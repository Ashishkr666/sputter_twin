"""2D Axisymmetric Multiphysics Plasma Solver for DC Magnetron Sputtering.

Solves the coupled 2D (r, z) plasma state of a planar circular DC magnetron
discharge on an axisymmetric grid:

1. Magnetized electron transport with cross-field mobility reduction:
       mu_perp(r, z) = mu_e0(r, z) / (1 + beta_e(r, z)^2)
   where beta_e = omega_ce / nu_en is the electron Hall parameter.
2. 2D electron temperature field T_e(r, z) shaped by magnetic-trap confinement.
3. 2D electron density field n_e(r, z) and ionization source:
       R_ion(r, z) = n_e(r, z) * n_g(r, z) * k_ion(T_e(r, z))
4. Cathode sheath: analytic Child-Langmuir potential V(r, z) and sheath width s(r).
5. Self-consistent 1D radial cathode ion current density J_i(r) and racetrack
   erosion profile, normalized to the magnetron power law W = V_d * I_d shared
   with :mod:`sputtertwin.physics.plasma`.

Model class and limitations (read before trusting absolute numbers)
-------------------------------------------------------------------
This is a **semi-empirical field model**, not a first-principles fluid or PIC
solve.  Specifically:

* Poisson's equation is *not* solved numerically.  The sheath potential is the
  analytic Child-Langmuir solution and the quasineutral bulk potential is a
  prescribed profile (default +15 V, i.e. ~3-5 Te); only the sheath width is
  obtained by inverting the Child-Langmuir law for the local current density.
* n_e(r, z) and T_e(r, z) are prescribed closure fields (Gaussian trap
  envelopes scaled by the magnetic solver's racetrack radius / trap thickness),
  not solutions of electron/ion continuity equations.  The one quantity that is
  *solved* self-consistently is the overall density (current) scale, fixed by
  the power balance W = V_d * I_d with the magnetron power law.
* Because the total cathode ion current is renormalized onto the power law, the
  absolute magnitude of the ionization rate coefficient cancels: only its
  spatial shape (via T_e and n_e) affects the results.
* Ions are treated as cold, unmagnetized and collisionless in the sheath; the
  ion density inside the sheath follows from flux continuity J_i = e * n_i * u_i.

Upgrade paths (see ``simulate_plasma_2d`` keyword arguments):
    * ``rarefaction=`` feeds the 2D gas rarefaction field
      (:mod:`sputtertwin.physics2d.rarefaction2d`) into n_g(r, z), nu_en,
      k_ion and the pressure feedback, i.e. genuine 2D multiphysics coupling.
    * ``cathode_flux_model="bohm"`` shapes J_i(r) by the Bohm sheath flux
      instead of the integrated ionization source.
    * ``ionization_rate_model="fit"`` swaps the Arrhenius rate coefficient for
      a Te-dependent fit to tabulated Ar electron-impact ionization data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from sputtertwin.numerics import trapezoid
from sputtertwin.physics.plasma import (
    effective_pressure_mtorr,
    magnetron_k_constant,
    magnetron_voltage_from_power,
)
from sputtertwin.physics2d.magnetic_field import (
    MagneticField2D,
    MagnetronGeometry,
    compute_magnetron_magnetic_field,
)
from sputtertwin.physics2d.rarefaction2d import GasRarefaction2D

__all__ = ["Plasma2DResult", "simulate_plasma_2d"]

# ---------------------------------------------------------------------------
# Fundamental physical constants (SI)
# ---------------------------------------------------------------------------
_EPSILON_0: float = 8.8541878128e-12  # Vacuum permittivity (F/m)
_E_CHARGE: float = 1.602176634e-19     # Elementary charge (C)
_M_E_KG: float = 9.1093837e-31         # Electron mass (kg)
_M_AR_KG: float = 39.948 * 1.6605390666e-27  # Argon ion mass (kg)
_K_B_J_K: float = 1.380649e-23         # Boltzmann constant (J/K)
_MTORR_TO_PA: float = 0.133322         # 1 mTorr in Pa

# ---------------------------------------------------------------------------
# Gas / collisional constants
# ---------------------------------------------------------------------------
_AMBIENT_GAS_TEMP_K: float = 300.0     # Room-temperature Ar background (K)
_SIGMA_EN_M2: float = 4.0e-20          # e-Ar momentum-transfer cross section (m^2)
_NU_EN_FLOOR_S: float = 1.0e-6         # Collision-frequency floor (s^-1)
# NOTE: the floor must stay far *below* any physical nu_en (~1e6 - 1e8 s^-1 at
# mTorr pressures). A floor comparable to nu_en silently destroys the electron
# Hall parameter beta_e = omega_ce / nu_en and hence the mobility reduction.
_B_FLOOR_TESLA: float = 1e-6           # |B| floor to avoid 0/0 (T)

# ---------------------------------------------------------------------------
# Ionization rate coefficients for Ar (m^3 s^-1), k = f(Te)
# ---------------------------------------------------------------------------
_E_ION_ARRHENIUS_EV: float = 15.76     # Ar ionization threshold (eV)
_K_IZ_ARRHENIUS: float = 2.0e-13       # Arrhenius pre-factor (legacy default)
_E_ION_FIT_EV: float = 17.44           # Fit activation energy (eV)
_K_IZ_FIT_PREFACTOR: float = 2.34e-14  # Fit pre-factor (m^3 s^-1)
_K_IZ_FIT_EXPONENT: float = 0.59       # Fit Te power-law exponent
# The "fit" closure is an empirical Te^0.59 * exp(-E/Te) form for Maxwellian-averaged
# Ar electron-impact ionization (order 1e-16 m^3/s at Te = 3 eV, 1e-14 at 10 eV).
# It is an alternative to the legacy Arrhenius form, not a certified dataset: because
# the cathode current is renormalized onto the power law, only its spatial shape
# (through Te and n_e) affects the solution.
_IONIZATION_MODELS: Tuple[str, ...] = ("arrhenius", "fit")
_CATHODE_FLUX_MODELS: Tuple[str, ...] = ("ionization_integral", "bohm")

# ---------------------------------------------------------------------------
# Magnetic-trap spatial scales (derived from the magnetic solver, not hardcoded)
# ---------------------------------------------------------------------------
# The factors below reproduce the historically calibrated lengths
# (12 mm / 10 mm radial, 12 mm / 15 mm axial, 25 mm ionization depth) for the
# reference 2-inch geometry, while scaling automatically with magnet geometry.
_TRAP_CONFINEMENT_R_FACTOR: float = 0.72  # x racetrack radius -> trap radial scale (m)
_TRAP_CONFINEMENT_Z_FACTOR: float = 1.50  # x trap thickness -> trap axial scale (m)
_DENSITY_R_FACTOR: float = 0.60           # x racetrack radius -> density radial scale (m)
_DENSITY_Z_FACTOR: float = 1.90           # x trap thickness -> density axial scale (m)
_DENSITY_Z_EXPONENT: float = 1.5          # Axial fall-off exponent of n_e
_Z_TRAP_EXTENT_FACTOR: float = 3.00       # x trap thickness -> ionization integral depth (m)
_MIN_TRAP_THICKNESS_M: float = 1.0e-4     # Guard for degenerate magnet geometries (m)

# ---------------------------------------------------------------------------
# Electron temperature closure
# ---------------------------------------------------------------------------
_TE_REF_EV: float = 3.0                # Te at the reference pressure (eV)
_TE_MIN_EV: float = 2.0                # Lower bound of the trap-peak Te (eV)
_TE_MAX_EV: float = 4.5                # Upper bound of the trap-peak Te (eV)
_TE_FLOOR_EV: float = 1.2              # Far-field (untrapped) Te (eV)
_TE_PRESSURE_EXPONENT: float = 0.25    # Te ~ P_eff^-0.25 scaling
_P_REF_MTORR: float = 5.0              # Reference effective pressure (mTorr)
_W_REF_W: float = 220.0                # Reference cathode power (W)

# ---------------------------------------------------------------------------
# Electron density calibration
# ---------------------------------------------------------------------------
_NE_REF_M3: float = 1.27e18            # ne at the reference operating point (m^-3)
_NE_PRESSURE_EXPONENT: float = 0.4     # ne ~ P_eff^0.4 scaling
_NE_FLOOR_M3: float = 1.0e10           # Positivity floor for n_e (m^-3)
_NE_BACKGROUND_FRACTION: float = 0.05  # Off-trap residual density fraction

# ---------------------------------------------------------------------------
# Magnetron power law (2D calibration; exponents shared with physics/plasma.py)
# ---------------------------------------------------------------------------
_K_2D_REF_POWER_W: float = 0.76        # k at the 2D reference point
_K_2D_REF_VOLTAGE_V: float = 395.0     # Reference discharge voltage (V)
_K_2D_REF_PRESSURE_MTORR: float = _P_REF_MTORR  # Reference effective pressure (mTorr)
_B_REF_TESLA: float = 0.045            # Reference parallel B at the racetrack (T)
_B_FACTOR_EXPONENT: float = 0.20       # V_d ~ B_parallel^0.2 confinement scaling

# ---------------------------------------------------------------------------
# Sputter-wind gas heating / rarefaction feedback on the effective pressure
# ---------------------------------------------------------------------------
_GAS_HEATING_COEFF_K: float = 75.0     # dT_gas per 100 W at the reference pressure (K)
_GAS_HEATING_PRESSURE_EXPONENT: float = 0.25
_GAS_HEATING_TRANSFER_FRACTION: float = 0.35  # Fraction of dT reaching the target front
_RAREFACTION_PRESSURE_EXPONENT: float = 0.30  # P_eff ~ T_gas^-0.30 feedback
_RAREFACTION_FRONT_DEPTH_M: float = 5.0e-3    # "Target front" depth for 2D averaging (m)

# ---------------------------------------------------------------------------
# Sheath constants
# ---------------------------------------------------------------------------
_SHEATH_MIN_M: float = 1.0e-3          # Lower bound on sheath thickness (m)
_SHEATH_MAX_M: float = 1.0e-2          # Upper bound on sheath thickness (m)
_SHEATH_J_FLOOR_FRACTION: float = 1e-3  # J_i floor as a fraction of the mean (A/m^2)
_BULK_PLASMA_POTENTIAL_V: float = 15.0  # Quasineutral bulk plasma potential (V)
_BULK_POTENTIAL_LENGTH_M: float = 0.010  # Bulk potential equilibration length (m)
_MIN_SHEATH_DROP_V: float = 1.0        # Floor for the ion acceleration drop (V)
_SMOOTH_SIGMA_M: float = 1.3e-3        # Cathode flux smoothing width (m, grid independent)
_FWHM_FRACTION: float = 0.5            # Half-maximum level for the groove width
_TOTAL_CURRENT_FLOOR_A: float = 1.0e-12  # Guard for the ion-current normalization (A)


@dataclass
class Plasma2DResult:
    """Complete 2D Multiphysics Simulation Result.

    Attributes:
        r_grid_m: 1D radial coordinates (m), shape ``(Nr,)``.
        z_grid_m: 1D axial coordinates from the target surface (m), shape ``(Nz,)``.
        V_2d: 2D electrostatic potential V(r, z) (V), shape ``(Nz, Nr)``.
            The cathode (z = 0) sits at ``-V_d``; the quasineutral bulk plasma
            sits at ``+V_p`` (~15 V by default).
        ne_2d: 2D electron density n_e(r, z) (m^-3), shape ``(Nz, Nr)``.
        ni_2d: 2D ion density n_i(r, z) (m^-3), shape ``(Nz, Nr)``.
        Te_2d: 2D electron temperature T_e(r, z) (eV), shape ``(Nz, Nr)``.
        R_ion_2d: 2D ionization rate R_ion(r, z) (m^-3 s^-1), shape ``(Nz, Nr)``.
        n_gas_2d_m3: 2D neutral Ar density used for collisions/ionization
            (m^-3), shape ``(Nz, Nr)``.
        hall_parameter_2d: 2D electron Hall parameter beta_e (dimensionless).
        mu_perp_2d: 2D cross-field electron mobility (m^2 V^-1 s^-1).
        sheath_thickness_1d: Cathode sheath thickness s(r) (m), shape ``(Nr,)``.
        bohm_velocity_1d: Bohm velocity at the cathode surface (m/s), ``(Nr,)``.
        ion_flux_profile_1d: Cathode surface ion flux Gamma_i(r) (m^-2 s^-1).
        current_density_1d: Cathode surface current density J_i(r) (A/m^2).
        voltage_v: Cathode discharge voltage V_d (V).
        current_a: Total discharge current I_d (A).
        power_w: Total cathode power W (W).
        total_ion_current_a: Ion component of the discharge current (A).
        effective_pressure_mtorr: Target-front effective pressure actually used in
            the power law (mTorr), i.e. after the sputter-wind rarefaction
            feedback. Lower than the setpoint pressure at high power.
        peak_ne_m3: Peak electron density in the magnetic trap (m^-3).
        racetrack_radius_m: Peak erosion radius (m).
        racetrack_fwhm_m: Full-width-at-half-maximum of the erosion groove (m).
    """

    r_grid_m: np.ndarray
    z_grid_m: np.ndarray
    V_2d: np.ndarray
    ne_2d: np.ndarray
    ni_2d: np.ndarray
    Te_2d: np.ndarray
    R_ion_2d: np.ndarray
    n_gas_2d_m3: np.ndarray
    hall_parameter_2d: np.ndarray
    mu_perp_2d: np.ndarray
    sheath_thickness_1d: np.ndarray
    bohm_velocity_1d: np.ndarray
    ion_flux_profile_1d: np.ndarray
    current_density_1d: np.ndarray
    voltage_v: float
    current_a: float
    power_w: float
    total_ion_current_a: float
    effective_pressure_mtorr: float
    peak_ne_m3: float
    racetrack_radius_m: float
    racetrack_fwhm_m: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert the key scalar metrics to a JSON-serializable dictionary."""
        return {
            "voltage_v": float(self.voltage_v),
            "current_a": float(self.current_a),
            "power_w": float(self.power_w),
            "total_ion_current_a": float(self.total_ion_current_a),
            "effective_pressure_mtorr": float(self.effective_pressure_mtorr),
            "peak_ne_m3": float(self.peak_ne_m3),
            "peak_te_ev": float(np.max(self.Te_2d)),
            "peak_ion_flux_m2_s": float(np.max(self.ion_flux_profile_1d)),
            "peak_current_density_a_m2": float(np.max(self.current_density_1d)),
            "mean_sheath_thickness_mm": float(np.mean(self.sheath_thickness_1d)) * 1e3,
            "max_hall_parameter": float(np.max(self.hall_parameter_2d)),
            "racetrack_radius_mm": float(self.racetrack_radius_m) * 1e3,
            "racetrack_fwhm_mm": float(self.racetrack_fwhm_m) * 1e3,
            "grid_r": int(self.r_grid_m.size),
            "grid_z": int(self.z_grid_m.size),
        }

    def summary(self) -> str:
        """Return a formatted multi-line summary of the 2D discharge solution."""
        d = self.to_dict()
        return (
            "Plasma2DResult Summary:\n"
            f"  Discharge Voltage:   {d['voltage_v']:.1f} V\n"
            f"  Discharge Current:   {d['current_a']:.3f} A\n"
            f"  Ion Current:         {d['total_ion_current_a']:.3f} A\n"
            f"  Cathode Power:       {d['power_w']:.1f} W\n"
            f"  Eff. Pressure (front):{d['effective_pressure_mtorr']:.2f} mTorr\n"
            f"  Peak ne:             {d['peak_ne_m3']:.3e} m^-3\n"
            f"  Peak Te:             {d['peak_te_ev']:.2f} eV\n"
            f"  Peak Ion Flux:       {d['peak_ion_flux_m2_s']:.3e} m^-2 s^-1\n"
            f"  Peak Current Density:{d['peak_current_density_a_m2']:.1f} A/m^2\n"
            f"  Mean Sheath:         {d['mean_sheath_thickness_mm']:.2f} mm\n"
            f"  Max Hall Parameter:  {d['max_hall_parameter']:.1f}\n"
            f"  Racetrack Radius:    {d['racetrack_radius_mm']:.2f} mm\n"
            f"  Racetrack FWHM:      {d['racetrack_fwhm_mm']:.2f} mm"
        )


# ---------------------------------------------------------------------------
# Field closures
# ---------------------------------------------------------------------------
def _ionization_rate_coefficient(
    Te_2d: np.ndarray,
    model: str = "arrhenius",
) -> np.ndarray:
    """Return the Ar electron-impact ionization rate coefficient k_ion(Te) [m^3/s].

    Args:
        Te_2d: Electron temperature field (eV).
        model: ``"arrhenius"`` (legacy single-threshold Arrhenius form) or
            ``"fit"`` (Te-dependent power-law/Arrhenius fit to tabulated Ar
            electron-impact ionization data).

    Returns:
        Rate coefficient field with the same shape as ``Te_2d``.

    Raises:
        ValueError: If ``model`` is not a supported option.
    """
    Te_safe = np.maximum(Te_2d, 0.5)
    if model == "arrhenius":
        return _K_IZ_ARRHENIUS * np.exp(-_E_ION_ARRHENIUS_EV / Te_safe)
    if model == "fit":
        return (
            _K_IZ_FIT_PREFACTOR
            * (Te_safe ** _K_IZ_FIT_EXPONENT)
            * np.exp(-_E_ION_FIT_EV / Te_safe)
        )
    raise ValueError(
        f"ionization_rate_model must be one of {_IONIZATION_MODELS}, got {model!r}"
    )


def _fwhm_linear(r_vec: np.ndarray, y: np.ndarray, fraction: float = _FWHM_FRACTION) -> float:
    """Full width of ``y`` at ``fraction`` of its peak, with sub-grid interpolation.

    Unlike a pure index-based width, this is (approximately) independent of the
    radial grid resolution and does not snap to grid points.

    Args:
        r_vec: Monotonically increasing radial coordinates (m).
        y: Radial profile (any positive quantity).
        fraction: Fraction of the peak defining the width (0.5 = FWHM).

    Returns:
        Interpolated width in meters (0.0 if the profile is degenerate).
    """
    peak = float(np.max(y)) if y.size else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        return 0.0

    level = fraction * peak
    above = np.flatnonzero(y >= level)
    if above.size < 2:
        return 0.0

    def _crossing(i_left: int) -> float:
        """Linear crossing point of ``level`` between i_left and i_left + 1."""
        y0, y1 = float(y[i_left]), float(y[i_left + 1])
        r0, r1 = float(r_vec[i_left]), float(r_vec[i_left + 1])
        dy = y1 - y0
        if abs(dy) < 1e-30:
            return r1
        return r0 + (level - y0) * (r1 - r0) / dy

    i_lo = int(above[0])
    i_hi = int(above[-1])
    r_left = _crossing(i_lo - 1) if i_lo > 0 else float(r_vec[0])
    r_right = _crossing(i_hi) if i_hi < r_vec.size - 1 else float(r_vec[-1])
    return float(max(r_right - r_left, 0.0))


def _validate_inputs(
    power_w: float,
    pressure_mtorr: float,
    ar_flow_sccm: float,
    target_radius_m: float,
    anode_distance_m: float,
    grid_r: int,
    grid_z: int,
    gamma_se: float,
    gas_temp_k: float,
    ionization_rate_model: str,
    cathode_flux_model: str,
    mag_field: Optional[MagneticField2D],
) -> None:
    """Validate ``simulate_plasma_2d`` inputs, raising ``ValueError`` on abuse.

    Mirrors the guard style of :func:`sputtertwin.physics.plasma.calculate_discharge_state`
    so both entry points fail identically on bad input.
    """
    _finite = {
        "power_w": power_w,
        "pressure_mtorr": pressure_mtorr,
        "ar_flow_sccm": ar_flow_sccm,
        "target_radius_m": target_radius_m,
        "anode_distance_m": anode_distance_m,
        "gamma_se": gamma_se,
        "gas_temp_k": gas_temp_k,
    }
    for name, value in _finite.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number, got {value!r}")

    if power_w <= 0.0:
        raise ValueError(f"power_w must be > 0, got {power_w}")
    if pressure_mtorr <= 0.0:
        raise ValueError(f"pressure_mtorr must be > 0, got {pressure_mtorr}")
    if ar_flow_sccm <= 0.0:
        raise ValueError(f"ar_flow_sccm must be > 0, got {ar_flow_sccm}")
    if target_radius_m <= 0.0:
        raise ValueError(f"target_radius_m must be > 0, got {target_radius_m}")
    if anode_distance_m <= 0.0:
        raise ValueError(f"anode_distance_m must be > 0, got {anode_distance_m}")
    if not (0.0 <= gamma_se < 1.0):
        raise ValueError(
            f"gamma_se must be in [0, 1) — values >= 1 produce a negative ion current,"
            f" got {gamma_se}"
        )
    if gas_temp_k <= 0.0:
        raise ValueError(f"gas_temp_k must be > 0, got {gas_temp_k}")

    if ionization_rate_model not in _IONIZATION_MODELS:
        raise ValueError(
            f"ionization_rate_model must be one of {_IONIZATION_MODELS},"
            f" got {ionization_rate_model!r}"
        )
    if cathode_flux_model not in _CATHODE_FLUX_MODELS:
        raise ValueError(
            f"cathode_flux_model must be one of {_CATHODE_FLUX_MODELS},"
            f" got {cathode_flux_model!r}"
        )

    if mag_field is None:
        for name, value in (("grid_r", grid_r), ("grid_z", grid_z)):
            if not isinstance(value, (int, np.integer)) or int(value) < 2:
                raise ValueError(
                    f"{name} must be an integer >= 2 (got {value!r});"
                    " pass a pre-computed mag_field to use its grid instead"
                )


def _grid_from_magnetic_field(mag_field: MagneticField2D) -> Tuple[np.ndarray, np.ndarray]:
    """Extract and validate the (r, z) grid carried by a pre-computed field."""
    r_vec = np.asarray(mag_field.r_grid_m, dtype=float)
    z_vec = np.asarray(mag_field.z_grid_m, dtype=float)
    for name, vec in (("r_grid_m", r_vec), ("z_grid_m", z_vec)):
        if vec.ndim != 1 or vec.size < 2:
            raise ValueError(
                f"mag_field.{name} must be a 1D array with >= 2 points, got shape {vec.shape}"
            )
        if np.any(np.diff(vec) <= 0.0):
            raise ValueError(f"mag_field.{name} must be strictly increasing")
    return r_vec, z_vec


def simulate_plasma_2d(
    power_w: float = 220.0,
    pressure_mtorr: float = 5.0,
    ar_flow_sccm: float = 20.0,
    target_radius_m: float = 0.050,
    anode_distance_m: float = 0.060,
    grid_r: int = 60,
    grid_z: int = 50,
    gamma_se: float = 0.10,
    mag_field: Optional[MagneticField2D] = None,
    *,
    gas_temp_k: float = _AMBIENT_GAS_TEMP_K,
    ionization_rate_model: str = "arrhenius",
    cathode_flux_model: str = "ionization_integral",
    rarefaction: Optional[GasRarefaction2D] = None,
    bulk_plasma_potential_v: Optional[float] = None,
    smooth_sigma_m: float = _SMOOTH_SIGMA_M,
    sheath_thickness_bounds_m: Tuple[float, float] = (_SHEATH_MIN_M, _SHEATH_MAX_M),
) -> Plasma2DResult:
    """Solve the 2D coupled magnetic-plasma equations on an axisymmetric grid (r, z).

    Args:
        power_w: Cathode electrical power (Watts, must be > 0).
        pressure_mtorr: Argon setpoint pressure (mTorr, must be > 0).
        ar_flow_sccm: Argon gas mass flow (sccm, must be > 0).
        target_radius_m: Target disk radius (m, must be > 0).
        anode_distance_m: Target-to-anode/wafer spacing (m, must be > 0).
        grid_r: Number of radial grid points (>= 2, ignored if ``mag_field`` given).
        grid_z: Number of axial grid points (>= 2, ignored if ``mag_field`` given).
        gamma_se: Secondary electron emission coefficient in [0, 1).
        mag_field: Pre-computed 2D magnetic field (computed if None). Its grid
            overrides ``grid_r`` / ``grid_z`` / ``target_radius_m``.
        gas_temp_k: Ambient Ar temperature (K).
        ionization_rate_model: ``"arrhenius"`` (default, legacy) or ``"fit"``.
        cathode_flux_model: ``"ionization_integral"`` (default) or ``"bohm"``.
        rarefaction: Optional 2D gas rarefaction field. When supplied, its local
            neutral density replaces the uniform n_g in the collision frequency,
            ionization rate and density scale, and its target-front mean
            rarefaction replaces the analytic gas-heating pressure feedback.
        bulk_plasma_potential_v: Quasineutral bulk plasma potential (V). Defaults
            to :data:`_BULK_PLASMA_POTENTIAL_V` (~3-5 Te). Note that ions are
            accelerated through ``V_d + V_p``, not ``V_d`` alone.
        smooth_sigma_m: Cathode flux smoothing width in **meters** (grid
            resolution independent). Set to 0 to disable smoothing.
        sheath_thickness_bounds_m: (min, max) clamp on the Child-Langmuir
            sheath thickness (m).

    Returns:
        Plasma2DResult containing complete 2D fields and cathode racetrack profiles.

    Raises:
        ValueError: If any input is non-finite, out of range, or if the supplied
            magnetic field / rarefaction grids are incompatible with the solve.
        RuntimeError: If the ionization integral produces no cathode ion current
            (degenerate grid or pressure configuration).
    """
    _validate_inputs(
        power_w=power_w,
        pressure_mtorr=pressure_mtorr,
        ar_flow_sccm=ar_flow_sccm,
        target_radius_m=target_radius_m,
        anode_distance_m=anode_distance_m,
        grid_r=grid_r,
        grid_z=grid_z,
        gamma_se=gamma_se,
        gas_temp_k=gas_temp_k,
        ionization_rate_model=ionization_rate_model,
        cathode_flux_model=cathode_flux_model,
        mag_field=mag_field,
    )

    # ------------------------------------------------------------------
    # 1. 2D Coordinate Grid Setup
    # ------------------------------------------------------------------
    if mag_field is not None:
        r_vec, z_vec = _grid_from_magnetic_field(mag_field)
    else:
        r_vec = np.linspace(0.0, target_radius_m, int(grid_r))
        z_vec = np.linspace(0.0, anode_distance_m, int(grid_z))
    grid_r = int(r_vec.size)
    grid_z = int(z_vec.size)
    dr = float(r_vec[1] - r_vec[0])

    # 'xy' indexing -> R, Z have shape (Nz, Nr), matching the field convention.
    R, Z = np.meshgrid(r_vec, z_vec, indexing="xy")

    # ------------------------------------------------------------------
    # 2. Neutral gas density (uniform, or 2D when rarefaction is coupled)
    # ------------------------------------------------------------------
    p_eff = effective_pressure_mtorr(pressure_mtorr, ar_flow_sccm)
    p_pa = p_eff * _MTORR_TO_PA
    n_g_ambient = p_pa / (_K_B_J_K * gas_temp_k)

    if rarefaction is not None:
        n_gas_2d = np.asarray(rarefaction.n_gas_2d_m3, dtype=float)
        if n_gas_2d.shape != (grid_z, grid_r):
            raise ValueError(
                "rarefaction.n_gas_2d_m3 has shape "
                f"{n_gas_2d.shape}, expected {(grid_z, grid_r)} for the plasma grid"
            )
        n_gas_2d = np.maximum(n_gas_2d, 1.0)  # physical density floor (m^-3)
    else:
        n_gas_2d = np.full((grid_z, grid_r), n_g_ambient)

    # ------------------------------------------------------------------
    # 3. Magnetic Field Solution
    # ------------------------------------------------------------------
    if mag_field is None:
        geom = MagnetronGeometry(target_radius_m=target_radius_m)
        mag_field = compute_magnetron_magnetic_field(
            geom=geom,
            grid_r_points=grid_r,
            grid_z_points=grid_z,
            z_max_m=anode_distance_m,
        )

    B_mag = mag_field.B_mag
    B_r = mag_field.B_r
    r_race = mag_field.racetrack_radius_m

    # Spatial scales are tied to the magnetic solution rather than hardcoded,
    # so the plasma fields track changes in magnet geometry / grid extent.
    trap_thickness = max(float(mag_field.trap_thickness_m), _MIN_TRAP_THICKNESS_M)
    r_scale_trap = max(_TRAP_CONFINEMENT_R_FACTOR * r_race, 1.0e-3)
    z_scale_trap = _TRAP_CONFINEMENT_Z_FACTOR * trap_thickness
    r_scale_density = max(_DENSITY_R_FACTOR * r_race, 1.0e-3)
    z_scale_density = _DENSITY_Z_FACTOR * trap_thickness
    z_trap_extent = min(_Z_TRAP_EXTENT_FACTOR * trap_thickness, float(z_vec[-1]))

    # ------------------------------------------------------------------
    # 4. 2D Electron Temperature Field T_e(r, z)
    # ------------------------------------------------------------------
    # Electrons are heated in the magnetic trap where the E x B drift closes;
    # Te peaks near the cathode racetrack and falls toward the bulk anode.
    te_trap_peak = float(
        np.clip(
            _TE_REF_EV * ((_P_REF_MTORR / p_eff) ** _TE_PRESSURE_EXPONENT),
            _TE_MIN_EV,
            _TE_MAX_EV,
        )
    )
    # Confinement weight is highest where the field is parallel (B_r / |B| -> 1).
    b_parallel_ratio = np.abs(B_r) / (B_mag + _B_FLOOR_TESLA)
    trap_confinement = (
        np.exp(-((R - r_race) / r_scale_trap) ** 2)
        * np.exp(-(Z / z_scale_trap) ** 2)
    )
    Te_2d = _TE_FLOOR_EV + (te_trap_peak - _TE_FLOOR_EV) * (
        0.3 + 0.7 * trap_confinement * b_parallel_ratio
    )

    # ------------------------------------------------------------------
    # 5. Cross-Field Electron Mobility mu_perp(r, z) and Hall parameter
    # ------------------------------------------------------------------
    v_th_e = np.sqrt((8.0 * _E_CHARGE * Te_2d) / (math.pi * _M_E_KG))
    nu_en = n_gas_2d * _SIGMA_EN_M2 * v_th_e
    omega_ce = (_E_CHARGE * B_mag) / _M_E_KG
    hall_parameter = omega_ce / (nu_en + _NU_EN_FLOOR_S)
    mu_e0 = _E_CHARGE / (_M_E_KG * nu_en)
    mu_perp = mu_e0 / (1.0 + hall_parameter ** 2)

    # ------------------------------------------------------------------
    # 6. 2D Electron Density and Ionization Source
    # ------------------------------------------------------------------
    k_ion = _ionization_rate_coefficient(Te_2d, model=ionization_rate_model)

    # Peak density scales with generator power and pressure.
    ne_scale = _NE_REF_M3 * (power_w / _W_REF_W) * ((p_eff / _P_REF_MTORR) ** _NE_PRESSURE_EXPONENT)
    density_shape = (
        np.exp(-((R - r_race) / r_scale_density) ** 2)
        * np.exp(-(Z / z_scale_density) ** _DENSITY_Z_EXPONENT)
    )
    ne_2d = ne_scale * (_NE_BACKGROUND_FRACTION + (1.0 - _NE_BACKGROUND_FRACTION) * density_shape)
    ne_2d = np.maximum(ne_2d, _NE_FLOOR_M3)

    # 2D ionization source term: R_ion = ne * ng * k_ion [m^-3 s^-1]
    R_ion_2d = ne_2d * n_gas_2d * k_ion

    # ------------------------------------------------------------------
    # 7. Cathode Ion Flux Profile Gamma_i(r)
    # ------------------------------------------------------------------
    # Bohm velocity at the cathode surface (used by the "bohm" flux model).
    u_bohm_r = np.sqrt((_E_CHARGE * Te_2d[0, :]) / _M_AR_KG)

    if cathode_flux_model == "bohm":
        # Sheath-limited flux: Gamma_i = exp(-1/2) * ne * u_Bohm
        gamma_i_r = np.exp(-0.5) * ne_2d[0, :] * u_bohm_r
    else:
        # Ionization produced above the cathode inside the magnetic trap:
        # Gamma_i(r) = int_0^z_trap R_ion(r, z) dz
        z_mask = z_vec <= z_trap_extent
        if int(np.count_nonzero(z_mask)) < 2:
            raise RuntimeError(
                "Ionization integral is degenerate: fewer than 2 axial grid points lie"
                f" below z_trap_extent = {z_trap_extent:.4f} m. Increase grid_z or"
                " anode_distance_m."
            )
        gamma_i_r = trapezoid(R_ion_2d[z_mask, :], z_vec[z_mask], axis=0)
        gamma_i_r = np.maximum(gamma_i_r, 0.0)

    # Smooth the cathode ion flux profile. The smoothing width is specified in
    # meters so the result does not silently change with grid resolution.
    if smooth_sigma_m > 0.0 and dr > 0.0:
        gamma_i_r = gaussian_filter(gamma_i_r, sigma=smooth_sigma_m / dr)

    # ------------------------------------------------------------------
    # 8. Self-Consistent Voltage Solution (magnetron power law)
    # ------------------------------------------------------------------
    # Total ion current from the 2D integral: I_ion = 2*pi*int Gamma_i(r) r dr * e
    total_integrated_flux = 2.0 * math.pi * trapezoid(gamma_i_r * r_vec, r_vec)
    i_ion_raw = float(total_integrated_flux * _E_CHARGE)

    if not math.isfinite(i_ion_raw) or i_ion_raw <= _TOTAL_CURRENT_FLOOR_A:
        raise RuntimeError(
            f"Ionization source produced no cathode ion current (I_ion = {i_ion_raw:.3e} A)."
            " Check pressure, grid resolution and magnetic field inputs."
        )

    # Power balance: W = V_d * I_d = V_d * I_ion * (1 + gamma_se)
    if rarefaction is not None:
        # 2D sputter-wind feedback: mean rarefaction in front of the target.
        front = z_vec <= _RAREFACTION_FRONT_DEPTH_M
        rarefaction_factor = float(np.mean(rarefaction.rarefaction_factor_2d[front, :]))
        p_eff_target = p_eff * min(max(rarefaction_factor, 1.0e-3), 1.0)
    else:
        # Sputter-wind thermal rarefaction feedback at the cathode front.
        delta_t_gas = (
            _GAS_HEATING_COEFF_K
            * (power_w / 100.0)
            * ((_P_REF_MTORR / p_eff) ** _GAS_HEATING_PRESSURE_EXPONENT)
        )
        t_gas_local = gas_temp_k + _GAS_HEATING_TRANSFER_FRACTION * delta_t_gas
        p_eff_target = p_eff * ((gas_temp_k / t_gas_local) ** _RAREFACTION_PRESSURE_EXPONENT)

    # 2D magnetic confinement scaling from the parallel B-field at the racetrack:
    # stronger parallel B -> better electron confinement -> lower V_d at fixed W.
    b_factor = (mag_field.b_parallel_peak_tesla / _B_REF_TESLA) ** _B_FACTOR_EXPONENT
    k_2d = (
        magnetron_k_constant(
            _K_2D_REF_POWER_W,
            _K_2D_REF_VOLTAGE_V,
            _K_2D_REF_PRESSURE_MTORR,
        )
        * b_factor
    )

    v_discharge = magnetron_voltage_from_power(power_w, p_eff_target, k_constant=k_2d)
    i_discharge = power_w / v_discharge
    i_ion_target = i_discharge / (1.0 + gamma_se)

    # Calibrate the 1D flux profile to exactly conserve the total cathode ion current
    flux_scale = i_ion_target / i_ion_raw
    gamma_i_r = gamma_i_r * flux_scale
    J_i_r = gamma_i_r * _E_CHARGE  # Cathode current density in A/m^2

    # ------------------------------------------------------------------
    # 9. Cathode Sheath: Child-Langmuir potential V(r, z) and ion density
    # ------------------------------------------------------------------
    v_bulk = (
        float(bulk_plasma_potential_v)
        if bulk_plasma_potential_v is not None
        else _BULK_PLASMA_POTENTIAL_V
    )
    sheath_min, sheath_max = sorted(sheath_thickness_bounds_m)

    # Child-Langmuir: s = sqrt(4/9 * eps0 * sqrt(2e/M) * Vd^(3/2) / J_i)
    # The density floor is relative to the mean current density so it never
    # dominates a physical racetrack profile.
    j_floor = _SHEATH_J_FLOOR_FRACTION * max(float(np.mean(J_i_r)), 1.0e-6)
    sheath_s = np.sqrt(
        (4.0 / 9.0)
        * _EPSILON_0
        * np.sqrt((2.0 * _E_CHARGE) / _M_AR_KG)
        * (v_discharge ** 1.5)
        / np.maximum(J_i_r, j_floor)
    )
    sheath_s = np.clip(sheath_s, sheath_min, sheath_max)  # Bound to a physical sheath

    # Vectorized (previously double Python loop over r, z).
    sheath_mask = Z <= sheath_s[None, :]
    z_over_s = np.clip(Z / sheath_s[None, :], 0.0, 1.0)
    # Child-Langmuir sheath potential: V(z) = -Vd * (1 - z/s)^(4/3)
    V_sheath = -v_discharge * (1.0 - z_over_s) ** (4.0 / 3.0)
    # Quasineutral bulk plasma potential (+V_p, relaxing over a few mm)
    V_bulk = v_bulk * (1.0 - np.exp(-np.maximum(Z - sheath_s[None, :], 0.0) / _BULK_POTENTIAL_LENGTH_M))
    V_2d = np.where(sheath_mask, V_sheath, V_bulk)

    # 10. 2D Ion Density n_i(r, z)
    # Outside the sheath: ni = ne (quasineutrality)
    # Inside the sheath: ni = Ji / (e * u_ion(z)) with u_ion from energy conservation
    delta_v = np.maximum(v_bulk - V_2d, _MIN_SHEATH_DROP_V)
    u_ion = np.sqrt((2.0 * _E_CHARGE * delta_v) / _M_AR_KG)
    ni_2d = np.where(sheath_mask, J_i_r[None, :] / (_E_CHARGE * u_ion), ne_2d)

    # ------------------------------------------------------------------
    # 11. Racetrack groove width (sub-grid interpolated FWHM)
    # ------------------------------------------------------------------
    fwhm = _fwhm_linear(r_vec, gamma_i_r)

    return Plasma2DResult(
        r_grid_m=r_vec,
        z_grid_m=z_vec,
        V_2d=V_2d,
        ne_2d=ne_2d,
        ni_2d=ni_2d,
        Te_2d=Te_2d,
        R_ion_2d=R_ion_2d,
        n_gas_2d_m3=n_gas_2d,
        hall_parameter_2d=hall_parameter,
        mu_perp_2d=mu_perp,
        sheath_thickness_1d=sheath_s,
        bohm_velocity_1d=u_bohm_r,
        ion_flux_profile_1d=gamma_i_r,
        current_density_1d=J_i_r,
        voltage_v=float(v_discharge),
        current_a=float(i_discharge),
        power_w=float(power_w),
        total_ion_current_a=float(i_ion_target),
        effective_pressure_mtorr=float(p_eff_target),
        peak_ne_m3=float(np.max(ne_2d)),
        racetrack_radius_m=float(r_race),
        racetrack_fwhm_m=fwhm,
    )
