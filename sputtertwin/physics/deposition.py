"""Deposition geometry engine and wafer profile simulation module for SputterTwin.

Models the 2D film thickness distribution across a wafer substrate from circular
erosion racetrack sputtering. Transport is fully wired into both the deposition
rate (via thermalization efficiency) and the radial profile (via scattering broadening).
All metrics are computed over the circular wafer area only. The racetrack radius
geometrically controls the profile non-uniformity. The profile model is bounded
and singularity-free at any physical pressure.

Calibration target (Slide 7 nominal benchmark):
    Power = 220 W, Pressure = 5.0 mTorr, Distance = 80 mm,
    Material = 'Ti', Time = 720 s
    => Mean thickness ~ 100.1 nm, Rate ~ 8.4 nm/min, NU ~ 1.2 %
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Union

import numpy as np

__all__ = [
    "DepositionResult",
    "simulate_deposition",
]

from sputtertwin.physics.plasma import DischargeState, calculate_discharge_state
from sputtertwin.physics.sputter_yield import (
    MATERIALS,
    TargetMaterial,
    calculate_sputter_yield,
)
from sputtertwin.physics.transport import (
    calculate_knudsen_number,
    calculate_mean_free_path,
    calculate_scattering_broadening,
    calculate_transmission_probability,
)

# Fundamental constants
_AVOGADRO: float = 6.02214076e23        # atoms / mol
_ELEMENTARY_CHARGE_C: float = 1.602176634e-19  # Coulombs

# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------
# _ETA_CHAMBER: geometric + sticking efficiency fraction.
# Calibrated so that at nominal conditions (220 W, 5 mTorr, Ti, 80 mm, 720 s)
# the mean thickness = 100.1 nm (rate = 8.34 nm/min) with exact ion current I_d / (1 + gamma_se).
_ETA_CHAMBER: float = 0.141336

# _N_THERM: effective thermalization collision count for the transport efficiency
# model. eta_transport = 1 / (1 + n_coll / N_THERM).
# Physically represents the mean number of collisions before a sputtered atom
# is effectively lost to the sidewalls rather than deposited on the wafer.
# At 5 mTorr / 80 mm (n_coll ~ 5.15): eta_transport ~ 0.660.
_N_THERM: float = 10.0

# _GAMMA_PROFILE: dimensionless profile non-uniformity pre-factor.
# Controls the depth of the parabolic radial profile:
#   c_eff = _GAMMA_PROFILE * (racetrack_radius_mm / distance_mm)^2 / BF^2
# Calibrated so that c_eff at nominal conditions gives NU = 1.20 %.
_GAMMA_PROFILE: float = 1.4922

# Maximum allowed grid size per axis (memory guard: 500^2 = 250 k points).
_MAX_GRID_SIZE: int = 500


@dataclass
class DepositionResult:
    """Simulation results for thin-film sputter deposition across a wafer.

    Attributes:
        thickness_map: 2D array of film thickness across the full square grid (nm).
            Points outside the circular wafer boundary are set to NaN.
            Shape: (grid_size, grid_size).
        mean_thickness_nm: Mean deposited film thickness over on-wafer grid points (nm).
        uniformity_percent: Thickness non-uniformity over on-wafer points, using the
            semiconductor fab standard: (max - min) / (2 * mean) * 100.
        std_thickness_nm: Standard deviation of thickness over on-wafer points (nm).
        deposition_rate_nm_min: Mean film growth rate over on-wafer points (nm/min).
        min_thickness_nm: Minimum thickness among on-wafer grid points (nm).
        max_thickness_nm: Maximum thickness among on-wafer grid points (nm).
        x_grid_mm: 1D array of x coordinates for grid columns (mm).
        y_grid_mm: 1D array of y coordinates for grid rows (mm).
        discharge_state: Plasma DischargeState instance (None if power <= 0).
        sputter_yield: Effective sputter yield (atoms / incident ion).
        mean_free_path_mm: Argon mean free path at operating pressure (mm).
        knudsen_number: Knudsen number Kn = lambda_mfp / distance.
        power_w: Cathode electrical power (W).
        pressure_mtorr: Argon set-point pressure (mTorr).
        time_s: Deposition time (s).
        n_wafer_points: Number of grid points that fall inside the circular wafer.
    """

    thickness_map: np.ndarray
    mean_thickness_nm: float
    uniformity_percent: float
    std_thickness_nm: float
    deposition_rate_nm_min: float
    min_thickness_nm: float
    max_thickness_nm: float
    x_grid_mm: np.ndarray
    y_grid_mm: np.ndarray
    discharge_state: Any
    sputter_yield: float
    mean_free_path_mm: float
    knudsen_number: float
    power_w: float
    pressure_mtorr: float
    time_s: float
    n_wafer_points: int

    def summary(self) -> str:
        """Return a formatted report similar to the Slide 7 dashboard."""
        return (
            f"Deposition Result Summary:\n"
            f"  Mean Thickness: {self.mean_thickness_nm:.1f} nm\n"
            f"  Deposition Rate: {self.deposition_rate_nm_min:.2f} nm/min\n"
            f"  Non-Uniformity: {self.uniformity_percent:.2f} %\n"
            f"  Min Thickness:  {self.min_thickness_nm:.1f} nm\n"
            f"  Max Thickness:  {self.max_thickness_nm:.1f} nm\n"
            f"  Std Deviation:  {self.std_thickness_nm:.2f} nm"
        )


def simulate_deposition(
    power_w: float,
    pressure_mtorr: float,
    ar_flow_sccm: float = 20.0,
    distance_mm: float = 80.0,
    deposition_time_s: float = 60.0,
    material: Union[str, TargetMaterial] = "Ti",
    grid_size: int = 5,
    wafer_radius_mm: float = 75.0,
    target_radius_mm: float = 50.0,
    racetrack_radius_mm: float = 25.0,
    target_erosion_mm: float = 0.0,
) -> DepositionResult:
    """Simulate thin film deposition thickness distribution across a wafer.

    Physics pipeline
    ----------------
    1. **Discharge** — `calculate_discharge_state` maps (power, p_eff) to
       discharge voltage, total ion current, and ion flux.
    2. **Sputter yield** — Yamamura-Tawara Y(Ei, Ti/Cu/Al) at the sheath ion energy.
    3. **Rate** — total sputtered atoms/s = I_ion / e * Y, attenuated by the
       thermalization transport efficiency eta_transport = 1/(1 + n_coll/N_THERM),
       then distributed over the wafer area via eta_chamber geometric factor.
    4. **Profile** — parabolic radial non-uniformity with coefficient
       c_eff = GAMMA_PROFILE * (r_race/d)^2 / BF^2,
       where BF = scattering broadening factor. Higher pressure → larger BF →
       smaller c_eff → more uniform film. Larger racetrack radius → higher c_eff
       → less uniform film. c_eff is always positive and < 1 for physical inputs,
       so no singularity is possible.
    5. **Metrics** — all statistics (mean, min, max, std, NU) are evaluated only
       over grid points that lie inside the circular wafer (R <= wafer_radius_mm).
       Off-wafer points in thickness_map are set to NaN.
    6. **Erosion** — target groove depth collimation: erosion broadens the profile
       slightly, increasing non-uniformity via c_eff_erosion = c_eff * (1 + 0.10 * erosion_mm).

    Calibration (Slide 7 nominal benchmark at 220 W, 5 mTorr, 80 mm, Ti, 720 s):
        Mean thickness: 100.1 nm   |  Rate: 8.34 nm/min  |  NU: 1.20 %

    Args:
        power_w: Cathode electrical power in Watts. If <= 0, returns zero deposition.
        pressure_mtorr: Argon set-point gas pressure in mTorr (must be > 0 and finite).
        ar_flow_sccm: Argon gas flow in sccm (must be > 0, default 20.0).
            Wired into effective pressure via plasma.py Option-B model.
        distance_mm: Target-to-substrate distance in mm (must be > 0 and finite).
        deposition_time_s: Process run duration in seconds (default 60.0).
        material: Target material symbol ('Ti', 'Cu', 'Al') or TargetMaterial instance.
        grid_size: Integer number of sample points along each wafer axis
            (must be int in [3, 500], default 5).
        wafer_radius_mm: Substrate wafer radius in mm (default 75.0 = 150 mm wafer).
        target_radius_mm: Target disk radius in mm (default 50.0 mm).
        racetrack_radius_mm: Radius of the magnetron erosion ring in mm (default 25.0 mm).
            Controls radial non-uniformity: larger ring → less uniform film.
        target_erosion_mm: Depth of target erosion groove in mm (default 0.0 mm).
            Increases non-uniformity via groove collimation.

    Returns:
        DepositionResult with thickness map, rate, uniformity, and plasma physics state.

    Raises:
        TypeError: If grid_size is not an int.
        ValueError: If any required parameter is out of range or non-finite.
        KeyError: If material string is not in MATERIALS.
    """
    # -----------------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------------
    # grid_size must be a true Python int (not float, bool, etc.)
    if not isinstance(grid_size, int) or isinstance(grid_size, bool):
        raise TypeError(
            f"grid_size must be an int, got {type(grid_size).__name__}"
        )
    if not (3 <= grid_size <= _MAX_GRID_SIZE):
        raise ValueError(
            f"grid_size must be in [3, {_MAX_GRID_SIZE}], got {grid_size}"
        )

    # Finiteness checks on critical float inputs
    for _name, _val in [
        ("power_w", power_w),
        ("pressure_mtorr", pressure_mtorr),
        ("ar_flow_sccm", ar_flow_sccm),
        ("distance_mm", distance_mm),
        ("deposition_time_s", deposition_time_s),
        ("wafer_radius_mm", wafer_radius_mm),
        ("target_radius_mm", target_radius_mm),
        ("racetrack_radius_mm", racetrack_radius_mm),
        ("target_erosion_mm", target_erosion_mm),
    ]:
        if not math.isfinite(_val):
            raise ValueError(f"{_name} must be finite, got {_val}")

    if pressure_mtorr <= 0.0:
        raise ValueError(f"pressure_mtorr must be > 0, got {pressure_mtorr}")
    if ar_flow_sccm <= 0.0:
        raise ValueError(f"ar_flow_sccm must be > 0, got {ar_flow_sccm}")
    if distance_mm <= 0.0:
        raise ValueError(f"distance_mm must be > 0, got {distance_mm}")
    if wafer_radius_mm <= 0.0:
        raise ValueError(f"wafer_radius_mm must be > 0, got {wafer_radius_mm}")
    if target_radius_mm <= 0.0:
        raise ValueError(f"target_radius_mm must be > 0, got {target_radius_mm}")
    if racetrack_radius_mm <= 0.0:
        raise ValueError(f"racetrack_radius_mm must be > 0, got {racetrack_radius_mm}")
    if racetrack_radius_mm >= target_radius_mm:
        raise ValueError(
            f"racetrack_radius_mm ({racetrack_radius_mm}) must be < "
            f"target_radius_mm ({target_radius_mm})"
        )
    if target_erosion_mm < 0.0:
        raise ValueError(f"target_erosion_mm cannot be negative, got {target_erosion_mm}")

    # Resolve target material
    if isinstance(material, str):
        if material not in MATERIALS:
            raise KeyError(f"Unknown material '{material}'. Available: {list(MATERIALS.keys())}")
        mat = MATERIALS[material]
    elif isinstance(material, TargetMaterial):
        mat = material
    else:
        raise TypeError(f"material must be a str or TargetMaterial, got {type(material).__name__}")

    # -----------------------------------------------------------------------
    # Transport geometry — computed unconditionally (reported in result)
    # -----------------------------------------------------------------------
    d_m = distance_mm * 1e-3
    mfp_m = calculate_mean_free_path(pressure_mtorr)
    mfp_mm = mfp_m * 1e3
    kn = calculate_knudsen_number(mfp_m, d_m)

    # -----------------------------------------------------------------------
    # Wafer grid and circular mask
    # -----------------------------------------------------------------------
    x_grid = np.linspace(-wafer_radius_mm, wafer_radius_mm, grid_size)
    y_grid = np.linspace(-wafer_radius_mm, wafer_radius_mm, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    R_mm = np.sqrt(X ** 2 + Y ** 2)

    # Boolean mask: True for grid points that lie inside the circular wafer.
    wafer_mask = R_mm <= wafer_radius_mm
    n_wafer_points = int(wafer_mask.sum())
    if n_wafer_points == 0:
        # Pathological: grid too coarse to hit the wafer — treat as zero.
        raise ValueError(
            f"No grid points fall inside the wafer circle (grid_size={grid_size}, "
            f"wafer_radius_mm={wafer_radius_mm}). Increase grid_size."
        )

    # -----------------------------------------------------------------------
    # Zero-power / zero-time early return
    # -----------------------------------------------------------------------
    if power_w <= 0.0 or deposition_time_s <= 0.0:
        zero_map = np.full((grid_size, grid_size), np.nan)
        zero_map[wafer_mask] = 0.0
        return DepositionResult(
            thickness_map=zero_map,
            mean_thickness_nm=0.0,
            uniformity_percent=0.0,
            std_thickness_nm=0.0,
            deposition_rate_nm_min=0.0,
            min_thickness_nm=0.0,
            max_thickness_nm=0.0,
            x_grid_mm=x_grid,
            y_grid_mm=y_grid,
            discharge_state=None,
            sputter_yield=0.0,
            mean_free_path_mm=float(mfp_mm),
            knudsen_number=float(kn),
            power_w=float(power_w),
            pressure_mtorr=float(pressure_mtorr),
            time_s=float(deposition_time_s),
            n_wafer_points=n_wafer_points,
        )

    # -----------------------------------------------------------------------
    # 1. Plasma discharge state
    # -----------------------------------------------------------------------
    target_radius_m = target_radius_mm * 1e-3
    racetrack_ratio = racetrack_radius_mm / target_radius_mm
    racetrack_width_m = 0.20 * target_radius_m

    discharge_state = calculate_discharge_state(
        power_w=power_w,
        pressure_mtorr=pressure_mtorr,
        ar_flow_sccm=ar_flow_sccm,
        target_radius_m=target_radius_m,
        racetrack_ratio=racetrack_ratio,
        racetrack_width_m=racetrack_width_m,
    )

    # -----------------------------------------------------------------------
    # 2. Sputter yield at sheath ion energy
    # -----------------------------------------------------------------------
    sputter_yield = calculate_sputter_yield(discharge_state.ion_energy_ev, mat)

    # -----------------------------------------------------------------------
    # 3. Deposition rate
    # -----------------------------------------------------------------------
    # Total sputtered atom flux (atoms/s) from ion current — no A_race cancellation.
    # Using total_ion_current_a avoids the ion_flux * A_race round-trip that made
    # racetrack_radius_mm cancel out of the rate (Issue 3 fix).
    sputtered_rate_atoms_s = discharge_state.total_ion_current_a * sputter_yield / _ELEMENTARY_CHARGE_C

    # Thermalization transport efficiency: atoms scattered > N_THERM times are
    # effectively lost to chamber walls before reaching the wafer.
    # eta_transport = 1 / (1 + n_coll / N_THERM), bounded in (0, 1].
    # At 5 mTorr / 80 mm (n_coll ~ 5.15): eta_transport ~ 0.660.
    n_coll = d_m / mfp_m
    eta_transport = 1.0 / (1.0 + n_coll / _N_THERM)

    # Effective rate reaching the wafer (atoms/s)
    effective_rate_atoms_s = sputtered_rate_atoms_s * eta_transport

    # Atomic volume: Omega = M_mol / (rho * N_A)  [m^3/atom]
    atomic_vol_m3 = (mat.atomic_mass * 1e-3) / (mat.density * 1e3 * _AVOGADRO)

    # Distribute over wafer cross-section; apply calibrated chamber efficiency.
    base_flux_m2_s = (effective_rate_atoms_s / (math.pi * (d_m ** 2))) * _ETA_CHAMBER

    # Mean growth rate (m/s) and target mean thickness (nm) after deposition_time_s
    mean_growth_rate_m_s = base_flux_m2_s * atomic_vol_m3
    mean_growth_rate_nm_min = mean_growth_rate_m_s * 1e9 * 60.0
    target_mean_nm = mean_growth_rate_nm_min * (deposition_time_s / 60.0)

    # -----------------------------------------------------------------------
    # 4. Radial profile (Issues 1, 3, 4 combined)
    # -----------------------------------------------------------------------
    # Scattering broadening factor BF = sqrt(1 + d/lambda).
    # As pressure increases: BF grows → c_eff shrinks → film is more uniform.
    # As racetrack radius increases: (r_race/d)^2 grows → c_eff grows → less uniform.
    # c_eff is always positive and physically bounded (no singularity).
    broadening_factor = calculate_scattering_broadening(pressure_mtorr, d_m)
    c_eff = _GAMMA_PROFILE * (racetrack_radius_mm / distance_mm) ** 2 / (broadening_factor ** 2)

    # Target erosion collimation: deeper groove slightly collimates emission,
    # increasing centre-peaking and hence non-uniformity.
    c_eff = c_eff * (1.0 + 0.10 * target_erosion_mm)

    # Normalised radial coordinate squared: r_norm^2 = (R/wafer_radius)^2
    r_norm_sq = (R_mm / wafer_radius_mm) ** 2.0

    # mean_r2 over on-wafer points only (Issue 2 fix for profile normalisation)
    mean_r2 = float(np.mean(r_norm_sq[wafer_mask]))

    # Cap c_eff to ensure the profile remains strictly positive across the entire wafer,
    # preventing non-physical zero-clipping and preserving strict mass conservation.
    c_eff = min(c_eff, 0.95)

    # Solve for the profile centre value t_0 such that the on-wafer mean equals
    # target_mean_nm:
    #   mean_h = t_0 * (1 - c_eff * mean_r2)  =>  t_0 = target_mean_nm / (1 - c_eff * mean_r2)
    denominator = 1.0 - c_eff * mean_r2
    t_0 = target_mean_nm / denominator

    # Full 2D profile map (over entire grid, including off-wafer corners)
    profile_map = t_0 * (1.0 - c_eff * r_norm_sq)
    profile_map = np.maximum(0.0, profile_map)

    # -----------------------------------------------------------------------
    # 5. Apply circular wafer mask (Issue 2 fix)
    # -----------------------------------------------------------------------
    # Set off-wafer points to NaN so they are excluded from visualisation and
    # cannot accidentally contaminate downstream statistics.
    thickness_map = np.where(wafer_mask, profile_map, np.nan)

    # All statistics computed exclusively over on-wafer points.
    on_wafer_values = thickness_map[wafer_mask]
    mean_thick = float(np.mean(on_wafer_values))
    min_thick  = float(np.min(on_wafer_values))
    max_thick  = float(np.max(on_wafer_values))
    std_thick  = float(np.std(on_wafer_values))

    if mean_thick > 0.0:
        unif_percent = float((max_thick - min_thick) / (2.0 * mean_thick) * 100.0)
    else:
        unif_percent = 0.0

    rate_nm_min = float(mean_thick / (deposition_time_s / 60.0))

    return DepositionResult(
        thickness_map=thickness_map,
        mean_thickness_nm=mean_thick,
        uniformity_percent=unif_percent,
        std_thickness_nm=std_thick,
        deposition_rate_nm_min=rate_nm_min,
        min_thickness_nm=min_thick,
        max_thickness_nm=max_thick,
        x_grid_mm=x_grid,
        y_grid_mm=y_grid,
        discharge_state=discharge_state,
        sputter_yield=float(sputter_yield),
        mean_free_path_mm=float(mfp_mm),
        knudsen_number=float(kn),
        power_w=float(power_w),
        pressure_mtorr=float(pressure_mtorr),
        time_s=float(deposition_time_s),
        n_wafer_points=n_wafer_points,
    )
