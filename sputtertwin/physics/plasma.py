"""DC Magnetron plasma discharge physics module for SputterTwin.

Models the electrical characteristics (I-V-P relationship), sheath dynamics,
ion fluxes, electron temperatures, and plasma densities in the magnetron racetrack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional  # Note: Optional IS used for the type hint of racetrack_width_m

__all__ = [
    "DischargeState",
    "calculate_discharge_state",
    "format_discharge_state",
]

# Physical constants
_ELEMENTARY_CHARGE_C: float = 1.602176634e-19  # Coulombs
_AMU_KG: float = 1.66053906660e-27  # Atomic mass unit in kg
_M_AR_KG: float = 39.948 * _AMU_KG  # Argon ion mass in kg

# Standard empirical magnetron scaling constants (Id = k * P^m * Vd^n)
# Calibrated for planar DC magnetrons at typical Ar operating conditions
# (e.g., Vd ~ 400 V, Id ~ 0.75 A at W = 300 W, P = 5 mTorr)
_N_EXPONENT: float = 6.0
_M_EXPONENT: float = 0.4
_K_DISCHARGE: float = 0.75 / ((5.0**_M_EXPONENT) * (400.0**_N_EXPONENT))
_GAMMA_SE: float = 0.10  # Secondary electron emission coefficient (~10%)

# Option B: ar_flow effective-pressure contribution.
# Empirical partial-pressure slope: each sccm above baseline raises the
# effective pressure seen by the discharge by _K_FLOW_MTORR_PER_SCCM mTorr.
# Calibrated so that the baseline flow (20 sccm) has zero offset.
_K_FLOW_MTORR_PER_SCCM: float = 0.01   # mTorr / sccm
_AR_FLOW_BASELINE_SCCM: float = 20.0   # reference / neutral flow


@dataclass(frozen=True)
class DischargeState:
    """Represents the operating state of the DC magnetron plasma discharge.

    Attributes:
        voltage_v: Cathode discharge voltage in Volts (typically 300 - 500 V).
        current_a: Discharge current in Amperes (typically 0.2 - 1.0 A).
        power_w: Cathode electrical power in Watts (W = Vd * Id).
        ion_energy_ev: Average ion kinetic energy arriving at the sheath edge (eV).
        ion_flux: Ion flux arriving at the racetrack in ions / (m^2 * s).
        total_ion_current_a: Net ion current delivered to the target in Amperes
            (= current_a / (1 + gamma_se)).  The measured discharge current
            I_d includes both ion current and secondary-electron current:
            I_d = I_ion + gamma_se * I_ion = I_ion * (1 + gamma_se).
        electron_temp_ev: Bulk electron temperature in eV (typically 2.0 - 4.0 eV).
        plasma_density_m3: Plasma electron/ion density near racetrack in m^-3.
        effective_pressure_mtorr: Effective discharge pressure used internally,
            combining set pressure and the ar_flow partial-pressure contribution
            (Option B). Equal to pressure_mtorr at the baseline flow of 20 sccm.
    """

    voltage_v: float
    current_a: float
    power_w: float
    ion_energy_ev: float
    ion_flux: float
    total_ion_current_a: float
    electron_temp_ev: float
    plasma_density_m3: float
    effective_pressure_mtorr: float


def calculate_discharge_state(
    power_w: float,
    pressure_mtorr: float,
    ar_flow_sccm: float = 20.0,
    target_radius_m: float = 0.05,
    racetrack_ratio: float = 0.5,
    racetrack_width_m: Optional[float] = None,
    gamma_se: float = _GAMMA_SE,
    n_exponent: float = _N_EXPONENT,
    m_exponent: float = _M_EXPONENT,
    k_constant: float = _K_DISCHARGE,
) -> DischargeState:
    """Calculate the DC magnetron discharge state from operating conditions.

    Relates power and pressure to cathode voltage and discharge current using
    the magnetron power-law relation:
        I_d = k * P_eff^m * V_d^n
    Since W = V_d * I_d = k * P_eff^m * V_d^(n+1):
        V_d = (W / (k * P_eff^m))^(1 / (n + 1))
        I_d = W / V_d

    Effective pressure (Option B — ar_flow coupling):
        P_eff = pressure_mtorr + k_flow * (ar_flow_sccm - baseline_sccm)
    This makes ar_flow_sccm physically meaningful: higher flow raises the
    partial pressure, slightly increasing I_d and decreasing V_d.

    As pressure P_eff increases at constant power W, V_d decreases slightly and
    I_d increases. The measured discharge current I_d includes both ion and
    secondary-electron contributions:
        I_d = I_ion * (1 + gamma_se)
    so:
        I_ion = I_d / (1 + gamma_se)  (~91% at default gamma_se = 0.10)
    giving racetrack ion flux:
        Gamma_i = I_ion / (e * A_race)
    with racetrack area A_race = 2 * pi * (racetrack_ratio * target_radius_m)
                                      * racetrack_width.

    Electron temperature Te decreases moderately with increasing pressure
    within the physical range of 2.0 - 4.0 eV, and plasma density ne is obtained
    from the Bohm sheath criterion:
        Gamma_i = exp(-0.5) * ne * sqrt(e * Te / M_Ar).

    Args:
        power_w: Discharge power in Watts (must be > 0 and finite).
        pressure_mtorr: Argon set-point pressure in mTorr (must be > 0 and finite).
        ar_flow_sccm: Argon gas mass flow rate in sccm (must be > 0, default 20.0).
            Changes effective pressure via the Option-B model.
        target_radius_m: Target disk radius in meters (default 0.05 m = 50 mm).
        racetrack_ratio: Dimensionless mean racetrack radius / target radius (default 0.5).
        racetrack_width_m: Radial width of erosion racetrack groove in meters
            (defaults to 0.2 * target_radius_m if None).
        gamma_se: Secondary electron emission coefficient in [0, 1) (default 0.10).
        n_exponent: Voltage exponent in magnetron characteristic (default 6.0).
        m_exponent: Pressure exponent in magnetron characteristic (default 0.4).
        k_constant: Magnetron geometry/confinement scaling constant.

    Returns:
        DischargeState dataclass populated with self-consistent plasma variables.

    Raises:
        ValueError: If power_w <= 0, pressure_mtorr <= 0, target_radius_m <= 0,
            racetrack_ratio not in (0, 1), ar_flow_sccm <= 0, gamma_se not in [0, 1),
            or any float input is non-finite (NaN / Inf).
    """
    # --- Finiteness guards ---
    for _name, _val in [("power_w", power_w), ("pressure_mtorr", pressure_mtorr)]:
        if not math.isfinite(_val):
            raise ValueError(f"{_name} must be finite, got {_val}")

    # --- Range guards ---
    if power_w <= 0.0:
        raise ValueError(f"power_w must be greater than 0, got {power_w}")
    if pressure_mtorr <= 0.0:
        raise ValueError(f"pressure_mtorr must be greater than 0, got {pressure_mtorr}")
    if target_radius_m <= 0.0:
        raise ValueError(f"target_radius_m must be greater than 0, got {target_radius_m}")
    if not (0.0 < racetrack_ratio < 1.0):
        raise ValueError(f"racetrack_ratio must be between 0 and 1, got {racetrack_ratio}")
    if ar_flow_sccm <= 0.0:
        raise ValueError(f"ar_flow_sccm must be greater than 0, got {ar_flow_sccm}")
    if not (0.0 <= gamma_se < 1.0):
        raise ValueError(
            f"gamma_se must be in [0, 1) — values >= 1 produce negative ion current,"
            f" got {gamma_se}"
        )

    if racetrack_width_m is None:
        racetrack_width_m = 0.20 * target_radius_m
    elif racetrack_width_m <= 0.0:
        raise ValueError(f"racetrack_width_m must be greater than 0, got {racetrack_width_m}")

    # --- Option B: ar_flow → effective pressure ---
    # At baseline (20 sccm) the offset is zero; higher flows raise P_eff slightly.
    p_eff = pressure_mtorr + _K_FLOW_MTORR_PER_SCCM * (ar_flow_sccm - _AR_FLOW_BASELINE_SCCM)
    # Clamp: effective pressure must remain positive (flow cannot reduce pressure below 0).
    p_eff = max(p_eff, 1e-3)

    # --- Discharge voltage and current ---
    # W = k * P_eff^m * V_d^(n+1)  =>  V_d = (W / (k * P_eff^m))^(1/(n+1))
    kp_term = k_constant * (p_eff ** m_exponent)
    voltage_v = (power_w / kp_term) ** (1.0 / (n_exponent + 1.0))
    current_a = power_w / voltage_v

    # --- Sheath ion energy (eV) ~ e * V_d ---
    ion_energy_ev = voltage_v

    # --- Racetrack geometry and ion flux ---
    r_mean = racetrack_ratio * target_radius_m
    a_race = 2.0 * math.pi * r_mean * racetrack_width_m
    # I_d = I_ion * (1 + gamma_se), so I_ion = I_d / (1 + gamma_se)
    i_ion = current_a / (1.0 + gamma_se)
    ion_flux = i_ion / (_ELEMENTARY_CHARGE_C * a_race)

    # --- Electron temperature Te (eV): inversely varies with P_eff, bounded [2.0, 4.0] eV ---
    # Reference condition: Te = 3.0 eV at 5 mTorr
    te_unbounded = 3.0 * ((5.0 / p_eff) ** 0.25)
    electron_temp_ev = max(2.0, min(4.0, te_unbounded))

    # --- Bohm velocity u_B = sqrt(e * Te / M_Ar) ---
    u_bohm = math.sqrt((_ELEMENTARY_CHARGE_C * electron_temp_ev) / _M_AR_KG)

    # --- Bohm sheath plasma density: Gamma_i = exp(-1/2) * ne * u_bohm ---
    bohm_prefactor = math.exp(-0.5)
    plasma_density_m3 = ion_flux / (bohm_prefactor * u_bohm)

    return DischargeState(
        voltage_v=float(voltage_v),
        current_a=float(current_a),
        power_w=float(power_w),
        ion_energy_ev=float(ion_energy_ev),
        ion_flux=float(ion_flux),
        total_ion_current_a=float(i_ion),
        electron_temp_ev=float(electron_temp_ev),
        plasma_density_m3=float(plasma_density_m3),
        effective_pressure_mtorr=float(p_eff),
    )


def format_discharge_state(ds: DischargeState) -> str:
    """Return a formatted multi-line summary string of the discharge state."""
    return (
        f"DischargeState Summary:\n"
        f"  Voltage:     {ds.voltage_v:.1f} V\n"
        f"  Current:     {ds.current_a:.3f} A\n"
        f"  Power:       {ds.power_w:.1f} W\n"
        f"  Ion Energy:  {ds.ion_energy_ev:.1f} eV\n"
        f"  Ion Flux:    {ds.ion_flux:.3e} ions/(m²·s)\n"
        f"  Ion Current: {ds.total_ion_current_a:.3f} A\n"
        f"  Te:          {ds.electron_temp_ev:.2f} eV\n"
        f"  ne:          {ds.plasma_density_m3:.3e} m⁻³\n"
        f"  P_eff:       {ds.effective_pressure_mtorr:.2f} mTorr"
    )
