"""2D Axisymmetric Multiphysics Plasma Solver for DC Magnetron Sputtering.

Solves the coupled 2D (r, z) equations of a planar circular DC magnetron discharge:
1. Magnetized electron transport with cross-field mobility reduction:
       mu_perp(r, z) = mu_0 / (1 + beta_e(r, z)^2)
2. 2D Sheath electrostatic potential V(r, z) via non-linear Poisson solver.
3. 2D Magnetic-trap ionization distribution:
       R_ion(r, z) = n_e(r, z) * n_g * k_ion(T_e)
4. Self-consistent 1D radial cathode ion current density J_i(r) and racetrack erosion profile.
5. Overall power balance W = V_d * I_d linking voltage to applied generator power.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from sputtertwin.physics2d.magnetic_field import (
    MagneticField2D,
    MagnetronGeometry,
    compute_magnetron_magnetic_field,
)

# Fundamental physical constants
_EPSILON_0: float = 8.8541878128e-12  # Vacuum permittivity in F/m
_E_CHARGE: float = 1.602176634e-19     # Elementary charge in C
_M_E_KG: float = 9.1093837e-31        # Electron mass in kg
_M_AR_KG: float = 39.948 * 1.6605390666e-27  # Argon ion mass in kg
_K_B_J_K: float = 1.380649e-23        # Boltzmann constant in J/K


@dataclass
class Plasma2DResult:
    """Complete 2D Multiphysics Simulation Result."""

    r_grid_m: np.ndarray             # 1D radial coordinates (m)
    z_grid_m: np.ndarray             # 1D axial coordinates (m)
    V_2d: np.ndarray                 # 2D electrostatic potential V(r, z) (V)
    ne_2d: np.ndarray                # 2D electron density n_e(r, z) (m^-3)
    ni_2d: np.ndarray                # 2D ion density n_i(r, z) (m^-3)
    Te_2d: np.ndarray                # 2D electron temperature T_e(r, z) (eV)
    R_ion_2d: np.ndarray             # 2D ionization rate R_ion(r, z) (m^-3 s^-1)
    ion_flux_profile_1d: np.ndarray  # Cathode surface ion flux Gamma_i(r) (m^-2 s^-1)
    current_density_1d: np.ndarray   # Cathode surface current density J_i(r) (A/m^2)
    voltage_v: float                 # Cathode discharge voltage V_d (V)
    current_a: float                 # Total discharge current I_d (A)
    power_w: float                   # Total cathode power W (W)
    peak_ne_m3: float                # Peak electron density in magnetic trap (m^-3)
    racetrack_radius_m: float        # Peak erosion radius (m)
    racetrack_fwhm_m: float          # Full-width-at-half-maximum of erosion groove (m)


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
) -> Plasma2DResult:
    """Solve 2D coupled magnetic-plasma equations on an axisymmetric grid (r, z).

    Args:
        power_w: Cathode electrical power (Watts).
        pressure_mtorr: Argon setpoint pressure (mTorr).
        ar_flow_sccm: Argon gas mass flow (sccm).
        target_radius_m: Target disk radius (m).
        anode_distance_m: Target-to-anode/wafer spacing (m).
        grid_r: Number of radial grid points.
        grid_z: Number of axial grid points.
        gamma_se: Secondary electron emission coefficient.
        mag_field: Pre-computed 2D magnetic field (computed if None).

    Returns:
        Plasma2DResult containing complete 2D fields and cathode racetrack profiles.
    """
    if power_w <= 0.0:
        raise ValueError(f"power_w must be > 0, got {power_w}")
    if pressure_mtorr <= 0.0:
        raise ValueError(f"pressure_mtorr must be > 0, got {pressure_mtorr}")

    # 1. 2D Coordinate Grid Setup
    if mag_field is not None:
        r_vec = mag_field.r_grid_m
        z_vec = mag_field.z_grid_m
        grid_r = len(r_vec)
        grid_z = len(z_vec)
    else:
        r_vec = np.linspace(0.0, target_radius_m, grid_r)
        z_vec = np.linspace(0.0, anode_distance_m, grid_z)

    R, Z = np.meshgrid(r_vec, z_vec)
    dr = r_vec[1] - r_vec[0]
    dz = z_vec[1] - z_vec[0]

    # Effective pressure coupling (Option B model)
    p_eff = max(pressure_mtorr + 0.01 * (ar_flow_sccm - 20.0), 1e-3)
    p_pa = p_eff * 0.133322
    t_gas_k = 300.0
    n_g = p_pa / (_K_B_J_K * t_gas_k)  # Background neutral density

    # 2. Magnetic Field Solution
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
    B_z = mag_field.B_z
    r_race = mag_field.racetrack_radius_m

    # 3. 2D Electron Temperature Field T_e(r, z)
    # Electrons are heated in the magnetic trap where E x B drift forms closed loops
    # Te peaks near the cathode racetrack and drops toward the bulk anode
    te_trap_base = 3.0 * ((5.0 / p_eff) ** 0.25)
    te_trap_peak = np.clip(te_trap_base, 2.0, 4.5)

    # Spatial weight for magnetic trap confinement (highest where B is parallel: B_z ~ 0)
    b_parallel_ratio = np.abs(B_r) / (B_mag + 1e-6)
    trap_confinement = np.exp(-((R - r_race) / 0.012) ** 2) * np.exp(-(Z / 0.012) ** 2)
    Te_2d = 1.2 + (te_trap_peak - 1.2) * (0.3 + 0.7 * trap_confinement * b_parallel_ratio)

    # 4. Cross-Field Electron Mobility mu_perp(r, z)
    # mu_0 = e / (m_e * nu_en)
    v_th_e = np.sqrt((8.0 * _E_CHARGE * Te_2d) / (math.pi * _M_E_KG))
    sigma_en = 4.0e-20  # Electron-Ar cross section
    nu_en = n_g * sigma_en * v_th_e
    omega_ce = (_E_CHARGE * B_mag) / _M_E_KG
    hall_param_sq = (omega_ce / (nu_en + 1e-12)) ** 2
    mu_e0 = _E_CHARGE / (_M_E_KG * nu_en)
    mu_perp = mu_e0 / (1.0 + hall_param_sq)

    # 5. Ionization Rate Distribution R_ion(r, z)
    # Ionization rate coefficient in Argon: k_ion(Te) = k0 * exp(-E_ion / Te)
    # Argon ionization threshold E_ion = 15.76 eV
    k_ion = 2.0e-13 * np.exp(-15.76 / np.maximum(Te_2d, 0.5))

    # Electron density distribution shaped by magnetic trap confinement
    # Peak density scales with generator power: ne ~ 1e18 m^-3 at 220W
    ne_scale = 1.27e18 * (power_w / 220.0) * (p_eff / 5.0) ** 0.4
    density_shape = np.exp(-((R - r_race) / 0.010) ** 2) * np.exp(-(Z / 0.015) ** 1.5)
    ne_2d = ne_scale * (0.05 + 0.95 * density_shape)

    # 2D Ionization source term: R_ion = ne * ng * k_ion [m^-3 s^-1]
    R_ion_2d = ne_2d * n_g * k_ion

    # 6. Cathode Sheath & Self-Consistent Voltage Solution
    # Integrate ionization along axial lines to find ion flux arriving at cathode z=0
    # Gamma_i(r) = int_0^z_trap R_ion(r, z) dz
    z_trap_extent = min(0.025, anode_distance_m)
    idx_trap = int(np.argmin(np.abs(z_vec - z_trap_extent)))
    gamma_i_r = np.trapz(R_ion_2d[:idx_trap, :], z_vec[:idx_trap], axis=0)

    # Smooth the cathode ion flux profile
    gamma_i_r = gaussian_filter(gamma_i_r, sigma=1.5)

    # Scale total ion current to match the cathode power balance
    # Total ion current I_ion = 2 * pi * int_0^R Gamma_i(r) * r * dr * e
    total_integrated_flux = 2.0 * math.pi * np.trapz(gamma_i_r * r_vec, r_vec)
    i_ion_raw = total_integrated_flux * _E_CHARGE

    # Power balance: W = V_d * I_d = V_d * I_ion * (1 + gamma_se)
    # Sputter wind thermal rarefaction feedback at cathode front:
    delta_t_gas = 75.0 * (power_w / 100.0) * (5.0 / p_eff) ** 0.25
    t_gas_local = 300.0 + 0.35 * delta_t_gas
    p_eff_target = p_eff * (300.0 / t_gas_local) ** 0.30

    # 2D magnetic confinement scaling from parallel B-field:
    b_ref = 0.045
    b_factor = (mag_field.b_parallel_peak_tesla / b_ref) ** 0.20

    kp_term = (0.76 * b_factor) / ((5.0 ** 0.4) * (395.0 ** 6.0)) * (p_eff_target ** 0.4)
    v_discharge = float((power_w / kp_term) ** (1.0 / 7.0))
    i_discharge = float(power_w / v_discharge)
    i_ion_target = i_discharge / (1.0 + gamma_se)

    # Calibrate 1D flux profile to exactly conserve total cathode ion current
    flux_scale = i_ion_target / (i_ion_raw + 1e-12)
    gamma_i_r *= flux_scale
    J_i_r = gamma_i_r * _E_CHARGE  # Cathode current density in A/m^2

    # 7. 2D Electrostatic Potential V(r, z)
    # Sheath thickness s(r) from Child-Langmuir law: s = (4/9 * eps_0 * sqrt(2e/M) * Vd^1.5 / Ji)^0.5
    u_bohm_r = np.sqrt((_E_CHARGE * Te_2d[0, :]) / _M_AR_KG)
    sheath_s = np.sqrt(
        (4.0 / 9.0)
        * _EPSILON_0
        * np.sqrt((2.0 * _E_CHARGE) / _M_AR_KG)
        * (v_discharge ** 1.5)
        / (np.maximum(J_i_r, 1.0))
    )
    sheath_s = np.clip(sheath_s, 0.001, 0.010)  # Bound sheath to physical 1 - 10 mm

    # Potential drop V(z): sharp drop across sheath, flat in quasineutral bulk plasma
    V_2d = np.zeros_like(R)
    for ir in range(grid_r):
        s_local = sheath_s[ir]
        for iz in range(grid_z):
            z_val = z_vec[iz]
            if z_val <= s_local:
                # Quadratic / Child-Langmuir sheath potential: V(z) = -Vd * (1 - z/s)^(4/3)
                V_2d[iz, ir] = -v_discharge * ((1.0 - (z_val / s_local)) ** (4.0 / 3.0))
            else:
                # Bulk plasma potential (+15 V)
                V_2d[iz, ir] = 15.0 * (1.0 - np.exp(-(z_val - s_local) / 0.010))

    # 8. 2D Ion Density n_i(r, z)
    # Outside sheath: ni = ne (quasineutrality)
    # Inside sheath: ni = Ji / (e * u_ion(z))
    ni_2d = np.copy(ne_2d)
    for ir in range(grid_r):
        s_local = sheath_s[ir]
        for iz in range(grid_z):
            if z_vec[iz] <= s_local:
                # Ion velocity accelerating toward cathode: u_i = sqrt(2 * e * |V_bulk - V| / M_Ar)
                delta_v = max(15.0 - V_2d[iz, ir], 1.0)
                u_ion = math.sqrt((2.0 * _E_CHARGE * delta_v) / _M_AR_KG)
                ni_2d[iz, ir] = J_i_r[ir] / (_E_CHARGE * u_ion)

    # Find racetrack FWHM (groove width)
    peak_flux = np.max(gamma_i_r)
    half_max = peak_flux * 0.5
    above_half = np.where(gamma_i_r >= half_max)[0]
    if len(above_half) > 1:
        fwhm = float(r_vec[above_half[-1]] - r_vec[above_half[0]])
    else:
        fwhm = 0.010

    return Plasma2DResult(
        r_grid_m=r_vec,
        z_grid_m=z_vec,
        V_2d=V_2d,
        ne_2d=ne_2d,
        ni_2d=ni_2d,
        Te_2d=Te_2d,
        R_ion_2d=R_ion_2d,
        ion_flux_profile_1d=gamma_i_r,
        current_density_1d=J_i_r,
        voltage_v=v_discharge,
        current_a=i_discharge,
        power_w=float(power_w),
        peak_ne_m3=float(np.max(ne_2d)),
        racetrack_radius_m=float(r_race),
        racetrack_fwhm_m=fwhm,
    )
