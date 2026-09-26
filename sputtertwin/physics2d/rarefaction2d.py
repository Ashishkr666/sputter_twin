"""2D Gas Rarefaction and Sputter Wind Thermal Fluid Model.

Models the momentum and thermal energy transfer from high-energy sputtered
metal atoms (Cu/Ti, ~5-10 eV) to the ambient Argon background gas:
1. 2D Gas Temperature Field T_gas(r, z) due to collisional collisional thermalization.
2. Local gas rarefaction: n_g(r, z) = P / (k_B * T_gas(r, z)).
3. Rarefaction factor eta_rare(r, z) = n_g(r, z) / n_g,0.
4. Reduced target-front effective pressure leading to voltage rise at high powers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Physical constants
_K_B_J_K: float = 1.380649e-23  # J/K
_E_CHARGE: float = 1.602176634e-19  # C


@dataclass
class GasRarefaction2D:
    """2D Gas Heating and Rarefaction Profile."""

    r_grid_m: np.ndarray        # 1D radial coordinates (m)
    z_grid_m: np.ndarray        # 1D axial coordinates (m)
    T_gas_2d_K: np.ndarray      # 2D background gas temperature (K)
    n_gas_2d_m3: np.ndarray     # 2D background gas number density (m^-3)
    rarefaction_factor_2d: np.ndarray # n_g(r, z) / n_g_ambient (<= 1.0)
    peak_gas_temp_k: float      # Maximum gas temperature in front of racetrack (K)
    min_rarefaction: float      # Minimum density ratio (peak rarefaction)
    effective_pressure_reduction_pct: float # Percent drop in target-front pressure


def compute_gas_rarefaction_2d(
    power_w: float,
    pressure_mtorr: float,
    racetrack_radius_m: float = 0.025,
    racetrack_fwhm_m: float = 0.010,
    target_radius_m: float = 0.050,
    anode_distance_m: float = 0.060,
    grid_r: int = 60,
    grid_z: int = 50,
    ambient_temp_k: float = 300.0,
    mean_sputter_energy_ev: float = 6.0,  # Average Thompson ejected atom energy ~ 6 eV
    energy_transfer_factor: float = 0.70, # Ar-Cu energy transfer efficiency ~ 70%
) -> GasRarefaction2D:
    """Calculate 2D gas temperature rise and density rarefaction in front of the target.

    Args:
        power_w: Cathode electrical power (W).
        pressure_mtorr: Argon setpoint pressure (mTorr).
        racetrack_radius_m: Mean radius of erosion ring (m).
        racetrack_fwhm_m: Width of racetrack ring (m).
        target_radius_m: Target radius (m).
        anode_distance_m: Spacing to substrate (m).
        grid_r: Radial grid resolution.
        grid_z: Axial grid resolution.
        ambient_temp_k: Room temperature (K).
        mean_sputter_energy_ev: Mean energy of sputtered particles (eV).
        energy_transfer_factor: Fractional momentum transfer per collision.

    Returns:
        GasRarefaction2D with complete 2D gas temperature and density fields.
    """
    r_vec = np.linspace(0.0, target_radius_m, grid_r)
    z_vec = np.linspace(0.0, anode_distance_m, grid_z)
    R, Z = np.meshgrid(r_vec, z_vec)

    # Ambient baseline gas density n_g0 = P / (k_B * T_0)
    p_pa = pressure_mtorr * 0.133322
    n_g0 = p_pa / (_K_B_J_K * ambient_temp_k)

    # Sputter wind power dissipation scales with cathode power and sputter yield
    # Higher power -> greater metal flux -> more gas heating
    # Delta T_max ~ 100 K at 100 W, up to ~ 450 K at 450 W
    delta_t_max = 85.0 * (power_w / 100.0) * (5.0 / max(pressure_mtorr, 1.0)) ** 0.3

    # Spatial heating profile: localized immediately above the racetrack ring (z < 25 mm)
    sigma_r = max(racetrack_fwhm_m * 0.8, 0.006)
    sigma_z = 0.018  # Thermalization mean free path scale

    radial_envelope = np.exp(-0.5 * ((R - racetrack_radius_m) / sigma_r) ** 2)
    axial_envelope = np.exp(-Z / sigma_z)
    heating_shape = radial_envelope * axial_envelope

    # 2D Gas Temperature Field
    T_gas_2d = ambient_temp_k + delta_t_max * heating_shape

    # 2D Gas Density Field: isobaric expansion n_g(r, z) = P / (k_B * T_g)
    n_gas_2d = p_pa / (_K_B_J_K * T_gas_2d)

    # Rarefaction factor: ratio of local density to ambient
    rarefaction_factor = n_gas_2d / n_g0

    # Summary metrics
    peak_temp = float(np.max(T_gas_2d))
    min_rare = float(np.min(rarefaction_factor))
    pressure_drop_pct = (1.0 - min_rare) * 100.0

    return GasRarefaction2D(
        r_grid_m=r_vec,
        z_grid_m=z_vec,
        T_gas_2d_K=T_gas_2d,
        n_gas_2d_m3=n_gas_2d,
        rarefaction_factor_2d=rarefaction_factor,
        peak_gas_temp_k=peak_temp,
        min_rarefaction=min_rare,
        effective_pressure_reduction_pct=pressure_drop_pct,
    )
