"""Gas transport and collisional scattering module for SputterTwin.

Implements kinetic theory calculations for sputtered atom transport through
ambient argon gas, including mean free path, Knudsen number, ballistic
transmission probability, and collisional scattering broadening.
"""

from __future__ import annotations

import math
from typing import TypedDict

__all__ = [
    "calculate_mean_free_path",
    "calculate_knudsen_number",
    "calculate_transmission_probability",
    "calculate_scattering_broadening",
    "calculate_transport_summary",
    "TransportSummary",
]


class TransportSummary(TypedDict):
    """Structured return type for :func:`calculate_transport_summary`."""

    mean_free_path_mm: float
    knudsen_number: float
    transmission_probability: float
    broadening_factor: float
    n_collisions: float
    regime: str

# Physical constants
_K_BOLTZMANN_J_K: float = 1.380649e-23  # Boltzmann constant in J/K
_PA_PER_MTORR: float = 0.133322  # 1 mTorr in Pascals
_DEFAULT_CROSS_SECTION_M2: float = 4.0e-19  # Typical Ar-metal cross section (~0.36 nm diameter)


def calculate_mean_free_path(
    pressure_mtorr: float,
    gas_temp_k: float = 300.0,
    cross_section_m2: float = _DEFAULT_CROSS_SECTION_M2,
) -> float:
    """Calculate the mean free path of sputtered particles in background gas.

    Uses classical gas kinetic theory:
        P_pa = pressure_mtorr * 0.133322
        n_g = P_pa / (k_B * T)
        lambda_mfp = 1 / (n_g * sigma) = (k_B * T) / (P_pa * sigma)

    Args:
        pressure_mtorr: Ambient gas pressure in mTorr (must be > 0).
        gas_temp_k: Background gas temperature in Kelvin (default 300.0 K).
        cross_section_m2: Collisional cross section in m^2
            (default 4.0e-19 m^2, corresponding to collision diameter ~0.36 nm).

    Returns:
        Mean free path in meters.

    Raises:
        ValueError: If pressure_mtorr <= 0, gas_temp_k <= 0, or cross_section_m2 <= 0.
    """
    if not math.isfinite(pressure_mtorr):
        raise ValueError(f"pressure_mtorr must be finite, got {pressure_mtorr}")
    if not math.isfinite(gas_temp_k):
        raise ValueError(f"gas_temp_k must be finite, got {gas_temp_k}")
        
    if pressure_mtorr <= 0.0:
        raise ValueError(f"pressure_mtorr must be greater than 0, got {pressure_mtorr}")
    if gas_temp_k <= 0.0:
        raise ValueError(f"gas_temp_k must be greater than 0, got {gas_temp_k}")
    if cross_section_m2 <= 0.0:
        raise ValueError(f"cross_section_m2 must be greater than 0, got {cross_section_m2}")

    p_pa = pressure_mtorr * _PA_PER_MTORR
    n_g = p_pa / (_K_BOLTZMANN_J_K * gas_temp_k)
    mean_free_path_m = 1.0 / (n_g * cross_section_m2)

    return float(mean_free_path_m)


def calculate_knudsen_number(mean_free_path_m: float, distance_m: float) -> float:
    """Calculate the Knudsen number for sputtered transport.

    Knudsen number characterizes the collisional transport regime:
        Kn = lambda_mfp / distance
    - Kn > 1.0: Ballistic / free-molecular regime (minimal scattering).
    - 0.1 <= Kn <= 1.0: Transition regime.
    - Kn < 0.1: Continuum / diffusive regime (multiple collisions).

    Args:
        mean_free_path_m: Particle mean free path in meters (must be >= 0).
        distance_m: Characteristic transport distance (e.g., target-to-substrate)
            in meters (must be > 0).

    Returns:
        Knudsen number (dimensionless float).

    Raises:
        ValueError: If distance_m <= 0 or mean_free_path_m < 0.
    """
    if not math.isfinite(distance_m):
        raise ValueError(f"distance_m must be finite, got {distance_m}")
    if distance_m <= 0.0:
        raise ValueError(f"distance_m must be greater than 0, got {distance_m}")
    if mean_free_path_m < 0.0:
        raise ValueError(f"mean_free_path_m cannot be negative, got {mean_free_path_m}")

    return float(mean_free_path_m / distance_m)


def calculate_transmission_probability(
    distance_m: float, mean_free_path_m: float
) -> float:
    """Calculate the unscattered ballistic transmission probability.

    The fraction of sputtered atoms arriving at distance d without experiencing
    any gas collisions is given by the Beer-Lambert attenuation:
        P_ballistic = exp(-distance_m / mean_free_path_m)

    Args:
        distance_m: Transport distance in meters (must be >= 0).
        mean_free_path_m: Mean free path in meters (must be > 0).

    Returns:
        Ballistic fraction between 0.0 and 1.0.

    Raises:
        ValueError: If distance_m < 0 or mean_free_path_m <= 0.
    """
    if not math.isfinite(distance_m):
        raise ValueError(f"distance_m must be finite, got {distance_m}")
    if distance_m < 0.0:
        raise ValueError(f"distance_m cannot be negative, got {distance_m}")
    if mean_free_path_m <= 0.0:
        raise ValueError(f"mean_free_path_m must be greater than 0, got {mean_free_path_m}")

    if distance_m == 0.0:
        return 1.0

    return float(math.exp(-distance_m / mean_free_path_m))


def calculate_scattering_broadening(
    pressure_mtorr: float,
    distance_m: float,
    gas_temp_k: float = 300.0,
    broadening_coefficient: float = 1.0,
) -> float:
    """Calculate the angular spread / diffusion broadening factor due to gas collisions.

    As sputtered atoms traverse the gas, random collisions deflect them from
    their initial ballistic trajectories. Based on random-walk collisional
    dispersion, the spatial/angular broadening relative to ballistic transport
    scales as:
        N_coll = distance_m / lambda_mfp
        Broadening = sqrt(1.0 + broadening_coefficient * N_coll)

    When distance_m == 0 or pressure_mtorr -> 0, Broadening == 1.0 (pure ballistic).
    As pressure and distance increase, Broadening increases monotonically above 1.0.

    Args:
        pressure_mtorr: Background gas pressure in mTorr (must be > 0).
        distance_m: Target-to-substrate transport distance in meters (must be >= 0).
        gas_temp_k: Gas temperature in Kelvin (default 300.0 K).
        broadening_coefficient: Dimensionless scaling coefficient for deflection
            per collision (default 1.0).

    Returns:
        Broadening factor (dimensionless float >= 1.0).

    Raises:
        ValueError: If pressure_mtorr <= 0, distance_m < 0, gas_temp_k <= 0,
            or broadening_coefficient < 0.
    """
    if not math.isfinite(pressure_mtorr):
        raise ValueError(f"pressure_mtorr must be finite, got {pressure_mtorr}")
    if not math.isfinite(distance_m):
        raise ValueError(f"distance_m must be finite, got {distance_m}")
    if not math.isfinite(gas_temp_k):
        raise ValueError(f"gas_temp_k must be finite, got {gas_temp_k}")
        
    if distance_m < 0.0:
        raise ValueError(f"distance_m cannot be negative, got {distance_m}")
    if broadening_coefficient < 0.0:
        raise ValueError(
            f"broadening_coefficient cannot be negative, got {broadening_coefficient}"
        )

    if distance_m == 0.0:
        return 1.0

    mfp = calculate_mean_free_path(pressure_mtorr, gas_temp_k=gas_temp_k)
    n_coll = distance_m / mfp

    broadening = math.sqrt(1.0 + broadening_coefficient * n_coll)
    return float(broadening)


def calculate_transport_summary(pressure_mtorr: float, distance_mm: float, gas_temp_k: float = 300.0) -> TransportSummary:
    d_m = distance_mm * 1e-3
    mfp_m = calculate_mean_free_path(pressure_mtorr, gas_temp_k)
    kn = calculate_knudsen_number(mfp_m, d_m)
    return TransportSummary(
        mean_free_path_mm=mfp_m * 1e3,
        knudsen_number=kn,
        transmission_probability=calculate_transmission_probability(d_m, mfp_m),
        broadening_factor=calculate_scattering_broadening(pressure_mtorr, d_m, gas_temp_k),
        n_collisions=d_m / mfp_m,
        regime='ballistic' if kn > 1.0 else ('transition' if kn >= 0.1 else 'continuum'),
    )
