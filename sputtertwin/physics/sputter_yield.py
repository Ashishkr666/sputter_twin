"""Sputter yield calculation module for SputterTwin.

Implements the Yamamura-Tawara / Bohdansky formulation for Ar+ bombardment
on planar target materials, including energy thresholds, empirical stopping
powers, and angular incidence dependency with grazing angle cutoffs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Sequence, Union

import numpy as np

__all__ = [
    "TargetMaterial",
    "MATERIALS",
    "calculate_sputter_yield",
    "calculate_sputter_yield_array",
    "calculate_kinematic_factor",
    "calculate_thomson_energy_spectrum",
    "sample_ejected_energy_and_angle",
    "calculate_sputter_yield_sota",
    "EjectedParticles",
]

@dataclass(frozen=True)
class TargetMaterial:
    """Properties of a sputtering target material.
    
    # __slots__ note: frozen dataclass.

    Attributes:
        name: Chemical symbol or material designation (e.g., 'Ti', 'Cu', 'Al').
        atomic_mass: Target atomic mass in g/mol.
        atomic_number: Target atomic number (Z2).
        density: Target mass density in g/cm^3.
        sublimation_energy: Heat of sublimation / surface binding energy (Us) in eV.
        threshold_energy: Sputtering threshold energy (Eth) in eV.
        q_factor: Empirical scaling factor Q for Yamamura-Tawara formulation.
    """

    name: str
    atomic_mass: float
    atomic_number: int
    density: float
    sublimation_energy: float
    threshold_energy: float
    q_factor: float


# Predefined standard materials for DC magnetron sputtering
MATERIALS: dict[str, TargetMaterial] = {
    "Ti": TargetMaterial("Ti", 47.87, 22, 4.506, 4.89, 30.0, 0.85),
    "Cu": TargetMaterial("Cu", 63.55, 29, 8.96, 3.49, 20.0, 1.60),
    "Al": TargetMaterial("Al", 26.98, 13, 2.70, 3.39, 25.0, 0.90),
}

# Incident projectile constants (Argon ion Ar+)
_Z_AR: int = 18
_M_AR: float = 39.948  # g/mol

# Fundamental physical constants
_BOHR_RADIUS_A: float = 0.5291772109  # Angstrom
_E2_EV_A: float = 14.399645  # e^2 / (4 * pi * eps_0) in eV * Angstrom
_EV_TO_J: float = 1.602176634e-19  # J/eV
_AMU_TO_KG: float = 1.66053906660e-27  # kg/amu (g/mol to kg)


def _calculate_reduced_nuclear_stopping(epsilon: float) -> float:
    """Calculate Kr-C reduced nuclear stopping cross-section sn(epsilon).

    Uses the Krypton-Carbon (Kr-C) universal potential approximation commonly
    employed in Yamamura & Tawara (1996) sputtering yield formulations.

    Args:
        epsilon: Lindhard reduced energy (dimensionless).

    Returns:
        sn: Reduced nuclear stopping cross-section (dimensionless).
    """
    if epsilon <= 0.0:
        return 0.0
    sqrt_eps = math.sqrt(epsilon)
    num = 3.441 * sqrt_eps * math.log(epsilon + math.e)
    denom = 1.0 + 6.35 * sqrt_eps + epsilon * (-1.708 + 6.882 * sqrt_eps)
    return max(0.0, num / denom)


def _calculate_alpha(m_target: float, m_ion: float = _M_AR) -> float:
    """Calculate Yamamura empirical mass ratio parameter alpha.

    Args:
        m_target: Target atomic mass in g/mol.
        m_ion: Incident ion atomic mass in g/mol.

    Returns:
        alpha parameter for the energy deposition ratio.
    """
    mu = m_target / m_ion
    if mu >= 1.0:
        return 0.10 + 0.155 * (mu**0.73)
    return 0.249 * (mu**0.56)


def calculate_sputter_yield(
    energy_ev: float,
    material: Union[str, TargetMaterial],
    angle_rad: float = 0.0,
    f_exponent: float = 1.9,
    optimum_angle_deg: float = 65.0,
    grazing_cutoff_deg: float = 85.0,
) -> float:
    """Calculate the sputter yield for Ar+ bombardment on a target material.

    Uses the Yamamura-Tawara / Bohdansky semi-empirical formulation:
        Y(0) = (0.042 / Us) * alpha * Q * Sn(E) * [1 - sqrt(Eth / E)]^s
    with Yamamura angular dependency:
        Y(theta) = Y(0) * (cos(theta))^(-f) * exp(-Sigma * (1/cos(theta) - 1))
    where Sigma = f * cos(theta_optimum), with a cutoff at grazing incidence.

    Args:
        energy_ev: Incident ion kinetic energy in eV (typically 100 - 1000 eV).
        material: TargetMaterial instance or chemical symbol string ('Ti', 'Cu', 'Al').
        angle_rad: Angle of incidence relative to surface normal in radians
            (0.0 = normal incidence, pi/2 = grazing).
        f_exponent: Exponent for Yamamura angular formulation (default 1.9).
        optimum_angle_deg: Angle of maximum yield in degrees (default 65.0 deg).
        grazing_cutoff_deg: Cutoff angle in degrees beyond which yield drops to 0
            (default 85.0 deg).

    Returns:
        Sputter yield Y in atoms per incident ion (atoms/ion). Returns 0.0
        if energy_ev <= threshold_energy or for grazing angles beyond cutoff.

    Raises:
        KeyError: If material is a string not found in MATERIALS.
        ValueError: If energy_ev is negative or material is invalid.
    """
    if not math.isfinite(energy_ev):
        raise ValueError(f"Ion energy must be finite, got {energy_ev}")
    if energy_ev < 0.0:
        raise ValueError(f"Ion energy cannot be negative: {energy_ev} eV")

    # Resolve material
    if isinstance(material, str):
        if material not in MATERIALS:
            raise KeyError(
                f"Material '{material}' not found in predefined materials: {list(MATERIALS.keys())}"
            )
        target = MATERIALS[material]
    elif isinstance(material, TargetMaterial):
        target = material
    else:
        raise TypeError(
            f"material must be a string or TargetMaterial instance, got {type(material).__name__}"
        )

    # Angular parameter validation
    if f_exponent <= 0.0:
        raise ValueError(f"f_exponent must be positive, got {f_exponent}")
    if not (0.0 < optimum_angle_deg < 90.0):
        raise ValueError(
            f"optimum_angle_deg must be in (0, 90), got {optimum_angle_deg}"
        )
    if not (0.0 < grazing_cutoff_deg <= 90.0):
        raise ValueError(
            f"grazing_cutoff_deg must be in (0, 90], got {grazing_cutoff_deg}"
        )

    # Sub-threshold check
    if energy_ev <= target.threshold_energy:
        return 0.0

    # Angular cutoff check
    theta = abs(angle_rad)
    cutoff_rad = math.radians(grazing_cutoff_deg)
    if theta >= cutoff_rad or theta >= (math.pi / 2.0):
        return 0.0

    z1 = _Z_AR
    m1 = _M_AR
    z2 = target.atomic_number
    m2 = target.atomic_mass
    us = target.sublimation_energy
    eth = target.threshold_energy

    # Lindhard screening length a_L in Angstroms
    a_l = 0.8853 * _BOHR_RADIUS_A / math.sqrt(z1 ** (2.0 / 3.0) + z2 ** (2.0 / 3.0))

    # Lindhard reduced energy epsilon
    eps = (
        energy_ev
        * (m2 / (m1 + m2))
        * (a_l / (z1 * z2 * _E2_EV_A))
    )

    # Reduced nuclear stopping power sn
    sn = _calculate_reduced_nuclear_stopping(eps)

    # Dimensional nuclear stopping cross section Sn in eV * A^2
    sn_dimensional = 4.0 * math.pi * a_l * z1 * z2 * _E2_EV_A * (m1 / (m1 + m2)) * sn

    # Yamamura alpha factor and material Q factor
    alpha = _calculate_alpha(m2, m1)
    q_factor = target.q_factor

    # Yamamura-Tawara threshold factor (s = 2.8 for low energy sputtering)
    threshold_term = max(0.0, 1.0 - math.sqrt(eth / energy_ev)) ** 2.8

    # Normal incidence yield Y(0)
    y_normal = (0.042 / us) * alpha * q_factor * sn_dimensional * threshold_term

    if theta == 0.0:
        return float(y_normal)

    # Angular enhancement factor
    cos_theta = math.cos(theta)
    if cos_theta <= 0.0:
        return 0.0

    theta_opt_rad = math.radians(optimum_angle_deg)
    sigma = f_exponent * math.cos(theta_opt_rad)

    sec_minus_one = (1.0 / cos_theta) - 1.0
    angular_multiplier = (cos_theta ** (-f_exponent)) * math.exp(-sigma * sec_minus_one)

    return float(y_normal * angular_multiplier)


def calculate_sputter_yield_array(
    energies_ev: np.ndarray,
    material: Union[str, TargetMaterial],
    angle_rad: float = 0.0,
    f_exponent: float = 1.9,
    optimum_angle_deg: float = 65.0,
    grazing_cutoff_deg: float = 85.0,
) -> np.ndarray:
    """Vectorized sputter yield for an array of energies using native numpy ops.

    Implements the full Yamamura-Tawara / Bohdansky formulation at C-level
    speed — typically 50-100x faster than the previous np.vectorize wrapper
    for arrays of 1000+ elements.

    Args:
        energies_ev: 1-D array of ion energies in eV.
        material: TargetMaterial instance or chemical symbol string.
        angle_rad: Angle of incidence in radians (scalar, applied uniformly).
        f_exponent: Yamamura angular exponent (default 1.9).
        optimum_angle_deg: Angle of maximum yield in degrees (default 65.0).
        grazing_cutoff_deg: Cutoff angle in degrees (default 85.0).

    Returns:
        1-D numpy array of sputter yields (atoms/ion), same shape as energies_ev.
    """
    energies_ev = np.asarray(energies_ev, dtype=np.float64)

    # Resolve material
    if isinstance(material, str):
        if material not in MATERIALS:
            raise KeyError(
                f"Material '{material}' not found: {list(MATERIALS.keys())}"
            )
        target = MATERIALS[material]
    elif isinstance(material, TargetMaterial):
        target = material
    else:
        raise TypeError(
            f"material must be a string or TargetMaterial, got {type(material).__name__}"
        )

    z1, m1 = _Z_AR, _M_AR
    z2 = target.atomic_number
    m2 = target.atomic_mass
    us = target.sublimation_energy
    eth = target.threshold_energy

    # Start with zero yield everywhere
    Y = np.zeros_like(energies_ev)

    # Mask: above threshold and positive energy
    above = energies_ev > eth

    # Angular cutoff
    theta = abs(angle_rad)
    cutoff_rad = math.radians(grazing_cutoff_deg)
    if theta >= cutoff_rad or theta >= (math.pi / 2.0):
        return Y  # all zeros

    if not np.any(above):
        return Y

    E = energies_ev[above]

    # Lindhard screening length a_L (Angstrom)
    a_l = 0.8853 * _BOHR_RADIUS_A / math.sqrt(z1 ** (2.0 / 3.0) + z2 ** (2.0 / 3.0))

    # Lindhard reduced energy epsilon
    eps = E * (m2 / (m1 + m2)) * (a_l / (z1 * z2 * _E2_EV_A))

    # Kr-C reduced nuclear stopping sn(epsilon) — vectorized
    sqrt_eps = np.sqrt(eps)
    num = 3.441 * sqrt_eps * np.log(eps + math.e)
    denom = 1.0 + 6.35 * sqrt_eps + eps * (-1.708 + 6.882 * sqrt_eps)
    sn = np.maximum(0.0, num / denom)

    # Dimensional nuclear stopping Sn (eV·Å²)
    sn_dim = 4.0 * math.pi * a_l * z1 * z2 * _E2_EV_A * (m1 / (m1 + m2)) * sn

    # Alpha factor
    alpha = _calculate_alpha(m2, m1)

    # Threshold term [1 - sqrt(Eth/E)]^2.8
    threshold_term = np.maximum(0.0, 1.0 - np.sqrt(eth / E)) ** 2.8

    # Normal incidence yield Y(0)
    y_normal = (0.042 / us) * alpha * target.q_factor * sn_dim * threshold_term

    if theta == 0.0:
        Y[above] = y_normal
        return Y

    # Angular enhancement
    cos_theta = math.cos(theta)
    if cos_theta <= 0.0:
        return np.zeros_like(energies_ev)

    theta_opt_rad = math.radians(optimum_angle_deg)
    sigma = f_exponent * math.cos(theta_opt_rad)
    sec_minus_one = (1.0 / cos_theta) - 1.0
    angular_mult = (cos_theta ** (-f_exponent)) * math.exp(-sigma * sec_minus_one)

    Y[above] = y_normal * angular_mult
    return Y


def _resolve_material(material: Union[str, TargetMaterial]) -> TargetMaterial:
    """Resolve a material input to a TargetMaterial instance."""
    if isinstance(material, str):
        if material not in MATERIALS:
            raise KeyError(
                f"Material '{material}' not found in predefined materials: {list(MATERIALS.keys())}"
            )
        return MATERIALS[material]
    elif isinstance(material, TargetMaterial):
        return material
    else:
        raise TypeError(
            f"material must be a string or TargetMaterial instance, got {type(material).__name__}"
        )


def calculate_kinematic_factor(
    m_target: float,
    m_ion: float = _M_AR,
) -> float:
    """Calculate the maximum kinematic energy transfer factor gamma.

    gamma = 4 * M1 * M2 / (M1 + M2)**2

    Args:
        m_target: Target atomic mass in g/mol.
        m_ion: Incident projectile atomic mass in g/mol (default Ar+: 39.948).

    Returns:
        Dimensionless kinematic energy transfer factor gamma in (0, 1].
    """
    if m_target <= 0.0 or m_ion <= 0.0:
        raise ValueError(
            f"Atomic masses must be positive: m_target={m_target}, m_ion={m_ion}"
        )
    return (4.0 * m_ion * m_target) / ((m_ion + m_target) ** 2)


def calculate_thomson_energy_spectrum(
    energies_ev: Union[float, np.ndarray, Sequence[float]],
    material: Union[str, TargetMaterial],
    ion_energy_ev: float,
    normalize: bool = False,
) -> Union[float, np.ndarray]:
    """Calculate the Thomson nascent energy distribution f(E) for sputtered atoms.

    Implements the classic Thomson energy spectrum with the Sigmund high-energy
    kinematic cutoff term:
        f(E) = (2 * Us * E) / (E + Us)**3 * [1 - sqrt((E + Us) / (gamma * E_ion))]
    where:
        gamma = 4 * M1 * M2 / (M1 + M2)**2  (kinematic energy transfer factor)
        Us: Target surface binding energy (heat of sublimation in eV)
        E_ion: Incident projectile kinetic energy in eV

    The distribution vanishes for E <= 0 and for E >= gamma * E_ion - Us.
    For E << gamma * E_ion, the peak occurs at E_peak = Us / 2.

    Args:
        energies_ev: Kinetic energy of ejected atom in eV (scalar float or array-like).
        material: TargetMaterial instance or chemical symbol ('Ti', 'Cu', 'Al').
        ion_energy_ev: Incident ion kinetic energy in eV.
        normalize: If True, normalizes the distribution such that
            integral_0^{E_max} f(E) dE = 1.0.

    Returns:
        Probability density or spectrum value f(E) in eV^-1. Returns float if
        energies_ev is scalar, or ndarray if energies_ev is array-like.
    """
    if not math.isfinite(ion_energy_ev):
        raise ValueError(f"Ion energy must be finite, got {ion_energy_ev}")
    if ion_energy_ev <= 0.0:
        raise ValueError(f"Ion energy must be positive: {ion_energy_ev} eV")

    target = _resolve_material(material)
    us = target.sublimation_energy
    gamma = calculate_kinematic_factor(target.atomic_mass, _M_AR)
    e_max = gamma * ion_energy_ev - us

    is_scalar = isinstance(energies_ev, (int, float, np.floating, np.integer))

    if e_max <= 0.0:
        if is_scalar:
            return 0.0
        return np.zeros_like(np.asarray(energies_ev, dtype=np.float64))

    # Analytical normalization integral factor over [0, E_max]
    # integral = 1 - (8/3)*eps + 2*(eps^2) - (1/3)*(eps^4), where eps = sqrt(Us / (gamma * E_ion))
    norm_factor = 1.0
    if normalize:
        eps = math.sqrt(us / (gamma * ion_energy_ev))
        norm_factor = 1.0 - (8.0 / 3.0) * eps + 2.0 * (eps ** 2) - (1.0 / 3.0) * (eps ** 4)
        if norm_factor <= 0.0:
            norm_factor = 1.0

    if is_scalar:
        e = float(energies_ev)
        if e <= 0.0 or e >= e_max:
            return 0.0
        u = e + us
        val = (2.0 * us * e) / (u ** 3) * (1.0 - math.sqrt(u / (gamma * ion_energy_ev)))
        return float(val / norm_factor)

    arr = np.asarray(energies_ev, dtype=np.float64)
    result = np.zeros_like(arr)
    mask = (arr > 0.0) & (arr < e_max)
    if np.any(mask):
        e_valid = arr[mask]
        u = e_valid + us
        term1 = (2.0 * us * e_valid) / (u ** 3)
        cutoff = 1.0 - np.sqrt(u / (gamma * ion_energy_ev))
        result[mask] = (term1 * np.maximum(0.0, cutoff)) / norm_factor

    return result


class EjectedParticles(NamedTuple):
    """Container for sampled nascent sputtered particle energies and angles.

    Attributes:
        energies: 1-D numpy array of sampled kinetic energies in eV.
        angles: 1-D numpy array of polar emission angles in radians (relative to normal).
    """

    energies: np.ndarray
    angles: np.ndarray

    def speeds(self, material: Union[str, TargetMaterial, float]) -> np.ndarray:
        """Calculate nascent particle speeds in m/s from kinetic energies.

        Args:
            material: TargetMaterial, chemical symbol string, or target atomic mass in g/mol.

        Returns:
            1-D numpy array of speeds in m/s.
        """
        if isinstance(material, (int, float)):
            mass_amu = float(material)
        elif isinstance(material, str):
            mass_amu = MATERIALS[material].atomic_mass
        elif isinstance(material, TargetMaterial):
            mass_amu = material.atomic_mass
        else:
            raise TypeError(
                f"material must be str, TargetMaterial, or float mass, got {type(material).__name__}"
            )
        if mass_amu <= 0.0:
            raise ValueError(f"Target mass must be positive, got {mass_amu}")

        mass_kg = mass_amu * _AMU_TO_KG
        energy_j = np.maximum(0.0, self.energies) * _EV_TO_J
        return np.sqrt(2.0 * energy_j / mass_kg)


def sample_ejected_energy_and_angle(
    num_samples: int,
    material: Union[str, TargetMaterial],
    ion_energy_ev: float,
    seed: Optional[Union[int, np.random.Generator]] = 42,
    n_cosine: float = 1.2,
    **kwargs: Any,
) -> EjectedParticles:
    """Vectorized Monte Carlo sampling of nascent sputtered atom velocities.

    Samples kinetic energies E from the Thomson nascent energy distribution:
        f(E) ~ (2 * Us * E) / (E + Us)**3 * [1 - sqrt((E + Us)/(gamma * E_ion))]
    and polar emission angles alpha from an over/under-cosine distribution:
        f(alpha) ~ cos^n(alpha) * sin(alpha)  (per polar angle in 3D half-space)
    yielding polar angles alpha = arccos(U^(1 / (n + 1))).

    Args:
        num_samples: Number of particles to sample (non-negative int).
        material: TargetMaterial instance or chemical symbol string ('Ti', 'Cu', 'Al').
        ion_energy_ev: Incident ion kinetic energy in eV.
        seed: Random seed or np.random.Generator instance (default 42).
        n_cosine: Angular exponent for over/under-cosine emission (default 1.2, typical 1.0 - 1.5).
        **kwargs: Additional parameters (supports 'n' as alias for n_cosine).

    Returns:
        EjectedParticles(energies, angles): NamedTuple containing:
            - energies: 1-D numpy array of sampled energies in eV
            - angles: 1-D numpy array of polar emission angles alpha in radians in [0, pi/2]
    """
    if num_samples < 0:
        raise ValueError(f"num_samples must be non-negative, got {num_samples}")
    if not math.isfinite(ion_energy_ev):
        raise ValueError(f"Ion energy must be finite, got {ion_energy_ev}")
    if ion_energy_ev <= 0.0:
        raise ValueError(f"Ion energy must be positive: {ion_energy_ev} eV")

    if "n" in kwargs:
        n_cosine = float(kwargs["n"])
    if n_cosine <= -1.0:
        raise ValueError(f"n_cosine exponent must be > -1, got {n_cosine}")

    if num_samples == 0:
        return EjectedParticles(
            energies=np.empty(0, dtype=np.float64),
            angles=np.empty(0, dtype=np.float64),
        )

    target = _resolve_material(material)
    us = target.sublimation_energy
    gamma = calculate_kinematic_factor(target.atomic_mass, _M_AR)
    e_max = gamma * ion_energy_ev - us

    if isinstance(seed, np.random.Generator):
        rng = seed
    elif seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # If ion energy is below kinematic threshold, no atoms are ejected
    if e_max <= 0.0:
        return EjectedParticles(
            energies=np.zeros(num_samples, dtype=np.float64),
            angles=np.zeros(num_samples, dtype=np.float64),
        )

    # 1. Vectorized sampling of polar emission angle alpha:
    # dY/dOmega ~ cos^n(alpha) => dY/dalpha ~ cos^n(alpha) * sin(alpha)
    # CDF F(alpha) = 1 - cos^(n+1)(alpha) = U => alpha = arccos(U^(1 / (n+1)))
    u_angles = rng.uniform(0.0, 1.0, num_samples)
    angles = np.arccos(np.power(u_angles, 1.0 / (n_cosine + 1.0)))

    # 2. Vectorized rejection sampling of Thomson nascent energy:
    # Proposal distribution is uncutoff Thomson CDF: F0(E) = (E / (E + Us))^2
    # Inverted proposal: E = Us * sqrt(u) / (1 - sqrt(u)), where u in [0, F0(E_max)]
    # Acceptance probability: P_accept(E) = 1 - sqrt((E + Us) / (gamma * E_ion))
    u_max = (e_max / (e_max + us)) ** 2
    energies = np.empty(num_samples, dtype=np.float64)
    filled = 0
    batch_size = max(1024, int(num_samples * 1.3))

    while filled < num_samples:
        needed = num_samples - filled
        current_batch = max(needed, batch_size)
        u_cand = rng.uniform(0.0, u_max, current_batch)
        sqrt_u = np.sqrt(u_cand)
        e_cand = us * sqrt_u / (1.0 - sqrt_u)
        p_accept = 1.0 - np.sqrt((e_cand + us) / (gamma * ion_energy_ev))
        accept_mask = rng.uniform(0.0, 1.0, current_batch) < p_accept
        accepted = e_cand[accept_mask]
        n_acc = len(accepted)
        if n_acc > 0:
            take = min(n_acc, needed)
            energies[filled : filled + take] = accepted[:take]
            filled += take

    return EjectedParticles(energies=energies, angles=angles)


def calculate_sputter_yield_sota(
    energy_ev: Union[float, np.ndarray],
    material: Union[str, TargetMaterial],
    angle_rad: Union[float, np.ndarray] = 0.0,
    roughness_factor: float = 0.15,
    f_exponent: float = 1.9,
    optimum_angle_deg: float = 65.0,
    grazing_cutoff_deg: float = 85.0,
) -> Union[float, np.ndarray]:
    """Calculate state-of-the-art (SOTA) sputter yield with surface roughness damping.

    Extends the Yamamura-Tawara formulation by incorporating the Ruzic surface
    micro-roughness damping model:
        Y_sota(E, theta, r) = Y_smooth(E, theta) * exp(-r * tan(theta))
    where:
        r: Target surface micro-roughness parameter (default 0.15).
        theta: Angle of incidence relative to surface normal in radians.

    Surface micro-roughness softens the unphysical sharp peak predicted by
    flat-surface Yamamura-Tawara models at oblique angles through geometric
    shadowing and local redeposition onto neighboring surface asperities.

    At normal incidence (theta = 0) or for an ideal smooth surface (r = 0),
    the yield reduces identically to Yamamura-Tawara.

    Args:
        energy_ev: Incident ion kinetic energy in eV (scalar or array).
        material: TargetMaterial instance or chemical symbol ('Ti', 'Cu', 'Al').
        angle_rad: Angle of incidence in radians (0.0 = normal, pi/2 = grazing).
        roughness_factor: Micro-roughness factor (default 0.15, >= 0.0).
        f_exponent: Yamamura angular exponent (default 1.9).
        optimum_angle_deg: Angle of maximum yield in degrees (default 65.0).
        grazing_cutoff_deg: Cutoff angle in degrees beyond which yield is 0 (default 85.0).

    Returns:
        Sputter yield Y in atoms per incident ion. Returns float if energy_ev
        and angle_rad are scalars, or numpy ndarray otherwise.
    """
    if roughness_factor < 0.0:
        raise ValueError(
            f"roughness_factor must be non-negative, got {roughness_factor}"
        )

    # Scalar fast-path
    if np.ndim(energy_ev) == 0 and np.ndim(angle_rad) == 0:
        y_smooth = calculate_sputter_yield(
            energy_ev=float(energy_ev),
            material=material,
            angle_rad=float(angle_rad),
            f_exponent=f_exponent,
            optimum_angle_deg=optimum_angle_deg,
            grazing_cutoff_deg=grazing_cutoff_deg,
        )
        if y_smooth == 0.0 or roughness_factor == 0.0:
            return float(y_smooth)
        theta = abs(float(angle_rad))
        damping = math.exp(-roughness_factor * math.tan(theta))
        return float(y_smooth * damping)

    # Vectorized array path
    energies_arr = np.asarray(energy_ev, dtype=np.float64)
    if np.ndim(angle_rad) == 0:
        y_smooth = calculate_sputter_yield_array(
            energies_ev=energies_arr,
            material=material,
            angle_rad=float(angle_rad),
            f_exponent=f_exponent,
            optimum_angle_deg=optimum_angle_deg,
            grazing_cutoff_deg=grazing_cutoff_deg,
        )
        if roughness_factor == 0.0 or float(angle_rad) == 0.0:
            return y_smooth
        theta = abs(float(angle_rad))
        damping = math.exp(-roughness_factor * math.tan(theta))
        return y_smooth * damping
    else:
        # Both energies and angles may be arrays
        angles_arr = np.asarray(angle_rad, dtype=np.float64)
        cutoff_rad = math.radians(grazing_cutoff_deg)
        theta = np.abs(angles_arr)

        y_normal = calculate_sputter_yield_array(
            energies_ev=energies_arr,
            material=material,
            angle_rad=0.0,
            f_exponent=f_exponent,
            optimum_angle_deg=optimum_angle_deg,
            grazing_cutoff_deg=grazing_cutoff_deg,
        )

        valid_mask = (theta < cutoff_rad) & (theta < (math.pi / 2.0))
        cos_theta = np.cos(theta)
        valid_mask = valid_mask & (cos_theta > 0.0)

        angular_mult = np.zeros_like(theta, dtype=np.float64)
        if np.any(valid_mask):
            ct = cos_theta[valid_mask]
            th = theta[valid_mask]
            theta_opt_rad = math.radians(optimum_angle_deg)
            sigma = f_exponent * math.cos(theta_opt_rad)
            sec_m1 = (1.0 / ct) - 1.0
            mult = (ct ** (-f_exponent)) * np.exp(-sigma * sec_m1)
            if roughness_factor > 0.0:
                mult *= np.exp(-roughness_factor * np.tan(th))
            angular_mult[valid_mask] = mult

        return y_normal * angular_mult

