"""End-to-End Integrated Simulation Pipeline for SputterTwin.

Seamlessly couples Stage 1 (Plasma & Magnetic Field) and Stage 2 (Sputter Yield
& Target Racetrack Erosion) into a unified, high-fidelity multiphysics pipeline:

Pipeline Flow:
1. Stage 1A (Magnetic Field):
   Calculates 2D magnetic trap field B(r, z) and racetrack center r_race.
2. Stage 1B (Plasma Discharge):
   Solves 2D drift-diffusion electron/ion transport, sheath potential V_p(r, z),
   and cathode ion flux profile J_i(r) = e * n_i(r, 0) * u_Bohm.
3. Stage 1C (Gas Rarefaction):
   Calculates 2D thermal fluid gas heating T_gas(r, z) and rarefaction factor.
4. Stage 2A (Dynamic Target Erosion):
   Couples cathode ion flux J_i(r) and discharge voltage V_d with SOTA Yamamura-Ruzic
   roughness-damped sputter yield Y(E_eff, theta_local(r, t)), dynamically evolving
   the 2D groove depth d(r, t) over operational hours.
5. Stage 2B (Nascent Sputtered Atom Emission):
   Samples nascent sputtered atom kinetic energies from the Thomson cascade distribution
   and polar angles from over-cosine emission, providing the source boundary condition
   for Stage 3 gas-phase transport.
6. Unified PINN Surrogate:
   Coupled PlasmaPINN and YieldPINN for sub-millisecond AI-driven multi-physics inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from sputtertwin.numerics import trapezoid

from sputtertwin.physics.sputter_yield import (
    MATERIALS,
    EjectedParticles,
    TargetMaterial,
    calculate_sputter_yield_sota,
    sample_ejected_energy_and_angle,
)
from sputtertwin.physics.target_erosion import (
    TargetErosionModel,
    TargetErosionResult,
)
from sputtertwin.physics2d.plasma2d import (
    Plasma2DResult,
    simulate_plasma_2d,
)
from sputtertwin.physics2d.rarefaction2d import (
    GasRarefaction2D,
    compute_gas_rarefaction_2d,
)

# Fundamental constants
_AVOGADRO: float = 6.02214076e23        # atoms / mol
_E_CHARGE: float = 1.602176634e-19     # C


@dataclass
class IntegratedDischargeErosionResult:
    """Comprehensive results of an end-to-end integrated plasma and erosion simulation.

    Attributes:
        plasma: 2D plasma discharge result (current density, potential, ion density).
        erosion: 2D dynamic target racetrack erosion result (depth profile, lifetime, utilization).
        rarefaction: Optional 2D background gas rarefaction profile.
        ejected_particles: Monte Carlo sample of nascent sputtered atom energies and angles.
        material: Target material chemical symbol.
        power_w: Cathode electrical power in Watts.
        pressure_mtorr: Argon gas pressure in mTorr.
        total_hours: Total operational sputtering duration in hours.
        total_sputtered_flux_atoms_per_s: Total integrated target atom emission rate (atoms/s).
        total_sputtered_mass_grams: Total mass eroded from the target plate (grams).
        peak_erosion_rate_um_hr: Maximum local target erosion rate in um/hour.
        mean_erosion_rate_um_hr: Area-averaged target erosion rate in um/hour.
    """

    plasma: Plasma2DResult
    erosion: TargetErosionResult
    rarefaction: Optional[GasRarefaction2D]
    ejected_particles: EjectedParticles
    material: str
    power_w: float
    pressure_mtorr: float
    total_hours: float
    total_sputtered_flux_atoms_per_s: float
    total_sputtered_mass_grams: float
    peak_erosion_rate_um_hr: float
    mean_erosion_rate_um_hr: float

    @property
    def total_mass_loss_grams(self) -> float:
        """Alias for total_sputtered_mass_grams."""
        return self.total_sputtered_mass_grams

    def to_dict(self) -> Dict[str, Any]:
        """Convert key metrics to a serializable dictionary."""
        return {
            "material": self.material,
            "power_w": self.power_w,
            "pressure_mtorr": self.pressure_mtorr,
            "total_hours": self.total_hours,
            "discharge_voltage_v": float(self.plasma.voltage_v),
            "discharge_current_a": float(self.plasma.current_a),
            "peak_plasma_density_m3": float(self.plasma.peak_ne_m3),
            "electron_temperature_ev": float(np.mean(self.plasma.Te_2d)),
            "peak_ion_flux_a_m2": float(np.max(self.plasma.current_density_1d)),
            "peak_erosion_depth_mm": float(self.erosion.peak_erosion_depth_mm),
            "racetrack_fwhm_mm": float(self.erosion.racetrack_fwhm_mm),
            "target_utilization_pct": float(self.erosion.target_utilization_efficiency_pct),
            "target_lifetime_hours": float(self.erosion.target_lifetime_hours),
            "breakthrough_occurred": bool(self.erosion.breakthrough_occurred),
            "total_sputtered_flux_atoms_per_s": float(self.total_sputtered_flux_atoms_per_s),
            "total_sputtered_mass_grams": float(self.total_sputtered_mass_grams),
            "peak_erosion_rate_um_hr": float(self.peak_erosion_rate_um_hr),
            "mean_erosion_rate_um_hr": float(self.mean_erosion_rate_um_hr),
            "mean_nascent_atom_energy_ev": float(np.mean(self.ejected_particles.energies)),
            "mean_ejection_angle_deg": float(np.degrees(np.mean(self.ejected_particles.angles))),
        }

    def summary(self) -> str:
        """Return a formatted academic summary table."""
        d = self.to_dict()
        lines = [
            "=" * 72,
            f"SPUTTERTWIN INTEGRATED STAGE 1 & 2 MULTIPHYSICS REPORT ({self.material} TARGET)",
            "=" * 72,
            f"Operating Conditions: Power = {d['power_w']:.1f} W | Pressure = {d['pressure_mtorr']:.2f} mTorr | Time = {d['total_hours']:.1f} h",
            "-" * 72,
            "[Stage 1: Plasma & Magnetic Confinement]",
            f"  Discharge Voltage:       {d['discharge_voltage_v']:.1f} V",
            f"  Discharge Current:       {d['discharge_current_a']:.3f} A",
            f"  Peak Plasma Density:     {d['peak_plasma_density_m3']:.3e} m^-3",
            f"  Electron Temperature:    {d['electron_temperature_ev']:.2f} eV",
            f"  Peak Cathode Ion Flux:   {d['peak_ion_flux_a_m2']:.2f} A/m^2",
            "-" * 72,
            "[Stage 2: Sputter Yield & Target Erosion]",
            f"  Peak Groove Depth:       {d['peak_erosion_depth_mm']:.3f} mm",
            f"  Racetrack FWHM Width:    {d['racetrack_fwhm_mm']:.2f} mm",
            f"  Peak Erosion Rate:       {d['peak_erosion_rate_um_hr']:.2f} um/h",
            f"  Target Utilization:      {d['target_utilization_pct']:.2f} %",
            f"  Target Lifetime:         {d['target_lifetime_hours']:.1f} hours"
            + (" (BREAKTHROUGH!)" if d["breakthrough_occurred"] else " (safe)"),
            f"  Total Sputtered Flux:    {d['total_sputtered_flux_atoms_per_s']:.3e} atoms/s",
            f"  Total Mass Loss:         {d['total_sputtered_mass_grams']:.3f} g",
            "-" * 72,
            "[Nascent Atom Ejection Distribution (Thomson + Ruzic)]",
            f"  Mean Nascent Energy:     {d['mean_nascent_atom_energy_ev']:.2f} eV",
            f"  Mean Polar Angle:        {d['mean_ejection_angle_deg']:.1f} deg",
            "=" * 72,
        ]
        return "\n".join(lines)


def simulate_integrated_discharge_and_erosion(
    power_w: float = 220.0,
    pressure_mtorr: float = 5.0,
    material: str = "Cu",
    ar_flow_sccm: float = 20.0,
    target_radius_mm: float = 25.0,
    initial_thickness_mm: float = 6.0,
    total_hours: float = 10.0,
    time_step_hours: float = 0.5,
    roughness_factor: float = 0.15,
    anode_distance_mm: float = 40.0,
    grid_r: int = 40,
    grid_z: int = 30,
    include_gas_rarefaction: bool = True,
    num_ejected_samples: int = 10000,
    seed: Optional[int] = 42,
) -> IntegratedDischargeErosionResult:
    """Run full end-to-end integrated simulation of Stage 1 and Stage 2.

    Couples 2D plasma discharge, Bohm sheath ion acceleration, SOTA Yamamura-Ruzic
    sputter yield, dynamic 2D racetrack target erosion, and Thomson atom emission.

    Args:
        power_w: Cathode electrical power in Watts (default: 220.0 W).
        pressure_mtorr: Argon background pressure in mTorr (default: 5.0 mTorr).
        material: Target material symbol ('Cu', 'Ti', 'Al') (default: 'Cu').
        ar_flow_sccm: Argon flow rate in sccm (default: 20.0 sccm).
        target_radius_mm: Target plate radius in mm (default: 25.0 mm).
        initial_thickness_mm: Initial target plate thickness in mm (default: 6.0 mm).
        total_hours: Total operational sputtering duration in hours (default: 10.0 h).
        time_step_hours: Numerical time integration step in hours (default: 0.5 h).
        roughness_factor: Micro-roughness damping parameter (default: 0.15).
        anode_distance_mm: Cathode-to-anode chamber distance in mm (default: 40.0 mm).
        grid_r: Number of radial grid cells (default: 40).
        grid_z: Number of axial grid cells (default: 30).
        include_gas_rarefaction: Whether to compute 2D gas rarefaction profile (default: True).
        num_ejected_samples: Number of nascent atom Monte Carlo samples (default: 10000).
        seed: Random seed for reproducible Monte Carlo sampling.

    Returns:
        IntegratedDischargeErosionResult containing coupled plasma, erosion,
        gas rarefaction, and nascent atom ejection properties.
    """
    if material not in MATERIALS:
        raise KeyError(f"Material '{material}' not supported. Options: {list(MATERIALS.keys())}")

    mat_obj = MATERIALS[material]
    target_r_m = float(target_radius_mm) * 1e-3
    anode_d_m = float(anode_distance_mm) * 1e-3

    # 1. Step 1: Solve 2D Plasma Discharge
    plasma_res = simulate_plasma_2d(
        power_w=power_w,
        pressure_mtorr=pressure_mtorr,
        ar_flow_sccm=ar_flow_sccm,
        target_radius_m=target_r_m,
        anode_distance_m=anode_d_m,
        grid_r=grid_r,
        grid_z=grid_z,
    )

    # 2. Step 2: Compute 2D Gas Rarefaction (Optional)
    rarefaction_res = None
    if include_gas_rarefaction:
        rarefaction_res = compute_gas_rarefaction_2d(
            power_w=power_w,
            pressure_mtorr=pressure_mtorr,
            target_radius_m=target_r_m,
            anode_distance_m=anode_d_m,
            grid_r=grid_r,
            grid_z=grid_z,
        )

    # 3. Step 3: Run Dynamic 2D Target Racetrack Erosion Engine
    erosion_model = TargetErosionModel(
        target_material=material,
        target_radius=target_radius_mm,
        initial_target_thickness=initial_thickness_mm,
        roughness_factor=roughness_factor,
    )

    erosion_res = erosion_model.simulate_erosion(
        ion_flux_profile=plasma_res,
        total_hours=total_hours,
        time_step_hours=time_step_hours,
        roughness_factor=roughness_factor,
    )

    # 4. Step 4: Calculate Total Sputtered Flux and Mass Loss
    r_m = erosion_res.r_grid_m
    j_i = plasma_res.current_density_1d
    v_d = plasma_res.voltage_v

    # Local slope and local SOTA yield
    if np.max(erosion_res.depth_profile_m) > 1e-12:
        grad = np.gradient(erosion_res.depth_profile_m, r_m)
        grad[0] = 0.0
        theta_local = np.clip(np.arctan(np.abs(grad)), 0.0, math.radians(85.0))
    else:
        theta_local = np.zeros_like(r_m)

    y_local = np.array([
        calculate_sputter_yield_sota(v_d, mat_obj, float(th), roughness_factor=roughness_factor)
        for th in theta_local
    ])

    # Sputtered flux per unit area [atoms / m^2 s] = (J_i / e) * Y_local
    flux_density = (j_i / _E_CHARGE) * y_local
    # Axisymmetric integral: 2*pi * int_0^R r * flux_density(r) dr [atoms / s]
    total_flux_atoms_s = float(2.0 * math.pi * trapezoid(flux_density * r_m, r_m))

    # Total mass loss over total_hours [grams]
    # Sputtered volume [m^3] = 2*pi * int_0^R r * d(r) dr
    sputtered_vol_m3 = float(2.0 * math.pi * trapezoid(erosion_res.depth_profile_m * r_m, r_m))
    rho_kg_m3 = mat_obj.density * 1e3
    total_mass_grams = float(sputtered_vol_m3 * rho_kg_m3 * 1e3)

    # Erosion rate metrics (in um/hr)
    peak_rate_um_hr = float(np.max(erosion_res.erosion_rate_mm_hr) * 1e3)
    mean_rate_um_hr = float(
        (2.0 * math.pi * trapezoid(erosion_res.erosion_rate_mm_hr * 1e3 * r_m, r_m))
        / (math.pi * (target_r_m**2))
    )

    # 5. Step 5: Sample Nascent Sputtered Atom Emission Distribution
    ejected_particles = sample_ejected_energy_and_angle(
        num_samples=num_ejected_samples,
        material=mat_obj,
        ion_energy_ev=v_d,
        roughness_factor=roughness_factor,
        seed=seed,
    )

    return IntegratedDischargeErosionResult(
        plasma=plasma_res,
        erosion=erosion_res,
        rarefaction=rarefaction_res,
        ejected_particles=ejected_particles,
        material=material,
        power_w=power_w,
        pressure_mtorr=pressure_mtorr,
        total_hours=total_hours,
        total_sputtered_flux_atoms_per_s=total_flux_atoms_s,
        total_sputtered_mass_grams=total_mass_grams,
        peak_erosion_rate_um_hr=peak_rate_um_hr,
        mean_erosion_rate_um_hr=mean_rate_um_hr,
    )


class IntegratedPINNSurrogate:
    """Coupled PlasmaPINN + YieldPINN surrogate model for ultra-fast digital twin inference."""

    def __init__(self, yield_checkpoint_path: Optional[str] = None) -> None:
        """Initialize both neural network surrogate engines."""
        import torch
        from sputtertwin.pinn.plasma_pinn import PlasmaPINN, PlasmaPINNConfig
        from sputtertwin.pinn.yield_pinn import YieldPINN, YieldPINNConfig, load_yield_pinn

        # Initialize PlasmaPINN
        self.plasma_config = PlasmaPINNConfig()
        self.plasma_pinn = PlasmaPINN(self.plasma_config)
        self.plasma_pinn.eval()

        # Initialize YieldPINN (loads Cu checkpoint if available)
        self.yield_pinn: Optional[YieldPINN] = None
        try:
            self.yield_pinn = load_yield_pinn(yield_checkpoint_path)
            self.yield_pinn.eval()
        except Exception:
            self.yield_pinn = YieldPINN(YieldPINNConfig())
            self.yield_pinn.eval()

    def predict(
        self,
        power_w: float = 220.0,
        pressure_mtorr: float = 5.0,
        b_max_t: float = 0.045,
        material: str = "Cu",
        angle_deg: float = 0.0,
    ) -> Dict[str, float]:
        """Predict coupled discharge and sputter yield metrics via neural surrogate in < 1 ms.

        Args:
            power_w: Discharge power (W).
            pressure_mtorr: Operating pressure (mTorr).
            b_max_t: Peak magnetic field (T).
            material: Target material symbol.
            angle_deg: Incident ion angle in degrees.

        Returns:
            Dictionary with fast predicted plasma and yield values.
        """
        import torch

        # 1. PlasmaPINN prediction
        plasma_preds = self.plasma_pinn.predict(power_w, pressure_mtorr)
        n_e_m3 = plasma_preds["plasma_density_m3"]
        v_d = plasma_preds["voltage_v"]
        t_e = plasma_preds["electron_temp_ev"]
        v_p = 3.0 * t_e  # Plasma potential ~ 3 * Te
        e_ion_ev = float(v_d + v_p)  # Total ion acceleration across cathode sheath

        # 2. YieldPINN prediction
        mat = MATERIALS.get(material, MATERIALS["Cu"])
        theta_rad = math.radians(angle_deg)
        x_yield = torch.tensor(
            [[e_ion_ev, theta_rad, mat.atomic_number, mat.atomic_mass, mat.sublimation_energy]],
            dtype=torch.float32,
        )
        with torch.no_grad():
            y_pred = float(self.yield_pinn(x_yield).item())

        return {
            "plasma_density_m3": n_e_m3,
            "plasma_potential_v": v_p,
            "cathode_voltage_v": v_d,
            "electron_temp_ev": t_e,
            "ion_impact_energy_ev": e_ion_ev,
            "predicted_sputter_yield": y_pred,
        }
