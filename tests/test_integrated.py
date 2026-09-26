"""Unit and Integration tests for the End-to-End Integrated Simulation Pipeline.

Validates the full coupling between Stage 1 (Plasma & Magnetic Field)
and Stage 2 (Sputter Yield & Dynamic Target Racetrack Erosion).
"""

from __future__ import annotations

import importlib
import math
import unittest

import torch
import numpy as np

from sputtertwin import (
    simulate_integrated_discharge_and_erosion,
    IntegratedDischargeErosionResult,
    IntegratedPINNSurrogate,
    TargetErosionModel,
    simulate_plasma_2d,
    calculate_sputter_yield_sota,
)


class TestIntegratedPipeline(unittest.TestCase):
    """Test suite for integrated Stage 1 and Stage 2 multiphysics workflow."""

    def test_integrated_simulation_copper(self):
        """Verify full Stage 1 -> Stage 2 coupled simulation for a Copper target."""
        res = simulate_integrated_discharge_and_erosion(
            power_w=220.0,
            pressure_mtorr=5.0,
            material="Cu",
            total_hours=5.0,
            time_step_hours=1.0,
            target_radius_mm=25.0,
            initial_thickness_mm=6.0,
            roughness_factor=0.15,
            grid_r=30,
            grid_z=25,
            num_ejected_samples=2000,
            seed=42,
        )

        self.assertIsInstance(res, IntegratedDischargeErosionResult)
        self.assertEqual(res.material, "Cu")
        self.assertEqual(res.power_w, 220.0)
        self.assertEqual(res.pressure_mtorr, 5.0)

        # 1. Stage 1 Plasma checks
        self.assertGreater(res.plasma.voltage_v, 300.0)
        self.assertLess(res.plasma.voltage_v, 600.0)
        self.assertGreater(res.plasma.current_a, 0.2)
        self.assertGreater(np.max(res.plasma.current_density_1d), 50.0)  # A/m^2

        # 2. Stage 2 Target Erosion checks
        self.assertGreater(res.erosion.peak_erosion_depth_mm, 0.5)
        self.assertLess(res.erosion.peak_erosion_depth_mm, 5.0)
        self.assertGreater(res.erosion.racetrack_fwhm_mm, 4.0)
        self.assertGreater(res.erosion.target_utilization_efficiency_pct, 10.0)
        self.assertLess(res.erosion.target_utilization_efficiency_pct, 50.0)
        self.assertGreater(res.erosion.target_lifetime_hours, 5.0)
        self.assertFalse(res.erosion.breakthrough_occurred)

        # 3. Mass and flux conservation
        self.assertGreater(res.total_sputtered_flux_atoms_per_s, 1e16)
        self.assertGreater(res.total_sputtered_mass_grams, 0.05)
        self.assertLess(res.total_sputtered_mass_grams, 20.0)
        self.assertGreater(res.peak_erosion_rate_um_hr, 100.0)

        # 4. Nascent atom emission properties (Thomson distribution)
        self.assertEqual(len(res.ejected_particles.energies), 2000)
        self.assertEqual(len(res.ejected_particles.angles), 2000)
        mean_e = float(np.mean(res.ejected_particles.energies))
        # For Cu with Us = 3.49 eV, mean energy is typically ~ 5 - 15 eV
        self.assertGreater(mean_e, 3.0)
        self.assertLess(mean_e, 30.0)

        # 5. Output dictionaries and report summary
        res_dict = res.to_dict()
        self.assertIn("discharge_voltage_v", res_dict)
        self.assertIn("peak_erosion_depth_mm", res_dict)
        self.assertIn("target_utilization_pct", res_dict)
        summary_str = res.summary()
        self.assertIn("SPUTTERTWIN INTEGRATED", summary_str)
        self.assertIn("Stage 1", summary_str)
        self.assertIn("Stage 2", summary_str)

    def test_integrated_simulation_titanium_and_aluminum(self):
        """Verify integrated simulation works reliably across multiple semiconductor materials."""
        # Titanium
        res_ti = simulate_integrated_discharge_and_erosion(
            power_w=200.0,
            pressure_mtorr=5.0,
            material="Ti",
            total_hours=2.0,
            grid_r=25,
            grid_z=20,
            num_ejected_samples=1000,
        )
        self.assertEqual(res_ti.material, "Ti")
        self.assertGreater(res_ti.erosion.peak_erosion_depth_mm, 0.1)

        # Aluminum
        res_al = simulate_integrated_discharge_and_erosion(
            power_w=200.0,
            pressure_mtorr=5.0,
            material="Al",
            total_hours=2.0,
            grid_r=25,
            grid_z=20,
            num_ejected_samples=1000,
        )
        self.assertEqual(res_al.material, "Al")
        self.assertGreater(res_al.erosion.peak_erosion_depth_mm, 0.1)

    def test_integrated_pinn_surrogate(self):
        """Verify coupled PlasmaPINN + YieldPINN surrogate model inference."""
        surrogate = IntegratedPINNSurrogate()
        preds = surrogate.predict(
            power_w=220.0,
            pressure_mtorr=5.0,
            b_max_t=0.045,
            material="Cu",
            angle_deg=0.0,
        )

        self.assertIn("plasma_density_m3", preds)
        self.assertIn("plasma_potential_v", preds)
        self.assertIn("cathode_voltage_v", preds)
        self.assertIn("predicted_sputter_yield", preds)

        # Check physical ranges
        self.assertGreater(preds["plasma_density_m3"], 0.0)
        self.assertGreater(preds["cathode_voltage_v"], 0.0)
        self.assertGreater(preds["predicted_sputter_yield"], 0.0)
        self.assertLess(preds["predicted_sputter_yield"], 10.0)


if __name__ == "__main__":
    unittest.main()
