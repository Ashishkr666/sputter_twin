"""Unit tests for the 2D Dynamic Target Racetrack Erosion Engine.

Validates:
1. TargetErosionModel initialization, atomic density n_target, and input validation.
2. Erosion rate calculation d(depth)/dt = J_i * Y(E, theta) / (e * n_target) and tilt enhancement.
3. Groove deepening over 10 hours with monotonic depth progression and groove widening.
4. Target utilization efficiency calculation (including exact analytical benchmark).
5. Target lifetime and breakthrough detection.
6. Direct coupling with 2D multiphysics plasma solver (Plasma2DResult).
7. 2D Cartesian surface reconstruction of racetrack groove.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from sputtertwin.physics.sputter_yield import (
    MATERIALS,
    calculate_sputter_yield,
)
from sputtertwin.physics.target_erosion import (
    TargetErosionModel,
    TargetErosionResult,
)
from sputtertwin.physics2d.plasma2d import simulate_plasma_2d


class TestTargetErosion(unittest.TestCase):
    """Test suite for TargetErosionModel and target racetrack erosion physics."""

    def setUp(self):
        self.radius_mm = 25.0
        self.thickness_mm = 6.0
        self.cu_model = TargetErosionModel(
            target_material="Cu",
            target_radius=self.radius_mm,
            initial_target_thickness=self.thickness_mm,
        )

    def test_target_erosion_model_initialization(self):
        """Verify model initialization, material resolution, and n_target values."""
        # Check Cu target
        self.assertEqual(self.cu_model.target_material, "Cu")
        self.assertAlmostEqual(self.cu_model.target_radius_mm, 25.0)
        self.assertAlmostEqual(self.cu_model.target_radius_m, 0.025)
        self.assertAlmostEqual(self.cu_model.initial_target_thickness_mm, 6.0)
        self.assertAlmostEqual(self.cu_model.initial_target_thickness_m, 0.006)

        # Theoretical n_target = (rho * 1e6 * N_A) / M
        rho_cu = 8.96
        m_cu = 63.55
        expected_n_cu = (rho_cu * 1e6 * 6.02214076e23) / m_cu
        self.assertAlmostEqual(self.cu_model.n_target, expected_n_cu, delta=1e24)

        # Check Ti and Al targets
        ti_model = TargetErosionModel(target_material="Ti", target_radius=30.0, initial_target_thickness=5.0)
        expected_n_ti = (4.506 * 1e6 * 6.02214076e23) / 47.87
        self.assertAlmostEqual(ti_model.n_target, expected_n_ti, delta=1e24)

        al_model = TargetErosionModel(target_material="Al", target_radius=25.0, initial_target_thickness=6.0)
        expected_n_al = (2.70 * 1e6 * 6.02214076e23) / 26.98
        self.assertAlmostEqual(al_model.n_target, expected_n_al, delta=1e24)

        # Invalid material check
        with self.assertRaises(KeyError):
            TargetErosionModel(target_material="Unobtainium")

        # Invalid geometry check
        with self.assertRaises(ValueError):
            TargetErosionModel(target_material="Cu", target_radius=-10.0)
        with self.assertRaises(ValueError):
            TargetErosionModel(target_material="Cu", initial_target_thickness=0.0)

    def test_erosion_rate_calculation(self):
        """Verify d(depth)/dt calculation, angular tilt enhancement, and unit scaling."""
        j_i = 150.0  # A/m^2
        v_d = 400.0  # V
        e_charge = 1.602176634e-19

        # 1. Normal incidence erosion rate
        rate_m_s = self.cu_model.calculate_erosion_rate(
            current_density_a_m2=j_i,
            discharge_voltage_v=v_d,
            theta_local_rad=0.0,
            unit="m/s",
        )
        y0 = calculate_sputter_yield(v_d, "Cu", 0.0)
        expected_rate_m_s = (j_i / e_charge) * y0 / self.cu_model.n_target
        self.assertAlmostEqual(rate_m_s, expected_rate_m_s, places=12)

        # 2. mm/h conversion check (1 m/s = 3.6e6 mm/h)
        rate_mm_h = self.cu_model.calculate_erosion_rate(
            current_density_a_m2=j_i,
            discharge_voltage_v=v_d,
            theta_local_rad=0.0,
            unit="mm/h",
        )
        self.assertAlmostEqual(rate_mm_h, rate_m_s * 3.6e6, places=6)

        # 3. Local surface tilt enhancement (theta > 0)
        theta_tilted = math.radians(35.0)
        rate_tilted = self.cu_model.calculate_erosion_rate(
            current_density_a_m2=j_i,
            discharge_voltage_v=v_d,
            theta_local_rad=theta_tilted,
            unit="mm/h",
        )
        # Due to Yamamura angular dependence, rate at 35 deg must be significantly higher than normal
        self.assertGreater(
            rate_tilted,
            rate_mm_h,
            "Groove tilt enhancement must increase erosion rate at oblique incidence",
        )

        # 4. Sub-threshold voltage must yield zero erosion
        sub_thresh_rate = self.cu_model.calculate_erosion_rate(
            current_density_a_m2=j_i,
            discharge_voltage_v=15.0,  # Below Cu Eth = 20 eV
            theta_local_rad=0.0,
        )
        self.assertEqual(sub_thresh_rate, 0.0)

    def test_groove_deepening_over_10_hours(self):
        """Verify dynamic groove depth evolution over 10 operational hours."""
        # Create a synthetic Gaussian racetrack profile peaked at r = 12.5 mm
        r_grid_m = np.linspace(0.0, 0.025, 60)
        r_race_m = 0.0125
        sigma_m = 0.004
        # J_i peaked at ~200 A/m^2
        j_i_profile = 200.0 * np.exp(-((r_grid_m - r_race_m) / sigma_m) ** 2)

        res = self.cu_model.simulate_erosion(
            ion_flux_profile=j_i_profile,
            discharge_voltage_v=400.0,
            total_hours=10.0,
            time_step_hours=0.5,
            r_grid=r_grid_m,
        )

        # 1. Output structure assertions
        self.assertIsInstance(res, TargetErosionResult)
        self.assertEqual(len(res.time_hours), 21)  # 0.0 to 10.0 h with 0.5 h steps
        self.assertEqual(res.depth_history_mm.shape, (21, 60))

        # 2. Monotonic deepening over time:
        # Check depth at 0h, 2.5h, 5h, 7.5h, 10h
        peak_depths = [np.max(res.depth_history_mm[step]) for step in [0, 5, 10, 15, 20]]
        for i in range(len(peak_depths) - 1):
            self.assertLess(
                peak_depths[i],
                peak_depths[i + 1],
                f"Peak depth must strictly increase over time: step {i} vs {i+1}",
            )

        # 3. Peak erosion depth value check
        # At 200 A/m^2, Cu rate is ~0.10 mm/h -> ~1.0 mm after 10 h
        self.assertGreater(res.peak_erosion_depth_mm, 0.5)
        self.assertLess(res.peak_erosion_depth_mm, 2.5)

        # 4. Groove widening (FWHM expansion due to tilt enhancement)
        # Compare FWHM of depth at 1 hour vs 10 hours
        fwhm_1h = self.cu_model.calculate_fwhm(res.depth_history_mm[2], res.r_grid_mm)
        fwhm_10h = self.cu_model.calculate_fwhm(res.depth_history_mm[-1], res.r_grid_mm)
        self.assertGreaterEqual(
            fwhm_10h,
            fwhm_1h * 0.999,
            "Groove FWHM must expand or remain stable as sidewalls erode",
        )
        self.assertGreater(res.racetrack_fwhm_mm, 4.0)
        self.assertLess(res.racetrack_fwhm_mm, 15.0)

        # 5. Check convenience properties on TargetErosionModel
        self.assertEqual(self.cu_model.peak_erosion_depth_mm, res.peak_erosion_depth_mm)
        self.assertEqual(self.cu_model.racetrack_fwhm_mm, res.racetrack_fwhm_mm)

    def test_target_utilization_calculation(self):
        """Verify target utilization efficiency calculation against analytical ground truth."""
        r_grid_m = np.linspace(0.0, 0.025, 100)
        r_target_m = 0.025
        t_target_m = 0.006

        # 1. Analytical test: Uniform flat erosion of depth d0 across entire target
        # V_sputtered = pi * R^2 * d0
        # V_total = pi * R^2 * T0
        # Utilization = d0 / T0 * 100%
        d0_m = 0.0015  # 1.5 mm
        flat_depth_profile = np.full_like(r_grid_m, d0_m)

        calc_util_pct = self.cu_model.calculate_target_utilization(
            flat_depth_profile, r_grid=r_grid_m
        )
        expected_util_pct = (d0_m / t_target_m) * 100.0  # 25.0%
        self.assertAlmostEqual(calc_util_pct, expected_util_pct, delta=0.5)

        # 2. Zero depth must yield 0% utilization
        zero_util = self.cu_model.calculate_target_utilization(
            np.zeros_like(r_grid_m), r_grid=r_grid_m
        )
        self.assertEqual(zero_util, 0.0)

        # 3. Realistic racetrack erosion utilization
        # Racetrack ring typically occupies ~20-40% of target area
        sigma = 0.004
        r_race = 0.0125
        racetrack_depth = 0.003 * np.exp(-((r_grid_m - r_race) / sigma) ** 2)
        race_util = self.cu_model.calculate_target_utilization(
            racetrack_depth, r_grid=r_grid_m
        )
        self.assertGreater(race_util, 5.0)
        self.assertLess(race_util, 35.0)

    def test_target_lifetime_and_breakthrough(self):
        """Verify target lifetime calculation and breakthrough detection."""
        r_grid_m = np.linspace(0.0, 0.025, 50)
        # High ion current density (~600 A/m^2) to drive breakthrough within 25 hours
        j_i_profile = 600.0 * np.exp(-((r_grid_m - 0.0125) / 0.004) ** 2)

        # 1. Run for 5 hours (before breakthrough)
        res_5h = self.cu_model.simulate_erosion(
            ion_flux_profile=j_i_profile,
            discharge_voltage_v=400.0,
            total_hours=5.0,
            time_step_hours=0.5,
            r_grid=r_grid_m,
        )
        self.assertFalse(res_5h.breakthrough_occurred)
        self.assertIsNone(res_5h.breakthrough_time_hours)
        self.assertGreater(res_5h.target_lifetime_hours, 15.0)
        self.assertLess(res_5h.target_lifetime_hours, 25.0)

        # 2. Run for 25 hours (triggers breakthrough through 6.0 mm target)
        res_25h = self.cu_model.simulate_erosion(
            ion_flux_profile=j_i_profile,
            discharge_voltage_v=400.0,
            total_hours=25.0,
            time_step_hours=0.5,
            r_grid=r_grid_m,
        )
        self.assertTrue(res_25h.breakthrough_occurred)
        self.assertIsNotNone(res_25h.breakthrough_time_hours)
        self.assertGreater(res_25h.breakthrough_time_hours, 18.0)
        self.assertLess(res_25h.breakthrough_time_hours, 22.0)
        # Clamped peak depth must not exceed initial thickness 6.0 mm
        self.assertAlmostEqual(res_25h.peak_erosion_depth_mm, 6.0, places=5)
        # Lifetime must equal breakthrough time
        self.assertAlmostEqual(
            res_25h.target_lifetime_hours,
            res_25h.breakthrough_time_hours,
            places=5,
        )

    def test_plasma2d_multiphysics_coupling(self):
        """Verify seamless end-to-end coupling with simulate_plasma_2d result."""
        # 1. Solve 2D plasma discharge
        plasma_res = simulate_plasma_2d(
            power_w=220.0,
            pressure_mtorr=5.0,
            ar_flow_sccm=20.0,
            target_radius_m=0.025,
            anode_distance_m=0.040,
            grid_r=40,
            grid_z=30,
        )

        # 2. Feed Plasma2DResult directly into TargetErosionModel
        model = TargetErosionModel(
            target_material="Cu",
            target_radius=25.0,
            initial_target_thickness=6.0,
        )
        res = model.simulate_erosion(
            ion_flux_profile=plasma_res,
            total_hours=10.0,
            time_step_hours=0.5,
        )

        # Verify all 4 required metrics are computed
        self.assertGreater(res.peak_erosion_depth_mm, 1.0)
        self.assertLess(res.peak_erosion_depth_mm, 5.0)

        self.assertGreater(res.racetrack_fwhm_mm, 4.0)
        self.assertLess(res.racetrack_fwhm_mm, 15.0)

        self.assertGreater(res.target_utilization_efficiency_pct, 10.0)
        self.assertLess(res.target_utilization_efficiency_pct, 40.0)

        self.assertGreater(res.target_lifetime_hours, 10.0)
        self.assertLess(res.target_lifetime_hours, 30.0)

        # 3. Test 2D Cartesian surface map reconstruction
        X, Y, Z = res.reconstruct_2d_depth_map(grid_size=40)
        self.assertEqual(Z.shape, (40, 40))
        # Peak depth on 2D disk should match 1D radial peak
        self.assertAlmostEqual(float(np.nanmax(Z)), res.peak_erosion_depth_mm, delta=0.1)


if __name__ == "__main__":
    unittest.main()
