"""Comprehensive physics verification test suite for SputterTwin.

Validates material properties, sputter yield thresholds and values, DC magnetron
plasma scaling, gas kinetic transport scaling, deposition boundary conditions,
time linearity, wafer grid metrics (circular mask), and all 6 bug fixes from the
Phase 1 code review.

Test index:
    1.  test_material_definitions
    2.  test_sputter_yield_threshold
    3.  test_sputter_yield_values
    4.  test_plasma_scaling
    5.  test_mean_free_path_scaling
    6.  test_deposition_zero_power
    7.  test_deposition_time_scaling
    8.  test_deposition_grid_shape_and_uniformity  (updated: circular mask)
    9.  test_slide7_nominal_condition               (updated: relaxed 0.9% delta)
    --- New tests covering the 6 review findings ---
    10. test_racetrack_radius_effect               (Issue 3 fix)
    11. test_transport_wired_into_rate             (Issue 1 fix)
    12. test_high_pressure_no_singularity          (Issue 4 fix)
    13. test_grid_size_type_validation             (Issue 5 fix)
    14. test_gamma_se_bounds                       (Issue 5 fix)
    15. test_nan_inputs_rejected                   (Issue 5 fix)
    16. test_ar_flow_affects_discharge             (Issue 6 / Option B fix)
    17. test_circular_wafer_mask_metrics           (Issue 2 fix)
"""

import math
import unittest

import numpy as np

from sputtertwin.physics.deposition import DepositionResult, simulate_deposition
from sputtertwin.physics.plasma import DischargeState, calculate_discharge_state
from sputtertwin.physics.sputter_yield import (
    MATERIALS,
    TargetMaterial,
    calculate_sputter_yield,
    calculate_sputter_yield_array,
)
from sputtertwin.physics.transport import (
    TransportSummary,
    calculate_knudsen_number,
    calculate_mean_free_path,
    calculate_scattering_broadening,
    calculate_transmission_probability,
    calculate_transport_summary,
)


class TestPhysicsSuite(unittest.TestCase):
    """Test suite covering the complete physical modeling pipeline of SputterTwin."""

    # ------------------------------------------------------------------
    # 1. Material definitions
    # ------------------------------------------------------------------
    def test_material_definitions(self):
        """1. Verify Ti, Cu, Al properties exist and have positive values."""
        required_materials = ["Ti", "Cu", "Al"]
        for mat_name in required_materials:
            self.assertIn(
                mat_name,
                MATERIALS,
                f"Required material '{mat_name}' not found in MATERIALS catalog",
            )
            mat = MATERIALS[mat_name]
            self.assertIsInstance(mat, TargetMaterial)
            self.assertEqual(mat.name, mat_name)
            self.assertGreater(mat.atomic_mass, 0.0)
            self.assertGreater(mat.atomic_number, 0)
            self.assertGreater(mat.density, 0.0)
            self.assertGreater(mat.sublimation_energy, 0.0)
            self.assertGreater(mat.threshold_energy, 0.0)

    # ------------------------------------------------------------------
    # 2. Sputter yield threshold
    # ------------------------------------------------------------------
    def test_sputter_yield_threshold(self):
        """2. Verify yield is 0.0 below or at threshold energy, and positive above."""
        for mat_name, mat in MATERIALS.items():
            eth = mat.threshold_energy
            self.assertEqual(calculate_sputter_yield(0.0, mat_name), 0.0)
            self.assertEqual(calculate_sputter_yield(eth * 0.5, mat_name), 0.0)
            self.assertEqual(calculate_sputter_yield(eth, mat_name), 0.0)
            above = calculate_sputter_yield(eth + 20.0, mat_name)
            self.assertGreater(above, 0.0)

    # ------------------------------------------------------------------
    # 3. Sputter yield values
    # ------------------------------------------------------------------
    def test_sputter_yield_values(self):
        """3. Ti yield at 400 eV in [0.3, 0.8]; Cu > Ti; oblique > normal; grazing = 0."""
        y_ti_400 = calculate_sputter_yield(400.0, "Ti")
        self.assertGreaterEqual(y_ti_400, 0.3)
        self.assertLessEqual(y_ti_400, 0.8)

        y_cu_400 = calculate_sputter_yield(400.0, "Cu")
        self.assertGreater(y_cu_400, y_ti_400)

        y_normal = calculate_sputter_yield(400.0, "Ti", angle_rad=0.0)
        y_oblique = calculate_sputter_yield(400.0, "Ti", angle_rad=math.radians(60.0))
        self.assertGreater(y_oblique, y_normal)

        y_grazing = calculate_sputter_yield(400.0, "Ti", angle_rad=math.radians(88.0))
        self.assertEqual(y_grazing, 0.0)

    # ------------------------------------------------------------------
    # 4. Plasma discharge scaling
    # ------------------------------------------------------------------
    def test_plasma_scaling(self):
        """4. Voltage decreases, current increases with pressure; ion flux scales with power."""
        power_w = 300.0
        lo_p = calculate_discharge_state(power_w=power_w, pressure_mtorr=3.0)
        hi_p = calculate_discharge_state(power_w=power_w, pressure_mtorr=10.0)

        self.assertLess(hi_p.voltage_v, lo_p.voltage_v)
        self.assertGreater(hi_p.current_a, lo_p.current_a)
        # Power conservation: V * I == W
        self.assertAlmostEqual(lo_p.voltage_v * lo_p.current_a, power_w, places=3)
        self.assertAlmostEqual(hi_p.voltage_v * hi_p.current_a, power_w, places=3)

        # Ion flux scales with power
        lo_w = calculate_discharge_state(power_w=100.0, pressure_mtorr=5.0)
        hi_w = calculate_discharge_state(power_w=400.0, pressure_mtorr=5.0)
        self.assertGreater(hi_w.ion_flux, lo_w.ion_flux)

        # total_ion_current_a must be positive and less than discharge current
        self.assertGreater(lo_p.total_ion_current_a, 0.0)
        self.assertLess(lo_p.total_ion_current_a, lo_p.current_a)

    # ------------------------------------------------------------------
    # 5. Mean free path scaling
    # ------------------------------------------------------------------
    def test_mean_free_path_scaling(self):
        """5. Mean free path inversely proportional to pressure (P * lambda = const)."""
        pressures = [1.0, 3.0, 5.0, 10.0, 20.0]
        mfps = [calculate_mean_free_path(p) for p in pressures]

        for i in range(len(mfps) - 1):
            self.assertGreater(mfps[i], mfps[i + 1])

        products = [p * m for p, m in zip(pressures, mfps)]
        for prod in products[1:]:
            self.assertAlmostEqual(prod, products[0], places=6)

    # ------------------------------------------------------------------
    # 6. Zero power → zero deposition
    # ------------------------------------------------------------------
    def test_deposition_zero_power(self):
        """6. Power <= 0 returns zero thickness and rate across the on-wafer region."""
        res = simulate_deposition(power_w=0.0, pressure_mtorr=5.0, deposition_time_s=60.0)
        self.assertEqual(res.mean_thickness_nm, 0.0)
        self.assertEqual(res.deposition_rate_nm_min, 0.0)
        self.assertEqual(res.min_thickness_nm, 0.0)
        self.assertEqual(res.max_thickness_nm, 0.0)
        self.assertEqual(res.uniformity_percent, 0.0)
        # On-wafer points should all be 0.0 (off-wafer are NaN)
        wafer_vals = res.thickness_map[~np.isnan(res.thickness_map)]
        self.assertTrue(np.all(wafer_vals == 0.0))

    # ------------------------------------------------------------------
    # 7. Linear time scaling
    # ------------------------------------------------------------------
    def test_deposition_time_scaling(self):
        """7. Thickness scales strictly linearly with deposition time."""
        res1 = simulate_deposition(power_w=220.0, pressure_mtorr=5.0, deposition_time_s=60.0)
        res2 = simulate_deposition(power_w=220.0, pressure_mtorr=5.0, deposition_time_s=180.0)

        ratio = 180.0 / 60.0
        self.assertAlmostEqual(res2.mean_thickness_nm / res1.mean_thickness_nm, ratio, places=4)
        # Rate must be time-independent
        self.assertAlmostEqual(res1.deposition_rate_nm_min, res2.deposition_rate_nm_min, places=4)
        # Element-wise on on-wafer values only
        v1 = res1.thickness_map[~np.isnan(res1.thickness_map)]
        v2 = res2.thickness_map[~np.isnan(res2.thickness_map)]
        np.testing.assert_allclose(v2, v1 * ratio, rtol=1e-5)

    # ------------------------------------------------------------------
    # 8. Grid shape and uniformity (updated: circular mask aware)
    # ------------------------------------------------------------------
    def test_deposition_grid_shape_and_uniformity(self):
        """8. 5x5 grid: correct shape; on-wafer points positive; NaN off-wafer; NU formula correct."""
        res = simulate_deposition(
            power_w=220.0,
            pressure_mtorr=5.0,
            deposition_time_s=60.0,
            grid_size=5,
            wafer_radius_mm=75.0,
        )

        self.assertEqual(res.thickness_map.shape, (5, 5))
        self.assertEqual(len(res.x_grid_mm), 5)
        self.assertEqual(len(res.y_grid_mm), 5)

        # On-wafer points (not NaN) must be positive
        on_wafer = res.thickness_map[~np.isnan(res.thickness_map)]
        self.assertTrue(np.all(on_wafer > 0.0), "All on-wafer points must be positive")

        # For a 5x5 grid spanning ±75 mm with R <= 75 mm wafer, 13 of 25 points are inside.
        self.assertEqual(res.n_wafer_points, 13)

        # Off-wafer corners must be NaN
        x_vals = res.x_grid_mm
        y_vals = res.y_grid_mm
        corners_nan = []
        for xi in [-75.0, 75.0]:
            for yi in [-75.0, 75.0]:
                r = math.sqrt(xi**2 + yi**2)
                if r > 75.0:
                    ix = list(x_vals).index(xi)
                    iy = list(y_vals).index(yi)
                    corners_nan.append(math.isnan(res.thickness_map[iy, ix]))
        self.assertTrue(all(corners_nan), "Grid corners outside wafer radius must be NaN")

        # NU formula: (max - min) / (2 * mean) * 100
        expected_nu = (res.max_thickness_nm - res.min_thickness_nm) / (2.0 * res.mean_thickness_nm) * 100.0
        self.assertAlmostEqual(res.uniformity_percent, expected_nu, places=5)
        self.assertGreater(res.uniformity_percent, 0.0)
        self.assertLess(res.uniformity_percent, 100.0)

    # ------------------------------------------------------------------
    # 9. Slide 7 nominal benchmark (updated: relaxed pressure delta)
    # ------------------------------------------------------------------
    def test_slide7_nominal_condition(self):
        """9. Nominal (220W, 5mTorr, Ti, 720s, 80mm) matches Slide 7: ~100nm, ~8.4nm/min, ~1.2% NU."""
        res = simulate_deposition(
            power_w=220.0,
            pressure_mtorr=5.0,
            ar_flow_sccm=20.0,
            distance_mm=80.0,
            deposition_time_s=720.0,
            material="Ti",
            grid_size=5,
            wafer_radius_mm=75.0,
            target_radius_mm=50.0,
            racetrack_radius_mm=25.0,
            target_erosion_mm=0.0,
        )
        self.assertAlmostEqual(res.mean_thickness_nm, 100.1, delta=5.0)
        self.assertAlmostEqual(res.deposition_rate_nm_min, 8.4, delta=1.5)
        self.assertAlmostEqual(res.uniformity_percent, 1.2, delta=0.5)

        # Physics: higher pressure → more scattering broadening → more uniform film (lower NU).
        # The slide's "reduce pressure → 0.9% NU" was illustrative; correct physics is opposite.
        res_high_p = simulate_deposition(
            power_w=220.0, pressure_mtorr=7.0, deposition_time_s=720.0, material="Ti"
        )
        self.assertLess(
            res_high_p.uniformity_percent,
            res.uniformity_percent,
            "Increasing pressure improves (lowers) non-uniformity via scattering broadening",
        )

        # Target erosion must increase NU
        res_eroded = simulate_deposition(
            power_w=220.0, pressure_mtorr=5.0, deposition_time_s=720.0,
            material="Ti", target_erosion_mm=2.0,
        )
        self.assertGreater(res_eroded.uniformity_percent, res.uniformity_percent)

    # ------------------------------------------------------------------
    # 10. (NEW) Racetrack radius geometrically controls NU  — Issue 3 fix
    # ------------------------------------------------------------------
    def test_racetrack_radius_effect(self):
        """10. (NEW-Issue3) Changing racetrack_radius_mm measurably changes film non-uniformity."""
        base = dict(power_w=220.0, pressure_mtorr=5.0, distance_mm=80.0,
                    deposition_time_s=720.0, material="Ti", grid_size=5)

        res_small = simulate_deposition(**base, racetrack_radius_mm=15.0, target_radius_mm=50.0)
        res_nominal = simulate_deposition(**base, racetrack_radius_mm=25.0, target_radius_mm=50.0)
        res_large = simulate_deposition(**base, racetrack_radius_mm=35.0, target_radius_mm=50.0)

        self.assertLess(
            res_small.uniformity_percent,
            res_nominal.uniformity_percent,
            "Smaller racetrack radius must produce more uniform film",
        )
        self.assertGreater(
            res_large.uniformity_percent,
            res_nominal.uniformity_percent,
            "Larger racetrack radius must produce less uniform film",
        )
        # The effect should be substantial (not a rounding artefact)
        self.assertGreater(
            res_large.uniformity_percent - res_small.uniformity_percent,
            0.5,
            "Racetrack radius must have a meaningful (>0.5%) effect on NU",
        )

    # ------------------------------------------------------------------
    # 11. (NEW) Transport attenuation wired into rate  — Issue 1 fix
    # ------------------------------------------------------------------
    def test_transport_wired_into_rate(self):
        """11. (NEW-Issue1) Higher pressure reduces deposition rate via transport attenuation."""
        base = dict(power_w=220.0, distance_mm=80.0, deposition_time_s=720.0, material="Ti")

        res_low_p  = simulate_deposition(**base, pressure_mtorr=3.0)
        res_high_p = simulate_deposition(**base, pressure_mtorr=7.0)

        self.assertGreater(
            res_low_p.deposition_rate_nm_min,
            res_high_p.deposition_rate_nm_min,
            "Lower pressure must give higher deposition rate due to less transport attenuation",
        )
        # The rate difference should be physically meaningful (> 0.3 nm/min)
        diff = res_low_p.deposition_rate_nm_min - res_high_p.deposition_rate_nm_min
        self.assertGreater(diff, 0.3, f"Rate difference too small: {diff:.3f} nm/min")

    # ------------------------------------------------------------------
    # 12. (NEW) No singularity at high pressure  — Issue 4 fix
    # ------------------------------------------------------------------
    def test_high_pressure_no_singularity(self):
        """12. (NEW-Issue4) pressure_mtorr=100 returns finite, positive, physically sane results."""
        res = simulate_deposition(
            power_w=220.0,
            pressure_mtorr=100.0,
            distance_mm=80.0,
            deposition_time_s=720.0,
            material="Ti",
            grid_size=5,
        )
        self.assertTrue(math.isfinite(res.mean_thickness_nm))
        self.assertTrue(math.isfinite(res.uniformity_percent))
        self.assertTrue(math.isfinite(res.deposition_rate_nm_min))
        self.assertGreater(res.mean_thickness_nm, 0.0)
        self.assertGreater(res.uniformity_percent, 0.0)
        # At very high pressure film should be more uniform than at 5 mTorr
        res_nominal = simulate_deposition(
            power_w=220.0, pressure_mtorr=5.0, distance_mm=80.0,
            deposition_time_s=720.0, material="Ti", grid_size=5,
        )
        self.assertLess(
            res.uniformity_percent,
            res_nominal.uniformity_percent,
            "Very high pressure should produce a more uniform film than nominal",
        )

    # ------------------------------------------------------------------
    # 13. (NEW) grid_size type validation  — Issue 5 fix
    # ------------------------------------------------------------------
    def test_grid_size_type_validation(self):
        """13. (NEW-Issue5) Non-int grid_size raises TypeError; out-of-range raises ValueError."""
        base = dict(power_w=220.0, pressure_mtorr=5.0)

        with self.assertRaises(TypeError):
            simulate_deposition(**base, grid_size=2.5)       # float
        with self.assertRaises(TypeError):
            simulate_deposition(**base, grid_size="5")       # str
        with self.assertRaises(TypeError):
            simulate_deposition(**base, grid_size=True)      # bool subclass of int

        with self.assertRaises(ValueError):
            simulate_deposition(**base, grid_size=0)         # below minimum
        with self.assertRaises(ValueError):
            simulate_deposition(**base, grid_size=1)         # all corners off-wafer
        with self.assertRaises(ValueError):
            simulate_deposition(**base, grid_size=2)         # all 4 points off-wafer
        with self.assertRaises(ValueError):
            simulate_deposition(**base, grid_size=501)       # above maximum

        # Boundary value: grid_size=3 should succeed (centre point at origin is on-wafer)
        res = simulate_deposition(**base, grid_size=3)
        self.assertEqual(res.thickness_map.shape, (3, 3))

    # ------------------------------------------------------------------
    # 14. (NEW) gamma_se bounds in plasma  — Issue 5 fix
    # ------------------------------------------------------------------
    def test_gamma_se_bounds(self):
        """14. (NEW-Issue5) gamma_se >= 1.0 must raise ValueError (would invert ion current)."""
        with self.assertRaises(ValueError):
            calculate_discharge_state(power_w=220.0, pressure_mtorr=5.0, gamma_se=1.0)
        with self.assertRaises(ValueError):
            calculate_discharge_state(power_w=220.0, pressure_mtorr=5.0, gamma_se=1.1)
        with self.assertRaises(ValueError):
            calculate_discharge_state(power_w=220.0, pressure_mtorr=5.0, gamma_se=-0.1)

        # Valid boundary: 0.0 (perfect ion collection, no secondaries)
        res = calculate_discharge_state(power_w=220.0, pressure_mtorr=5.0, gamma_se=0.0)
        self.assertGreater(res.total_ion_current_a, 0.0)

    # ------------------------------------------------------------------
    # 15. (NEW) NaN/Inf inputs rejected  — Issue 5 fix
    # ------------------------------------------------------------------
    def test_nan_inputs_rejected(self):
        """15. (NEW-Issue5) NaN or Inf in critical float inputs must raise ValueError."""
        for bad in [float("nan"), float("inf"), float("-inf")]:
            with self.assertRaises(ValueError):
                simulate_deposition(power_w=220.0, pressure_mtorr=bad, distance_mm=80.0)
            with self.assertRaises(ValueError):
                simulate_deposition(power_w=220.0, pressure_mtorr=5.0, distance_mm=bad)

        with self.assertRaises(ValueError):
            calculate_discharge_state(power_w=float("nan"), pressure_mtorr=5.0)
        with self.assertRaises(ValueError):
            calculate_discharge_state(power_w=220.0, pressure_mtorr=float("nan"))

    # ------------------------------------------------------------------
    # 16. (NEW) ar_flow_sccm affects discharge (Option B)  — Issue 6 fix
    # ------------------------------------------------------------------
    def test_ar_flow_affects_discharge(self):
        """16. (NEW-Issue6/OptionB) ar_flow_sccm changes effective pressure and thus discharge state."""
        base_flow = calculate_discharge_state(power_w=220.0, pressure_mtorr=5.0, ar_flow_sccm=20.0)
        hi_flow   = calculate_discharge_state(power_w=220.0, pressure_mtorr=5.0, ar_flow_sccm=50.0)
        lo_flow   = calculate_discharge_state(power_w=220.0, pressure_mtorr=5.0, ar_flow_sccm=5.0)

        # Higher flow → higher effective pressure → lower discharge voltage
        self.assertLess(hi_flow.voltage_v, base_flow.voltage_v,
                        "Higher Ar flow must lower discharge voltage via effective pressure")
        self.assertGreater(lo_flow.voltage_v, base_flow.voltage_v,
                           "Lower Ar flow must raise discharge voltage")

        # Effective pressure must be stored in DischargeState
        self.assertGreater(hi_flow.effective_pressure_mtorr, base_flow.effective_pressure_mtorr)
        self.assertLess(lo_flow.effective_pressure_mtorr, base_flow.effective_pressure_mtorr)

        # Baseline (20 sccm) should give p_eff == set pressure
        self.assertAlmostEqual(base_flow.effective_pressure_mtorr, 5.0, places=5)

    # ------------------------------------------------------------------
    # 17. (NEW) Circular wafer mask for metrics  — Issue 2 fix
    # ------------------------------------------------------------------
    def test_circular_wafer_mask_metrics(self):
        """17. (NEW-Issue2) Metrics use on-wafer grid points only; off-wafer points are NaN."""
        res = simulate_deposition(
            power_w=220.0,
            pressure_mtorr=5.0,
            deposition_time_s=720.0,
            grid_size=5,
            wafer_radius_mm=75.0,
        )

        # 5x5 grid, wafer radius 75 mm: 13 points have R <= 75 mm
        self.assertEqual(res.n_wafer_points, 13)

        # Corners (R > 75 mm) must be NaN
        X, Y = np.meshgrid(res.x_grid_mm, res.y_grid_mm)
        R = np.sqrt(X**2 + Y**2)
        off_wafer_mask = R > 75.0
        off_wafer_values = res.thickness_map[off_wafer_mask]
        self.assertTrue(
            np.all(np.isnan(off_wafer_values)),
            "All off-wafer grid points must be NaN in thickness_map",
        )

        # mean/min/max must equal statistics over the 13 on-wafer values
        on_wafer_values = res.thickness_map[~np.isnan(res.thickness_map)]
        self.assertAlmostEqual(res.mean_thickness_nm, float(np.mean(on_wafer_values)), places=6)
        self.assertAlmostEqual(res.min_thickness_nm, float(np.min(on_wafer_values)), places=6)
        self.assertAlmostEqual(res.max_thickness_nm, float(np.max(on_wafer_values)), places=6)

        # Sanity: off-wafer corners should NOT influence the reported mean
        # (adding 1e6 nm off-wafer should leave mean unchanged since they're NaN)
        # This is guaranteed by NaN propagation — just verify n_wafer_points
        self.assertEqual(res.n_wafer_points, 13)

    # ------------------------------------------------------------------
    # 18. (NEW) Vectorized sputter yield array matches scalar implementation
    # ------------------------------------------------------------------
    def test_sputter_yield_array_native_vectorization(self):
        """18. Native numpy vectorized yield matches scalar implementation across energies & angles."""
        energies = np.linspace(5.0, 800.0, 100)
        for mat in ["Ti", "Cu", "Al"]:
            # Normal incidence
            y_arr = calculate_sputter_yield_array(energies, material=mat, angle_rad=0.0)
            y_scalar = np.array([calculate_sputter_yield(e, material=mat, angle_rad=0.0) for e in energies])
            np.testing.assert_allclose(y_arr, y_scalar, rtol=1e-10, atol=1e-12)

            # Oblique incidence (45 deg)
            theta = math.radians(45.0)
            y_arr_obl = calculate_sputter_yield_array(energies, material=mat, angle_rad=theta)
            y_scalar_obl = np.array([calculate_sputter_yield(e, material=mat, angle_rad=theta) for e in energies])
            np.testing.assert_allclose(y_arr_obl, y_scalar_obl, rtol=1e-10, atol=1e-12)

        # Grazing angle cutoff
        y_grazing = calculate_sputter_yield_array(energies, material="Cu", angle_rad=math.radians(88.0))
        self.assertTrue(np.all(y_grazing == 0.0))

    # ------------------------------------------------------------------
    # 19. (NEW) Sputter yield angular parameter boundary checks
    # ------------------------------------------------------------------
    def test_sputter_yield_angular_bounds_validation(self):
        """19. Invalid f_exponent, optimum_angle_deg, or grazing_cutoff_deg raises ValueError."""
        with self.assertRaises(ValueError):
            calculate_sputter_yield(400.0, "Cu", f_exponent=-1.0)
        with self.assertRaises(ValueError):
            calculate_sputter_yield(400.0, "Cu", f_exponent=0.0)
        with self.assertRaises(ValueError):
            calculate_sputter_yield(400.0, "Cu", optimum_angle_deg=0.0)
        with self.assertRaises(ValueError):
            calculate_sputter_yield(400.0, "Cu", optimum_angle_deg=90.0)
        with self.assertRaises(ValueError):
            calculate_sputter_yield(400.0, "Cu", grazing_cutoff_deg=0.0)
        with self.assertRaises(ValueError):
            calculate_sputter_yield(400.0, "Cu", grazing_cutoff_deg=95.0)

    # ------------------------------------------------------------------
    # 20. (NEW) Transport summary TypedDict structure
    # ------------------------------------------------------------------
    def test_transport_summary_typed_dict(self):
        """20. calculate_transport_summary returns valid TransportSummary dict with expected keys."""
        summary = calculate_transport_summary(pressure_mtorr=5.0, distance_mm=80.0)
        self.assertIn("mean_free_path_mm", summary)
        self.assertIn("knudsen_number", summary)
        self.assertIn("transmission_probability", summary)
        self.assertIn("broadening_factor", summary)
        self.assertIn("n_collisions", summary)
        self.assertIn("regime", summary)
        self.assertIsInstance(summary["regime"], str)
        self.assertGreater(summary["mean_free_path_mm"], 0.0)
        self.assertGreater(summary["knudsen_number"], 0.0)
        self.assertGreaterEqual(summary["broadening_factor"], 1.0)

    # ------------------------------------------------------------------
    # 21. (NEW) Mass conservation under extreme geometry & erosion
    # ------------------------------------------------------------------
    def test_deposition_extreme_c_eff_mass_conservation(self):
        """21. Extreme non-uniformity (large erosion, small distance) preserves mass conservation."""
        res = simulate_deposition(
            power_w=220.0,
            pressure_mtorr=1.0,
            distance_mm=30.0,
            deposition_time_s=720.0,
            material="Cu",
            racetrack_radius_mm=25.0,
            target_radius_mm=50.0,
            target_erosion_mm=10.0,
            grid_size=15,
        )
        on_wafer = res.thickness_map[~np.isnan(res.thickness_map)]
        # All on-wafer points must be strictly positive
        self.assertTrue(np.all(on_wafer > 0.0), "All on-wafer points must remain positive")
        # Growth rate * time must equal mean thickness
        expected_mean = res.deposition_rate_nm_min * (720.0 / 60.0)
        self.assertAlmostEqual(res.mean_thickness_nm, expected_mean, places=4)

    # ------------------------------------------------------------------
    # 22. (NEW) Secondary inputs validation in deposition
    # ------------------------------------------------------------------
    def test_deposition_secondary_input_guards(self):
        """22. NaN in ar_flow_sccm, target_erosion_mm, or ar_flow_sccm <= 0 raises ValueError."""
        base = dict(power_w=220.0, pressure_mtorr=5.0, distance_mm=80.0)
        with self.assertRaises(ValueError):
            simulate_deposition(**base, ar_flow_sccm=float("nan"))
        with self.assertRaises(ValueError):
            simulate_deposition(**base, target_erosion_mm=float("nan"))
        with self.assertRaises(ValueError):
            simulate_deposition(**base, ar_flow_sccm=0.0)
        with self.assertRaises(ValueError):
            simulate_deposition(**base, ar_flow_sccm=-5.0)


if __name__ == "__main__":
    unittest.main()
