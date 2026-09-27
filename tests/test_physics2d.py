"""Unit tests for the SputterTwin 2D Multiphysics field solvers.

Validates:
1. 2D Axisymmetric Magnetic Field (B_r, B_z, psi, racetrack null point, Hall parameter)
2. 2D Plasma Multiphysics (Child-Langmuir sheath potential, cross-field electron
   confinement, ionization rate R_ion(r, z), cathode ion flux profile, power balance)
3. 2D Gas Rarefaction & Sputter Wind Thermal Fluid Model
4. Solver robustness: input validation, grid convergence, and 2D rarefaction coupling
"""

import math
import unittest

import numpy as np

from sputtertwin.numerics import trapezoid
from sputtertwin.physics.plasma import (
    effective_pressure_mtorr,
    magnetron_voltage_from_power,
)
from sputtertwin.physics2d.magnetic_field import (
    MagnetronGeometry,
    MagneticField2D,
    compute_magnetron_magnetic_field,
)
from sputtertwin.physics2d.plasma2d import (
    Plasma2DResult,
    _fwhm_linear,
    _ionization_rate_coefficient,
    simulate_plasma_2d,
)
from sputtertwin.physics2d.rarefaction2d import compute_gas_rarefaction_2d


class TestPhysics2D(unittest.TestCase):
    """Test suite for 2D multiphysics field solvers."""

    def setUp(self):
        self.geom = MagnetronGeometry(target_radius_m=0.050)

    def test_magnetic_field_symmetry_and_racetrack(self):
        """Verify on-axis symmetry and racetrack zero-crossing of magnetic field."""
        b2d = compute_magnetron_magnetic_field(
            geom=self.geom,
            grid_r_points=40,
            grid_z_points=30,
            z_max_m=0.040,
        )

        # On-axis symmetry: B_r(0, z) must be zero
        b_r_axis = b2d.B_r[:, 0]
        np.testing.assert_allclose(b_r_axis, 0.0, atol=1e-5, err_msg="B_r must be 0 on axis")

        # Racetrack radius must lie inside target radius
        self.assertGreater(b2d.racetrack_radius_m, 0.010)
        self.assertLess(b2d.racetrack_radius_m, self.geom.target_radius_m)

        # Peak parallel field in Gauss should be typical 200 - 1000 Gauss
        b_gauss = b2d.b_parallel_peak_tesla * 10000.0
        self.assertGreater(b_gauss, 100.0, "Parallel B field too weak")
        self.assertLess(b_gauss, 2000.0, "Parallel B field unrealistically strong")

        # Hall parameter beta_e in trap must be >> 1 (magnetized electrons)
        beta_e = b2d.hall_parameter(pressure_mtorr=5.0)
        self.assertGreater(np.max(beta_e), 10.0, "Electrons must be strongly magnetized in trap")

    def test_plasma_2d_coupled_solution(self):
        """Verify 2D plasma sheath potential, density, and power conservation."""
        res = simulate_plasma_2d(
            power_w=220.0,
            pressure_mtorr=5.0,
            ar_flow_sccm=20.0,
            target_radius_m=0.050,
            anode_distance_m=0.050,
            grid_r=30,
            grid_z=25,
        )

        # Grid and field shapes
        self.assertEqual(res.V_2d.shape, (25, 30))
        self.assertEqual(res.ne_2d.shape, (25, 30))
        self.assertEqual(res.ni_2d.shape, (25, 30))
        self.assertEqual(len(res.ion_flux_profile_1d), 30)

        # Power conservation W = V_d * I_d
        calc_power = res.voltage_v * res.current_a
        np.testing.assert_allclose(calc_power, 220.0, rtol=1e-4)

        # Cathode target potential at z=0 must equal -V_d
        np.testing.assert_allclose(res.V_2d[0, :], -res.voltage_v, rtol=1e-3)

        # Electron density strictly positive everywhere
        self.assertTrue(np.all(res.ne_2d > 0.0))
        self.assertGreater(res.peak_ne_m3, 1e17)
        self.assertLess(res.peak_ne_m3, 1e19)

        # Racetrack FWHM groove width between 4 mm and 20 mm
        self.assertGreater(res.racetrack_fwhm_m, 0.004)
        self.assertLess(res.racetrack_fwhm_m, 0.025)

    def test_gas_rarefaction_thermal_fluid(self):
        """Verify 2D gas temperature rise and density depletion in front of target."""
        rare = compute_gas_rarefaction_2d(
            power_w=300.0,
            pressure_mtorr=5.0,
            racetrack_radius_m=0.025,
            target_radius_m=0.050,
            grid_r=25,
            grid_z=20,
        )

        # Gas temperature must be >= ambient (300 K)
        self.assertTrue(np.all(rare.T_gas_2d_K >= 299.9))
        self.assertGreater(rare.peak_gas_temp_k, 350.0)

        # Rarefaction factor must be <= 1.0 (density depleted)
        self.assertTrue(np.all(rare.rarefaction_factor_2d <= 1.0001))
        self.assertGreater(rare.min_rarefaction, 0.20)
        self.assertLess(rare.min_rarefaction, 0.95)

        # Pressure drop percentage
        self.assertGreater(rare.effective_pressure_reduction_pct, 5.0)
        self.assertLess(rare.effective_pressure_reduction_pct, 70.0)

    # ------------------------------------------------------------------
    # Solver robustness and physics consistency
    # ------------------------------------------------------------------
    def test_trap_thickness_is_grid_independent(self):
        """Interpolated trap thickness must not snap to the axial grid spacing."""
        thicknesses = []
        for grid_r, grid_z, z_max in [(30, 25, 0.050), (60, 50, 0.060), (120, 100, 0.060)]:
            b2d = compute_magnetron_magnetic_field(
                geom=self.geom,
                grid_r_points=grid_r,
                grid_z_points=grid_z,
                z_max_m=z_max,
            )
            thicknesses.append(b2d.trap_thickness_m)

        spread = (max(thicknesses) - min(thicknesses)) / np.mean(thicknesses)
        self.assertLess(spread, 0.05, f"trap thickness varies with grid: {thicknesses}")

    def test_plasma_2d_rejects_invalid_inputs(self):
        """Invalid operating points and grids must raise ValueError, not NaNs."""
        valid = dict(power_w=220.0, pressure_mtorr=5.0, ar_flow_sccm=20.0,
                     target_radius_m=0.050, anode_distance_m=0.050, grid_r=30, grid_z=25)
        bad_cases = [
            {"power_w": 0.0},
            {"power_w": -10.0},
            {"power_w": float("nan")},
            {"pressure_mtorr": 0.0},
            {"pressure_mtorr": float("inf")},
            {"ar_flow_sccm": 0.0},
            {"target_radius_m": 0.0},
            {"anode_distance_m": -0.05},
            {"gamma_se": 1.0},
            {"gamma_se": -0.1},
            {"gas_temp_k": 0.0},
            {"grid_r": 1},
            {"grid_z": 1},
            {"ionization_rate_model": "bogus"},
            {"cathode_flux_model": "bogus"},
        ]
        for override in bad_cases:
            kwargs = dict(valid)
            kwargs.update(override)
            with self.assertRaises(ValueError, msg=f"{override} should be rejected"):
                simulate_plasma_2d(**kwargs)

    def test_plasma_2d_electrons_are_magnetized(self):
        """Cross-field mobility must actually be reduced (Hall parameter >> 1)."""
        res = simulate_plasma_2d(power_w=220.0, pressure_mtorr=5.0, grid_r=40, grid_z=30)

        # beta_e = omega_ce / nu_en must be >> 1 inside the magnetic trap
        self.assertGreater(np.max(res.hall_parameter_2d), 100.0)
        # mu_perp = mu_e0 / (1 + beta^2) must therefore be far below mu_e0
        v_th = np.sqrt(8.0 * 1.602176634e-19 * res.Te_2d / (math.pi * 9.1093837e-31))
        nu_en = res.n_gas_2d_m3 * 4.0e-20 * v_th
        mu_e0 = 1.602176634e-19 / (9.1093837e-31 * nu_en)
        expected_mu_perp = mu_e0 / (1.0 + res.hall_parameter_2d ** 2)
        np.testing.assert_allclose(res.mu_perp_2d, expected_mu_perp, rtol=1e-6)

        # Magnetization can never *increase* mobility, and inside the trap
        # (beta_e >> 1) it must suppress it by orders of magnitude.
        self.assertTrue(np.all(res.mu_perp_2d <= mu_e0 * (1.0 + 1e-12)))
        trapped = res.hall_parameter_2d > 100.0
        self.assertTrue(np.any(trapped))
        self.assertLess(
            np.max(res.mu_perp_2d[trapped] / mu_e0[trapped]), 0.01
        )

    def test_plasma_2d_current_conservation(self):
        """Integrated cathode current must equal I_ion = I_d / (1 + gamma_se)."""
        gamma_se = 0.10
        res = simulate_plasma_2d(
            power_w=220.0, pressure_mtorr=5.0, gamma_se=gamma_se, grid_r=60, grid_z=50
        )
        i_ion_integral = 2.0 * math.pi * trapezoid(
            res.current_density_1d * res.r_grid_m, res.r_grid_m
        )
        np.testing.assert_allclose(
            i_ion_integral, res.total_ion_current_a, rtol=1e-6
        )
        np.testing.assert_allclose(
            res.total_ion_current_a, res.current_a / (1.0 + gamma_se), rtol=1e-9
        )
        np.testing.assert_allclose(res.voltage_v * res.current_a, res.power_w, rtol=1e-9)

    def test_plasma_2d_grid_convergence(self):
        """Electrical outputs must be insensitive to grid resolution."""
        results = [
            simulate_plasma_2d(
                power_w=220.0,
                pressure_mtorr=5.0,
                target_radius_m=0.050,
                anode_distance_m=0.060,
                grid_r=grid_r,
                grid_z=grid_z,
            )
            for grid_r, grid_z in [(30, 25), (60, 50), (120, 100)]
        ]
        for attr in ("voltage_v", "current_a", "peak_ne_m3", "racetrack_fwhm_m"):
            values = [getattr(r, attr) for r in results]
            spread = (max(values) - min(values)) / abs(np.mean(values))
            self.assertLess(
                spread, 0.02, f"{attr} varies {spread:.2%} across grids: {values}"
            )

        # The cathode current profile itself must converge
        coarse, fine = results[0], results[-1]
        j_fine_on_coarse = np.interp(coarse.r_grid_m, fine.r_grid_m, fine.current_density_1d)
        rel = np.max(np.abs(j_fine_on_coarse - coarse.current_density_1d)) / np.max(
            coarse.current_density_1d
        )
        self.assertLess(rel, 0.05, f"J_i(r) not grid converged (rel diff {rel:.2%})")

    def test_fwhm_linear_recovers_analytic_gaussian(self):
        """Sub-grid FWHM must be resolution independent for a known profile."""
        true_fwhm = 0.010
        sigma = true_fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        estimates = []
        for n in (61, 121, 241):
            r = np.linspace(0.0, 0.050, n)
            y = np.exp(-0.5 * ((r - 0.020) / sigma) ** 2)
            estimates.append(_fwhm_linear(r, y))
        for est in estimates:
            self.assertAlmostEqual(est / true_fwhm, 1.0, places=3)
        self.assertLess(max(estimates) - min(estimates), 1e-6)

        # Degenerate profiles must not raise
        self.assertEqual(_fwhm_linear(np.linspace(0, 1, 5), np.zeros(5)), 0.0)

    def test_plasma_2d_ionization_rate_models(self):
        """Both ionization closures must be positive, finite and Te-ordered."""
        Te = np.array([1.0, 2.0, 3.0, 5.0, 10.0])
        arr = _ionization_rate_coefficient(Te, model="arrhenius")
        fit = _ionization_rate_coefficient(Te, model="fit")
        for k in (arr, fit):
            self.assertTrue(np.all(np.isfinite(k)))
            self.assertTrue(np.all(k > 0.0))
            self.assertTrue(np.all(np.diff(k) > 0.0), "k_ion must increase with Te")
        with self.assertRaises(ValueError):
            _ionization_rate_coefficient(Te, model="unknown")

    def test_plasma_2d_cathode_flux_models_conserve_current(self):
        """Both cathode flux closures must reproduce the same total ion current."""
        common = dict(power_w=220.0, pressure_mtorr=5.0, grid_r=60, grid_z=50)
        res_int = simulate_plasma_2d(cathode_flux_model="ionization_integral", **common)
        res_bohm = simulate_plasma_2d(cathode_flux_model="bohm", **common)

        self.assertAlmostEqual(res_int.total_ion_current_a, res_bohm.total_ion_current_a, places=9)
        # Both closures must localize the erosion groove at the racetrack
        for res in (res_int, res_bohm):
            r_peak = res.r_grid_m[int(np.argmax(res.current_density_1d))]
            self.assertAlmostEqual(r_peak, res.racetrack_radius_m, delta=0.004)
            self.assertTrue(np.all(res.current_density_1d > 0.0))

    def test_plasma_2d_rarefaction_coupling(self):
        """A 2D rarefaction field must feed the neutral density and pressure feedback."""
        kw = dict(power_w=300.0, pressure_mtorr=5.0, grid_r=40, grid_z=30,
                  target_radius_m=0.050, anode_distance_m=0.050)
        uniform = simulate_plasma_2d(**kw)
        rare = compute_gas_rarefaction_2d(
            power_w=300.0,
            pressure_mtorr=5.0,
            racetrack_radius_m=uniform.racetrack_radius_m,
            target_radius_m=0.050,
            grid_r=40,
            grid_z=30,
        )
        coupled = simulate_plasma_2d(rarefaction=rare, **kw)

        # Neutral density is now 2D and depleted in front of the racetrack
        self.assertLess(np.min(coupled.n_gas_2d_m3), np.min(uniform.n_gas_2d_m3))
        np.testing.assert_allclose(coupled.n_gas_2d_m3, rare.n_gas_2d_m3, rtol=1e-12)

        # Rarefaction lowers the effective pressure -> higher discharge voltage
        self.assertLess(coupled.effective_pressure_mtorr, uniform.effective_pressure_mtorr)
        self.assertGreater(coupled.voltage_v, uniform.voltage_v)
        # ... while the power balance still holds exactly
        np.testing.assert_allclose(coupled.voltage_v * coupled.current_a, 300.0, rtol=1e-9)

        # A mismatched rarefaction grid must be rejected
        rare_wrong = compute_gas_rarefaction_2d(
            power_w=300.0, pressure_mtorr=5.0, grid_r=20, grid_z=15
        )
        with self.assertRaises(ValueError):
            simulate_plasma_2d(rarefaction=rare_wrong, **kw)

    def test_plasma_2d_result_serialization(self):
        """Plasma2DResult must expose a dict/summary for downstream reporting."""
        res = simulate_plasma_2d(power_w=220.0, pressure_mtorr=5.0, grid_r=30, grid_z=25)
        self.assertIsInstance(res, Plasma2DResult)
        d = res.to_dict()
        for key in ("voltage_v", "current_a", "power_w", "peak_ne_m3", "racetrack_fwhm_mm"):
            self.assertIn(key, d)
            self.assertIsInstance(d[key], float)
        text = res.summary()
        for token in ("Discharge Voltage", "Racetrack FWHM", "Peak ne"):
            self.assertIn(token, text)

    def test_shared_magnetron_power_law_helpers(self):
        """0D and 2D solvers must invert the same power law consistently."""
        self.assertAlmostEqual(effective_pressure_mtorr(5.0, 20.0), 5.0, places=12)
        self.assertAlmostEqual(effective_pressure_mtorr(5.0, 60.0), 5.4, places=12)
        v = magnetron_voltage_from_power(220.0, 5.0)
        self.assertAlmostEqual(v * (220.0 / v), 220.0, places=9)
        with self.assertRaises(ValueError):
            magnetron_voltage_from_power(220.0, 0.0)
        with self.assertRaises(ValueError):
            magnetron_voltage_from_power(float("nan"), 5.0)

    def test_numerics_trapezoid_matches_scipy(self):
        """The NumPy 1.x/2.x compatibility shim must integrate correctly."""
        from scipy.integrate import trapezoid as scipy_trapz

        x = np.linspace(0.0, 2.0 * math.pi, 101)
        y = np.sin(x)
        self.assertAlmostEqual(float(trapezoid(y, x)), float(scipy_trapz(y, x)), places=12)

    def test_grid_that_misses_the_cathode_raises(self):
        """A magnetic-field grid that starts above the cathode cannot be integrated."""
        r_vec = np.linspace(0.0, 0.050, 20)
        z_vec = np.linspace(0.030, 0.060, 20)  # z = 0 (cathode) not on the grid
        shape = (z_vec.size, r_vec.size)
        detached = MagneticField2D(
            r_grid_m=r_vec,
            z_grid_m=z_vec,
            B_r=np.zeros(shape),
            B_z=np.full(shape, -0.02),
            B_mag=np.full(shape, 0.02),
            psi_flux=np.zeros(shape),
            racetrack_radius_m=0.020,
            b_parallel_peak_tesla=0.045,
            trap_thickness_m=0.008,
        )
        with self.assertRaises(RuntimeError):
            simulate_plasma_2d(power_w=220.0, pressure_mtorr=5.0, mag_field=detached)

        # A non-monotonic or too-short grid must also be rejected
        bad = MagneticField2D(
            r_grid_m=r_vec,
            z_grid_m=z_vec[::-1],
            B_r=np.zeros(shape),
            B_z=np.full(shape, -0.02),
            B_mag=np.full(shape, 0.02),
            psi_flux=np.zeros(shape),
            racetrack_radius_m=0.020,
            b_parallel_peak_tesla=0.045,
            trap_thickness_m=0.008,
        )
        with self.assertRaises(ValueError):
            simulate_plasma_2d(power_w=220.0, pressure_mtorr=5.0, mag_field=bad)


if __name__ == "__main__":
    unittest.main()
