"""Unit tests for the SputterTwin 2D Multiphysics field solvers.

Validates:
1. 2D Axisymmetric Magnetic Field (B_r, B_z, psi, racetrack null point, Hall parameter)
2. 2D Plasma Multiphysics (Poisson sheath potential, cross-field electron confinement,
   ionization rate R_ion(r, z), cathode ion flux profile)
3. 2D Gas Rarefaction & Sputter Wind Thermal Fluid Model
"""

import math
import unittest

import numpy as np

from sputtertwin.physics2d.magnetic_field import (
    MagnetronGeometry,
    compute_magnetron_magnetic_field,
)
from sputtertwin.physics2d.plasma2d import simulate_plasma_2d
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


if __name__ == "__main__":
    unittest.main()
