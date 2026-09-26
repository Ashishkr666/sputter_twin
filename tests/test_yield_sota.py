"""Comprehensive unit test suite for SOTA sputter yield, Thomson spectrum, and Monte Carlo sampling.

Validates:
1. Thomson nascent energy distribution:
   - Peak occurs at E_peak = Us / 2 (e.g. ~1.75 eV for Cu with Us=3.49 eV).
   - High-energy kinematic cutoff at gamma * E_ion - Us.
   - Zero yield for sub-threshold or negative energies.
   - Exact analytical probability normalization (integral = 1.0).
2. Monte Carlo sampling of nascent sputtered velocities:
   - Vectorized sampling of energy E and polar emission angle alpha.
   - Over/under-cosine distribution f(alpha) ~ cos^n(alpha) * sin(alpha).
   - Reproducibility with random seeds.
   - Physical bounds and particle speeds in m/s.
3. State-of-the-art (SOTA) sputter yield with surface micro-roughness damping:
   - Equivalence to Yamamura-Tawara when roughness_factor = 0.0.
   - Normal incidence invariance (roughness does not damp normal emission).
   - Softening/damping of unphysical sharp peaks at oblique angles.
   - Monotonic damping with increasing roughness factor.
   - Preservation of energy threshold and grazing angle cutoff.
   - Vectorized array support.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from sputtertwin.physics.sputter_yield import (
    MATERIALS,
    EjectedParticles,
    TargetMaterial,
    calculate_kinematic_factor,
    calculate_sputter_yield,
    calculate_sputter_yield_array,
    calculate_sputter_yield_sota,
    calculate_thomson_energy_spectrum,
    sample_ejected_energy_and_angle,
)


class TestKinematicFactor(unittest.TestCase):
    """Tests for the kinematic energy transfer factor gamma."""

    def test_equal_masses(self):
        """Gamma is 1.0 when projectile and target have identical mass."""
        gamma = calculate_kinematic_factor(39.948, 39.948)
        self.assertAlmostEqual(gamma, 1.0, places=9)

    def test_standard_materials(self):
        """Gamma matches analytical 4*m1*m2 / (m1+m2)^2 for Ar+ bombardment."""
        m_ar = 39.948
        for name, mat in MATERIALS.items():
            expected = 4.0 * m_ar * mat.atomic_mass / ((m_ar + mat.atomic_mass) ** 2)
            actual = calculate_kinematic_factor(mat.atomic_mass, m_ar)
            self.assertAlmostEqual(actual, expected, places=7)
            self.assertGreater(actual, 0.0)
            self.assertLessEqual(actual, 1.0)

    def test_invalid_masses(self):
        """Non-positive atomic masses raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_kinematic_factor(-10.0, 39.948)
        with self.assertRaises(ValueError):
            calculate_kinematic_factor(63.55, 0.0)


class TestThomsonEnergySpectrum(unittest.TestCase):
    """Tests for Thomson nascent energy spectrum f(E)."""

    def test_thomson_peak_copper(self):
        """Verify Thomson peak occurs at E_peak = Us / 2 (~1.745 eV for Cu with Us=3.49 eV)."""
        mat = MATERIALS["Cu"]
        us = mat.sublimation_energy
        expected_peak = us / 2.0  # 1.745 eV

        # Evaluate on fine energy grid
        energies = np.linspace(0.01, 10.0, 100000)

        # For 500 eV ion energy, peak is very close to Us/2 (~1.70 eV, within 0.06 eV)
        f_500 = calculate_thomson_energy_spectrum(energies, "Cu", 500.0)
        peak_e_500 = energies[np.argmax(f_500)]
        self.assertAlmostEqual(peak_e_500, expected_peak, delta=0.06)

        # In asymptotic limit of high ion energy (e.g. 50 keV), peak approaches Us/2 even closer
        f_high = calculate_thomson_energy_spectrum(energies, "Cu", 50000.0)
        peak_e_high = energies[np.argmax(f_high)]
        self.assertAlmostEqual(peak_e_high, expected_peak, delta=0.01)

    def test_thomson_peak_titanium_and_aluminum(self):
        """Verify Thomson peak scales as Us / 2 across different materials."""
        energies = np.linspace(0.01, 10.0, 100000)

        # Ti (Us = 4.89 eV -> peak ~ 2.445 eV)
        us_ti = MATERIALS["Ti"].sublimation_energy
        f_ti = calculate_thomson_energy_spectrum(energies, "Ti", 50000.0)
        peak_ti = energies[np.argmax(f_ti)]
        self.assertAlmostEqual(peak_ti, us_ti / 2.0, delta=0.01)

        # Al (Us = 3.39 eV -> peak ~ 1.695 eV)
        us_al = MATERIALS["Al"].sublimation_energy
        f_al = calculate_thomson_energy_spectrum(energies, "Al", 50000.0)
        peak_al = energies[np.argmax(f_al)]
        self.assertAlmostEqual(peak_al, us_al / 2.0, delta=0.01)

    def test_thomson_scalar_and_vector_inputs(self):
        """Verify calculate_thomson_energy_spectrum supports both scalar and array inputs."""
        # Scalar
        val_scalar = calculate_thomson_energy_spectrum(1.75, "Cu", 500.0)
        self.assertIsInstance(val_scalar, float)
        self.assertGreater(val_scalar, 0.0)

        # 1-D Array
        e_arr = np.array([0.5, 1.745, 3.0, 10.0])
        val_arr = calculate_thomson_energy_spectrum(e_arr, "Cu", 500.0)
        self.assertIsInstance(val_arr, np.ndarray)
        self.assertEqual(val_arr.shape, e_arr.shape)
        self.assertTrue(np.all(val_arr >= 0.0))

    def test_thomson_kinematic_cutoffs(self):
        """Verify spectrum is strictly zero outside physical energy domain (0, gamma*E_ion - Us)."""
        mat = MATERIALS["Cu"]
        gamma = calculate_kinematic_factor(mat.atomic_mass)
        e_ion = 300.0
        e_max = gamma * e_ion - mat.sublimation_energy

        # At or below 0 eV
        self.assertEqual(calculate_thomson_energy_spectrum(0.0, "Cu", e_ion), 0.0)
        self.assertEqual(calculate_thomson_energy_spectrum(-5.0, "Cu", e_ion), 0.0)

        # At or above maximum kinematic limit
        self.assertEqual(calculate_thomson_energy_spectrum(e_max, "Cu", e_ion), 0.0)
        self.assertEqual(calculate_thomson_energy_spectrum(e_max + 10.0, "Cu", e_ion), 0.0)

        # Sub-threshold ion energy (gamma * E_ion <= Us)
        sub_e_ion = mat.sublimation_energy / gamma * 0.5
        self.assertEqual(calculate_thomson_energy_spectrum(1.0, "Cu", sub_e_ion), 0.0)

    def test_thomson_probability_normalization(self):
        """Verify that with normalize=True, Thomson spectrum integrates to 1.0."""
        for e_ion in [200.0, 500.0, 1000.0]:
            mat = MATERIALS["Cu"]
            gamma = calculate_kinematic_factor(mat.atomic_mass)
            e_max = gamma * e_ion - mat.sublimation_energy

            # Dense integration grid from 0 to E_max
            e_grid = np.linspace(1e-4, e_max - 1e-4, 100000)
            pdf = calculate_thomson_energy_spectrum(e_grid, "Cu", e_ion, normalize=True)

            integral = np.trapezoid(pdf, e_grid) if hasattr(np, "trapezoid") else np.trapz(pdf, e_grid)
            self.assertAlmostEqual(integral, 1.0, delta=1e-3)

    def test_thomson_invalid_inputs(self):
        """Verify invalid energy and material inputs raise appropriate exceptions."""
        with self.assertRaises(ValueError):
            calculate_thomson_energy_spectrum(1.0, "Cu", -100.0)
        with self.assertRaises(ValueError):
            calculate_thomson_energy_spectrum(1.0, "Cu", float("nan"))
        with self.assertRaises(KeyError):
            calculate_thomson_energy_spectrum(1.0, "UnknownElement", 500.0)


class TestEjectedSampling(unittest.TestCase):
    """Tests for Monte Carlo sampling of nascent ejected particle energies and angles."""

    def test_sampling_structure_and_types(self):
        """Verify return type, unpacking, and array shapes."""
        particles = sample_ejected_energy_and_angle(500, "Cu", 500.0, seed=42)
        self.assertIsInstance(particles, EjectedParticles)
        self.assertIsInstance(particles, tuple)
        self.assertEqual(len(particles), 2)
        self.assertEqual(len(particles.energies), 500)
        self.assertEqual(len(particles.angles), 500)

        # Unpacking support
        e, a = sample_ejected_energy_and_angle(200, "Ti", 400.0, seed=123)
        self.assertEqual(len(e), 200)
        self.assertEqual(len(a), 200)

    def test_sampling_reproducibility(self):
        """Verify identical seed yields identical Monte Carlo samples."""
        p1 = sample_ejected_energy_and_angle(1000, "Cu", 600.0, seed=42)
        p2 = sample_ejected_energy_and_angle(1000, "Cu", 600.0, seed=42)
        p3 = sample_ejected_energy_and_angle(1000, "Cu", 600.0, seed=99)

        np.testing.assert_array_equal(p1.energies, p2.energies)
        np.testing.assert_array_equal(p1.angles, p2.angles)
        self.assertFalse(np.array_equal(p1.energies, p3.energies))

    def test_sampling_physical_bounds(self):
        """Sampled energies and angles must lie strictly within physical kinematic bounds."""
        mat = MATERIALS["Cu"]
        gamma = calculate_kinematic_factor(mat.atomic_mass)
        e_ion = 500.0
        e_max = gamma * e_ion - mat.sublimation_energy

        particles = sample_ejected_energy_and_angle(5000, "Cu", e_ion, seed=42)
        self.assertTrue(np.all(particles.energies > 0.0))
        self.assertTrue(np.all(particles.energies < e_max))
        self.assertTrue(np.all(particles.angles >= 0.0))
        self.assertTrue(np.all(particles.angles <= math.pi / 2.0))

    def test_sampling_energy_distribution_shape(self):
        """Sampled energy distribution mode aligns near Us/2 and mean is physically plausible."""
        us = MATERIALS["Cu"].sublimation_energy
        particles = sample_ejected_energy_and_angle(50000, "Cu", 500.0, seed=42)

        # Histogram peak
        counts, bin_edges = np.histogram(particles.energies, bins=100, range=(0.0, 10.0))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        hist_peak = bin_centers[np.argmax(counts)]

        self.assertAlmostEqual(hist_peak, us / 2.0, delta=0.35)
        # Average sputtered energy is typically 10 - 25 eV
        self.assertGreater(particles.energies.mean(), 5.0)
        self.assertLess(particles.energies.mean(), 35.0)

    def test_sampling_angular_distribution(self):
        """Verify angular distribution follows over/under-cosine law."""
        # For n_cosine=1.0, mean angle ~ 45 deg = 0.785 rad
        # For higher n (e.g. 2.0), distribution shifts towards surface normal (smaller alpha)
        p_n1 = sample_ejected_energy_and_angle(30000, "Cu", 500.0, seed=42, n_cosine=1.0)
        p_n2 = sample_ejected_energy_and_angle(30000, "Cu", 500.0, seed=42, n_cosine=2.0)

        self.assertLess(p_n2.angles.mean(), p_n1.angles.mean())

        # Test alias 'n'
        p_alias = sample_ejected_energy_and_angle(100, "Cu", 500.0, seed=42, n=1.5)
        self.assertEqual(len(p_alias.angles), 100)

    def test_sampling_speeds_calculation(self):
        """Verify particle speeds calculated from energies are physically consistent."""
        particles = sample_ejected_energy_and_angle(1000, "Cu", 500.0, seed=42)
        speeds = particles.speeds("Cu")

        self.assertEqual(len(speeds), 1000)
        self.assertTrue(np.all(speeds > 0.0))
        # Typical nascent sputtered speeds for Cu are ~2,000 - 15,000 m/s
        self.assertGreater(speeds.mean(), 1000.0)
        self.assertLess(speeds.mean(), 20000.0)

    def test_sampling_edge_cases(self):
        """Verify zero samples, invalid parameters, and sub-threshold behavior."""
        p_zero = sample_ejected_energy_and_angle(0, "Cu", 500.0)
        self.assertEqual(len(p_zero.energies), 0)
        self.assertEqual(len(p_zero.angles), 0)

        with self.assertRaises(ValueError):
            sample_ejected_energy_and_angle(-5, "Cu", 500.0)
        with self.assertRaises(ValueError):
            sample_ejected_energy_and_angle(100, "Cu", -50.0)
        with self.assertRaises(ValueError):
            sample_ejected_energy_and_angle(100, "Cu", 500.0, n_cosine=-1.5)

        # Sub-threshold ion energy
        p_sub = sample_ejected_energy_and_angle(10, "Cu", 1.0)
        np.testing.assert_array_equal(p_sub.energies, np.zeros(10))
        np.testing.assert_array_equal(p_sub.angles, np.zeros(10))


class TestSputterYieldSOTA(unittest.TestCase):
    """Tests for SOTA sputter yield calculation with surface roughness damping."""

    def test_zero_roughness_matches_yamamura_exactly(self):
        """When roughness_factor is 0.0, SOTA yield is identical to standard Yamamura-Tawara."""
        for mat in ["Ti", "Cu", "Al"]:
            for energy in [150.0, 400.0, 800.0]:
                for angle_deg in [0.0, 30.0, 45.0, 65.0, 80.0]:
                    theta = math.radians(angle_deg)
                    y_std = calculate_sputter_yield(energy, mat, angle_rad=theta)
                    y_sota = calculate_sputter_yield_sota(
                        energy, mat, angle_rad=theta, roughness_factor=0.0
                    )
                    self.assertAlmostEqual(y_sota, y_std, places=12)

    def test_normal_incidence_invariance(self):
        """At normal incidence (theta = 0.0), roughness damping has no effect."""
        for mat in ["Ti", "Cu", "Al"]:
            for r in [0.0, 0.15, 0.30, 0.50]:
                y_norm_sota = calculate_sputter_yield_sota(
                    500.0, mat, angle_rad=0.0, roughness_factor=r
                )
                y_norm_std = calculate_sputter_yield(500.0, mat, angle_rad=0.0)
                self.assertAlmostEqual(y_norm_sota, y_norm_std, places=12)

    def test_oblique_peak_softening(self):
        """At oblique angles, micro-roughness softens the unphysical sharp peak."""
        theta_peak = math.radians(65.0)
        for mat in ["Ti", "Cu", "Al"]:
            y_smooth = calculate_sputter_yield(500.0, mat, angle_rad=theta_peak)
            y_sota = calculate_sputter_yield_sota(
                500.0, mat, angle_rad=theta_peak, roughness_factor=0.15
            )

            # SOTA with roughness must be strictly lower than smooth peak
            self.assertGreater(y_smooth, 0.0)
            self.assertLess(y_sota, y_smooth)

            # Check expected damping factor exp(-r * tan(theta))
            expected_damping = math.exp(-0.15 * math.tan(theta_peak))
            self.assertAlmostEqual(y_sota / y_smooth, expected_damping, places=9)

    def test_monotonic_roughness_damping(self):
        """Higher roughness factor leads to progressively stronger damping at oblique angles."""
        theta = math.radians(60.0)
        yields = [
            calculate_sputter_yield_sota(500.0, "Cu", angle_rad=theta, roughness_factor=r)
            for r in [0.0, 0.10, 0.20, 0.35]
        ]
        # Strictly decreasing
        self.assertTrue(yields[0] > yields[1] > yields[2] > yields[3])

    def test_subthreshold_and_grazing_cutoffs(self):
        """SOTA yield respects threshold energy and grazing angle cutoffs."""
        eth = MATERIALS["Cu"].threshold_energy
        # Sub-threshold
        self.assertEqual(
            calculate_sputter_yield_sota(eth * 0.5, "Cu", angle_rad=0.5, roughness_factor=0.15),
            0.0,
        )
        self.assertEqual(
            calculate_sputter_yield_sota(eth, "Cu", angle_rad=0.5, roughness_factor=0.15),
            0.0,
        )

        # Grazing angle cutoff (>= 85 degrees)
        self.assertEqual(
            calculate_sputter_yield_sota(
                500.0, "Cu", angle_rad=math.radians(85.0), roughness_factor=0.15
            ),
            0.0,
        )
        self.assertEqual(
            calculate_sputter_yield_sota(
                500.0, "Cu", angle_rad=math.radians(89.0), roughness_factor=0.15
            ),
            0.0,
        )

    def test_vectorized_energy_array(self):
        """Verify calculate_sputter_yield_sota works seamlessly with energy numpy arrays."""
        energies = np.array([10.0, 100.0, 300.0, 500.0, 800.0])
        theta = math.radians(45.0)

        # Array call
        y_array = calculate_sputter_yield_sota(
            energies, "Cu", angle_rad=theta, roughness_factor=0.15
        )
        self.assertEqual(y_array.shape, energies.shape)

        # Pointwise comparisons
        for i, e in enumerate(energies):
            y_scalar = calculate_sputter_yield_sota(
                float(e), "Cu", angle_rad=theta, roughness_factor=0.15
            )
            self.assertAlmostEqual(y_array[i], y_scalar, places=10)

    def test_vectorized_angle_array(self):
        """Verify calculate_sputter_yield_sota works with angle numpy arrays."""
        angles = np.radians(np.array([0.0, 30.0, 60.0, 75.0, 86.0]))
        y_angles = calculate_sputter_yield_sota(500.0, "Cu", angle_rad=angles, roughness_factor=0.15)
        self.assertEqual(y_angles.shape, angles.shape)
        self.assertEqual(y_angles[-1], 0.0)  # 86 deg is beyond cutoff

        for i, th in enumerate(angles):
            y_pt = calculate_sputter_yield_sota(500.0, "Cu", angle_rad=float(th), roughness_factor=0.15)
            self.assertAlmostEqual(y_angles[i], y_pt, places=10)

    def test_invalid_parameters(self):
        """Verify negative roughness factor and invalid material raise errors."""
        with self.assertRaises(ValueError):
            calculate_sputter_yield_sota(500.0, "Cu", roughness_factor=-0.1)
        with self.assertRaises(ValueError):
            calculate_sputter_yield_sota(-100.0, "Cu")
        with self.assertRaises(KeyError):
            calculate_sputter_yield_sota(500.0, "InvalidElement")


if __name__ == "__main__":
    unittest.main()
