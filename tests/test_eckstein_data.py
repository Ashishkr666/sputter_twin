"""Tests for Wolfgang Eckstein experimental sputtering yield benchmark dataset.

Verifies:
- Data integrity, schema, and reference validity.
- Energy and angle range validity (50 - 1000 eV, 0 - 80 degrees).
- Physical sanity checks (positive yields, threshold behavior, angular maximum).
- get_eckstein_data helper function return types, filtering, and error handling.
- Consistency with SputterTwin Yamamura-Tawara model calculations.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pytest

from sputtertwin.data import (
    ECKSTEIN_REFERENCES,
    ECKSTEIN_YIELD_DATA,
    SputterYieldPoint,
    get_eckstein_data,
)
from sputtertwin.physics.sputter_yield import (
    MATERIALS,
    TargetMaterial,
    calculate_sputter_yield,
)


class TestEcksteinDataIntegrity(unittest.TestCase):
    """Test data integrity, schema consistency, and bibliographic references."""

    def test_dataset_non_empty(self):
        """Verify the compiled benchmark dataset is populated."""
        self.assertGreater(len(ECKSTEIN_YIELD_DATA), 0)
        self.assertGreaterEqual(len(ECKSTEIN_YIELD_DATA), 80)

    def test_schema_and_types(self):
        """Verify every point has valid columns (ion, target, energy_ev, angle_deg, yield_atoms_per_ion, source)."""
        valid_targets = {"Cu", "Ti", "Al"}
        for i, pt in enumerate(ECKSTEIN_YIELD_DATA):
            self.assertIsInstance(pt, SputterYieldPoint, f"Point {i} is not a SputterYieldPoint")
            self.assertEqual(pt.ion, "Ar+", f"Point {i} has unexpected ion: {pt.ion}")
            self.assertIn(pt.target, valid_targets, f"Point {i} has invalid target: {pt.target}")
            self.assertIsInstance(pt.energy_ev, float)
            self.assertTrue(math.isfinite(pt.energy_ev))
            self.assertIsInstance(pt.angle_deg, float)
            self.assertTrue(math.isfinite(pt.angle_deg))
            self.assertIsInstance(pt.yield_atoms_per_ion, float)
            self.assertTrue(math.isfinite(pt.yield_atoms_per_ion))
            self.assertIsInstance(pt.source, str)
            self.assertGreater(len(pt.source.strip()), 0)

            # Dictionary representation
            d = pt.to_dict()
            expected_keys = {"ion", "target", "energy_ev", "angle_deg", "yield_atoms_per_ion", "source"}
            self.assertEqual(set(d.keys()), expected_keys)

    def test_literature_references_dictionary(self):
        """Verify the reference bibliography is well-formed with known citations."""
        required_refs = [
            "Eckstein2007",
            "YamamuraTawara1996",
            "LaegreidWehner1961",
            "RosenbergWehner1962",
            "Oechsner1973",
        ]
        for ref_key in required_refs:
            self.assertIn(ref_key, ECKSTEIN_REFERENCES)
            self.assertIsInstance(ECKSTEIN_REFERENCES[ref_key], str)
            self.assertGreater(len(ECKSTEIN_REFERENCES[ref_key]), 10)


class TestEcksteinRangeValidity(unittest.TestCase):
    """Test energy, angle, and yield range validity."""

    def test_energy_range_50_to_1000_ev(self):
        """Verify all data points lie strictly within 50 eV to 1000 eV."""
        energies = [pt.energy_ev for pt in ECKSTEIN_YIELD_DATA]
        self.assertEqual(min(energies), 50.0)
        self.assertEqual(max(energies), 1000.0)
        for pt in ECKSTEIN_YIELD_DATA:
            self.assertGreaterEqual(pt.energy_ev, 50.0, f"Energy below 50 eV: {pt}")
            self.assertLessEqual(pt.energy_ev, 1000.0, f"Energy above 1000 eV: {pt}")

    def test_angle_range_0_to_80_deg(self):
        """Verify all angles lie strictly within 0 deg to 80 deg."""
        angles = [pt.angle_deg for pt in ECKSTEIN_YIELD_DATA]
        self.assertEqual(min(angles), 0.0)
        self.assertEqual(max(angles), 80.0)
        for pt in ECKSTEIN_YIELD_DATA:
            self.assertGreaterEqual(pt.angle_deg, 0.0, f"Angle below 0 deg: {pt}")
            self.assertLessEqual(pt.angle_deg, 80.0, f"Angle above 80 deg: {pt}")

    def test_yield_positivity_and_physical_bounds(self):
        """Verify all yields are positive and within realistic physical bounds (0.01 to 10.0 atoms/ion)."""
        for pt in ECKSTEIN_YIELD_DATA:
            self.assertGreater(pt.yield_atoms_per_ion, 0.0, f"Yield must be positive: {pt}")
            self.assertLessEqual(pt.yield_atoms_per_ion, 10.0, f"Yield unreasonably high: {pt}")

    def test_all_materials_represented(self):
        """Verify Cu, Ti, and Al targets all have normal and oblique incidence data."""
        for mat in ["Cu", "Ti", "Al"]:
            pts = [p for p in ECKSTEIN_YIELD_DATA if p.target == mat]
            self.assertGreaterEqual(len(pts), 20, f"Insufficient data points for {mat}")

            normal_pts = [p for p in pts if p.angle_deg == 0.0]
            oblique_pts = [p for p in pts if p.angle_deg > 0.0]
            self.assertGreaterEqual(len(normal_pts), 15, f"Insufficient normal incidence points for {mat}")
            self.assertGreaterEqual(len(oblique_pts), 8, f"Insufficient oblique points for {mat}")


class TestEcksteinPhysicalTrends(unittest.TestCase):
    """Test physical consistency: energy monotonicity, relative yields, and angular peak."""

    def test_relative_material_yields(self):
        """Verify at 600 eV and 1000 eV, Cu yield > Al yield > Ti yield."""
        for energy in [600.0, 1000.0]:
            cu_pts = [p.yield_atoms_per_ion for p in ECKSTEIN_YIELD_DATA if p.target == "Cu" and p.energy_ev == energy and p.angle_deg == 0.0]
            al_pts = [p.yield_atoms_per_ion for p in ECKSTEIN_YIELD_DATA if p.target == "Al" and p.energy_ev == energy and p.angle_deg == 0.0]
            ti_pts = [p.yield_atoms_per_ion for p in ECKSTEIN_YIELD_DATA if p.target == "Ti" and p.energy_ev == energy and p.angle_deg == 0.0]

            cu_mean = float(np.mean(cu_pts))
            al_mean = float(np.mean(al_pts))
            ti_mean = float(np.mean(ti_pts))

            self.assertGreater(cu_mean, al_mean, f"Cu yield should exceed Al at {energy} eV")
            self.assertGreater(al_mean, ti_mean, f"Al yield should exceed Ti at {energy} eV")

    def test_energy_dependence_trend(self):
        """Verify yield increases monotonically with energy from 50 to 1000 eV at normal incidence."""
        for mat in ["Cu", "Ti", "Al"]:
            data = get_eckstein_data(mat, normal_incidence_only=True)
            # Group by energy and compute average yield
            unique_energies = np.unique(data.energy_ev)
            avg_yields = [
                float(np.mean(data.yield_atoms_per_ion[data.energy_ev == e]))
                for e in unique_energies
            ]
            # Ensure strictly increasing with energy
            for k in range(len(avg_yields) - 1):
                self.assertLess(
                    avg_yields[k],
                    avg_yields[k + 1],
                    f"Yield for {mat} did not increase between {unique_energies[k]} eV and {unique_energies[k+1]} eV",
                )

    def test_angular_dependence_peak(self):
        """Verify oblique yield increases to a maximum around 60-70 deg and drops at 80 deg."""
        for mat in ["Cu", "Ti", "Al"]:
            data = get_eckstein_data(mat, min_energy_ev=1000.0, max_energy_ev=1000.0)
            angles = data.angle_deg
            yields = data.yield_atoms_per_ion

            # Normal incidence yield
            y0 = float(yields[angles == 0.0][0])

            # Peak yield should occur at an angle between 50 and 70 degrees
            max_idx = int(np.argmax(yields))
            peak_angle = angles[max_idx]
            peak_yield = yields[max_idx]

            self.assertGreaterEqual(peak_angle, 50.0)
            self.assertLessEqual(peak_angle, 70.0)
            self.assertGreater(peak_yield, y0, f"Peak yield should be greater than normal yield for {mat}")

            # Yield at 80 degrees should drop significantly from the peak
            y80 = float(yields[angles == 80.0][0])
            self.assertLess(y80, peak_yield, f"Yield at 80 deg should drop below peak for {mat}")


class TestGetEcksteinDataHelper(unittest.TestCase):
    """Test helper function get_eckstein_data features, filters, and formats."""

    def test_default_material_cu(self):
        """Default call returns Cu dataset."""
        data = get_eckstein_data()
        self.assertEqual(len(data.target), len(data))
        self.assertTrue(all(t == "Cu" for t in data.target))

    def test_case_insensitivity(self):
        """Material names are case-insensitive ('cu', 'CU', 'Cu')."""
        d_lower = get_eckstein_data("cu")
        d_upper = get_eckstein_data("CU")
        d_mixed = get_eckstein_data("Cu")
        self.assertEqual(len(d_lower), len(d_mixed))
        self.assertEqual(len(d_upper), len(d_mixed))
        np.testing.assert_array_equal(d_lower.yield_atoms_per_ion, d_mixed.yield_atoms_per_ion)

    def test_target_material_interop(self):
        """Passing a TargetMaterial instance directly resolves material name."""
        target_obj = MATERIALS["Ti"]
        data = get_eckstein_data(target_obj)
        self.assertTrue(all(t == "Ti" for t in data.target))

    def test_all_materials_filter(self):
        """Requesting 'all' or None returns all benchmark points across Cu, Ti, and Al."""
        data_all = get_eckstein_data("all")
        data_none = get_eckstein_data(None)
        self.assertEqual(len(data_all), len(ECKSTEIN_YIELD_DATA))
        self.assertEqual(len(data_none), len(ECKSTEIN_YIELD_DATA))
        self.assertEqual(set(data_all.target), {"Cu", "Ti", "Al"})

    def test_invalid_material_raises_error(self):
        """Unknown material raises ValueError."""
        with self.assertRaises(ValueError):
            get_eckstein_data("Iron")
        with self.assertRaises(ValueError):
            get_eckstein_data("Gold")

    def test_return_type_dict(self):
        """return_type='dict' returns EcksteinDataset with numpy arrays."""
        data = get_eckstein_data("Al", return_type="dict")
        self.assertIn("energy_ev", data)
        self.assertIn("yield_atoms_per_ion", data)
        self.assertIn("angle_deg", data)
        self.assertIsInstance(data["energy_ev"], np.ndarray)
        self.assertIsInstance(data.energy_ev, np.ndarray)
        self.assertEqual(data.num_points, len(data["energy_ev"]))
        self.assertEqual(data.size, len(data["energy_ev"]))

    def test_return_type_records(self):
        """return_type='records' returns a list of dictionaries."""
        records = get_eckstein_data("Al", return_type="records")
        self.assertIsInstance(records, list)
        self.assertIsInstance(records[0], dict)
        self.assertEqual(
            set(records[0].keys()),
            {"ion", "target", "energy_ev", "angle_deg", "yield_atoms_per_ion", "source"},
        )

    def test_return_type_array(self):
        """return_type='array' returns a numpy structured array."""
        arr = get_eckstein_data("Cu", return_type="array")
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(
            arr.dtype.names,
            ("ion", "target", "energy_ev", "angle_deg", "yield_atoms_per_ion", "source"),
        )
        self.assertGreater(len(arr), 0)

    def test_return_type_dataframe(self):
        """return_type='dataframe' returns pandas.DataFrame."""
        df = get_eckstein_data("Ti", return_type="dataframe")
        import pandas as pd
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("yield_atoms_per_ion", df.columns)
        self.assertEqual(len(df), len(get_eckstein_data("Ti")))

    def test_invalid_return_type_raises_error(self):
        """Invalid return_type raises ValueError."""
        with self.assertRaises(ValueError):
            get_eckstein_data("Cu", return_type="invalid_format")

    def test_energy_and_angle_filtering(self):
        """Verify min/max energy and angle filtering options."""
        filtered = get_eckstein_data(
            "Cu",
            min_energy_ev=200.0,
            max_energy_ev=600.0,
            min_angle_deg=0.0,
            max_angle_deg=0.0,
        )
        self.assertTrue(all(filtered.energy_ev >= 200.0))
        self.assertTrue(all(filtered.energy_ev <= 600.0))
        self.assertTrue(all(filtered.angle_deg == 0.0))
        self.assertGreater(len(filtered), 0)

    def test_normal_incidence_only_filter(self):
        """normal_incidence_only=True returns only angle_deg == 0."""
        data = get_eckstein_data("all", normal_incidence_only=True)
        self.assertTrue(all(a == 0.0 for a in data.angle_deg))


class TestModelBenchmarkComparison(unittest.TestCase):
    """Test agreement between Eckstein benchmark data and SputterTwin physics model."""

    def test_cu_model_vs_experiment_agreement(self):
        """Verify SputterTwin Yamamura-Tawara formulation agrees with Cu experimental data."""
        data = get_eckstein_data("Cu", normal_incidence_only=True)
        # Select common energies >= 100 eV
        mask = data.energy_ev >= 100.0
        energies = data.energy_ev[mask]
        exp_yields = data.yield_atoms_per_ion[mask]

        model_yields = np.array([
            calculate_sputter_yield(float(e), "Cu", angle_rad=0.0)
            for e in energies
        ])

        # Relative error should be within realistic experimental tolerance (< 25% on average)
        rel_errors = np.abs(model_yields - exp_yields) / exp_yields
        mean_rel_error = float(np.mean(rel_errors))
        self.assertLess(mean_rel_error, 0.25, f"Mean relative error too high: {mean_rel_error:.2%}")

    def test_ti_model_vs_experiment_agreement(self):
        """Verify SputterTwin Yamamura-Tawara formulation agrees with Ti experimental data."""
        data = get_eckstein_data("Ti", normal_incidence_only=True)
        mask = data.energy_ev >= 100.0
        energies = data.energy_ev[mask]
        exp_yields = data.yield_atoms_per_ion[mask]

        model_yields = np.array([
            calculate_sputter_yield(float(e), "Ti", angle_rad=0.0)
            for e in energies
        ])

        rel_errors = np.abs(model_yields - exp_yields) / exp_yields
        mean_rel_error = float(np.mean(rel_errors))
        self.assertLess(mean_rel_error, 0.25, f"Mean relative error too high: {mean_rel_error:.2%}")

    def test_angular_enhancement_model_vs_experiment(self):
        """Verify oblique angular enhancement ratio Y(60 deg) / Y(0 deg) matches within 25%."""
        for mat in ["Cu", "Ti", "Al"]:
            data = get_eckstein_data(mat, min_energy_ev=1000.0, max_energy_ev=1000.0)
            y0_exp = float(data.yield_atoms_per_ion[data.angle_deg == 0.0][0])
            y60_exp = float(data.yield_atoms_per_ion[data.angle_deg == 60.0][0])
            exp_ratio = y60_exp / y0_exp

            y0_model = calculate_sputter_yield(1000.0, mat, angle_rad=0.0)
            y60_model = calculate_sputter_yield(1000.0, mat, angle_rad=math.radians(60.0))
            model_ratio = y60_model / y0_model

            rel_diff = abs(model_ratio - exp_ratio) / exp_ratio
            self.assertLess(
                rel_diff,
                0.25,
                f"Angular enhancement discrepancy for {mat}: exp={exp_ratio:.2f}, model={model_ratio:.2f}",
            )


if __name__ == "__main__":
    unittest.main()
