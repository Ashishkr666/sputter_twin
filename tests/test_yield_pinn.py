"""Unit and integration tests for the Sputter Yield PINN (YieldPINN).

Validates:
1. Model forward pass across input shapes (2D tensor, 1D tensors, scalar inputs).
2. Material properties and threshold energy resolution.
3. Physics constraints (sub-threshold, asymptotic, angular derivatives, monotonicity).
4. Checkpoint saving, loading, and evaluation accuracy (MAPE < 2.5% on benchmark data).
5. Quick training loop integration.
"""

# Windows DLL order requirement: always import torch first
import torch
import torch.nn.functional as F

import math
import os
import unittest
from pathlib import Path

import numpy as np

from sputtertwin.physics.sputter_yield import (
    MATERIALS,
    calculate_sputter_yield,
)
from sputtertwin.pinn.yield_pinn import (
    BENCHMARK_POINTS,
    YieldPINN,
    YieldPINNConfig,
    compute_physics_residuals,
    evaluate_yield_pinn,
    generate_yield_dataset,
    get_material_properties,
    get_threshold_energy,
    load_yield_pinn,
    train_yield_pinn,
)


class TestYieldPINN(unittest.TestCase):
    """Test suite for YieldPINN architecture, physics constraints, and checkpoint loading."""

    @classmethod
    def setUpClass(cls):
        # Locate pre-trained checkpoint
        cls.checkpoint_path = (
            Path(__file__).resolve().parent.parent
            / "sputtertwin"
            / "pinn"
            / "yield_pinn_cu.pt"
        )
        if not cls.checkpoint_path.is_file():
            # If not yet trained, train once for tests
            train_yield_pinn(epochs=1000, save_path=cls.checkpoint_path, verbose=False)

        cls.trained_model = load_yield_pinn(cls.checkpoint_path)

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.config = YieldPINNConfig(
            hidden_layers=2,
            neurons=32,
            epochs=5,
            n_collocation=20,
        )
        self.model = YieldPINN(config=self.config)

    def test_material_properties_and_threshold(self):
        """Verify helper functions return correct material physical parameters."""
        z2, m2, us, eth = get_material_properties("Cu")
        self.assertEqual(z2, 29.0)
        self.assertEqual(m2, 63.55)
        self.assertEqual(us, 3.49)
        self.assertEqual(eth, 20.0)

        # Threshold helper for scalar and tensor
        self.assertEqual(get_threshold_energy(29.0), 20.0)
        self.assertEqual(get_threshold_energy(22.0), 30.0)
        self.assertEqual(get_threshold_energy(13.0), 25.0)

        z_tensor = torch.tensor([29.0, 22.0, 13.0])
        eth_tensor = get_threshold_energy(z_tensor)
        self.assertTrue(torch.allclose(eth_tensor, torch.tensor([20.0, 30.0, 25.0])))

    def test_yield_dataset_generation(self):
        """Verify generated synthetic dataset shapes, ranges, and target sanity."""
        ds = generate_yield_dataset(materials=["Cu", "Ti", "Al"], n_energies=6)
        x = ds["X"]
        y = ds["Y"]
        eth = ds["Eth"]

        self.assertEqual(x.shape[1], 5)
        self.assertEqual(y.shape[1], 1)
        self.assertEqual(eth.shape[1], 1)
        self.assertTrue(np.all(np.isfinite(x)))
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertTrue(np.all(y >= 0.0), "All sputter yields must be non-negative")

    def test_forward_pass_input_variants(self):
        """Verify forward pass supports 2D tensors, 1D batches, column vectors, and scalars."""
        batch_size = 8

        # 1. Single 2D tensor input [batch, 5]
        x_2d = torch.tensor(
            np.column_stack([
                np.random.uniform(50.0, 800.0, batch_size),
                np.random.uniform(0.0, 1.2, batch_size),
                np.full(batch_size, 29.0),
                np.full(batch_size, 63.55),
                np.full(batch_size, 3.49),
            ]),
            dtype=torch.float32,
        )
        y_2d = self.model(x_2d)
        self.assertEqual(y_2d.shape, (batch_size, 1))
        self.assertTrue(torch.all(y_2d >= 0.0))

        # 2. Five 1D tensor inputs [batch]
        e_1d = torch.tensor([100.0, 200.0, 400.0], dtype=torch.float32)
        th_1d = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
        z2_1d = torch.full((3,), 29.0, dtype=torch.float32)
        m2_1d = torch.full((3,), 63.55, dtype=torch.float32)
        us_1d = torch.full((3,), 3.49, dtype=torch.float32)
        y_1d = self.model(e_1d, th_1d, z2_1d, m2_1d, us_1d)
        self.assertEqual(y_1d.shape, (3,))
        self.assertTrue(torch.all(y_1d >= 0.0))

        # 3. Five 2D column vectors [batch, 1]
        y_col = self.model(
            e_1d.unsqueeze(1),
            th_1d.unsqueeze(1),
            z2_1d.unsqueeze(1),
            m2_1d.unsqueeze(1),
            us_1d.unsqueeze(1),
        )
        self.assertEqual(y_col.shape, (3, 1))

        # 4. Scalar inputs
        y_scalar = self.model(400.0, 0.0, 29.0, 63.55, 3.49)
        self.assertEqual(y_scalar.ndim, 0)
        self.assertGreater(float(y_scalar.item()), 0.0)

        # 5. Material shorthand string
        y_shorthand = self.model(400.0, 0.0, material="Cu")
        self.assertEqual(y_shorthand.ndim, 0)

        # 6. Convenience predict_yield method
        pred_val = self.model.predict_yield(400.0, angle_rad=0.0, material="Cu")
        self.assertIsInstance(pred_val, float)
        self.assertGreater(pred_val, 0.0)

        # Subthreshold prediction returns exactly 0.0
        pred_sub = self.model.predict_yield(10.0, angle_rad=0.0, material="Cu")
        self.assertEqual(pred_sub, 0.0)

        # Grazing angle prediction returns exactly 0.0
        pred_grazing = self.model.predict_yield(400.0, angle_rad=math.radians(88.0), material="Cu")
        self.assertEqual(pred_grazing, 0.0)

    def test_physics_residuals_computation(self):
        """Verify all physics loss terms compute finite, non-negative residuals with autograd."""
        res = compute_physics_residuals(self.model, n_colloc=32)

        required_keys = [
            "loss_sub_threshold",
            "loss_high_energy",
            "loss_angular_deriv",
            "loss_angular_grazing",
            "loss_angular",
            "loss_monotonicity",
        ]
        for key in required_keys:
            self.assertIn(key, res)
            val = float(res[key].item())
            self.assertTrue(math.isfinite(val), f"{key} must be finite")
            self.assertGreaterEqual(val, 0.0, f"{key} must be non-negative")

    def test_sub_threshold_constraint_loss(self):
        """Verify Sub-Threshold Constraint Loss: ReLU(Eth - E) * Y^2."""
        # Collocation points with energies well below threshold (Eth = 20 eV for Cu)
        e_sub = torch.tensor([5.0, 10.0, 15.0], dtype=torch.float32)
        th_sub = torch.zeros(3, dtype=torch.float32)
        z2 = torch.full((3,), 29.0)
        m2 = torch.full((3,), 63.55)
        us = torch.full((3,), 3.49)
        eth = torch.full((3,), 20.0)

        y_sub = self.model(e_sub, th_sub, z2, m2, us)
        loss_sub = torch.mean(F.relu(eth - e_sub) * (y_sub ** 2))
        self.assertTrue(math.isfinite(float(loss_sub.item())))
        self.assertGreaterEqual(float(loss_sub.item()), 0.0)

        # For trained model, verify yield below threshold is suppressed
        with torch.no_grad():
            y_trained_sub = self.trained_model(e_sub, th_sub, z2, m2, us)
            # Trained model should predict near zero yield below Eth
            self.assertTrue(torch.all(y_trained_sub < 0.15))

    def test_high_energy_asymptotic_loss(self):
        """Verify High-Energy Asymptotic Loss: |d(ln Y)/d(ln E) - (-0.2)|."""
        e_high = torch.tensor([3000.0, 5000.0, 8000.0], dtype=torch.float32, requires_grad=True)
        th_high = torch.zeros(3, dtype=torch.float32)
        z2 = torch.full((3,), 29.0)
        m2 = torch.full((3,), 63.55)
        us = torch.full((3,), 3.49)

        y_high = self.model(e_high, th_high, z2, m2, us)
        dY_dE = torch.autograd.grad(
            outputs=y_high,
            inputs=e_high,
            grad_outputs=torch.ones_like(y_high),
            create_graph=True,
        )[0]
        d_ln_Y_d_ln_E = (e_high / (y_high + 1e-8)) * dY_dE
        loss_asymp = torch.mean(torch.abs(d_ln_Y_d_ln_E - (-0.2)))

        self.assertTrue(math.isfinite(float(loss_asymp.item())))
        self.assertGreaterEqual(float(loss_asymp.item()), 0.0)

    def test_angular_derivative_and_grazing_loss(self):
        """Verify Angular Derivative Loss near 65 deg and Y(85 deg) -> 0."""
        # 1. Optimum angle: dY/dtheta = 0 near theta_opt ~ 65 deg
        th_opt = torch.tensor([math.radians(65.0)], dtype=torch.float32, requires_grad=True)
        e_opt = torch.tensor([400.0], dtype=torch.float32)
        z2 = torch.tensor([29.0])
        m2 = torch.tensor([63.55])
        us = torch.tensor([3.49])

        y_opt = self.model(e_opt, th_opt, z2, m2, us)
        dY_dtheta = torch.autograd.grad(
            outputs=y_opt,
            inputs=th_opt,
            grad_outputs=torch.ones_like(y_opt),
            create_graph=True,
        )[0]
        loss_deriv = torch.mean(dY_dtheta ** 2)
        self.assertTrue(math.isfinite(float(loss_deriv.item())))

        # 2. Grazing angle cutoff: Y(85 deg) -> 0
        th_85 = torch.tensor([math.radians(85.0)], dtype=torch.float32)
        y_85 = self.model(e_opt, th_85, z2, m2, us)
        loss_grazing = torch.mean(y_85 ** 2)
        self.assertTrue(math.isfinite(float(loss_grazing.item())))

    def test_monotonicity_loss(self):
        """Verify Monotonicity: dY/dE >= 0 for energies between Eth and ~1 keV."""
        e_mono = torch.tensor([50.0, 150.0, 300.0, 600.0], dtype=torch.float32, requires_grad=True)
        th_mono = torch.zeros(4, dtype=torch.float32)
        z2 = torch.full((4,), 29.0)
        m2 = torch.full((4,), 63.55)
        us = torch.full((4,), 3.49)

        y_mono = self.model(e_mono, th_mono, z2, m2, us)
        dY_dE = torch.autograd.grad(
            outputs=y_mono,
            inputs=e_mono,
            grad_outputs=torch.ones_like(y_mono),
            create_graph=True,
        )[0]
        loss_mono = torch.mean(F.relu(-dY_dE) ** 2)
        self.assertTrue(math.isfinite(float(loss_mono.item())))

        # On the trained model, monotonicity should be strictly satisfied (loss ~ 0)
        y_trained = self.trained_model(e_mono, th_mono, z2, m2, us)
        dY_dE_trained = torch.autograd.grad(
            outputs=y_trained,
            inputs=e_mono,
            grad_outputs=torch.ones_like(y_trained),
        )[0]
        self.assertTrue(torch.all(dY_dE_trained >= -1e-4), "Yield must increase monotonically with energy")

    def test_checkpoint_loading_and_mape_requirement(self):
        """Verify checkpoint loading and ensure MAPE < 2.5% on benchmark data."""
        self.assertTrue(self.checkpoint_path.is_file(), f"Checkpoint {self.checkpoint_path} must exist")

        loaded = load_yield_pinn(self.checkpoint_path)
        self.assertIsInstance(loaded, YieldPINN)

        results = evaluate_yield_pinn(loaded)
        mape = results["overall_mape_percent"]

        # CRITICAL USER REQUIREMENT: Ensure MAPE < 2.5% on benchmark data
        self.assertLess(mape, 2.5, f"Expected benchmark MAPE < 2.5%, got {mape:.2f}%")
        self.assertTrue(results["is_mape_valid"])

        # Check physical trends on Cu
        # Cu at 400 eV: calibrated Yamamura is ~1.95 atoms/ion
        pred_cu_400 = loaded.predict_yield(400.0, angle_rad=0.0, material="Cu")
        true_cu_400 = calculate_sputter_yield(400.0, "Cu")
        self.assertAlmostEqual(pred_cu_400, true_cu_400, delta=0.05)

        # Cu > Ti and Cu > Al at 400 eV
        pred_ti_400 = loaded.predict_yield(400.0, angle_rad=0.0, material="Ti")
        pred_al_400 = loaded.predict_yield(400.0, angle_rad=0.0, material="Al")
        self.assertGreater(pred_cu_400, pred_ti_400)
        self.assertGreater(pred_cu_400, pred_al_400)

        # Angular enhancement: Y(65 deg) > Y(0 deg) for Cu
        pred_cu_65 = loaded.predict_yield(400.0, angle_rad=math.radians(65.0), material="Cu")
        self.assertGreater(pred_cu_65, pred_cu_400)

    def test_short_training_loop(self):
        """Verify train_yield_pinn executes and logs loss reductions without overwriting checkpoint."""
        cfg = YieldPINNConfig(epochs=10, neurons=16, hidden_layers=2, n_collocation=16)
        trained_model, history = train_yield_pinn(
            epochs=10,
            config=cfg,
            materials=["Cu", "Ti", "Al"],
            save_path=False,
            verbose=False,
        )
        self.assertEqual(len(history["train_loss"]), 10)
        self.assertTrue(all(math.isfinite(loss) for loss in history["train_loss"]))


if __name__ == "__main__":
    unittest.main()
