"""Unit and integration tests for the SputterTwin PINN modules.

Validates PlasmaPINN architecture, positive value constraints, bounded electron
temperatures, physics residual calculations, synthetic data generation, and training loop.
"""

import math
import unittest

import numpy as np
import torch

from sputtertwin.pinn.plasma_pinn import (
    PlasmaPINN,
    PlasmaPINNConfig,
    compute_physics_residuals,
    evaluate_plasma_pinn,
    generate_plasma_dataset,
    train_plasma_pinn,
)


class TestPlasmaPINN(unittest.TestCase):
    """Test suite for the PlasmaPINN neural surrogate and physics constraints."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.config = PlasmaPINNConfig(
            hidden_layers=2,
            neurons=32,
            epochs=10,
            batch_size=16,
            n_collocation=50,
        )
        self.model = PlasmaPINN(config=self.config)

    def test_plasma_dataset_generation(self):
        """Verify synthetic plasma dataset generation shapes and feature properties."""
        ds = generate_plasma_dataset(n_power=5, n_pressure=5, n_flow=2)
        x = ds["X"]
        y = ds["Y"]

        self.assertEqual(x.shape, (50, 3))
        self.assertEqual(y.shape, (50, 5))
        self.assertTrue(np.all(np.isfinite(x)))
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertTrue(np.all(y > 0.0), "All physical plasma outputs must be strictly positive")

    def test_plasma_pinn_forward_and_bounds(self):
        """Verify forward pass output dimensions and strict physical bounds enforcement."""
        batch_size = 10
        # Physical inputs: Power (50-500W), Pressure (1-20 mTorr), Flow (10-40 sccm)
        x_phys = torch.tensor(
            np.column_stack([
                np.random.uniform(50.0, 500.0, batch_size),
                np.random.uniform(1.0, 20.0, batch_size),
                np.random.uniform(10.0, 40.0, batch_size),
            ]),
            dtype=torch.float32,
        )

        outputs = self.model(x_phys)
        self.assertEqual(outputs.shape, (batch_size, 5))

        vd = outputs[:, 0].detach().numpy()
        id_ = outputs[:, 1].detach().numpy()
        gi = outputs[:, 2].detach().numpy()
        te = outputs[:, 3].detach().numpy()
        ne = outputs[:, 4].detach().numpy()

        # Strict physics bounds
        self.assertTrue(np.all(vd > 0.0), "Discharge voltage must be strictly positive")
        self.assertTrue(np.all(id_ > 0.0), "Discharge current must be strictly positive")
        self.assertTrue(np.all(gi > 0.0), "Ion flux must be strictly positive")
        self.assertTrue(np.all((te >= 2.0) & (te <= 4.0)), "Electron temperature must be bounded in [2.0, 4.0] eV")
        self.assertTrue(np.all(ne > 0.0), "Plasma density must be strictly positive")

    def test_physics_residuals_computation(self):
        """Verify that all 4 physics loss terms compute finite, non-negative residuals."""
        x_colloc = torch.tensor(
            np.column_stack([
                np.random.uniform(100.0, 400.0, 20),
                np.random.uniform(2.0, 10.0, 20),
                np.full(20, 20.0),
            ]),
            dtype=torch.float32,
        )

        res = compute_physics_residuals(self.model, x_colloc)
        for key in ["loss_power", "loss_iv", "loss_flux", "loss_bohm"]:
            self.assertIn(key, res)
            val = res[key].item()
            self.assertTrue(math.isfinite(val))
            self.assertGreaterEqual(val, 0.0)

    def test_predict_convenience_api(self):
        """Verify model.predict() scalar convenience method."""
        pred = self.model.predict(power_w=220.0, pressure_mtorr=5.0, ar_flow_sccm=20.0)
        expected_keys = [
            "voltage_v",
            "current_a",
            "ion_flux",
            "electron_temp_ev",
            "plasma_density_m3",
            "power_calc_w",
        ]
        for k in expected_keys:
            self.assertIn(k, pred)
            self.assertTrue(math.isfinite(pred[k]))
            self.assertGreater(pred[k], 0.0)

        # Te bounded in [2, 4]
        self.assertGreaterEqual(pred["electron_temp_ev"], 2.0)
        self.assertLessEqual(pred["electron_temp_ev"], 4.0)

    def test_short_training_loop_convergence(self):
        """Verify that train_plasma_pinn executes and logs loss reductions."""
        ds = generate_plasma_dataset(n_power=4, n_pressure=4, n_flow=2)
        cfg = PlasmaPINNConfig(
            epochs=25,
            neurons=16,
            hidden_layers=2,
            n_collocation=20,
        )
        trained_model, history = train_plasma_pinn(config=cfg, dataset=ds, verbose=False)
        self.assertEqual(len(history["train_loss"]), 25)
        self.assertEqual(len(history["val_loss"]), 25)
        self.assertTrue(all(math.isfinite(loss) for loss in history["train_loss"]))

        metrics = evaluate_plasma_pinn(trained_model, ds)
        self.assertIn("voltage_v", metrics)
        self.assertIn("power_conservation", metrics)


if __name__ == "__main__":
    unittest.main()
