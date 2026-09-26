"""Physics-Informed Neural Network (PINN) for DC Magnetron Plasma Discharge.

Surrogate modeling for the plasma discharge stage in DC magnetron sputtering:
Maps operational inputs:
    [Power W (W), Pressure P (mTorr), Argon Flow (sccm)]
to discharge state observables:
    [Voltage V_d (V), Current I_d (A), Ion Flux Gamma_i (ions/(m^2*s)),
     Electron Temperature T_e (eV), Plasma Density n_e (m^-3)]

Physics constraints enforced during training:
1. Power Conservation: W = V_d * I_d
2. Magnetron I-V-P Characteristic: I_d = k * (P_eff)^m * (V_d)^n
3. Ion Flux Relation: Gamma_i = I_ion / (e * A_race), where I_ion = I_d / (1 + gamma_se)
4. Bohm Sheath Criterion: Gamma_i = exp(-0.5) * n_e * sqrt(e * T_e / M_Ar)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sputtertwin.physics.plasma import (
    _ELEMENTARY_CHARGE_C,
    _GAMMA_SE,
    _K_DISCHARGE,
    _K_FLOW_MTORR_PER_SCCM,
    _AR_FLOW_BASELINE_SCCM,
    _M_AR_KG,
    _M_EXPONENT,
    _N_EXPONENT,
    DischargeState,
    calculate_discharge_state,
)

__all__ = [
    "PlasmaPINN",
    "PlasmaPINNConfig",
    "generate_plasma_dataset",
    "compute_physics_residuals",
    "train_plasma_pinn",
    "evaluate_plasma_pinn",
]


@dataclass
class PlasmaPINNConfig:
    """Hyperparameter and training configuration for PlasmaPINN."""

    hidden_layers: int = 4
    neurons: int = 64
    activation: str = "tanh"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lambda_data: float = 1.0
    lambda_power: float = 0.5
    lambda_iv: float = 0.2
    lambda_flux: float = 0.2
    lambda_bohm: float = 0.1
    epochs: int = 2000
    batch_size: int = 64
    n_collocation: int = 1000
    target_radius_m: float = 0.05
    racetrack_ratio: float = 0.5
    racetrack_width_m: Optional[float] = None


class PlasmaPINN(nn.Module):
    """Physics-Informed Neural Network for DC Magnetron Plasma Discharge."""

    def __init__(
        self,
        config: Optional[PlasmaPINNConfig] = None,
        x_mean: Optional[np.ndarray] = None,
        x_std: Optional[np.ndarray] = None,
        y_mean: Optional[np.ndarray] = None,
        y_std: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.config = config or PlasmaPINNConfig()

        # Build MLP trunk with smooth activation (tanh for well-behaved autodiff)
        activation_cls = nn.Tanh if self.config.activation == "tanh" else nn.GELU
        layers: list[nn.Module] = [nn.Linear(3, self.config.neurons), activation_cls()]
        for _ in range(self.config.hidden_layers - 1):
            layers.extend([nn.Linear(self.config.neurons, self.config.neurons), activation_cls()])
        layers.append(nn.Linear(self.config.neurons, 5))
        self.net = nn.Sequential(*layers)

        # Normalization buffers for seamless integration
        self.register_buffer(
            "x_mean",
            torch.tensor(x_mean if x_mean is not None else np.zeros(3), dtype=torch.float32),
        )
        self.register_buffer(
            "x_std",
            torch.tensor(x_std if x_std is not None else np.ones(3), dtype=torch.float32),
        )
        self.register_buffer(
            "y_mean",
            torch.tensor(y_mean if y_mean is not None else np.zeros(5), dtype=torch.float32),
        )
        self.register_buffer(
            "y_std",
            torch.tensor(y_std if y_std is not None else np.ones(5), dtype=torch.float32),
        )

    def forward_normalized(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Forward pass taking normalized inputs [batch, 3] -> normalized outputs [batch, 5]."""
        return self.net(x_norm)

    def forward(self, x_phys: torch.Tensor) -> torch.Tensor:
        """Forward pass taking physical inputs [W, P, flow] and returning physical outputs.

        Enforces physical bounds:
        - V_d > 0 (softplus)
        - I_d > 0 (softplus)
        - Gamma_i > 0 (softplus)
        - T_e bounded in [2.0, 4.0] eV (sigmoid scaling)
        - n_e > 0 (softplus)
        """
        # Normalize inputs
        x_norm = (x_phys - self.x_mean) / (self.x_std + 1e-8)
        y_norm = self.net(x_norm)
        # Denormalize to approximate physical space
        y_phys_approx = y_norm * (self.y_std + 1e-8) + self.y_mean

        # Apply physics-preserving output activations
        vd = F.softplus(y_phys_approx[:, 0:1])
        id_ = F.softplus(y_phys_approx[:, 1:2])
        gi = F.softplus(y_phys_approx[:, 2:3])
        # Smoothly bound electron temperature to physically admissible [2.0, 4.0] eV
        te = 2.0 + 2.0 * torch.sigmoid((y_phys_approx[:, 3:4] - 3.0) / 0.5)
        ne = F.softplus(y_phys_approx[:, 4:5])

        return torch.cat([vd, id_, gi, te, ne], dim=1)

    def predict(
        self,
        power_w: float,
        pressure_mtorr: float,
        ar_flow_sccm: float = 20.0,
    ) -> Dict[str, float]:
        """High-level evaluation method returning a dictionary of physical scalar predictions."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor([[power_w, pressure_mtorr, ar_flow_sccm]], dtype=torch.float32)
            out = self.forward(x).cpu().numpy()[0]
            return {
                "voltage_v": float(out[0]),
                "current_a": float(out[1]),
                "ion_flux": float(out[2]),
                "electron_temp_ev": float(out[3]),
                "plasma_density_m3": float(out[4]),
                "power_calc_w": float(out[0] * out[1]),
            }


def generate_plasma_dataset(
    n_power: int = 25,
    n_pressure: int = 25,
    n_flow: int = 3,
    power_range: Tuple[float, float] = (50.0, 500.0),
    pressure_range: Tuple[float, float] = (1.0, 20.0),
    flow_range: Tuple[float, float] = (10.0, 40.0),
    noise_std: float = 0.0,
    random_seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate synthetic plasma discharge dataset using the calibrated analytical physics engine.

    Returns:
        Dictionary containing inputs (X), ground-truth observables (Y), and feature names.
    """
    rng = np.random.default_rng(random_seed)
    powers = np.linspace(power_range[0], power_range[1], n_power)
    pressures = np.linspace(pressure_range[0], pressure_range[1], n_pressure)
    flows = np.linspace(flow_range[0], flow_range[1], n_flow)

    records = []
    for w in powers:
        for p in pressures:
            for f in flows:
                ds: DischargeState = calculate_discharge_state(
                    power_w=float(w),
                    pressure_mtorr=float(p),
                    ar_flow_sccm=float(f),
                )
                records.append([
                    w,
                    p,
                    f,
                    ds.voltage_v,
                    ds.current_a,
                    ds.ion_flux,
                    ds.electron_temp_ev,
                    ds.plasma_density_m3,
                ])

    data = np.array(records, dtype=np.float64)
    x = data[:, :3]
    y = data[:, 3:]

    if noise_std > 0.0:
        # Add relative Gaussian noise to simulate experimental instrumentation variance
        y *= (1.0 + rng.normal(0.0, noise_std, size=y.shape))

    return {
        "X": x,
        "Y": y,
        "feature_names": ["power_w", "pressure_mtorr", "ar_flow_sccm"],
        "target_names": [
            "voltage_v",
            "current_a",
            "ion_flux",
            "electron_temp_ev",
            "plasma_density_m3",
        ],
    }


def compute_physics_residuals(
    model: PlasmaPINN,
    x_colloc_phys: torch.Tensor,
    target_radius_m: float = 0.05,
    racetrack_ratio: float = 0.5,
    racetrack_width_m: Optional[float] = None,
    gamma_se: float = _GAMMA_SE,
    n_exp: float = _N_EXPONENT,
    m_exp: float = _M_EXPONENT,
    k_const: float = _K_DISCHARGE,
) -> Dict[str, torch.Tensor]:
    """Evaluate physics governing equations at collocation points.

    Returns normalized mean squared residuals for:
    1. L_power: Power conservation (W - V_d * I_d) / W
    2. L_iv: Magnetron characteristic (I_d - k * P_eff^m * V_d^n) / I_d
    3. L_flux: Ion flux conservation (Gamma_i - I_ion / (e * A_race)) / Gamma_i
    4. L_bohm: Bohm sheath plasma density criterion (ne - Gamma_i / (exp(-0.5) * u_bohm)) / ne
    """
    if racetrack_width_m is None:
        racetrack_width_m = 0.20 * target_radius_m

    r_mean = racetrack_ratio * target_radius_m
    a_race = 2.0 * math.pi * r_mean * racetrack_width_m

    # Forward pass on collocation points in physical coordinates
    preds = model(x_colloc_phys)
    vd = preds[:, 0:1]
    id_ = preds[:, 1:2]
    gi = preds[:, 2:3]
    te = preds[:, 3:4]
    ne = preds[:, 4:5]

    w_input = x_colloc_phys[:, 0:1]
    p_input = x_colloc_phys[:, 1:2]
    f_input = x_colloc_phys[:, 2:3]

    # Effective pressure coupling (Option B model from plasma.py)
    p_eff = p_input + _K_FLOW_MTORR_PER_SCCM * (f_input - _AR_FLOW_BASELINE_SCCM)
    p_eff = torch.clamp(p_eff, min=1e-3)

    # 1. Power Conservation Residual: W = V_d * I_d (dimensionless relative error)
    res_power = (w_input - (vd * id_)) / (w_input + 1e-6)
    loss_power = torch.mean(res_power ** 2)

    # 2. Magnetron I-V Characteristic Residual: I_d = k * P_eff^m * V_d^n
    id_expected = k_const * (p_eff ** m_exp) * (vd ** n_exp)
    res_iv = (id_ - id_expected) / (torch.clamp(id_expected, min=0.1) + 1e-6)
    loss_iv = torch.mean(res_iv ** 2)

    # 3. Ion Flux Conservation: Gamma_i = I_ion / (e * A_race)
    i_ion = id_ / (1.0 + gamma_se)
    gi_expected = i_ion / (_ELEMENTARY_CHARGE_C * a_race)
    res_flux = (gi - gi_expected) / (torch.clamp(gi_expected, min=1e18) + 1e-6)
    loss_flux = torch.mean(res_flux ** 2)

    # 4. Bohm Sheath Criterion: Gamma_i = exp(-0.5) * n_e * sqrt(e * T_e / M_Ar)
    u_bohm = torch.sqrt((_ELEMENTARY_CHARGE_C * te) / _M_AR_KG)
    ne_expected = gi / (math.exp(-0.5) * u_bohm)
    res_bohm = (ne - ne_expected) / (torch.clamp(ne_expected, min=1e15) + 1e-6)
    loss_bohm = torch.mean(res_bohm ** 2)

    return {
        "loss_power": loss_power,
        "loss_iv": loss_iv,
        "loss_flux": loss_flux,
        "loss_bohm": loss_bohm,
    }


def train_plasma_pinn(
    config: Optional[PlasmaPINNConfig] = None,
    dataset: Optional[Dict[str, np.ndarray]] = None,
    verbose: bool = True,
) -> Tuple[PlasmaPINN, Dict[str, list[float]]]:
    """Train the PlasmaPINN using combined supervised data loss and physics residuals."""
    cfg = config or PlasmaPINNConfig()
    ds = dataset or generate_plasma_dataset()

    x_all = ds["X"]
    y_all = ds["Y"]

    # Shuffle and split: 80% train, 20% validation
    n_samples = len(x_all)
    indices = np.random.permutation(n_samples)
    n_train = int(0.8 * n_samples)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]

    # Compute normalization statistics from training set
    x_mean = np.mean(x_train, axis=0)
    x_std = np.std(x_train, axis=0)
    y_mean = np.mean(y_train, axis=0)
    y_std = np.std(y_train, axis=0)

    # Initialize model
    model = PlasmaPINN(
        config=cfg,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=200, min_lr=1e-6
    )

    t_x_train = torch.tensor(x_train, dtype=torch.float32)
    t_y_train = torch.tensor(y_train, dtype=torch.float32)
    t_x_val = torch.tensor(x_val, dtype=torch.float32)
    t_y_val = torch.tensor(y_val, dtype=torch.float32)

    history: Dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "data_loss": [],
        "physics_loss": [],
        "loss_power": [],
        "loss_iv": [],
        "loss_flux": [],
        "loss_bohm": [],
    }

    if verbose:
        print(f"Starting PlasmaPINN training: {cfg.epochs} epochs, {n_train} train points...")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # 1. Supervised Data Loss (evaluated in normalized output space for numerical stability)
        y_pred_phys = model(t_x_train)
        # Normalized MSE across all 5 outputs
        y_pred_norm = (y_pred_phys - model.y_mean) / (model.y_std + 1e-8)
        y_true_norm = (t_y_train - model.y_mean) / (model.y_std + 1e-8)
        loss_data = F.mse_loss(y_pred_norm, y_true_norm)

        # 2. Physics Residual Loss evaluated at random collocation points across operating domain
        w_coll = torch.FloatTensor(cfg.n_collocation, 1).uniform_(50.0, 500.0)
        p_coll = torch.FloatTensor(cfg.n_collocation, 1).uniform_(1.0, 20.0)
        f_coll = torch.FloatTensor(cfg.n_collocation, 1).uniform_(10.0, 40.0)
        x_colloc = torch.cat([w_coll, p_coll, f_coll], dim=1)

        phys_res = compute_physics_residuals(
            model=model,
            x_colloc_phys=x_colloc,
            target_radius_m=cfg.target_radius_m,
            racetrack_ratio=cfg.racetrack_ratio,
            racetrack_width_m=cfg.racetrack_width_m,
        )

        loss_phys = (
            cfg.lambda_power * phys_res["loss_power"]
            + cfg.lambda_iv * phys_res["loss_iv"]
            + cfg.lambda_flux * phys_res["loss_flux"]
            + cfg.lambda_bohm * phys_res["loss_bohm"]
        )

        total_loss = cfg.lambda_data * loss_data + loss_phys
        total_loss.backward()
        optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            y_val_phys = model(t_x_val)
            y_val_norm = (y_val_phys - model.y_mean) / (model.y_std + 1e-8)
            y_val_true_norm = (t_y_val - model.y_mean) / (model.y_std + 1e-8)
            val_loss = F.mse_loss(y_val_norm, y_val_true_norm).item()

        scheduler.step(val_loss)

        # Record metrics
        history["train_loss"].append(float(total_loss.item()))
        history["val_loss"].append(val_loss)
        history["data_loss"].append(float(loss_data.item()))
        history["physics_loss"].append(float(loss_phys.item()))
        history["loss_power"].append(float(phys_res["loss_power"].item()))
        history["loss_iv"].append(float(phys_res["loss_iv"].item()))
        history["loss_flux"].append(float(phys_res["loss_flux"].item()))
        history["loss_bohm"].append(float(phys_res["loss_bohm"].item()))

        if verbose and (epoch % 200 == 0 or epoch == cfg.epochs):
            print(
                f"Epoch {epoch:4d}/{cfg.epochs} | "
                f"Train Loss: {total_loss.item():.5f} | "
                f"Data Loss: {loss_data.item():.5f} | "
                f"Physics: {loss_phys.item():.5f} | "
                f"Val Loss: {val_loss:.5f} | "
                f"Pwr Err: {phys_res['loss_power'].item():.2e}"
            )

    return model, history


def evaluate_plasma_pinn(
    model: PlasmaPINN,
    test_dataset: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Evaluate trained PlasmaPINN on test data, computing accuracy and physics adherence metrics."""
    ds = test_dataset or generate_plasma_dataset(n_power=10, n_pressure=10, n_flow=2, random_seed=999)
    x_test = torch.tensor(ds["X"], dtype=torch.float32)
    y_test = ds["Y"]

    model.eval()
    with torch.no_grad():
        y_pred = model(x_test).cpu().numpy()

    target_names = ds["target_names"]
    metrics: Dict[str, Any] = {}

    # Mean Absolute Percentage Error (MAPE) and R2 for each output variable
    for i, name in enumerate(target_names):
        actual = y_test[:, i]
        predicted = y_pred[:, i]
        abs_rel_err = np.abs((predicted - actual) / (actual + 1e-12)) * 100.0
        mape = float(np.mean(abs_rel_err))
        max_err = float(np.max(abs_rel_err))

        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))

        metrics[name] = {
            "mape_percent": mape,
            "max_error_percent": max_err,
            "r2_score": r2,
        }

    # Physical Power Conservation check: |W - V_d * I_d| / W
    w_true = ds["X"][:, 0]
    p_calc = y_pred[:, 0] * y_pred[:, 1]
    power_err_pct = np.abs((w_true - p_calc) / w_true) * 100.0
    metrics["power_conservation"] = {
        "mean_error_percent": float(np.mean(power_err_pct)),
        "max_error_percent": float(np.max(power_err_pct)),
    }

    return metrics


if __name__ == "__main__":
    print("=" * 70)
    print("SputterTwin Plasma Discharge PINN (Step 1 Implementation)")
    print("=" * 70)

    # 1. Generate synthetic dataset from existing analytical model
    dataset = generate_plasma_dataset(n_power=20, n_pressure=20, n_flow=3)
    print(f"Generated {len(dataset['X'])} training samples.")

    # 2. Train PINN with coupled data + physics loss
    config = PlasmaPINNConfig(epochs=1000, neurons=64, hidden_layers=4)
    model, history = train_plasma_pinn(config=config, dataset=dataset, verbose=True)

    # 3. Evaluate accuracy against independent benchmark grid
    print("\nEvaluating trained model on independent test grid...")
    results = evaluate_plasma_pinn(model)

    print("\n" + "-" * 70)
    print(f"{'Target Parameter':<22} | {'MAPE (%)':<10} | {'Max Error (%)':<15} | {'R² Score':<10}")
    print("-" * 70)
    for target in dataset["target_names"]:
        res = results[target]
        print(f"{target:<22} | {res['mape_percent']:<10.3f} | {res['max_error_percent']:<15.3f} | {res['r2_score']:<10.5f}")

    pwr = results["power_conservation"]
    print("-" * 70)
    print(f"Power Conservation Error: Mean = {pwr['mean_error_percent']:.3f}%, Max = {pwr['max_error_percent']:.3f}%")
    print("=" * 70)

    # Save trained checkpoint
    save_path = "plasma_pinn_cu.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Trained model checkpoint saved successfully to: {save_path}")
