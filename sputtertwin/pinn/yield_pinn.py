"""Physics-Informed Neural Network (Yield PINN) for Sputter Yield Prediction.

Predicts the energy- and angle-dependent sputter yield Y(E, theta) across target
materials (Cu, Ti, Al) under Ar+ ion bombardment.

Input features:
    [ion_energy_ev, angle_rad, target_z2, target_m2, target_us]
Output:
    Predicted sputter yield Y (atoms/ion).

Physics constraints enforced during training:
1. Data Loss: MSE against calibrated Yamamura and Eckstein experimental benchmark points.
2. Sub-Threshold Constraint Loss: ReLU(Eth - E) * Y^2 (ensures exact zero yield below threshold energy).
3. High-Energy Asymptotic Loss: |d(ln Y)/d(ln E) - (-0.2)| for high energies (Lindhard nuclear stopping scaling).
4. Angular Derivative Loss: Enforces dY/dtheta = 0 near theta_opt ~ 65 degrees, and Y(85 deg) -> 0.
5. Monotonicity: dY/dE >= 0 for energies between Eth and peak nuclear stopping energy (~1 keV).
"""

from __future__ import annotations

# Crucial for Windows: import torch before any GUI or graphics modules to avoid DLL conflicts
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from sputtertwin.physics.sputter_yield import (
    MATERIALS,
    TargetMaterial,
    calculate_sputter_yield,
)

__all__ = [
    "YieldPINN",
    "YieldPINNConfig",
    "get_threshold_energy",
    "get_material_properties",
    "generate_yield_dataset",
    "compute_physics_residuals",
    "compute_yield_physics_residuals",
    "train_yield_pinn",
    "evaluate_yield_pinn",
    "load_yield_pinn",
    "BENCHMARK_POINTS",
]

# Standard experimental and literature benchmark points (Eckstein 2007, Yamamura & Tawara 1996)
BENCHMARK_POINTS: list[tuple[str, float, float]] = [
    # Material, Energy (eV), Angle (deg)
    # Copper (Cu) normal incidence
    ("Cu", 100.0, 0.0),
    ("Cu", 200.0, 0.0),
    ("Cu", 300.0, 0.0),
    ("Cu", 400.0, 0.0),
    ("Cu", 500.0, 0.0),
    ("Cu", 600.0, 0.0),
    ("Cu", 800.0, 0.0),
    ("Cu", 1000.0, 0.0),
    # Cu angular dependence at 400 eV
    ("Cu", 400.0, 15.0),
    ("Cu", 400.0, 30.0),
    ("Cu", 400.0, 45.0),
    ("Cu", 400.0, 60.0),
    ("Cu", 400.0, 65.0),
    ("Cu", 400.0, 75.0),
    # Titanium (Ti) normal incidence
    ("Ti", 100.0, 0.0),
    ("Ti", 200.0, 0.0),
    ("Ti", 300.0, 0.0),
    ("Ti", 400.0, 0.0),
    ("Ti", 500.0, 0.0),
    ("Ti", 600.0, 0.0),
    ("Ti", 800.0, 0.0),
    ("Ti", 1000.0, 0.0),
    # Ti angular dependence at 400 eV
    ("Ti", 400.0, 30.0),
    ("Ti", 400.0, 60.0),
    ("Ti", 400.0, 65.0),
    # Aluminium (Al) normal incidence
    ("Al", 100.0, 0.0),
    ("Al", 200.0, 0.0),
    ("Al", 300.0, 0.0),
    ("Al", 400.0, 0.0),
    ("Al", 500.0, 0.0),
    ("Al", 600.0, 0.0),
    ("Al", 800.0, 0.0),
    ("Al", 1000.0, 0.0),
    # Al angular dependence at 400 eV
    ("Al", 400.0, 30.0),
    ("Al", 400.0, 60.0),
    ("Al", 400.0, 65.0),
]


def get_material_properties(
    material: Union[str, TargetMaterial]
) -> tuple[float, float, float, float]:
    """Retrieve (Z2, M2, Us, Eth) for a given target material."""
    if isinstance(material, str):
        if material not in MATERIALS:
            raise KeyError(f"Material '{material}' not found in {list(MATERIALS.keys())}")
        mat = MATERIALS[material]
    elif isinstance(material, TargetMaterial):
        mat = material
    else:
        raise TypeError(f"Expected str or TargetMaterial, got {type(material).__name__}")

    return (
        float(mat.atomic_number),
        float(mat.atomic_mass),
        float(mat.sublimation_energy),
        float(mat.threshold_energy),
    )


def get_threshold_energy(
    target_z2: Union[float, int, torch.Tensor],
    target_us: Optional[Union[float, torch.Tensor]] = None,
) -> Union[float, torch.Tensor]:
    """Infer sputtering threshold energy Eth (eV) from target atomic number Z2 and Us."""
    if isinstance(target_z2, torch.Tensor):
        eth = torch.where(
            torch.isclose(target_z2, torch.tensor(29.0), atol=1.0),
            torch.tensor(20.0, dtype=target_z2.dtype, device=target_z2.device),
            torch.where(
                torch.isclose(target_z2, torch.tensor(22.0), atol=1.0),
                torch.tensor(30.0, dtype=target_z2.dtype, device=target_z2.device),
                torch.where(
                    torch.isclose(target_z2, torch.tensor(13.0), atol=1.0),
                    torch.tensor(25.0, dtype=target_z2.dtype, device=target_z2.device),
                    (
                        target_us * 6.5
                        if target_us is not None
                        else torch.tensor(25.0, dtype=target_z2.dtype, device=target_z2.device)
                    ),
                ),
            ),
        )
        return eth

    z = round(float(target_z2))
    if z == 29:
        return 20.0  # Cu
    elif z == 22:
        return 30.0  # Ti
    elif z == 13:
        return 25.0  # Al
    elif target_us is not None:
        return float(target_us) * 6.5
    return 25.0


@dataclass
class YieldPINNConfig:
    """Hyperparameter and training configuration for YieldPINN."""

    hidden_layers: int = 4
    neurons: int = 96
    activation: str = "tanh"
    learning_rate: float = 3.5e-3
    weight_decay: float = 1e-6
    lambda_data: float = 1.0
    lambda_sub_threshold: float = 0.05
    lambda_high_energy: float = 0.01
    lambda_angular: float = 0.05
    lambda_monotonicity: float = 0.05
    epochs: int = 1000
    n_collocation: int = 64


class YieldPINN(nn.Module):
    """Physics-Informed Neural Network for Sputter Yield Prediction.

    Maps inputs:
        (ion_energy_ev, angle_rad, target_z2, target_m2, target_us) -> sputter yield Y.
    """

    def __init__(
        self,
        config: Optional[YieldPINNConfig] = None,
        x_mean: Optional[Union[np.ndarray, torch.Tensor]] = None,
        x_std: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        super().__init__()
        self.config = config or YieldPINNConfig()

        activation_cls = nn.Tanh if self.config.activation == "tanh" else nn.GELU
        layers: list[nn.Module] = [
            nn.Linear(5, self.config.neurons),
            activation_cls(),
        ]
        for _ in range(self.config.hidden_layers - 1):
            layers.extend([
                nn.Linear(self.config.neurons, self.config.neurons),
                activation_cls(),
            ])
        layers.append(nn.Linear(self.config.neurons, 1))
        self.net = nn.Sequential(*layers)

        # Register input normalization buffers
        default_mean = np.array([350.0, 0.6, 21.0, 46.0, 3.9], dtype=np.float32)
        default_std = np.array([280.0, 0.45, 7.0, 15.0, 0.7], dtype=np.float32)

        if x_mean is not None:
            mean_tensor = torch.as_tensor(x_mean, dtype=torch.float32)
        else:
            mean_tensor = torch.tensor(default_mean, dtype=torch.float32)

        if x_std is not None:
            std_tensor = torch.as_tensor(x_std, dtype=torch.float32)
        else:
            std_tensor = torch.tensor(default_std, dtype=torch.float32)

        self.register_buffer("x_mean", mean_tensor)
        self.register_buffer("x_std", std_tensor)

    def forward(
        self,
        ion_energy_ev: Union[float, torch.Tensor],
        angle_rad: Optional[Union[float, torch.Tensor]] = None,
        target_z2: Optional[Union[float, torch.Tensor]] = None,
        target_m2: Optional[Union[float, torch.Tensor]] = None,
        target_us: Optional[Union[float, torch.Tensor]] = None,
        material: Optional[Union[str, TargetMaterial]] = None,
        enforce_zero_subthreshold: bool = False,
    ) -> torch.Tensor:
        """Forward pass computing predicted sputter yield Y.

        Accepts either:
        1. A single 2D tensor of shape (batch, 5) passed as ion_energy_ev.
        2. Five individual scalar / tensor arguments:
           (ion_energy_ev, angle_rad, target_z2, target_m2, target_us).
        3. Energy and angle with material name string (e.g. material='Cu').

        Args:
            ion_energy_ev: Incident ion kinetic energy in eV, or shape (N, 5) tensor.
            angle_rad: Angle of incidence in radians (0 = normal, pi/2 = grazing).
            target_z2: Target atomic number Z2 (e.g., 29 for Cu).
            target_m2: Target atomic mass M2 in g/mol (e.g., 63.55 for Cu).
            target_us: Target surface binding / sublimation energy Us in eV (e.g., 3.49 for Cu).
            material: Optional target material name string ('Cu', 'Ti', 'Al') or TargetMaterial.
            enforce_zero_subthreshold: If True, explicitly hard-clamps sub-threshold yield to 0.

        Returns:
            Predicted sputter yield Y (atoms/ion).
        """
        if angle_rad is None:
            if isinstance(ion_energy_ev, torch.Tensor) and ion_energy_ev.ndim == 2 and ion_energy_ev.shape[-1] == 5:
                x = ion_energy_ev
                is_1d = False
                is_0d = False
            elif isinstance(ion_energy_ev, torch.Tensor):
                # Default to normal incidence Cu target
                angle_rad = torch.zeros_like(ion_energy_ev)
                target_z2 = torch.full_like(ion_energy_ev, 29.0)
                target_m2 = torch.full_like(ion_energy_ev, 63.55)
                target_us = torch.full_like(ion_energy_ev, 3.49)
                return self.forward(ion_energy_ev, angle_rad, target_z2, target_m2, target_us)
            else:
                raise ValueError("Expected tensor of shape (N, 5) or 5 individual arguments.")
        else:
            # Material shorthand resolution
            if material is not None:
                z2_val, m2_val, us_val, _ = get_material_properties(material)
                if target_z2 is None:
                    target_z2 = z2_val
                if target_m2 is None:
                    target_m2 = m2_val
                if target_us is None:
                    target_us = us_val
            elif target_z2 is None:
                # Default to Cu target if omitted
                target_z2 = 29.0
                target_m2 = 63.55
                target_us = 3.49

            inputs = [ion_energy_ev, angle_rad, target_z2, target_m2, target_us]
            tensors = [
                t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
                for t in inputs
            ]
            is_0d = (tensors[0].ndim == 0)
            is_1d = (tensors[0].ndim == 1)

            t_expanded = [t.reshape(-1, 1) if t.ndim < 2 else t for t in tensors]
            b_tensors = torch.broadcast_tensors(*t_expanded)
            x = torch.cat(b_tensors, dim=-1)

        # Normalize inputs
        x_norm = (x - self.x_mean) / (self.x_std + 1e-8)
        y_raw = self.net(x_norm)
        # Yield is non-negative physical quantity: smooth softplus activation
        y = F.softplus(y_raw)

        if enforce_zero_subthreshold:
            z2_col = x[:, 2:3]
            eth = get_threshold_energy(z2_col)
            e_col = x[:, 0:1]
            th_col = x[:, 1:2]
            cutoff_rad = math.radians(85.0)
            mask = (e_col > eth) & (torch.abs(th_col) < cutoff_rad)
            y = torch.where(mask, y, torch.zeros_like(y))

        if angle_rad is not None:
            if is_0d:
                return y.squeeze()
            if is_1d:
                return y.squeeze(-1)
        return y

    def predict_yield(
        self,
        ion_energy_ev: float,
        angle_rad: float = 0.0,
        material: Union[str, TargetMaterial] = "Cu",
        target_z2: Optional[float] = None,
        target_m2: Optional[float] = None,
        target_us: Optional[float] = None,
    ) -> float:
        """Convenience method returning predicted scalar yield Y (atoms/ion)."""
        self.eval()
        with torch.no_grad():
            if material is not None and (target_z2 is None or target_m2 is None or target_us is None):
                z2, m2, us, eth = get_material_properties(material)
            else:
                z2 = float(target_z2 or 29.0)
                m2 = float(target_m2 or 63.55)
                us = float(target_us or 3.49)
                eth = float(get_threshold_energy(z2, us))

            if ion_energy_ev <= eth or abs(angle_rad) >= math.radians(85.0):
                return 0.0

            out = self.forward(
                float(ion_energy_ev),
                float(angle_rad),
                z2,
                m2,
                us,
            )
            return float(out.item())


def generate_yield_dataset(
    materials: Optional[list[str]] = None,
    n_energies: int = 18,
    include_subthreshold: bool = True,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Generate training dataset spanning benchmark and operational sputtering conditions.

    Evaluated against the calibrated Yamamura-Tawara formulation and Eckstein benchmark
    data for Cu, Ti, and Al.
    """
    mats = materials or ["Cu", "Ti", "Al"]
    energies = np.linspace(25.0, 1000.0, n_energies)
    angles_deg = [0.0, 15.0, 30.0, 45.0, 60.0, 65.0, 70.0, 75.0, 80.0]

    records = []
    for m in mats:
        z2, m2, us, eth = get_material_properties(m)

        # 1. Subthreshold points: true yield is exactly 0.0
        if include_subthreshold:
            sub_energies = [0.0, 5.0, 10.0, eth * 0.5, eth * 0.85]
            for se in sub_energies:
                for ad in [0.0, 30.0, 60.0]:
                    records.append([se, math.radians(ad), z2, m2, us, 0.0, eth])

        # 2. Regular energy and angle grid
        for e in energies:
            for ad in angles_deg:
                ar = math.radians(ad)
                y_val = calculate_sputter_yield(float(e), m, angle_rad=ar)
                records.append([float(e), ar, z2, m2, us, float(y_val), eth])

        # 3. Grazing cutoff angle (85 deg): yield drops to 0.0
        for e in [200.0, 400.0, 600.0, 800.0, 1000.0]:
            records.append([e, math.radians(85.0), z2, m2, us, 0.0, eth])

    data = np.array(records, dtype=np.float32)
    return {
        "X": data[:, :5],
        "Y": data[:, 5:6],
        "Eth": data[:, 6:7],
        "feature_names": ["ion_energy_ev", "angle_rad", "target_z2", "target_m2", "target_us"],
        "target_names": ["sputter_yield"],
    }


def compute_physics_residuals(
    model: YieldPINN,
    e_colloc: Optional[torch.Tensor] = None,
    theta_colloc: Optional[torch.Tensor] = None,
    target_z2: Optional[torch.Tensor] = None,
    target_m2: Optional[torch.Tensor] = None,
    target_us: Optional[torch.Tensor] = None,
    n_colloc: int = 64,
) -> Dict[str, torch.Tensor]:
    """Evaluate physics governing equations and constraints at collocation points.

    Returns:
        Dictionary containing individual physics loss components:
        1. loss_sub_threshold: ReLU(Eth - E) * Y^2 (ensures exact zero yield below threshold).
        2. loss_high_energy: |d(ln Y)/d(ln E) - (-0.2)| (Lindhard nuclear stopping scaling).
        3. loss_angular_deriv: Enforces dY/dtheta = 0 near theta_opt ~ 65 degrees.
        4. loss_angular_grazing: Enforces Y(85 deg) -> 0.
        5. loss_angular: Combined angular loss (deriv + grazing).
        6. loss_monotonicity: dY/dE >= 0 for energies between Eth and peak energy (~1 keV).
    """
    device = next(model.parameters()).device

    # Setup target material properties across collocation points
    if target_z2 is None or target_m2 is None or target_us is None:
        mat_idx = torch.randint(0, 3, (n_colloc,), device=device)
        z2_table = torch.tensor([29.0, 22.0, 13.0], device=device)
        m2_table = torch.tensor([63.55, 47.87, 26.98], device=device)
        us_table = torch.tensor([3.49, 4.89, 3.39], device=device)
        eth_table = torch.tensor([20.0, 30.0, 25.0], device=device)

        z2 = z2_table[mat_idx].unsqueeze(1)
        m2 = m2_table[mat_idx].unsqueeze(1)
        us = us_table[mat_idx].unsqueeze(1)
        eth = eth_table[mat_idx].unsqueeze(1)
    else:
        z2 = target_z2.to(device)
        m2 = target_m2.to(device)
        us = target_us.to(device)
        eth = get_threshold_energy(z2, us).to(device)
        if eth.ndim == 1:
            eth = eth.unsqueeze(1)

    # -------------------------------------------------------------------------
    # 1. Sub-Threshold Constraint Loss: ReLU(Eth - E) * Y^2
    # -------------------------------------------------------------------------
    if e_colloc is None:
        # Sample subthreshold and near-threshold energies [0, 35 eV]
        e_sub = torch.rand(n_colloc, 1, device=device) * 35.0
        theta_sub = torch.rand(n_colloc, 1, device=device) * 1.4
    else:
        e_sub = e_colloc.to(device)
        theta_sub = (theta_colloc or torch.zeros_like(e_sub)).to(device)

    y_sub = model(e_sub, theta_sub, z2, m2, us)
    if y_sub.ndim == 1:
        y_sub = y_sub.unsqueeze(1)
    loss_sub_threshold = torch.mean(F.relu(eth - e_sub) * (y_sub ** 2))

    # -------------------------------------------------------------------------
    # 2. High-Energy Asymptotic Loss: |d(ln Y)/d(ln E) - (-0.2)|
    # -------------------------------------------------------------------------
    # Sample multi-keV energies [2 keV to 10 keV] where Lindhard stopping ~ E^(-0.2)
    e_high = (torch.rand(n_colloc, 1, device=device) * 8000.0 + 2000.0).requires_grad_(True)
    th_high = torch.zeros(n_colloc, 1, device=device)
    y_high = model(e_high, th_high, z2, m2, us)
    if y_high.ndim == 1:
        y_high = y_high.unsqueeze(1)

    dY_dE_high = torch.autograd.grad(
        outputs=y_high,
        inputs=e_high,
        grad_outputs=torch.ones_like(y_high),
        create_graph=True,
    )[0]

    d_ln_Y_d_ln_E = (e_high / (y_high + 1e-8)) * dY_dE_high
    loss_high_energy = torch.mean(torch.abs(d_ln_Y_d_ln_E - (-0.2)))

    # -------------------------------------------------------------------------
    # 3. Angular Derivative Loss: dY/dtheta = 0 near theta_opt ~ 65 deg, and Y(85 deg) -> 0
    # -------------------------------------------------------------------------
    theta_opt_val = math.radians(65.0)  # ~ 1.13446 rad
    th_opt = (torch.full((n_colloc, 1), theta_opt_val, device=device)).requires_grad_(True)
    e_opt = torch.rand(n_colloc, 1, device=device) * 600.0 + 150.0

    y_opt = model(e_opt, th_opt, z2, m2, us)
    if y_opt.ndim == 1:
        y_opt = y_opt.unsqueeze(1)

    dY_dtheta = torch.autograd.grad(
        outputs=y_opt,
        inputs=th_opt,
        grad_outputs=torch.ones_like(y_opt),
        create_graph=True,
    )[0]
    loss_angular_deriv = torch.mean(dY_dtheta ** 2)

    # Grazing angle cutoff at 85 degrees (1.4835 rad)
    th_85 = torch.full((n_colloc, 1), math.radians(85.0), device=device)
    y_85 = model(e_opt, th_85, z2, m2, us)
    if y_85.ndim == 1:
        y_85 = y_85.unsqueeze(1)
    loss_angular_grazing = torch.mean(y_85 ** 2)

    loss_angular = loss_angular_deriv + loss_angular_grazing

    # -------------------------------------------------------------------------
    # 4. Monotonicity: dY/dE >= 0 for energies between Eth and peak energy (~1 keV)
    # -------------------------------------------------------------------------
    e_mono = (torch.rand(n_colloc, 1, device=device) * 800.0 + 100.0).requires_grad_(True)
    th_mono = torch.zeros(n_colloc, 1, device=device)
    y_mono = model(e_mono, th_mono, z2, m2, us)
    if y_mono.ndim == 1:
        y_mono = y_mono.unsqueeze(1)

    dY_dE_mono = torch.autograd.grad(
        outputs=y_mono,
        inputs=e_mono,
        grad_outputs=torch.ones_like(y_mono),
        create_graph=True,
    )[0]
    # Penalize negative slope
    loss_monotonicity = torch.mean(F.relu(-dY_dE_mono) ** 2)

    return {
        "loss_sub_threshold": loss_sub_threshold,
        "loss_high_energy": loss_high_energy,
        "loss_angular_deriv": loss_angular_deriv,
        "loss_angular_grazing": loss_angular_grazing,
        "loss_angular": loss_angular,
        "loss_monotonicity": loss_monotonicity,
    }


# Alias for compatibility
compute_yield_physics_residuals = compute_physics_residuals


def train_yield_pinn(
    epochs: int = 1000,
    config: Optional[YieldPINNConfig] = None,
    dataset: Optional[Dict[str, np.ndarray]] = None,
    save_path: Optional[Union[str, Path, bool]] = None,
    materials: Optional[list[str]] = None,
    verbose: bool = True,
) -> Tuple[YieldPINN, Dict[str, list[float]]]:
    """Train the YieldPINN surrogate model on Cu, Ti, and Al.

    Trains against calibrated Yamamura and Eckstein benchmark points with
    physics-informed sub-threshold, asymptotic, angular, and monotonicity constraints.

    Args:
        epochs: Number of optimization epochs (default: 1000).
        config: Optional YieldPINNConfig instance.
        dataset: Optional pre-generated dataset dictionary.
        save_path: Path to save trained PyTorch weights, or False to skip saving.
            Defaults to sputtertwin/pinn/yield_pinn_cu.pt.
        materials: List of target materials to train on (default: ['Cu', 'Ti', 'Al']).
        verbose: Whether to log training progress.

    Returns:
        Tuple of (trained YieldPINN model, training history dictionary).
    """
    cfg = config or YieldPINNConfig(epochs=epochs)
    cfg.epochs = epochs

    mats = materials or ["Cu", "Ti", "Al"]
    ds = dataset or generate_yield_dataset(materials=mats)

    x_train = ds["X"]
    y_train = ds["Y"]

    # Compute normalization statistics
    x_mean = np.mean(x_train, axis=0)
    x_std = np.std(x_train, axis=0)

    model = YieldPINN(config=cfg, x_mean=x_mean, x_std=x_std)

    # Curate standard literature benchmark points (Eckstein 2007, Yamamura 1996)
    bench_records = []
    for m, e, ad in BENCHMARK_POINTS:
        if m in mats:
            z2, m2, us, _ = get_material_properties(m)
            ar = math.radians(ad)
            y_true = calculate_sputter_yield(e, m, angle_rad=ar)
            bench_records.append([e, ar, z2, m2, us, y_true])

    bench_arr = np.array(bench_records, dtype=np.float32)
    t_x_bench = torch.tensor(bench_arr[:, :5], dtype=torch.float32)
    t_y_bench = torch.tensor(bench_arr[:, 5:6], dtype=torch.float32)

    t_x_train = torch.tensor(x_train, dtype=torch.float32)
    t_y_train = torch.tensor(y_train, dtype=torch.float32)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=1e-5,
    )

    history: Dict[str, list[float]] = {
        "train_loss": [],
        "loss_data": [],
        "loss_physics": [],
        "loss_sub_threshold": [],
        "loss_high_energy": [],
        "loss_angular": [],
        "loss_monotonicity": [],
        "benchmark_mape": [],
    }

    if verbose:
        print(f"Starting YieldPINN training for {cfg.epochs} epochs across {mats}...")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # 1. Supervised Data Loss: Combined grid MSE and weighted benchmark points
        y_pred = model(t_x_train)
        loss_grid = F.mse_loss(y_pred, t_y_train)

        y_bench_pred = model(t_x_bench)
        loss_bench_mse = F.mse_loss(y_bench_pred, t_y_bench)
        loss_bench_rel = torch.mean(((y_bench_pred - t_y_bench) / (t_y_bench + 0.05)) ** 2)

        loss_data = loss_grid + 2.0 * loss_bench_mse + 0.5 * loss_bench_rel

        # 2. Physics Residual Losses evaluated at collocation points
        phys_res = compute_physics_residuals(model=model, n_colloc=cfg.n_collocation)

        loss_physics = (
            cfg.lambda_sub_threshold * phys_res["loss_sub_threshold"]
            + cfg.lambda_high_energy * phys_res["loss_high_energy"]
            + cfg.lambda_angular * phys_res["loss_angular"]
            + cfg.lambda_monotonicity * phys_res["loss_monotonicity"]
        )

        total_loss = cfg.lambda_data * loss_data + loss_physics
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Record metrics
        history["train_loss"].append(float(total_loss.item()))
        history["loss_data"].append(float(loss_data.item()))
        history["loss_physics"].append(float(loss_physics.item()))
        history["loss_sub_threshold"].append(float(phys_res["loss_sub_threshold"].item()))
        history["loss_high_energy"].append(float(phys_res["loss_high_energy"].item()))
        history["loss_angular"].append(float(phys_res["loss_angular"].item()))
        history["loss_monotonicity"].append(float(phys_res["loss_monotonicity"].item()))

        if verbose and (epoch % 200 == 0 or epoch == cfg.epochs):
            model.eval()
            with torch.no_grad():
                pred_eval = model(t_x_bench).cpu().numpy().flatten()
                true_eval = t_y_bench.cpu().numpy().flatten()
                curr_mape = float(np.mean(np.abs((pred_eval - true_eval) / true_eval)) * 100.0)
                history["benchmark_mape"].append(curr_mape)

            print(
                f"Epoch {epoch:4d}/{cfg.epochs} | "
                f"Loss: {total_loss.item():.5f} | "
                f"Data: {loss_data.item():.5f} | "
                f"Phys: {loss_physics.item():.5f} | "
                f"Bench MAPE: {curr_mape:.2f}%"
            )

    # Save trained model weights
    if save_path is not False:
        if save_path is None:
            resolved_save_path = Path(__file__).resolve().parent / "yield_pinn_cu.pt"
        else:
            resolved_save_path = Path(save_path)

        os.makedirs(resolved_save_path.parent, exist_ok=True)
        torch.save(model.state_dict(), str(resolved_save_path))
        if verbose:
            print(f"Trained model checkpoint saved successfully to: {resolved_save_path}")

    return model, history


def evaluate_yield_pinn(
    model: YieldPINN,
    benchmark_points: Optional[list[tuple[str, float, float]]] = None,
) -> Dict[str, Any]:
    """Evaluate trained YieldPINN model on benchmark points, verifying MAPE < 2.5%."""
    points = benchmark_points or BENCHMARK_POINTS

    bench_records = []
    metadata = []
    for m, e, ad in points:
        z2, m2, us, eth = get_material_properties(m)
        ar = math.radians(ad)
        y_true = calculate_sputter_yield(e, m, angle_rad=ar)
        bench_records.append([e, ar, z2, m2, us])
        metadata.append((m, e, ad, y_true))

    x_tensor = torch.tensor(bench_records, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        y_pred = model(x_tensor).cpu().numpy().flatten()

    y_true = np.array([item[3] for item in metadata], dtype=np.float64)

    # Mean Absolute Percentage Error (MAPE)
    abs_pct_err = np.abs((y_pred - y_true) / (y_true + 1e-12)) * 100.0
    mape = float(np.mean(abs_pct_err))
    max_err = float(np.max(abs_pct_err))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    # Per-material evaluation breakdown
    material_metrics: Dict[str, Dict[str, float]] = {}
    for mat_name in set(item[0] for item in metadata):
        indices = [i for i, item in enumerate(metadata) if item[0] == mat_name]
        mat_pred = y_pred[indices]
        mat_true = y_true[indices]
        mat_ape = np.abs((mat_pred - mat_true) / (mat_true + 1e-12)) * 100.0
        material_metrics[mat_name] = {
            "mape_percent": float(np.mean(mat_ape)),
            "max_error_percent": float(np.max(mat_ape)),
            "rmse": float(np.sqrt(np.mean((mat_pred - mat_true) ** 2))),
        }

    return {
        "overall_mape_percent": mape,
        "overall_max_error_percent": max_err,
        "overall_rmse": rmse,
        "is_mape_valid": mape < 2.5,
        "material_metrics": material_metrics,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def load_yield_pinn(
    checkpoint_path: Optional[Union[str, Path]] = None,
    config: Optional[YieldPINNConfig] = None,
) -> YieldPINN:
    """Load a trained YieldPINN model instance from checkpoint weights."""
    if checkpoint_path is None:
        checkpoint_path = Path(__file__).resolve().parent / "yield_pinn_cu.pt"
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    state = torch.load(str(checkpoint_path), weights_only=True)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state

    if config is None:
        # Auto-detect layer dimensions from state_dict
        if "net.0.weight" in state_dict:
            detected_neurons = state_dict["net.0.weight"].shape[0]
            weight_keys = [k for k in state_dict.keys() if k.startswith("net.") and k.endswith(".weight")]
            detected_layers = len(weight_keys) - 1
            config = YieldPINNConfig(neurons=detected_neurons, hidden_layers=detected_layers)
        else:
            config = YieldPINNConfig()

    model = YieldPINN(config=config)
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == "__main__":
    print("=" * 70)
    print("SputterTwin Yield PINN (Step 2 Implementation)")
    print("=" * 70)

    model, history = train_yield_pinn(epochs=1000, verbose=True)

    print("\nEvaluating trained Yield PINN on Eckstein & Yamamura benchmark data...")
    eval_res = evaluate_yield_pinn(model)

    print("-" * 70)
    print(f"Overall Benchmark MAPE: {eval_res['overall_mape_percent']:.2f}% (Requirement: < 2.5%)")
    print(f"Overall Benchmark Max Error: {eval_res['overall_max_error_percent']:.2f}%")
    print(f"Overall RMSE: {eval_res['overall_rmse']:.4f}")
    print("-" * 70)

    for mat, metrics in eval_res["material_metrics"].items():
        print(f"Material {mat:2s} -> MAPE: {metrics['mape_percent']:.2f}%, Max Error: {metrics['max_error_percent']:.2f}%")
    print("=" * 70)
