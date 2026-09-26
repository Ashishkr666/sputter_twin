"""Physics-Informed Neural Networks (PINNs) package for SputterTwin.

Extends the analytical physics engine with deep surrogate models,
physics-constrained learning, and inverse problem solvers.
"""

from sputtertwin.pinn.plasma_pinn import (
    PlasmaPINN,
    PlasmaPINNConfig,
    generate_plasma_dataset,
    train_plasma_pinn,
    evaluate_plasma_pinn,
)
from sputtertwin.pinn.yield_pinn import (
    YieldPINN,
    YieldPINNConfig,
    generate_yield_dataset,
    compute_physics_residuals,
    compute_yield_physics_residuals,
    train_yield_pinn,
    evaluate_yield_pinn,
    load_yield_pinn,
    BENCHMARK_POINTS,
)

__all__ = [
    "PlasmaPINN",
    "PlasmaPINNConfig",
    "generate_plasma_dataset",
    "train_plasma_pinn",
    "evaluate_plasma_pinn",
    "YieldPINN",
    "YieldPINNConfig",
    "generate_yield_dataset",
    "compute_physics_residuals",
    "compute_yield_physics_residuals",
    "train_yield_pinn",
    "evaluate_yield_pinn",
    "load_yield_pinn",
    "BENCHMARK_POINTS",
]
