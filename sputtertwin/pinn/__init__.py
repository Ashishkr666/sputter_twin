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

__all__ = [
    "PlasmaPINN",
    "PlasmaPINNConfig",
    "generate_plasma_dataset",
    "train_plasma_pinn",
    "evaluate_plasma_pinn",
]
