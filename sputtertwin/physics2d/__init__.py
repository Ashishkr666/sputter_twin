"""2D Multiphysics Simulation Package for SputterTwin.

Implements 2D axisymmetric (r, z) field solvers for planar DC magnetron sputtering:
1. Magnetic Field (B_r, B_z, A_theta, E x B racetrack location)
2. 2D Drift-Diffusion Plasma Discharge & Sheath Potential V(r, z)
3. 2D Gas Rarefaction / Sputter Wind Thermal Fluid Model
4. 2D Field-Coupled Thin-Film Wafer Deposition Profile
"""

from sputtertwin.physics2d.magnetic_field import (
    MagnetronGeometry,
    MagneticField2D,
    compute_magnetron_magnetic_field,
)
from sputtertwin.physics2d.plasma2d import (
    Plasma2DResult,
    simulate_plasma_2d,
)
from sputtertwin.physics2d.rarefaction2d import (
    GasRarefaction2D,
    compute_gas_rarefaction_2d,
)

__all__ = [
    "MagnetronGeometry",
    "MagneticField2D",
    "compute_magnetron_magnetic_field",
    "Plasma2DResult",
    "simulate_plasma_2d",
    "GasRarefaction2D",
    "compute_gas_rarefaction_2d",
]
