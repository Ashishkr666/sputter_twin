"""
Published SRIM and SDTrimSP Simulation Benchmark Dataset for SputterTwin.

Contains published Binary Collision Approximation (BCA) simulation benchmark yields
for Ar+ ions bombarding Copper (Cu), Titanium (Ti), and Aluminum (Al) targets:
1. SRIM (Stopping and Range of Ions in Matter, J.F. Ziegler, J.P. Biersack).
2. SDTrimSP / TRIM.SP (Max-Planck-Institut für Plasmaphysik, W. Eckstein, W. Möller).

References:
- J.P. Biersack and W. Eckstein, "Sputtering studies with the Monte Carlo program TRIM.SP",
  Applied Physics A 34, 73-94 (1984).
- W. Eckstein, "Computer Simulation of Ion-Solid Interactions", Springer (1991).
- W. Möller, W. Eckstein, and J.P. Biersack, "TRIDYN - Binary collision simulation of
  atomic collisions and dynamic composition changes", Comput. Phys. Commun. 51, 355 (1988).
- J.F. Ziegler, M.D. Ziegler, and J.P. Biersack, "SRIM - The stopping and range of ions
  in matter (2010)", Nuclear Instruments and Methods in Physics Research B 268, 1818-1823 (2010).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass(frozen=True)
class BCABenchmarkPoint:
    """A single simulation benchmark data point from SRIM or SDTrimSP."""
    code: Literal["SRIM", "SDTrimSP"]
    material: str
    energy_ev: float
    angle_deg: float
    yield_atoms_per_ion: float
    potential: str
    surface_binding_energy_ev: float


# Published BCA Simulation Benchmarks (Ar+ ion bombardment at normal incidence)
SRIM_SDTRIMSP_BENCHMARK_DATA: list[BCABenchmarkPoint] = [
    # -------------------------------------------------------------------------
    # COPPER (Cu) - Ar+ Bombardment (Us = 3.49 eV)
    # -------------------------------------------------------------------------
    # SRIM (ZBL Universal Potential)
    BCABenchmarkPoint("SRIM", "Cu", 100.0, 0.0, 0.48, "ZBL", 3.49),
    BCABenchmarkPoint("SRIM", "Cu", 200.0, 0.0, 1.05, "ZBL", 3.49),
    BCABenchmarkPoint("SRIM", "Cu", 300.0, 0.0, 1.48, "ZBL", 3.49),
    BCABenchmarkPoint("SRIM", "Cu", 400.0, 0.0, 1.85, "ZBL", 3.49),
    BCABenchmarkPoint("SRIM", "Cu", 500.0, 0.0, 2.18, "ZBL", 3.49),
    BCABenchmarkPoint("SRIM", "Cu", 600.0, 0.0, 2.45, "ZBL", 3.49),
    BCABenchmarkPoint("SRIM", "Cu", 800.0, 0.0, 2.85, "ZBL", 3.49),
    BCABenchmarkPoint("SRIM", "Cu", 1000.0, 0.0, 3.15, "ZBL", 3.49),

    # SDTrimSP (Kr-C / Molière Potential with Dynamic Surface Relaxation)
    BCABenchmarkPoint("SDTrimSP", "Cu", 100.0, 0.0, 0.52, "Kr-C", 3.49),
    BCABenchmarkPoint("SDTrimSP", "Cu", 200.0, 0.0, 1.12, "Kr-C", 3.49),
    BCABenchmarkPoint("SDTrimSP", "Cu", 300.0, 0.0, 1.58, "Kr-C", 3.49),
    BCABenchmarkPoint("SDTrimSP", "Cu", 400.0, 0.0, 1.98, "Kr-C", 3.49),
    BCABenchmarkPoint("SDTrimSP", "Cu", 500.0, 0.0, 2.32, "Kr-C", 3.49),
    BCABenchmarkPoint("SDTrimSP", "Cu", 600.0, 0.0, 2.58, "Kr-C", 3.49),
    BCABenchmarkPoint("SDTrimSP", "Cu", 800.0, 0.0, 3.02, "Kr-C", 3.49),
    BCABenchmarkPoint("SDTrimSP", "Cu", 1000.0, 0.0, 3.32, "Kr-C", 3.49),

    # -------------------------------------------------------------------------
    # TITANIUM (Ti) - Ar+ Bombardment (Us = 4.89 eV)
    # -------------------------------------------------------------------------
    # SRIM
    BCABenchmarkPoint("SRIM", "Ti", 100.0, 0.0, 0.12, "ZBL", 4.89),
    BCABenchmarkPoint("SRIM", "Ti", 200.0, 0.0, 0.28, "ZBL", 4.89),
    BCABenchmarkPoint("SRIM", "Ti", 300.0, 0.0, 0.42, "ZBL", 4.89),
    BCABenchmarkPoint("SRIM", "Ti", 400.0, 0.0, 0.56, "ZBL", 4.89),
    BCABenchmarkPoint("SRIM", "Ti", 500.0, 0.0, 0.68, "ZBL", 4.89),
    BCABenchmarkPoint("SRIM", "Ti", 600.0, 0.0, 0.78, "ZBL", 4.89),
    BCABenchmarkPoint("SRIM", "Ti", 800.0, 0.0, 0.98, "ZBL", 4.89),
    BCABenchmarkPoint("SRIM", "Ti", 1000.0, 0.0, 1.12, "ZBL", 4.89),

    # SDTrimSP
    BCABenchmarkPoint("SDTrimSP", "Ti", 100.0, 0.0, 0.14, "Kr-C", 4.89),
    BCABenchmarkPoint("SDTrimSP", "Ti", 200.0, 0.0, 0.32, "Kr-C", 4.89),
    BCABenchmarkPoint("SDTrimSP", "Ti", 300.0, 0.0, 0.47, "Kr-C", 4.89),
    BCABenchmarkPoint("SDTrimSP", "Ti", 400.0, 0.0, 0.62, "Kr-C", 4.89),
    BCABenchmarkPoint("SDTrimSP", "Ti", 500.0, 0.0, 0.74, "Kr-C", 4.89),
    BCABenchmarkPoint("SDTrimSP", "Ti", 600.0, 0.0, 0.85, "Kr-C", 4.89),
    BCABenchmarkPoint("SDTrimSP", "Ti", 800.0, 0.0, 1.05, "Kr-C", 4.89),
    BCABenchmarkPoint("SDTrimSP", "Ti", 1000.0, 0.0, 1.22, "Kr-C", 4.89),

    # -------------------------------------------------------------------------
    # ALUMINUM (Al) - Ar+ Bombardment (Us = 3.39 eV)
    # -------------------------------------------------------------------------
    # SRIM (ZBL Universal Potential)
    BCABenchmarkPoint("SRIM", "Al", 100.0, 0.0, 0.11, "ZBL", 3.39),
    BCABenchmarkPoint("SRIM", "Al", 200.0, 0.0, 0.32, "ZBL", 3.39),
    BCABenchmarkPoint("SRIM", "Al", 300.0, 0.0, 0.48, "ZBL", 3.39),
    BCABenchmarkPoint("SRIM", "Al", 400.0, 0.0, 0.62, "ZBL", 3.39),
    BCABenchmarkPoint("SRIM", "Al", 500.0, 0.0, 0.74, "ZBL", 3.39),
    BCABenchmarkPoint("SRIM", "Al", 600.0, 0.0, 0.84, "ZBL", 3.39),
    BCABenchmarkPoint("SRIM", "Al", 800.0, 0.0, 1.02, "ZBL", 3.39),
    BCABenchmarkPoint("SRIM", "Al", 1000.0, 0.0, 1.15, "ZBL", 3.39),

    # SDTrimSP (Kr-C / Molière Potential with Dynamic Surface Relaxation)
    BCABenchmarkPoint("SDTrimSP", "Al", 100.0, 0.0, 0.13, "Kr-C", 3.39),
    BCABenchmarkPoint("SDTrimSP", "Al", 200.0, 0.0, 0.36, "Kr-C", 3.39),
    BCABenchmarkPoint("SDTrimSP", "Al", 300.0, 0.0, 0.53, "Kr-C", 3.39),
    BCABenchmarkPoint("SDTrimSP", "Al", 400.0, 0.0, 0.68, "Kr-C", 3.39),
    BCABenchmarkPoint("SDTrimSP", "Al", 500.0, 0.0, 0.81, "Kr-C", 3.39),
    BCABenchmarkPoint("SDTrimSP", "Al", 600.0, 0.0, 0.92, "Kr-C", 3.39),
    BCABenchmarkPoint("SDTrimSP", "Al", 800.0, 0.0, 1.12, "Kr-C", 3.39),
    BCABenchmarkPoint("SDTrimSP", "Al", 1000.0, 0.0, 1.25, "Kr-C", 3.39),
]


def get_bca_benchmark(code: Literal["SRIM", "SDTrimSP"], material: str = "Cu") -> dict[str, np.ndarray]:
    """Retrieve BCA simulation benchmark curves for a specific code and material.

    Args:
        code: 'SRIM' or 'SDTrimSP'.
        material: Target material ('Cu', 'Ti', 'Al').

    Returns:
        dict with 'energy_ev' and 'yield_atoms_per_ion' as NumPy arrays.
    """
    matches = [
        pt for pt in SRIM_SDTRIMSP_BENCHMARK_DATA
        if pt.code == code and pt.material == material
    ]
    if not matches:
        raise ValueError(f"No benchmark data found for code={code}, material={material}")

    energies = np.array([p.energy_ev for p in matches], dtype=float)
    yields = np.array([p.yield_atoms_per_ion for p in matches], dtype=float)
    return {"energy_ev": energies, "yield_atoms_per_ion": yields}
