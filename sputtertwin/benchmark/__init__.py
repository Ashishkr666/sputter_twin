"""COMSOL Multiphysics benchmark automation package for SputterTwin.

Bridges SputterTwin physics and PINN surrogates directly with local COMSOL
installations via the MPh API and batch execution engines.
"""

from sputtertwin.benchmark.comsol_runner import (
    ComsolBenchmark,
    check_comsol_installation,
)

__all__ = [
    "ComsolBenchmark",
    "check_comsol_installation",
]
