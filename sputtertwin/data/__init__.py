"""Experimental benchmark datasets for SputterTwin.

Includes published experimental sputter yield benchmarks from Wolfgang Eckstein
(Max-Planck-Institut für Plasmaphysik, Garching) and Yamamura & Tawara (1996).
"""

from sputtertwin.data.eckstein_sputter_yield_data import (
    ECKSTEIN_YIELD_DATA,
    ECKSTEIN_REFERENCES,
    SputterYieldPoint,
    get_eckstein_data,
)
from sputtertwin.data.srim_sdtrimsp_data import (
    SRIM_SDTRIMSP_BENCHMARK_DATA,
    BCABenchmarkPoint,
    get_bca_benchmark,
)

__all__ = [
    "ECKSTEIN_YIELD_DATA",
    "ECKSTEIN_REFERENCES",
    "SputterYieldPoint",
    "get_eckstein_data",
    "SRIM_SDTRIMSP_BENCHMARK_DATA",
    "BCABenchmarkPoint",
    "get_bca_benchmark",
]

