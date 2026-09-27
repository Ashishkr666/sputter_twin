"""Numerical compatibility helpers shared across SputterTwin physics modules.

NumPy 2.0 promoted the trapezoidal integration routine from ``np.trapz`` to
``np.trapezoid`` (the old name was removed in the 2.0 release).  SputterTwin
supports ``numpy>=1.24``, so every module integrates through :data:`trapezoid`
instead of touching the NumPy namespace directly.
"""

from __future__ import annotations

import numpy as np

__all__ = ["trapezoid"]

#: Trapezoidal integration with the NumPy 2.x name, falling back to the legacy
#: alias on NumPy 1.x installations.
if hasattr(np, "trapezoid"):  # NumPy >= 2.0
    trapezoid = np.trapezoid
else:  # pragma: no cover - exercised only on NumPy 1.x
    trapezoid = np.trapz
