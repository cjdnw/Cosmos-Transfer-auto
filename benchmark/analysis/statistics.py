"""Statistics helpers (bootstrap stubs)."""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


def mean_std(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Any]:
    """Simple i.i.d. bootstrap CI. Prefer Episode/Scene resampling in Pilot."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"low": None, "high": None, "n": 0}
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(sample.mean())
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return {"low": low, "high": high, "n": int(arr.size), "n_boot": n_boot}
