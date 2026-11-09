"""Normalization fit on train/dev pool only; freeze before test."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np


@dataclass
class NormStats:
    method: str
    mins: Dict[str, float]
    maxs: Dict[str, float]
    means: Dict[str, float]
    stds: Dict[str, float]
    clip_z: float = 3.0
    eps: float = 1e-8
    frozen: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "NormStats":
        return cls(
            method=d["method"],
            mins={k: float(v) for k, v in d["mins"].items()},
            maxs={k: float(v) for k, v in d["maxs"].items()},
            means={k: float(v) for k, v in d["means"].items()},
            stds={k: float(v) for k, v in d["stds"].items()},
            clip_z=float(d.get("clip_z", 3.0)),
            eps=float(d.get("eps", 1e-8)),
            frozen=bool(d.get("frozen", False)),
        )


def fit_normalization(
    records: Iterable[Dict[str, Optional[float]]],
    keys: List[str],
    *,
    method: str = "minmax",
    clip_z: float = 3.0,
    eps: float = 1e-8,
) -> NormStats:
    """
    Fit normalization statistics on a train/dev pool.

    `records` is an iterable of dicts mapping metric key → float or None.
    """
    buckets: Dict[str, List[float]] = {k: [] for k in keys}
    for rec in records:
        for k in keys:
            v = rec.get(k)
            if v is None:
                continue
            if isinstance(v, float) and not np.isfinite(v):
                continue
            buckets[k].append(float(v))

    mins, maxs, means, stds = {}, {}, {}, {}
    for k in keys:
        vals = buckets[k]
        if not vals:
            mins[k] = 0.0
            maxs[k] = 1.0
            means[k] = 0.0
            stds[k] = 1.0
            continue
        arr = np.asarray(vals, dtype=np.float64)
        mins[k] = float(arr.min())
        maxs[k] = float(arr.max())
        means[k] = float(arr.mean())
        stds[k] = float(arr.std()) if arr.std() > eps else 1.0

    return NormStats(
        method=method,
        mins=mins,
        maxs=maxs,
        means=means,
        stds=stds,
        clip_z=clip_z,
        eps=eps,
        frozen=True,
    )


def apply_normalization(value: Optional[float], key: str, stats: NormStats) -> Optional[float]:
    """Map a raw value to approximately [0,1] using frozen stats."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    v = float(value)
    if stats.method == "minmax":
        lo = stats.mins.get(key, 0.0)
        hi = stats.maxs.get(key, 1.0)
        if abs(hi - lo) < stats.eps:
            return 0.5
        out = (v - lo) / (hi - lo)
        return float(np.clip(out, 0.0, 1.0))
    if stats.method == "zscore_clip":
        mu = stats.means.get(key, 0.0)
        sd = stats.stds.get(key, 1.0)
        z = (v - mu) / (sd + stats.eps)
        z = float(np.clip(z, -stats.clip_z, stats.clip_z))
        # map clipped z to [0,1]
        return float((z + stats.clip_z) / (2 * stats.clip_z))
    raise ValueError(f"Unknown normalization method: {stats.method}")
