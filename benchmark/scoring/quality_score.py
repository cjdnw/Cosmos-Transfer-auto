"""Training-free quality score Q_i = w_c C_i + w_t T_i + w_n N_i."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .normalization import NormStats, apply_normalization


DEFAULT_WEIGHTS = {
    "control": 0.45,   # hypothesis only
    "temporal": 0.35,  # hypothesis only
    "novelty": 0.20,   # hypothesis only
}


def _renormalize_weights(
    weights: Dict[str, float],
    present: Dict[str, bool],
) -> Dict[str, float]:
    active = {k: w for k, w in weights.items() if present.get(k, False) and w > 0}
    s = sum(active.values())
    if s <= 0:
        return {k: 0.0 for k in weights}
    return {k: (active[k] / s if k in active else 0.0) for k in weights}


def compute_quality_score(
    *,
    control: Optional[float],
    temporal: Optional[float],
    novelty: Optional[float],
    weights: Optional[Dict[str, float]] = None,
    mode: str = "weighted_sum",
    missing_policy: str = "renormalize",
    hard_gate: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Compute Q_i. Sub-scores expected in [0,1], higher better.

    Modes:
      - weighted_sum / uniform
      - multiplicative: C * T * N (missing → skip that factor)
      - hard_gate_then_weighted: return None if gates fail, else weighted_sum
    """
    weights = dict(weights or DEFAULT_WEIGHTS)
    if mode == "uniform":
        keys = ["control", "temporal", "novelty"]
        weights = {k: 1.0 / len(keys) for k in keys}

    present = {
        "control": control is not None,
        "temporal": temporal is not None,
        "novelty": novelty is not None,
    }
    vals = {"control": control, "temporal": temporal, "novelty": novelty}

    detail: Dict[str, Any] = {
        "mode": mode,
        "weights_requested": weights,
        "present": present,
        "weights_note": "Defaults are hypotheses only; not tuned on real test.",
    }

    if hard_gate or mode == "hard_gate_then_weighted":
        hg = hard_gate or {}
        if present["control"] and control < float(hg.get("min_control", 0.0)):
            return None, {**detail, "hard_gate_failed": "control"}
        if present["temporal"] and temporal < float(hg.get("min_temporal", 0.0)):
            return None, {**detail, "hard_gate_failed": "temporal"}

    if mode == "multiplicative":
        factors = [vals[k] for k in ("control", "temporal", "novelty") if present[k]]
        if not factors:
            return None, {**detail, "reason": "all_missing"}
        q = float(np.prod(factors))
        detail["weights_used"] = None
        return q, detail

    # weighted_sum / uniform / hard_gate_then_weighted
    if missing_policy == "renormalize":
        w_used = _renormalize_weights(weights, present)
    else:
        w_used = weights
    detail["weights_used"] = w_used

    if not any(present.values()):
        return None, {**detail, "reason": "all_missing"}

    q = 0.0
    for k, w in w_used.items():
        if present[k]:
            q += w * float(vals[k])
    return float(q), detail


def score_records(
    records: List[Dict[str, Any]],
    *,
    scoring_cfg: Dict[str, Any],
    norm_stats: Optional[NormStats] = None,
    raw_keys: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Attach quality scores to records in-place (returns same list).

    raw_keys maps logical name → record field for raw values before norm.
    If norm_stats given, apply normalization first.
    """
    raw_keys = raw_keys or {
        "control": "C_raw",
        "temporal": "T_raw",
        "novelty": "N_raw",
    }
    weights = scoring_cfg.get("weights", DEFAULT_WEIGHTS)
    mode = scoring_cfg.get("mode", "weighted_sum")
    missing_policy = scoring_cfg.get("normalization", {}).get("missing_policy", "renormalize")
    hard_gate = scoring_cfg.get("hard_gate") if "hard_gate" in mode or scoring_cfg.get("hard_gate") else None

    for rec in records:
        c = rec.get(raw_keys["control"])
        t = rec.get(raw_keys["temporal"])
        n = rec.get(raw_keys["novelty"])
        if norm_stats is not None:
            c = apply_normalization(c, "control", norm_stats) if c is not None else None
            # For temporal/novelty we store already-[0,1] aggregates usually;
            # still allow norm if raw keys differ.
            t = apply_normalization(t, "temporal", norm_stats) if t is not None else None
            n = apply_normalization(n, "novelty", norm_stats) if n is not None else None
        # If aggregates already in [0,1], skip forcing norm — values passed through.
        # Re-read from rec if we want direct C_i/T_i/N_i fields:
        c = rec.get("C", c)
        t = rec.get("T", t)
        n = rec.get("N", n)

        q, detail = compute_quality_score(
            control=c,
            temporal=t,
            novelty=n,
            weights=weights,
            mode=mode,
            missing_policy=missing_policy,
            hard_gate=scoring_cfg.get("hard_gate") if mode == "hard_gate_then_weighted" else None,
        )
        rec["C"] = c
        rec["T"] = t
        rec["N"] = n
        rec["Q"] = q
        rec["Q_detail"] = detail
    return records
