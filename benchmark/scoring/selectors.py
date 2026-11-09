"""Two-stage selection: quality gate + diversity-aware pick of K."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from benchmark.metrics.diversity import coverage_score


def quality_gate(
    records: Sequence[Dict[str, Any]],
    *,
    min_control: float = 0.20,
    min_temporal: float = 0.20,
    min_quality: float = 0.25,
    reject_near_duplicates: bool = True,
    near_duplicate_cosine_threshold: float = 0.98,
    features: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Stage A: filter failing samples.
    Returns (passed, rejected_with_reason).
    """
    passed: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    # First pass: threshold gates
    provisional: List[Dict[str, Any]] = []
    provisional_idx: List[int] = []
    for i, rec in enumerate(records):
        reason = None
        c, t, q = rec.get("C"), rec.get("T"), rec.get("Q")
        if c is not None and c < min_control:
            reason = "low_control"
        elif t is not None and t < min_temporal:
            reason = "low_temporal"
        elif q is None or q < min_quality:
            reason = "low_quality"
        if reason:
            rejected.append({**rec, "reject_reason": reason})
        else:
            provisional.append(rec)
            provisional_idx.append(i)

    if not reject_near_duplicates or features is None or len(provisional) < 2:
        return provisional, rejected

    # Near-dup rejection: keep first of each near-duplicate cluster (stable order by Q desc)
    order = sorted(
        range(len(provisional)),
        key=lambda j: (provisional[j].get("Q") is not None, provisional[j].get("Q") or -1),
        reverse=True,
    )
    kept_local: List[int] = []
    kept_feats: List[np.ndarray] = []
    for j in order:
        global_i = provisional_idx[j]
        feat = features[global_i]
        is_dup = False
        for kf in kept_feats:
            sim = float(np.dot(feat, kf) / ((np.linalg.norm(feat) + 1e-8) * (np.linalg.norm(kf) + 1e-8)))
            if sim >= near_duplicate_cosine_threshold:
                is_dup = True
                break
        if is_dup:
            rejected.append({**provisional[j], "reject_reason": "near_duplicate"})
        else:
            kept_local.append(j)
            kept_feats.append(feat)

    kept_local_sorted = sorted(kept_local)  # restore original relative order among kept
    passed = [provisional[j] for j in kept_local_sorted]
    return passed, rejected


def select_random(records: Sequence[Dict[str, Any]], k: int, seed: int = 0) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(seed)
    n = len(records)
    if n == 0:
        return []
    k = min(k, n)
    idx = rng.choice(n, size=k, replace=False)
    idx = sorted(idx.tolist())
    return [records[i] for i in idx]


def select_topk_quality(records: Sequence[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    ranked = sorted(
        records,
        key=lambda r: (r.get("Q") is not None, r.get("Q") if r.get("Q") is not None else -1.0),
        reverse=True,
    )
    return ranked[: min(k, len(ranked))]


def select_diversity_only(
    records: Sequence[Dict[str, Any]],
    features: np.ndarray,
    k: int,
) -> List[Dict[str, Any]]:
    """Greedy farthest-first traversal in feature space."""
    n = len(records)
    if n == 0:
        return []
    k = min(k, n)
    # start from medoid-ish: highest mean distance? use index 0 of max norm for determinism: max quality then
    start = int(np.argmax([r.get("Q") or 0.0 for r in records])) if records else 0
    selected = [start]
    selected_set = {start}
    feats = features.astype(np.float64)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    while len(selected) < k:
        best_i, best_gain = None, -1.0
        for i in range(n):
            if i in selected_set:
                continue
            # min distance (1 - cosine) to selected
            sims = feats[selected] @ feats[i]
            gain = float(1.0 - sims.max())
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        selected_set.add(best_i)
    return [records[i] for i in selected]


def select_quality_gate_greedy_diversity(
    records: Sequence[Dict[str, Any]],
    features: np.ndarray,
    k: int,
    *,
    lambda_quality: float = 0.6,
    gate_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Stage A gate + Stage B greedy maximizing
      marginal gain ≈ λ * Q_i + (1-λ) * min_distance_to_S
    """
    gate_cfg = gate_cfg or {}
    passed, rejected = quality_gate(records, features=features, **{
        key: gate_cfg[key]
        for key in (
            "min_control",
            "min_temporal",
            "min_quality",
            "reject_near_duplicates",
            "near_duplicate_cosine_threshold",
        )
        if key in gate_cfg
    })

    # Map passed back to feature rows via uid
    uid_to_idx = {r.get("uid"): i for i, r in enumerate(records)}
    pass_idx = [uid_to_idx[r["uid"]] for r in passed if r.get("uid") in uid_to_idx]
    if not pass_idx:
        return [], {"rejected": rejected, "passed": 0, "strategy": "quality_gate_greedy_diversity"}

    sub_records = [records[i] for i in pass_idx]
    sub_feats = features[pass_idx]
    n = len(sub_records)
    k = min(k, n)
    feats = sub_feats / (np.linalg.norm(sub_feats, axis=1, keepdims=True) + 1e-8)

    # Start with highest Q
    start = int(np.argmax([r.get("Q") or 0.0 for r in sub_records]))
    selected = [start]
    selected_set = {start}

    while len(selected) < k:
        best_i, best_score = None, -1e9
        for i in range(n):
            if i in selected_set:
                continue
            q = float(sub_records[i].get("Q") or 0.0)
            sims = feats[selected] @ feats[i]
            dist = float(1.0 - sims.max())
            score = lambda_quality * q + (1.0 - lambda_quality) * dist
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        selected_set.add(best_i)

    chosen = [sub_records[i] for i in selected]
    cov = coverage_score(feats[selected], feats)
    meta = {
        "strategy": "quality_gate_greedy_diversity",
        "passed": len(passed),
        "rejected_count": len(rejected),
        "rejected": [{"uid": r.get("uid"), "reason": r.get("reject_reason")} for r in rejected],
        "lambda_quality": lambda_quality,
        "coverage": cov,
        "mean_quality": float(np.mean([c.get("Q") or 0.0 for c in chosen])) if chosen else None,
    }
    return chosen, meta


def select_facility_location_stub(
    records: Sequence[Dict[str, Any]],
    features: np.ndarray,
    k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Facility Location selection — STUB in this PR.
    Falls back to quality_gate_greedy_diversity for smoke continuity.
    """
    chosen, meta = select_quality_gate_greedy_diversity(
        records, features, k, lambda_quality=0.5, gate_cfg={"min_quality": 0.0, "min_control": 0.0, "min_temporal": 0.0}
    )
    meta["strategy"] = "facility_location_stub"
    meta["note"] = "Facility Location not fully implemented; greedy diversity fallback."
    return chosen, meta


def run_selection(
    records: Sequence[Dict[str, Any]],
    features: np.ndarray,
    *,
    strategy: str,
    k: int,
    seed: int = 0,
    lambda_quality: float = 0.6,
    gate_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    gate_cfg = gate_cfg or {}
    if strategy == "random":
        return select_random(records, k, seed=seed), {"strategy": "random", "seed": seed}
    if strategy == "topk_quality":
        return select_topk_quality(records, k), {"strategy": "topk_quality"}
    if strategy == "diversity_only":
        return select_diversity_only(records, features, k), {"strategy": "diversity_only"}
    if strategy == "facility_location":
        return select_facility_location_stub(records, features, k)
    if strategy in ("quality_gate_greedy_diversity", "combined_score_diversity"):
        return select_quality_gate_greedy_diversity(
            records, features, k, lambda_quality=lambda_quality, gate_cfg=gate_cfg
        )
    raise ValueError(f"Unknown selection strategy: {strategy}")
