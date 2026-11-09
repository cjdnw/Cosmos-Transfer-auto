"""Selector determinism tests."""
from __future__ import annotations

import numpy as np

from benchmark.scoring.selectors import (
    quality_gate,
    run_selection,
    select_random,
    select_topk_quality,
)


def _records(n=6):
    return [
        {"uid": f"u{i}", "C": 0.9 - 0.1 * i, "T": 0.8, "Q": 0.9 - 0.1 * i, "N": 0.5}
        for i in range(n)
    ]


def test_random_deterministic():
    recs = _records()
    a = select_random(recs, k=3, seed=123)
    b = select_random(recs, k=3, seed=123)
    assert [r["uid"] for r in a] == [r["uid"] for r in b]
    c = select_random(recs, k=3, seed=999)
    assert [r["uid"] for r in a] != [r["uid"] for r in c] or len(recs) < 3


def test_topk_quality_order():
    recs = _records()
    top = select_topk_quality(recs, k=2)
    assert top[0]["uid"] == "u0"
    assert top[1]["uid"] == "u1"


def test_quality_gate_rejects_low():
    recs = _records()
    recs[0]["C"] = 0.01
    recs[0]["Q"] = 0.01
    feats = np.eye(len(recs), dtype=np.float32)
    passed, rejected = quality_gate(
        recs, min_control=0.2, min_temporal=0.2, min_quality=0.25, features=feats
    )
    assert all(r["uid"] != "u0" for r in passed)
    assert any(r.get("reject_reason") == "low_control" for r in rejected)


def test_greedy_diversity_deterministic():
    recs = _records()
    rng = np.random.RandomState(0)
    feats = rng.randn(len(recs), 8).astype(np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    a, meta_a = run_selection(
        recs,
        feats,
        strategy="quality_gate_greedy_diversity",
        k=3,
        lambda_quality=0.6,
        gate_cfg={"min_control": 0.0, "min_temporal": 0.0, "min_quality": 0.0},
    )
    b, meta_b = run_selection(
        recs,
        feats,
        strategy="quality_gate_greedy_diversity",
        k=3,
        lambda_quality=0.6,
        gate_cfg={"min_control": 0.0, "min_temporal": 0.0, "min_quality": 0.0},
    )
    assert [r["uid"] for r in a] == [r["uid"] for r in b]
    assert meta_a["strategy"] == "quality_gate_greedy_diversity"
