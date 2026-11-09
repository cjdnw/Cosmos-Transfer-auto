"""Quality score unit tests."""
from __future__ import annotations

from benchmark.scoring.quality_score import DEFAULT_WEIGHTS, compute_quality_score


def test_default_weights_sum_to_one():
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9


def test_weighted_sum():
    q, detail = compute_quality_score(control=1.0, temporal=0.0, novelty=0.0, mode="weighted_sum")
    assert abs(q - 0.45) < 1e-9
    assert detail["weights_note"]


def test_renormalize_missing():
    q, detail = compute_quality_score(
        control=1.0, temporal=None, novelty=None, mode="weighted_sum", missing_policy="renormalize"
    )
    assert abs(q - 1.0) < 1e-9
    assert abs(detail["weights_used"]["control"] - 1.0) < 1e-9


def test_multiplicative():
    q, _ = compute_quality_score(control=0.5, temporal=0.5, novelty=1.0, mode="multiplicative")
    assert abs(q - 0.25) < 1e-9


def test_hard_gate():
    q, detail = compute_quality_score(
        control=0.05,
        temporal=0.9,
        novelty=0.9,
        mode="hard_gate_then_weighted",
        hard_gate={"min_control": 0.15, "min_temporal": 0.15},
    )
    assert q is None
    assert detail.get("hard_gate_failed") == "control"
