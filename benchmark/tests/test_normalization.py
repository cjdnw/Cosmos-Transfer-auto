"""Tests for score normalization fit/freeze behavior."""
from __future__ import annotations

from benchmark.scoring.normalization import apply_normalization, fit_normalization


def test_minmax_fit_and_apply():
    records = [
        {"control": 0.0, "temporal": 10.0},
        {"control": 1.0, "temporal": 20.0},
        {"control": 0.5, "temporal": None},
    ]
    stats = fit_normalization(records, ["control", "temporal"], method="minmax")
    assert stats.frozen is True
    assert stats.mins["control"] == 0.0
    assert stats.maxs["control"] == 1.0
    assert apply_normalization(0.5, "control", stats) == 0.5
    assert apply_normalization(10.0, "temporal", stats) == 0.0
    assert apply_normalization(20.0, "temporal", stats) == 1.0
    assert apply_normalization(None, "control", stats) is None


def test_zscore_clip():
    records = [{"control": float(x)} for x in range(10)]
    stats = fit_normalization(records, ["control"], method="zscore_clip", clip_z=2.0)
    mid = apply_normalization(stats.means["control"], "control", stats)
    assert 0.4 <= mid <= 0.6
