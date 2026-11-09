"""Ablation helpers (stub tables for Pilot+)."""
from __future__ import annotations

from typing import Any, Dict, List


SCORING_COMPONENT_ABLATIONS = [
    "control_only",
    "temporal_only",
    "novelty_only",
    "control_temporal",
    "control_novelty",
    "temporal_novelty",
    "control_temporal_novelty",
]

SELECTION_STRATEGY_ABLATIONS = [
    "random",
    "topk_quality",
    "diversity_only",
    "quality_gate_diversity",
    "combined_score_diversity",
    "low_quality_negative_control",
]


def list_planned_ablations() -> Dict[str, List[str]]:
    return {
        "scoring_components": SCORING_COMPONENT_ABLATIONS,
        "selection_strategies": SELECTION_STRATEGY_ABLATIONS,
        "note": "No results fabricated in this PR.",
    }
