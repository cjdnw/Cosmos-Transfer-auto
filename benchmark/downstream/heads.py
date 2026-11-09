"""Lightweight heads for downstream validation (stubs)."""
from __future__ import annotations

from typing import Any, Dict, Optional


class LinearSegmentationHead:
    """Stub: linear projection over frozen features → per-pixel/class logits."""

    def __init__(self, in_dim: int, num_classes: int):
        self.in_dim = in_dim
        self.num_classes = num_classes

    def info(self) -> Dict[str, Any]:
        return {
            "type": "linear_segmentation",
            "in_dim": self.in_dim,
            "num_classes": self.num_classes,
            "status": "stub",
        }


class ClassificationHead:
    def __init__(self, in_dim: int, num_classes: int):
        self.in_dim = in_dim
        self.num_classes = num_classes

    def info(self) -> Dict[str, Any]:
        return {"type": "classification", "in_dim": self.in_dim, "num_classes": self.num_classes, "status": "stub"}
