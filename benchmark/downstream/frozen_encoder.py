"""Frozen vision encoder wrapper (DINOv2 optional; smoke uses pixel features)."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class FrozenEncoder:
    """
    Validation-only encoder. Does NOT participate in quality score learning.
    Torch/DINOv2 path returns MISSING_DEP on this CPU box unless extras installed.
    """

    def __init__(self, name: str = "dinov2_vits14"):
        self.name = name
        self.backend = "unavailable"
        self.status = "MISSING_DEP"
        try:
            import torch  # noqa: F401

            self.status = "MISSING_WEIGHTS"
            self.detail = "torch present but DINOv2 weights not loaded in this PR"
        except ImportError:
            self.detail = "torch not installed"

    def encode(self, rgb: np.ndarray) -> np.ndarray:
        raise RuntimeError(f"MISSING_DEP: FrozenEncoder unavailable ({self.detail})")

    def info(self) -> Dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": getattr(self, "detail", None)}
