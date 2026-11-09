"""Frozen vision encoder wrapper. DINOv2 loads when torch and weights are available.

This module does not train. Downstream head training stays a separate stub.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from benchmark.metrics.dinov2_encoder import Dinov2LoadError, load_dinov2_encoder


class FrozenEncoder:
    """
    Validation-only encoder. Does not participate in quality-score learning.
    """

    def __init__(self, name: str = "dinov2_vits14", **dino_cfg: Any):
        self.name = name
        cfg = {"model_name": name, **dino_cfg}
        self._encoder = None
        self.error: Optional[str] = None
        try:
            self._encoder = load_dinov2_encoder(cfg)
        except Dinov2LoadError as exc:
            self.backend = "unavailable"
            self.status = exc.reason
            self.error = str(exc)
            self.detail: Dict[str, Any] = {"error": str(exc), "attempts": exc.attempts}
            return
        self.backend = "dinov2"
        self.status = "ok"
        self.detail = dict(self._encoder.info)

    def encode(self, rgb: np.ndarray) -> np.ndarray:
        if self._encoder is None:
            raise RuntimeError(f"{self.status}: FrozenEncoder unavailable ({self.error})")
        return self._encoder.encode_video(rgb)

    def info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "status": self.status,
            "detail": self.detail,
        }
