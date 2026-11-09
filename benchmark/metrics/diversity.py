"""Visual diversity / novelty features (frozen encoder or CPU pixel-PCA fallback)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .control import MetricResult


def _rgb_to_feature_matrix(rgb: np.ndarray, max_frames: int = 8) -> np.ndarray:
    """Flatten subsampled frames into a vector (smoke encoder)."""
    t = rgb.shape[0]
    idx = np.linspace(0, t - 1, num=min(max_frames, t)).astype(int)
    samples = rgb[idx].astype(np.float32) / 255.0
    # Downsample spatially
    try:
        import cv2  # type: ignore

        small = np.stack(
            [cv2.resize(f, (16, 16), interpolation=cv2.INTER_AREA) for f in samples],
            axis=0,
        )
    except ImportError:
        small = samples[:, :: max(1, samples.shape[1] // 16), :: max(1, samples.shape[2] // 16)]
    return small.reshape(-1)


class FeatureEncoder:
    """Configurable feature encoder with a frozen DINOv2 path and pixel_pca fallback."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self.requested = cfg.get("encoder", "pixel_pca")
        self.feature_dim = int(cfg.get("feature_dim", 64))
        dino_cfg = cfg.get("dinov2") or {}
        self.allow_fallback = bool(dino_cfg.get("allow_fallback", True))
        self.backend = "pixel_pca"
        self.status = "ok"
        self.detail: Dict[str, Any] = {}
        self._pca_components: Optional[np.ndarray] = None
        self._pca_mean: Optional[np.ndarray] = None
        self._dino = None

        if self.requested not in ("pixel_pca", "dinov2"):
            self.detail["warning"] = f"unknown encoder {self.requested!r}; using pixel_pca"
            self.requested = "pixel_pca"

        if self.requested == "dinov2":
            ok = self._try_init_dinov2(dino_cfg)
            if not ok:
                self.status = str(self.detail.get("reason") or "MISSING_DEP")
                self.detail["need"] = "torch + dinov2 weights"
                if self.allow_fallback:
                    self.backend = "pixel_pca"
                    self.detail["fallback"] = "pixel_pca"
                else:
                    self.backend = "unavailable"

    def _try_init_dinov2(self, dino_cfg: Dict[str, Any]) -> bool:
        from .dinov2_encoder import Dinov2LoadError, load_dinov2_encoder

        try:
            self._dino = load_dinov2_encoder(dino_cfg)
        except Dinov2LoadError as exc:
            self.detail["reason"] = exc.reason
            self.detail["error"] = str(exc)
            self.detail["attempts"] = exc.attempts
            return False
        self.backend = "dinov2"
        self.status = "ok"
        self.detail.update(self._dino.info)
        return True

    def fit_pca(self, rgb_list: Sequence[np.ndarray]) -> None:
        """Fit a simple PCA on stacked pixel features (train/dev pool only).

        DINOv2 keeps its native embedding. A PCA fit on a handful of videos would
        collapse that space, so this is a no-op unless the active backend is pixel_pca.
        """
        if self.backend != "pixel_pca":
            return
        mats = [_rgb_to_feature_matrix(rgb) for rgb in rgb_list]
        X = np.stack(mats, axis=0)
        mean = X.mean(axis=0, keepdims=True)
        Xc = X - mean
        # Economy SVD
        try:
            _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        except np.linalg.LinAlgError:
            vt = np.eye(Xc.shape[1])[: self.feature_dim]
        k = min(self.feature_dim, vt.shape[0])
        self._pca_mean = mean.ravel()
        self._pca_components = vt[:k]
        self.feature_dim = k

    def encode(self, rgb: np.ndarray) -> np.ndarray:
        if self.backend == "unavailable":
            raise RuntimeError(
                f"{self.status}: feature encoder unavailable ({self.detail.get('error', '')})"
            )
        if self.backend == "dinov2":
            if self._dino is None:
                raise RuntimeError("MISSING_WEIGHTS: DINOv2 encoder was not initialized")
            return self._dino.encode_video(rgb)
        vec = _rgb_to_feature_matrix(rgb)
        if self._pca_components is None:
            # Unfitted: deterministic hash-like projection via fixed RNG
            rng = np.random.RandomState(0)
            proj = rng.randn(self.feature_dim, vec.shape[0]).astype(np.float32)
            proj /= np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8
            feat = proj @ vec
        else:
            centered = vec - self._pca_mean
            feat = self._pca_components @ centered
        # L2 normalize
        feat = feat.astype(np.float32)
        feat /= np.linalg.norm(feat) + 1e-8
        return feat


def pairwise_cosine_similarity(features: np.ndarray) -> np.ndarray:
    """features: (N, D) L2-normalized → (N, N) cosine similarity."""
    f = features.astype(np.float64)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    return f @ f.T


def near_duplicate_rate(sim: np.ndarray, threshold: float = 0.98) -> float:
    n = sim.shape[0]
    if n < 2:
        return 0.0
    count = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if sim[i, j] >= threshold:
                count += 1
    return float(count) / float(total) if total else 0.0


def novelty_against_pool(
    feat: np.ndarray,
    pool: np.ndarray,
    *,
    metric: str = "min_cosine_distance",
) -> float:
    """
    Novelty vs a real-train (or candidate) feature pool.
    Returns value in [0,1] where higher = more novel (1 - max cosine sim).
    NOTE: farther ≠ more valuable; selection must still gate on quality.
    """
    if pool.size == 0:
        return 1.0
    f = feat / (np.linalg.norm(feat) + 1e-8)
    p = pool / (np.linalg.norm(pool, axis=1, keepdims=True) + 1e-8)
    sims = p @ f
    return float(1.0 - np.max(sims))


def coverage_score(selected_feats: np.ndarray, candidate_feats: np.ndarray) -> float:
    """
    Simple coverage: fraction of candidates whose nearest selected neighbor
    has cosine sim >= 0.5 (soft facility coverage proxy).
    """
    if selected_feats.size == 0 or candidate_feats.size == 0:
        return 0.0
    s = selected_feats / (np.linalg.norm(selected_feats, axis=1, keepdims=True) + 1e-8)
    c = candidate_feats / (np.linalg.norm(candidate_feats, axis=1, keepdims=True) + 1e-8)
    sim = c @ s.T  # (Nc, Ns)
    max_sim = sim.max(axis=1)
    return float((max_sim >= 0.5).mean())


def compute_diversity_metrics(
    features: np.ndarray,
    *,
    real_pool: Optional[np.ndarray] = None,
    near_dup_threshold: float = 0.98,
) -> Dict[str, MetricResult]:
    results: Dict[str, MetricResult] = {}
    if features.ndim != 2 or features.shape[0] == 0:
        results["near_duplicate_rate"] = MetricResult(
            "near_duplicate_rate", None, "missing_data"
        )
        return results
    sim = pairwise_cosine_similarity(features)
    nd = near_duplicate_rate(sim, threshold=near_dup_threshold)
    results["near_duplicate_rate"] = MetricResult(
        "near_duplicate_rate", nd, "ok", detail={"threshold": near_dup_threshold}
    )
    if real_pool is not None and real_pool.size:
        novelties = [
            novelty_against_pool(features[i], real_pool) for i in range(features.shape[0])
        ]
        results["mean_novelty_vs_real"] = MetricResult(
            "mean_novelty_vs_real", float(np.mean(novelties)), "ok"
        )
    return results
