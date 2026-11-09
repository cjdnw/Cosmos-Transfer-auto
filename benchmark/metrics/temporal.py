"""Temporal consistency metrics (CPU-friendly with optical-flow fallback)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .control import MetricResult


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    return (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    )


def _farneback_flow(prev_gray: np.ndarray, next_gray: np.ndarray) -> Optional[np.ndarray]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    p = np.clip(prev_gray, 0, 255).astype(np.uint8)
    n = np.clip(next_gray, 0, 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(
        p, n, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow  # H,W,2


def _warp_back(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp `frame` (next) back to previous coords using forward flow prev→next."""
    try:
        import cv2  # type: ignore

        h, w = flow.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        if frame.ndim == 2:
            return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        chans = [
            cv2.remap(frame[..., c], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            for c in range(frame.shape[2])
        ]
        return np.stack(chans, axis=-1)
    except ImportError:
        # Integer approx warp
        h, w = flow.shape[:2]
        ys, xs = np.indices((h, w))
        src_x = np.clip(np.rint(xs + flow[..., 0]).astype(int), 0, w - 1)
        src_y = np.clip(np.rint(ys + flow[..., 1]).astype(int), 0, h - 1)
        return frame[src_y, src_x]


def temporal_warping_error(
    rgb: np.ndarray,
    *,
    flow_backend: str = "auto",
    max_flow_mag: float = 40.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Optical-flow-guided temporal warping error (lower raw error = more consistent).

    Returns mean absolute error on luminance after warping, plus detail dict.
    Robot ego-motion will increase this score — do not treat motion as flicker alone.
    """
    if rgb.shape[0] < 2:
        return 0.0, {"pairs": 0, "backend": "n/a"}

    gray = _to_gray(rgb)
    errors = []
    backend_used = None
    for t in range(gray.shape[0] - 1):
        prev_g, next_g = gray[t], gray[t + 1]
        flow = None
        if flow_backend in ("auto", "farneback"):
            flow = _farneback_flow(prev_g, next_g)
            if flow is not None:
                backend_used = "farneback"
        if flow is None:
            # Frame-diff fallback: treat zero flow → error = |next-prev|
            backend_used = "frame_diff"
            err = np.abs(next_g.astype(np.float32) - prev_g.astype(np.float32))
            errors.append(float(err.mean()) / 255.0)
            continue

        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        valid = mag <= max_flow_mag
        warped = _warp_back(next_g.astype(np.float32), flow)
        err = np.abs(warped - prev_g.astype(np.float32))
        if valid.any():
            errors.append(float(err[valid].mean()) / 255.0)
        else:
            errors.append(float(err.mean()) / 255.0)

    return float(np.mean(errors)) if errors else 0.0, {
        "pairs": len(errors),
        "backend": backend_used or flow_backend,
    }


def flickering_score(rgb: np.ndarray) -> float:
    """
    Mean absolute frame-to-frame luminance difference / 255.
    Lower raw = less flicker. Returned as raw error in [0, ~1].
    """
    if rgb.shape[0] < 2:
        return 0.0
    gray = _to_gray(rgb)
    diffs = np.abs(np.diff(gray.astype(np.float32), axis=0))
    return float(diffs.mean() / 255.0)


def compute_temporal_metrics(
    rgb: np.ndarray,
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, MetricResult]:
    cfg = cfg or {}
    results: Dict[str, MetricResult] = {}

    warp_cfg = cfg.get("warping_error", {})
    if warp_cfg.get("enabled", True):
        try:
            err, detail = temporal_warping_error(
                rgb,
                flow_backend=str(warp_cfg.get("flow_backend", "auto")),
                max_flow_mag=float(warp_cfg.get("max_flow_mag", 40.0)),
            )
            results["temporal_warping_error"] = MetricResult(
                "temporal_warping_error", err, "ok", reference_tag="n/a", detail=detail
            )
        except Exception as e:
            results["temporal_warping_error"] = MetricResult(
                "temporal_warping_error", None, "error", detail={"error": repr(e)}
            )

    flick_cfg = cfg.get("flickering", {})
    if flick_cfg.get("enabled", True):
        try:
            flick = flickering_score(rgb)
            results["flickering"] = MetricResult(
                "flickering", flick, "ok", reference_tag="n/a"
            )
        except Exception as e:
            results["flickering"] = MetricResult(
                "flickering", None, "error", detail={"error": repr(e)}
            )

    lpips_cfg = cfg.get("temporal_lpips", {})
    if lpips_cfg.get("enabled", False):
        results["temporal_lpips"] = MetricResult(
            "temporal_lpips", None, "MISSING_DEP", detail={"need": "torch+lpips"}
        )

    return results


def aggregate_temporal_score(metric_results: Dict[str, MetricResult]) -> Optional[float]:
    """Map temporal errors to T_i in [0,1], higher = more consistent."""
    parts = []
    if "temporal_warping_error" in metric_results and metric_results["temporal_warping_error"].status == "ok":
        e = float(metric_results["temporal_warping_error"].value)
        parts.append(float(np.exp(-5.0 * e)))  # heuristic map
    if "flickering" in metric_results and metric_results["flickering"].status == "ok":
        e = float(metric_results["flickering"].value)
        parts.append(float(np.exp(-5.0 * e)))
    if not parts:
        return None
    return float(np.mean(parts))
