"""Control fidelity metrics (Proxy vs re-extracted attributes from generated video).

IMPORTANT:
  - Compare generated-video re-extracted maps to the *control signal*, not control to itself.
  - AutoSeg / Depth-Anything / OpenCV controls are tagged Proxy, never Ground Truth.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np

from .io_utils import align_lengths, load_map_sequence


@dataclass
class MetricResult:
    name: str
    value: Optional[float]
    status: str  # ok | skipped | MISSING_DEP | missing_data | error
    reference_tag: str = "Proxy"
    detail: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def segmentation_miou(
    pred_labels: np.ndarray,
    ref_labels: np.ndarray,
    *,
    roi_mask: Optional[np.ndarray] = None,
    min_pixels_per_class: int = 1,
) -> float:
    """Mean IoU over classes present in the reference (integer label maps)."""
    pred_labels, ref_labels = align_lengths(pred_labels, ref_labels)
    if roi_mask is not None:
        roi_mask, = align_lengths(roi_mask.astype(bool))
        # broadcast ROI across time if needed
        if roi_mask.ndim == 2:
            roi = np.broadcast_to(roi_mask, pred_labels.shape)
        else:
            roi = roi_mask[: pred_labels.shape[0]]
    else:
        roi = np.ones(pred_labels.shape, dtype=bool)

    pred = pred_labels.astype(np.int64)[roi]
    ref = ref_labels.astype(np.int64)[roi]
    classes = np.unique(ref)
    ious = []
    for c in classes:
        pred_c = pred == c
        ref_c = ref == c
        inter = np.logical_and(pred_c, ref_c).sum()
        union = np.logical_or(pred_c, ref_c).sum()
        if ref_c.sum() < min_pixels_per_class:
            continue
        if union == 0:
            continue
        ious.append(float(inter) / float(union))
    if not ious:
        return 0.0
    return float(np.mean(ious))


def edge_f1(
    pred_edge: np.ndarray,
    ref_edge: np.ndarray,
    *,
    binary_threshold: float = 0.5,
    roi_mask: Optional[np.ndarray] = None,
) -> float:
    """Binary edge F1 between predicted and reference edge maps."""
    pred_edge, ref_edge = align_lengths(pred_edge, ref_edge)

    def _binarize(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64)
        # Auto-scale if looks like 0-255
        if np.nanmax(x) > 1.5:
            x = x / 255.0
        return x >= binary_threshold

    pred = _binarize(pred_edge)
    ref = _binarize(ref_edge)
    if roi_mask is not None:
        if roi_mask.ndim == 2:
            roi = np.broadcast_to(roi_mask.astype(bool), pred.shape)
        else:
            roi = roi_mask[: pred.shape[0]].astype(bool)
        pred = pred & roi
        ref = ref & roi

    tp = np.logical_and(pred, ref).sum()
    fp = np.logical_and(pred, np.logical_not(ref)).sum()
    fn = np.logical_and(np.logical_not(pred), ref).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(2 * precision * recall / (precision + recall + 1e-8))


def depth_si_rmse(
    pred_depth: np.ndarray,
    ref_depth: np.ndarray,
    *,
    invalid_as_nan: bool = True,
    roi_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Scale-invariant RMSE (Eigen et al.):
      SI-RMSE = sqrt( mean(d^2) - (mean(d))^2 ) where d = log(pred) - log(ref)
    Lower is better; callers map to [0,1] consistency separately.
    """
    pred_depth, ref_depth = align_lengths(pred_depth, ref_depth)
    pred = pred_depth.astype(np.float64)
    ref = ref_depth.astype(np.float64)
    if invalid_as_nan:
        pred = np.where(pred <= 0, np.nan, pred)
        ref = np.where(ref <= 0, np.nan, ref)
    mask = np.isfinite(pred) & np.isfinite(ref)
    if roi_mask is not None:
        if roi_mask.ndim == 2:
            roi = np.broadcast_to(roi_mask.astype(bool), pred.shape)
        else:
            roi = roi_mask[: pred.shape[0]].astype(bool)
        mask = mask & roi
    if mask.sum() < 2:
        return float("nan")
    d = np.log(pred[mask]) - np.log(ref[mask])
    return float(np.sqrt(np.mean(d ** 2) - (np.mean(d) ** 2)))


def reextract_seg_from_rgb(rgb: np.ndarray, n_clusters: int = 4) -> np.ndarray:
    """
    Cheap Proxy re-extraction for smoke: k-means-like quantization on RGB.
    NOT a real segmentation model — used only when a dedicated re-extractor is absent.
    """
    t, h, w, c = rgb.shape
    # Use luminance + simple color bins for determinism without sklearn.
    lum = (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    )
    # Quantize into n_clusters bins
    edges = np.linspace(0, 256, n_clusters + 1)
    labels = np.digitize(lum, edges[1:-1], right=False).astype(np.int32)
    return labels


def reextract_edge_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """Sobel/Canny-like edge magnitude via numpy; OpenCV Canny if available."""
    try:
        import cv2  # type: ignore

        edges = []
        for t in range(rgb.shape[0]):
            gray = cv2.cvtColor(rgb[t], cv2.COLOR_RGB2GRAY)
            e = cv2.Canny(gray, 80, 160).astype(np.float32) / 255.0
            edges.append(e)
        return np.stack(edges, axis=0)
    except ImportError:
        # Numpy Sobel fallback
        gray = (
            0.299 * rgb[..., 0].astype(np.float32)
            + 0.587 * rgb[..., 1].astype(np.float32)
            + 0.114 * rgb[..., 2].astype(np.float32)
        )
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        ky = kx.T
        out = []
        for t in range(gray.shape[0]):
            g = gray[t]
            # pad
            p = np.pad(g, 1, mode="edge")
            gx = (
                kx[0, 0] * p[:-2, :-2]
                + kx[0, 1] * p[:-2, 1:-1]
                + kx[0, 2] * p[:-2, 2:]
                + kx[1, 0] * p[1:-1, :-2]
                + kx[1, 1] * p[1:-1, 1:-1]
                + kx[1, 2] * p[1:-1, 2:]
                + kx[2, 0] * p[2:, :-2]
                + kx[2, 1] * p[2:, 1:-1]
                + kx[2, 2] * p[2:, 2:]
            )
            gy = (
                ky[0, 0] * p[:-2, :-2]
                + ky[0, 1] * p[:-2, 1:-1]
                + ky[0, 2] * p[:-2, 2:]
                + ky[1, 0] * p[1:-1, :-2]
                + ky[1, 1] * p[1:-1, 1:-1]
                + ky[1, 2] * p[1:-1, 2:]
                + ky[2, 0] * p[2:, :-2]
                + ky[2, 1] * p[2:, 1:-1]
                + ky[2, 2] * p[2:, 2:]
            )
            mag = np.sqrt(gx ** 2 + gy ** 2)
            mag = mag / (mag.max() + 1e-8)
            out.append(mag)
        return np.stack(out, axis=0)


def reextract_depth_proxy_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    Inverse-luminance pseudo-depth for smoke only (Proxy of a Proxy).
    Real Pilot should re-run Video-Depth-Anything on generated frames.
    """
    lum = (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    )
    # Darker → farther heuristic inverted: use (1 + lum) so values > 0
    return (255.0 - lum) + 1.0


def compute_control_metrics_for_video(
    generated_rgb: np.ndarray,
    control_paths: Dict[str, Any],
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, MetricResult]:
    """
    Compute Proxy control fidelity for one generated video.

    `control_paths` maps control type → path (dir/npy/mp4). Missing keys → missing_data.
    Re-extraction uses cheap CPU proxies for smoke; Pilot should swap in real models.
    """
    cfg = cfg or {}
    results: Dict[str, MetricResult] = {}

    # Segmentation
    seg_cfg = cfg.get("segmentation", {})
    if seg_cfg.get("enabled", True):
        seg_path = control_paths.get("seg") if control_paths else None
        if not seg_path:
            results["seg_miou"] = MetricResult(
                "seg_miou", None, "missing_data", reference_tag="Proxy"
            )
        else:
            try:
                ref = load_map_sequence(seg_path, dtype=np.int32)
                # If ref looks continuous 0-255, quantize similarly
                if ref.max() > 20:
                    # treat as color-coded seg vis → coarse bins
                    ref = (ref.astype(np.float32) / 32).astype(np.int32)
                pred = reextract_seg_from_rgb(generated_rgb, n_clusters=int(ref.max()) + 1)
                pred, ref = align_lengths(pred, ref)
                # spatial resize if needed
                if pred.shape[1:] != ref.shape[1:]:
                    pred = _resize_labels(pred, ref.shape[1:])
                val = segmentation_miou(
                    pred,
                    ref,
                    min_pixels_per_class=int(seg_cfg.get("min_pixels_per_class", 1)),
                )
                results["seg_miou"] = MetricResult(
                    "seg_miou", val, "ok", reference_tag="Proxy", detail={"note": "reextract=quantize_luminance"}
                )
            except Exception as e:
                results["seg_miou"] = MetricResult(
                    "seg_miou", None, "error", reference_tag="Proxy", detail={"error": repr(e)}
                )

    # Edge
    edge_cfg = cfg.get("edge", {})
    if edge_cfg.get("enabled", True):
        edge_path = control_paths.get("edge") if control_paths else None
        if not edge_path:
            results["edge_f1"] = MetricResult(
                "edge_f1", None, "missing_data", reference_tag="Proxy"
            )
        else:
            try:
                ref = load_map_sequence(edge_path, dtype=np.float32)
                pred = reextract_edge_from_rgb(generated_rgb)
                pred, ref = align_lengths(pred, ref)
                if pred.shape[1:] != ref.shape[1:]:
                    pred = _resize_float(pred, ref.shape[1:])
                val = edge_f1(
                    pred,
                    ref,
                    binary_threshold=float(edge_cfg.get("binary_threshold", 0.5)),
                )
                results["edge_f1"] = MetricResult(
                    "edge_f1", val, "ok", reference_tag="Proxy"
                )
            except Exception as e:
                results["edge_f1"] = MetricResult(
                    "edge_f1", None, "error", reference_tag="Proxy", detail={"error": repr(e)}
                )

    # Depth
    depth_cfg = cfg.get("depth", {})
    if depth_cfg.get("enabled", True):
        depth_path = control_paths.get("depth") if control_paths else None
        if not depth_path:
            results["depth_si_rmse"] = MetricResult(
                "depth_si_rmse", None, "missing_data", reference_tag="Proxy"
            )
        else:
            try:
                ref = load_map_sequence(depth_path, dtype=np.float32)
                pred = reextract_depth_proxy_from_rgb(generated_rgb)
                pred, ref = align_lengths(pred, ref)
                if pred.shape[1:] != ref.shape[1:]:
                    pred = _resize_float(pred, ref.shape[1:])
                # Ensure ref > 0
                ref = np.where(ref <= 0, np.nan, ref)
                val = depth_si_rmse(
                    pred,
                    ref,
                    invalid_as_nan=bool(depth_cfg.get("invalid_as_nan", True)),
                )
                results["depth_si_rmse"] = MetricResult(
                    "depth_si_rmse",
                    val,
                    "ok" if np.isfinite(val) else "error",
                    reference_tag="Proxy",
                    detail={"note": "reextract=inverse_luminance_proxy"},
                )
            except Exception as e:
                results["depth_si_rmse"] = MetricResult(
                    "depth_si_rmse", None, "error", reference_tag="Proxy", detail={"error": repr(e)}
                )

    return results


def _resize_labels(labels: np.ndarray, hw: tuple) -> np.ndarray:
    try:
        import cv2  # type: ignore

        out = []
        for t in range(labels.shape[0]):
            out.append(
                cv2.resize(labels[t].astype(np.int32), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
            )
        return np.stack(out, axis=0)
    except ImportError:
        # Nearest via simple indexing
        ys = (np.linspace(0, labels.shape[1] - 1, hw[0])).astype(int)
        xs = (np.linspace(0, labels.shape[2] - 1, hw[1])).astype(int)
        return labels[:, ys][:, :, xs]


def _resize_float(arr: np.ndarray, hw: tuple) -> np.ndarray:
    try:
        import cv2  # type: ignore

        out = []
        for t in range(arr.shape[0]):
            out.append(
                cv2.resize(arr[t].astype(np.float32), (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
            )
        return np.stack(out, axis=0)
    except ImportError:
        ys = (np.linspace(0, arr.shape[1] - 1, hw[0])).astype(int)
        xs = (np.linspace(0, arr.shape[2] - 1, hw[1])).astype(int)
        return arr[:, ys][:, :, xs].astype(np.float32)


def aggregate_control_score(metric_results: Dict[str, MetricResult]) -> Optional[float]:
    """
    Aggregate control metrics into C_i in [0,1], higher better.
    - seg_miou, edge_f1 already in [0,1]
    - depth_si_rmse: map via exp(-si_rmse) heuristic
    """
    parts = []
    if "seg_miou" in metric_results and metric_results["seg_miou"].status == "ok":
        parts.append(float(metric_results["seg_miou"].value))
    if "edge_f1" in metric_results and metric_results["edge_f1"].status == "ok":
        parts.append(float(metric_results["edge_f1"].value))
    if "depth_si_rmse" in metric_results and metric_results["depth_si_rmse"].status == "ok":
        si = float(metric_results["depth_si_rmse"].value)
        parts.append(float(np.exp(-si)))
    if not parts:
        return None
    return float(np.mean(parts))
