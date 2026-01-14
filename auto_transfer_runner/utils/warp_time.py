from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TimeStepDebug:
    mode_requested: str
    mode_used: str
    fallback_reason: str | None
    alpha_stats: dict[str, float] | None
    alpha: list[float] | None


def validate_time_steps(time_steps: List[List[float]], segment_frames: List[int]) -> None:
    if len(time_steps) != len(segment_frames):
        raise ValueError("time_steps length must match segment_frames length")
    for idx, frames_in_segment in enumerate(segment_frames):
        expected = max(frames_in_segment - 2, 0)
        segment_steps = time_steps[idx]
        if frames_in_segment == 2 and segment_steps:
            raise ValueError(f"segment {idx} should have no time_steps for segment_frames=2")
        if len(segment_steps) != expected:
            raise ValueError(
                f"segment {idx} time_steps length mismatch: expected {expected} got {len(segment_steps)}"
            )
        prev = 0.0
        for step in segment_steps:
            if not (0.0 < step < 1.0):
                raise ValueError(f"segment {idx} time_steps must be in (0,1), got {step}")
            if step <= prev:
                raise ValueError(f"segment {idx} time_steps must be strictly increasing")
            prev = step


def count_guided_output_frames(
    frames: Iterable[object],
    segment_frames: List[int],
    time_steps: List[List[float]],
) -> int:
    frames_list = list(frames)
    if len(frames_list) != len(segment_frames) + 1:
        raise ValueError("frames length must equal len(segment_frames)+1")
    validate_time_steps(time_steps, segment_frames)
    total = 1
    for frames_in_segment in segment_frames:
        total += frames_in_segment - 1
    return total


def compute_time_steps(
    frame_map: List[int],
    segment_frames: List[int],
    parquet_path: str | None,
    pose_indices: List[int] | None = None,
    pose_name_prefix: str | list[str] | None = "position.zarm_",
    sam_masks: Optional[List[np.ndarray] | str | Path] = None,
    image_wh: Tuple[int, int] = (848, 480),
    mode: str = "sam_zarm_hybrid",
    cache_path: Optional[str | Path] = None,
    debug: bool = False,
    force_mode: bool = False,
) -> List[List[float]] | tuple[List[List[float]], TimeStepDebug]:
    if len(frame_map) < 2:
        raise ValueError("frame_map must have at least 2 entries")
    if len(segment_frames) != len(frame_map) - 1:
        raise ValueError("segment_frames must be len(frame_map)-1")
    if mode not in {"uniform", "zarm_pos", "sam", "sam_zarm_hybrid", "sam_cache", "sam_iou"}:
        raise ValueError(f"Unsupported time_steps mode: {mode}")

    mode_for_logic = "sam" if mode in {"sam_cache", "sam_iou"} else mode

    cache_key = None
    cache_path = Path(cache_path) if cache_path else None
    if cache_path:
        cache_key = _build_cache_key(
            frame_map=frame_map,
            segment_frames=segment_frames,
            parquet_path=Path(parquet_path) if parquet_path else None,
            sam_masks=sam_masks,
            mode=mode,
            pose_indices=pose_indices,
            pose_name_prefix=pose_name_prefix,
            image_wh=image_wh,
        )
        if cache_path.exists():
            cached = _load_time_steps_cache(cache_path, cache_key)
            if cached is not None and not debug:
                return cached

    fallback_reason = None
    mode_used = mode_for_logic
    alpha_stats = None
    alpha_values: list[float] | None = None

    if mode_for_logic == "uniform":
        time_steps = _uniform_time_steps(frame_map, segment_frames)
    else:
        d_state = None
        d_vis = None
        alpha = None

        if mode_for_logic in {"zarm_pos", "sam_zarm_hybrid"} or mode_for_logic == "sam":
            if parquet_path:
                d_state = _compute_state_deltas(
                    parquet_path=Path(parquet_path),
                    frame_count=frame_map[-1] + 1,
                    pose_indices=pose_indices,
                    pose_name_prefix=pose_name_prefix,
                )

        if mode_for_logic in {"sam", "sam_zarm_hybrid"}:
            d_vis = _compute_sam_deltas(
                sam_masks=sam_masks,
                frame_count=frame_map[-1] + 1,
            )

        if mode_for_logic == "sam" and d_vis is None:
            if force_mode:
                raise RuntimeError("SAM masks unavailable for time_steps mode=sam (force_mode enabled)")
            if d_state is None:
                mode_used = "uniform"
                fallback_reason = fallback_reason or "sam_unavailable_no_state"
            else:
                mode_used = "zarm_pos"
                fallback_reason = fallback_reason or "sam_unavailable"

        if mode_for_logic == "sam_zarm_hybrid":
            if d_vis is None:
                if force_mode:
                    raise RuntimeError("SAM masks unavailable for time_steps mode=sam_zarm_hybrid (force_mode enabled)")
                mode_used = "zarm_pos"
                fallback_reason = fallback_reason or "sam_unavailable"
            elif d_state is None:
                mode_used = "sam"
                fallback_reason = fallback_reason or "zarm_unavailable"

        if mode_used == "sam":
            d = d_vis
        elif mode_used == "zarm_pos":
            d = d_state
        elif mode_used == "sam_zarm_hybrid":
            d, alpha = _blend_deltas(d_vis, d_state)
        else:
            d = None

        if d is None:
            if force_mode:
                raise RuntimeError("Unable to compute time_steps from SAM masks (force_mode enabled)")
            time_steps = _uniform_time_steps(frame_map, segment_frames)
            mode_used = "uniform"
            fallback_reason = fallback_reason or "no_motion_signal"
        else:
            time_steps = _time_steps_from_deltas(
                frame_map=frame_map,
                segment_frames=segment_frames,
                deltas=d,
            )
            validate_time_steps(time_steps, segment_frames)
            if alpha is not None:
                alpha_values = alpha.tolist()
                alpha_stats = _summary_stats(alpha)

    if cache_path and cache_key:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "mode_requested": mode,
            "mode_used": mode_used,
            "fallback_reason": fallback_reason,
            "alpha_stats": alpha_stats,
        }
        _write_time_steps_cache(cache_path, cache_key, time_steps, meta)

    if debug:
        debug_payload = TimeStepDebug(
            mode_requested=mode,
            mode_used=mode_used,
            fallback_reason=fallback_reason,
            alpha_stats=alpha_stats,
            alpha=alpha_values,
        )
        return time_steps, debug_payload
    return time_steps


def _load_time_steps_cache(cache_path: Path, cache_key: str) -> List[List[float]] | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("cache_key") != cache_key:
        return None
    time_steps = payload.get("time_steps")
    if not isinstance(time_steps, list):
        return None
    return time_steps


def _write_time_steps_cache(
    cache_path: Path,
    cache_key: str,
    time_steps: List[List[float]],
    meta: dict[str, object],
) -> None:
    payload = {
        "cache_key": cache_key,
        "meta": meta,
        "time_steps": time_steps,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_cache_key(
    *,
    frame_map: List[int],
    segment_frames: List[int],
    parquet_path: Path | None,
    sam_masks: Optional[List[np.ndarray] | str | Path],
    mode: str,
    pose_indices: List[int] | None,
    pose_name_prefix: str | list[str] | None,
    image_wh: Tuple[int, int],
) -> str:
    sam_mask_root = None
    sam_mtime = None
    if isinstance(sam_masks, (str, Path)):
        sam_mask_root = str(sam_masks)
        if Path(sam_masks).exists():
            sam_mtime = Path(sam_masks).stat().st_mtime
    parquet_mtime = parquet_path.stat().st_mtime if parquet_path and parquet_path.exists() else None
    payload = {
        "frame_map": frame_map,
        "segment_frames": segment_frames,
        "mode": mode,
        "pose_indices": pose_indices,
        "pose_name_prefix": pose_name_prefix,
        "parquet_path": str(parquet_path) if parquet_path else None,
        "parquet_mtime": parquet_mtime,
        "sam_mask_root": sam_mask_root,
        "sam_mask_mtime": sam_mtime,
        "image_wh": image_wh,
        "version": 1,
    }
    digest = hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


def _uniform_time_steps(frame_map: List[int], segment_frames: List[int]) -> List[List[float]]:
    time_steps: List[List[float]] = []
    for idx, frames_in_segment in enumerate(segment_frames):
        if frames_in_segment == 2:
            time_steps.append([])
            continue
        start = frame_map[idx]
        end = frame_map[idx + 1]
        steps = []
        for frame_idx in range(start + 1, end):
            steps.append((frame_idx - start) / (end - start))
        time_steps.append(steps)
    validate_time_steps(time_steps, segment_frames)
    return time_steps


def _time_steps_from_deltas(
    *,
    frame_map: List[int],
    segment_frames: List[int],
    deltas: np.ndarray,
) -> List[List[float]]:
    eps = 1e-6
    time_steps: List[List[float]] = []
    for idx, frames_in_segment in enumerate(segment_frames):
        if frames_in_segment == 2:
            time_steps.append([])
            continue
        start = frame_map[idx]
        end = frame_map[idx + 1]
        segment_deltas = deltas[start + 1 : end + 1]
        if segment_deltas.size == 0:
            time_steps.append([])
            continue
        cumulative = np.cumsum(segment_deltas)
        total = cumulative[-1]
        if total < eps:
            steps = []
            for frame_idx in range(start + 1, end):
                steps.append((frame_idx - start) / (end - start))
            time_steps.append(steps)
            continue
        steps = (cumulative[:-1] / total).tolist()
        clamped = _clamp_strictly_increasing(steps)
        time_steps.append(clamped)
    validate_time_steps(time_steps, segment_frames)
    return time_steps


def _clamp_strictly_increasing(values: List[float]) -> List[float]:
    if not values:
        return values
    eps = 1e-4
    result = []
    prev = 0.0
    for value in values:
        clamped = min(max(value, eps), 1.0 - eps)
        if clamped <= prev:
            clamped = min(prev + eps, 1.0 - eps)
        result.append(clamped)
        prev = clamped
    return result


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def _blend_deltas(
    d_vis: np.ndarray | None,
    d_state: np.ndarray | None,
    *,
    alpha_value: float = 0.5,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if d_vis is None or d_state is None:
        return None, None
    vis_norm = _robust_normalize(d_vis)
    state_norm = _robust_normalize(d_state)
    alpha = np.full_like(vis_norm, alpha_value, dtype=float)
    blended = alpha * vis_norm + (1.0 - alpha) * state_norm
    return blended, alpha


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    median = np.median(values[values > 0]) if np.any(values > 0) else np.median(values)
    if not np.isfinite(median) or median <= 0:
        median = 1.0
    return values / (median + 1e-6)


def _compute_state_deltas(
    *,
    parquet_path: Path,
    frame_count: int,
    pose_indices: List[int] | None,
    pose_name_prefix: str | list[str] | None,
) -> np.ndarray | None:
    if not parquet_path.exists():
        return None
    states, state_names = _load_parquet_states(parquet_path)
    if states.shape[0] < frame_count:
        return None
    if pose_indices is None:
        pose_indices = _select_pose_indices(
            state_names,
            subset_prefixes=pose_name_prefix,
        )
    if not pose_indices:
        return None
    dataset_root = _find_dataset_root(parquet_path)
    stats_std = _load_state_std(dataset_root, state_dim=states.shape[1])
    pose = states[:, pose_indices]
    if stats_std is not None:
        std_sel = stats_std[pose_indices]
        pose = pose / (std_sel + 1e-6)
    else:
        std_sel = np.std(pose, axis=0)
        pose = pose / (std_sel + 1e-6)
    deltas = np.zeros(states.shape[0], dtype=float)
    deltas[1:] = np.linalg.norm(pose[1:] - pose[:-1], axis=1)
    return deltas


def _load_state_std(dataset_root: Path, *, state_dim: int) -> np.ndarray | None:
    stats_path = dataset_root / "meta" / "stats.json"
    if not stats_path.exists():
        return None
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    state_stats = None
    if isinstance(stats.get("observation"), dict):
        state_stats = stats["observation"].get("state")
    if state_stats is None and isinstance(stats.get("observations"), dict):
        state_stats = stats["observations"].get("state")
    if state_stats is None and isinstance(stats.get("observation.state"), dict):
        state_stats = stats.get("observation.state")
    if state_stats is None and isinstance(stats.get("features"), dict):
        features = stats["features"]
        if isinstance(features.get("observation.state"), dict):
            state_stats = features.get("observation.state")
    if not isinstance(state_stats, dict):
        return None
    std = state_stats.get("std")
    if not isinstance(std, list) or len(std) != state_dim:
        return None
    return np.asarray(std, dtype=float)


def _load_parquet_states(parquet_path: Path) -> tuple[np.ndarray, list[str]]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    names = set(table.schema.names)
    if "observation.state" in names:
        state_values = table.column("observation.state").to_pylist()
    elif "observation" in names:
        obs_values = table.column("observation").to_pylist()
        state_values = []
        for item in obs_values:
            if isinstance(item, dict):
                state_values.append(item.get("state"))
            else:
                state_values.append(item["state"])
    else:
        raise ValueError("Parquet missing observation.state column")
    states = np.asarray(state_values, dtype=float)
    if states.ndim == 1 and len(states) > 0:
        states = np.stack(state_values, axis=0).astype(float)
    if states.ndim != 2:
        raise ValueError(f"Unexpected observation.state shape: {states.shape}")
    state_names = _load_state_names(_find_dataset_root(parquet_path), state_dim=states.shape[1])
    return states, state_names


def _find_dataset_root(parquet_path: Path) -> Path:
    for parent in [parquet_path.parent, *parquet_path.parents]:
        if (parent / "meta" / "info.json").exists():
            return parent
    return parquet_path.parent


def _load_state_names(dataset_root: Path, *, state_dim: int) -> list[str]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return [f"state_{idx}" for idx in range(state_dim)]
    info = json.loads(info_path.read_text(encoding="utf-8"))
    state_info = None
    if isinstance(info.get("observation"), dict):
        state_info = info["observation"].get("state")
    if state_info is None and isinstance(info.get("observations"), dict):
        state_info = info["observations"].get("state")
    if state_info is None and isinstance(info.get("observation.state"), dict):
        state_info = info.get("observation.state")
    if state_info is None and isinstance(info.get("features"), dict):
        features = info["features"]
        if isinstance(features.get("observation.state"), dict):
            state_info = features.get("observation.state")
    names = None
    if isinstance(state_info, dict):
        names = state_info.get("names")
    if isinstance(names, dict):
        dexhand_names = list(names.get("dexhand", []))
        sensors_names = list(names.get("sensors_raw", []))
        names = dexhand_names + sensors_names
    if not isinstance(names, list) or len(names) != state_dim:
        return [f"state_{idx}" for idx in range(state_dim)]
    return names


def _select_pose_indices(state_names: list[str], *, subset_prefixes: str | list[str] | None) -> list[int]:
    if subset_prefixes is None:
        return list(range(len(state_names)))
    prefixes = [subset_prefixes] if isinstance(subset_prefixes, str) else list(subset_prefixes)
    keep = [idx for idx, name in enumerate(state_names) if any(name.startswith(prefix) for prefix in prefixes)]
    return keep


def _compute_sam_deltas(
    *,
    sam_masks: Optional[List[np.ndarray] | str | Path],
    frame_count: int,
) -> np.ndarray | None:
    masks = _load_sam_masks(sam_masks, frame_count=frame_count)
    if masks is None:
        return None
    if len(masks) < frame_count:
        return None
    centroids = []
    for mask in masks[:frame_count]:
        if mask is None or mask.size == 0:
            centroids.append(None)
            continue
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            centroids.append(None)
            continue
        centroids.append((float(np.mean(xs)), float(np.mean(ys))))
    centroids = _median_smooth_centroid(centroids, window=3)

    d_vis = np.zeros(frame_count, dtype=float)
    for idx in range(1, frame_count):
        centroid_prev = centroids[idx - 1]
        centroid_curr = centroids[idx]
        if centroid_prev is None or centroid_curr is None:
            continue
        delta_c = np.linalg.norm(np.subtract(centroid_curr, centroid_prev))
        d_vis[idx] = delta_c
    return d_vis


def _median_smooth_centroid(values: list[tuple[float, float] | None], window: int) -> list[tuple[float, float] | None]:
    if window <= 1 or not values:
        return values
    half = window // 2
    out: list[tuple[float, float] | None] = []
    for idx in range(len(values)):
        xs = []
        ys = []
        for j in range(max(0, idx - half), min(len(values), idx + half + 1)):
            if values[j] is None:
                continue
            xs.append(values[j][0])
            ys.append(values[j][1])
        if not xs:
            out.append(None)
        else:
            out.append((float(np.median(xs)), float(np.median(ys))))
    return out


def _load_sam_masks(
    sam_masks: Optional[List[np.ndarray] | str | Path],
    *,
    frame_count: int,
) -> list[np.ndarray] | None:
    if sam_masks is None:
        return None
    if isinstance(sam_masks, (str, Path)):
        mask_root = Path(sam_masks)
        if mask_root.is_dir():
            mask_files = sorted(
                list(mask_root.glob("*.png"))
                + list(mask_root.glob("*.jpg"))
                + list(mask_root.glob("*.jpeg"))
                + list(mask_root.glob("*.npy"))
            )
            if not mask_files:
                return None
            return [_load_mask_file(path) for path in mask_files]
        if mask_root.is_file():
            return [_load_mask_file(mask_root)]
        return None
    if isinstance(sam_masks, list):
        if not sam_masks:
            return None
        if isinstance(sam_masks[0], np.ndarray):
            return sam_masks[:frame_count]
        return [_load_mask_file(Path(item)) for item in sam_masks]
    return None


def _load_mask_file(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        data = np.load(path)
        return _normalize_mask(data)
    from PIL import Image

    with Image.open(path) as img:
        data = np.asarray(img.convert("L"))
    return _normalize_mask(data)


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim > 2:
        mask = mask[..., 0]
    return mask > 0
