from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image, ImageDraw

from ..config import ArmSwapConfig, CacheConfig, PostprocessConfig, PreprocessConfig
from ..steps.arm_swap import ensure_qwen_sam_cache, find_mask_dir_with_files, run_arm_swap_sam_masks
from ..steps.postprocess import probe_video_stats, restore_duration
from ..steps.preprocess import (
    PreprocessSettings,
    frame_map_path_for,
    write_keyframes_from_indices,
    _compute_motion_diffs,
    _load_parquet_states,
    _load_state_stats,
    _select_pose_indices,
)
from ..utils.dataset_paths import find_dataset_root, parquet_path_for_video
from ..utils.hashing import short_hash, video_cache_key
from ..utils.logging_utils import append_log, console_print


@dataclass(frozen=True)
class ProxyPreflightSettings:
    enable: bool = False
    iou_threshold: float = 0.9
    iou_high_threshold: float = 0.98
    motion_low_quantile: float = 0.25
    max_iters: int = 3
    restore_mode: str = "rife"
    rife_backend: str | None = None
    cache_iou_csv: bool = False


def _append_preflight_iou_csv(
    csv_path: Path,
    iteration: int,
    iou_series: list[float],
    original_frames: list[int] | None = None,
    keyframes_before: set[int] | None = None,
    keyframes_after: set[int] | None = None,
    bad_frames: set[int] | None = None,
) -> None:
    write_header = not csv_path.exists()
    if original_frames is None:
        original_frames = list(range(len(iou_series)))
    if len(original_frames) != len(iou_series):
        raise ValueError(
            "preflight iou csv requires original_frames to match iou_series length "
            f"({len(original_frames)} vs {len(iou_series)})",
        )
    keyframes_before = keyframes_before or set()
    keyframes_after = keyframes_after or set()
    bad_frames = bad_frames or set()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                [
                    "iteration",
                    "proxy_frame",
                    "original_frame",
                    "iou",
                    "is_keyframe_before",
                    "is_keyframe_after",
                    "is_bad_frame",
                ],
            )
        for frame_idx, (orig_frame, iou) in enumerate(zip(original_frames, iou_series)):
            writer.writerow(
                [
                    iteration,
                    frame_idx,
                    orig_frame,
                    f"{iou:.6f}",
                    int(orig_frame in keyframes_before),
                    int(orig_frame in keyframes_after),
                    int(orig_frame in bad_frames),
                ],
            )


def _load_frame_map(frame_map_path: Path) -> list[int]:
    payload = json.loads(frame_map_path.read_text(encoding="utf-8"))
    indices = payload.get("sampled_indices")
    if not isinstance(indices, list) or not indices:
        raise ValueError(f"Invalid frame_map: {frame_map_path}")
    return [int(idx) for idx in indices]


def _mask_files(mask_dir: Path) -> List[Path]:
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.npy")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(mask_dir.glob(pattern))
    return sorted({p for p in files if p.is_file()})


def _uniform_frame_map(total_frames: int, target_frames: int) -> list[int]:
    if total_frames <= 0 or target_frames <= 0:
        raise ValueError("frame map requires positive frame counts")
    if target_frames >= total_frames:
        return list(range(total_frames))
    if target_frames == 1:
        return [0]
    step = (total_frames - 1) / (target_frames - 1)
    indices: list[int] = []
    prev = -1
    for i in range(target_frames):
        idx = int(round(i * step))
        if idx <= prev:
            idx = prev + 1
        if idx >= total_frames:
            idx = total_frames - 1
        indices.append(idx)
        prev = idx
    indices[-1] = total_frames - 1
    return indices


def _load_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        data = np.load(path)
    else:
        with Image.open(path) as img:
            data = np.array(img.convert("L"))
    return data > 0


def _compute_iou_series(mask_dir_a: Path, mask_dir_b: Path) -> list[float]:
    masks_a = _mask_files(mask_dir_a)
    masks_b = _mask_files(mask_dir_b)
    if len(masks_a) != len(masks_b):
        raise ValueError(
            f"Mask count mismatch: {mask_dir_a}={len(masks_a)} {mask_dir_b}={len(masks_b)}",
        )
    iou_series: list[float] = []
    for mask_a, mask_b in zip(masks_a, masks_b):
        a = _load_mask(mask_a)
        b = _load_mask(mask_b)
        union = np.logical_or(a, b).sum()
        if union == 0:
            iou = 1.0
        else:
            iou = float(np.logical_and(a, b).sum()) / float(union)
        iou_series.append(iou)
    return iou_series


def _write_iou_plot_png(
    *,
    png_path: Path,
    iou_series: list[float],
    iteration: int,
    size: tuple[int, int] = (1280, 720),
    margin: int = 60,
) -> None:
    if not iou_series:
        return
    width, height = size
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    left = margin
    right = width - margin
    top = margin
    bottom = height - margin
    draw.rectangle([left, top, right, bottom], outline="black", width=2)

    series_len = len(iou_series)
    max_idx = max(series_len - 1, 1)
    points = []
    for idx, iou in enumerate(iou_series):
        x = left + (right - left) * idx / max_idx
        clamped = max(0.0, min(1.0, float(iou)))
        y = bottom - (bottom - top) * clamped
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill="blue", width=2)
    else:
        draw.ellipse(
            [points[0][0] - 2, points[0][1] - 2, points[0][0] + 2, points[0][1] + 2],
            fill="blue",
        )

    label = f"proxy preflight IoU (iter={iteration})"
    draw.text((left, max(0, top - 30)), label, fill="black")
    draw.text((left, bottom + 8), "frame index", fill="black")
    draw.text((5, top), "IoU", fill="black")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(png_path)


def _motion_per_frame(
    states: np.ndarray,
    frame_indices: Iterable[int],
    state_names: list[str],
    dexhand_names: set[str],
    pose_subset: str,
    stats_mean: np.ndarray | None,
    stats_std: np.ndarray | None,
    smooth_win: int,
    total_frames: int,
    log_path: Path,
) -> np.ndarray:
    keep_idx, _ = _select_pose_indices(state_names, dexhand_names, subset=pose_subset, log_path=log_path)
    diffs = _compute_motion_diffs(
        states,
        keep_idx=keep_idx,
        stats_mean=stats_mean,
        stats_std=stats_std,
        smooth_win=smooth_win,
        log_path=log_path,
    )
    motion = np.zeros(total_frames, dtype=float)
    for idx, frame_idx in enumerate(frame_indices):
        if 0 <= frame_idx < total_frames:
            motion[frame_idx] = diffs[idx]
    return motion


def _pick_insert_index(
    *,
    bad_frame_pos: int,
    current_indices: list[int],
    used_indices: set[int],
) -> tuple[int | None, str]:
    if bad_frame_pos <= 0 or bad_frame_pos >= len(current_indices):
        return None, f"bad_frame_pos out of range (pos={bad_frame_pos})"
    prev_idx = bad_frame_pos - 1
    prev_frame = current_indices[prev_idx]
    bad_frame = current_indices[bad_frame_pos]
    if bad_frame - prev_frame <= 1:
        return None, (
            "no free frame between prev_frame="
            f"{prev_frame} and bad_frame={bad_frame} (gap={bad_frame - prev_frame})"
        )
    candidate = (prev_frame + bad_frame) // 2
    for frame in range(candidate, prev_frame, -1):
        if frame not in used_indices:
            return frame, f"picked descending from candidate={candidate} prev_frame={prev_frame}"
    for frame in range(candidate + 1, bad_frame):
        if frame not in used_indices:
            return frame, f"picked ascending from candidate={candidate} bad_frame={bad_frame}"
    return None, f"no free frame between prev_frame={prev_frame} and bad_frame={bad_frame} (candidate={candidate})"


def _reallocate_keyframes(
    *,
    current_indices: list[int],
    bad_frames: list[int],
    iou_series: list[float],
    motion_series: np.ndarray,
    target_frames: int,
    iou_high_threshold: float,
    motion_low_quantile: float,
    log_path: Path,
) -> list[int]:
    used_indices = set(current_indices)
    insertions: list[int] = []
    insertion_details: list[str] = []
    insertion_failures: list[str] = []
    index_positions = {frame: pos for pos, frame in enumerate(current_indices)}
    for bad_frame in bad_frames:
        if bad_frame in used_indices:
            bad_frame_pos = index_positions.get(bad_frame)
            if bad_frame_pos is None:
                insertion_failures.append(f"bad_frame={bad_frame} missing from index_positions")
                continue
            candidate, detail = _pick_insert_index(
                bad_frame_pos=bad_frame_pos,
                current_indices=current_indices,
                used_indices=used_indices,
            )
            if candidate is not None:
                insertions.append(candidate)
                used_indices.add(candidate)
                insertion_details.append(
                    "bad_frame="
                    f"{bad_frame} (pos={bad_frame_pos}) -> "
                    f"insert={candidate} ({detail})"
                )
            else:
                insertion_failures.append(
                    f"bad_frame={bad_frame} (pos={bad_frame_pos}) ({detail})"
                )
        else:
            insertions.append(bad_frame)
            used_indices.add(bad_frame)
            insertion_details.append(f"bad_frame={bad_frame} -> insert self")

    if not insertions:
        if insertion_failures:
            append_log(
                log_path,
                "[STEP1.5.5] No valid insertion points. Details: "
                + "; ".join(insertion_failures),
            )
        append_log(log_path, "[STEP1.5.5] No valid insertion points found; skipping reallocation.")
        return current_indices
    if insertion_details:
        append_log(
            log_path,
            "[STEP1.5.5] Insertion candidates: " + "; ".join(insertion_details),
        )

    removal_candidates = [
        idx
        for idx in current_indices
        if idx not in {0, current_indices[-1]} and idx not in insertions
    ]
    if not removal_candidates:
        append_log(log_path, "[STEP1.5.5] No removable keyframes found; skipping reallocation.")
        return current_indices

    keyframe_motion = np.array([motion_series[idx] for idx in removal_candidates])
    motion_threshold = float(np.quantile(keyframe_motion, motion_low_quantile)) if keyframe_motion.size else 0.0
    scored = []
    for idx in removal_candidates:
        iou = iou_series[idx] if idx < len(iou_series) else 1.0
        motion = motion_series[idx]
        scored.append((iou, motion, idx))

    preferred = [entry for entry in scored if entry[0] >= iou_high_threshold and entry[1] <= motion_threshold]
    preferred.sort(key=lambda item: (-item[0], item[1]))
    scored.sort(key=lambda item: (-item[0], item[1]))

    removals_needed = len(insertions)
    removals: list[int] = []
    for pool in (preferred, scored):
        for iou, motion, idx in pool:
            if idx in removals:
                continue
            removals.append(idx)
            if len(removals) >= removals_needed:
                break
        if len(removals) >= removals_needed:
            break

    if len(removals) < removals_needed:
        append_log(
            log_path,
            f"[STEP1.5.5] Unable to remove enough keyframes (need={removals_needed} got={len(removals)}).",
        )
        return current_indices

    updated = sorted(set(current_indices + insertions) - set(removals))
    if len(updated) != target_frames:
        append_log(
            log_path,
            f"[STEP1.5.5] Keyframe count mismatch after reallocation: {len(updated)} vs {target_frames}.",
        )
        return current_indices
    return updated


def _ensure_proxy_masks(
    *,
    proxy_video: Path,
    bbox_json: Path,
    workdir: Path,
    mask_dir: Path,
    arm_swap_cfg: ArmSwapConfig,
    env: Dict[str, str],
    log_path: Path,
) -> Path:
    if not bbox_json.exists() or bbox_json.stat().st_size == 0:
        proxy_masks = ensure_qwen_sam_cache(
            project_root=arm_swap_cfg.project_root,
            conda_env=arm_swap_cfg.conda_env,
            orig_video=proxy_video,
            workdir=workdir,
            qwen_prompt=arm_swap_cfg.qwen_prompt,
            env=env,
            log_path=log_path,
            log_label="STEP1.5.5",
        )
        return proxy_masks
    mask_source = find_mask_dir_with_files(mask_dir)
    if mask_source:
        return mask_source
    run_arm_swap_sam_masks(
        project_root=arm_swap_cfg.project_root,
        conda_env=arm_swap_cfg.conda_env,
        orig_video=proxy_video,
        workdir=workdir,
        bbox_json=bbox_json,
        mask_dir=mask_dir,
        cleanup_workdir=False,
        env=env,
        log_path=log_path,
        log_label="STEP1.5.5",
    )
    mask_source = find_mask_dir_with_files(mask_dir)
    if not mask_source:
        raise FileNotFoundError(f"Proxy SAM masks missing after generation: {mask_dir}")
    return mask_source


def run_proxy_iou_preflight(
    *,
    original_video: Path,
    keyframes_video: Path,
    keyframes_frame_map: Path | None,
    preprocess_cfg: PreprocessConfig,
    num_video_frames_per_chunk: int,
    num_conditional_frames: int,
    postprocess_cfg: PostprocessConfig,
    arm_swap_cfg: ArmSwapConfig,
    cache_cfg: CacheConfig,
    proxy_cfg: ProxyPreflightSettings,
    work_dir: Path,
    env: Dict[str, str],
    log_path: Path,
    run_gpus: list[str] | None,
    job_index: int,
) -> tuple[Path, Path]:
    if not proxy_cfg.enable:
        if keyframes_frame_map is None:
            raise ValueError("keyframes_frame_map is required when proxy preflight is disabled.")
        return keyframes_video, keyframes_frame_map

    if proxy_cfg.max_iters <= 0:
        append_log(log_path, "[STEP1.5.5] max_iters <= 0; skipping proxy preflight.")
        if keyframes_frame_map is None:
            raise ValueError("keyframes_frame_map is required when proxy preflight is disabled.")
        return keyframes_video, keyframes_frame_map

    dataset_root = find_dataset_root(original_video)
    parquet_path = parquet_path_for_video(original_video, dataset_root=dataset_root)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found for proxy preflight: {parquet_path}")

    original_stats = probe_video_stats(original_video, env=env, log_path=log_path)
    if original_stats.frame_count is None or original_stats.frame_count <= 0:
        raise RuntimeError("Original video frame_count required for proxy preflight.")

    states, frame_indices, state_names, dexhand_names = _load_parquet_states(
        parquet_path,
        dataset_root=dataset_root,
        log_path=log_path,
    )
    stats_mean, stats_std = _load_state_stats(dataset_root, state_dim=states.shape[1], log_path=log_path)
    motion_series = _motion_per_frame(
        states=states,
        frame_indices=frame_indices,
        state_names=state_names,
        dexhand_names=dexhand_names,
        pose_subset=preprocess_cfg.parquet_pose_subset,
        stats_mean=stats_mean,
        stats_std=stats_std,
        smooth_win=preprocess_cfg.keyframe_smooth_win,
        total_frames=original_stats.frame_count,
        log_path=log_path,
    )

    current_video = keyframes_video
    current_frame_map = keyframes_frame_map
    target_frames = None
    if current_frame_map is not None:
        target_frames = len(_load_frame_map(current_frame_map))
    if target_frames is None or target_frames <= 1:
        target_frames = (
            preprocess_cfg.keyframe_target_frames
            if preprocess_cfg.keyframe_target_frames is not None
            else num_video_frames_per_chunk
        )
        if target_frames <= 1:
            append_log(log_path, "[STEP1.5.5] Invalid keyframe count; skipping proxy preflight.")
            if keyframes_frame_map is None:
                raise ValueError("keyframes_frame_map is required when skipping proxy preflight.")
            return keyframes_video, keyframes_frame_map
    if original_stats.duration_sec is None or original_stats.duration_sec <= 0:
        raise RuntimeError("Original video duration required for keyframes FPS calculation.")
    speed_fps = target_frames / original_stats.duration_sec

    keyframes_dir = keyframes_video.parent
    opt_video_path = keyframes_dir / "keyframes_opt.mp4"
    opt_frame_map_path = frame_map_path_for(opt_video_path)
    proxy_dir = keyframes_dir / "proxy_preflight"
    proxy_dir.mkdir(parents=True, exist_ok=True)

    if (
        cache_cfg.enable
        and opt_video_path.exists()
        and opt_video_path.stat().st_size > 0
        and opt_frame_map_path.exists()
        and opt_frame_map_path.stat().st_size > 0
    ):
        append_log(log_path, f"[STEP1.5.5] cache hit: {opt_video_path}")
        console_print(f"[STEP1.5.5] cache hit: {opt_video_path.name}")
        return opt_video_path, opt_frame_map_path

    preprocess_settings = PreprocessSettings(
        width=preprocess_cfg.width,
        height=preprocess_cfg.height,
        keep_aspect=preprocess_cfg.keep_aspect,
        pre_chunk_mode=preprocess_cfg.pre_chunk_mode,
        pre_chunk_policy=preprocess_cfg.pre_chunk_policy,
        pre_chunk_clip_where=preprocess_cfg.pre_chunk_clip_where,
        pre_chunk_max_adjust=preprocess_cfg.pre_chunk_max_adjust,
        pre_chunk_force_chunks=preprocess_cfg.pre_chunk_force_chunks,
        num_video_frames_per_chunk=num_video_frames_per_chunk,
        num_conditional_frames=num_conditional_frames,
        keyframe_sampling=preprocess_cfg.keyframe_sampling,
        keyframe_target_frames=preprocess_cfg.keyframe_target_frames,
        keyframe_smooth_win=preprocess_cfg.keyframe_smooth_win,
        keyframe_max_gap=preprocess_cfg.keyframe_max_gap,
        parquet_pose_subset=preprocess_cfg.parquet_pose_subset,
    )

    if current_frame_map is None:
        uniform_video_path = keyframes_dir / "keyframes_uniform.mp4"
        uniform_frame_map_path = frame_map_path_for(uniform_video_path)
        if not uniform_video_path.exists() or not uniform_frame_map_path.exists():
            uniform_indices = _uniform_frame_map(original_stats.frame_count, target_frames)
            write_keyframes_from_indices(
                video_path=original_video,
                output_video_path=uniform_video_path,
                frame_map_path=uniform_frame_map_path,
                keyframe_indices_path=None,
                original_total_frames=original_stats.frame_count,
                indices=sorted(set(uniform_indices)),
                settings=preprocess_settings,
                env=env,
                log_path=log_path,
                title="STEP1.5.5 Keyframe extract (uniform)",
                output_fps=speed_fps,
            )
        current_video = uniform_video_path
        current_frame_map = uniform_frame_map_path

    sam_cache_key = video_cache_key(original_video, mode=cache_cfg.fingerprint_mode)
    sam_workdir = arm_swap_cfg.workdir_root / sam_cache_key
    masks_dir = ensure_qwen_sam_cache(
        project_root=arm_swap_cfg.project_root,
        conda_env=arm_swap_cfg.conda_env,
        orig_video=original_video,
        workdir=sam_workdir,
        qwen_prompt=arm_swap_cfg.qwen_prompt,
        env=env,
        log_path=log_path,
        log_label="STEP1.5.5",
    )
    qwen_bbox_json = masks_dir.parent / "qwen_boxes.json"
    if not qwen_bbox_json.exists() or qwen_bbox_json.stat().st_size == 0:
        masks_dir = ensure_qwen_sam_cache(
            project_root=arm_swap_cfg.project_root,
            conda_env=arm_swap_cfg.conda_env,
            orig_video=original_video,
            workdir=sam_workdir,
            qwen_prompt=arm_swap_cfg.qwen_prompt,
            env=env,
            log_path=log_path,
            log_label="STEP1.5.5",
        )
        qwen_bbox_json = masks_dir.parent / "qwen_boxes.json"
        if not qwen_bbox_json.exists() or qwen_bbox_json.stat().st_size == 0:
            raise FileNotFoundError(f"Qwen bbox json not found for proxy preflight: {qwen_bbox_json}")

    rife_backend = proxy_cfg.rife_backend or postprocess_cfg.rife_backend

    for iteration in range(proxy_cfg.max_iters):
        append_log(log_path, f"[STEP1.5.5] iter={iteration} proxy preflight start")
        current_indices = _load_frame_map(current_frame_map)
        keyframes_before = set(current_indices)

        proxy_restore_path = proxy_dir / "proxy_restore.mp4"
        if proxy_restore_path.exists():
            proxy_restore_path.unlink()
        shutil.copy2(current_video, proxy_restore_path)

        restore_duration(
            output_dir=proxy_dir,
            input_video_path=original_video,
            original_stats=original_stats,
            mode=proxy_cfg.restore_mode,
            rife_bin=postprocess_cfg.rife_bin,
            rife_model=postprocess_cfg.rife_model,
            rife_backend=rife_backend,
            rife_pytorch_repo=postprocess_cfg.rife_pytorch_repo,
            rife_pytorch_model_dir=postprocess_cfg.rife_pytorch_model_dir,
            rife_pytorch_env=postprocess_cfg.rife_pytorch_env,
            rife_scale=postprocess_cfg.rife_scale,
            rife_keep_audio=postprocess_cfg.rife_keep_audio,
            rife_time_steps_mode=postprocess_cfg.rife_time_steps_mode,
            rife_time_steps_cache=(Path(postprocess_cfg.rife_time_steps_cache) if postprocess_cfg.rife_time_steps_cache else None),
            rife_sam_mask_dir=(Path(postprocess_cfg.rife_sam_mask_dir) if postprocess_cfg.rife_sam_mask_dir else (work_dir / "rife_sam_cache")),
            fingerprint_mode=cache_cfg.fingerprint_mode,
            rife_time_steps_force_sam=postprocess_cfg.rife_time_steps_force_sam,
            rife_sam_generate=postprocess_cfg.rife_sam_generate,
            rife_sam_bbox_json=(Path(postprocess_cfg.rife_sam_bbox_json) if postprocess_cfg.rife_sam_bbox_json else None),
            rife_sam_cleanup_workdir=postprocess_cfg.rife_sam_cleanup_workdir,
            rife_sam_project_root=arm_swap_cfg.project_root,
            rife_sam_conda_env=arm_swap_cfg.conda_env,
            rife_sam_qwen_prompt=arm_swap_cfg.qwen_prompt,
            rife_sam_erode=arm_swap_cfg.erode,
            rife_sam_feather=arm_swap_cfg.feather,
            rife_sam_dilate=arm_swap_cfg.dilate,
            rife_sam_no_workdir_write=arm_swap_cfg.no_workdir_write,
            rife_sam_cache_workdir=sam_workdir,
            rife_pose_subset=postprocess_cfg.rife_pose_subset,
            rife_time_steps_debug=postprocess_cfg.rife_time_steps_debug,
            rife_cache_root=work_dir / "rife_cache",
            frame_map_path=current_frame_map,
            run_gpus=run_gpus,
            job_index=job_index,
            env=env,
            log_path=log_path,
            log_label="STEP1.5.5",
        )

        mask_suffix = short_hash(",".join(str(idx) for idx in current_indices), 8)
        proxy_mask_dir = _ensure_proxy_masks(
            proxy_video=proxy_restore_path,
            bbox_json=qwen_bbox_json,
            workdir=proxy_dir / f"proxy_sam_workdir_{mask_suffix}",
            mask_dir=proxy_dir / f"proxy_masks_arm_{mask_suffix}",
            arm_swap_cfg=arm_swap_cfg,
            env=env,
            log_path=log_path,
        )

        iou_series = _compute_iou_series(masks_dir, proxy_mask_dir)
        iou_path = proxy_dir / "proxy_iou.json"
        iou_path.write_text(json.dumps(iou_series, ensure_ascii=False, indent=2), encoding="utf-8")
        bad_frames = [idx for idx, iou in enumerate(iou_series) if iou < proxy_cfg.iou_threshold]
        append_log(
            log_path,
            f"[STEP1.5.5] iter={iteration} bad_frames={len(bad_frames)}",
        )
        if bad_frames:
            append_log(
                log_path,
                f"[STEP1.5.5] iter={iteration} bad_frame_indices={bad_frames}",
            )
        if not bad_frames:
            if proxy_cfg.cache_iou_csv:
                csv_path = keyframes_dir / "proxy_preflight_iou.csv"
                append_log(
                    log_path,
                    f"[STEP1.5.5] iter={iteration} write proxy_preflight_iou.csv -> {csv_path}",
                )
                console_print(f"[STEP1.5.5] CSV -> {csv_path}")
                _append_preflight_iou_csv(
                    csv_path,
                    iteration,
                    iou_series,
                    keyframes_before=keyframes_before,
                    keyframes_after=keyframes_before,
                    bad_frames=set(),
                )
                plot_dir = proxy_dir / "iou_plots"
                png_path = plot_dir / f"proxy_preflight_iou_iter_{iteration}.png"
                _write_iou_plot_png(
                    png_path=png_path,
                    iou_series=iou_series,
                    iteration=iteration,
                )
            append_log(log_path, f"[STEP1.5.5] iter={iteration} IoU OK; stop.")
            break

        updated_indices = _reallocate_keyframes(
            current_indices=current_indices,
            bad_frames=bad_frames,
            iou_series=iou_series,
            motion_series=motion_series,
            target_frames=target_frames,
            iou_high_threshold=proxy_cfg.iou_high_threshold,
            motion_low_quantile=proxy_cfg.motion_low_quantile,
            log_path=log_path,
        )
        if proxy_cfg.cache_iou_csv:
            csv_path = keyframes_dir / "proxy_preflight_iou.csv"
            append_log(
                log_path,
                f"[STEP1.5.5] iter={iteration} write proxy_preflight_iou.csv -> {csv_path}",
            )
            console_print(f"[STEP1.5.5] CSV -> {csv_path}")
            _append_preflight_iou_csv(
                csv_path,
                iteration,
                iou_series,
                keyframes_before=keyframes_before,
                keyframes_after=set(updated_indices),
                bad_frames=set(bad_frames),
            )
            plot_dir = proxy_dir / "iou_plots"
            png_path = plot_dir / f"proxy_preflight_iou_iter_{iteration}.png"
            _write_iou_plot_png(
                png_path=png_path,
                iou_series=iou_series,
                iteration=iteration,
            )
        if updated_indices == current_indices:
            append_log(log_path, f"[STEP1.5.5] iter={iteration} no changes; stop.")
            break

        write_keyframes_from_indices(
            video_path=original_video,
            output_video_path=opt_video_path,
            frame_map_path=opt_frame_map_path,
            keyframe_indices_path=None,
            original_total_frames=original_stats.frame_count,
            indices=updated_indices,
            settings=preprocess_settings,
            env=env,
            log_path=log_path,
            title="STEP1.5.5 Keyframe extract (proxy iou)",
            output_fps=speed_fps,
        )
        current_video = opt_video_path
        current_frame_map = opt_frame_map_path

    if current_video == keyframes_video and current_frame_map == keyframes_frame_map:
        shutil.copy2(keyframes_video, opt_video_path)
        opt_frame_map_path.write_text(
            keyframes_frame_map.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        current_video = opt_video_path
        current_frame_map = opt_frame_map_path

    return current_video, current_frame_map
