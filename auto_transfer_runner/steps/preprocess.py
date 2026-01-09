from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from ..utils.cleanup import safe_unlink
from ..utils.dataset_paths import find_dataset_root, parquet_path_for_video
from ..utils.locking import FileLock, FileLockConfig
from ..utils.logging_utils import append_log, console_print
from ..utils.subprocess_utils import run_cmd


@dataclass(frozen=True)
class PreprocessSettings:
    width: int | None
    height: int | None
    keep_aspect: bool
    pre_chunk_mode: str
    pre_chunk_policy: str
    pre_chunk_clip_where: str
    pre_chunk_max_adjust: int
    pre_chunk_force_chunks: int | None
    num_video_frames_per_chunk: int
    num_conditional_frames: int
    keyframe_sampling: str
    keyframe_target_frames: int | None
    keyframe_smooth_win: int
    keyframe_max_gap: int
    parquet_pose_subset: str


def frame_map_path_for(pre_video_path: Path) -> Path:
    return pre_video_path.with_name(f"{pre_video_path.stem}__frame_map.json")


def preprocess_video(
    *,
    video_path: Path,
    settings: PreprocessSettings,
    output_root: Path,
    input_root: Path | None,
    env: Dict[str, str],
    log_path: Path,
    lock_timeout_sec: int,
    stale_lock_sec: int,
) -> Path:
    keyframe_video = _maybe_generate_keyframes(
        video_path=video_path,
        settings=settings,
        output_root=output_root,
        input_root=input_root,
        env=env,
        log_path=log_path,
    )
    if keyframe_video is not None:
        append_log(log_path, f"[PREPROCESS KEYFRAME] output={keyframe_video}")
        console_print(f"[PREPROCESS KEYFRAME] {keyframe_video.name}")
        return keyframe_video
    output_path = _build_output_path(
        video_path,
        settings,
        output_root=output_root,
        input_root=input_root,
    )
    effective_fps = _probe_video_fps(video_path, env=env, log_path=log_path)
    if output_path.exists() and output_path.stat().st_size > 0:
        append_log(log_path, f"[PREPROCESS CACHE HIT] {output_path}")
        console_print(f"[PREPROCESS CACHE HIT] {output_path.name}")
        return _align_preprocess_chunks(
            pre_video_path=output_path,
            settings=settings,
            input_fps=effective_fps,
            env=env,
            log_path=log_path,
            lock_timeout_sec=lock_timeout_sec,
            stale_lock_sec=stale_lock_sec,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_name(f".{output_path.name}.lock")
    lock_cfg = FileLockConfig(timeout_sec=lock_timeout_sec, stale_lock_sec=stale_lock_sec)

    with FileLock(lock_path, lock_cfg):
        if output_path.exists() and output_path.stat().st_size > 0:
            append_log(log_path, f"[PREPROCESS CACHE HIT AFTER LOCK] {output_path}")
            console_print(f"[PREPROCESS CACHE HIT] {output_path.name}")
            return _align_preprocess_chunks(
                pre_video_path=output_path,
                settings=settings,
                input_fps=effective_fps,
                env=env,
                log_path=log_path,
                lock_timeout_sec=lock_timeout_sec,
                stale_lock_sec=stale_lock_sec,
            )

        tmp_path = output_path.with_name(f".{output_path.stem}__tmp{output_path.suffix}")
        safe_unlink(tmp_path, log_path=log_path)

        scale_filter = _build_scale_filter(
            width=settings.width,
            height=settings.height,
            keep_aspect=settings.keep_aspect,
        )
        filters = [scale_filter, f"fps={effective_fps}"]
        filter_chain = ",".join(filters)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            filter_chain,
            "-vsync",
            "cfr",
            "-r",
            str(effective_fps),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(tmp_path),
        ]
        run_cmd(
            cmd,
            cwd=None,
            env=env or dict(os.environ),
            log_path=log_path,
            title="STEP1: Preprocess video (CFR/scale)",
        )
        tmp_path.replace(output_path)
        append_log(log_path, f"[PREPROCESS DONE] output={output_path}")
        console_print(f"[PREPROCESS DONE] {output_path.name}")
        return _align_preprocess_chunks(
            pre_video_path=output_path,
            settings=settings,
            input_fps=effective_fps,
            env=env,
            log_path=log_path,
            lock_timeout_sec=lock_timeout_sec,
            stale_lock_sec=stale_lock_sec,
        )


def _evenize(value: int | None) -> int | None:
    if value is None:
        return None
    if value <= 0:
        raise ValueError("preprocess width/height must be positive")
    if value % 2 == 0:
        return value
    return max(2, value - 1)


def _build_output_path(
    video_path: Path,
    settings: PreprocessSettings,
    *,
    output_root: Path,
    input_root: Path | None,
) -> Path:
    width = _evenize(settings.width)
    height = _evenize(settings.height)
    fps_label = "orig"
    w_label = str(width) if width is not None else "auto"
    h_label = str(height) if height is not None else "auto"
    aspect_label = "ka1" if settings.keep_aspect else "ka0"
    name = f"{video_path.stem}__pre_fps{fps_label}_w{w_label}_h{h_label}_{aspect_label}{video_path.suffix}"
    relative_dir = None
    if input_root is not None:
        try:
            relative_dir = video_path.parent.relative_to(input_root)
        except ValueError:
            relative_dir = None
    output_dir = output_root if relative_dir is None else output_root / relative_dir
    return output_dir / name


def _align_preprocess_chunks(
    *,
    pre_video_path: Path,
    settings: PreprocessSettings,
    input_fps: float,
    env: Dict[str, str],
    log_path: Path,
    lock_timeout_sec: int,
    stale_lock_sec: int,
) -> Path:
    force_chunks = settings.pre_chunk_force_chunks
    mode = settings.pre_chunk_mode
    if force_chunks is None and mode == "none":
        return pre_video_path

    total_frames = _probe_frame_count(pre_video_path, env=env, log_path=log_path)
    L = settings.num_video_frames_per_chunk
    C = settings.num_conditional_frames
    S = L - C
    if S <= 0:
        append_log(log_path, f"[PRE_CHUNK SKIP] invalid S=L-C ({L}-{C})")
        console_print("[PRE_CHUNK SKIP] invalid chunk config")
        return pre_video_path

    if force_chunks is not None:
        mode = "speed"
        n = force_chunks
        policy_label = "force"
    else:
        if total_frames <= L:
            n_up = 1
            n_down = 1
        else:
            n_up = 1 + math.ceil((total_frames - L) / S)
            n_down = 1 + math.floor((total_frames - L) / S)

        if settings.pre_chunk_policy == "up":
            n = n_up
        else:
            n = n_down
        policy_label = settings.pre_chunk_policy

    target_frames = L + (n - 1) * S
    drop = total_frames - target_frames
    if (
        force_chunks is None
        and settings.pre_chunk_max_adjust is not None
        and abs(drop) > settings.pre_chunk_max_adjust
    ):
        append_log(
            log_path,
            (
                "[PRE_CHUNK SKIP] adjust too large "
                f"T={total_frames} T_target={target_frames} drop={drop} "
                f"max_adjust={settings.pre_chunk_max_adjust}"
            ),
        )
        console_print("[PRE_CHUNK SKIP] adjust too large")
        return pre_video_path

    if mode == "clip" and target_frames > total_frames:
        append_log(
            log_path,
            f"[PRE_CHUNK SKIP] clip cannot increase frames: T={total_frames} T_target={target_frames}",
        )
        console_print("[PRE_CHUNK SKIP] clip cannot increase frames")
        return pre_video_path
    if mode == "speed" and target_frames > total_frames:
        append_log(
            log_path,
            f"[PRE_CHUNK SKIP] speed cannot increase frames: T={total_frames} T_target={target_frames}",
        )
        console_print("[PRE_CHUNK SKIP] speed cannot increase frames")
        return pre_video_path

    aligned_path = _build_aligned_output_path(
        pre_video_path,
        mode=mode,
        policy=policy_label,
        L=L,
        C=C,
        n=n,
        T=total_frames,
        T_target=target_frames,
    )
    if aligned_path.exists() and aligned_path.stat().st_size > 0:
        append_log(log_path, f"[PRE_CHUNK CACHE HIT] {aligned_path}")
        console_print(f"[PRE_CHUNK CACHE HIT] {aligned_path.name}")
        return aligned_path

    aligned_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = aligned_path.with_name(f".{aligned_path.name}.lock")
    lock_cfg = FileLockConfig(timeout_sec=lock_timeout_sec, stale_lock_sec=stale_lock_sec)
    with FileLock(lock_path, lock_cfg):
        if aligned_path.exists() and aligned_path.stat().st_size > 0:
            append_log(log_path, f"[PRE_CHUNK CACHE HIT AFTER LOCK] {aligned_path}")
            console_print(f"[PRE_CHUNK CACHE HIT] {aligned_path.name}")
            return aligned_path

        tmp_path = aligned_path.with_name(f".{aligned_path.stem}__tmp{aligned_path.suffix}")
        safe_unlink(tmp_path, log_path=log_path)

        if mode == "clip":
            clip_where = settings.pre_chunk_clip_where
            if clip_where == "tail":
                filters = None
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(pre_video_path),
                    "-frames:v",
                    str(target_frames),
                    "-vsync",
                    "cfr",
                    "-r",
                    str(input_fps),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(tmp_path),
                ]
            else:
                if clip_where == "head":
                    drop_head = drop
                elif clip_where == "center":
                    drop_head = drop // 2
                else:
                    raise ValueError(f"Unknown pre_chunk_clip_where: {clip_where}")
                end_frame = drop_head + target_frames
                filters = f"trim=start_frame={drop_head}:end_frame={end_frame},setpts=PTS-STARTPTS"
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(pre_video_path),
                    "-vf",
                    filters,
                    "-vsync",
                    "cfr",
                    "-r",
                    str(input_fps),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(tmp_path),
                ]
            k_value = None
        elif mode == "speed":
            target_fps = input_fps * (target_frames / total_frames)
            filters = f"fps={target_fps:.6f}"
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(pre_video_path),
                "-vf",
                filters,
                "-frames:v",
                str(target_frames),
                "-vsync",
                "cfr",
                "-r",
                f"{target_fps:.6f}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(tmp_path),
            ]
        else:
            raise ValueError(f"Unknown pre_chunk_mode: {mode}")

        title = f"STEP1.5 Align chunks ({mode})"
        run_cmd(
            cmd,
            cwd=None,
            env=env or dict(os.environ),
            log_path=log_path,
            title=title,
        )
        tmp_path.replace(aligned_path)

        if mode == "speed":
            frame_map_path = frame_map_path_for(aligned_path)
            frame_map = _build_frame_map(total_frames, target_frames)
            frame_map_payload = {
                "original_total_frames": total_frames,
                "sampled_indices": frame_map,
            }
            frame_map_path.write_text(json.dumps(frame_map_payload, ensure_ascii=False), encoding="utf-8")

        log_parts = [
            f"[PRE_CHUNK DONE] mode={mode}",
            f"policy={policy_label}",
            f"T={total_frames}",
            f"T_target={target_frames}",
            f"drop={drop}",
            f"n={n}",
        ]
        if mode == "speed":
            log_parts.append(f"fps={target_fps:.6f}")
        log_parts.append(f"output={aligned_path}")
        append_log(log_path, " ".join(log_parts))
        console_print(" ".join(log_parts))
        return aligned_path


def _build_frame_map(total_frames: int, target_frames: int) -> list[int]:
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


def _build_aligned_output_path(
    pre_video_path: Path,
    *,
    mode: str,
    policy: str,
    L: int,
    C: int,
    n: int,
    T: int,
    T_target: int,
) -> Path:
    suffix = f"__chunk_{mode}_{policy}_L{L}_C{C}_n{n}_T{T}_Tt{T_target}"
    return pre_video_path.with_name(f"{pre_video_path.stem}{suffix}{pre_video_path.suffix}")


def _probe_frame_count(pre_video_path: Path, *, env: Dict[str, str], log_path: Path) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(pre_video_path),
    ]
    append_log(log_path, "[PRE_CHUNK] " + " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env or dict(os.environ),
        check=False,
    )
    if result.returncode != 0:
        append_log(log_path, f"[PRE_CHUNK] ffprobe failed: {result.stderr.strip()}")
        raise RuntimeError("ffprobe failed while counting frames.")
    output = result.stdout.strip()
    if not output.isdigit():
        append_log(log_path, f"[PRE_CHUNK] ffprobe output not integer: {output}")
        raise RuntimeError("ffprobe returned non-integer frame count.")
    return int(output)


def _build_scale_filter(*, width: int | None, height: int | None, keep_aspect: bool) -> str:
    width = _evenize(width)
    height = _evenize(height)

    if keep_aspect:
        if width and height:
            return (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                "scale=trunc(iw/2)*2:trunc(ih/2)*2"
            )
        if width:
            return f"scale={width}:-2"
        if height:
            return f"scale=-2:{height}"
    else:
        if width and height:
            return f"scale={width}:{height}"
        if width:
            return f"scale={width}:trunc(ih/2)*2"
        if height:
            return f"scale=trunc(iw/2)*2:{height}"

    return "scale=trunc(iw/2)*2:trunc(ih/2)*2"


def _keyframe_output_dir(video_path: Path, *, output_root: Path, input_root: Path | None) -> Path:
    relative_dir = None
    if input_root is not None:
        try:
            relative_dir = video_path.parent.relative_to(input_root)
        except ValueError:
            relative_dir = None
    base_dir = output_root if relative_dir is None else output_root / relative_dir
    return base_dir / f"{video_path.stem}__keyframes"


def _keyframe_target_from_chunks(
    *,
    total_frames: int,
    settings: PreprocessSettings,
) -> int:
    L = settings.num_video_frames_per_chunk
    C = settings.num_conditional_frames
    S = L - C
    if S <= 0:
        return L
    if settings.pre_chunk_force_chunks is not None:
        n = settings.pre_chunk_force_chunks
    else:
        if total_frames <= L:
            n_up = 1
            n_down = 1
        else:
            n_up = 1 + math.ceil((total_frames - L) / S)
            n_down = 1 + math.floor((total_frames - L) / S)
        n = n_up if settings.pre_chunk_policy == "up" else n_down
    return L + (n - 1) * S


def _log_warning(log_path: Path, message: str) -> None:
    append_log(log_path, f"[WARN] {message}")
    console_print(f"[WARN] {message}")


def _log_info(log_path: Path, message: str) -> None:
    append_log(log_path, message)
    console_print(message)


def _load_state_names(dataset_root: Path, *, state_dim: int) -> tuple[list[str], set[str]]:
    info_path = dataset_root / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

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
    dexhand_names: list[str] = []
    if isinstance(state_info, dict):
        names = state_info.get("names")
    if isinstance(names, dict):
        dexhand_names = list(names.get("dexhand", []))
        sensors_names = list(names.get("sensors_raw", []))
        names = dexhand_names + sensors_names
    if not isinstance(names, list):
        raise ValueError("Unable to parse observation.state names from meta/info.json")
    if len(names) != state_dim:
        raise ValueError(
            f"observation.state names length mismatch: got {len(names)} expected {state_dim}"
        )
    return names, set(dexhand_names)


def _load_state_stats(dataset_root: Path, *, state_dim: int, log_path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    stats_path = dataset_root / "meta" / "stats.json"
    if not stats_path.exists():
        _log_warning(log_path, f"stats.json not found at {stats_path}; skipping normalization")
        return None, None
    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
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
        _log_warning(log_path, "stats.json missing observation.state; skipping normalization")
        return None, None
    mean = state_stats.get("mean")
    std = state_stats.get("std")
    if not isinstance(mean, list) or not isinstance(std, list):
        _log_warning(log_path, "stats.json missing mean/std arrays; skipping normalization")
        return None, None
    if len(mean) != state_dim or len(std) != state_dim:
        _log_warning(
            log_path,
            f"stats.json mean/std length mismatch (mean={len(mean)} std={len(std)} expected={state_dim}); skipping",
        )
        return None, None
    return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)


def _load_parquet_states(
    parquet_path: Path,
    *,
    dataset_root: Path,
    log_path: Path,
) -> tuple[np.ndarray, list[int], list[str], set[str]]:
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
                try:
                    state_values.append(item["state"])
                except Exception as exc:
                    raise ValueError("Unable to parse observation.state from parquet observation column") from exc
    else:
        raise ValueError("Parquet missing observation.state column")

    if "frame_index" in names:
        frame_indices = [int(x) for x in table.column("frame_index").to_pylist()]
    else:
        frame_indices = list(range(len(state_values)))

    states = np.asarray(state_values, dtype=float)
    if states.ndim == 1 and len(states) > 0:
        try:
            states = np.stack(state_values, axis=0).astype(float)
        except Exception as exc:
            raise ValueError(f"Unexpected observation.state shape: {states.shape}") from exc
    if states.ndim != 2:
        raise ValueError(f"Unexpected observation.state shape: {states.shape}")
    state_names, dexhand_names = _load_state_names(dataset_root, state_dim=states.shape[1])
    return states, frame_indices, state_names, dexhand_names


def _moving_average(values: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return values
    kernel = np.ones(win, dtype=float) / float(win)
    return np.convolve(values, kernel, mode="same")


def _enforce_max_gap(
    indices: list[int],
    motion_scores: np.ndarray,
    *,
    max_gap: int,
    target_frames: int,
) -> list[int]:
    if max_gap <= 0:
        return indices

    indices = sorted(set(indices))
    if indices[0] != 0:
        indices[0] = 0

    expanded: list[int] = [indices[0]]
    for start, end in zip(indices[:-1], indices[1:]):
        gap = end - start
        if gap > max_gap:
            inserts = int(math.ceil(gap / max_gap)) - 1
            for step in range(1, inserts + 1):
                pos = int(round(start + (gap * step) / (inserts + 1)))
                if pos <= expanded[-1]:
                    pos = expanded[-1] + 1
                if pos < end:
                    expanded.append(pos)
        expanded.append(end)

    expanded = sorted(set(expanded))
    while len(expanded) > target_frames:
        removable: list[int] = []
        for i in range(1, len(expanded) - 1):
            prev_idx = expanded[i - 1]
            next_idx = expanded[i + 1]
            if next_idx - prev_idx <= max_gap:
                removable.append(expanded[i])
        if not removable:
            break
        removable.sort(key=lambda idx: motion_scores[idx])
        expanded.remove(removable[0])

    if len(expanded) > target_frames:
        expanded = expanded[:target_frames - 1] + [expanded[-1]]

    expanded[0] = 0
    return expanded


def _select_pose_indices(
    state_names: list[str],
    dexhand_names: set[str],
    *,
    subset: str,
    log_path: Path,
) -> tuple[list[int], str]:
    subset = subset or "full"
    if subset == "full":
        keep_idx = list(range(len(state_names)))
    else:
        prefixes: list[str] = []
        if subset in {"zarm_pos", "zarm_pos_vel", "zarm_pos_vel_hand"}:
            prefixes.append("position.zarm_")
        if subset in {"zarm_pos_vel", "zarm_pos_vel_hand"}:
            prefixes.append("velocity.zarm_")
        include_dexhand = subset == "zarm_pos_vel_hand"
        keep_idx = [
            idx
            for idx, name in enumerate(state_names)
            if any(name.startswith(prefix) for prefix in prefixes)
            or (include_dexhand and name in dexhand_names)
        ]
    if not keep_idx:
        _log_warning(
            log_path,
            (
                f"parquet_pose_subset={subset} produced 0 dims "
                f"(prefixes={prefixes}); falling back to full"
            ),
        )
        keep_idx = list(range(len(state_names)))
        subset = "full"

    sample_names = [state_names[idx] for idx in keep_idx[:10]]
    _log_info(
        log_path,
        (
            "[KEYFRAME] parquet_pose_subset="
            f"{subset} dims={len(keep_idx)} sample_names={sample_names}"
        ),
    )
    return keep_idx, subset


def _compute_motion_diffs(
    states: np.ndarray,
    *,
    keep_idx: list[int],
    stats_mean: np.ndarray | None,
    stats_std: np.ndarray | None,
    smooth_win: int,
    log_path: Path,
) -> np.ndarray:
    pose = states[:, keep_idx]
    if stats_mean is not None and stats_std is not None:
        mean_sel = stats_mean[keep_idx]
        std_sel = stats_std[keep_idx]
        pose = (pose - mean_sel) / (std_sel + 1e-6)
    else:
        _log_warning(log_path, "Normalization skipped (stats.json unavailable or invalid)")

    T = pose.shape[0]
    diffs = np.zeros(T, dtype=float)
    diffs[1:] = np.linalg.norm(pose[1:] - pose[:-1], axis=1)
    diffs = _moving_average(diffs, smooth_win)
    if diffs.size:
        p10, p50, p90 = np.percentile(diffs, [10, 50, 90]).tolist()
        _log_info(log_path, f"[KEYFRAME] diffs_p10={p10:.6f} p50={p50:.6f} p90={p90:.6f}")
    return diffs


def _select_keyframes_from_states(
    states: np.ndarray,
    *,
    state_names: list[str],
    dexhand_names: set[str],
    target_frames: int,
    smooth_win: int,
    max_gap: int,
    pose_subset: str,
    stats_mean: np.ndarray | None,
    stats_std: np.ndarray | None,
    log_path: Path,
) -> list[int]:
    T = states.shape[0]
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    if T < target_frames:
        raise ValueError(f"Video frame count T={T} is less than target N={target_frames}")

    keep_idx, _ = _select_pose_indices(
        state_names,
        dexhand_names,
        subset=pose_subset,
        log_path=log_path,
    )
    diffs = _compute_motion_diffs(
        states,
        keep_idx=keep_idx,
        stats_mean=stats_mean,
        stats_std=stats_std,
        smooth_win=smooth_win,
        log_path=log_path,
    )
    nonzero = diffs[diffs > 0]
    if nonzero.size > 0:
        eps = 0.2 * float(np.percentile(nonzero, 10))
    else:
        eps = 1e-6
    diffs = np.maximum(diffs, eps)

    cumulative = np.cumsum(diffs)
    if target_frames == 1:
        indices = [0]
    else:
        delta = (cumulative[-1] - cumulative[0]) / (target_frames - 1)
        indices = []
        prev = -1
        for m in range(target_frames):
            target = cumulative[0] + delta * m
            idx = int(np.searchsorted(cumulative, target, side="left"))
            if idx <= prev:
                idx = prev + 1
            if idx >= T:
                idx = T - 1
            indices.append(idx)
            prev = idx
        indices[-1] = T - 1

    indices = _enforce_max_gap(indices, diffs, max_gap=max_gap, target_frames=target_frames)
    if len(indices) != target_frames:
        if len(indices) < target_frames:
            raise ValueError("Failed to sample enough keyframes after enforcing max_gap")
        indices = indices[:target_frames - 1] + [indices[-1]]
    indices[0] = 0
    indices[-1] = T - 1
    return indices


def _build_select_filter(indices: Iterable[int]) -> str:
    parts = [f"eq(n\\,{idx})" for idx in indices]
    return "+".join(parts)


def _maybe_generate_keyframes(
    *,
    video_path: Path,
    settings: PreprocessSettings,
    output_root: Path,
    input_root: Path | None,
    env: Dict[str, str],
    log_path: Path,
) -> Path | None:
    if settings.keyframe_sampling != "parquet_motion":
        return None

    try:
        dataset_root = find_dataset_root(video_path)
        parquet_path = parquet_path_for_video(video_path, dataset_root=dataset_root)
    except ValueError as exc:
        raise ValueError(f"Failed to derive parquet path for input_video_path={video_path}") from exc
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet not found for input video.\n"
            f"input_video_path={video_path}\n"
            f"parquet_path={parquet_path}"
        )

    states, frame_indices, state_names, dexhand_names = _load_parquet_states(
        parquet_path,
        dataset_root=dataset_root,
        log_path=log_path,
    )
    stats_mean, stats_std = _load_state_stats(dataset_root, state_dim=states.shape[1], log_path=log_path)
    total_frames = states.shape[0]
    video_frames = _probe_frame_count(video_path, env=env, log_path=log_path)
    append_log(log_path, f"[KEYFRAME] parquet={parquet_path}")
    append_log(log_path, f"[KEYFRAME] video_frames={video_frames} parquet_frames={total_frames}")
    console_print(f"[KEYFRAME] parquet={parquet_path}")
    console_print(f"[KEYFRAME] video_frames={video_frames} parquet_frames={total_frames}")
    if settings.keyframe_target_frames is not None:
        target_frames = settings.keyframe_target_frames
    elif settings.pre_chunk_mode == "speed" or settings.pre_chunk_force_chunks is not None:
        target_frames = _keyframe_target_from_chunks(total_frames=video_frames, settings=settings)
    else:
        target_frames = settings.num_video_frames_per_chunk
    if target_frames <= 0:
        raise ValueError("keyframe_target_frames must be positive")
    if total_frames < target_frames:
        raise ValueError(f"Video frames too short: T={total_frames} < N={target_frames}")

    keyframe_indices = _select_keyframes_from_states(
        states,
        state_names=state_names,
        dexhand_names=dexhand_names,
        target_frames=target_frames,
        smooth_win=settings.keyframe_smooth_win,
        max_gap=settings.keyframe_max_gap,
        pose_subset=settings.parquet_pose_subset,
        stats_mean=stats_mean,
        stats_std=stats_std,
        log_path=log_path,
    )
    mapped_indices = [frame_indices[idx] for idx in keyframe_indices]
    if any(mapped_indices[i] >= mapped_indices[i + 1] for i in range(len(mapped_indices) - 1)):
        raise ValueError("Keyframe indices are not strictly increasing after mapping frame_index")
    if max(mapped_indices) >= video_frames:
        raise ValueError(
            f"Keyframe index exceeds video frame count: max={max(mapped_indices)} video_frames={video_frames}"
        )

    output_dir = _keyframe_output_dir(video_path, output_root=output_root, input_root=input_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    keyframe_indices_path = output_dir / "keyframe_indices.txt"
    keyframe_indices_path.write_text(
        "\n".join(str(idx) for idx in mapped_indices) + "\n",
        encoding="utf-8",
    )
    output_video_path = output_dir / "keyframes.mp4"
    meta = {
        "input_video_path": str(video_path),
        "parquet_path": str(parquet_path),
        "N": target_frames,
        "max_gap": settings.keyframe_max_gap,
        "smooth_win": settings.keyframe_smooth_win,
        "keyframe_indices": mapped_indices,
        "output_video_path": str(output_video_path),
        "output_fps": 16,
    }
    (output_dir / "keyframe_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    select_filter = _build_select_filter(mapped_indices)
    scale_filter = _build_scale_filter(
        width=settings.width,
        height=settings.height,
        keep_aspect=settings.keep_aspect,
    )
    extract_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"{scale_filter},select='{select_filter}',setpts=N/(16*TB)",
        "-vsync",
        "cfr",
        "-r",
        "16",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video_path),
    ]
    run_cmd(
        extract_cmd,
        cwd=None,
        env=env or dict(os.environ),
        log_path=log_path,
        title="STEP1 Keyframe extract (parquet motion)",
    )

    gaps = [b - a for a, b in zip(mapped_indices[:-1], mapped_indices[1:])]
    max_gap_actual = max(gaps) if gaps else 0
    append_log(
        log_path,
        (
            "[KEYFRAME] "
            f"min={min(mapped_indices)} max={max(mapped_indices)} "
            f"max_gap_actual={max_gap_actual}"
        ),
    )
    console_print(
        "[KEYFRAME] "
        f"min={min(mapped_indices)} max={max(mapped_indices)} "
        f"max_gap_actual={max_gap_actual}"
    )
    return output_video_path


def _parse_fraction(value: str) -> float:
    if "/" in value:
        num_str, den_str = value.split("/", 1)
        den = float(den_str)
        if den == 0:
            return 0.0
        return float(num_str) / den
    return float(value)


def _probe_video_fps(video_path: Path, *, env: Dict[str, str], log_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]
    append_log(log_path, f"[PREPROCESS] ffprobe fps: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env or dict(os.environ),
        check=False,
    )
    if result.returncode != 0:
        append_log(log_path, f"[PREPROCESS] ffprobe failed: {result.stderr.strip()}")
        raise RuntimeError("ffprobe failed while probing fps.")
    fps = _parse_fraction(result.stdout.strip())
    if fps <= 0:
        append_log(log_path, f"[PREPROCESS] invalid fps: {fps}")
        raise RuntimeError("ffprobe returned invalid fps.")
    append_log(log_path, f"[PREPROCESS] probed fps={fps}")
    return fps
