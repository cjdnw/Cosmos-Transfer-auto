from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from ..utils.hashing import video_fingerprint
from ..utils.locking import FileLock, FileLockConfig
from ..utils.subprocess_utils import run_cmd, conda_run
from ..utils.logging_utils import append_log, console_print


class ControlType(str, Enum):
    DEPTH = "depth"
    EDGE = "edge"
    BLUR = "blur"
    SEG = "seg"
    COMBO = "combo"


def find_vis_mp4(control_dir: Path, video_path: Path) -> Optional[Path]:
    """Find a *_vis.mp4 in output directory, tolerant to naming differences."""
    if not control_dir.exists():
        return None

    candidates = list(control_dir.glob("*_vis.mp4"))
    if not candidates:
        return None

    stem = video_path.stem
    name = video_path.name

    for p in candidates:
        if stem in p.name or name in p.name:
            return p

    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def generate_control_depth(
    video_path: Path,
    output_dir: Path,
    depth_root: Path,
    conda_env: str,
    encoder: str,
    grayscale: bool,
    env: Dict[str, str],
    log_path: Path,
) -> Path:
    """Step3(depth): run Video-Depth-Anything and return the produced *_vis.mp4."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = conda_run(
        conda_env,
        [
            "python",
            "run.py",
            "--input_video",
            str(video_path),
            "--output_dir",
            str(output_dir),
            "--encoder",
            str(encoder),
        ],
    )
    if grayscale:
        cmd.append("--grayscale")

    run_cmd(cmd, cwd=depth_root, env=env, log_path=log_path, title="STEP3: Generate control (depth)")

    vis = find_vis_mp4(output_dir, video_path)
    if vis is None or not vis.exists() or vis.stat().st_size == 0:
        raise FileNotFoundError(f"Depth vis mp4 not found in: {output_dir}")

    return vis


def ensure_cached_depth_control(
    video_path: Path,
    work_dir: Path,
    depth_root: Path,
    conda_env: str,
    encoder: str,
    grayscale: bool,
    env: Dict[str, str],
    log_path: Path,
    enable_cache: bool,
    force_regen: bool,
    fingerprint_mode: str,
    lock_timeout_sec: int,
    stale_lock_sec: int,
) -> Path:
    """
    depth cache strategy:
    - cache_dir = <work_dir>/control_cache/depth/<fingerprint>/
    - If a valid *_vis.mp4 already exists: reuse it (skip generation)
    - Otherwise: lock and generate once (safe for multi-GPU concurrency)
    """
    fp = video_fingerprint(video_path, mode=fingerprint_mode)
    cache_dir = work_dir / "control_cache" / "depth" / fp
    cache_dir.mkdir(parents=True, exist_ok=True)

    if enable_cache and (not force_regen):
        vis = find_vis_mp4(cache_dir, video_path)
        if vis is not None and vis.exists() and vis.stat().st_size > 0:
            append_log(log_path, f"[CACHE HIT] depth control: {vis}")
            console_print(f"[CACHE HIT] depth control for {video_path.name}")
            return vis

    lock_path = cache_dir / ".lock"
    lock_cfg = FileLockConfig(timeout_sec=lock_timeout_sec, stale_lock_sec=stale_lock_sec)

    with FileLock(lock_path, lock_cfg):
        if enable_cache and (not force_regen):
            vis = find_vis_mp4(cache_dir, video_path)
            if vis is not None and vis.exists() and vis.stat().st_size > 0:
                append_log(log_path, f"[CACHE HIT AFTER LOCK] depth control: {vis}")
                console_print(f"[CACHE HIT] depth control ready after lock for {video_path.name}")
                return vis

        console_print(f"[CACHE MISS] generating depth control for {video_path.name}")
        vis = generate_control_depth(
            video_path=video_path,
            output_dir=cache_dir,
            depth_root=depth_root,
            conda_env=conda_env,
            encoder=encoder,
            grayscale=grayscale,
            env=env,
            log_path=log_path,
        )
        append_log(log_path, f"[CACHE GENERATED] depth control: {vis}")
        console_print(f"[CACHE GENERATED] {vis.name}")
        return vis


def generate_control(
    control_type: ControlType,
    video_path: Path,
    work_dir: Path,
    depth_root: Path,
    depth_conda_env: str,
    encoder: str,
    grayscale: bool,
    enable_cache: bool,
    force_regen: bool,
    fingerprint_mode: str,
    lock_timeout_sec: int,
    stale_lock_sec: int,
    non_cached_output_dir: Path,
    env: Dict[str, str],
    log_path: Path,
) -> Path:
    """
    Unified control interface:
    - depth: implemented + cached
    - edge/blur/seg/combo: interfaces reserved (to be implemented later)
    """
    if control_type == ControlType.DEPTH:
        return ensure_cached_depth_control(
            video_path=video_path,
            work_dir=work_dir,
            depth_root=depth_root,
            conda_env=depth_conda_env,
            encoder=encoder,
            grayscale=grayscale,
            env=env,
            log_path=log_path,
            enable_cache=enable_cache,
            force_regen=force_regen,
            fingerprint_mode=fingerprint_mode,
            lock_timeout_sec=lock_timeout_sec,
            stale_lock_sec=stale_lock_sec,
        )

    # Reserved for future implementation
    raise NotImplementedError(
        f"Control type '{control_type.value}' not implemented yet. "
        f"Implement it in auto_transfer_runner/steps/control.py."
    )
