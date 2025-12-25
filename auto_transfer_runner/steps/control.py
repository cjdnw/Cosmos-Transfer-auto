from __future__ import annotations

import shutil
from contextlib import nullcontext
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


def find_seg_mp4(control_dir: Path, video_path: Path) -> Optional[Path]:
    """Find a seg mp4 in output directory (prefer names containing video stem)."""
    if not control_dir.exists():
        return None

    candidates = list(control_dir.glob("*.mp4"))
    if not candidates:
        return None

    stem = video_path.stem
    name = video_path.name
    preferred = [p for p in candidates if stem in p.name or name in p.name]
    target_list = preferred if preferred else candidates
    target_list.sort(key=lambda p: (p.stat().st_size, p.stat().st_mtime), reverse=True)
    return target_list[0]


def find_control_mp4(control_dir: Path, video_path: Path, suffix_hint: str | None = None) -> Optional[Path]:
    """Find a control mp4 in output directory (prefer suffix + video stem)."""
    if not control_dir.exists():
        return None

    candidates = list(control_dir.glob("*.mp4"))
    if not candidates:
        return None

    stem = video_path.stem
    name = video_path.name
    preferred = candidates
    if suffix_hint:
        preferred = [p for p in candidates if suffix_hint in p.name]
    preferred = [p for p in preferred if stem in p.name or name in p.name] or preferred
    preferred.sort(key=lambda p: (p.stat().st_size, p.stat().st_mtime), reverse=True)
    return preferred[0]


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


def _pick_seg_mp4(tmp_dir: Path, video_path: Path) -> Path:
    mp4s = list(tmp_dir.rglob("*.mp4"))
    if not mp4s:
        raise FileNotFoundError(f"Seg mp4 not found under: {tmp_dir}")

    stem = video_path.stem
    name = video_path.name
    preferred = [p for p in mp4s if stem in p.name or name in p.name]
    target_list = preferred if preferred else mp4s
    target_list.sort(key=lambda p: (p.stat().st_size, p.stat().st_mtime), reverse=True)
    return target_list[0]


def generate_control_seg(
    video_path: Path,
    dest_dir: Path,
    seg_root: Path,
    conda_env: str,
    level: str,
    use_other_level: bool,
    detect_stride: int,
    save_intermediate: bool,
    save_npy: bool,
    enable_highlight: bool,
    enable_overlay: bool,
    overlay_style: str,
    env: Dict[str, str],
    log_path: Path,
) -> Path:
    """
    Step3(seg): run AutoSeg-SAM2 and return the produced control mp4.
    Generation happens in a temp dir under dest_dir to keep cache clean.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = dest_dir / "_seg_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = tmp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # AutoSeg-SAM2 expects a directory of extracted frames; pre-extract with ffmpeg.
    extract_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-q:v",
        "2",
        "-start_number",
        "1",
        str(frames_dir / "%06d.jpg"),
    ]
    run_cmd(extract_cmd, cwd=seg_root, env=env, log_path=log_path, title="STEP3: Extract frames for seg")

    mask_cmd = conda_run(
        conda_env,
        [
            "python",
            "auto-mask-batch.py",
            "--video_path",
            str(frames_dir),
            "--output_dir",
            str(tmp_dir),
            "--level",
            str(level),
            "--use_other_level",
            str(int(use_other_level)),
            "--detect_stride",
            str(detect_stride),
            "--save_intermediate",
            str(int(save_intermediate)),
            "--save_npy",
            str(int(save_npy)),
        ],
    )

    run_cmd(mask_cmd, cwd=seg_root, env=env, log_path=log_path, title="STEP3: Generate control (seg mask)")

    vis_cmd = conda_run(
        conda_env,
        [
            "python",
            "visulization.py",
            "--video_path",
            str(frames_dir),
            "--output_dir",
            str(tmp_dir),
            "--level",
            str(level),
            "--enable_highlight",
            str(int(enable_highlight)),
            "--enable_overlay",
            str(int(enable_overlay)),
            "--overlay_style",
            str(overlay_style),
        ],
    )

    run_cmd(vis_cmd, cwd=seg_root, env=env, log_path=log_path, title="STEP3: Generate control (seg vis)")

    seg_mp4 = _pick_seg_mp4(tmp_dir, video_path)
    final_path = dest_dir / seg_mp4.name
    if final_path.exists():
        final_path.unlink()
    shutil.move(str(seg_mp4), final_path)

    # Clean temp dir after moving the final mp4
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return final_path


def _edge_output_path(output_dir: Path, video_path: Path) -> Path:
    return output_dir / f"{video_path.stem}_canny.mp4"


def _blur_output_path(output_dir: Path, video_path: Path) -> Path:
    return output_dir / f"{video_path.stem}_blur.mp4"


def generate_control_edge(
    video_path: Path,
    output_dir: Path,
    cv_root: Path,
    conda_env: str,
    low: int,
    high: int,
    env: Dict[str, str],
    log_path: Path,
) -> Path:
    """Step3(edge): run gen_controls_cv_mp4.py (canny) and return the produced mp4."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = _edge_output_path(output_dir, video_path)

    cmd = conda_run(
        conda_env,
        [
            "python",
            "gen_controls_cv_mp4.py",
            "--input",
            str(video_path),
            "--mode",
            "canny",
            "--output-dir",
            str(output_dir),
            "--low",
            str(low),
            "--high",
            str(high),
        ],
    )

    run_cmd(cmd, cwd=cv_root, env=env, log_path=log_path, title="STEP3: Generate control (edge)")

    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    fallback = find_control_mp4(output_dir, video_path, suffix_hint="canny")
    if fallback is None or fallback.stat().st_size == 0:
        raise FileNotFoundError(f"Edge control mp4 not found in: {output_dir}")
    return fallback


def generate_control_blur(
    video_path: Path,
    output_dir: Path,
    cv_root: Path,
    conda_env: str,
    blur_sigma: float,
    low: int,
    high: int,
    env: Dict[str, str],
    log_path: Path,
) -> Path:
    """Step3(blur): run gen_controls_cv_mp4.py (blur) and return the produced mp4."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = _blur_output_path(output_dir, video_path)

    cmd = conda_run(
        conda_env,
        [
            "python",
            "gen_controls_cv_mp4.py",
            "--input",
            str(video_path),
            "--mode",
            "blur",
            "--output-dir",
            str(output_dir),
            "--blur-sigma",
            str(blur_sigma),
            "--low",
            str(low),
            "--high",
            str(high),
        ],
    )

    run_cmd(cmd, cwd=cv_root, env=env, log_path=log_path, title="STEP3: Generate control (blur)")

    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    fallback = find_control_mp4(output_dir, video_path, suffix_hint="blur")
    if fallback is None or fallback.stat().st_size == 0:
        raise FileNotFoundError(f"Blur control mp4 not found in: {output_dir}")
    return fallback


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


def ensure_cached_seg_control(
    video_path: Path,
    work_dir: Path,
    seg_root: Path,
    conda_env: str,
    level: str,
    use_other_level: bool,
    detect_stride: int,
    save_intermediate: bool,
    save_npy: bool,
    enable_highlight: bool,
    enable_overlay: bool,
    overlay_style: str,
    env: Dict[str, str],
    log_path: Path,
    enable_cache: bool,
    force_regen: bool,
    fingerprint_mode: str,
    lock_timeout_sec: int,
    stale_lock_sec: int,
    non_cached_output_dir: Path,
) -> Path:
    """
    seg cache strategy (mirrors depth):
    - cache_dir = <work_dir>/control_cache/seg/<fingerprint>/
    - If a valid *.mp4 already exists: reuse it (skip generation)
    - Otherwise: lock and generate once (safe for multi-GPU concurrency)
    - When cache is disabled, generate into per-job non_cached_output_dir
    """
    if enable_cache:
        fp = video_fingerprint(video_path, mode=fingerprint_mode)
        cache_dir = work_dir / "control_cache" / "seg" / fp
    else:
        cache_dir = non_cached_output_dir

    cache_dir.mkdir(parents=True, exist_ok=True)

    if enable_cache and (not force_regen):
        seg_mp4 = find_seg_mp4(cache_dir, video_path)
        if seg_mp4 is not None and seg_mp4.exists() and seg_mp4.stat().st_size > 0:
            append_log(log_path, f"[CACHE HIT] seg control: {seg_mp4}")
            console_print(f"[CACHE HIT] seg control for {video_path.name}")
            return seg_mp4

    lock_path = cache_dir / ".lock"
    lock_cfg = FileLockConfig(timeout_sec=lock_timeout_sec, stale_lock_sec=stale_lock_sec)

    # If caching is disabled, the output dir is per-job unique; we skip the lock.
    lock_context = FileLock(lock_path, lock_cfg) if enable_cache else nullcontext()

    with lock_context:
        if enable_cache and (not force_regen):
            seg_mp4 = find_seg_mp4(cache_dir, video_path)
            if seg_mp4 is not None and seg_mp4.exists() and seg_mp4.stat().st_size > 0:
                append_log(log_path, f"[CACHE HIT AFTER LOCK] seg control: {seg_mp4}")
                console_print(f"[CACHE HIT] seg control ready after lock for {video_path.name}")
                return seg_mp4

        console_print(f"[CACHE MISS] generating seg control for {video_path.name}")
        seg_mp4 = generate_control_seg(
            video_path=video_path,
            dest_dir=cache_dir,
            seg_root=seg_root,
            conda_env=conda_env,
            level=level,
            use_other_level=use_other_level,
            detect_stride=detect_stride,
            save_intermediate=save_intermediate,
            save_npy=save_npy,
            enable_highlight=enable_highlight,
            enable_overlay=enable_overlay,
            overlay_style=overlay_style,
            env=env,
            log_path=log_path,
        )
        append_log(log_path, f"[CACHE GENERATED] seg control: {seg_mp4}")
        console_print(f"[CACHE GENERATED] {seg_mp4.name}")
        return seg_mp4


def ensure_cached_edge_control(
    video_path: Path,
    work_dir: Path,
    cv_root: Path,
    conda_env: str,
    low: int,
    high: int,
    env: Dict[str, str],
    log_path: Path,
    enable_cache: bool,
    force_regen: bool,
    fingerprint_mode: str,
    lock_timeout_sec: int,
    stale_lock_sec: int,
    non_cached_output_dir: Path,
) -> Path:
    """
    edge cache strategy (mirrors seg):
    - cache_dir = <work_dir>/control_cache/edge/<fingerprint>/
    - If a valid *.mp4 already exists: reuse it (skip generation)
    - Otherwise: lock and generate once (safe for multi-GPU concurrency)
    - When cache is disabled, generate into per-job non_cached_output_dir
    """
    if enable_cache:
        fp = video_fingerprint(video_path, mode=fingerprint_mode)
        cache_dir = work_dir / "control_cache" / "edge" / fp
    else:
        cache_dir = non_cached_output_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    expected = _edge_output_path(cache_dir, video_path)

    if enable_cache and (not force_regen):
        if expected.exists() and expected.stat().st_size > 0:
            append_log(log_path, f"[CACHE HIT] edge control: {expected}")
            console_print(f"[CACHE HIT] edge control for {video_path.name}")
            return expected

        fallback = find_control_mp4(cache_dir, video_path, suffix_hint="canny")
        if fallback is not None and fallback.stat().st_size > 0:
            append_log(log_path, f"[CACHE HIT] edge control: {fallback}")
            console_print(f"[CACHE HIT] edge control for {video_path.name}")
            return fallback

    lock_path = cache_dir / ".lock"
    lock_cfg = FileLockConfig(timeout_sec=lock_timeout_sec, stale_lock_sec=stale_lock_sec)
    lock_context = FileLock(lock_path, lock_cfg) if enable_cache else nullcontext()

    with lock_context:
        if enable_cache and (not force_regen):
            if expected.exists() and expected.stat().st_size > 0:
                append_log(log_path, f"[CACHE HIT AFTER LOCK] edge control: {expected}")
                console_print(f"[CACHE HIT] edge control ready after lock for {video_path.name}")
                return expected

            fallback = find_control_mp4(cache_dir, video_path, suffix_hint="canny")
            if fallback is not None and fallback.stat().st_size > 0:
                append_log(log_path, f"[CACHE HIT AFTER LOCK] edge control: {fallback}")
                console_print(f"[CACHE HIT] edge control ready after lock for {video_path.name}")
                return fallback

        console_print(f"[CACHE MISS] generating edge control for {video_path.name}")
        edge_mp4 = generate_control_edge(
            video_path=video_path,
            output_dir=cache_dir,
            cv_root=cv_root,
            conda_env=conda_env,
            low=low,
            high=high,
            env=env,
            log_path=log_path,
        )
        append_log(log_path, f"[CACHE GENERATED] edge control: {edge_mp4}")
        console_print(f"[CACHE GENERATED] {edge_mp4.name}")
        return edge_mp4


def ensure_cached_blur_control(
    video_path: Path,
    work_dir: Path,
    cv_root: Path,
    conda_env: str,
    blur_sigma: float,
    low: int,
    high: int,
    env: Dict[str, str],
    log_path: Path,
    enable_cache: bool,
    force_regen: bool,
    fingerprint_mode: str,
    lock_timeout_sec: int,
    stale_lock_sec: int,
    non_cached_output_dir: Path,
) -> Path:
    """
    blur cache strategy (mirrors seg):
    - cache_dir = <work_dir>/control_cache/blur/<fingerprint>/
    - If a valid *.mp4 already exists: reuse it (skip generation)
    - Otherwise: lock and generate once (safe for multi-GPU concurrency)
    - When cache is disabled, generate into per-job non_cached_output_dir
    """
    if enable_cache:
        fp = video_fingerprint(video_path, mode=fingerprint_mode)
        cache_dir = work_dir / "control_cache" / "blur" / fp
    else:
        cache_dir = non_cached_output_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    expected = _blur_output_path(cache_dir, video_path)

    if enable_cache and (not force_regen):
        if expected.exists() and expected.stat().st_size > 0:
            append_log(log_path, f"[CACHE HIT] blur control: {expected}")
            console_print(f"[CACHE HIT] blur control for {video_path.name}")
            return expected

        fallback = find_control_mp4(cache_dir, video_path, suffix_hint="blur")
        if fallback is not None and fallback.stat().st_size > 0:
            append_log(log_path, f"[CACHE HIT] blur control: {fallback}")
            console_print(f"[CACHE HIT] blur control for {video_path.name}")
            return fallback

    lock_path = cache_dir / ".lock"
    lock_cfg = FileLockConfig(timeout_sec=lock_timeout_sec, stale_lock_sec=stale_lock_sec)
    lock_context = FileLock(lock_path, lock_cfg) if enable_cache else nullcontext()

    with lock_context:
        if enable_cache and (not force_regen):
            if expected.exists() and expected.stat().st_size > 0:
                append_log(log_path, f"[CACHE HIT AFTER LOCK] blur control: {expected}")
                console_print(f"[CACHE HIT] blur control ready after lock for {video_path.name}")
                return expected

            fallback = find_control_mp4(cache_dir, video_path, suffix_hint="blur")
            if fallback is not None and fallback.stat().st_size > 0:
                append_log(log_path, f"[CACHE HIT AFTER LOCK] blur control: {fallback}")
                console_print(f"[CACHE HIT] blur control ready after lock for {video_path.name}")
                return fallback

        console_print(f"[CACHE MISS] generating blur control for {video_path.name}")
        blur_mp4 = generate_control_blur(
            video_path=video_path,
            output_dir=cache_dir,
            cv_root=cv_root,
            conda_env=conda_env,
            blur_sigma=blur_sigma,
            low=low,
            high=high,
            env=env,
            log_path=log_path,
        )
        append_log(log_path, f"[CACHE GENERATED] blur control: {blur_mp4}")
        console_print(f"[CACHE GENERATED] {blur_mp4.name}")
        return blur_mp4


def generate_control(
    control_type: ControlType,
    video_path: Path,
    work_dir: Path,
    depth_root: Path,
    seg_root: Path,
    cv_root: Path,
    depth_conda_env: str,
    seg_conda_env: str,
    cv_conda_env: str,
    encoder: str,
    grayscale: bool,
    seg_level: str,
    seg_use_other_level: bool,
    seg_detect_stride: int,
    seg_save_intermediate: bool,
    seg_save_npy: bool,
    seg_enable_highlight: bool,
    seg_enable_overlay: bool,
    seg_overlay_style: str,
    edge_low: int,
    edge_high: int,
    blur_sigma: float,
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
    - seg: implemented + cached (AutoSeg-SAM2)
    - edge/blur: implemented + cached (gen_controls_cv_mp4.py)
    - combo: reserved
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

    if control_type == ControlType.SEG:
        return ensure_cached_seg_control(
            video_path=video_path,
            work_dir=work_dir,
            seg_root=seg_root,
            conda_env=seg_conda_env,
            level=seg_level,
            use_other_level=seg_use_other_level,
            detect_stride=seg_detect_stride,
            save_intermediate=seg_save_intermediate,
            save_npy=seg_save_npy,
            enable_highlight=seg_enable_highlight,
            enable_overlay=seg_enable_overlay,
            overlay_style=seg_overlay_style,
            env=env,
            log_path=log_path,
            enable_cache=enable_cache,
            force_regen=force_regen,
            fingerprint_mode=fingerprint_mode,
            lock_timeout_sec=lock_timeout_sec,
            stale_lock_sec=stale_lock_sec,
            non_cached_output_dir=non_cached_output_dir,
        )

    if control_type == ControlType.EDGE:
        return ensure_cached_edge_control(
            video_path=video_path,
            work_dir=work_dir,
            cv_root=cv_root,
            conda_env=cv_conda_env,
            low=edge_low,
            high=edge_high,
            env=env,
            log_path=log_path,
            enable_cache=enable_cache,
            force_regen=force_regen,
            fingerprint_mode=fingerprint_mode,
            lock_timeout_sec=lock_timeout_sec,
            stale_lock_sec=stale_lock_sec,
            non_cached_output_dir=non_cached_output_dir,
        )

    if control_type == ControlType.BLUR:
        return ensure_cached_blur_control(
            video_path=video_path,
            work_dir=work_dir,
            cv_root=cv_root,
            conda_env=cv_conda_env,
            blur_sigma=blur_sigma,
            low=edge_low,
            high=edge_high,
            env=env,
            log_path=log_path,
            enable_cache=enable_cache,
            force_regen=force_regen,
            fingerprint_mode=fingerprint_mode,
            lock_timeout_sec=lock_timeout_sec,
            stale_lock_sec=stale_lock_sec,
            non_cached_output_dir=non_cached_output_dir,
        )

    # Reserved for future implementation
    raise NotImplementedError(
        f"Control type '{control_type.value}' not implemented yet. "
        f"Implement it in auto_transfer_runner/steps/control.py."
    )
