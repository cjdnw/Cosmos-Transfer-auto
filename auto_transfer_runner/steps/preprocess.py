from __future__ import annotations

import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from ..utils.cleanup import safe_unlink
from ..utils.locking import FileLock, FileLockConfig
from ..utils.logging_utils import append_log, console_print
from ..utils.subprocess_utils import run_cmd


@dataclass(frozen=True)
class PreprocessSettings:
    fps: int
    width: int | None
    height: int | None
    keep_aspect: bool
    pre_chunk_mode: str
    pre_chunk_policy: str
    pre_chunk_clip_where: str
    pre_chunk_max_adjust: int
    num_video_frames_per_chunk: int
    num_conditional_frames: int


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
    output_path = _build_output_path(
        video_path,
        settings,
        output_root=output_root,
        input_root=input_root,
    )
    if output_path.exists() and output_path.stat().st_size > 0:
        append_log(log_path, f"[PREPROCESS CACHE HIT] {output_path}")
        console_print(f"[PREPROCESS CACHE HIT] {output_path.name}")
        return _align_preprocess_chunks(
            pre_video_path=output_path,
            settings=settings,
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
        filters = [scale_filter, f"fps={settings.fps}"]
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
            str(settings.fps),
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
    w_label = str(width) if width is not None else "auto"
    h_label = str(height) if height is not None else "auto"
    aspect_label = "ka1" if settings.keep_aspect else "ka0"
    name = f"{video_path.stem}__pre_fps{settings.fps}_w{w_label}_h{h_label}_{aspect_label}{video_path.suffix}"
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
    env: Dict[str, str],
    log_path: Path,
    lock_timeout_sec: int,
    stale_lock_sec: int,
) -> Path:
    mode = settings.pre_chunk_mode
    if mode == "none":
        return pre_video_path

    total_frames = _probe_frame_count(pre_video_path, env=env, log_path=log_path)
    L = settings.num_video_frames_per_chunk
    C = settings.num_conditional_frames
    S = L - C
    if S <= 0:
        append_log(log_path, f"[PRE_CHUNK SKIP] invalid S=L-C ({L}-{C})")
        console_print("[PRE_CHUNK SKIP] invalid chunk config")
        return pre_video_path

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

    target_frames = L + (n - 1) * S
    drop = total_frames - target_frames
    if settings.pre_chunk_max_adjust is not None and abs(drop) > settings.pre_chunk_max_adjust:
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

    aligned_path = _build_aligned_output_path(
        pre_video_path,
        mode=mode,
        policy=settings.pre_chunk_policy,
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
                    str(settings.fps),
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
                    str(settings.fps),
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
            k_value = total_frames / target_frames
            filters = f"setpts=PTS/{k_value},fps={settings.fps}"
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
                str(settings.fps),
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

        log_parts = [
            f"[PRE_CHUNK DONE] mode={mode}",
            f"policy={settings.pre_chunk_policy}",
            f"T={total_frames}",
            f"T_target={target_frames}",
            f"drop={drop}",
            f"n={n}",
        ]
        if mode == "speed":
            log_parts.append(f"k={k_value}")
        log_parts.append(f"output={aligned_path}")
        append_log(log_path, " ".join(log_parts))
        console_print(" ".join(log_parts))
        return aligned_path


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
