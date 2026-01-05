from __future__ import annotations

import os
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
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_name(f".{output_path.name}.lock")
    lock_cfg = FileLockConfig(timeout_sec=lock_timeout_sec, stale_lock_sec=stale_lock_sec)

    with FileLock(lock_path, lock_cfg):
        if output_path.exists() and output_path.stat().st_size > 0:
            append_log(log_path, f"[PREPROCESS CACHE HIT AFTER LOCK] {output_path}")
            console_print(f"[PREPROCESS CACHE HIT] {output_path.name}")
            return output_path

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
        return output_path


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
