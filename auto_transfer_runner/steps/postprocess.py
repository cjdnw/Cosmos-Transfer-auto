from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ..utils.cleanup import safe_unlink
from ..utils.logging_utils import append_log, console_print
from ..utils.subprocess_utils import run_cmd


@dataclass(frozen=True)
class VideoStats:
    duration_sec: float | None
    frame_count: int | None
    fps: float | None


def _parse_fraction(value: str) -> float:
    if "/" in value:
        num_str, den_str = value.split("/", 1)
        den = float(den_str)
        if den == 0:
            return 0.0
        return float(num_str) / den
    return float(value)


def get_pad_dims(width: int, height: int, align: int = 32) -> tuple[int, int]:
    pad_w = ((width + align - 1) // align) * align
    pad_h = ((height + align - 1) // align) * align
    return pad_w, pad_h


def probe_video_dimensions(video_path: Path, *, env: Dict[str, str], log_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(video_path),
    ]
    append_log(log_path, f"[POST_RESTORE] ffprobe dimensions: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env or dict(os.environ),
        check=False,
    )
    if result.returncode != 0:
        append_log(log_path, result.stdout.strip())
        append_log(log_path, result.stderr.strip())
        raise RuntimeError(f"ffprobe failed for {video_path} (code={result.returncode}).")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found for {video_path}.")
    stream = streams[0]
    width = stream.get("width")
    height = stream.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        raise RuntimeError(f"Failed to parse dimensions for {video_path}.")
    append_log(log_path, f"[POST_RESTORE] dimensions width={width} height={height}")
    return width, height


def find_transfer_output_mp4(output_dir: Path) -> Path:
    mp4s: List[Path] = [p for p in output_dir.glob("*.mp4") if p.is_file()]
    if not mp4s:
        raise FileNotFoundError(f"No mp4 outputs found in transfer output_dir: {output_dir}")
    if len(mp4s) > 1:
        return max(mp4s, key=lambda p: p.stat().st_size)
    return mp4s[0]


def has_transfer_output_mp4(output_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    for mp4 in output_dir.glob("*.mp4"):
        if mp4.is_file() and mp4.stat().st_size > 0:
            return True
    return False


def probe_video_stats(video_path: Path, *, env: Dict[str, str], log_path: Path) -> VideoStats:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,avg_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    append_log(log_path, f"[POST_RESTORE] ffprobe: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env or dict(os.environ),
        check=False,
    )
    if result.returncode != 0:
        append_log(log_path, result.stdout.strip())
        append_log(log_path, result.stderr.strip())
        raise RuntimeError(f"ffprobe failed for {video_path} (code={result.returncode}).")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    duration_str = fmt.get("duration")
    duration_sec = float(duration_str) if duration_str else None
    frame_count = None
    fps = None
    if streams:
        stream = streams[0]
        frames_str = stream.get("nb_read_frames")
        if frames_str and frames_str.isdigit():
            frame_count = int(frames_str)
        avg_frame_rate = stream.get("avg_frame_rate", "0/0")
        fps = _parse_fraction(avg_frame_rate)
    if duration_sec is None and frame_count and fps:
        duration_sec = frame_count / fps if fps > 0 else None
    append_log(
        log_path,
        f"[POST_RESTORE] stats duration={duration_sec} frames={frame_count} fps={fps}",
    )
    return VideoStats(duration_sec=duration_sec, frame_count=frame_count, fps=fps)


def restore_duration(
    *,
    output_dir: Path,
    original_stats: VideoStats,
    pre_fps: int | None,
    mode: str,
    fps_mode: str,
    rife_bin: str | None,
    rife_model: str | None,
    env: Dict[str, str],
    log_path: Path,
) -> Path:
    if original_stats.duration_sec is None or original_stats.duration_sec <= 0:
        raise ValueError("Original duration is required for post-restore.")

    transfer_mp4 = find_transfer_output_mp4(output_dir)
    tmp_path = transfer_mp4.with_name(f".{transfer_mp4.stem}__post_restore_tmp{transfer_mp4.suffix}")

    current_stats = probe_video_stats(transfer_mp4, env=env, log_path=log_path)
    if current_stats.duration_sec is None or current_stats.duration_sec <= 0:
        raise RuntimeError("Failed to probe current duration for post-restore.")

    ratio = original_stats.duration_sec / current_stats.duration_sec
    target_fps: float
    if fps_mode == "original":
        target_fps = original_stats.fps or 0.0
        if target_fps <= 0:
            if pre_fps is not None:
                append_log(log_path, "[POST_RESTORE] original fps unavailable, fallback to pre_fps")
                target_fps = float(pre_fps)
            elif current_stats.fps and current_stats.fps > 0:
                append_log(log_path, "[POST_RESTORE] original fps unavailable, fallback to current fps")
                target_fps = current_stats.fps
            else:
                raise RuntimeError("Unable to resolve target fps for post-restore.")
    else:
        if pre_fps is None:
            append_log(log_path, "[POST_RESTORE] pre_fps=none, fallback to original fps")
            target_fps = original_stats.fps or 0.0
            if target_fps <= 0:
                raise RuntimeError("Original fps unavailable for post-restore.")
        else:
            target_fps = float(pre_fps)
    target_frames = round(original_stats.duration_sec * target_fps)
    if (
        fps_mode == "original"
        and original_stats.frame_count
        and original_stats.frame_count > 0
        and original_stats.duration_sec > 0
    ):
        target_frames = original_stats.frame_count
        target_fps = target_frames / original_stats.duration_sec
        append_log(
            log_path,
            f"[POST_RESTORE] force frame_count={target_frames} => fps={target_fps:.6f}",
        )

    if mode == "speed":
        filters = f"setpts=PTS*{ratio:.6f},fps={target_fps}"
    elif mode == "minterpolate":
        filters = f"setpts=PTS*{ratio:.6f},minterpolate=fps={target_fps}:mi_mode=mci"
    elif mode == "rife":
        if not rife_bin:
            raise ValueError("rife_bin must be set when post_restore_mode=rife")
        rife_cache_root = Path("/mnt/data/ganrunhe/auto_transfer/rife_cache")
        rife_workdir = rife_cache_root / output_dir.name
        if rife_workdir.exists():
            shutil.rmtree(rife_workdir, ignore_errors=True)
        rife_workdir.mkdir(parents=True, exist_ok=True)
        speed_path = rife_workdir / f"{transfer_mp4.stem}__post_restore_speed{transfer_mp4.suffix}"
        input_frames_dir = rife_workdir / "input_frames"
        output_frames_dir = rife_workdir / "output_frames"
        final_output = rife_workdir / f"{transfer_mp4.stem}__post_restore_final{transfer_mp4.suffix}"
        try:
            safe_unlink(speed_path, log_path=log_path)
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(transfer_mp4),
                "-vf",
                f"setpts=PTS*{ratio:.6f}",
                "-vsync",
                "vfr",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(speed_path),
            ]
            run_cmd(
                cmd,
                cwd=None,
                env=env or dict(os.environ),
                log_path=log_path,
                title="STEP5.5 Post-restore speed adjust (RIFE prep)",
            )
            orig_w, orig_h = probe_video_dimensions(speed_path, env=env, log_path=log_path)
            pad_w, pad_h = get_pad_dims(orig_w, orig_h, align=32)
            target_frames = round(original_stats.duration_sec * target_fps)
            if input_frames_dir.exists():
                shutil.rmtree(input_frames_dir, ignore_errors=True)
            if output_frames_dir.exists():
                shutil.rmtree(output_frames_dir, ignore_errors=True)
            input_frames_dir.mkdir(parents=True, exist_ok=True)
            output_frames_dir.mkdir(parents=True, exist_ok=True)
            pad_filter = None
            if pad_w != orig_w or pad_h != orig_h:
                pad_filter = f"pad=ceil(iw/32)*32:ceil(ih/32)*32:(ow-iw)/2:(oh-ih)/2"
            extract_cmd = [
                "ffmpeg",
                "-y",
                "-vsync",
                "0",
                "-i",
                str(speed_path),
            ]
            if pad_filter:
                extract_cmd += ["-vf", pad_filter]
            extract_cmd.append(str(input_frames_dir / "%08d.png"))
            run_cmd(
                extract_cmd,
                cwd=None,
                env=env or dict(os.environ),
                log_path=log_path,
                title="STEP5.5 Post-restore RIFE extract frames",
            )
            rife_cmd = [
                rife_bin,
                "-i",
                str(input_frames_dir),
                "-o",
                str(output_frames_dir),
                "-n",
                str(target_frames),
                "-f",
                "%08d.png",
            ]
            if rife_model:
                rife_cmd += ["-m", rife_model]
            rife_env = dict(env) if env is not None else dict(os.environ)
            rife_env.pop("DISPLAY", None)
            rife_env.pop("WAYLAND_DISPLAY", None)
            run_cmd(
                rife_cmd,
                cwd=None,
                env=rife_env,
                log_path=log_path,
                title="STEP5.5 Post-restore RIFE interpolation",
            )
            encode_cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(target_fps),
                "-i",
                str(output_frames_dir / "%08d.png"),
            ]
            if pad_w != orig_w or pad_h != orig_h:
                crop_filter = f"crop={orig_w}:{orig_h}:(in_w-{orig_w})/2:(in_h-{orig_h})/2"
                encode_cmd += ["-vf", crop_filter]
            encode_cmd += [
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(final_output),
            ]
            run_cmd(
                encode_cmd,
                cwd=None,
                env=env or dict(os.environ),
                log_path=log_path,
                title="STEP5.5 Post-restore RIFE encode",
            )
            shutil.move(str(final_output), str(transfer_mp4))
        finally:
            shutil.rmtree(rife_workdir, ignore_errors=True)
        append_log(
            log_path,
            (
                "[POST_RESTORE DONE] "
                f"mode={mode} target_duration={original_stats.duration_sec:.3f} "
                f"current_duration={current_stats.duration_sec:.3f} "
                f"ratio={ratio:.6f} target_frames={target_frames} output={transfer_mp4}"
            ),
        )
        console_print(
            f"[POST_RESTORE DONE] mode={mode} target_frames={target_frames} output={transfer_mp4.name}",
        )
        return transfer_mp4
    else:
        raise ValueError(f"Unknown post_restore_mode: {mode}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(transfer_mp4),
        "-vf",
        filters,
        "-frames:v",
        str(target_frames),
        "-vsync",
        "cfr",
        "-r",
        str(target_fps),
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
        title="STEP5.5 Post-restore duration",
    )
    tmp_path.replace(transfer_mp4)
    append_log(
        log_path,
        (
            "[POST_RESTORE DONE] "
            f"mode={mode} target_duration={original_stats.duration_sec:.3f} "
            f"current_duration={current_stats.duration_sec:.3f} "
            f"ratio={ratio:.6f} target_frames={target_frames} output={transfer_mp4}"
        ),
    )
    console_print(
        f"[POST_RESTORE DONE] mode={mode} target_frames={target_frames} output={transfer_mp4.name}",
    )
    return transfer_mp4
