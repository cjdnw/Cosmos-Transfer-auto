from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ..utils.cleanup import safe_unlink
from ..utils.logging_utils import append_log, console_print
from ..utils.subprocess_utils import conda_run, run_cmd


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


def _build_atempo_filter(tempo: float) -> str:
    if tempo <= 0:
        raise ValueError("Audio tempo must be > 0.")
    parts = []
    while tempo < 0.5:
        parts.append("atempo=0.5")
        tempo /= 0.5
    while tempo > 2.0:
        parts.append("atempo=2.0")
        tempo /= 2.0
    parts.append(f"atempo={tempo:.6f}")
    return ",".join(parts)


def has_audio_stream(video_path: Path, *, env: Dict[str, str], log_path: Path) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "json",
        str(video_path),
    ]
    append_log(log_path, f"[POST_RESTORE] ffprobe audio: {' '.join(cmd)}")
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
        raise RuntimeError(f"ffprobe audio check failed for {video_path} (code={result.returncode}).")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    has_audio = bool(streams)
    append_log(log_path, f"[POST_RESTORE] audio_stream={has_audio}")
    return has_audio


def _check_pytorch_cuda_available(*, conda_env: str, env: Dict[str, str], log_path: Path) -> None:
    base_cmd = ["python3", "-c", "import torch; print(torch.cuda.is_available())"]
    cmd = conda_run(conda_env, base_cmd) if conda_env else base_cmd
    append_log(log_path, f"[POST_RESTORE] PyTorch CUDA check: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env or dict(os.environ),
        check=False,
    )
    append_log(log_path, result.stdout.strip())
    append_log(log_path, result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError("PyTorch CUDA check failed. Ensure torch is installed and importable.")
    if "True" not in result.stdout:
        raise RuntimeError("CUDA is not available for PyTorch RIFE backend.")


def run_rife_pytorch(
    *,
    input_mp4: Path,
    output_mp4: Path,
    target_fps: float,
    scale: float,
    repo_dir: Path,
    model_dir: Path,
    keep_audio: bool,
    conda_env: str,
    env: Dict[str, str],
    log_path: Path,
) -> None:
    repo_dir = repo_dir.resolve()
    script_path = repo_dir / "inference_video.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"inference_video.py not found under {repo_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"RIFE model dir not found: {model_dir}")

    _check_pytorch_cuda_available(conda_env=conda_env, env=env, log_path=log_path)

    start_ts = dt.datetime.now().isoformat(timespec="seconds")
    start_perf = time.perf_counter()
    append_log(
        log_path,
        (
            "[POST_RESTORE] RIFE pytorch start "
            f"time={start_ts} input={input_mp4} output={output_mp4} fps={target_fps} scale={scale}"
        ),
    )

    tmp_output = output_mp4.with_name(f".{output_mp4.stem}__rife_pytorch{output_mp4.suffix}")
    safe_unlink(tmp_output, log_path=log_path)

    base_cmd = [
        "python3",
        str(script_path),
        "--video",
        str(input_mp4),
        "--fps",
        str(int(round(target_fps))),
        "--output",
        str(tmp_output),
        "--model",
        str(model_dir),
        "--scale",
        str(scale),
    ]
    cmd = conda_run(conda_env, base_cmd) if conda_env else base_cmd
    run_cmd(
        cmd,
        cwd=repo_dir,
        env=env or dict(os.environ),
        log_path=log_path,
        title="STEP5.5 Post-restore RIFE (pytorch) inference_video",
    )

    final_output = tmp_output
    if keep_audio:
        if has_audio_stream(input_mp4, env=env, log_path=log_path):
            mux_output = output_mp4.with_name(f".{output_mp4.stem}__rife_mux{output_mp4.suffix}")
            safe_unlink(mux_output, log_path=log_path)
            mux_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(tmp_output),
                "-i",
                str(input_mp4),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-shortest",
                str(mux_output),
            ]
            run_cmd(
                mux_cmd,
                cwd=None,
                env=env or dict(os.environ),
                log_path=log_path,
                title="STEP5.5 Post-restore RIFE (pytorch) mux audio",
            )
            safe_unlink(tmp_output, log_path=log_path)
            final_output = mux_output
        else:
            append_log(log_path, "[POST_RESTORE] keep_audio=true but no audio stream found; skip mux.")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(final_output), str(output_mp4))

    elapsed = time.perf_counter() - start_perf
    end_ts = dt.datetime.now().isoformat(timespec="seconds")
    append_log(
        log_path,
        (
            "[POST_RESTORE] RIFE pytorch done "
            f"time={end_ts} elapsed_sec={elapsed:.2f} output={output_mp4} fps={target_fps}"
        ),
    )


def restore_duration(
    *,
    output_dir: Path,
    original_stats: VideoStats,
    pre_fps: int | None,
    mode: str,
    fps_mode: str,
    rife_bin: str | None,
    rife_model: str | None,
    rife_backend: str,
    rife_pytorch_repo: str | None,
    rife_pytorch_model_dir: str | None,
    rife_pytorch_env: str,
    rife_scale: float,
    rife_keep_audio: bool,
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

    target_fps: float | None = None
    target_frames: int | None = None
    if original_stats.frame_count and original_stats.frame_count > 0:
        if original_stats.fps and original_stats.fps > 0:
            target_frames = original_stats.frame_count
            target_fps = original_stats.fps
            append_log(
                log_path,
                f"[POST_RESTORE] match original frames={target_frames} fps={target_fps:.6f}",
            )
        elif original_stats.duration_sec and original_stats.duration_sec > 0:
            target_frames = original_stats.frame_count
            target_fps = target_frames / original_stats.duration_sec
            append_log(
                log_path,
                f"[POST_RESTORE] match original frames={target_frames} fps={target_fps:.6f} (derived)",
            )

    if target_fps is None:
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

    if target_fps <= 0:
        raise RuntimeError("Target fps invalid for post-restore.")

    if target_frames is None:
        target_frames = round(original_stats.duration_sec * target_fps)

    target_duration = target_frames / target_fps
    ratio = target_duration / current_stats.duration_sec

    if mode == "speed":
        filters = f"setpts=PTS*{ratio:.6f},fps={target_fps}"
    elif mode == "minterpolate":
        filters = f"setpts=PTS*{ratio:.6f},minterpolate=fps={target_fps}:mi_mode=mci"
    elif mode == "rife":
        if rife_backend == "ncnn_vulkan":
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
        if rife_backend == "pytorch":
            if not rife_pytorch_repo:
                raise ValueError("rife_pytorch_repo must be set when rife_backend=pytorch")
            repo_dir = Path(rife_pytorch_repo)
            model_dir = Path(rife_pytorch_model_dir) if rife_pytorch_model_dir else repo_dir / "train_log"
            speed_path = transfer_mp4.with_name(f".{transfer_mp4.stem}__post_restore_speed{transfer_mp4.suffix}")
            try:
                safe_unlink(speed_path, log_path=log_path)
                if rife_keep_audio and has_audio_stream(transfer_mp4, env=env, log_path=log_path):
                    tempo = 1.0 / ratio
                    atempo = _build_atempo_filter(tempo)
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(transfer_mp4),
                        "-filter_complex",
                        f"[0:v]setpts=PTS*{ratio:.6f}[v];[0:a]{atempo}[a]",
                        "-map",
                        "[v]",
                        "-map",
                        "[a]",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(speed_path),
                    ]
                else:
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
                    title="STEP5.5 Post-restore speed adjust (RIFE pytorch prep)",
                )
                run_rife_pytorch(
                    input_mp4=speed_path,
                    output_mp4=transfer_mp4,
                    target_fps=target_fps,
                    scale=rife_scale,
                    repo_dir=repo_dir,
                    model_dir=model_dir,
                    keep_audio=rife_keep_audio,
                    conda_env=rife_pytorch_env,
                    env=env,
                    log_path=log_path,
                )
            finally:
                safe_unlink(speed_path, log_path=log_path)
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
        raise ValueError(f"Unknown RIFE backend: {rife_backend}")
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
