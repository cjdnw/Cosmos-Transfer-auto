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


def _load_frame_map(frame_map_path: Path) -> List[int]:
    indices: List[int] = []
    with frame_map_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            indices.append(int(record["original_index"]))
    return indices


def _validate_frame_map(indices: List[int]) -> None:
    if not indices:
        raise ValueError("Frame map is empty.")
    last = -1
    for idx in indices:
        if idx < 0:
            raise ValueError("Frame map contains negative indices.")
        if idx <= last:
            raise ValueError("Frame map is not strictly increasing.")
        last = idx


def _build_gap_plan(indices: List[int]) -> List[int]:
    gaps: List[int] = []
    for i in range(len(indices) - 1):
        gap = indices[i + 1] - indices[i]
        if gap <= 0:
            raise ValueError("Frame map gap is non-positive.")
        gaps.append(gap)
    return gaps


def _extract_frames(
    *,
    input_video: Path,
    output_dir: Path,
    pad_filter: str | None,
    env: Dict[str, str],
    log_path: Path,
    title: str,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-vsync",
        "0",
        "-i",
        str(input_video),
    ]
    if pad_filter:
        cmd += ["-vf", pad_filter]
    cmd.append(str(output_dir / "%08d.png"))
    run_cmd(cmd, cwd=None, env=env or dict(os.environ), log_path=log_path, title=title)


def _encode_frames(
    *,
    input_dir: Path,
    output_path: Path,
    target_fps: float,
    crop_filter: str | None,
    env: Dict[str, str],
    log_path: Path,
    title: str,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(target_fps),
        "-i",
        str(input_dir / "%08d.png"),
    ]
    if crop_filter:
        cmd += ["-vf", crop_filter]
    cmd += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    run_cmd(cmd, cwd=None, env=env or dict(os.environ), log_path=log_path, title=title)


def _mux_audio(
    *,
    video_path: Path,
    audio_source: Path,
    env: Dict[str, str],
    log_path: Path,
    title: str,
) -> None:
    tmp_output = video_path.with_name(f".{video_path.stem}__audio{video_path.suffix}")
    safe_unlink(tmp_output, log_path=log_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_source),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        str(tmp_output),
    ]
    run_cmd(cmd, cwd=None, env=env or dict(os.environ), log_path=log_path, title=title)
    tmp_output.replace(video_path)


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


def _copy_frame(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _interpolate_segment_ncnn(
    *,
    rife_bin: str,
    rife_model: str | None,
    segment_input_dir: Path,
    segment_output_dir: Path,
    output_frames: int,
    env: Dict[str, str],
    log_path: Path,
) -> None:
    if segment_output_dir.exists():
        shutil.rmtree(segment_output_dir, ignore_errors=True)
    segment_output_dir.mkdir(parents=True, exist_ok=True)
    rife_cmd = [
        rife_bin,
        "-i",
        str(segment_input_dir),
        "-o",
        str(segment_output_dir),
        "-n",
        str(output_frames),
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
        title="STEP5.5 Post-restore RIFE segment interpolation",
    )


def _interpolate_segment_pytorch(
    *,
    segment_input_dir: Path,
    segment_video_path: Path,
    segment_output_video: Path,
    segment_output_frames_dir: Path,
    output_frames: int,
    scale: float,
    repo_dir: Path,
    model_dir: Path,
    conda_env: str,
    env: Dict[str, str],
    log_path: Path,
) -> None:
    if segment_video_path.exists():
        safe_unlink(segment_video_path, log_path=log_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "2",
        "-i",
        str(segment_input_dir / "%08d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(segment_video_path),
    ]
    run_cmd(
        cmd,
        cwd=None,
        env=env or dict(os.environ),
        log_path=log_path,
        title="STEP5.5 Post-restore RIFE (pytorch) segment video",
    )
    run_rife_pytorch(
        input_mp4=segment_video_path,
        output_mp4=segment_output_video,
        target_fps=float(output_frames),
        scale=scale,
        repo_dir=repo_dir,
        model_dir=model_dir,
        keep_audio=False,
        conda_env=conda_env,
        env=env,
        log_path=log_path,
    )
    _extract_frames(
        input_video=segment_output_video,
        output_dir=segment_output_frames_dir,
        pad_filter=None,
        env=env,
        log_path=log_path,
        title="STEP5.5 Post-restore RIFE (pytorch) segment extract frames",
    )


def _interpolate_with_frame_map(
    *,
    transfer_mp4: Path,
    frame_map_path: Path,
    target_fps: float,
    target_frames: int,
    ratio: float,
    rife_backend: str,
    rife_bin: str | None,
    rife_model: str | None,
    rife_pytorch_repo: str | None,
    rife_pytorch_model_dir: str | None,
    rife_pytorch_env: str,
    rife_scale: float,
    rife_keep_audio: bool,
    env: Dict[str, str],
    log_path: Path,
) -> None:
    indices = _load_frame_map(frame_map_path)
    _validate_frame_map(indices)
    gaps = _build_gap_plan(indices)

    rife_workdir = transfer_mp4.parent / f".{transfer_mp4.stem}__post_restore_rife"
    if rife_workdir.exists():
        shutil.rmtree(rife_workdir, ignore_errors=True)
    rife_workdir.mkdir(parents=True, exist_ok=True)

    speed_path = rife_workdir / f"{transfer_mp4.stem}__post_restore_speed{transfer_mp4.suffix}"
    audio_source = rife_workdir / f"{transfer_mp4.stem}__post_restore_audio{transfer_mp4.suffix}"
    input_frames_dir = rife_workdir / "input_frames"
    output_frames_dir = rife_workdir / "output_frames"
    segment_input_dir = rife_workdir / "segment_input"
    segment_output_dir = rife_workdir / "segment_output"
    segment_video = rife_workdir / "segment.mp4"
    segment_output_video = rife_workdir / "segment_rife.mp4"

    try:
        shutil.copyfile(transfer_mp4, audio_source)
        safe_unlink(speed_path, log_path=log_path)
        speed_cmd = [
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
            speed_cmd,
            cwd=None,
            env=env or dict(os.environ),
            log_path=log_path,
            title="STEP5.5 Post-restore speed adjust (frame map prep)",
        )

        orig_w, orig_h = probe_video_dimensions(speed_path, env=env, log_path=log_path)
        pad_w, pad_h = get_pad_dims(orig_w, orig_h, align=32)
        pad_filter = None
        crop_filter = None
        if pad_w != orig_w or pad_h != orig_h:
            pad_filter = f"pad=ceil(iw/32)*32:ceil(ih/32)*32:(ow-iw)/2:(oh-ih)/2"
            crop_filter = f"crop={orig_w}:{orig_h}:(in_w-{orig_w})/2:(in_h-{orig_h})/2"

        _extract_frames(
            input_video=speed_path,
            output_dir=input_frames_dir,
            pad_filter=pad_filter,
            env=env,
            log_path=log_path,
            title="STEP5.5 Post-restore RIFE extract frames (frame map)",
        )
        input_frames = sorted(input_frames_dir.glob("*.png"))
        if len(input_frames) != len(indices):
            raise RuntimeError(
                "Frame map size does not match extracted frames "
                f"({len(indices)} != {len(input_frames)})."
            )
        if indices[-1] + 1 != target_frames:
            raise RuntimeError(
                "Frame map last index does not match target frames "
                f"({indices[-1] + 1} != {target_frames})."
            )
        if output_frames_dir.exists():
            shutil.rmtree(output_frames_dir, ignore_errors=True)
        output_frames_dir.mkdir(parents=True, exist_ok=True)

        output_index = 1
        for i, gap in enumerate(gaps):
            if segment_input_dir.exists():
                shutil.rmtree(segment_input_dir, ignore_errors=True)
            segment_input_dir.mkdir(parents=True, exist_ok=True)
            _copy_frame(input_frames[i], segment_input_dir / "00000001.png")
            _copy_frame(input_frames[i + 1], segment_input_dir / "00000002.png")

            output_count = gap + 1
            if rife_backend == "ncnn_vulkan":
                if not rife_bin:
                    raise ValueError("rife_bin must be set when post_restore_mode=rife")
                _interpolate_segment_ncnn(
                    rife_bin=rife_bin,
                    rife_model=rife_model,
                    segment_input_dir=segment_input_dir,
                    segment_output_dir=segment_output_dir,
                    output_frames=output_count,
                    env=env,
                    log_path=log_path,
                )
            elif rife_backend == "pytorch":
                if not rife_pytorch_repo:
                    raise ValueError("rife_pytorch_repo must be set when rife_backend=pytorch")
                repo_dir = Path(rife_pytorch_repo)
                model_dir = (
                    Path(rife_pytorch_model_dir)
                    if rife_pytorch_model_dir
                    else repo_dir / "train_log"
                )
                _interpolate_segment_pytorch(
                    segment_input_dir=segment_input_dir,
                    segment_video_path=segment_video,
                    segment_output_video=segment_output_video,
                    segment_output_frames_dir=segment_output_dir,
                    output_frames=output_count,
                    scale=rife_scale,
                    repo_dir=repo_dir,
                    model_dir=model_dir,
                    conda_env=rife_pytorch_env,
                    env=env,
                    log_path=log_path,
                )
            else:
                raise ValueError(f"Unknown RIFE backend: {rife_backend}")

            segment_frames = sorted(segment_output_dir.glob("*.png"))
            if len(segment_frames) < output_count:
                raise RuntimeError(
                    "Segment interpolation produced fewer frames than expected "
                    f"({len(segment_frames)} < {output_count})."
                )
            for frame_path in segment_frames[:-1]:
                dest = output_frames_dir / f"{output_index:08d}.png"
                _copy_frame(frame_path, dest)
                output_index += 1

        last_frame = output_frames_dir / f"{output_index:08d}.png"
        _copy_frame(input_frames[-1], last_frame)
        output_index += 1

        produced_frames = output_index - 1
        if produced_frames != target_frames:
            raise RuntimeError(
                "Post-restore frame count mismatch "
                f"({produced_frames} != {target_frames})."
            )

        _encode_frames(
            input_dir=output_frames_dir,
            output_path=transfer_mp4,
            target_fps=target_fps,
            crop_filter=crop_filter,
            env=env,
            log_path=log_path,
            title="STEP5.5 Post-restore RIFE encode (frame map)",
        )

        if rife_keep_audio and has_audio_stream(audio_source, env=env, log_path=log_path):
            _mux_audio(
                video_path=transfer_mp4,
                audio_source=audio_source,
                env=env,
                log_path=log_path,
                title="STEP5.5 Post-restore RIFE mux audio",
            )
    finally:
        shutil.rmtree(rife_workdir, ignore_errors=True)


def restore_duration(
    *,
    output_dir: Path,
    original_stats: VideoStats,
    mode: str,
    frame_map_path: Path | None,
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
    if original_stats.fps and original_stats.fps > 0:
        target_fps = original_stats.fps
    else:
        raise RuntimeError("Original fps unavailable for post-restore.")

    if original_stats.frame_count and original_stats.frame_count > 0:
        target_frames = original_stats.frame_count
        append_log(
            log_path,
            f"[POST_RESTORE] match original frames={target_frames} fps={target_fps:.6f}",
        )
    elif original_stats.duration_sec and original_stats.duration_sec > 0:
        target_frames = round(original_stats.duration_sec * target_fps)
        append_log(
            log_path,
            f"[POST_RESTORE] match original duration fps={target_fps:.6f} frames={target_frames}",
        )

    target_duration = target_frames / target_fps
    ratio = target_duration / current_stats.duration_sec

    if mode == "speed":
        filters = f"setpts=PTS*{ratio:.6f},fps={target_fps}"
    elif mode == "minterpolate":
        filters = f"setpts=PTS*{ratio:.6f},minterpolate=fps={target_fps}:mi_mode=mci"
    elif mode == "rife":
        if frame_map_path is not None and frame_map_path.exists():
            _interpolate_with_frame_map(
                transfer_mp4=transfer_mp4,
                frame_map_path=frame_map_path,
                target_fps=target_fps,
                target_frames=target_frames,
                ratio=ratio,
                rife_backend=rife_backend,
                rife_bin=rife_bin,
                rife_model=rife_model,
                rife_pytorch_repo=rife_pytorch_repo,
                rife_pytorch_model_dir=rife_pytorch_model_dir,
                rife_pytorch_env=rife_pytorch_env,
                rife_scale=rife_scale,
                rife_keep_audio=rife_keep_audio,
                env=env,
                log_path=log_path,
            )
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
            if current_stats.fps and current_stats.fps > 0:
                speed_fps = current_stats.fps / ratio
            else:
                speed_fps = target_fps / ratio
            append_log(log_path, f"[POST_RESTORE] pytorch speed_fps={speed_fps:.6f}")
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
                        "-r",
                        f"{speed_fps:.6f}",
                        "-vsync",
                        "cfr",
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
                        "-r",
                        f"{speed_fps:.6f}",
                        "-vsync",
                        "cfr",
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
