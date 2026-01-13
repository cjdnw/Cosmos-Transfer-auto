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
from ..utils.hashing import short_hash
from ..utils.logging_utils import append_log, console_print
from ..utils.subprocess_utils import conda_run, run_cmd
from ..utils.warp_time import compute_time_steps, count_guided_output_frames, validate_time_steps
from .arm_swap import run_arm_swap_sam_masks
from ..utils.dataset_paths import parquet_path_for_video


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


def _mask_dir_has_files(mask_dir: Path) -> bool:
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.npy"]
    for pattern in patterns:
        if any(mask_dir.glob(pattern)):
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


def _probe_frame_count(video_path: Path, *, env: Dict[str, str], log_path: Path) -> int:
    stats = probe_video_stats(video_path, env=env, log_path=log_path)
    if stats.frame_count is None or stats.frame_count <= 0:
        raise RuntimeError(f"Failed to resolve frame count for {video_path}")
    return stats.frame_count


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


def _load_frame_map(
    *,
    frame_map_path: Path | None,
    expected_input_frames: int,
    expected_output_frames: int | None,
    log_path: Path,
) -> dict[str, list[int]] | None:
    if frame_map_path is None:
        return None
    if not frame_map_path.exists():
        append_log(log_path, f"[POST_RESTORE] frame_map not found: {frame_map_path}")
        return None
    data = json.loads(frame_map_path.read_text(encoding="utf-8"))
    sample_indices = data.get("sampled_indices")
    original_total_frames = data.get("original_total_frames")
    if not isinstance(sample_indices, list) or not sample_indices:
        append_log(log_path, f"[POST_RESTORE] invalid frame_map content: {frame_map_path}")
        return None
    if expected_input_frames > 0 and len(sample_indices) != expected_input_frames:
        append_log(
            log_path,
            (
                "[POST_RESTORE] frame_map size mismatch "
                f"expected={expected_input_frames} got={len(sample_indices)}"
            ),
        )
        return None
    if sample_indices[0] != 0:
        append_log(log_path, "[POST_RESTORE] frame_map must start at frame 0")
        return None
    if any(sample_indices[i] >= sample_indices[i + 1] for i in range(len(sample_indices) - 1)):
        append_log(log_path, "[POST_RESTORE] frame_map indices must be strictly increasing")
        return None
    if isinstance(original_total_frames, int) and expected_output_frames is not None:
        if original_total_frames != expected_output_frames:
            append_log(
                log_path,
                (
                    "[POST_RESTORE] frame_map output mismatch "
                    f"expected={expected_output_frames} map_total={original_total_frames}"
                ),
            )
            return None
    return {"sampled_indices": sample_indices}


def _select_rife_gpu_id(
    *,
    run_gpus: list[str] | None,
    job_index: int,
    env: Dict[str, str] | None,
) -> str | None:
    if env:
        explicit_gpu = env.get("AUTO_TRANSFER_RIFE_GPU_ID")
        if explicit_gpu:
            return explicit_gpu
    if run_gpus:
        return run_gpus[job_index % len(run_gpus)]
    return None


def _segment_frames_from_map(sample_indices: list[int]) -> list[int]:
    segment_frames = []
    for start, end in zip(sample_indices[:-1], sample_indices[1:]):
        frames = end - start + 1
        if frames < 2:
            raise RuntimeError("Invalid segment length for guided interpolation")
        segment_frames.append(frames)
    return segment_frames


def _pose_prefixes_for_subset(subset: str) -> list[str] | None:
    if subset == "full":
        return None
    prefixes: list[str] = []
    if subset in {"zarm_pos", "zarm_pos_vel", "zarm_pos_vel_hand"}:
        prefixes.append("position.zarm_")
    if subset in {"zarm_pos_vel", "zarm_pos_vel_hand"}:
        prefixes.append("velocity.zarm_")
    return prefixes or ["position.zarm_"]


def _run_rife_guided(
    *,
    input_frames_dir: Path,
    output_frames_dir: Path,
    sample_indices: list[int],
    rife_bin: str,
    rife_model: str | None,
    rife_gpu_id: str | None,
    env: Dict[str, str],
    log_path: Path,
) -> None:
    input_frames = sorted(input_frames_dir.glob("*.png"))
    if len(input_frames) != len(sample_indices):
        raise RuntimeError("frame_map does not match extracted frame count")
    if len(input_frames) < 2:
        raise RuntimeError("RIFE guided interpolation requires at least 2 frames")

    segment_root = output_frames_dir.parent / "guided_segments"
    segment_root.mkdir(parents=True, exist_ok=True)
    out_index = 1

    for idx in range(len(input_frames) - 1):
        start_idx = sample_indices[idx]
        end_idx = sample_indices[idx + 1]
        segment_frames = end_idx - start_idx + 1
        if segment_frames < 2:
            raise RuntimeError("Invalid segment length for guided interpolation")

        segment_in = segment_root / f"in_{idx:06d}"
        segment_out = segment_root / f"out_{idx:06d}"
        if segment_in.exists():
            shutil.rmtree(segment_in, ignore_errors=True)
        if segment_out.exists():
            shutil.rmtree(segment_out, ignore_errors=True)
        segment_in.mkdir(parents=True, exist_ok=True)
        segment_out.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(input_frames[idx], segment_in / "00000001.png")
        shutil.copyfile(input_frames[idx + 1], segment_in / "00000002.png")

        rife_cmd = [
            rife_bin,
            "-i",
            str(segment_in),
            "-o",
            str(segment_out),
            "-n",
            str(segment_frames),
            "-f",
            "%08d.png",
        ]
        if rife_gpu_id is not None:
            rife_cmd += ["-g", str(rife_gpu_id)]
        if rife_model:
            rife_cmd += ["-m", rife_model]
        append_log(log_path, f"[POST_RESTORE] RIFE guided segment {idx} cmd: {' '.join(rife_cmd)}")
        rife_env = dict(env) if env is not None else dict(os.environ)
        rife_env.pop("DISPLAY", None)
        rife_env.pop("WAYLAND_DISPLAY", None)
        run_cmd(
            rife_cmd,
            cwd=None,
            env=rife_env,
            log_path=log_path,
            title=f"STEP5.5 Post-restore RIFE guided segment {idx}",
        )

        segment_frames_paths = sorted(segment_out.glob("*.png"))
        if not segment_frames_paths:
            raise RuntimeError("RIFE guided segment produced no frames")
        if idx > 0:
            segment_frames_paths = segment_frames_paths[1:]
        for frame_path in segment_frames_paths:
            target_name = f"{out_index:08d}.png"
            shutil.copyfile(frame_path, output_frames_dir / target_name)
            out_index += 1

    append_log(log_path, f"[POST_RESTORE] guided RIFE output frames={out_index - 1}")


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


def run_rife_pytorch_guided(
    *,
    input_frames_dir: Path,
    output_frames_dir: Path,
    frame_map: list[int],
    segment_frames: list[int],
    time_steps: list[list[float]],
    scale: float,
    repo_dir: Path,
    model_dir: Path,
    conda_env: str,
    env: Dict[str, str],
    log_path: Path,
) -> None:
    repo_dir = repo_dir.resolve()
    script_path = repo_dir / "guided_interpolation_cli.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"guided_interpolation_cli.py not found under {repo_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"RIFE model dir not found: {model_dir}")
    if not input_frames_dir.exists():
        raise FileNotFoundError(f"Input frames dir missing: {input_frames_dir}")

    validate_time_steps(time_steps, segment_frames)
    count_guided_output_frames(range(len(frame_map)), segment_frames, time_steps)

    frame_map_path = output_frames_dir.parent / "frame_map.json"
    frame_map_payload = {"sampled_indices": frame_map}
    frame_map_path.write_text(json.dumps(frame_map_payload, ensure_ascii=False), encoding="utf-8")

    time_steps_payload = json.dumps(time_steps, ensure_ascii=False)
    time_steps_path = output_frames_dir.parent / "time_steps.json"
    time_steps_path.write_text(time_steps_payload, encoding="utf-8")

    output_frames_dir.mkdir(parents=True, exist_ok=True)

    input_frames = sorted(p for p in input_frames_dir.iterdir() if p.is_file())
    if not input_frames:
        raise FileNotFoundError(f"No input frames found in {input_frames_dir}")

    output_prefix = output_frames_dir.as_posix().rstrip("/") + "/"
    cmd = [
        "python3",
        str(script_path),
        "--imgs",
        *[str(p) for p in input_frames],
        "--output",
        output_prefix,
        "--model",
        str(model_dir),
        "--scale",
        str(scale),
        "--frame_map",
        *[str(idx) for idx in frame_map],
        "--segment_frames",
        *[str(count) for count in segment_frames],
        "--time_steps_json",
        time_steps_payload,
    ]
    cmd = conda_run(conda_env, cmd) if conda_env else cmd
    run_cmd(
        cmd,
        cwd=repo_dir,
        env=env or dict(os.environ),
        log_path=log_path,
        title="STEP5.5 Post-restore RIFE (pytorch) guided interpolation",
    )


def restore_duration(
    *,
    output_dir: Path,
    input_video_path: Path,
    original_stats: VideoStats,
    mode: str,
    rife_bin: str | None,
    rife_model: str | None,
    rife_backend: str,
    rife_pytorch_repo: str | None,
    rife_pytorch_model_dir: str | None,
    rife_pytorch_env: str,
    rife_scale: float,
    rife_keep_audio: bool,
    rife_time_steps_mode: str,
    rife_time_steps_cache: Path | None,
    rife_sam_mask_dir: Path | None,
    rife_time_steps_force_sam: bool,
    rife_sam_generate: bool,
    rife_sam_bbox_json: Path | None,
    rife_sam_cleanup_workdir: bool,
    rife_sam_project_root: Path,
    rife_sam_conda_env: str,
    rife_pose_subset: str,
    rife_time_steps_debug: bool,
    rife_cache_root: Path,
    frame_map_path: Path | None,
    run_gpus: list[str] | None,
    job_index: int,
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
        target_fps = original_stats.fps or 0.0
        if target_fps <= 0:
            if current_stats.fps and current_stats.fps > 0:
                append_log(log_path, "[POST_RESTORE] original fps unavailable, fallback to current fps")
                target_fps = current_stats.fps
            else:
                raise RuntimeError("Unable to resolve target fps for post-restore.")

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
            rife_gpu_id = _select_rife_gpu_id(run_gpus=run_gpus, job_index=job_index, env=env)
            append_log(log_path, f"[POST_RESTORE] rife_gpu_id={rife_gpu_id}")
            console_print(f"[POST_RESTORE] rife_gpu_id={rife_gpu_id}")
            rife_workdir = rife_cache_root / f"{output_dir.name}_{short_hash(str(output_dir.resolve()), 8)}"
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
                frame_map = _load_frame_map(
                    frame_map_path=frame_map_path,
                    expected_input_frames=len(list(input_frames_dir.glob("*.png"))),
                    expected_output_frames=target_frames,
                    log_path=log_path,
                )
                if frame_map:
                    guided_dir = rife_workdir / "guided_frames"
                    guided_dir.mkdir(parents=True, exist_ok=True)
                    _run_rife_guided(
                        input_frames_dir=input_frames_dir,
                        output_frames_dir=guided_dir,
                        sample_indices=frame_map["sampled_indices"],
                        rife_bin=rife_bin,
                        rife_model=rife_model,
                        rife_gpu_id=rife_gpu_id,
                        env=env,
                        log_path=log_path,
                    )
                    output_frames_dir = guided_dir
                else:
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
                    if rife_gpu_id is not None:
                        rife_cmd += ["-g", str(rife_gpu_id)]
                    if rife_model:
                        rife_cmd += ["-m", rife_model]
                    append_log(log_path, f"[POST_RESTORE] RIFE cmd: {' '.join(rife_cmd)}")
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
            rife_workdir = rife_cache_root / f"{output_dir.name}_{short_hash(str(output_dir.resolve()), 8)}__pytorch"
            try:
                if rife_workdir.exists():
                    shutil.rmtree(rife_workdir, ignore_errors=True)
                rife_workdir.mkdir(parents=True, exist_ok=True)
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
                frame_map = _load_frame_map(
                    frame_map_path=frame_map_path,
                    expected_input_frames=_probe_frame_count(speed_path, env=env, log_path=log_path),
                    expected_output_frames=target_frames,
                    log_path=log_path,
                )
                if frame_map:
                    input_frames_dir = rife_workdir / "input_frames"
                    output_frames_dir = rife_workdir / "output_frames"
                    if input_frames_dir.exists():
                        shutil.rmtree(input_frames_dir, ignore_errors=True)
                    if output_frames_dir.exists():
                        shutil.rmtree(output_frames_dir, ignore_errors=True)
                    input_frames_dir.mkdir(parents=True, exist_ok=True)
                    output_frames_dir.mkdir(parents=True, exist_ok=True)
                    orig_w, orig_h = probe_video_dimensions(speed_path, env=env, log_path=log_path)
                    pad_w, pad_h = get_pad_dims(orig_w, orig_h, align=32)
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
                        title="STEP5.5 Post-restore RIFE pytorch extract frames",
                    )
                    segment_frames = _segment_frames_from_map(frame_map["sampled_indices"])
                    if rife_time_steps_cache is None:
                        rife_time_steps_cache = rife_cache_root / "time_steps.json"
                    parquet_path = None
                    if rife_time_steps_mode != "uniform":
                        try:
                            parquet_path = parquet_path_for_video(input_video_path)
                        except ValueError as exc:
                            append_log(log_path, f"[POST_RESTORE] parquet lookup failed: {exc}")
                    image_w, image_h = probe_video_dimensions(input_video_path, env=env, log_path=log_path)
                    sam_mask_dir = rife_sam_mask_dir
                    if rife_time_steps_mode in {"sam", "sam_zarm_hybrid"}:
                        if sam_mask_dir is None:
                            sam_mask_dir = rife_cache_root / f"{output_dir.name}_sam_masks"
                        if not _mask_dir_has_files(sam_mask_dir):
                            if rife_sam_generate:
                                if rife_sam_bbox_json is None:
                                    raise ValueError(
                                        "rife_sam_bbox_json must be set when rife_sam_generate is enabled."
                                    )
                                if not rife_sam_bbox_json.exists():
                                    raise FileNotFoundError(
                                        f"rife_sam_bbox_json not found: {rife_sam_bbox_json}"
                                    )
                                sam_workdir = rife_cache_root / f"{output_dir.name}_sam_workdir"
                                run_arm_swap_sam_masks(
                                    project_root=rife_sam_project_root,
                                    conda_env=rife_sam_conda_env,
                                    orig_video=input_video_path,
                                    workdir=sam_workdir,
                                    bbox_json=rife_sam_bbox_json,
                                    mask_dir=sam_mask_dir,
                                    cleanup_workdir=rife_sam_cleanup_workdir,
                                    env=env,
                                    log_path=log_path,
                                )
                            else:
                                append_log(
                                    log_path,
                                    f"[POST_RESTORE] SAM mask dir missing or empty: {sam_mask_dir}",
                                )
                    time_steps_result = compute_time_steps(
                        frame_map=frame_map["sampled_indices"],
                        segment_frames=segment_frames,
                        parquet_path=str(parquet_path) if parquet_path else None,
                        pose_indices=None,
                        pose_name_prefix=_pose_prefixes_for_subset(rife_pose_subset),
                        sam_masks=sam_mask_dir,
                        image_wh=(image_w, image_h),
                        mode=rife_time_steps_mode,
                        cache_path=rife_time_steps_cache,
                        debug=rife_time_steps_debug,
                        force_mode=rife_time_steps_force_sam,
                    )
                    if rife_time_steps_debug:
                        time_steps, debug_info = time_steps_result
                    else:
                        time_steps = time_steps_result
                        debug_info = None
                    validate_time_steps(time_steps, segment_frames)
                    if debug_info:
                        append_log(
                            log_path,
                            (
                                "[POST_RESTORE] time_steps mode="
                                f"{debug_info.mode_requested} used={debug_info.mode_used} "
                                f"fallback={debug_info.fallback_reason} sam_quality={debug_info.sam_quality_ratio}"
                            ),
                        )
                        if debug_info.alpha_stats:
                            append_log(
                                log_path,
                                (
                                    "[POST_RESTORE] time_steps alpha_stats="
                                    f"{debug_info.alpha_stats}"
                                ),
                            )
                        for idx, steps in enumerate(time_steps):
                            if steps:
                                stats = {
                                    "min": min(steps),
                                    "mean": sum(steps) / len(steps),
                                    "max": max(steps),
                                }
                            else:
                                stats = {"min": 0.0, "mean": 0.0, "max": 0.0}
                            sam_ratio = None
                            if debug_info.sam_q is not None:
                                start = frame_map["sampled_indices"][idx]
                                end = frame_map["sampled_indices"][idx + 1]
                                sam_slice = debug_info.sam_q[start + 1 : end + 1]
                                if sam_slice:
                                    sam_ratio = sum(1 for val in sam_slice if val >= 0.7) / len(sam_slice)
                            append_log(
                                log_path,
                                (
                                    f"[POST_RESTORE] segment={idx} t_stats={stats} "
                                    f"sam_ratio={sam_ratio}"
                                ),
                            )
                    run_rife_pytorch_guided(
                        input_frames_dir=input_frames_dir,
                        output_frames_dir=output_frames_dir,
                        frame_map=frame_map["sampled_indices"],
                        segment_frames=segment_frames,
                        time_steps=time_steps,
                        scale=rife_scale,
                        repo_dir=repo_dir,
                        model_dir=model_dir,
                        conda_env=rife_pytorch_env,
                        env=env,
                        log_path=log_path,
                    )
                    output_frames = sorted(output_frames_dir.glob("*.png"))
                    if not output_frames:
                        prefix = output_frames_dir.name
                        misplaced = sorted(output_frames_dir.parent.glob(f"{prefix}*.png"))
                        if misplaced:
                            output_frames_dir.mkdir(parents=True, exist_ok=True)
                            for frame_path in misplaced:
                                suffix = frame_path.name[len(prefix) :].lstrip("_")
                                target = output_frames_dir / (suffix or frame_path.name)
                                shutil.move(str(frame_path), str(target))
                            output_frames = sorted(output_frames_dir.glob("*.png"))
                    if len(output_frames) != target_frames:
                        raise RuntimeError(
                            f"RIFE pytorch guided output frame mismatch: got={len(output_frames)} expected={target_frames}"
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
                        str(transfer_mp4),
                    ]
                    run_cmd(
                        encode_cmd,
                        cwd=None,
                        env=env or dict(os.environ),
                        log_path=log_path,
                        title="STEP5.5 Post-restore RIFE pytorch encode",
                    )
                else:
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
                if rife_workdir.exists():
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
