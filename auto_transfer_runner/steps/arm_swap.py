from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..utils.logging_utils import append_log
from ..utils.logging_utils import console_print
from ..utils.subprocess_utils import conda_run, run_cmd


def _find_transfer_output_mp4(output_dir: Path) -> Path:
    mp4s: List[Path] = [p for p in output_dir.glob("*.mp4") if p.is_file()]
    if not mp4s:
        raise FileNotFoundError(f"No mp4 outputs found in transfer output_dir: {output_dir}")
    if len(mp4s) > 1:
        return max(mp4s, key=lambda p: p.stat().st_size)
    return mp4s[0]


def _cleanup_workdir(workdir: Path) -> None:
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)


def _parse_fraction(value: str) -> float:
    if "/" in value:
        num_str, den_str = value.split("/", 1)
        den = float(den_str)
        if den == 0:
            return 0.0
        return float(num_str) / den
    return float(value)


def _probe_video_info(video_path: Path, log_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    append_log(log_path, f"[STEP6] ffprobe: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        append_log(log_path, result.stdout)
        append_log(log_path, result.stderr)
        raise RuntimeError(f"ffprobe failed for {video_path} (code={result.returncode}). See log: {log_path}")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe returned no video streams for {video_path}")
    stream = streams[0]
    width = int(stream["width"])
    avg_frame_rate = stream.get("avg_frame_rate", "0/0")
    fps_float = _parse_fraction(avg_frame_rate)
    fps = int(round(fps_float))
    if fps <= 0 or width <= 0:
        raise RuntimeError(f"Invalid video info from ffprobe: fps={fps}, width={width}")
    append_log(log_path, f"[STEP6] ffprobe parsed fps={fps} width={width}")
    return fps, width


def _probe_video_dims(video_path: Path, log_path: Path) -> tuple[int, int]:
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
    append_log(log_path, f"[STEP6] ffprobe dims: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        append_log(log_path, result.stdout)
        append_log(log_path, result.stderr)
        raise RuntimeError(f"ffprobe failed for {video_path} (code={result.returncode}). See log: {log_path}")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe returned no video streams for {video_path}")
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video dims from ffprobe: width={width}, height={height}")
    append_log(log_path, f"[STEP6] ffprobe parsed width={width} height={height}")
    return width, height


def _mask_candidates(workdir: Path) -> list[Path]:
    if not workdir.exists():
        return []
    return [path for path in workdir.rglob("*.mp4") if "mask" in path.name.lower()]


def _select_mask_pair(mask_videos: list[Path]) -> tuple[Path, Path] | None:
    if len(mask_videos) < 2:
        return None
    if len(mask_videos) == 2:
        return mask_videos[0], mask_videos[1]
    tag_groups = {
        "orig": {"orig", "source", "input"},
        "skin": {"skin", "swap", "transfer", "target", "output"},
    }
    lower_map = {path: path.name.lower() for path in mask_videos}
    orig_candidates = [p for p in mask_videos if any(tag in lower_map[p] for tag in tag_groups["orig"])]
    skin_candidates = [p for p in mask_videos if any(tag in lower_map[p] for tag in tag_groups["skin"])]
    if orig_candidates and skin_candidates:
        orig = max(orig_candidates, key=lambda p: p.stat().st_size)
        skin = max(skin_candidates, key=lambda p: p.stat().st_size)
        if orig != skin:
            return orig, skin
    sorted_masks = sorted(mask_videos, key=lambda p: p.stat().st_size, reverse=True)
    return sorted_masks[0], sorted_masks[1]


def _compute_mean_mask_iou(
    mask_a: Path,
    mask_b: Path,
    *,
    log_path: Path,
) -> float | None:
    width_a, height_a = _probe_video_dims(mask_a, log_path)
    width_b, height_b = _probe_video_dims(mask_b, log_path)
    if (width_a, height_a) != (width_b, height_b):
        append_log(
            log_path,
            f"[STEP6] mask dims mismatch: {mask_a.name} ({width_a}x{height_a}) vs "
            f"{mask_b.name} ({width_b}x{height_b}); skip IoU",
        )
        return None
    frame_size = width_a * height_a
    cmd_a = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(mask_a),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    cmd_b = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(mask_b),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    proc_a = subprocess.Popen(cmd_a, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc_b = subprocess.Popen(cmd_b, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    total_iou = 0.0
    frame_count = 0
    try:
        while True:
            if proc_a.stdout is None or proc_b.stdout is None:
                break
            data_a = proc_a.stdout.read(frame_size)
            data_b = proc_b.stdout.read(frame_size)
            if len(data_a) < frame_size or len(data_b) < frame_size:
                break
            mask_a_frame = np.frombuffer(data_a, dtype=np.uint8).reshape(height_a, width_a) > 0
            mask_b_frame = np.frombuffer(data_b, dtype=np.uint8).reshape(height_a, width_a) > 0
            union = np.logical_or(mask_a_frame, mask_b_frame).sum()
            if union == 0:
                iou = 1.0
            else:
                intersection = np.logical_and(mask_a_frame, mask_b_frame).sum()
                iou = float(intersection) / float(union)
            total_iou += iou
            frame_count += 1
    finally:
        if proc_a.stdout:
            proc_a.stdout.close()
        if proc_b.stdout:
            proc_b.stdout.close()
        stderr_a = proc_a.stderr.read().decode("utf-8", errors="ignore") if proc_a.stderr else ""
        stderr_b = proc_b.stderr.read().decode("utf-8", errors="ignore") if proc_b.stderr else ""
        proc_a.wait()
        proc_b.wait()
        if proc_a.returncode != 0:
            append_log(log_path, f"[STEP6] ffmpeg mask decode failed for {mask_a}: {stderr_a}")
        if proc_b.returncode != 0:
            append_log(log_path, f"[STEP6] ffmpeg mask decode failed for {mask_b}: {stderr_b}")
    if frame_count == 0:
        append_log(log_path, "[STEP6] no overlapping mask frames; skip IoU")
        return None
    return total_iou / float(frame_count)


def _log_mask_iou(workdir: Path, log_path: Path) -> None:
    mask_videos = _mask_candidates(workdir)
    mask_pair = _select_mask_pair(mask_videos)
    if mask_pair is None:
        append_log(log_path, "[STEP6] mask videos not found; skip mean IoU")
        return
    mask_a, mask_b = mask_pair
    append_log(log_path, f"[STEP6] mask IoU candidates: {mask_a} | {mask_b}")
    mean_iou = _compute_mean_mask_iou(mask_a, mask_b, log_path=log_path)
    if mean_iou is None:
        return
    append_log(log_path, f"[STEP6] mask mean IoU={mean_iou:.6f}")
    console_print(f"[STEP6] mask mean IoU={mean_iou:.6f}")


def run_arm_swap(
    *,
    project_root: Path,
    conda_env: str,
    output_dir: Path,
    output_base_dir: Path,
    workdir: Path,
    no_workdir_write: bool,
    cleanup_workdir: bool,
    orig_video: Path,
    qwen_prompt: str,
    erode: int,
    feather: int,
    dilate: int,
    qwen_only: bool,
    sam_only: bool,
    bbox_json: Path | None,
    env: Dict[str, str],
    log_path: Path,
) -> Path:
    """
    Step6: run robot arm swap (qwen3vl + sam3).

    Command:
      python -m robot_arm_swap --orig <input> --skin <transfer_output> --workdir <dir>
        --fps <fps> --width <width> --qwen-prompt "<prompt>" --out <output>
        [--qwen-only | --sam-only --bbox-json <path>]
    """
    output_base_dir.mkdir(parents=True, exist_ok=True)
    workdir.parent.mkdir(parents=True, exist_ok=True)

    transfer_mp4 = _find_transfer_output_mp4(output_dir)
    output_path = output_base_dir / transfer_mp4.name

    workdir.mkdir(parents=True, exist_ok=True)
    fps, width = _probe_video_info(orig_video, log_path)
    cmd = conda_run(
        conda_env,
        [
            "python",
            "-m",
            "robot_arm_swap",
            "--orig",
            str(orig_video),
            "--skin",
            str(transfer_mp4),
            "--workdir",
            str(workdir),
            "--fps",
            str(fps),
            "--width",
            str(width),
            "--qwen-prompt",
            qwen_prompt,
            "--erode",
            str(erode),
            "--feather",
            str(feather),
            "--dilate",
            str(dilate),
            "--out",
            str(output_path),
        ],
    )
    if qwen_only:
        cmd.append("--qwen-only")
    if sam_only:
        cmd.append("--sam-only")
    if bbox_json is not None:
        cmd.extend(["--bbox-json", str(bbox_json)])
    if no_workdir_write:
        cmd.append("--no-workdir-write")
    try:
        run_cmd(cmd, cwd=project_root, env=env, log_path=log_path, title="STEP6: Robot arm swap")
        if qwen_only:
            if bbox_json is not None:
                if not bbox_json.exists() or bbox_json.stat().st_size == 0:
                    raise FileNotFoundError(f"Arm swap bbox json not found after run: {bbox_json}")
                return bbox_json
            return output_path
        _log_mask_iou(workdir, log_path)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise FileNotFoundError(f"Arm swap output not found after run: {output_path}")
        return output_path
    finally:
        if cleanup_workdir:
            _cleanup_workdir(workdir)
