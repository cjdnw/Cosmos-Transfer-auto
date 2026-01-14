from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
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


def _has_mask_files(mask_dir: Path) -> bool:
    if not mask_dir.exists():
        return False
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.npy"]
    for pattern in patterns:
        if any(mask_dir.glob(pattern)):
            return True
    return False


def _parse_fraction(value: str) -> float:
    if "/" in value:
        num_str, den_str = value.split("/", 1)
        den = float(den_str)
        if den == 0:
            return 0.0
        return float(num_str) / den
    return float(value)


def _probe_video_info(video_path: Path, log_path: Path, step_label: str) -> tuple[int, int]:
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
    append_log(log_path, f"[{step_label}] ffprobe: {' '.join(cmd)}")
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
    append_log(log_path, f"[{step_label}] ffprobe parsed fps={fps} width={width}")
    return fps, width


def _probe_video_dims(video_path: Path, log_path: Path, step_label: str) -> tuple[int, int]:
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
    append_log(log_path, f"[{step_label}] ffprobe dims: {' '.join(cmd)}")
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
    append_log(log_path, f"[{step_label}] ffprobe parsed width={width} height={height}")
    return width, height


def _probe_video_meta(video_path: Path, log_path: Path, step_label: str) -> tuple[int, int, int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(video_path),
    ]
    append_log(log_path, f"[{step_label}] ffprobe meta: {' '.join(cmd)}")
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
    avg_frame_rate = stream.get("avg_frame_rate", "0/0")
    fps_float = _parse_fraction(avg_frame_rate)
    fps = int(round(fps_float))
    frame_count = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
    if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
        raise RuntimeError(
            f"Invalid video meta from ffprobe: fps={fps} width={width} height={height} frames={frame_count}"
        )
    append_log(
        log_path,
        f"[{step_label}] ffprobe parsed fps={fps} width={width} height={height} frames={frame_count}",
    )
    return fps, width, height, frame_count


def _build_video_signature(video_path: Path) -> Dict[str, str | int]:
    stat = video_path.stat()
    digest = hashlib.sha256(f"{video_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")).hexdigest()
    return {
        "path": str(video_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
    }


def _load_cache_meta(cache_meta_path: Path) -> Dict[str, object] | None:
    if not cache_meta_path.exists():
        return None
    try:
        return json.loads(cache_meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _cache_meta_matches(existing: Dict[str, object], expected: Dict[str, object]) -> bool:
    keys = ["orig_signature", "fps", "width", "height", "frames", "prompt", "bbox_json"]
    for key in keys:
        if existing.get(key) != expected.get(key):
            return False
    return True


def ensure_qwen_sam_cache(
    *,
    project_root: Path,
    conda_env: str,
    orig_video: Path,
    workdir: Path,
    qwen_prompt: str,
    env: Dict[str, str],
    log_path: Path,
    log_label: str,
) -> Path:
    cache_root = workdir / "qwen_sam"
    cache_root.mkdir(parents=True, exist_ok=True)
    qwen_boxes_json = cache_root / "qwen_boxes.json"
    sam_track_dir = cache_root / "sam3_track"
    masks_dir = cache_root / "masks_arm"
    cache_meta_path = cache_root / "cache_meta.json"

    fps, width, height, frame_count = _probe_video_meta(orig_video, log_path, log_label)
    expected_meta = {
        "orig_signature": _build_video_signature(orig_video),
        "fps": fps,
        "width": width,
        "height": height,
        "frames": frame_count,
        "prompt": qwen_prompt,
        "bbox_json": str(qwen_boxes_json),
    }

    cache_meta = _load_cache_meta(cache_meta_path)
    if cache_meta and not _cache_meta_matches(cache_meta, expected_meta):
        append_log(log_path, f"[{log_label}] qwen_sam cache_meta mismatch; invalidating cache.")
        shutil.rmtree(cache_root, ignore_errors=True)
        cache_root.mkdir(parents=True, exist_ok=True)

    if qwen_boxes_json.exists() and _has_mask_files(masks_dir) and cache_meta:
        append_log(log_path, f"[{log_label}] qwen_sam cache hit: {cache_root}")
        return masks_dir

    if not qwen_boxes_json.exists():
        cmd = conda_run(
            conda_env,
            [
                "python",
                "-m",
                "robot_arm_swap",
                "--orig",
                str(orig_video),
                "--workdir",
                str(cache_root),
                "--fps",
                str(fps),
                "--width",
                str(width),
                "--qwen-prompt",
                qwen_prompt,
                "--qwen-only",
                "--bbox-json",
                str(qwen_boxes_json),
            ],
        )
        run_cmd(cmd, cwd=project_root, env=env, log_path=log_path, title=f"{log_label}: Qwen bbox")
        if not qwen_boxes_json.exists() or qwen_boxes_json.stat().st_size == 0:
            raise FileNotFoundError(f"Qwen bbox json not found after run: {qwen_boxes_json}")

    if not _has_mask_files(masks_dir):
        cmd = conda_run(
            conda_env,
            [
                "python",
                "-m",
                "robot_arm_swap",
                "--orig",
                str(orig_video),
                "--workdir",
                str(cache_root),
                "--fps",
                str(fps),
                "--width",
                str(width),
                "--sam-only",
                "--bbox-json",
                str(qwen_boxes_json),
            ],
        )
        run_cmd(cmd, cwd=project_root, env=env, log_path=log_path, title=f"{log_label}: SAM masks")
        if not _has_mask_files(masks_dir):
            raise FileNotFoundError(f"SAM mask output not found after run: {masks_dir}")

    cache_meta_payload = dict(expected_meta)
    cache_meta_payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cache_meta_payload["sam_track_dir"] = str(sam_track_dir)
    cache_meta_payload["masks_dir"] = str(masks_dir)
    cache_meta_path.write_text(json.dumps(cache_meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    append_log(log_path, f"[{log_label}] qwen_sam cache ready: {cache_root}")
    return masks_dir


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
    log_title: str = "STEP6: Robot arm swap",
    log_label: str = "STEP6",
) -> Path:
    """Legacy arm swap runner (qwen+sam or qwen-only/sam-only)."""
    output_base_dir.mkdir(parents=True, exist_ok=True)
    workdir.parent.mkdir(parents=True, exist_ok=True)

    transfer_mp4 = _find_transfer_output_mp4(output_dir)
    output_path = output_base_dir / transfer_mp4.name

    workdir.mkdir(parents=True, exist_ok=True)
    fps, width = _probe_video_info(orig_video, log_path, log_label)
    cmd = [
        "python",
        "-m",
        "robot_arm_swap",
        "--orig",
        str(orig_video),
        "--workdir",
        str(workdir),
        "--fps",
        str(fps),
        "--width",
        str(width),
    ]
    if qwen_only:
        cmd.extend(["--qwen-prompt", qwen_prompt, "--qwen-only"])
    elif sam_only:
        cmd.append("--sam-only")
    else:
        cmd.extend(
            [
                "--skin",
                str(transfer_mp4),
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
    if bbox_json is not None:
        cmd.extend(["--bbox-json", str(bbox_json)])
    if no_workdir_write:
        cmd.append("--no-workdir-write")
    cmd = conda_run(conda_env, cmd)
    try:
        run_cmd(cmd, cwd=project_root, env=env, log_path=log_path, title=log_title)
        if qwen_only:
            if bbox_json is not None:
                if not bbox_json.exists() or bbox_json.stat().st_size == 0:
                    raise FileNotFoundError(f"Arm swap bbox json not found after run: {bbox_json}")
                return bbox_json
            return output_path
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise FileNotFoundError(f"Arm swap output not found after run: {output_path}")
        return output_path
    finally:
        if cleanup_workdir and no_workdir_write:
            _cleanup_workdir(workdir)


def run_arm_swap_swap_only(
    *,
    project_root: Path,
    conda_env: str,
    orig_video: Path,
    skin_video: Path,
    workdir: Path,
    arm_mask_dir: Path,
    output_path: Path,
    erode: int,
    feather: int,
    dilate: int,
    no_workdir_write: bool,
    arm_swap_workdir_write: bool,
    env: Dict[str, str],
    log_path: Path,
    log_title: str = "STEP6: Robot arm swap (swap-only)",
) -> Path:
    workdir.parent.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not arm_mask_dir.exists():
        raise FileNotFoundError(f"Arm mask dir not found: {arm_mask_dir}")
    fps, width = _probe_video_info(orig_video, log_path, "STEP6")
    cmd = conda_run(
        conda_env,
        [
            "python",
            "-m",
            "robot_arm_swap",
            "--orig",
            str(orig_video),
            "--skin",
            str(skin_video),
            "--workdir",
            str(workdir),
            "--fps",
            str(fps),
            "--width",
            str(width),
            "--swap-only",
            "--arm-mask-dir",
            str(arm_mask_dir),
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
    if arm_swap_workdir_write:
        cmd.append("--arm-swap-workdir-write")
    elif no_workdir_write:
        cmd.append("--no-workdir-write")
    run_cmd(cmd, cwd=project_root, env=env, log_path=log_path, title=log_title)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise FileNotFoundError(f"Arm swap output not found after run: {output_path}")
    return output_path


def run_arm_swap_sam_masks(
    *,
    project_root: Path,
    conda_env: str,
    orig_video: Path,
    workdir: Path,
    bbox_json: Path,
    mask_dir: Path,
    cleanup_workdir: bool,
    env: Dict[str, str],
    log_path: Path,
) -> Path:
    """
    Generate SAM masks only (no skin swap).

    Command:
      python -m robot_arm_swap --orig <input> --workdir <dir> --fps <fps> --width <width>
        --sam-only --bbox-json <path>
        --mask-lossless-frames-dir <mask_dir>
    """
    workdir.parent.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    fps, width = _probe_video_info(orig_video, log_path, "STEP5.5")
    cmd = conda_run(
        conda_env,
        [
            "python",
            "-m",
            "robot_arm_swap",
            "--orig",
            str(orig_video),
            "--workdir",
            str(workdir),
            "--fps",
            str(fps),
            "--width",
            str(width),
            "--sam-only",
            "--bbox-json",
            str(bbox_json),
            "--mask-lossless-frames-dir",
            str(mask_dir),
        ],
    )
    try:
        run_cmd(cmd, cwd=project_root, env=env, log_path=log_path, title="STEP5.5: SAM mask generation")
        if not _has_mask_files(mask_dir):
            raise FileNotFoundError(f"SAM mask output not found after run: {mask_dir}")
        return mask_dir
    finally:
        if cleanup_workdir:
            _cleanup_workdir(workdir)
