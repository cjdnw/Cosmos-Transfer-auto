import argparse
import importlib.util
import json
import math
import logging
import os
import sys
import tempfile
import subprocess
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from robot_arm_swap.config import QWEN3_VL_WRAPPER_DIR, SAM3_VIDEO_TRACKER_DIR
from robot_arm_swap.external_wrappers import load_qwen3_vl_class, load_sam3_video_tracker_class
from robot_arm_swap.utils import assert_ffmpeg, ensure_dir, sorted_glob, CmdError, run_cmd
from robot_arm_swap import extract_frames, composite, encode_video

LOGGER = logging.getLogger(__name__)


def _log_mask_iou(mean_iou: float, min_iou: float, max_iou: float) -> None:
    message = f"[IoU] mean={mean_iou:.6f} min={min_iou:.6f} max={max_iou:.6f}"
    print(message)
    LOGGER.info(message)


def _load_frames_from_dir(image_dir: str):
    frame_paths = sorted_glob(os.path.join(image_dir, "*.*"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {image_dir}")
    frames = []
    for path in frame_paths:
        frame = cv2.imread(path)
        if frame is None:
            raise FileNotFoundError(path)
        frames.append(frame)
    return frames, frame_paths


def _write_masks_from_sam3(sam3_mask_dir: str, out_mask_dir: str) -> None:
    mask_paths = sorted_glob(os.path.join(sam3_mask_dir, "mask_union_*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No SAM3 union masks found in {sam3_mask_dir}")
    ensure_dir(out_mask_dir)
    for idx, mask_path in enumerate(mask_paths, start=1):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        cv2.imwrite(os.path.join(out_mask_dir, f"{idx:06d}.png"), mask)


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.float32)
    if mask.max() <= 1.0:
        mask = mask * 255.0
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    return mask


def _bbox_from_mask(mask: np.ndarray) -> List[int]:
    mask = _normalize_mask(mask)
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("Mask contains no foreground pixels to derive bbox.")
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    return [x1, y1, x2, y2]


def _load_masks_from_dir(mask_dir: str) -> List[np.ndarray]:
    mask_paths = sorted_glob(os.path.join(mask_dir, "*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No mask frames found in {mask_dir}")
    masks = []
    for path in mask_paths:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        masks.append(mask)
    return masks


def _mask_to_coco_polygons(mask: np.ndarray) -> List[List[float]]:
    mask = _normalize_mask(mask)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[List[float]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        points = contour.reshape(-1, 2).astype(float).flatten().tolist()
        if len(points) >= 6:
            polygons.append(points)
    return polygons


def _write_coco_parquet(masks: List[np.ndarray], out_path: str) -> None:
    import json as _json
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows: List[str] = []
    for mask in masks:
        polygons = _mask_to_coco_polygons(mask)
        rows.append(_json.dumps(polygons, ensure_ascii=False))
    ensure_dir(os.path.dirname(out_path) or ".")
    new_column = pa.array(rows)
    if os.path.exists(out_path):
        existing = pq.read_table(out_path)
        if existing.num_rows != len(rows):
            raise ValueError(
                "Existing parquet row count does not match masks. "
                f"Expected {len(rows)}, got {existing.num_rows}."
            )
        column_name = "data"
        if column_name in existing.column_names:
            suffix = 1
            while f"{column_name}_{suffix}" in existing.column_names:
                suffix += 1
            column_name = f"{column_name}_{suffix}"
        table = existing.append_column(column_name, new_column)
    else:
        table = pa.Table.from_arrays([new_column], names=["data"])
    pq.write_table(table, out_path)


def _write_mask_frames(masks: List[np.ndarray], out_dir: str) -> None:
    ensure_dir(out_dir)
    for idx, mask in enumerate(masks, start=1):
        mask = _normalize_mask(mask)
        cv2.imwrite(os.path.join(out_dir, f"{idx:06d}.png"), mask)


def _write_lossless_mp4(frames_dir: str, fps: int, out_path: str) -> None:
    assert_ffmpeg()
    ensure_dir(os.path.dirname(out_path) or ".")
    frame_pattern = os.path.join(frames_dir, "%06d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        frame_pattern,
        "-c:v",
        "libx264",
        "-crf",
        "0",
        "-preset",
        "veryslow",
        "-pix_fmt",
        "yuv444p",
        out_path,
    ]
    run_cmd(cmd)


def _detect_frame_ext(frames_orig_dir: str, frames_skin_dir: str, preferred_ext: Optional[str]) -> Optional[str]:
    candidate_exts = []
    if preferred_ext:
        candidate_exts.append(preferred_ext.lower())
    for ext in ("png", "jpg", "jpeg"):
        if ext not in candidate_exts:
            candidate_exts.append(ext)
    for ext in candidate_exts:
        orig_paths = sorted_glob(os.path.join(frames_orig_dir, f"*.{ext}"))
        skin_paths = sorted_glob(os.path.join(frames_skin_dir, f"*.{ext}"))
        if orig_paths and skin_paths:
            return ext
    return None


def _compute_iou_stats(
    masks_orig_dir: str,
    masks_skin_dir: str,
    preferred_ext: Optional[str] = None,
) -> Optional[Tuple[float, float, float]]:
    if not os.path.isdir(masks_orig_dir) or not os.path.isdir(masks_skin_dir):
        print("mask frames not found; skip IoU")
        return None

    frame_ext = _detect_frame_ext(masks_orig_dir, masks_skin_dir, preferred_ext)
    if not frame_ext:
        print("mask frames not found; skip IoU")
        return None

    orig_paths = sorted_glob(os.path.join(masks_orig_dir, f"*.{frame_ext}"))
    skin_paths = sorted_glob(os.path.join(masks_skin_dir, f"*.{frame_ext}"))
    if not orig_paths or not skin_paths:
        print("mask frames not found; skip IoU")
        return None

    first_orig = cv2.imread(orig_paths[0], cv2.IMREAD_GRAYSCALE)
    first_skin = cv2.imread(skin_paths[0], cv2.IMREAD_GRAYSCALE)
    if first_orig is None or first_skin is None:
        print("mask frames not found; skip IoU")
        return None

    height, width = first_orig.shape[:2]
    if first_skin.shape[:2] != (height, width):
        raise RuntimeError("IoU input frames have mismatched dimensions.")

    assert_ffmpeg()
    frame_size = width * height
    orig_cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-f", "image2",
        "-i", os.path.join(masks_orig_dir, f"%06d.{frame_ext}"),
        "-pix_fmt", "gray",
        "-f", "rawvideo",
        "-",
    ]
    skin_cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-f", "image2",
        "-i", os.path.join(masks_skin_dir, f"%06d.{frame_ext}"),
        "-pix_fmt", "gray",
        "-f", "rawvideo",
        "-",
    ]
    orig_proc = subprocess.Popen(orig_cmd, stdout=subprocess.PIPE)
    skin_proc = subprocess.Popen(skin_cmd, stdout=subprocess.PIPE)
    ious = []
    try:
        while True:
            orig_bytes = orig_proc.stdout.read(frame_size)
            skin_bytes = skin_proc.stdout.read(frame_size)
            if len(orig_bytes) < frame_size or len(skin_bytes) < frame_size:
                break
            orig_frame = np.frombuffer(orig_bytes, dtype=np.uint8)
            skin_frame = np.frombuffer(skin_bytes, dtype=np.uint8)
            orig_mask = orig_frame > 0
            skin_mask = skin_frame > 0
            union = np.logical_or(orig_mask, skin_mask)
            union_count = int(union.sum())
            if union_count == 0:
                iou = 1.0
            else:
                iou = float(np.logical_and(orig_mask, skin_mask).sum()) / union_count
            ious.append(iou)
    finally:
        if orig_proc.stdout:
            orig_proc.stdout.close()
        if skin_proc.stdout:
            skin_proc.stdout.close()
        orig_proc.wait()
        skin_proc.wait()

    if not ious:
        print("mask frames not found; skip IoU")
        return None

    mean_iou = float(sum(ious)) / len(ious)
    min_iou = float(min(ious))
    max_iou = float(max(ious))
    _log_mask_iou(mean_iou, min_iou, max_iou)
    return mean_iou, min_iou, max_iou


def _estimate_sam3_video_bytes(frames_bgr):
    if not torch.cuda.is_available() or not frames_bgr:
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    h, w = frames_bgr[0].shape[:2]
    bytes_per_frame = h * w * 3 * 4
    estimated_bytes = bytes_per_frame * len(frames_bgr)
    target_bytes = free_bytes * 0.7
    return {
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "estimated_bytes": estimated_bytes,
        "target_bytes": target_bytes,
        "frames": len(frames_bgr),
        "height": h,
        "width": w,
    }


def _auto_adjust_sam3_scale(frames_bgr, scale: int, enabled: bool) -> int:
    if not enabled:
        return scale

    stats = _estimate_sam3_video_bytes(frames_bgr)
    if stats is None:
        return scale

    required_scale = max(1, math.ceil(math.sqrt(stats["estimated_bytes"] / stats["target_bytes"])))
    print(
        "[SAM3] Auto scale check: "
        f"frames={stats['frames']} size={stats['width']}x{stats['height']} "
        f"estimated={stats['estimated_bytes'] / (1024 ** 3):.2f} GiB "
        f"free={stats['free_bytes'] / (1024 ** 3):.2f} GiB "
        f"target={stats['target_bytes'] / (1024 ** 3):.2f} GiB "
        f"requested_scale={scale} required_scale={required_scale}"
    )

    if required_scale > scale:
        print(f"[SAM3] Auto scale enabled. Using scale={required_scale}.")
        return required_scale

    return scale


def _resize_frame_to_width(frame_bgr, width: int):
    h, w = frame_bgr.shape[:2]
    if width <= 0 or w == width:
        return frame_bgr
    scale = width / float(w)
    new_h = int(round(h * scale))
    if new_h % 2 == 1:
        new_h += 1
    return cv2.resize(frame_bgr, (width, new_h), interpolation=cv2.INTER_AREA)


def _read_video_frames(video_path: str, fps: int, width: int) -> List:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if not input_fps or input_fps <= 1e-6:
        input_fps = float(fps)
    frame_interval = 1.0 / float(input_fps)
    target_interval = 1.0 / float(fps)
    next_sample_time = 0.0
    current_time = 0.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if current_time + 1e-9 >= next_sample_time:
            frames.append(_resize_frame_to_width(frame, width))
            next_sample_time += target_interval
        current_time += frame_interval
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from: {video_path}")
    return frames


def _parse_ffmpeg_rate(rate: str) -> Optional[float]:
    if not rate or rate == "0/0":
        return None
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            den_val = float(den)
            if den_val == 0:
                return None
            return float(num) / den_val
        except ValueError:
            return None
    try:
        return float(rate)
    except ValueError:
        return None


def _probe_video_fps(video_path: str) -> Optional[float]:
    assert_ffmpeg()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,avg_frame_rate",
        "-of",
        "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    streams = payload.get("streams") or []
    if not streams:
        return None
    stream = streams[0]
    fps = _parse_ffmpeg_rate(stream.get("avg_frame_rate", ""))
    if fps is None:
        fps = _parse_ffmpeg_rate(stream.get("r_frame_rate", ""))
    return fps


def _transcode_to_h264_yuv420p(video_path: str, temp_dir: str, fallback_fps: int) -> str:
    assert_ffmpeg()
    fps = _probe_video_fps(video_path) or float(fallback_fps)
    with tempfile.NamedTemporaryFile(
        suffix=".mp4",
        prefix="temp_h264_",
        dir=temp_dir,
        delete=False,
    ) as tmp:
        out_path = tmp.name
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        f"{fps:.6f}",
        "-c:a",
        "copy",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        os.unlink(out_path)
        raise CmdError(f"ffmpeg transcode failed.\n{result.stderr}")
    return out_path


def _read_video_frames_with_fallback(
    video_path: str,
    fps: int,
    width: int,
    temp_dir: str,
) -> Tuple[List, Optional[str]]:
    try:
        return _read_video_frames(video_path, fps, width), None
    except RuntimeError as exc:
        message = str(exc)
        if "No frames read from" not in message and "Cannot open video" not in message:
            raise
    temp_path = _transcode_to_h264_yuv420p(video_path, temp_dir, fps)
    return _read_video_frames(temp_path, fps, width), temp_path


def _save_temp_frame(frame_bgr) -> str:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    if not cv2.imwrite(tmp_path, frame_bgr):
        raise RuntimeError("Failed to write temporary frame for Qwen3-VL.")
    return tmp_path


def _load_local_sam3_tracker_class():
    module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sam3_video_tracker.py"))
    spec = importlib.util.spec_from_file_location("sam3_video_tracker_local", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load local sam3_video_tracker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "SAM3VideoTracker"):
        raise AttributeError("Local sam3_video_tracker.py is missing SAM3VideoTracker.")
    return module.SAM3VideoTracker


def _composite_frames_in_memory(
    orig_frames,
    skin_frames,
    masks,
    dilate: int,
    erode: int,
    feather: int,
):
    import numpy as np

    n = min(len(orig_frames), len(skin_frames), len(masks))
    kernel = np.ones((3, 3), np.uint8)
    feather_size = feather if feather and feather % 2 == 1 else (feather + 1 if feather else 0)
    outputs = []
    for i in range(n):
        orig = orig_frames[i]
        skin = skin_frames[i]
        m = masks[i]
        if dilate > 0:
            m = cv2.dilate(m, kernel, iterations=dilate)
        if erode > 0:
            m = cv2.erode(m, kernel, iterations=erode)
        alpha = (m.astype(np.float32) / 255.0)
        if feather_size and feather_size > 0:
            alpha = cv2.GaussianBlur(alpha, (feather_size, feather_size), 0)
            alpha = np.clip(alpha, 0.0, 1.0)
        alpha = alpha[..., None]
        out = (orig.astype(np.float32) * alpha + skin.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        outputs.append(out)
    return outputs


def _encode_video_from_frames(frames_bgr, fps: int, out_path: str, audio_from: Optional[str]) -> None:
    assert_ffmpeg()
    if not frames_bgr:
        raise RuntimeError("No frames to encode.")
    h, w = frames_bgr[0].shape[:2]
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
    ]
    if audio_from:
        cmd += ["-i", audio_from, "-map", "0:v:0", "-map", "1:a?"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio_from:
        cmd += ["-c:a", "copy", "-shortest"]
    cmd += [out_path]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        assert process.stdin is not None
        for frame in frames_bgr:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
    if return_code != 0:
        raise CmdError(f"ffmpeg failed with exit code {return_code}.\n{stderr}")


def main():
    ap = argparse.ArgumentParser(description="End-to-end pipeline: extract -> qwen3vl boxes -> sam3 track -> composite -> encode.")
    ap.add_argument("--orig", help="Original video path")
    ap.add_argument("--skin", help="Reskinned video path")
    ap.add_argument("--workdir", default="work", help="Working directory")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--ext", default="jpg", choices=["jpg", "jpeg", "png"])
    ap.add_argument("--dilate", type=int, default=2)
    ap.add_argument("--erode", type=int, default=0)
    ap.add_argument("--feather", type=int, default=9)
    ap.add_argument("--out", default="output_swap.mp4")
    ap.add_argument("--qwen-wrapper-dir", default=QWEN3_VL_WRAPPER_DIR)
    ap.add_argument("--sam3-wrapper-dir", default=SAM3_VIDEO_TRACKER_DIR)
    ap.add_argument("--qwen-model-path", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--sam3-model-name", default="facebook/sam3")
    ap.add_argument("--qwen-prompt", default="robotic arm")
    ap.add_argument("--sam3-scale", type=int, default=1)
    ap.add_argument(
        "--sam3-auto-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-increase SAM3 scale to fit available GPU memory",
    )
    ap.add_argument(
        "--sam3-preprocess-device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for SAM3 preprocessing to reduce GPU peak memory",
    )
    ap.add_argument("--sam3-mask-threshold", type=float, default=0.0)
    ap.add_argument("--sam3-save-mp4", action="store_true", help="Save SAM3 mask/mp4 previews in workdir")
    ap.add_argument(
        "--no-workdir-write",
        "--in-memory",
        dest="no_workdir_write",
        action="store_true",
        help="Do not write intermediate frame/mask PNGs; keep them in memory when possible.",
    )
    ap.add_argument("--qwen-only", action="store_true", help="Only run Qwen bbox inference")
    ap.add_argument("--sam-only", action="store_true", help="Only run SAM tracking to generate masks using --bbox-json")
    ap.add_argument("--swap-only", action="store_true", help="Only run swap using --arm-mask-dir (skip Qwen detection)")
    ap.add_argument("--arm-mask-dir", help="Mask directory for swap-only (e.g., masks_arm from SAM)")
    ap.add_argument(
        "--arm-swap-workdir-write",
        action="store_true",
        help="When using --swap-only, also track masks on the skin video and compute IoU.",
    )
    ap.add_argument("--mask-coco-parquet", help="Write COCO polygon JSON strings to this parquet file (mask mode).")
    ap.add_argument("--mask-lossless-mp4", help="Write a lossless mask mp4 to this path (mask mode).")
    ap.add_argument(
        "--mask-lossless-frames-dir",
        help="Write mask PNG frames to this directory when creating lossless mp4 (mask mode).",
    )
    ap.add_argument(
        "--bbox-json",
        help="Path to Qwen bbox json (output for --qwen-only, input for --sam-only)",
    )
    args = ap.parse_args()

    mode_count = sum([args.qwen_only, args.sam_only, args.swap_only])
    if mode_count > 1:
        ap.error("--qwen-only, --sam-only, and --swap-only are mutually exclusive")
    if args.sam_only and not args.bbox_json:
        ap.error("--sam-only requires --bbox-json")
    if args.swap_only and not args.arm_mask_dir:
        ap.error("--swap-only requires --arm-mask-dir")
    if args.arm_swap_workdir_write and args.no_workdir_write:
        ap.error("--arm-swap-workdir-write cannot be used with --no-workdir-write")
    if args.mask_lossless_frames_dir and not args.mask_lossless_mp4:
        ap.error("--mask-lossless-frames-dir requires --mask-lossless-mp4")
    if not args.sam_only:
        if args.mask_coco_parquet or args.mask_lossless_mp4 or args.mask_lossless_frames_dir:
            ap.error("Mask output options require --sam-only")
    if not args.orig:
        ap.error("--orig is required")
    if args.swap_only and not args.skin:
        ap.error("--swap-only requires --skin")
    if not (args.qwen_only or args.sam_only or args.swap_only) and not args.skin:
        ap.error("--skin is required unless using --qwen-only/--sam-only/--swap-only")

    frames_orig = os.path.join(args.workdir, "frames_orig")
    frames_skin = os.path.join(args.workdir, "frames_skin")
    masks_arm = os.path.join(args.workdir, "masks_arm")
    masks_skin = os.path.join(args.workdir, "masks_skin")
    sam3_out = os.path.join(args.workdir, "sam3_track")
    sam3_out_skin = os.path.join(args.workdir, "sam3_track_skin")
    out_frames = os.path.join(args.workdir, "out_frames")
    qwen_json = args.bbox_json or os.path.join(args.workdir, "qwen_boxes.json")

    use_in_memory = args.no_workdir_write
    want_swap = args.skin is not None and not args.qwen_only and not args.sam_only
    write_qwen_json = True
    if use_in_memory and not args.bbox_json and not args.qwen_only:
        write_qwen_json = False
        qwen_json = None

    temp_videos = []
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if not use_in_memory:
        ensure_dir(args.workdir)

        # 1) extract frames
        from robot_arm_swap.extract_frames import extract
        extract(args.orig, frames_orig, args.fps, args.width, args.ext)
        if want_swap:
            extract(args.skin, frames_skin, args.fps, args.width, args.ext)
        frames_bgr = None
        skin_frames_bgr = None
    else:
        try:
            frames_bgr, temp_path = _read_video_frames_with_fallback(
                args.orig,
                args.fps,
                args.width,
                repo_root,
            )
            if temp_path:
                temp_videos.append(temp_path)
            skin_frames_bgr = None
            if want_swap:
                skin_frames_bgr, temp_path = _read_video_frames_with_fallback(
                    args.skin,
                    args.fps,
                    args.width,
                    repo_root,
                )
                if temp_path:
                    temp_videos.append(temp_path)
        finally:
            for path in temp_videos:
                if os.path.exists(path):
                    os.unlink(path)

    if args.swap_only:
        masks_arm = args.arm_mask_dir

    # 2) Qwen3-VL bbox on first frame
    if not (args.sam_only or args.swap_only):
        QwenVL3 = load_qwen3_vl_class(args.qwen_wrapper_dir)
        qwen = QwenVL3(args.qwen_model_path)
        if use_in_memory:
            first_frame_path = _save_temp_frame(frames_bgr[0])
        else:
            first_frame_path = os.path.join(frames_orig, f"000001.{args.ext}")
        try:
            qwen_results = qwen.predict(args.qwen_prompt, first_frame_path)
        finally:
            if use_in_memory:
                os.unlink(first_frame_path)
        if not qwen_results:
            raise RuntimeError("Qwen3-VL returned empty bbox list. Check prompt or model.")
        if write_qwen_json:
            ensure_dir(os.path.dirname(qwen_json) or ".")
            with open(qwen_json, "w", encoding="utf-8") as f:
                json.dump(qwen_results, f, ensure_ascii=False, indent=2)

        if args.qwen_only:
            return

    # 3) SAM3 tracking
    union_masks = None
    if not args.swap_only:
        if args.sam_only:
            if not os.path.exists(qwen_json):
                raise FileNotFoundError(qwen_json)
            with open(qwen_json, "r", encoding="utf-8") as f:
                qwen_results = json.load(f)

        boxes_raw = [item["bbox_2d"] for item in qwen_results if "bbox_2d" in item]
        if not boxes_raw:
            raise RuntimeError("Qwen3-VL results contain no bbox_2d entries.")

        if use_in_memory:
            SAM3VideoTracker = _load_local_sam3_tracker_class()
        else:
            SAM3VideoTracker = load_sam3_video_tracker_class(args.sam3_wrapper_dir)
        tracker = SAM3VideoTracker(args.sam3_model_name)
        if frames_bgr is None:
            frames_bgr, _ = _load_frames_from_dir(frames_orig)
        sam3_scale = _auto_adjust_sam3_scale(frames_bgr, args.sam3_scale, args.sam3_auto_scale)
        sam3_stats = _estimate_sam3_video_bytes(frames_bgr)
        sam3_preprocess_device = args.sam3_preprocess_device
        if sam3_preprocess_device == "auto":
            if torch.cuda.is_available():
                if sam3_stats and sam3_stats["estimated_bytes"] > sam3_stats["target_bytes"]:
                    sam3_preprocess_device = "cpu"
                    print("[SAM3] Auto preprocess device: using CPU to avoid GPU peak during normalization.")
                else:
                    sam3_preprocess_device = "cuda"
            else:
                sam3_preprocess_device = "cpu"
        if use_in_memory:
            union_masks = tracker.track_with_first_frame_boxes(
                frames_bgr=frames_bgr,
                first_frame_boxes_xyxy=boxes_raw,
                scale=sam3_scale,
                mask_threshold=args.sam3_mask_threshold,
                save_per_object=False,
                out_dir=sam3_out,
                fps=float(args.fps),
                save_mp4=False,
                processing_device=sam3_preprocess_device,
                return_union_masks=True,
                write_outputs=False,
            )
        else:
            tracker.track_with_first_frame_boxes(
                frames_bgr=frames_bgr,
                first_frame_boxes_xyxy=boxes_raw,
                scale=sam3_scale,
                mask_threshold=args.sam3_mask_threshold,
                save_per_object=False,
                out_dir=sam3_out,
                fps=float(args.fps),
                save_mp4=args.sam3_save_mp4,
                processing_device=sam3_preprocess_device,
            )
            _write_masks_from_sam3(os.path.join(sam3_out, "mask"), masks_arm)
            if want_swap:
                skin_frames_bgr, _ = _load_frames_from_dir(frames_skin)
                tracker.track_with_first_frame_boxes(
                    frames_bgr=skin_frames_bgr,
                    first_frame_boxes_xyxy=boxes_raw,
                    scale=sam3_scale,
                    mask_threshold=args.sam3_mask_threshold,
                    save_per_object=False,
                    out_dir=sam3_out_skin,
                    fps=float(args.fps),
                    save_mp4=args.sam3_save_mp4,
                    processing_device=sam3_preprocess_device,
                )
                _write_masks_from_sam3(os.path.join(sam3_out_skin, "mask"), masks_skin)
                _compute_iou_stats(masks_arm, masks_skin, "png")
    else:
        if use_in_memory:
            masks = _load_masks_from_dir(masks_arm)
            union_masks = [_normalize_mask(mask) for mask in masks]
        if args.arm_swap_workdir_write:
            masks_for_bbox = _load_masks_from_dir(masks_arm)
            if not masks_for_bbox:
                raise RuntimeError("No masks found for swap-only mask directory.")
            boxes_raw = [_bbox_from_mask(masks_for_bbox[0])]
            if use_in_memory:
                SAM3VideoTracker = _load_local_sam3_tracker_class()
            else:
                SAM3VideoTracker = load_sam3_video_tracker_class(args.sam3_wrapper_dir)
            tracker = SAM3VideoTracker(args.sam3_model_name)
            if skin_frames_bgr is None:
                skin_frames_bgr, _ = _load_frames_from_dir(frames_skin)
            sam3_scale = _auto_adjust_sam3_scale(skin_frames_bgr, args.sam3_scale, args.sam3_auto_scale)
            sam3_stats = _estimate_sam3_video_bytes(skin_frames_bgr)
            sam3_preprocess_device = args.sam3_preprocess_device
            if sam3_preprocess_device == "auto":
                if torch.cuda.is_available():
                    if sam3_stats and sam3_stats["estimated_bytes"] > sam3_stats["target_bytes"]:
                        sam3_preprocess_device = "cpu"
                        print("[SAM3] Auto preprocess device: using CPU to avoid GPU peak during normalization.")
                    else:
                        sam3_preprocess_device = "cuda"
                else:
                    sam3_preprocess_device = "cpu"
            tracker.track_with_first_frame_boxes(
                frames_bgr=skin_frames_bgr,
                first_frame_boxes_xyxy=boxes_raw,
                scale=sam3_scale,
                mask_threshold=args.sam3_mask_threshold,
                save_per_object=False,
                out_dir=sam3_out_skin,
                fps=float(args.fps),
                save_mp4=args.sam3_save_mp4,
                processing_device=sam3_preprocess_device,
            )
            _write_masks_from_sam3(os.path.join(sam3_out_skin, "mask"), masks_skin)
            _compute_iou_stats(masks_arm, masks_skin, "png")

    if args.sam_only:
        if use_in_memory:
            if not union_masks:
                raise RuntimeError("SAM3 did not return masks in memory mode.")
            masks = [_normalize_mask(mask) for mask in union_masks]
        else:
            masks = _load_masks_from_dir(masks_arm)

        if args.mask_coco_parquet:
            _write_coco_parquet(masks, args.mask_coco_parquet)

        if args.mask_lossless_mp4:
            frames_dir = args.mask_lossless_frames_dir
            if not frames_dir:
                out_dir = os.path.dirname(args.mask_lossless_mp4) or "."
                frames_dir = os.path.join(out_dir, "mask_frames")
            _write_mask_frames(masks, frames_dir)
            _write_lossless_mp4(frames_dir, args.fps, args.mask_lossless_mp4)

        return

    # 4) composite
    if use_in_memory:
        if not union_masks:
            raise RuntimeError("SAM3 did not return masks in memory mode.")
        if skin_frames_bgr is None:
            raise RuntimeError("Missing reskinned frames for compositing.")
        composite_frames = _composite_frames_in_memory(
            frames_bgr,
            skin_frames_bgr,
            union_masks,
            args.dilate,
            args.erode,
            args.feather,
        )
        _encode_video_from_frames(composite_frames, args.fps, args.out, args.skin)
    else:
        sys.argv = [
            "composite",
            "--orig-frames", frames_orig,
            "--skin-frames", frames_skin,
            "--masks", masks_arm,
            "--out-frames", out_frames,
            "--dilate", str(args.dilate),
            "--erode", str(args.erode),
            "--feather", str(args.feather),
            "--trim-to-min",
        ]
        composite.main()

        # 5) encode with audio from skin video
        sys.argv = [
            "encode_video",
            "--frames", out_frames,
            "--fps", str(args.fps),
            "--audio-from", args.skin,
            "--out", args.out,
        ]
        encode_video.main()

if __name__ == "__main__":
    main()
