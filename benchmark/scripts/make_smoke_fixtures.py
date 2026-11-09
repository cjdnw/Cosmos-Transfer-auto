#!/usr/bin/env python3
"""Generate SYNTHETIC smoke fixtures mimicking auto_transfer work_dir layout."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SMOKE_ROOT = ROOT / "benchmark" / "testdata" / "smoke"
WORK_DIR = SMOKE_ROOT / "work_dir"

N_VIDEOS = 8
N_FRAMES = 6
H, W = 48, 64


def _write_rgb_frames(dir_path: Path, frames: np.ndarray) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # type: ignore

        for t in range(frames.shape[0]):
            bgr = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(dir_path / f"frame_{t:04d}.png"), bgr)
    except ImportError:
        from PIL import Image

        for t in range(frames.shape[0]):
            Image.fromarray(frames[t]).save(dir_path / f"frame_{t:04d}.png")


def _write_map_frames(dir_path: Path, maps: np.ndarray, *, as_int: bool = False) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    for t in range(maps.shape[0]):
        np.save(dir_path / f"frame_{t:04d}.npy", maps[t].astype(np.int32 if as_int else np.float32))


def synthesize_clip(seed: int, quality: str) -> dict:
    """
    quality: high | mid | low | dup_of_0
    Returns dict with rgb, seg, edge, depth arrays.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(N_FRAMES)[:, None, None]

    # Base spatial pattern
    yy, xx = np.mgrid[0:H, 0:W]
    base = (xx[None] * 2 + yy[None] + t * 3).astype(np.float32)

    if quality == "high":
        noise = rng.randn(N_FRAMES, H, W).astype(np.float32) * 3.0
        flicker = 0.0
    elif quality == "mid":
        noise = rng.randn(N_FRAMES, H, W).astype(np.float32) * 12.0
        flicker = 8.0
    elif quality == "low":
        noise = rng.randn(N_FRAMES, H, W).astype(np.float32) * 40.0
        # Strong temporal flicker
        flicker = 40.0 * ((t % 2) * 2 - 1)
    else:  # dup — caller clones
        noise = rng.randn(N_FRAMES, H, W).astype(np.float32) * 3.0
        flicker = 0.0

    lum = np.clip(80 + 40 * np.sin(base / 15.0) + noise + flicker, 0, 255).astype(np.uint8)
    rgb = np.stack([lum, np.roll(lum, 1, axis=2), np.roll(lum, 2, axis=1)], axis=-1)

    # Seg: 4 coarse regions
    seg = ((xx[None] // (W // 2)) + 2 * (yy[None] // (H // 2))).astype(np.int32)
    seg = np.broadcast_to(seg, (N_FRAMES, H, W)).copy()
    if quality == "low":
        seg = (seg + rng.randint(0, 3, size=seg.shape)) % 4

    # Edge: gradient magnitude proxy
    edge = (np.abs(np.diff(lum.astype(np.float32), axis=2, prepend=lum[:, :, :1])) > 8).astype(np.float32)

    # Depth: smooth field
    depth = (50 + 0.5 * xx[None] + 0.3 * yy[None] + 0.1 * t).astype(np.float32)
    depth = np.broadcast_to(depth, (N_FRAMES, H, W)).copy()
    if quality == "low":
        depth = depth + rng.randn(*depth.shape).astype(np.float32) * 20

    return {"rgb": rgb, "seg": seg, "edge": edge, "depth": depth}


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    (SMOKE_ROOT / "README.md").write_text(
        "# SYNTHETIC smoke fixtures\n\n"
        "**These are NOT real Cosmos-Transfer2.5 outputs.**\n"
        "Tiny fake RGB frame dirs + fake control maps for CPU pipeline tests only.\n"
        "Do not cite numbers from this directory as experimental results.\n",
        encoding="utf-8",
    )

    qualities = ["high", "high", "mid", "mid", "low", "low", "dup_of_0", "mid"]
    manifest_rows = []
    first_rgb = None

    for i, q in enumerate(qualities):
        seed = 1000 + i
        if q == "dup_of_0":
            assert first_rgb is not None
            # Near-duplicate of video 0 with tiny noise
            clip = synthesize_clip(1000, "high")
            clip["rgb"] = np.clip(
                first_rgb.astype(np.int16)
                + np.random.RandomState(999).randint(-1, 2, size=first_rgb.shape),
                0,
                255,
            ).astype(np.uint8)
        else:
            clip = synthesize_clip(seed, q)
            if i == 0:
                first_rgb = clip["rgb"].copy()

        job_id = f"smokejob{i:02d}"
        variant_idx = i % 2
        stem = f"synth_ep{i:02d}"
        input_video = WORK_DIR / "input_videos" / f"{stem}.mp4"
        # Mark input as a frames dir for existence checks
        input_frames = WORK_DIR / "input_videos" / stem / "frames"
        # Input slightly different
        input_rgb = np.clip(clip["rgb"].astype(np.int16) - 5, 0, 255).astype(np.uint8)
        _write_rgb_frames(input_frames, input_rgb)
        # placeholder mp4 path string (frames used if present — we point input_video to frames parent)
        # For manifest compatibility store path to frames parent as "video"
        input_video_path = str(input_frames.parent)

        out_dir = WORK_DIR / "output_video" / f"{stem}_var_{variant_idx + 1}" / stem
        frames_dir = out_dir / "frames"
        _write_rgb_frames(frames_dir, clip["rgb"])

        # control_cache layout
        fp = f"fp_{stem}"
        control_paths = {}
        for ctype, arr, as_int in (
            ("seg", clip["seg"], True),
            ("edge", clip["edge"], False),
            ("depth", clip["depth"], False),
        ):
            cdir = WORK_DIR / "control_cache" / ctype / fp
            _write_map_frames(cdir, arr, as_int=as_int)
            control_paths[ctype] = str(cdir)
        control_paths["blur"] = None

        prompt_path = WORK_DIR / "prompt" / f"{stem}__{job_id}__prompt.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(f"SYNTHETIC prompt for {stem} quality={q}\n", encoding="utf-8")

        json_path = WORK_DIR / "json" / f"{stem}__{job_id}__multicontrol.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps({"name": stem, "seed": seed, "synthetic": True, "quality_label": q}, indent=2),
            encoding="utf-8",
        )

        row = {
            "status": "done",
            "input_video": input_video_path,
            "variant_idx": variant_idx,
            "job_id": job_id,
            "control_type": "multicontrol",
            "prompt_path": str(prompt_path),
            "json_path": str(json_path),
            "output_dir": str(out_dir),
            "arm_swap_output": None,
            "post_restore_output": None,
            "log_path": str(WORK_DIR / "logs" / f"{stem}__{job_id}.log"),
            "control_vis_path": None,
            "control_paths": control_paths,
            "_synthetic_quality_label": q,
        }
        (WORK_DIR / "logs").mkdir(parents=True, exist_ok=True)
        (WORK_DIR / "logs" / f"{stem}__{job_id}.log").write_text("SYNTHETIC log\n", encoding="utf-8")
        manifest_rows.append(row)

    # One failed row to test missing-field tagging
    manifest_rows.append(
        {
            "status": "failed",
            "input_video": str(WORK_DIR / "input_videos" / "missing.mp4"),
            "variant_idx": 0,
            "job_id": "smokejob_fail",
            "control_type": "multicontrol",
            "prompt_path": None,
            "json_path": None,
            "output_dir": str(WORK_DIR / "output_video" / "missing"),
            "arm_swap_output": None,
            "log_path": None,
            "error": "RuntimeError('SYNTHETIC failure')",
        }
    )

    manifest_path = WORK_DIR / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote SYNTHETIC fixtures under {WORK_DIR}")
    print(f"  videos: {N_VIDEOS} done + 1 failed")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
