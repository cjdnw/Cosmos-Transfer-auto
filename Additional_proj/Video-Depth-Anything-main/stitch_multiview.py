#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def run_cmd(cmd: List[str]) -> None:
    """Run a command and raise with readable error output on failure."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"{' '.join(shlex.quote(c) for c in cmd)}\n\n"
            f"STDERR:\n{proc.stderr}\n"
            f"STDOUT:\n{proc.stdout}\n"
        )


def which_or_die(exe: str) -> None:
    from shutil import which
    if which(exe) is None:
        raise RuntimeError(
            f"Required executable '{exe}' not found in PATH. "
            f"Please install ffmpeg (which includes ffprobe) and try again."
        )


def ffprobe_dims(path: str) -> Tuple[int, int]:
    """Return (width, height) for the first video stream."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        path
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"ffprobe failed for: {path}\n{proc.stderr}")
    w_str, h_str = proc.stdout.strip().split("x")
    return int(w_str), int(h_str)


def parse_list_arg(values: List[str]) -> List[str]:
    """
    Allow:
      --videos a.mp4 b.mp4 c.mp4
    or:
      --videos "a.mp4,b.mp4,c.mp4"
    """
    out: List[str] = []
    for v in values:
        parts = [p.strip() for p in v.split(",") if p.strip()]
        out.extend(parts)
    return out


def ensure_files_exist(paths: List[str], label: str) -> None:
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        msg = "\n".join(missing)
        raise FileNotFoundError(f"Missing {label} file(s):\n{msg}")


def build_hstack_filter(n: int, target_h: int) -> str:
    """
    For each input i:
      [i:v]scale=-2:target_h:flags=lanczos,setsar=1[v{i}]
    Then:
      [v0][v1]...[v{n-1}]hstack=inputs=n[v]
    """
    parts = []
    for i in range(n):
        parts.append(f"[{i}:v]scale=-2:{target_h}:flags=lanczos,setsar=1[v{i}]")
    stack_inputs = "".join([f"[v{i}]" for i in range(n)])
    parts.append(f"{stack_inputs}hstack=inputs={n}[v]")
    return ";".join(parts)


def hstack_videos(
    inputs: List[str],
    output_path: str,
    keep_audio_from_first: bool = False,
    crf: int = 18,
    preset: str = "medium",
) -> None:
    # Determine target height = min height across inputs (safe, avoids upscaling)
    dims = [ffprobe_dims(p) for p in inputs]
    target_h = min(h for _, h in dims)

    n = len(inputs)
    vf = build_hstack_filter(n=n, target_h=target_h)

    cmd = ["ffmpeg", "-y"]
    for p in inputs:
        cmd += ["-i", p]

    cmd += ["-filter_complex", vf, "-map", "[v]"]

    if keep_audio_from_first:
        # Keep audio from first input only (common for multi-cam). If none exists, ffmpeg may error;
        # user can disable with default behavior.
        cmd += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]

    # Video encode
    cmd += [
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ]

    run_cmd(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Horizontally stitch multi-view RGB mp4 and depth mp4 into ultra-wide videos (width-wise hstack)."
    )
    parser.add_argument(
        "--videos", nargs="+", required=True,
        help="RGB video inputs. Provide as space-separated list or a single comma-separated string."
    )
    parser.add_argument(
        "--depths", nargs="+", required=True,
        help="Depth video inputs (same number/order as --videos). Provide as space-separated list or comma-separated."
    )
    parser.add_argument(
        "--outdir", required=True,
        help="Output directory."
    )
    parser.add_argument(
        "--name", default=None,
        help="Base output name (without extension). Default: stitched_<N>views"
    )
    parser.add_argument(
        "--keep-audio", action="store_true",
        help="Keep audio from the first RGB input only. Default: no audio in output."
    )
    parser.add_argument(
        "--crf", type=int, default=18,
        help="x264 CRF (lower is higher quality). Default: 18."
    )
    parser.add_argument(
        "--preset", type=str, default="medium",
        help="x264 preset. Default: medium. Options: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow."
    )

    args = parser.parse_args()

    which_or_die("ffmpeg")
    which_or_die("ffprobe")

    videos = parse_list_arg(args.videos)
    depths = parse_list_arg(args.depths)

    if len(videos) < 2:
        raise ValueError("Need at least 2 inputs for --videos to stitch horizontally.")
    if len(depths) < 2:
        raise ValueError("Need at least 2 inputs for --depths to stitch horizontally.")
    if len(videos) != len(depths):
        raise ValueError(f"--videos count ({len(videos)}) must match --depths count ({len(depths)}).")

    ensure_files_exist(videos, "video")
    ensure_files_exist(depths, "depth")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = args.name if args.name else f"stitched_{len(videos)}views"

    rgb_out = str(outdir / f"{base}_rgb.mp4")
    depth_out = str(outdir / f"{base}_depth.mp4")

    print(f"[1/2] Stitching RGB -> {rgb_out}")
    hstack_videos(
        inputs=videos,
        output_path=rgb_out,
        keep_audio_from_first=args.keep_audio,
        crf=args.crf,
        preset=args.preset,
    )

    print(f"[2/2] Stitching DEPTH -> {depth_out}")
    # Depth usually has no audio; force no audio regardless of --keep-audio
    hstack_videos(
        inputs=depths,
        output_path=depth_out,
        keep_audio_from_first=False,
        crf=args.crf,
        preset=args.preset,
    )

    print("Done.")
    print(f"Outputs:\n  {rgb_out}\n  {depth_out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
