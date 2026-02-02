import argparse
import os
from robot_arm_swap.utils import ensure_dir, assert_ffmpeg, run_cmd

def extract(video_path: str, out_dir: str, fps: int, width: int, ext: str) -> None:
    ensure_dir(out_dir)
    vf = f"fps={fps},scale={width}:-2"
    out_pat = os.path.join(out_dir, f"%06d.{ext}")
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", vf]
    if ext.lower() in ("jpg", "jpeg"):
        cmd += ["-q:v", "2"]
    cmd += [out_pat]
    run_cmd(cmd)

def main():
    ap = argparse.ArgumentParser(description="Extract frames from original & reskinned videos with unified fps/scale.")
    ap.add_argument("--orig", required=True, help="Path to original.mp4")
    ap.add_argument("--skin", required=True, help="Path to reskinned.mp4")
    ap.add_argument("--out-orig", default="work/frames_orig", help="Output dir for original frames")
    ap.add_argument("--out-skin", default="work/frames_skin", help="Output dir for reskinned frames")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--ext", default="jpg", choices=["jpg", "jpeg", "png"], help="Frame image format")
    args = ap.parse_args()

    assert_ffmpeg()
    extract(args.orig, args.out_orig, args.fps, args.width, args.ext)
    extract(args.skin, args.out_skin, args.fps, args.width, args.ext)

if __name__ == "__main__":
    main()
