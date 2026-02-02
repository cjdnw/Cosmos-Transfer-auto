import argparse
import os
from robot_arm_swap.utils import assert_ffmpeg, run_cmd

def main():
    ap = argparse.ArgumentParser(description="Encode frames to mp4, optionally mux audio from a source video.")
    ap.add_argument("--frames", required=True, help="Dir of frames (e.g., work/out_frames/%06d.png)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="output_swap.mp4")
    ap.add_argument("--audio-from", default=None, help="Video file to copy audio from (e.g., reskinned.mp4)")
    args = ap.parse_args()

    assert_ffmpeg()

    frame_pat = os.path.join(args.frames, "%06d.png")
    cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", frame_pat]

    if args.audio_from:
        cmd += ["-i", args.audio_from, "-map", "0:v:0", "-map", "1:a?"]

    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]

    if args.audio_from:
        cmd += ["-c:a", "copy", "-shortest"]

    cmd += [args.out]
    run_cmd(cmd)

    print("Video saved to:", args.out)

if __name__ == "__main__":
    main()
