import os
import cv2
import numpy as np
from tqdm import tqdm

def _auto_out(input_mp4: str, suffix: str, output: str | None):
    if output is not None:
        return output
    base = os.path.splitext(os.path.basename(input_mp4))[0]
    return base + suffix + ".mp4"

def _open_writer(out_path: str, fps: float, w: int, h: int, is_color=True):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=is_color)

# =========================
# Gaussian blur (replaces bilateral)
# =========================
def gen_blur_mp4(input_mp4: str, output_mp4: str, sigma: float = 8.0):
    """
    Generate blurry video using Gaussian blur.
    - sigma controls blur strength (bigger => blurrier)
    - kernel size is auto-computed by OpenCV when ksize=(0,0)
    """
    cap = cv2.VideoCapture(input_mp4)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_mp4}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = _open_writer(output_mp4, fps, w, h, is_color=True)
    pbar = tqdm(total=n if n > 0 else None, desc="gaussian blur")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Gaussian blur: very visibly blurry compared to bilateral
        out = cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma)

        writer.write(out)
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    print("Saved:", output_mp4)

def gen_canny_mp4(input_mp4: str, output_mp4: str, low=100, high=200, to_3ch=True):
    cap = cv2.VideoCapture(input_mp4)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_mp4}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Write 3-channel mp4 for compatibility with downstream readers
    writer = _open_writer(output_mp4, fps, w, h, is_color=True)
    pbar = tqdm(total=n if n > 0 else None, desc="canny edge")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, low, high)

        if to_3ch:
            out = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        else:
            # Not recommended: single-channel video may be less compatible
            out = edge

        writer.write(out)
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    print("Saved:", output_mp4)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Generate blur/canny control videos (mp4)")
    parser.add_argument("--input", type=str, required=True, help="input .mp4")
    parser.add_argument("--mode", type=str, choices=["blur", "canny", "both"], required=True)

    # output: for blur/canny mode; for both, output-dir is recommended
    parser.add_argument("--output", type=str, default=None, help="output .mp4 (optional)")
    parser.add_argument("--output-dir", type=str, default=None, help="output directory for mode=both")

    # blur params (Gaussian)
    parser.add_argument("--blur-sigma", type=float, default=8.0, help="Gaussian blur sigma (bigger => blurrier)")

    # canny params
    parser.add_argument("--low", type=int, default=100, help="Canny low threshold")
    parser.add_argument("--high", type=int, default=200, help="Canny high threshold")

    args = parser.parse_args()

    if args.mode in ["blur", "canny"]:
        if args.mode == "blur":
            out = _auto_out(args.input, "_blur", args.output)
            gen_blur_mp4(args.input, out, sigma=args.blur_sigma)
        else:
            out = _auto_out(args.input, "_canny", args.output)
            gen_canny_mp4(args.input, out, low=args.low, high=args.high, to_3ch=True)

    else:  # both
        out_dir = args.output_dir or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.input))[0]

        out_blur = os.path.join(out_dir, base + "_blur.mp4")
        out_canny = os.path.join(out_dir, base + "_canny.mp4")

        gen_blur_mp4(args.input, out_blur, sigma=args.blur_sigma)
        gen_canny_mp4(args.input, out_canny, low=args.low, high=args.high, to_3ch=True)
