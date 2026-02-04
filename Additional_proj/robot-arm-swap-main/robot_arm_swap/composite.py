import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

from robot_arm_swap.utils import ensure_dir, sorted_glob, read_image, read_mask

def main():
    ap = argparse.ArgumentParser(description="Composite original arm region onto reskinned frames using masks.")
    ap.add_argument("--orig-frames", required=True, help="Dir of original frames (jpg/png)")
    ap.add_argument("--skin-frames", required=True, help="Dir of reskinned frames (jpg/png)")
    ap.add_argument("--masks", required=True, help="Dir of mask pngs (000001.png ...)")
    ap.add_argument("--out-frames", default="work/out_frames", help="Output dir for composited frames (png)")
    ap.add_argument("--dilate", type=int, default=2, help="Mask dilation iterations")
    ap.add_argument("--erode", type=int, default=0, help="Mask erosion iterations")
    ap.add_argument("--feather", type=int, default=9, help="Gaussian blur kernel size (odd), 0 disables")
    ap.add_argument("--trim-to-min", action="store_true", help="If frame counts mismatch, trim to the min length")
    args = ap.parse_args()

    ensure_dir(args.out_frames)

    orig_paths = sorted_glob(os.path.join(args.orig_frames, "*.*"))
    skin_paths = sorted_glob(os.path.join(args.skin_frames, "*.*"))
    mask_paths = sorted_glob(os.path.join(args.masks, "*.png"))

    if not (len(orig_paths) and len(skin_paths) and len(mask_paths)):
        raise RuntimeError("Empty inputs: check your frame/mask directories.")

    n = min(len(orig_paths), len(skin_paths), len(mask_paths))
    if (len(orig_paths) != len(skin_paths) or len(orig_paths) != len(mask_paths)) and not args.trim_to_min:
        raise RuntimeError(
            f"Frame count mismatch: orig={len(orig_paths)}, skin={len(skin_paths)}, masks={len(mask_paths)}. "
            "Use --trim-to-min to proceed or align videos first."
        )

    kernel = np.ones((3, 3), np.uint8)
    feather = args.feather if args.feather and args.feather % 2 == 1 else (args.feather + 1 if args.feather else 0)

    for i in tqdm(range(n), desc="composite", unit="frame"):
        orig = read_image(orig_paths[i])
        skin = read_image(skin_paths[i])
        m = read_mask(mask_paths[i])

        if args.dilate > 0:
            m = cv2.dilate(m, kernel, iterations=args.dilate)
        if args.erode > 0:
            m = cv2.erode(m, kernel, iterations=args.erode)

        alpha = (m.astype(np.float32) / 255.0)
        if feather and feather > 0:
            alpha = cv2.GaussianBlur(alpha, (feather, feather), 0)
            alpha = np.clip(alpha, 0.0, 1.0)
        alpha = alpha[..., None]
        out = (orig.astype(np.float32) * alpha + skin.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

        cv2.imwrite(os.path.join(args.out_frames, f"{i+1:06d}.png"), out)

    print("Composited frames saved to:", args.out_frames)

if __name__ == "__main__":
    main()
