import argparse
import json
import os
from typing import List, Optional

import cv2
import torch
from torch.nn import functional as F

from guided_interpolation import guided_non_uniform_interpolation, validate_time_steps


def _load_model(model_dir: str):
    try:
        try:
            from model.RIFE_HDv2 import Model

            model = Model()
            model.load_model(model_dir, -1)
            print("Loaded v2.x HD model.")
        except Exception:
            from train_log.RIFE_HDv3 import Model

            model = Model()
            model.load_model(model_dir, -1)
            print("Loaded v3.x HD model.")
    except Exception:
        from model.RIFE_HD import Model

        model = Model()
        model.load_model(model_dir, -1)
        print("Loaded v1.x HD model")
    if not hasattr(model, "version"):
        model.version = 0
    model.eval()
    model.device()
    return model


def _load_frames(paths: List[str], device: torch.device):
    frames = []
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Unable to read image: {path}")
        tensor = (torch.tensor(img.transpose(2, 0, 1)).to(device) / 255.0).unsqueeze(0)
        frames.append(tensor)
    return frames


def _pad_frames(frames: List[torch.Tensor]):
    _, _, h, w = frames[0].shape
    ph = ((h - 1) // 64 + 1) * 64
    pw = ((w - 1) // 64 + 1) * 64
    padding = (0, pw - w, 0, ph - h)
    padded = [F.pad(frame, padding) for frame in frames]
    return padded, h, w


def _parse_time_steps_json(time_steps_json: Optional[str]):
    if not time_steps_json:
        return None
    return json.loads(time_steps_json)


def _compute_segment_frames(frame_map: List[int]) -> List[int]:
    return [frame_map[i + 1] - frame_map[i] + 1 for i in range(len(frame_map) - 1)]


def main():
    parser = argparse.ArgumentParser(description="Guided non-uniform interpolation CLI")
    parser.add_argument("--imgs", nargs="+", required=True, help="Ordered input frame paths.")
    parser.add_argument("--frame_map", nargs="+", type=int, help="Original indices of sampled frames.")
    parser.add_argument(
        "--segment_frames", nargs="+", type=int, help="Output frame counts for each segment."
    )
    parser.add_argument(
        "--time_steps_json",
        type=str,
        default=None,
        help="JSON list of list[float] time steps, e.g. [[0.5],[0.1,0.2]].",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Scale for model inference.")
    parser.add_argument("--output", type=str, default="output_guided", help="Output directory.")
    parser.add_argument("--model", dest="modelDir", type=str, default="train_log")

    args = parser.parse_args()

    if args.segment_frames is None:
        if args.frame_map is None:
            raise ValueError("Either --frame_map or --segment_frames must be provided.")
        segment_frames = _compute_segment_frames(args.frame_map)
    else:
        segment_frames = args.segment_frames
    for idx, segment_count in enumerate(segment_frames):
        if segment_count < 2:
            raise ValueError(f"segment_frames[{idx}] must be >= 2.")

    time_steps = _parse_time_steps_json(args.time_steps_json)
    time_steps = validate_time_steps(time_steps, segment_frames)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    model = _load_model(args.modelDir)
    frames = _load_frames(args.imgs, device)
    padded_frames, h, w = _pad_frames(frames)

    output_frames = guided_non_uniform_interpolation(
        model=model,
        frames=padded_frames,
        frame_map=args.frame_map,
        segment_frames=args.segment_frames,
        scale=args.scale,
        time_steps=time_steps,
    )

    os.makedirs(args.output, exist_ok=True)
    for idx, frame in enumerate(output_frames):
        img = (frame[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        cv2.imwrite(os.path.join(args.output, f"{idx + 1:08d}.png"), img)


if __name__ == "__main__":
    main()
