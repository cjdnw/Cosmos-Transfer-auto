# sam3_video_tracker.py
# Minimal SAM3 Video Tracking wrapper + main entry (single-file runnable)
#
# Goal:
# - Input: video OR frames directory + first-frame bboxes (xyxy, pixel coords), typically from Qwen3-VL
# - Output: per-frame masks (union + label mask) and masked RGB frames (background=0)
#
# Usage:
#   python sam3_video_tracker.py
#
# Edit the "USER CONFIG" section at the bottom for your paths and parameters.

from __future__ import annotations

import os
import glob
from typing import List, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

import torch
import huggingface_hub
if not hasattr(huggingface_hub, "is_offline_mode"):
    def is_offline_mode() -> bool:
        return os.getenv("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}

    huggingface_hub.is_offline_mode = is_offline_mode

from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
from transformers.utils import logging as hf_logging
import inspect


def _patch_torchvision_normalize_inplace() -> None:
    """Force torchvision normalize to run in-place to reduce peak GPU memory.

    Transformers' fast image/video processors may normalize tensors out-of-place,
    briefly allocating a second full-size tensor (often the direct cause of CUDA OOM
    on long/high-res videos). For inference pipelines, in-place normalize is usually
    safe and materially lowers peak VRAM.
    """
    # Patch torchvision v2
    try:
        import torchvision.transforms.v2.functional as Fv2  # type: ignore
        if getattr(Fv2.normalize, "_sam3_inplace_patched", False):
            return
        _orig = Fv2.normalize

        def _normalize(*args, **kwargs):
            kwargs["inplace"] = True
            return _orig(*args, **kwargs)

        _normalize._sam3_inplace_patched = True  # type: ignore[attr-defined]
        Fv2.normalize = _normalize  # type: ignore[assignment]
    except Exception:
        pass

    # Patch torchvision v1 (fallback in some envs)
    try:
        import torchvision.transforms.functional as F1  # type: ignore
        if getattr(F1.normalize, "_sam3_inplace_patched", False):
            return
        _orig1 = F1.normalize

        def _normalize1(*args, **kwargs):
            kwargs["inplace"] = True
            return _orig1(*args, **kwargs)

        _normalize1._sam3_inplace_patched = True  # type: ignore[attr-defined]
        F1.normalize = _normalize1  # type: ignore[assignment]
    except Exception:
        pass



def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _load_frames_from_dir(image_dir: str) -> List[np.ndarray]:
    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.png"))
        + glob.glob(os.path.join(image_dir, "*.jpg"))
        + glob.glob(os.path.join(image_dir, "*.jpeg"))
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {image_dir}")

    frames = []
    for p in image_paths:
        img = cv2.imread(p)  # BGR
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {p}")
        frames.append(img)
    return frames


def _load_frames_from_video(video_path: str, max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], float]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 30.0

    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)  # BGR
        i += 1
        if max_frames is not None and i >= max_frames:
            break
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from: {video_path}")
    return frames, float(fps)


def _bgr_to_pil_rgb(img_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _resize_bgr(img_bgr: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return img_bgr
    h, w = img_bgr.shape[:2]
    return cv2.resize(img_bgr, (w // scale, h // scale), interpolation=cv2.INTER_AREA)


def _resize_mask_nearest(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def _apply_union_mask(image_bgr: np.ndarray, union_mask01: np.ndarray) -> np.ndarray:
    out = image_bgr.copy()
    out[union_mask01 == 0] = 0
    return out


def _to_uint_for_cv(mask_label: np.ndarray) -> np.ndarray:
    """
    cv2.imwrite supports uint8/uint16 for PNG.
    Use uint8 when possible, otherwise uint16.
    """
    m = int(mask_label.max()) if mask_label.size else 0
    if m <= 255:
        return mask_label.astype(np.uint8)
    return mask_label.astype(np.uint16)


class SAM3VideoTracker:
    """
    Minimal SAM3 video tracker wrapper (Transformers Sam3TrackerVideo).

    Key idea:
    - Initialize an inference session with all frames
    - Add prompts (boxes/points/masks) on the first frame
    - Propagate masks through the video via propagate_in_video_iterator()

    This wrapper focuses on first-frame bounding boxes (xyxy, pixel coords).
    """

    def __init__(self, model_name: str = "facebook/sam3", device: Optional[str] = None, dtype: Optional[torch.dtype] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if dtype is None:
            # HF docs use bf16 on GPU; keep fp32 on CPU.
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype

        print(f"[SAM3VideoTracker] Loading '{model_name}' on {self.device} (dtype={self.dtype}) ...")
        hf_logging.disable_progress_bar()
        self.model = Sam3TrackerVideoModel.from_pretrained(model_name).to(self.device, dtype=self.dtype)
        self.processor = Sam3TrackerVideoProcessor.from_pretrained(model_name)
        self.model.eval()
        print("[SAM3VideoTracker] Loaded.")

    @torch.inference_mode()
    def track_with_first_frame_boxes(
        self,
        frames_bgr: List[np.ndarray],
        first_frame_boxes_xyxy: List[List[int]],
        scale: int = 1,
        mask_threshold: float = 0.0,
        save_per_object: bool = False,
        out_dir: str = "work/sam3_video_out",
        fps: Optional[float] = None,
        save_mp4: bool = False,
        processing_device: Optional[Union[str, torch.device]] = None,
        patch_normalize_inplace: bool = True,
        return_union_masks: bool = False,
        write_outputs: bool = True,
    ) -> Optional[List[np.ndarray]]:
        """
        Track objects defined by first-frame bboxes through the whole video.

        Parameters
        ----------
        frames_bgr:
            List of frames in BGR (uint8), e.g. from cv2 (video or image dir).
        first_frame_boxes_xyxy:
            List of bboxes in pixel xyxy format on the ORIGINAL frames coordinate system.
        scale:
            Optional downscale factor before feeding frames to SAM3. Boxes will be scaled accordingly.
        mask_threshold:
            Threshold used by processor.post_process_masks when binarize=True.
        save_per_object:
            If True, also saves per-object binary masks per frame (mask_obj{obj_id}_{frame}.png).
        out_dir:
            Output root directory.
        fps:
            FPS used for output mp4 if save_mp4=True. If None, defaults to 30.
        save_mp4:
            If True, saves `mask_union.mp4` and `masked_rgb.mp4` in out_dir.
        """
        if not frames_bgr:
            raise ValueError("frames_bgr is empty")
        if not first_frame_boxes_xyxy:
            raise ValueError("first_frame_boxes_xyxy is empty")
        if scale < 1:
            raise ValueError("scale must be >= 1")

        if patch_normalize_inplace:
            _patch_torchvision_normalize_inplace()

        if processing_device is None:
            # Default: preprocess on the same device as inference
            processing_device = self.device
        if isinstance(processing_device, str):
            processing_device = torch.device(processing_device)

        # Validate consistent frame sizes
        h0, w0 = frames_bgr[0].shape[:2]
        for i, fr in enumerate(frames_bgr):
            if fr.shape[:2] != (h0, w0):
                raise ValueError(f"Frame size mismatch at idx {i}: {fr.shape[:2]} vs {(h0, w0)}")

        if write_outputs or save_mp4:
            _ensure_dir(out_dir)
        if write_outputs:
            mask_dir = os.path.join(out_dir, "mask")
            masked_dir = os.path.join(out_dir, "masked_rgb")
            _ensure_dir(mask_dir)
            _ensure_dir(masked_dir)
        else:
            mask_dir = None
            masked_dir = None

        # Scale frames (optional) before passing to SAM3 video session
        frames_s = [_resize_bgr(fr, scale) for fr in frames_bgr]
        hs, ws = frames_s[0].shape[:2]

        # Scale boxes to match scaled frames
        boxes_s = [[int(round(v / scale)) for v in box] for box in first_frame_boxes_xyxy]

        # Convert frames to PIL RGB for the processor
        video_frames = [_bgr_to_pil_rgb(fr) for fr in frames_s]

        # Init session
        # Different Transformers versions use different parameter names for the preprocessing device.
        init_kwargs = {
            "video": video_frames,
            "inference_device": self.device,
            "dtype": self.dtype,
        }
        try:
            sig = inspect.signature(self.processor.init_video_session)
            params = sig.parameters
            if "processing_device" in params:
                init_kwargs["processing_device"] = processing_device
            elif "device" in params:
                # Some versions call it "device" (used for preprocessing)
                init_kwargs["device"] = processing_device
        except Exception:
            # If introspection fails, proceed without preprocessing override
            pass

        inference_session = self.processor.init_video_session(**init_kwargs)

        # Add first-frame boxes as prompts
        ann_frame_idx = 0
        obj_ids = list(range(1, len(boxes_s) + 1))  # 1..N
        # Sam3TrackerVideo expects batched inputs: batch=1
        input_boxes = [boxes_s]  # shape: [ [box1, box2, ...] ]
        self.processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=obj_ids,
            input_boxes=input_boxes,
        )

        _ = self.model(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
        )

        # Optional: get segmentation on frame 0 immediately (not required)
        # outputs0 = self.model(inference_session=inference_session, frame_idx=0)

        # Video writers (optional)
        if save_mp4:
            if fps is None:
                fps = 30.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            mask_writer = cv2.VideoWriter(os.path.join(out_dir, "mask_union.mp4"), fourcc, float(fps), (w0, h0), isColor=False)
            rgb_writer = cv2.VideoWriter(os.path.join(out_dir, "masked_rgb.mp4"), fourcc, float(fps), (w0, h0), isColor=True)
        else:
            mask_writer = None
            rgb_writer = None

        # Propagate through the entire video
        union_masks = [] if return_union_masks else None
        for out in self.model.propagate_in_video_iterator(inference_session,start_frame_idx=ann_frame_idx):
            frame_idx = int(out.frame_idx)

            # Post-process to the session's (scaled) frame size
            # Returns: (num_objs, num_channels, H, W) where num_channels usually = 1
            masks_t = self.processor.post_process_masks(
                [out.pred_masks],
                original_sizes=[[inference_session.video_height, inference_session.video_width]],
                binarize=True,
                mask_threshold=float(mask_threshold),
            )[0]

            # -> (N, H, W) uint8 {0,1}
            if isinstance(masks_t, torch.Tensor):
                masks_np = masks_t.detach().cpu().numpy()
            else:
                masks_np = np.array(masks_t)

            if masks_np.ndim != 4:
                raise RuntimeError(f"Unexpected masks shape: {masks_np.shape}")
            masks_np = masks_np[:, 0, :, :].astype(np.uint8)

            # Build label mask on scaled space
            label_s = np.zeros((hs, ws), dtype=np.int32)
            for i, obj_id in enumerate(inference_session.obj_ids):
                m = masks_np[i]
                label_s[m > 0] = int(obj_id)  # "last write wins" if overlap

            # Resize to original size if needed
            if scale != 1:
                label = _resize_mask_nearest(label_s, w0, h0).astype(np.int32)
            else:
                label = label_s

            union01 = (label > 0).astype(np.uint8)
            union_mask = union01 * 255
            masked_rgb = _apply_union_mask(frames_bgr[frame_idx], union01)

            # Save per-frame outputs
            if return_union_masks:
                union_masks.append(union_mask)
            if write_outputs:
                cv2.imwrite(os.path.join(mask_dir, f"mask_label_{frame_idx:04d}.png"), _to_uint_for_cv(label))
                cv2.imwrite(os.path.join(mask_dir, f"mask_union_{frame_idx:04d}.png"), union_mask)
                cv2.imwrite(os.path.join(masked_dir, f"masked_rgb_{frame_idx:04d}.png"), masked_rgb)

            if save_per_object and write_outputs:
                for i, obj_id in enumerate(inference_session.obj_ids):
                    m_s = masks_np[i]
                    if scale != 1:
                        m = _resize_mask_nearest(m_s, w0, h0).astype(np.uint8)
                    else:
                        m = m_s
                    cv2.imwrite(os.path.join(mask_dir, f"mask_obj{int(obj_id)}_{frame_idx:04d}.png"), m * 255)

            if mask_writer is not None:
                mask_writer.write(union_mask.astype(np.uint8))
                rgb_writer.write(masked_rgb)

            if frame_idx % 10 == 0:
                print(f"[SAM3VideoTracker] processed frame {frame_idx}")

        if mask_writer is not None:
            mask_writer.release()
            rgb_writer.release()

        if write_outputs or save_mp4:
            print("[SAM3VideoTracker] Done. Saved to:", out_dir)
        else:
            print("[SAM3VideoTracker] Done.")

        return union_masks


if __name__ == "__main__":
    # =========================
    # USER CONFIG (edit here)
    # =========================
    # Input: choose ONE
    image_dir = ""          # e.g. "train/bottle/rgb" (png/jpg frames)
    video_path = "/mnt/data/ganrunhe/input_video/Robo_real.mp4"         # e.g. "/path/to/input.mp4"
    max_frames = None       # only used for video_path; set int to limit frames for quick test

    # Output
    out_dir = "work/sam3_video_out"

    # Optional downscale for speed (frames + bboxes will be scaled together)
    scale = 1  # =1 no scaling; =2 downscale by 2

    # First-frame bboxes (pixel xyxy on ORIGINAL frames)
    # Typically obtained from Qwen3-VL after converting 0..1000 normalized coords to pixels.
    qwen_results = [
        {"bbox_2d": [89, 287, 277, 480], "label": "robotic arm"},
        {"bbox_2d": [596, 258, 773, 480], "label": "robotic arm"},
    ]
    boxes_raw = [d["bbox_2d"] for d in qwen_results]

    # SAM3 params
    mask_threshold = 0.0
    save_per_object = False

    # Optional: also export mp4 videos (union mask + masked rgb)
    save_mp4 = True
    out_fps = 30.0  # only used when save_mp4=True and input is frames-dir

    # =========================
    # 1) Load frames
    # =========================
    if image_dir and video_path:
        raise ValueError("Set only one of image_dir or video_path")
    if (not image_dir) and (not video_path):
        raise ValueError("Please set image_dir OR video_path in USER CONFIG")

    if image_dir:
        frames_bgr = _load_frames_from_dir(image_dir)
        fps = out_fps
    else:
        frames_bgr, fps = _load_frames_from_video(video_path, max_frames=max_frames)

    if not boxes_raw:
        raise ValueError("qwen_results is empty. Please provide at least one bbox_2d (pixel xyxy).")

    # =========================
    # 2) Run SAM3 video tracking
    # =========================
    tracker = SAM3VideoTracker("facebook/sam3")
    tracker.track_with_first_frame_boxes(
        frames_bgr=frames_bgr,
        first_frame_boxes_xyxy=boxes_raw,
        scale=scale,
        mask_threshold=mask_threshold,
        save_per_object=save_per_object,
        out_dir=out_dir,
        fps=fps,
        save_mp4=save_mp4,
    )
