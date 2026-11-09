"""I/O helpers for frames and control maps (CPU-friendly)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def _sorted_frame_files(directory: Path, patterns: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        files.extend(directory.glob(pat))
    files = [p for p in files if p.is_file()]
    files.sort(key=lambda p: p.name)
    return files


def load_rgb_sequence(path: PathLike, max_frames: Optional[int] = None) -> np.ndarray:
    """
    Load RGB frames as uint8 array (T, H, W, 3).

    Accepts:
      - directory of .png/.jpg/.npy frames
      - .mp4 / .avi via OpenCV (MISSING_DEP-style error if cv2 absent and path is video)
    """
    path = Path(path)
    if path.is_dir():
        files = _sorted_frame_files(path, ("*.png", "*.jpg", "*.jpeg", "*.npy"))
        if not files:
            raise FileNotFoundError(f"No frames in directory: {path}")
        if max_frames is not None:
            files = files[:max_frames]
        frames = []
        for f in files:
            if f.suffix.lower() == ".npy":
                arr = np.load(f)
            else:
                arr = _imread_rgb(f)
            frames.append(arr)
        return np.stack(frames, axis=0)

    if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        return _load_video_cv2(path, max_frames=max_frames)

    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim == 3:  # H,W,C single frame
            arr = arr[None, ...]
        return arr.astype(np.uint8)

    raise ValueError(f"Unsupported RGB path: {path}")


def load_map_sequence(
    path: PathLike,
    *,
    dtype: np.dtype = np.float32,
    max_frames: Optional[int] = None,
) -> np.ndarray:
    """
    Load a single-channel (or label) map sequence as (T, H, W).

    Accepts directory of .png/.npy, a single .npy (T,H,W), or video via OpenCV.
    """
    path = Path(path)
    if path.is_dir():
        files = _sorted_frame_files(path, ("*.png", "*.jpg", "*.jpeg", "*.npy"))
        if not files:
            raise FileNotFoundError(f"No map frames in directory: {path}")
        if max_frames is not None:
            files = files[:max_frames]
        maps = []
        for f in files:
            if f.suffix.lower() == ".npy":
                m = np.load(f)
            else:
                m = _imread_gray(f)
            if m.ndim == 3:
                m = m[..., 0]
            maps.append(m.astype(dtype))
        return np.stack(maps, axis=0)

    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(dtype)
        if arr.ndim == 2:
            arr = arr[None, ...]
        return arr

    if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        rgb = _load_video_cv2(path, max_frames=max_frames)
        # Take first channel / luminance
        gray = (
            0.299 * rgb[..., 0].astype(np.float32)
            + 0.587 * rgb[..., 1].astype(np.float32)
            + 0.114 * rgb[..., 2].astype(np.float32)
        )
        return gray.astype(dtype)

    raise ValueError(f"Unsupported map path: {path}")


def _imread_rgb(path: Path) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError as e:
        # Pillow fallback
        from PIL import Image

        img = np.asarray(Image.open(path).convert("RGB"))
        return img
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _imread_gray(path: Path) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError:
        from PIL import Image

        return np.asarray(Image.open(path).convert("L"))
    g = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if g is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    return g


def _load_video_cv2(path: Path, max_frames: Optional[int] = None) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError("MISSING_DEP: opencv (cv2) required to read video files") from e
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise FileNotFoundError(f"Empty video: {path}")
    return np.stack(frames, axis=0)


def align_lengths(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Truncate all arrays on axis 0 to the minimum length."""
    t = min(a.shape[0] for a in arrays)
    return tuple(a[:t] for a in arrays)
