from __future__ import annotations

from pathlib import Path


def find_videos_root(video_path: Path) -> Path:
    """Locate the nearest parent directory named "videos" for a given video path."""
    for parent in [video_path.parent, *video_path.parents]:
        if parent.name == "videos":
            return parent
    raise ValueError(f"Cannot locate videos/ root for input video: {video_path}")


def parquet_path_for_video(video_path: Path) -> Path:
    """Map a video path to its matching parquet under the dataset data/ directory."""
    videos_root = find_videos_root(video_path)
    dataset_root = videos_root.parent
    relative = video_path.relative_to(videos_root)
    parts = relative.parts
    if len(parts) < 2:
        raise ValueError(
            "Expected video path under videos/<camera_subpath>/chunk-XXX/file-YYY.mp4, "
            f"got: {video_path}"
        )
    data_relative = Path(*parts[1:]).with_suffix(".parquet")
    return dataset_root / "data" / data_relative


def task_parquet_for_video(video_path: Path) -> Path:
    """Locate meta/task.parquet from the dataset root derived by videos/ parent."""
    videos_root = find_videos_root(video_path)
    dataset_root = videos_root.parent
    return dataset_root / "meta" / "task.parquet"
