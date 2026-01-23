from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict


def find_videos_root(video_path: Path) -> Path:
    """Locate the nearest parent directory named "videos" for a given video path."""
    for parent in [video_path.parent, *video_path.parents]:
        if parent.name == "videos":
            return parent
    raise ValueError(f"Cannot locate videos/ root for input video: {video_path}")


def find_dataset_root(video_path: Path) -> Path:
    """Locate the dataset root by using videos/ parent, with meta/info.json fallback."""
    try:
        videos_root = find_videos_root(video_path)
        return videos_root.parent
    except ValueError:
        for parent in [video_path.parent, *video_path.parents]:
            if (parent / "meta" / "info.json").exists():
                return parent
    raise ValueError(f"Cannot locate dataset root for input video: {video_path}")


def _load_info(dataset_root: Path) -> Dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_template(info: Dict[str, Any], key: str) -> str | None:
    if key in info:
        return info[key]
    if "paths" in info and isinstance(info["paths"], dict) and key in info["paths"]:
        return info["paths"][key]
    return None


def _template_to_regex(template: str) -> re.Pattern[str]:
    parts = []
    cursor = 0
    for match in re.finditer(r"{([a-zA-Z0-9_]+)(:[^}]+)?}", template):
        start, end = match.span()
        if start > cursor:
            parts.append(re.escape(template[cursor:start]))
        name = match.group(1)
        fmt = match.group(2) or ""
        if "d" in fmt:
            pattern = r"(?P<%s>\\d+)" % name
        else:
            pattern = r"(?P<%s>[^/]+)" % name
        parts.append(pattern)
        cursor = end
    if cursor < len(template):
        parts.append(re.escape(template[cursor:]))
    return re.compile("^" + "".join(parts) + "$")


def _parse_template_values(template: str, relative_path: str) -> Dict[str, object]:
    regex = _template_to_regex(template)
    match = regex.match(relative_path)
    if match is None:
        raise ValueError(
            f"Video path '{relative_path}' does not match template '{template}'"
        )
    values: Dict[str, object] = match.groupdict()
    for match in re.finditer(r"{([a-zA-Z0-9_]+)(:[^}]+)?}", template):
        name = match.group(1)
        fmt = match.group(2) or ""
        if "d" in fmt and name in values:
            values[name] = int(values[name])
    return values


def parquet_path_for_video(video_path: Path, *, dataset_root: Path | None = None) -> Path:
    """Map a video path to its matching parquet under the dataset data/ directory."""
    dataset_root = dataset_root or find_dataset_root(video_path)
    info = _load_info(dataset_root)
    video_template = _get_template(info, "video_path")
    data_template = _get_template(info, "data_path")
    if video_template and data_template:
        relative_path = video_path.relative_to(dataset_root).as_posix()
        try:
            values = _parse_template_values(video_template, relative_path)
        except ValueError:
            values = None
        if values is not None:
            data_relative = data_template.format(**values)
            return dataset_root / data_relative

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
    """Locate meta/tasks.parquet (or task.parquet) from the dataset root."""
    dataset_root = find_dataset_root(video_path)
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if tasks_path.exists():
        return tasks_path
    return dataset_root / "meta" / "task.parquet"
