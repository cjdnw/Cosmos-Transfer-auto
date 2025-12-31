from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .text import sanitize_name


@dataclass(frozen=True)
class PathBundle:
    """一个 job 对应的所有输出路径（prompt/json/output/log）。"""
    prompt_path: Path
    prompt_bracket_path: Path
    prompt_replacements_path: Path
    control_dir: Path
    json_path: Path
    output_dir: Path
    log_path: Path


def _variant_folder_name(input_root: Path, variant_idx: int) -> str:
    return f"{input_root.name}_var_{variant_idx + 1}"


def _relative_parent(input_root: Path, video_path: Path) -> Path:
    relative = video_path.relative_to(input_root)
    if relative.parent == Path("."):
        return Path()
    return relative.parent


def make_output_dir(
    output_root: Path,
    *,
    input_root: Path,
    video_path: Path,
    variant_idx: int,
) -> Path:
    variant_folder = _variant_folder_name(input_root, variant_idx)
    relative_parent = _relative_parent(input_root, video_path)
    return output_root / variant_folder / relative_parent / video_path.stem


def make_job_paths(
    work_dir: Path,
    video_path: Path,
    job_id: str,
    control_type: str,
    *,
    input_root: Path,
    variant_idx: int,
) -> PathBundle:
    """为单个视频 job 生成独立路径，确保并行不覆盖。"""
    prompt_dir = work_dir / "prompt"
    control_dir = work_dir / "control" / control_type
    json_dir = work_dir / "json"
    output_root = work_dir / "output_video"
    logs_dir = work_dir / "logs"

    stem = sanitize_name(video_path.stem)
    base = f"{stem}__{job_id}"

    output_dir = make_output_dir(
        output_root,
        input_root=input_root,
        video_path=video_path,
        variant_idx=variant_idx,
    )

    return PathBundle(
        prompt_path=prompt_dir / f"{base}__prompt.txt",
        prompt_bracket_path=prompt_dir / f"{base}__prompt_bracket.txt",
        prompt_replacements_path=prompt_dir / f"{base}__prompt_replacements.json",
        control_dir=control_dir / base,  # 非缓存控制类型预留
        json_path=json_dir / f"{base}__{control_type}.json",
        output_dir=output_dir,
        log_path=logs_dir / f"{base}.log",
    )
