from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .text import sanitize_name


@dataclass(frozen=True)
class PathBundle:
    """一个 job 对应的所有输出路径（prompt/json/output/log）。"""
    prompt_path: Path
    control_dir: Path
    json_path: Path
    output_dir: Path
    log_path: Path


def make_job_paths(work_dir: Path, video_path: Path, job_id: str, control_type: str) -> PathBundle:
    """为单个视频 job 生成独立路径，确保并行不覆盖。"""
    prompt_dir = work_dir / "prompt"
    control_dir = work_dir / "control" / control_type
    json_dir = work_dir / "json"
    output_root = work_dir / "output_video"
    logs_dir = work_dir / "logs"

    stem = sanitize_name(video_path.stem)
    base = f"{stem}__{job_id}"

    return PathBundle(
        prompt_path=prompt_dir / f"{base}__prompt_3step.txt",
        control_dir=control_dir / base,  # 非缓存控制类型预留
        json_path=json_dir / f"{base}__{control_type}.json",
        output_dir=output_root / f"{base}__{control_type}",
        log_path=logs_dir / f"{base}.log",
    )
