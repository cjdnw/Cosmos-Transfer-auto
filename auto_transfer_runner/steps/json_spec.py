from __future__ import annotations

import json
from pathlib import Path


def write_json_spec_depth(
    json_path: Path,
    name: str,
    prompt_path: Path,
    video_path: Path,
    guidance: float,
    num_steps: int,
    control_vis_path: Path,
    control_weight: float,
) -> None:
    """Step4：生成 depth 格式的 json spec。"""
    json_path.parent.mkdir(parents=True, exist_ok=True)

    spec = {
        "name": name,
        "prompt_path": str(prompt_path),
        "video_path": str(video_path),
        "guidance": guidance,
        "num_steps": num_steps,
        "depth": {
            "control_path": str(control_vis_path),
            "control_weight": control_weight,
        },
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=4)
