from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..utils.subprocess_utils import run_cmd, conda_run


def generate_prompt(
    video_path: Path,
    out_prompt: Path,
    autoprompt_root: Path,
    conda_env: str,
    fps: float,
    size: str,
    variant: str,
    model_type: str,
    env: Dict[str, str],
    log_path: Path,
    reuse_if_exists: bool,
) -> None:
    """
    Step2：调用 AutoPrompt 生成 prompt。

    等效命令示例：
      cd /mnt/data/ganrunhe/AutoPrompt
      python robot_video_to_prompt.py \
        --video <in> \
        --out <out> \
        --size 7B --variant instruct --model_type Qwen2.5 --fps 4.0
    """
    out_prompt.parent.mkdir(parents=True, exist_ok=True)

    if reuse_if_exists and out_prompt.exists() and out_prompt.stat().st_size > 0:
        return

    cmd = conda_run(
        conda_env,
        [
            "python",
            "robot_video_to_prompt.py",
            "--video",
            str(video_path),
            "--out",
            str(out_prompt),
            "--size",
            str(size),
            "--variant",
            str(variant),
            "--model_type",
            str(model_type),
            "--fps",
            str(fps),
        ],
    )
    run_cmd(cmd, cwd=autoprompt_root, env=env, log_path=log_path, title="STEP2: Generate prompt (AutoPrompt/Qwen)")
