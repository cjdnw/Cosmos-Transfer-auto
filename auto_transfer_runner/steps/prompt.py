from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict

from ..utils.subprocess_utils import run_cmd, conda_run
from ..utils.logging_utils import append_log


def _conda_env_exists(env_name: str) -> bool:
    """
    Check whether a conda env exists by name. If conda is unavailable or the query fails,
    return True to avoid false negatives that would block execution.
    """
    try:
        p = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        if p.returncode != 0:
            return True
        data = json.loads(p.stdout or "{}")
        env_paths = data.get("envs", [])
        return any(Path(ep).name == env_name for ep in env_paths)
    except Exception:
        return True


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

    env_name = conda_env
    fallback_env = "qwen25vl"
    if not _conda_env_exists(env_name):
        append_log(log_path, f"[WARN] Conda env not found: {env_name}. Falling back to {fallback_env}.")
        env_name = fallback_env

    cmd = conda_run(
        env_name,
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
