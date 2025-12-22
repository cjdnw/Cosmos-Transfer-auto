from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List

from ..utils.cleanup import safe_unlink
from ..utils.subprocess_utils import run_cmd, conda_run
from ..utils.logging_utils import append_log, console_print


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


def _model_args(size: str, variant: str, model_type: str) -> List[str]:
    return ["--size", str(size), "--variant", str(variant), "--model_type", str(model_type)]


def _run_prompt_step(
    env_name: str,
    autoprompt_root: Path,
    args: List[str],
    env: Dict[str, str],
    log_path: Path,
    title: str,
) -> None:
    cmd = conda_run(env_name, ["python", "robot_video_to_prompt.py"] + args)
    run_cmd(cmd, cwd=autoprompt_root, env=env, log_path=log_path, title=title)


def generate_prompt(
    video_path: Path,
    out_prompt: Path,
    prompt_bracket_path: Path,
    prompt_replacements_path: Path,
    work_dir: Path,
    autoprompt_root: Path,
    conda_env: str,
    fps: float,
    size: str,
    variant: str,
    model_type: str,
    env: Dict[str, str],
    log_path: Path,
    reuse_if_exists: bool,
    seed: int,
    cache_dir: Path,
    num_alts: int,
    min_replace_terms: int = 1,
    max_replace_terms: int = 3,
) -> None:
    """
    Step2：调用 AutoPrompt 生成 prompt。

    等效命令示例（简化）：
      python robot_video_to_prompt.py step1_base --video <in> --fps 4 --out <base>
      python robot_video_to_prompt.py step2_bracket --base_in <base> --out <bracket>
      python robot_video_to_prompt.py step3_replacements --bracket_in <bracket> --out_json <repl.json>
      python robot_video_to_prompt.py step3_apply --bracket_in <bracket> --repl_json <repl.json> --seed <seed> --out <final>
    """
    out_prompt.parent.mkdir(parents=True, exist_ok=True)

    if reuse_if_exists and out_prompt.exists() and out_prompt.stat().st_size > 0:
        return

    env_name = conda_env
    fallback_env = "qwen25vl"
    if not _conda_env_exists(env_name):
        append_log(log_path, f"[WARN] Conda env not found: {env_name}. Falling back to {fallback_env}.")
        env_name = fallback_env

    prompt_cache_root = cache_dir
    prompt_cache_root.mkdir(parents=True, exist_ok=True)
    # Step1: build/reuse cache
    step1_args = [
        "step1_base",
        "--video",
        str(video_path),
        "--fps",
        str(fps),
        "--out_base",
        str(prompt_cache_root / "placeholder_base.txt"),  # will be overridden by cache dir logic
        "--cache_dir",
        str(prompt_cache_root),
        *_model_args(size, variant, model_type),
    ]
    _run_prompt_step(
        env_name=env_name,
        autoprompt_root=autoprompt_root,
        args=step1_args,
        env=env,
        log_path=log_path,
        title="STEP2.1: Generate/reuse base prompt (step1_base with cache)",
    )

    # Step23: combined bracket+replacements+apply, reusing cache when available
    step23_args = [
        "step23",
        "--base_in",
        str(prompt_cache_root / "placeholder_base.txt"),
        "--video",
        str(video_path),
        "--cache_dir",
        str(prompt_cache_root),
        "--seed",
        str(seed),
        "--out_final",
        str(out_prompt),
        "--out_bracket",
        str(prompt_bracket_path),
        "--out_repl_json",
        str(prompt_replacements_path),
        "--num_alts",
        str(num_alts),
        "--min_replace_terms",
        str(min_replace_terms),
        "--max_replace_terms",
        str(max_replace_terms),
        *_model_args(size, variant, model_type),
    ]

    _run_prompt_step(
        env_name=env_name,
        autoprompt_root=autoprompt_root,
        args=step23_args,
        env=env,
        log_path=log_path,
        title="STEP2.2: Bracket+replacements+apply (step23, cached)",
    )

    append_log(log_path, f"[STEP2 DONE] prompt={out_prompt}")

    # Cleanup intermediates immediately after use (bracket + replacements JSON)
    safe_unlink(prompt_bracket_path, log_path=log_path)
    safe_unlink(prompt_replacements_path, log_path=log_path)
