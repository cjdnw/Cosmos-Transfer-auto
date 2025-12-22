from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List

from ..utils.cleanup import safe_unlink
from ..utils.hashing import video_fingerprint
from ..utils.locking import FileLock, FileLockConfig
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
    num_alts: int,
    min_replace_terms: int = 1,
    max_replace_terms: int = 3,
    fingerprint_mode: str = "stat",
    lock_timeout_sec: int = 6 * 3600,
    stale_lock_sec: int = 3 * 3600,
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

    fingerprint = video_fingerprint(video_path, mode=fingerprint_mode)
    cache_dir = work_dir / "prompt_cache" / fingerprint
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_path = cache_dir / "base.txt"
    bracket_path = cache_dir / "bracket.txt"
    repl_path = cache_dir / "repl.json"
    lock = FileLock(cache_dir / ".lock", FileLockConfig(timeout_sec=lock_timeout_sec, stale_lock_sec=stale_lock_sec))

    # Fast path: cached bracket+repl exist -> only apply
    if bracket_path.exists() and repl_path.exists():
        step3_apply_args = [
            "step3_apply",
            "--bracket_in",
            str(bracket_path),
            "--repl_json",
            str(repl_path),
            "--seed",
            str(seed),
            "--min_replace_terms",
            str(min_replace_terms),
            "--max_replace_terms",
            str(max_replace_terms),
            "--out_final",
            str(out_prompt),
        ]
        _run_prompt_step(
            env_name=env_name,
            autoprompt_root=autoprompt_root,
            args=step3_apply_args,
            env=env,
            log_path=log_path,
            title="STEP2: Apply cached prompt (step3_apply)",
        )
        append_log(log_path, f"[CACHE HIT] prompt cached at {cache_dir}")
        console_print(f"[CACHE HIT] prompt for {video_path.name}")
        return

    with lock:
        # Re-check inside lock
        if bracket_path.exists() and repl_path.exists():
            step3_apply_args = [
                "step3_apply",
                "--bracket_in",
                str(bracket_path),
                "--repl_json",
                str(repl_path),
                "--seed",
                str(seed),
                "--min_replace_terms",
                str(min_replace_terms),
                "--max_replace_terms",
                str(max_replace_terms),
                "--out_final",
                str(out_prompt),
            ]
            _run_prompt_step(
                env_name=env_name,
                autoprompt_root=autoprompt_root,
                args=step3_apply_args,
                env=env,
                log_path=log_path,
                title="STEP2: Apply cached prompt (step3_apply)",
            )
            append_log(log_path, f"[CACHE HIT AFTER LOCK] prompt cached at {cache_dir}")
            console_print(f"[CACHE HIT] prompt ready for {video_path.name}")
            return

        append_log(log_path, f"[CACHE MISS] prompt cache generating at {cache_dir}")
        console_print(f"[CACHE MISS] prompt for {video_path.name}")

        # Step1 base
        step1_args = [
            "step1_base",
            "--video",
            str(video_path),
            "--fps",
            str(fps),
            "--out_base",
            str(base_path),
            *_model_args(size, variant, model_type),
            "--cache_dir",
            str(cache_dir),
        ]
        _run_prompt_step(
            env_name=env_name,
            autoprompt_root=autoprompt_root,
            args=step1_args,
            env=env,
            log_path=log_path,
            title="STEP2.1: Generate base prompt (step1_base)",
        )

        # Step2 bracket
        step2_args = [
            "step2_bracket",
            "--base_in",
            str(base_path),
            "--out_bracket",
            str(bracket_path),
            *_model_args(size, variant, model_type),
        ]
        _run_prompt_step(
            env_name=env_name,
            autoprompt_root=autoprompt_root,
            args=step2_args,
            env=env,
            log_path=log_path,
            title="STEP2.2: Bracket prompt (step2_bracket)",
        )

        # Step3 replacements
        step3_args = [
            "step3_replacements",
            "--bracket_in",
            str(bracket_path),
            "--out_repl_json",
            str(repl_path),
            "--num_alts",
            str(num_alts),
            *_model_args(size, variant, model_type),
        ]
        _run_prompt_step(
            env_name=env_name,
            autoprompt_root=autoprompt_root,
            args=step3_args,
            env=env,
            log_path=log_path,
            title="STEP2.3: Generate replacements (step3_replacements)",
        )

    # Apply using cached files (outside lock)
    step3_apply_args = [
        "step3_apply",
        "--bracket_in",
        str(bracket_path),
        "--repl_json",
        str(repl_path),
        "--seed",
        str(seed),
        "--min_replace_terms",
        str(min_replace_terms),
        "--max_replace_terms",
        str(max_replace_terms),
        "--out_final",
        str(out_prompt),
    ]
    _run_prompt_step(
        env_name=env_name,
        autoprompt_root=autoprompt_root,
        args=step3_apply_args,
        env=env,
        log_path=log_path,
        title="STEP2.4: Apply replacements (step3_apply)",
    )

    append_log(log_path, f"[STEP2 DONE] prompt={out_prompt}")

    # No cleanup: bracket/repl stay in cache for future reuse
    safe_unlink(out_prompt, log_path=None) if False else None
