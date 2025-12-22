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


def ensure_cached_base_prompt(
    *,
    video_path: Path,
    cache_dir: Path,
    env_name: str,
    autoprompt_root: Path,
    fps: float,
    size: str,
    variant: str,
    model_type: str,
    env: Dict[str, str],
    log_path: Path,
    use_cache: bool,
    lock_timeout_sec: int,
    stale_lock_sec: int,
) -> Path:
    """Generate or reuse the base prompt for a video (step1), keyed by fingerprint."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_prompt = cache_dir / "base.txt"
    lock = FileLock(cache_dir / ".lock", FileLockConfig(timeout_sec=lock_timeout_sec, stale_lock_sec=stale_lock_sec))

    if use_cache and base_prompt.exists() and base_prompt.stat().st_size > 0:
        append_log(log_path, f"[PROMPT CACHE HIT] {base_prompt}")
        console_print(f"[PROMPT CACHE HIT] {video_path.name}")
        return base_prompt

    with lock:
        if use_cache and base_prompt.exists() and base_prompt.stat().st_size > 0:
            append_log(log_path, f"[PROMPT CACHE HIT AFTER LOCK] {base_prompt}")
            console_print(f"[PROMPT CACHE HIT] {video_path.name}")
            return base_prompt

        console_print(f"[PROMPT CACHE MISS] generating base prompt for {video_path.name}")
        append_log(log_path, f"[PROMPT CACHE MISS] {base_prompt}")

        args = [
            "step1_base",
            "--video",
            str(video_path),
            "--fps",
            str(fps),
            "--out",
            str(base_prompt),
            *_model_args(size, variant, model_type),
        ]
        _run_prompt_step(
            env_name=env_name,
            autoprompt_root=autoprompt_root,
            args=args,
            env=env,
            log_path=log_path,
            title="STEP2.1: Generate base prompt (step1_base)",
        )

        if not base_prompt.exists() or base_prompt.stat().st_size == 0:
            raise FileNotFoundError(f"Base prompt not found after generation: {base_prompt}")

        append_log(log_path, f"[PROMPT CACHE GENERATED] {base_prompt}")
        console_print(f"[PROMPT CACHE GENERATED] {video_path.name}")
        return base_prompt


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
    enable_cache: bool,
    force_regen: bool,
    fingerprint_mode: str,
    lock_timeout_sec: int,
    stale_lock_sec: int,
    seed: int,
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

    fingerprint = video_fingerprint(video_path, mode=fingerprint_mode)
    cache_dir = work_dir / "prompt_cache" / fingerprint
    use_cache = enable_cache and (not force_regen)

    base_prompt_path = ensure_cached_base_prompt(
        video_path=video_path,
        cache_dir=cache_dir,
        env_name=env_name,
        autoprompt_root=autoprompt_root,
        fps=fps,
        size=size,
        variant=variant,
        model_type=model_type,
        env=env,
        log_path=log_path,
        use_cache=use_cache,
        lock_timeout_sec=lock_timeout_sec,
        stale_lock_sec=stale_lock_sec,
    )

    step2_args = [
        "step2_bracket",
        "--base_in",
        str(base_prompt_path),
        "--out",
        str(prompt_bracket_path),
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

    step3a_args = [
        "step3_replacements",
        "--bracket_in",
        str(prompt_bracket_path),
        "--out_json",
        str(prompt_replacements_path),
        *_model_args(size, variant, model_type),
    ]
    _run_prompt_step(
        env_name=env_name,
        autoprompt_root=autoprompt_root,
        args=step3a_args,
        env=env,
        log_path=log_path,
        title="STEP2.3: Generate replacements (step3_replacements)",
    )

    step3b_args = [
        "step3_apply",
        "--bracket_in",
        str(prompt_bracket_path),
        "--repl_json",
        str(prompt_replacements_path),
        "--seed",
        str(seed),
        "--min_replace_terms",
        str(min_replace_terms),
        "--max_replace_terms",
        str(max_replace_terms),
        "--out",
        str(out_prompt),
    ]
    _run_prompt_step(
        env_name=env_name,
        autoprompt_root=autoprompt_root,
        args=step3b_args,
        env=env,
        log_path=log_path,
        title="STEP2.4: Apply replacements (step3_apply)",
    )

    append_log(log_path, f"[STEP2 DONE] prompt={out_prompt}")

    # Cleanup intermediates immediately after use (bracket + replacements JSON)
    safe_unlink(prompt_bracket_path, log_path=log_path)
    safe_unlink(prompt_replacements_path, log_path=log_path)
