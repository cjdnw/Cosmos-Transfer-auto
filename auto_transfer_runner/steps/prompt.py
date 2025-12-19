from __future__ import annotations

from pathlib import Path
from typing import Dict

import shutil

from ..utils.hashing import video_fingerprint
from ..utils.locking import FileLock, FileLockConfig
from ..utils.logging_utils import append_log, console_print
from ..utils.subprocess_utils import run_cmd, conda_run
from ..utils.text import sanitize_name


def generate_prompt(
    video_path: Path,
    out_prompt: Path,
    autoprompt_root: Path,
    conda_env: str,
    fps: float,
    size: str,
    variant: str,
    model_family: str,
    env: Dict[str, str],
    log_path: Path,
    reuse_if_exists: bool,
    work_dir: Path,
    fingerprint_mode: str,
) -> None:
    """
    Step2：调用 AutoPrompt 生成 prompt。

    你的原始命令：
      cd /mnt/data/ganrunhe/AutoPrompt
      python robot_video_to_prompt.py \
        --video <in> \
        --out <out> \
        --size 7B --variant instruct --fps 4.0
    """
    out_prompt.parent.mkdir(parents=True, exist_ok=True)

    fingerprint = video_fingerprint(video_path, mode=fingerprint_mode)
    cache_dir = work_dir / "prompt_cache" / fingerprint
    cache_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{sanitize_name(video_path.stem)}__{fingerprint}"
    cached_prompt = cache_dir / f"{base_name}.txt"
    lock_path = cache_dir / ".lock"
    lock_cfg = FileLockConfig()

    def cache_side_outputs(base: Path) -> Dict[str, Path]:
        root = base.with_suffix("")
        return {
            "base": cache_dir / f"{root.name}_base.txt",
            "bracketed": cache_dir / f"{root.name}_bracketed.txt",
            "replacements": cache_dir / f"{root.name}_replacements.json",
        }

    cached_paths = cache_side_outputs(cached_prompt)

    def copy_cached_if_ready() -> bool:
        if cached_prompt.exists() and cached_prompt.stat().st_size > 0:
            shutil.copy2(cached_prompt, out_prompt)
            append_log(log_path, f"[CACHE HIT] prompt: {cached_prompt}")
            console_print(f"[PROMPT][GPU {env.get('ATR_GPU_ID', '?')}] cache hit for {video_path.name}")
            return True
        return False

    if reuse_if_exists and copy_cached_if_ready():
        return

    if lock_path.exists():
        console_print(f"[PROMPT][GPU {env.get('ATR_GPU_ID', '?')}] waiting for prompt lock {lock_path.name}")
        append_log(log_path, f"[CACHE WAIT] prompt lock exists: {lock_path}")

    with FileLock(lock_path, lock_cfg):
        if reuse_if_exists and copy_cached_if_ready():
            return

        cmd = conda_run(
            conda_env,
            [
                "python",
                "robot_video_to_prompt.py",
                "--video",
                str(video_path),
                "--out",
                str(cached_prompt),
                "--model_family",
                str(model_family),
                "--size",
                str(size),
                "--variant",
                str(variant),
                "--fps",
                str(fps),
            ],
        )
        console_print(f"[PROMPT][GPU {env.get('ATR_GPU_ID', '?')}] generating prompt for {video_path.name}")
        run_cmd(cmd, cwd=autoprompt_root, env=env, log_path=log_path, title="STEP2: Generate prompt (AutoPrompt/Qwen)")

        # Cache side artifacts for reuse (if they were written)
        for path in cached_paths.values():
            if path.exists():
                continue
            maybe_path = cache_dir / path.name
            if maybe_path.exists():
                maybe_path.rename(path)

        shutil.copy2(cached_prompt, out_prompt)
