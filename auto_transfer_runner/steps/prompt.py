from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Dict

from ..utils.hashing import short_hash, video_fingerprint
from ..utils.logging_utils import append_log, console_print
from ..utils.subprocess_utils import conda_run, run_cmd
from ..utils.text import sanitize_name


def _with_pythonpath(env: Dict[str, str], autoprompt_root: Path) -> Dict[str, str]:
    merged = env.copy()
    root_str = str(autoprompt_root)
    existing = merged.get("PYTHONPATH", "")
    merged["PYTHONPATH"] = root_str if not existing else f"{root_str}:{existing}"
    return merged


def _run_prompt_step1(
    video_path: Path,
    base_out: Path,
    autoprompt_root: Path,
    conda_env: str,
    fps: float,
    size: str,
    variant: str,
    model_family: str,
    env: Dict[str, str],
    log_path: Path,
) -> None:
    cmd = conda_run(
        conda_env,
        [
            "python",
            "-m",
            "auto_transfer_runner.steps.prompt_worker",
            "step1",
            "--video",
            str(video_path),
            "--base_out",
            str(base_out),
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
    run_cmd(cmd, cwd=autoprompt_root, env=_with_pythonpath(env, autoprompt_root), log_path=log_path,
            title="STEP2: Prompt step1 (video -> base)")


def _run_prompt_step23(
    base_in: Path,
    bracket_out: Path,
    replacements_out: Path,
    final_out: Path,
    autoprompt_root: Path,
    conda_env: str,
    size: str,
    variant: str,
    model_family: str,
    seed: int,
    reuse_bracket: bool,
    env: Dict[str, str],
    log_path: Path,
) -> None:
    cmd = conda_run(
        conda_env,
        [
            "python",
            "-m",
            "auto_transfer_runner.steps.prompt_worker",
            "step23",
            "--base_in",
            str(base_in),
            "--bracket_out",
            str(bracket_out),
            "--replacements_out",
            str(replacements_out),
            "--final_out",
            str(final_out),
            "--model_family",
            str(model_family),
            "--size",
            str(size),
            "--variant",
            str(variant),
            "--seed",
            str(seed),
        ],
    )
    if reuse_bracket and bracket_out.exists():
        cmd.append("--reuse_bracket")
    run_cmd(cmd, cwd=autoprompt_root, env=_with_pythonpath(env, autoprompt_root), log_path=log_path,
            title="STEP2: Prompt step2+3 (text refinement)")


def _ensure_step1_base(
    fingerprint: str,
    video_path: Path,
    base_out: Path,
    autoprompt_root: Path,
    conda_env: str,
    fps: float,
    size: str,
    variant: str,
    model_family: str,
    env: Dict[str, str],
    log_path: Path,
    step1_status,
    step1_paths,
    job_token: str,
) -> Path:
    """
    Guarantee that step1 (video -> base prompt) runs only once per video fingerprint.
    step1_status/step1_paths are multiprocessing.Manager dictionaries shared by workers.
    """
    base_out.parent.mkdir(parents=True, exist_ok=True)

    owner_token = f"{job_token}:{time.time_ns()}"
    gpu_tag = env.get("ATR_GPU_ID", "?")

    while True:
        state = step1_status.get(fingerprint)
        if state == "ready":
            cached = step1_paths.get(fingerprint)
            if cached:
                cached_path = Path(cached)
                if cached_path.exists() and cached_path.stat().st_size > 0:
                    return cached_path
            step1_status.pop(fingerprint, None)
            step1_paths.pop(fingerprint, None)
            continue

        if state and state.startswith("running"):
            if base_out.exists() and base_out.stat().st_size > 0:
                step1_paths[fingerprint] = str(base_out)
                step1_status[fingerprint] = "ready"
                return base_out
            console_print(f"[PROMPT][GPU {gpu_tag}] waiting for step1 of {video_path.name} (owner={state})")
            append_log(log_path, f"[PROMPT WAIT] step1 running elsewhere ({state})")
            time.sleep(2)
            continue

        if state == "failed":
            step1_status.pop(fingerprint, None)
            step1_paths.pop(fingerprint, None)
            continue

        existing = step1_status.setdefault(fingerprint, f"running:{owner_token}")
        if existing == f"running:{owner_token}":
            console_print(f"[PROMPT][GPU {gpu_tag}] step1 start for {video_path.name}")
            try:
                _run_prompt_step1(
                    video_path=video_path,
                    base_out=base_out,
                    autoprompt_root=autoprompt_root,
                    conda_env=conda_env,
                    fps=fps,
                    size=size,
                    variant=variant,
                    model_family=model_family,
                    env=env,
                    log_path=log_path,
                )
                step1_paths[fingerprint] = str(base_out)
                step1_status[fingerprint] = "ready"
                return base_out
            except Exception as exc:
                step1_status[fingerprint] = "failed"
                append_log(log_path, f"[PROMPT FAIL] step1 error: {repr(exc)}")
                raise
        else:
            # Another worker claimed between get/setdefault
            time.sleep(1)


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
    variant_idx: int,
    job_token: str,
    step1_status,
    step1_paths,
) -> None:
    """
    Step2：生成三步式 prompt。

    - step1：只读取视频，生成 base prompt（单视频仅跑一次，多卡等待完成）。
    - step2/3：基于 base prompt 文本生成最终 prompt（可按 variant seed 重复）。
    """
    out_prompt.parent.mkdir(parents=True, exist_ok=True)

    fingerprint = video_fingerprint(video_path, mode=fingerprint_mode)
    cache_dir = work_dir / "prompt_cache" / fingerprint
    cache_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{sanitize_name(video_path.stem)}__{fingerprint}"
    base_prompt_path = cache_dir / f"{base_name}__step1_base.txt"
    bracket_path = cache_dir / f"{base_name}__step2_bracketed.txt"
    replacements_path = cache_dir / f"{base_name}__v{variant_idx:03d}_replacements.json"
    final_prompt_path = cache_dir / f"{base_name}__v{variant_idx:03d}.txt"

    def copy_cached_if_ready() -> bool:
        if final_prompt_path.exists() and final_prompt_path.stat().st_size > 0:
            shutil.copy2(final_prompt_path, out_prompt)
            append_log(log_path, f"[CACHE HIT] prompt: {final_prompt_path}")
            console_print(f"[PROMPT] cache hit for {video_path.name} (variant {variant_idx})")
            return True
        return False

    if reuse_if_exists and copy_cached_if_ready():
        return

    seed = int(short_hash(f"{job_token}__v{variant_idx}", 8), 16)

    # Step1: run once per fingerprint
    ensured_base = _ensure_step1_base(
        fingerprint=fingerprint,
        video_path=video_path,
        base_out=base_prompt_path,
        autoprompt_root=autoprompt_root,
        conda_env=conda_env,
        fps=fps,
        size=size,
        variant=variant,
        model_family=model_family,
        env=env,
        log_path=log_path,
        step1_status=step1_status,
        step1_paths=step1_paths,
        job_token=job_token,
    )

    # Step2+3: always tied to variant seed
    console_print(f"[PROMPT][GPU {env.get('ATR_GPU_ID', '?')}] step2/3 start for {video_path.name} (variant {variant_idx})")
    _run_prompt_step23(
        base_in=ensured_base,
        bracket_out=bracket_path,
        replacements_out=replacements_path,
        final_out=final_prompt_path,
        autoprompt_root=autoprompt_root,
        conda_env=conda_env,
        size=size,
        variant=variant,
        model_family=model_family,
        seed=seed,
        reuse_bracket=reuse_if_exists,
        env=env,
        log_path=log_path,
    )

    shutil.copy2(final_prompt_path, out_prompt)
    append_log(log_path, f"[PROMPT DONE] {out_prompt}")
    console_print(f"[PROMPT] generated for {video_path.name} (variant {variant_idx})")
