from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .config import GlobalConfig
from .steps.prompt import generate_prompt
from .steps.control import ControlType, generate_control
from .steps.json_spec import write_json_spec_depth
from .steps.inference import run_transfer_inference
from .utils.logging_utils import append_log, console_print
from .utils.paths import make_job_paths
from .utils.text import sanitize_name
from .utils.manifest import append_manifest
from .utils.cleanup import cleanup_job_intermediates


@dataclass(frozen=True)
class Job:
    """One video task. job_id makes all outputs unique across batch + multi-GPU runs."""
    video_path: Path
    job_id: str
    variant_idx: int = 0


def run_job(job: Job, cfg: GlobalConfig, env: Dict[str, str]) -> None:
    """Run the full pipeline for a single video: Step2 -> Step5."""
    repos = cfg.repos
    paths = make_job_paths(repos.work_dir, job.video_path, job.job_id, cfg.run.control_type)

    # Log header
    append_log(paths.log_path, "#" * 90)
    append_log(paths.log_path, f"JOB_ID={job.job_id}")
    append_log(paths.log_path, f"VARIANT_IDX={job.variant_idx}")
    append_log(paths.log_path, f"VIDEO={job.video_path}")
    append_log(paths.log_path, "#" * 90)

    console_print(f"[JOB START] {job.video_path.name}  variant={job.variant_idx}  job_id={job.job_id}")

    try:
        # Step2: prompt
        generate_prompt(
            video_path=job.video_path,
            out_prompt=paths.prompt_path,
            autoprompt_root=repos.autoprompt_root,
            conda_env=cfg.prompt.conda_env,
            model_family=cfg.prompt.model_family,
            fps=cfg.prompt.fps,
            size=cfg.prompt.size,
            variant=cfg.prompt.variant,
            env=env,
            log_path=paths.log_path,
            reuse_if_exists=cfg.prompt.reuse_if_exists,
        )

        # Step3: control
        control_type = ControlType(cfg.run.control_type)
        control_vis = generate_control(
            control_type=control_type,
            video_path=job.video_path,
            work_dir=repos.work_dir,
            depth_root=repos.depth_root,
            depth_conda_env=cfg.depth.conda_env,
            encoder=cfg.depth.encoder,
            grayscale=cfg.depth.grayscale,
            enable_cache=cfg.cache.enable,
            force_regen=cfg.cache.force_regen,
            fingerprint_mode=cfg.cache.fingerprint_mode,
            lock_timeout_sec=cfg.cache.lock_timeout_sec,
            stale_lock_sec=cfg.cache.stale_lock_sec,
            non_cached_output_dir=paths.control_dir,
            env=env,
            log_path=paths.log_path,
        )

        # Step4: json spec (depth only)
        name = f"{sanitize_name(job.video_path.stem)}__{job.job_id}__{cfg.run.control_type}"
        if control_type == ControlType.DEPTH:
            write_json_spec_depth(
                json_path=paths.json_path,
                name=name,
                prompt_path=paths.prompt_path,
                video_path=job.video_path,
                guidance=cfg.transfer.guidance,
                num_steps=cfg.transfer.num_steps,
                control_vis_path=control_vis,
                control_weight=cfg.transfer.control_weight,
            )
        else:
            raise NotImplementedError("Only depth json spec is implemented for now.")

        append_log(paths.log_path, f"[STEP4] Wrote json: {paths.json_path}")
        console_print(f"[STEP4] json={paths.json_path.name}")

        # Step5: inference
        run_transfer_inference(
            transfer_root=repos.transfer_root,
            json_path=paths.json_path,
            output_dir=paths.output_dir,
            hf_cache=repos.hf_cache,
            env=env,
            log_path=paths.log_path,
            skip_if_output_exists=cfg.transfer.skip_if_output_exists,
        )

        append_log(paths.log_path, f"[DONE] output_dir={paths.output_dir}")
        console_print(f"[JOB DONE]  out={paths.output_dir}")

        # Manifest (success)
        append_manifest(
            repos.work_dir,
            {
                "status": "done",
                "input_video": str(job.video_path),
                "variant_idx": job.variant_idx,
                "job_id": job.job_id,
                "control_type": cfg.run.control_type,
                "prompt_path": str(paths.prompt_path),
                "json_path": str(paths.json_path),
                "output_dir": str(paths.output_dir),
                "log_path": str(paths.log_path),
                "control_vis_path": str(control_vis),
            },
        )

        # Cleanup intermediates (keep control_cache + final output)
        if cfg.cleanup.enable:
            cleanup_job_intermediates(
                prompt_path=paths.prompt_path,
                json_path=paths.json_path,
                non_cached_control_dir=paths.control_dir,
                log_path=paths.log_path,
                delete_prompt=cfg.cleanup.delete_prompt,
                delete_json=cfg.cleanup.delete_json,
                delete_non_cached_control_dir=cfg.cleanup.delete_non_cached_control_dir,
                delete_log=cfg.cleanup.delete_log,
            )

    except Exception as e:
        append_log(paths.log_path, f"[FAILED] {repr(e)}")
        console_print(f"[JOB FAIL] {job.video_path.name}  variant={job.variant_idx}  job_id={job.job_id}")

        append_manifest(
            repos.work_dir,
            {
                "status": "failed",
                "input_video": str(job.video_path),
                "variant_idx": job.variant_idx,
                "job_id": job.job_id,
                "control_type": cfg.run.control_type,
                "prompt_path": str(paths.prompt_path),
                "json_path": str(paths.json_path),
                "output_dir": str(paths.output_dir),
                "log_path": str(paths.log_path),
                "error": repr(e),
            },
        )
        raise
