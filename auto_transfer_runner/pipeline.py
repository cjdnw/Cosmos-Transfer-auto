from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, TypeVar

from .config import GlobalConfig
from .steps.prompt import generate_prompt
from .steps.control import ControlType, generate_control, generate_controls_combo
from .steps.preprocess import preprocess_video, PreprocessSettings
from .steps.json_spec import (
    write_json_spec_blur,
    write_json_spec_depth,
    write_json_spec_edge,
    write_json_spec_seg,
    write_json_spec_multicontrol,
)
from .steps.inference import run_transfer_inference
from .steps.arm_swap import run_arm_swap
from .utils.logging_utils import append_log, console_print
from .utils.paths import PathBundle, make_job_paths, make_output_dir
from .utils.text import sanitize_name
from .utils.manifest import append_manifest
from .utils.cleanup import cleanup_job_intermediates
from .utils.hashing import short_hash
from .utils.checkpoints import (
    checkpoint_dir_for_job,
    job_done_marker_path,
    mark_done,
    any_marker_exists,
    marker_exists,
    marker_path,
)


T = TypeVar("T")


@dataclass(frozen=True)
class Job:
    """One video task. job_id makes all outputs unique across batch + multi-GPU runs."""
    video_path: Path
    job_id: str
    variant_idx: int = 0
    input_root: Path | None = None


def run_job(job: Job, cfg: GlobalConfig, env: Dict[str, str]) -> None:
    """Run the full pipeline for a single video: Step2 -> Step5."""
    repos = cfg.repos
    input_root = job.input_root or job.video_path.parent
    paths = make_job_paths(
        repos.work_dir,
        job.video_path,
        job.job_id,
        cfg.run.control_type,
        input_root=input_root,
        variant_idx=job.variant_idx,
    )
    checkpoint_dir = checkpoint_dir_for_job(repos.work_dir, paths.output_dir)
    job_done_marker = job_done_marker_path(checkpoint_dir)
    step2_marker = marker_path(checkpoint_dir, "step2_prompt")
    step3_marker = marker_path(checkpoint_dir, "step3_control")
    step4_marker = marker_path(checkpoint_dir, "step4_json")
    step5_marker = marker_path(checkpoint_dir, "step5_infer")
    step6_marker = marker_path(checkpoint_dir, "step6_arm_swap")

    _bootstrap_checkpoints(
        checkpoint_dir=checkpoint_dir,
        job_done_marker=job_done_marker,
        step2_marker=step2_marker,
        step3_marker=step3_marker,
        step4_marker=step4_marker,
        step5_marker=step5_marker,
        step6_marker=step6_marker,
        paths=paths,
        cfg=cfg,
        input_root=input_root,
        job=job,
    )

    if marker_exists(job_done_marker) and paths.output_dir.exists() and any(paths.output_dir.iterdir()):
        append_log(paths.log_path, f"[SKIP] job checkpoint done: {job_done_marker}")
        console_print(f"[SKIP] job already done (checkpoint): {job.video_path.name} variant={job.variant_idx}")
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
                "arm_swap_output": None,
                "log_path": str(paths.log_path),
                "skip_reason": "checkpoint",
            },
        )
        return

    def _log_duration(step_label: str, duration_sec: float) -> None:
        msg = f"[{step_label}] 耗时 {duration_sec:.2f}s"
        append_log(paths.log_path, msg)
        console_print(msg)

    def _run_with_timing(step_label: str, fn: Callable[[], T]) -> T:
        start = perf_counter()
        try:
            return fn()
        finally:
            _log_duration(step_label, perf_counter() - start)

    # Log header
    append_log(paths.log_path, "#" * 90)
    append_log(paths.log_path, f"JOB_ID={job.job_id}")
    append_log(paths.log_path, f"VARIANT_IDX={job.variant_idx}")
    append_log(paths.log_path, f"VIDEO={job.video_path}")
    append_log(paths.log_path, "#" * 90)

    console_print(f"[JOB START] {job.video_path.name}  variant={job.variant_idx}  job_id={job.job_id}")
    job_start = perf_counter()
    job_success = False
    arm_swap_output: Path | None = None
    preprocessed_video: Path | None = None

    try:
        preprocessed_video = _run_with_timing(
            "STEP1 Preprocess",
            lambda: preprocess_video(
                video_path=job.video_path,
                settings=PreprocessSettings(
                    fps=cfg.preprocess.fps,
                    width=cfg.preprocess.width,
                    height=cfg.preprocess.height,
                    keep_aspect=cfg.preprocess.keep_aspect,
                    pre_chunk_mode=cfg.preprocess.pre_chunk_mode,
                    pre_chunk_policy=cfg.preprocess.pre_chunk_policy,
                    pre_chunk_clip_where=cfg.preprocess.pre_chunk_clip_where,
                    pre_chunk_max_adjust=cfg.preprocess.pre_chunk_max_adjust,
                    pre_chunk_force_chunks=cfg.preprocess.pre_chunk_force_chunks,
                    pre_chunk_restore_length=cfg.preprocess.pre_chunk_restore_length,
                    num_video_frames_per_chunk=cfg.transfer.num_video_frames_per_chunk,
                    num_conditional_frames=cfg.transfer.num_conditional_frames,
                ),
                output_root=repos.work_dir / "prepro_video",
                input_root=input_root,
                env=env,
                log_path=paths.log_path,
                lock_timeout_sec=cfg.cache.lock_timeout_sec,
                stale_lock_sec=cfg.cache.stale_lock_sec,
            ),
        )

        # Step2: prompt
        if marker_exists(step2_marker) and paths.prompt_path.exists() and paths.prompt_path.stat().st_size > 0:
            append_log(paths.log_path, f"[SKIP] step2 checkpoint exists: {step2_marker}")
            console_print(f"[SKIP] step2 prompt (checkpoint): {paths.prompt_path.name}")
        else:
            _run_with_timing(
                "STEP2 Prompt",
                lambda: generate_prompt(
                    video_path=job.video_path,
                    out_prompt=paths.prompt_path,
                    prompt_bracket_path=paths.prompt_bracket_path,
                    prompt_replacements_path=paths.prompt_replacements_path,
                    work_dir=repos.work_dir,
                    autoprompt_root=repos.autoprompt_root,
                    conda_env=cfg.prompt.conda_env,
                    fps=cfg.prompt.fps,
                    size=cfg.prompt.size,
                    variant=cfg.prompt.variant,
                    model_type=cfg.prompt.model_type,
                    env=env,
                    log_path=paths.log_path,
                    reuse_if_exists=cfg.prompt.reuse_if_exists,
                    enable_cache=cfg.cache.enable,
                    force_regen=cfg.cache.force_regen,
                    fingerprint_mode=cfg.cache.fingerprint_mode,
                    lock_timeout_sec=cfg.cache.lock_timeout_sec,
                    stale_lock_sec=cfg.cache.stale_lock_sec,
                    seed=int(short_hash(job.job_id, 8), 16),
                    sim2real_only=cfg.prompt.sim2real_only,
                    min_replace_terms=cfg.prompt.min_replace_terms,
                    max_replace_terms=cfg.prompt.max_replace_terms,
                    replace_all_terms=cfg.prompt.replace_all_terms,
                    replacements_per_term=cfg.prompt.replacements_per_term,
                    interactive_replace=cfg.prompt.interactive_replace,
                ),
            )
            mark_done(step2_marker)

        # Step3: control
        control_video = preprocessed_video or job.video_path
        control_type = ControlType(cfg.run.control_type)
        if control_type == ControlType.COMBO:
            control_paths = _run_with_timing(
                "STEP3 Control",
                lambda: generate_controls_combo(
                    video_path=control_video,
                    work_dir=repos.work_dir,
                    depth_root=repos.depth_root,
                    seg_root=repos.seg_root,
                    cv_root=repos.cv_root,
                    depth_conda_env=cfg.depth.conda_env,
                    seg_conda_env=cfg.seg.conda_env,
                    cv_conda_env=cfg.cv.conda_env,
                    encoder=cfg.depth.encoder,
                    grayscale=cfg.depth.grayscale,
                    seg_level=cfg.seg.level,
                    seg_use_other_level=cfg.seg.use_other_level,
                    seg_detect_stride=cfg.seg.detect_stride,
                    seg_save_intermediate=cfg.seg.save_intermediate,
                    seg_save_npy=cfg.seg.save_npy,
                    seg_enable_highlight=cfg.seg.enable_highlight,
                    seg_enable_overlay=cfg.seg.enable_overlay,
                    seg_overlay_style=cfg.seg.overlay_style,
                    edge_low=cfg.cv.canny_low,
                    edge_high=cfg.cv.canny_high,
                    enable_cache=cfg.cache.enable,
                    force_regen=cfg.cache.force_regen,
                    fingerprint_mode=cfg.cache.fingerprint_mode,
                    lock_timeout_sec=cfg.cache.lock_timeout_sec,
                    stale_lock_sec=cfg.cache.stale_lock_sec,
                    non_cached_output_dir=paths.control_dir,
                    env=env,
                    log_path=paths.log_path,
                ),
            )
            control_path = None
        else:
            control_paths = None
            control_path = _run_with_timing(
                "STEP3 Control",
                lambda: generate_control(
                    control_type=control_type,
                    video_path=control_video,
                    work_dir=repos.work_dir,
                    depth_root=repos.depth_root,
                    seg_root=repos.seg_root,
                    cv_root=repos.cv_root,
                    depth_conda_env=cfg.depth.conda_env,
                    seg_conda_env=cfg.seg.conda_env,
                    cv_conda_env=cfg.cv.conda_env,
                    encoder=cfg.depth.encoder,
                    grayscale=cfg.depth.grayscale,
                    seg_level=cfg.seg.level,
                    seg_use_other_level=cfg.seg.use_other_level,
                    seg_detect_stride=cfg.seg.detect_stride,
                    seg_save_intermediate=cfg.seg.save_intermediate,
                    seg_save_npy=cfg.seg.save_npy,
                    seg_enable_highlight=cfg.seg.enable_highlight,
                    seg_enable_overlay=cfg.seg.enable_overlay,
                    seg_overlay_style=cfg.seg.overlay_style,
                    edge_low=cfg.cv.canny_low,
                    edge_high=cfg.cv.canny_high,
                    blur_sigma=cfg.cv.blur_sigma,
                    enable_cache=cfg.cache.enable,
                    force_regen=cfg.cache.force_regen,
                    fingerprint_mode=cfg.cache.fingerprint_mode,
                    lock_timeout_sec=cfg.cache.lock_timeout_sec,
                    stale_lock_sec=cfg.cache.stale_lock_sec,
                    non_cached_output_dir=paths.control_dir,
                    env=env,
                    log_path=paths.log_path,
                ),
            )
        if not marker_exists(step3_marker):
            mark_done(step3_marker)

        # Step4: json spec
        name = sanitize_name(job.video_path.stem)
        seed = cfg.transfer.seed if cfg.transfer.seed is not None else int(short_hash(job.job_id, 8), 16)
        if marker_exists(step4_marker) and paths.json_path.exists() and paths.json_path.stat().st_size > 0:
            append_log(paths.log_path, f"[SKIP] step4 checkpoint exists: {step4_marker}")
            console_print(f"[SKIP] step4 json (checkpoint): {paths.json_path.name}")
        elif control_type == ControlType.DEPTH:
            _run_with_timing(
                "STEP4 JSON",
                lambda: write_json_spec_depth(
                    json_path=paths.json_path,
                    name=name,
                    prompt_path=paths.prompt_path,
                    video_path=control_video,
                    seed=seed,
                    transfer_cfg=cfg.transfer,
                    control_path=control_path,
                ),
            )
            mark_done(step4_marker)
        elif control_type == ControlType.SEG:
            _run_with_timing(
                "STEP4 JSON",
                lambda: write_json_spec_seg(
                    json_path=paths.json_path,
                    name=name,
                    prompt_path=paths.prompt_path,
                    video_path=control_video,
                    seed=seed,
                    transfer_cfg=cfg.transfer,
                    control_path=control_path,
                ),
            )
            mark_done(step4_marker)
        elif control_type == ControlType.EDGE:
            _run_with_timing(
                "STEP4 JSON",
                lambda: write_json_spec_edge(
                    json_path=paths.json_path,
                    name=name,
                    prompt_path=paths.prompt_path,
                    video_path=control_video,
                    seed=seed,
                    transfer_cfg=cfg.transfer,
                    control_path=control_path,
                ),
            )
            mark_done(step4_marker)
        elif control_type == ControlType.BLUR:
            _run_with_timing(
                "STEP4 JSON",
                lambda: write_json_spec_blur(
                    json_path=paths.json_path,
                    name=name,
                    prompt_path=paths.prompt_path,
                    video_path=control_video,
                    seed=seed,
                    transfer_cfg=cfg.transfer,
                    control_path=control_path,
                ),
            )
            mark_done(step4_marker)
        elif control_type == ControlType.COMBO:
            if control_paths is None:
                raise RuntimeError("Combo control paths were not generated.")
            _run_with_timing(
                "STEP4 JSON",
                lambda: write_json_spec_multicontrol(
                    json_path=paths.json_path,
                    name=name,
                    prompt_path=paths.prompt_path,
                    video_path=control_video,
                    seed=seed,
                    transfer_cfg=cfg.transfer,
                    depth_path=control_paths["depth"],
                    edge_path=control_paths["edge"],
                    seg_path=control_paths["seg"],
                ),
            )
            mark_done(step4_marker)
        else:
            raise NotImplementedError("Only depth/seg/edge/blur/combo json spec are implemented for now.")

        with paths.json_path.open("r", encoding="utf-8") as f:
            json_resolution = json.load(f).get("resolution")
        append_log(paths.log_path, f"[STEP4] resolution={json_resolution}")
        console_print(f"[STEP4] resolution={json_resolution}")
        append_log(paths.log_path, f"[STEP4] Wrote json: {paths.json_path}")
        console_print(f"[STEP4] json={paths.json_path.name}")

        # Step5: inference
        if marker_exists(step5_marker) and paths.output_dir.exists() and any(paths.output_dir.iterdir()):
            append_log(paths.log_path, f"[SKIP] step5 checkpoint exists: {step5_marker}")
            console_print(f"[SKIP] step5 inference (checkpoint): {paths.output_dir.name}")
        else:
            _run_with_timing(
                "STEP5 Inference",
                lambda: run_transfer_inference(
                    transfer_root=repos.transfer_root,
                    json_path=paths.json_path,
                    output_dir=paths.output_dir,
                    hf_cache=repos.hf_cache,
                    env=env,
                    log_path=paths.log_path,
                    skip_if_output_exists=cfg.transfer.skip_if_output_exists,
                ),
            )
            mark_done(step5_marker)

        append_log(paths.log_path, f"[DONE] output_dir={paths.output_dir}")
        console_print(f"[JOB DONE]  out={paths.output_dir}")

        if cfg.arm_swap.enable:
            arm_swap_workdir = cfg.arm_swap.workdir_root / paths.output_dir.name
            arm_swap_output_base_dir = make_output_dir(
                cfg.arm_swap.output_root,
                input_root=input_root,
                video_path=job.video_path,
                variant_idx=job.variant_idx,
            )
            if marker_exists(step6_marker):
                append_log(paths.log_path, f"[SKIP] step6 checkpoint exists: {step6_marker}")
                console_print(f"[SKIP] step6 arm swap (checkpoint): {paths.output_dir.name}")
            else:
                arm_swap_output = _run_with_timing(
                    "STEP6 Arm Swap",
                    lambda: run_arm_swap(
                        project_root=cfg.arm_swap.project_root,
                        conda_env=cfg.arm_swap.conda_env,
                        output_dir=paths.output_dir,
                        output_base_dir=arm_swap_output_base_dir,
                        workdir=arm_swap_workdir,
                        orig_video=preprocessed_video or job.video_path,
                        qwen_prompt=cfg.arm_swap.qwen_prompt,
                        erode=cfg.arm_swap.erode,
                        feather=cfg.arm_swap.feather,
                        dilate=cfg.arm_swap.dilate,
                        env=env,
                        log_path=paths.log_path,
                    ),
                )
                append_log(paths.log_path, f"[DONE] arm_swap_output={arm_swap_output}")
                console_print(f"[ARM SWAP DONE] out={arm_swap_output}")
                mark_done(step6_marker)

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
                "arm_swap_output": str(arm_swap_output) if arm_swap_output is not None else None,
                "log_path": str(paths.log_path),
                "control_vis_path": str(control_path) if control_path is not None else None,
                "control_paths": (
                    {k: str(v) for k, v in control_paths.items()} if control_paths is not None else None
                ),
            },
        )

        # Cleanup intermediates (keep control_cache + final output)
        if cfg.cleanup.enable:
            _run_with_timing(
                "CLEANUP",
                lambda: cleanup_job_intermediates(
                    prompt_path=paths.prompt_path,
                    prompt_intermediate_paths=[paths.prompt_bracket_path, paths.prompt_replacements_path],
                    json_path=paths.json_path,
                    non_cached_control_dir=paths.control_dir,
                    log_path=paths.log_path,
                    delete_prompt=cfg.cleanup.delete_prompt,
                    delete_json=cfg.cleanup.delete_json,
                    delete_non_cached_control_dir=cfg.cleanup.delete_non_cached_control_dir,
                    delete_log=cfg.cleanup.delete_log,
                ),
            )

        mark_done(job_done_marker)
        job_success = True

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
                "arm_swap_output": str(arm_swap_output) if arm_swap_output is not None else None,
                "log_path": str(paths.log_path),
                "error": repr(e),
            },
        )
        raise

    finally:
        job_duration = perf_counter() - job_start
        status = "success" if job_success else "failed"
        _log_duration(f"JOB TOTAL ({status})", job_duration)


def _bootstrap_checkpoints(
    *,
    checkpoint_dir: Path,
    job_done_marker: Path,
    step2_marker: Path,
    step3_marker: Path,
    step4_marker: Path,
    step5_marker: Path,
    step6_marker: Path,
    paths: PathBundle,
    cfg: GlobalConfig,
    input_root: Path,
    job: Job,
) -> None:
    """Infer step markers from existing artifacts when no checkpoint files exist."""
    if any_marker_exists(checkpoint_dir):
        return

    if paths.prompt_path.exists() and paths.prompt_path.stat().st_size > 0:
        mark_done(step2_marker)

    if paths.json_path.exists() and paths.json_path.stat().st_size > 0:
        mark_done(step3_marker)
        mark_done(step4_marker)

    if paths.output_dir.exists() and any(paths.output_dir.iterdir()):
        mark_done(step5_marker)

    if cfg.arm_swap.enable:
        if paths.output_dir.exists() and any(paths.output_dir.iterdir()):
            arm_swap_output_base_dir = make_output_dir(
                cfg.arm_swap.output_root,
                input_root=input_root,
                video_path=job.video_path,
                variant_idx=job.variant_idx,
            )
            if _arm_swap_output_exists(paths.output_dir, arm_swap_output_base_dir):
                mark_done(step6_marker)
                mark_done(job_done_marker)
    else:
        if marker_exists(step5_marker):
            mark_done(job_done_marker)


def _arm_swap_output_exists(output_dir: Path, output_base_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    mp4s = sorted([p for p in output_dir.glob("*.mp4") if p.is_file()])
    if not mp4s:
        return False
    output_base_dir.mkdir(parents=True, exist_ok=True)
    expected = output_base_dir / mp4s[0].name
    return expected.exists() and expected.stat().st_size > 0
