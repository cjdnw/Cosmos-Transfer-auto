from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, TypeVar

from .config import ArmSwapConfig, GlobalConfig
from .steps.prompt import generate_prompt
from .steps.control import ControlType, generate_control, generate_controls_combo
from .steps.preprocess import preprocess_video, PreprocessSettings, frame_map_path_for
from .steps.json_spec import (
    write_json_spec_blur,
    write_json_spec_depth,
    write_json_spec_edge,
    write_json_spec_seg,
    write_json_spec_multicontrol,
)
from .steps.inference import run_transfer_inference
from .steps.arm_swap import (
    ensure_qwen_sam_cache,
    find_mask_dir_with_files,
    run_arm_swap_swap_only,
    sync_mask_dir,
)
from .steps.postprocess import (
    probe_video_stats,
    restore_duration,
    VideoStats,
    find_transfer_output_mp4,
    has_transfer_output_mp4,
)
from .steps.keyframe_preflight import ProxyPreflightSettings, run_proxy_iou_preflight
from .utils.logging_utils import append_log, console_print
from .utils.paths import PathBundle, make_job_paths, make_output_dir
from .utils.text import sanitize_name
from .utils.manifest import append_manifest
from .utils.cleanup import cleanup_job_intermediates, safe_rmtree
from .utils.hashing import short_hash, video_cache_key
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
    job_index: int = 0
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
    step0_marker = marker_path(checkpoint_dir, "step0_probe")
    step1_marker = marker_path(checkpoint_dir, "step1_preprocess")
    step1_5_5_marker = marker_path(checkpoint_dir, "step1_5_5_preflight")
    step2_marker = marker_path(checkpoint_dir, "step2_prompt")
    step3_marker = marker_path(checkpoint_dir, "step3_control")
    step4_marker = marker_path(checkpoint_dir, "step4_json")
    step5_marker = marker_path(checkpoint_dir, "step5_infer")
    step5_restore_marker = marker_path(checkpoint_dir, "step5_restore")
    step6_marker = marker_path(checkpoint_dir, "step6_arm_swap")

    _bootstrap_checkpoints(
        checkpoint_dir=checkpoint_dir,
        job_done_marker=job_done_marker,
        step0_marker=step0_marker,
        step1_marker=step1_marker,
        step1_5_5_marker=step1_5_5_marker,
        step2_marker=step2_marker,
        step3_marker=step3_marker,
        step4_marker=step4_marker,
        step5_marker=step5_marker,
        step5_restore_marker=step5_restore_marker,
        step6_marker=step6_marker,
        paths=paths,
        cfg=cfg,
        input_root=input_root,
        job=job,
    )

    if marker_exists(job_done_marker) and has_transfer_output_mp4(paths.output_dir):
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
                "post_restore_output": (
                    str(find_transfer_output_mp4(paths.output_dir))
                    if cfg.postprocess.restore_duration
                    else None
                ),
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
    original_stats: VideoStats | None = None
    frame_map_path: Path | None = None

    try:
        if cfg.postprocess.restore_duration:
            original_stats = _run_with_timing(
                "STEP0 Probe Original",
                lambda: probe_video_stats(job.video_path, env=env, log_path=paths.log_path),
            )
            mark_done(step0_marker)
        preprocessed_video = _run_with_timing(
            "STEP1 Preprocess",
            lambda: preprocess_video(
                video_path=job.video_path,
                settings=PreprocessSettings(
                    width=cfg.preprocess.width,
                    height=cfg.preprocess.height,
                    keep_aspect=cfg.preprocess.keep_aspect,
                    pre_chunk_mode=cfg.preprocess.pre_chunk_mode,
                    pre_chunk_policy=cfg.preprocess.pre_chunk_policy,
                    pre_chunk_clip_where=cfg.preprocess.pre_chunk_clip_where,
                    pre_chunk_max_adjust=cfg.preprocess.pre_chunk_max_adjust,
                    pre_chunk_force_chunks=cfg.preprocess.pre_chunk_force_chunks,
                    num_video_frames_per_chunk=cfg.transfer.num_video_frames_per_chunk,
                    num_conditional_frames=cfg.transfer.num_conditional_frames,
                    keyframe_sampling=cfg.preprocess.keyframe_sampling,
                    keyframe_target_frames=cfg.preprocess.keyframe_target_frames,
                    keyframe_smooth_win=cfg.preprocess.keyframe_smooth_win,
                    keyframe_max_gap=cfg.preprocess.keyframe_max_gap,
                    parquet_pose_subset=cfg.preprocess.parquet_pose_subset,
                ),
                output_root=repos.work_dir / "prepro_video",
                input_root=input_root,
                env=env,
                log_path=paths.log_path,
                lock_timeout_sec=cfg.cache.lock_timeout_sec,
                stale_lock_sec=cfg.cache.stale_lock_sec,
            ),
        )
        mark_done(step1_marker)
        if preprocessed_video is not None:
            candidate_map_path = frame_map_path_for(preprocessed_video)
            if candidate_map_path.exists():
                frame_map_path = candidate_map_path

        if (
            preprocessed_video is not None
            and cfg.preprocess.proxy_preflight_enable
            and cfg.preprocess.keyframe_sampling in {"parquet_motion", "uniform"}
            and (cfg.preprocess.keyframe_sampling != "parquet_motion" or frame_map_path is not None)
        ):
            proxy_settings = ProxyPreflightSettings(
                enable=cfg.preprocess.proxy_preflight_enable,
                iou_threshold=cfg.preprocess.proxy_iou_threshold,
                iou_high_threshold=cfg.preprocess.proxy_iou_high_threshold,
                motion_low_quantile=cfg.preprocess.proxy_motion_low_quantile,
                max_iters=cfg.preprocess.proxy_max_iters,
                restore_mode=cfg.preprocess.proxy_restore_mode,
                rife_backend=cfg.preprocess.proxy_rife_backend,
                cache_iou_csv=cfg.preprocess.proxy_preflight_cache_csv,
                removal_strategy=cfg.preprocess.proxy_removal_strategy,
                removal_buckets=cfg.preprocess.proxy_removal_buckets,
            )
            step_label = (
                "STEP1.5.5 Proxy IoU Preflight (uniform)"
                if cfg.preprocess.keyframe_sampling == "uniform"
                else "STEP1.5.5 Proxy IoU Preflight"
            )
            preprocessed_video, frame_map_path = _run_with_timing(
                step_label,
                lambda: run_proxy_iou_preflight(
                    original_video=job.video_path,
                    keyframes_video=preprocessed_video,
                    keyframes_frame_map=frame_map_path,
                    preprocess_cfg=cfg.preprocess,
                    num_video_frames_per_chunk=cfg.transfer.num_video_frames_per_chunk,
                    num_conditional_frames=cfg.transfer.num_conditional_frames,
                    postprocess_cfg=cfg.postprocess,
                    arm_swap_cfg=cfg.arm_swap,
                    cache_cfg=cfg.cache,
                    proxy_cfg=proxy_settings,
                    work_dir=cfg.repos.work_dir,
                    env=env,
                    log_path=paths.log_path,
                    run_gpus=cfg.run.gpus,
                    job_index=job.job_index,
                ),
            )
            mark_done(step1_5_5_marker)

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
        if marker_exists(step5_marker) and has_transfer_output_mp4(paths.output_dir):
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
                    disable_guardrails=cfg.transfer.disable_guardrails,
                    skip_if_output_exists=cfg.transfer.skip_if_output_exists,
                ),
            )
            mark_done(step5_marker)

        # Step5.5: post-restore duration
        if cfg.postprocess.restore_duration:
            if marker_exists(step5_restore_marker) and has_transfer_output_mp4(paths.output_dir):
                append_log(paths.log_path, f"[SKIP] step5.5 checkpoint exists: {step5_restore_marker}")
                console_print(f"[SKIP] step5.5 post-restore (checkpoint): {paths.output_dir.name}")
            else:
                if original_stats is None:
                    raise RuntimeError("Missing original video stats for post-restore.")
                _run_with_timing(
                    "STEP5.5 Post-restore",
                    lambda: restore_duration(
                        output_dir=paths.output_dir,
                        input_video_path=job.video_path,
                        original_stats=original_stats,
                        mode=cfg.postprocess.restore_mode,
                        rife_bin=cfg.postprocess.rife_bin,
                        rife_model=cfg.postprocess.rife_model,
                        rife_backend=cfg.postprocess.rife_backend,
                        rife_pytorch_repo=cfg.postprocess.rife_pytorch_repo,
                        rife_pytorch_model_dir=cfg.postprocess.rife_pytorch_model_dir,
                        rife_pytorch_env=cfg.postprocess.rife_pytorch_env,
                        rife_scale=cfg.postprocess.rife_scale,
                        rife_keep_audio=cfg.postprocess.rife_keep_audio,
                        rife_time_steps_mode=cfg.postprocess.rife_time_steps_mode,
                        rife_time_steps_cache=(
                            Path(cfg.postprocess.rife_time_steps_cache)
                            if cfg.postprocess.rife_time_steps_cache
                            else None
                        ),
                        rife_sam_mask_dir=(
                            Path(cfg.postprocess.rife_sam_mask_dir)
                            if cfg.postprocess.rife_sam_mask_dir
                            else (cfg.repos.work_dir / "rife_sam_cache")
                        ),
                        fingerprint_mode=cfg.cache.fingerprint_mode,
                        rife_time_steps_force_sam=cfg.postprocess.rife_time_steps_force_sam,
                        rife_sam_generate=cfg.postprocess.rife_sam_generate,
                        rife_sam_bbox_json=(
                            Path(cfg.postprocess.rife_sam_bbox_json)
                            if cfg.postprocess.rife_sam_bbox_json
                            else None
                        ),
                        rife_sam_cleanup_workdir=cfg.postprocess.rife_sam_cleanup_workdir,
                        rife_sam_project_root=cfg.arm_swap.project_root,
                        rife_sam_conda_env=cfg.arm_swap.conda_env,
                        rife_sam_qwen_prompt=cfg.arm_swap.qwen_prompt,
                        rife_sam_erode=cfg.arm_swap.erode,
                        rife_sam_feather=cfg.arm_swap.feather,
                        rife_sam_dilate=cfg.arm_swap.dilate,
                        rife_sam_no_workdir_write=cfg.arm_swap.no_workdir_write,
                        rife_sam_cache_workdir=cfg.arm_swap.workdir_root
                        / video_cache_key(job.video_path, mode=cfg.cache.fingerprint_mode),
                        rife_pose_subset=cfg.postprocess.rife_pose_subset,
                        rife_time_steps_debug=cfg.postprocess.rife_time_steps_debug,
                        rife_cache_root=cfg.repos.work_dir / "rife_cache",
                        frame_map_path=frame_map_path,
                        run_gpus=cfg.run.gpus,
                        job_index=job.job_index,
                        env=env,
                        log_path=paths.log_path,
                    ),
                )
                mark_done(step5_restore_marker)

        append_log(paths.log_path, f"[DONE] output_dir={paths.output_dir}")
        console_print(f"[JOB DONE]  out={paths.output_dir}")

        if cfg.arm_swap.enable:
            sam_cache_key = video_cache_key(job.video_path, mode=cfg.cache.fingerprint_mode)
            arm_swap_workdir = cfg.arm_swap.workdir_root / sam_cache_key
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
                transfer_mp4 = find_transfer_output_mp4(paths.output_dir)
                sam_cache_root = cfg.repos.work_dir / "rife_sam_cache"
                sam_cache_dir = sam_cache_root / sam_cache_key
                sam_cache_root.mkdir(parents=True, exist_ok=True)
                cached_mask_dir = find_mask_dir_with_files(sam_cache_dir)
                if cached_mask_dir is not None:
                    masks_dir = cached_mask_dir
                else:
                    masks_dir = ensure_qwen_sam_cache(
                        project_root=cfg.arm_swap.project_root,
                        conda_env=cfg.arm_swap.conda_env,
                        orig_video=job.video_path,
                        workdir=arm_swap_workdir,
                        qwen_prompt=cfg.arm_swap.qwen_prompt,
                        env=env,
                        log_path=paths.log_path,
                        log_label="STEP6",
                    )
                    sync_mask_dir(masks_dir, sam_cache_dir)
                    masks_dir = find_mask_dir_with_files(sam_cache_dir) or masks_dir
                arm_swap_output = _run_with_timing(
                    "STEP6 Arm Swap",
                    lambda: run_arm_swap_swap_only(
                        project_root=cfg.arm_swap.project_root,
                        conda_env=cfg.arm_swap.conda_env,
                        orig_video=job.video_path,
                        skin_video=transfer_mp4,
                        workdir=arm_swap_workdir,
                        arm_mask_dir=masks_dir,
                        output_path=arm_swap_output_base_dir / transfer_mp4.name,
                        erode=cfg.arm_swap.erode,
                        feather=cfg.arm_swap.feather,
                        dilate=cfg.arm_swap.dilate,
                        no_workdir_write=cfg.arm_swap.no_workdir_write,
                        arm_swap_workdir_write=cfg.arm_swap.workdir_write,
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
                "post_restore_output": (
                    str(find_transfer_output_mp4(paths.output_dir))
                    if cfg.postprocess.restore_duration
                    else None
                ),
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
        _copy_fail_jsons(
            work_dir=repos.work_dir,
            job_id=job.job_id,
            json_paths=[paths.prompt_replacements_path, paths.json_path],
            log_path=paths.log_path,
        )

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
        _cleanup_job_cache_dirs(cfg=cfg, job=job, output_dir=paths.output_dir, log_path=paths.log_path)


def _copy_fail_jsons(
    *,
    work_dir: Path,
    job_id: str,
    json_paths: list[Path],
    log_path: Path,
) -> None:
    log_fail_dir = work_dir / "log_fail"
    log_fail_dir.mkdir(parents=True, exist_ok=True)
    for path in json_paths:
        if not path.exists():
            continue
        target = log_fail_dir / f"{path.stem}__{job_id}{path.suffix}"
        try:
            shutil.copy2(path, target)
            append_log(log_path, f"[FAILED] saved json to {target}")
        except Exception as exc:
            append_log(log_path, f"[FAILED] could not save json {path}: {exc!r}")


def _bootstrap_checkpoints(
    *,
    checkpoint_dir: Path,
    job_done_marker: Path,
    step0_marker: Path,
    step1_marker: Path,
    step1_5_5_marker: Path,
    step2_marker: Path,
    step3_marker: Path,
    step4_marker: Path,
    step5_marker: Path,
    step5_restore_marker: Path,
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

    if has_transfer_output_mp4(paths.output_dir):
        mark_done(step5_marker)
        if cfg.postprocess.restore_duration and marker_exists(step5_restore_marker):
            mark_done(step5_restore_marker)

    if cfg.arm_swap.enable:
        if has_transfer_output_mp4(paths.output_dir):
            arm_swap_output_base_dir = make_output_dir(
                cfg.arm_swap.output_root,
                input_root=input_root,
                video_path=job.video_path,
                variant_idx=job.variant_idx,
            )
            if _arm_swap_output_exists(paths.output_dir, arm_swap_output_base_dir, cfg.arm_swap):
                mark_done(step6_marker)
                if not cfg.postprocess.restore_duration or marker_exists(step5_restore_marker):
                    mark_done(job_done_marker)
    else:
        if marker_exists(step5_marker) and (
            not cfg.postprocess.restore_duration or marker_exists(step5_restore_marker)
        ):
            mark_done(job_done_marker)


def _arm_swap_output_exists(
    output_dir: Path,
    output_base_dir: Path,
    arm_swap_cfg: ArmSwapConfig,
) -> bool:
    if arm_swap_cfg.qwen_only and not arm_swap_cfg.sam_only:
        if arm_swap_cfg.bbox_json is None:
            return False
        return arm_swap_cfg.bbox_json.exists() and arm_swap_cfg.bbox_json.stat().st_size > 0
    if not output_dir.exists():
        return False
    mp4s = sorted([p for p in output_dir.glob("*.mp4") if p.is_file()])
    if not mp4s:
        return False
    output_base_dir.mkdir(parents=True, exist_ok=True)
    expected = output_base_dir / mp4s[0].name
    return expected.exists() and expected.stat().st_size > 0


def _cleanup_job_cache_dirs(
    *,
    cfg: GlobalConfig,
    job: Job,
    output_dir: Path,
    log_path: Path,
) -> None:
    if cfg.postprocess.restore_duration:
        rife_cache_root = cfg.repos.work_dir / "rife_cache"
        output_hash = short_hash(str(output_dir.resolve()), 8)
        for suffix in ("", "__pytorch"):
            safe_rmtree(
                rife_cache_root / f"{output_dir.name}_{output_hash}{suffix}",
                log_path=log_path,
            )
