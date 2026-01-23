from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    DEFAULT_AUTOPROMPT_ROOT,
    DEFAULT_DEPTH_ROOT,
    DEFAULT_SEG_ROOT,
    DEFAULT_CV_ROOT,
    DEFAULT_HF_CACHE,
    DEFAULT_TRANSFER_ROOT,
    DEFAULT_WORK_DIR,
    CACHE_DEFAULTS,
    DEPTH_DEFAULTS,
    SEG_DEFAULTS,
    CV_DEFAULTS,
    PREPROCESS_DEFAULTS,
    POSTPROCESS_DEFAULTS,
    PROMPT_DEFAULTS,
    TRANSFER_DEFAULTS,
    ARM_SWAP_DEFAULTS,
    RepoPaths,
    PromptConfig,
    DepthConfig,
    SegConfig,
    CvControlConfig,
    PreprocessConfig,
    PostprocessConfig,
    TransferConfig,
    ArmSwapConfig,
    CacheConfig,
    CleanupConfig,
    RunConfig,
    GlobalConfig,
)
from .pipeline import Job, run_job
from .utils.hashing import short_hash
from .utils.logging_utils import console_print, format_console_line
from .utils.paths import make_output_dir
from .utils.dataset_paths import task_parquet_for_video
from .utils.checkpoints import (
    CHECKPOINT_DIRNAME,
    checkpoint_dir_for_job,
    clear_checkpoint_dir,
    job_done_marker_path,
)

RESOLUTION_CHOICES = [
    "1080",
    "1024",
    "720",
    "720p",
    "720robocasa",
    "480",
    "480p",
    "512",
    "256",
]


def now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def selection_file_name(
    input_dir: Path,
    num_videos: int,
    shuffle: bool,
    shuffle_seed: Optional[int],
    selective_task: int,
) -> str:
    key = f"{input_dir.resolve()}|{num_videos}|{shuffle}|{shuffle_seed}|{selective_task}"
    digest = short_hash(key, 10)
    return f"selection_{digest}.json"


def _task_parquet_for_video(video_path: Path, input_root: Path) -> Optional[Path]:
    """Find tasks.parquet (or task.parquet) using the dataset root derived from videos/."""
    try:
        candidate = task_parquet_for_video(video_path)
    except ValueError:
        candidate = None
    if candidate is not None and candidate.is_file():
        return candidate
    input_root = input_root.resolve()
    for parent in [video_path.parent, *video_path.parents]:
        for candidate in ("tasks.parquet", "task.parquet"):
            fallback = parent / "meta" / candidate
            if fallback.is_file():
                return fallback
        if parent == input_root:
            break
    return None


def _hash_file(path: Path) -> str:
    import hashlib

    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_videos(
    input_dir: Path,
    num_videos: int,
    shuffle: bool,
    selective_task_max: int,
    shuffle_seed: Optional[int] = None,
) -> List[Path]:
    """Select unique input videos from a folder tree (recursive)."""
    videos = sorted([p for p in input_dir.rglob("*.mp4") if p.is_file()])
    if shuffle:
        rng = random.Random(shuffle_seed)
        rng.shuffle(videos)
    if selective_task_max > 0:
        max_per_task = selective_task_max
        selected: List[Path] = []
        task_counts: Dict[str, int] = {}
        task_paths: Dict[str, Path] = {}
        task_total_videos: Dict[str, int] = {}
        task_selected_videos: Dict[str, int] = {}
        missing_task_parquet = 0
        for video in videos:
            task_parquet = _task_parquet_for_video(video, input_dir)
            if task_parquet is None:
                selected.append(video)
                missing_task_parquet += 1
                continue
            task_key = _hash_file(task_parquet)
            task_paths.setdefault(task_key, task_parquet)
            task_total_videos[task_key] = task_total_videos.get(task_key, 0) + 1
            task_counts.setdefault(task_key, 0)
            if task_counts[task_key] >= max_per_task:
                continue
            task_counts[task_key] += 1
            task_selected_videos[task_key] = task_selected_videos.get(task_key, 0) + 1
            selected.append(video)
        videos = selected
        duplicate_task_videos = sum(
            max(0, count - 1) for count in task_total_videos.values()
        )
        console_print(
            "[SELECTIVE_TASK] max_per_task="
            f"{max_per_task} unique_tasks={len(task_total_videos)} "
            f"missing_task_parquet={missing_task_parquet} "
            f"task_videos={sum(task_total_videos.values())} "
            f"duplicate_task_videos={duplicate_task_videos} "
            f"selected={len(videos)}"
        )
        for task_key, total_count in sorted(task_total_videos.items()):
            selected_count = task_selected_videos.get(task_key, 0)
            task_path = task_paths.get(task_key)
            console_print(
                "[SELECTIVE_TASK] task_parquet="
                f"{task_path} total={total_count} selected={selected_count}"
            )
    if num_videos and num_videos > 0:
        videos = videos[:num_videos]
    return videos


def parse_gpus(gpus: str) -> List[str]:
    if not gpus.strip():
        return ["0"]
    return [x.strip() for x in gpus.split(",") if x.strip()]


def build_config(args: argparse.Namespace) -> GlobalConfig:
    repos = RepoPaths(
        work_dir=Path(args.work_dir),
        autoprompt_root=Path(args.autoprompt_root),
        depth_root=Path(args.depth_root),
        seg_root=Path(args.seg_root),
        cv_root=Path(args.cv_root),
        transfer_root=Path(args.transfer_root),
        hf_cache=Path(args.hf_cache),
    )

    prompt = PromptConfig(
        conda_env=args.qwen_env,
        fps=args.prompt_fps,
        size=args.prompt_size,
        variant=args.prompt_variant,
        model_type=args.prompt_model_type,
        reuse_if_exists=args.reuse_prompt,
        sim2real_only=args.sim2real_only,
        min_replace_terms=args.min_terms,
        max_replace_terms=args.max_terms,
        replace_all_terms=args.replace_all_terms,
        replacements_per_term=args.replacements_per_term,
        interactive_replace=args.interactive_replace,
    )

    depth = DepthConfig(
        conda_env=args.depth_env,
        encoder=args.depth_encoder,
        grayscale=(not args.no_depth_grayscale),
    )

    seg = SegConfig(
        conda_env=args.seg_env,
        level=args.seg_level,
        use_other_level=args.seg_use_other_level,
        detect_stride=args.seg_detect_stride,
        save_intermediate=args.seg_save_intermediate,
        save_npy=args.seg_save_npy,
        enable_highlight=args.seg_enable_highlight,
        enable_overlay=(not args.seg_disable_overlay),
        overlay_style=args.seg_overlay_style,
    )

    cv = CvControlConfig(
        conda_env=args.cv_env,
        blur_sigma=args.blur_sigma,
        canny_low=args.edge_low,
        canny_high=args.edge_high,
    )

    preprocess = PreprocessConfig(
        width=args.pre_width,
        height=args.pre_height,
        keep_aspect=(not args.pre_no_keep_aspect),
        pre_chunk_mode=args.pre_chunk_mode,
        pre_chunk_policy=args.pre_chunk_policy,
        pre_chunk_clip_where=args.pre_chunk_clip_where,
        pre_chunk_max_adjust=args.pre_chunk_max_adjust,
        pre_chunk_force_chunks=args.pre_chunk_force_chunks,
        keyframe_sampling=args.keyframe_sampling,
        keyframe_target_frames=args.keyframe_target_frames,
        keyframe_smooth_win=args.keyframe_smooth_win,
        keyframe_max_gap=args.keyframe_max_gap,
        parquet_pose_subset=args.parquet_pose_subset,
        proxy_preflight_enable=args.proxy_preflight,
        proxy_iou_threshold=args.proxy_iou_threshold,
        proxy_iou_high_threshold=args.proxy_iou_high_threshold,
        proxy_motion_low_quantile=args.proxy_motion_low_quantile,
        proxy_max_iters=args.proxy_max_iters,
        proxy_restore_mode=args.proxy_restore_mode,
        proxy_rife_backend=args.proxy_rife_backend,
        proxy_preflight_cache_csv=args.proxy_preflight_cache_csv,
        proxy_removal_strategy=args.proxy_removal_strategy,
        proxy_removal_buckets=args.proxy_removal_buckets,
    )

    postprocess = PostprocessConfig(
        restore_duration=args.post_restore_duration,
        restore_mode=args.post_restore_mode,
        rife_bin=args.rife_bin,
        rife_model=args.rife_model,
        rife_backend=args.rife_backend,
        rife_pytorch_repo=args.rife_pytorch_repo,
        rife_pytorch_model_dir=(
            args.rife_pytorch_model_dir
            if args.rife_pytorch_model_dir is not None
            else (str(Path(args.rife_pytorch_repo) / "train_log") if args.rife_pytorch_repo else None)
        ),
        rife_pytorch_env=args.rife_pytorch_env,
        rife_scale=args.rife_scale,
        rife_keep_audio=args.rife_keep_audio,
        rife_time_steps_mode=args.rife_time_steps_mode,
        rife_time_steps_cache=args.rife_time_steps_cache,
        rife_sam_mask_dir=args.rife_sam_mask_dir,
        rife_time_steps_force_sam=args.rife_time_steps_force_sam,
        rife_sam_generate=args.rife_sam_generate,
        rife_sam_bbox_json=args.rife_sam_bbox_json,
        rife_sam_cleanup_workdir=(not args.rife_sam_keep_workdir),
        rife_pose_subset=args.rife_pose_subset,
        rife_time_steps_debug=args.rife_time_steps_debug,
    )

    transfer = TransferConfig(
        seed=args.seed,
        num_conditional_frames=args.num_conditional_frames,
        resolution=args.resolution,
        sigma_max=args.sigma_max,
        num_video_frames_per_chunk=args.num_video_frames_per_chunk,
        show_control_condition=args.show_control_condition,
        show_input=args.show_input,
        keep_input_resolution=(not args.disable_keep_input_resolution),
        image_context_path=args.image_context_path,
        guidance=args.guidance,
        num_steps=args.num_steps,
        control_weight=args.control_weight,
        depth_control_weight=(
            args.depth_control_weight if args.depth_control_weight is not None else args.control_weight
        ),
        edge_control_weight=(
            args.edge_control_weight if args.edge_control_weight is not None else args.control_weight
        ),
        seg_control_weight=(
            args.seg_control_weight if args.seg_control_weight is not None else args.control_weight
        ),
        vis_control_weight=(
            args.vis_control_weight if args.vis_control_weight is not None else args.control_weight
        ),
        disable_guardrails=args.disable_guardrails,
        skip_if_output_exists=(not args.force_infer),
    )

    if args.arm_swap_workdir_write:
        args.arm_swap_no_workdir_write = False

    arm_swap = ArmSwapConfig(
        enable=args.enable_arm_swap,
        no_workdir_write=args.arm_swap_no_workdir_write,
        workdir_write=args.arm_swap_workdir_write,
        qwen_prompt=args.arm_swap_prompt,
        erode=args.arm_swap_erode,
        feather=args.arm_swap_feather,
        dilate=args.arm_swap_dilate,
        split_run=(args.arm_swap_split and not (args.arm_swap_qwen_only or args.arm_swap_sam_only)),
        cleanup_bbox_json=(not args.arm_swap_keep_bbox_json),
        qwen_only=args.arm_swap_qwen_only,
        sam_only=args.arm_swap_sam_only,
        bbox_json=(Path(args.arm_swap_bbox_json) if args.arm_swap_bbox_json else None),
    )

    cache = CacheConfig(
        enable=(not args.no_cache_control),
        force_regen=args.force_regen_control,
        fingerprint_mode=args.fingerprint_mode,
        lock_timeout_sec=args.lock_timeout_sec,
        stale_lock_sec=args.stale_lock_sec,
    )

    cleanup = CleanupConfig(
        enable=(not args.no_cleanup),
        delete_prompt=(not args.keep_prompt),
        delete_json=(not args.keep_json),
        delete_log=args.cleanup_logs,
        delete_non_cached_control_dir=(not args.keep_control_dir),
    )

    run = RunConfig(
        control_type=args.control_type,
        gpus=parse_gpus(args.gpus),
        shuffle=args.shuffle,
        num_videos=args.num_videos,
    )

    return GlobalConfig(
        repos=repos,
        prompt=prompt,
        depth=depth,
        seg=seg,
        transfer=transfer,
        arm_swap=arm_swap,
        cache=cache,
        cleanup=cleanup,
        run=run,
        cv=cv,
        preprocess=preprocess,
        postprocess=postprocess,
    )


def worker_loop(gpu_id: str, q: mp.Queue, cfg: GlobalConfig) -> None:
    """One GPU == one worker process. Each worker runs Step2->Step5 for jobs it pulls."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["AUTO_TRANSFER_RIFE_GPU_ID"] = str(gpu_id)
    # Also update the process environment so console logs include the GPU tag.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["AUTO_TRANSFER_RIFE_GPU_ID"] = str(gpu_id)

    while True:
        item = q.get()
        if item is None:
            return

        job = Job(
            video_path=Path(item["video_path"]),
            job_id=item["job_id"],
            variant_idx=item.get("variant_idx", 0),
            job_index=item.get("job_index", 0),
            input_root=Path(item["input_root"]),
        )
        try:
            run_job(job, cfg, env)
        except Exception:
            # Per-job failure does not stop the whole batch.
            continue


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="run_batch.py",
        description="Cosmos-Transfer2.5 batch automation (multi-GPU, cached depth/seg/edge/blur control).",
        epilog=(
            "Examples:\n"
            "  # Batch a whole folder\n"
            "  python scripts/run_batch.py --input_dir /path/to/mp4s --gpus 0 --control_type depth\n\n"
            "  # Repeat each video K times (K variants per video)\n"
            "  python scripts/run_batch.py --input_dir /path/to/mp4s --gpus 0,1,2,3 --variants_per_video 5 --console\n\n"
            "Notes:\n"
            "  - Each variant regenerates prompt and runs inference again.\n"
            "  - Depth control is cached, so repeated variants of the same video will reuse it.\n"
            "  - Mapping is written to <work_dir>/manifest.jsonl\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Basic inputs
    ap.add_argument("--input_dir", required=True, help="Directory that contains source .mp4 files (recursive)")
    ap.add_argument("--num_videos", type=int, default=0, help="How many unique input videos to process; 0 = all")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle videos before selecting the first N")
    ap.add_argument("--shuffle_seed", type=int, default=None, help="Seed for shuffle to keep selection deterministic")
    ap.add_argument(
        "--selection_file",
        type=str,
        default=None,
        help="Optional JSON file to persist/reuse selected videos (default: <work_dir>/selection_<hash>.json)",
    )
    ap.add_argument(
        "--selective_task",
        type=int,
        nargs="?",
        const=3,
        default=0,
        metavar="N",
        help="Select videos by tasks.parquet content; keep at most N videos per task (default: 3)",
    )
    ap.add_argument("--gpus", type=str, default="", help="GPU list, e.g. 0,1,2,3; empty = 0")
    ap.add_argument("--console", action="store_true", help="Print high-level progress to console (stdout)")

    # Repeat each video K times
    ap.add_argument("--variants_per_video", type=int, default=1,
                    help="Repeat each input video K times to generate K different outputs (default: 1)")

    # Paths (all configurable via CLI)
    ap.add_argument("--work_dir", type=str, default=str(DEFAULT_WORK_DIR), help="Work/output directory")
    ap.add_argument("--autoprompt_root", type=str, default=str(DEFAULT_AUTOPROMPT_ROOT), help="AutoPrompt repo path")
    ap.add_argument("--depth_root", type=str, default=str(DEFAULT_DEPTH_ROOT), help="Video-Depth-Anything repo path")
    ap.add_argument("--seg_root", type=str, default=str(DEFAULT_SEG_ROOT), help="AutoSeg-SAM2 repo path")
    ap.add_argument("--cv_root", type=str, default=str(DEFAULT_CV_ROOT), help="Path containing gen_controls_cv_mp4.py")
    ap.add_argument("--transfer_root", type=str, default=str(DEFAULT_TRANSFER_ROOT), help="Cosmos-Transfer2.5 repo path (contains .venv/)")
    ap.add_argument("--hf_cache", type=str, default=str(DEFAULT_HF_CACHE), help="HuggingFace cache root")

    # Control type (depth/seg/edge/blur/combo implemented)
    ap.add_argument("--control_type", type=str, default="depth", choices=["depth", "edge", "blur", "seg", "combo"],
                    help="Control type; depth/seg/edge/blur/combo are implemented")

    # Step2 (prompt)
    ap.add_argument("--qwen_env", type=str, default=PROMPT_DEFAULTS.conda_env, help="Conda env for Step2 (AutoPrompt)")
    ap.add_argument("--prompt_fps", type=float, default=PROMPT_DEFAULTS.fps, help="FPS for robot_video_to_prompt.py")
    ap.add_argument("--prompt_size", type=str, default=PROMPT_DEFAULTS.size, help="Model size argument passed to robot_video_to_prompt.py")
    ap.add_argument("--prompt_variant", type=str, default=PROMPT_DEFAULTS.variant, help="Variant argument passed to robot_video_to_prompt.py")
    ap.add_argument(
        "--prompt_model_type",
        type=str,
        default=PROMPT_DEFAULTS.model_type,
        choices=["Qwen3", "Qwen2.5"],
        help="Model family for robot_video_to_prompt.py (Qwen3 or Qwen2.5)",
    )
    ap.add_argument("--reuse_prompt", action="store_true", help="Reuse prompt file if it already exists (default: no)")
    ap.add_argument(
        "--sim2real_only",
        action="store_true",
        help="Sim2Real task: only run step1_base and skip bracket/replacement steps",
    )
    ap.add_argument(
        "--min_terms",
        type=int,
        default=PROMPT_DEFAULTS.min_replace_terms,
        help="Minimum number of variable terms to replace in the prompt (default: 1)",
    )
    ap.add_argument(
        "--max_terms",
        type=int,
        default=PROMPT_DEFAULTS.max_replace_terms,
        help="Maximum number of variable terms to replace in the prompt (default: 3)",
    )
    ap.add_argument(
        "--replace_all_terms",
        action="store_true",
        help="Replace all bracketed terms (ignore --min_terms/--max_terms limits)",
    )
    ap.add_argument(
        "--replacements_per_term",
        type=int,
        default=PROMPT_DEFAULTS.replacements_per_term,
        help="How many alternative descriptors to generate for each bracketed term (default: 6)",
    )
    ap.add_argument(
        "--interactive_replace",
        action="store_true",
        help="Interactively choose bracket terms and replacements instead of auto-generating candidates",
    )

    # Step3 (depth)
    ap.add_argument("--depth_env", type=str, default=DEPTH_DEFAULTS.conda_env, help="Conda env for Step3 (Depth Anything)")
    ap.add_argument("--depth_encoder", type=str, default=DEPTH_DEFAULTS.encoder, help="Encoder for depth model (e.g., vitb)")
    ap.add_argument("--no_depth_grayscale", action="store_true", help="Disable --grayscale for depth generation")

    # Step3 (seg)
    ap.add_argument("--seg_env", type=str, default=SEG_DEFAULTS.conda_env, help="Conda env for Step3 (Seg, AutoSeg-SAM2)")
    ap.add_argument("--seg_level", type=str, default=SEG_DEFAULTS.level, choices=["small", "middle", "large"],
                    help="Seg level passed to auto-mask-batch.py (small/middle/large)")
    ap.add_argument("--seg_use_other_level", action="store_true", help="Enable AutoSeg-SAM2 --use_other_level (default: off)")
    ap.add_argument("--seg_detect_stride", type=int, default=SEG_DEFAULTS.detect_stride, help="Stride for detection in auto-mask-batch.py")
    ap.add_argument("--seg_save_intermediate", action="store_true", help="Keep intermediate seg outputs (default: off)")
    ap.add_argument("--seg_save_npy", action="store_true", help="Save npy outputs for seg (default: off)")
    ap.add_argument("--seg_enable_highlight", action="store_true", help="Enable highlight overlay in seg visualization")
    ap.add_argument("--seg_disable_overlay", action="store_true", help="Disable overlay in seg visualization (default: overlay on)")
    ap.add_argument("--seg_overlay_style", type=str, default=SEG_DEFAULTS.overlay_style,
                    help="Overlay style for seg visualization (passed to visulization.py)")

    # Step3 (edge/blur)
    ap.add_argument("--cv_env", type=str, default=CV_DEFAULTS.conda_env, help="Conda env for Step3 (edge/blur, sam)")
    ap.add_argument("--blur_sigma", type=float, default=CV_DEFAULTS.blur_sigma, help="Gaussian blur sigma for blur control")
    ap.add_argument("--edge_low", type=int, default=CV_DEFAULTS.canny_low, help="Canny low threshold for edge control")
    ap.add_argument("--edge_high", type=int, default=CV_DEFAULTS.canny_high, help="Canny high threshold for edge control")

    # Step0 (preprocess)
    ap.add_argument("--pre_width", type=int, default=PREPROCESS_DEFAULTS.width,
                    help="Preprocess output width (default: keep original)")
    ap.add_argument("--pre_height", type=int, default=PREPROCESS_DEFAULTS.height,
                    help="Preprocess output height (default: keep original)")
    ap.add_argument("--pre_no_keep_aspect", action="store_true",
                    help="Disable aspect-ratio preservation in preprocessing (default: keep aspect)")
    ap.add_argument(
        "--pre_chunk_mode",
        type=str,
        default=PREPROCESS_DEFAULTS.pre_chunk_mode,
        choices=["none", "clip", "speed"],
        help="Align preprocess video to Transfer chunk size; none|clip|speed (speed=uniform frame drop, lower fps)",
    )
    ap.add_argument(
        "--pre_chunk_policy",
        type=str,
        default=PREPROCESS_DEFAULTS.pre_chunk_policy,
        choices=["down", "up"],
        help="Chunk alignment policy; down uses fewer chunks, up uses more (default: down)",
    )
    ap.add_argument(
        "--pre_chunk_clip_where",
        type=str,
        default=PREPROCESS_DEFAULTS.pre_chunk_clip_where,
        choices=["tail", "head", "center"],
        help="Clip strategy for pre_chunk_mode=clip (default: tail)",
    )
    ap.add_argument(
        "--pre_chunk_max_adjust",
        type=int,
        default=PREPROCESS_DEFAULTS.pre_chunk_max_adjust,
        help="Skip pre-chunk alignment when |T-T_target| exceeds this value (default: 40)",
    )
    ap.add_argument(
        "--pre_chunk_force_chunks",
        type=int,
        default=PREPROCESS_DEFAULTS.pre_chunk_force_chunks,
        help="Force chunk count (uses speed mode, uniform frame drop, ignores clip/policy options)",
    )
    ap.add_argument(
        "--keyframe_sampling",
        type=str,
        default=PREPROCESS_DEFAULTS.keyframe_sampling,
        choices=["uniform", "parquet_motion"],
        help="Keyframe sampling mode during preprocess: uniform (default) or parquet_motion",
    )
    ap.add_argument(
        "--keyframe_target_frames",
        type=int,
        default=PREPROCESS_DEFAULTS.keyframe_target_frames,
        help="Target keyframe count when keyframe_sampling is enabled (default: num_video_frames_per_chunk)",
    )
    ap.add_argument(
        "--keyframe_smooth_win",
        type=int,
        default=PREPROCESS_DEFAULTS.keyframe_smooth_win,
        help="Moving average window for parquet motion smoothing (default: 5)",
    )
    ap.add_argument(
        "--keyframe_max_gap",
        type=int,
        default=PREPROCESS_DEFAULTS.keyframe_max_gap,
        help="Max gap between sampled frames for parquet motion sampling (default: 8)",
    )
    ap.add_argument(
        "--parquet_pose_subset",
        type=str,
        default=PREPROCESS_DEFAULTS.parquet_pose_subset,
        choices=["zarm_pos", "zarm_pos_vel", "zarm_pos_vel_hand", "full"],
        help="Pose subset for parquet_motion diffs (default: zarm_pos_vel)",
    )
    ap.add_argument(
        "--proxy_preflight",
        action="store_true",
        default=PREPROCESS_DEFAULTS.proxy_preflight_enable,
        help="Enable STEP1.5.5 proxy IoU preflight for keyframe reallocation (default: off)",
    )
    ap.add_argument(
        "--proxy_iou_threshold",
        type=float,
        default=PREPROCESS_DEFAULTS.proxy_iou_threshold,
        help="IoU threshold for proxy preflight bad frames (default: 0.90)",
    )
    ap.add_argument(
        "--proxy_iou_high_threshold",
        type=float,
        default=PREPROCESS_DEFAULTS.proxy_iou_high_threshold,
        help="High IoU threshold for removal candidates (default: 0.98)",
    )
    ap.add_argument(
        "--proxy_motion_low_quantile",
        type=float,
        default=PREPROCESS_DEFAULTS.proxy_motion_low_quantile,
        help="Motion quantile for low-motion removal candidates (default: 0.25)",
    )
    ap.add_argument(
        "--proxy_max_iters",
        type=int,
        default=PREPROCESS_DEFAULTS.proxy_max_iters,
        help="Max iterations for proxy preflight (default: 3)",
    )
    ap.add_argument(
        "--proxy_restore_mode",
        type=str,
        default=PREPROCESS_DEFAULTS.proxy_restore_mode,
        choices=["speed", "minterpolate", "rife"],
        help="Restore mode for proxy preflight (default: rife)",
    )
    ap.add_argument(
        "--proxy_rife_backend",
        type=str,
        default=PREPROCESS_DEFAULTS.proxy_rife_backend,
        choices=["ncnn_vulkan", "pytorch"],
        help="Override RIFE backend for proxy preflight (default: use --rife_backend)",
    )
    ap.add_argument(
        "--proxy_preflight_cache_csv",
        action="store_true",
        default=PREPROCESS_DEFAULTS.proxy_preflight_cache_csv,
        help="Cache proxy preflight IoU results per iteration to CSV in prepro mp4 root (default: off)",
    )
    ap.add_argument(
        "--proxy_removal_strategy",
        type=str,
        default=PREPROCESS_DEFAULTS.proxy_removal_strategy,
        choices=["ranked", "bucketed"],
        help="Removal strategy for proxy preflight (default: ranked)",
    )
    ap.add_argument(
        "--proxy_removal_buckets",
        type=int,
        default=PREPROCESS_DEFAULTS.proxy_removal_buckets,
        help="Bucket count for bucketed proxy removals (default: 6)",
    )

    # Step5.5 post-restore (duration)
    ap.add_argument(
        "--post_restore_duration",
        action="store_true",
        help="Restore Transfer output to original duration",
    )
    ap.add_argument(
        "--post_restore_mode",
        type=str,
        default=POSTPROCESS_DEFAULTS.restore_mode,
        choices=["speed", "minterpolate", "rife"],
        help="Post-restore mode: speed (setpts+fps), minterpolate (smoother, slower), or rife (RIFE interpolation)",
    )
    ap.add_argument(
        "--rife_bin",
        type=str,
        default=POSTPROCESS_DEFAULTS.rife_bin,
        help="RIFE binary path (used when --post_restore_mode=rife)",
    )
    ap.add_argument(
        "--rife_model",
        type=str,
        default=POSTPROCESS_DEFAULTS.rife_model,
        help="RIFE model directory (optional; used when --post_restore_mode=rife)",
    )
    ap.add_argument(
        "--rife_backend",
        type=str,
        default=POSTPROCESS_DEFAULTS.rife_backend,
        choices=["ncnn_vulkan", "pytorch"],
        help="RIFE backend: ncnn_vulkan (default) or pytorch (inference_video.py)",
    )
    ap.add_argument(
        "--rife_pytorch_repo",
        type=str,
        default=POSTPROCESS_DEFAULTS.rife_pytorch_repo,
        help="Practical-RIFE repo path (required when --rife_backend=pytorch)",
    )
    ap.add_argument(
        "--rife_pytorch_model_dir",
        type=str,
        default=POSTPROCESS_DEFAULTS.rife_pytorch_model_dir,
        help="Practical-RIFE model dir (default: <rife_pytorch_repo>/train_log)",
    )
    ap.add_argument(
        "--rife_pytorch_env",
        type=str,
        default=POSTPROCESS_DEFAULTS.rife_pytorch_env,
        help="Conda env for Practical-RIFE (default: rife)",
    )
    ap.add_argument(
        "--rife_scale",
        type=float,
        default=POSTPROCESS_DEFAULTS.rife_scale,
        help="RIFE scale (1.0 default; 0.5 can be faster)",
    )
    ap.add_argument(
        "--rife_keep_audio",
        action="store_true",
        help="Keep input audio when using RIFE post-restore (default: false)",
    )
    ap.add_argument(
        "--rife_time_steps_mode",
        type=str,
        default=POSTPROCESS_DEFAULTS.rife_time_steps_mode,
        choices=["uniform", "zarm_pos", "sam", "sam_zarm_hybrid", "sam_cache", "sam_iou"],
        help="Guided interpolation time steps mode for RIFE pytorch (default: sam_zarm_hybrid)",
    )
    ap.add_argument(
        "--rife_time_steps_cache",
        type=str,
        default=POSTPROCESS_DEFAULTS.rife_time_steps_cache,
        help="Cache path for guided time_steps (default: <work_dir>/rife_cache/time_steps.json)",
    )
    ap.add_argument(
        "--rife_sam_mask_dir",
        type=str,
        default=POSTPROCESS_DEFAULTS.rife_sam_mask_dir,
        help="SAM mask directory for guided time_steps (default: <work_dir>/rife_sam_cache)",
    )
    ap.add_argument(
        "--rife_time_steps_force_sam",
        action="store_true",
        help="Force SAM-based time_steps (no fallback to zarm/uniform)",
    )
    ap.add_argument(
        "--rife_sam_generate",
        action="store_true",
        help="Generate SAM masks via robot_arm_swap when missing (requires --rife_sam_bbox_json)",
    )
    ap.add_argument(
        "--rife_sam_bbox_json",
        type=str,
        default=POSTPROCESS_DEFAULTS.rife_sam_bbox_json,
        help="BBox json path for SAM mask generation (from Qwen)",
    )
    ap.add_argument(
        "--rife_sam_keep_workdir",
        action="store_true",
        help="Keep SAM generation workdir (default: cleanup after run)",
    )
    ap.add_argument(
        "--rife_pose_subset",
        type=str,
        default=POSTPROCESS_DEFAULTS.rife_pose_subset,
        choices=["zarm_pos", "zarm_pos_vel", "zarm_pos_vel_hand", "full"],
        help="Pose subset for guided time_steps (default: zarm_pos)",
    )
    ap.add_argument(
        "--rife_time_steps_debug",
        action="store_true",
        help="Log per-segment time_steps stats and SAM reliability ratios",
    )

    # Step4/5 parameters
    ap.add_argument("--seed", type=int, default=TRANSFER_DEFAULTS.seed, help="Seed in json spec (default: 2025)")
    ap.add_argument("--num_conditional_frames", type=int, default=TRANSFER_DEFAULTS.num_conditional_frames,
                    help="num_conditional_frames in json spec (default: 1)")
    ap.add_argument(
        "--resolution",
        type=str,
        default=TRANSFER_DEFAULTS.resolution,
        help=(
            "resolution key in json spec for Transfer2.5 buckets "
            f"(choices: {', '.join(RESOLUTION_CHOICES)}; default: 720)"
        ),
    )
    ap.add_argument("--sigma_max", type=float, default=TRANSFER_DEFAULTS.sigma_max,
                    help="sigma_max in json spec (default: None)")
    ap.add_argument("--num_video_frames_per_chunk", type=int, default=TRANSFER_DEFAULTS.num_video_frames_per_chunk,
                    help="num_video_frames_per_chunk in json spec (default: 93)")
    ap.add_argument("--show_control_condition", action="store_true",
                    help="Set show_control_condition=true in json spec (default: false)")
    ap.add_argument("--show_input", action="store_true", help="Set show_input=true in json spec (default: false)")
    ap.add_argument("--disable_keep_input_resolution", action="store_true",
                    help="Set keep_input_resolution=false in json spec (default: true)")
    ap.add_argument("--image_context_path", type=str, default=TRANSFER_DEFAULTS.image_context_path,
                    help="image_context_path in json spec (default: null)")
    ap.add_argument("--guidance", type=float, default=TRANSFER_DEFAULTS.guidance, help="guidance value in json spec")
    ap.add_argument("--num_steps", type=int, default=TRANSFER_DEFAULTS.num_steps, help="num_steps in json spec (default: 35)")
    ap.add_argument("--control_weight", type=float, default=TRANSFER_DEFAULTS.control_weight, help="control_weight in json spec")
    ap.add_argument(
        "--disable_guardrails",
        action="store_true",
        help="Pass --disable_guardrails to Cosmos-Transfer2.5 inference",
    )
    ap.add_argument(
        "--depth_control_weight",
        type=float,
        default=None,
        help="depth control_weight override (default: --control_weight)",
    )
    ap.add_argument(
        "--edge_control_weight",
        type=float,
        default=None,
        help="edge control_weight override (default: --control_weight)",
    )
    ap.add_argument(
        "--seg_control_weight",
        type=float,
        default=None,
        help="seg control_weight override (default: --control_weight)",
    )
    ap.add_argument(
        "--vis_control_weight",
        type=float,
        default=None,
        help="vis(blur) control_weight override (default: --control_weight)",
    )
    ap.add_argument("--force_infer", action="store_true", help="Force inference even if output directory is not empty")

    # Step6: robot arm swap (qwen3vl + sam3)
    ap.add_argument("--enable_arm_swap", action="store_true",
                    help="Enable robot arm swap after transfer (qwen3vl + sam3)")
    ap.add_argument(
        "--arm_swap_no_workdir_write",
        action="store_true",
        default=ARM_SWAP_DEFAULTS.no_workdir_write,
        help="Use in-memory arm swap pipeline (avoid PNG workdir writes, default: enabled)",
    )
    ap.add_argument(
        "--arm_swap_workdir_write",
        action="store_true",
        default=ARM_SWAP_DEFAULTS.workdir_write,
        help="Enable swap-only workdir writes (skin masks + IoU outputs)",
    )
    ap.add_argument("--arm_swap_prompt", type=str, default=ARM_SWAP_DEFAULTS.qwen_prompt,
                    help="Qwen prompt for robot arm swap (default: 'robotic arm')")
    ap.add_argument("--arm_swap_erode", type=int, default=ARM_SWAP_DEFAULTS.erode,
                    help="Erode size for arm swap mask (default: 2)")
    ap.add_argument("--arm_swap_feather", type=int, default=ARM_SWAP_DEFAULTS.feather,
                    help="Feather size for arm swap mask (default: 3)")
    ap.add_argument("--arm_swap_dilate", type=int, default=ARM_SWAP_DEFAULTS.dilate,
                    help="Dilate size for arm swap mask (default: 0)")
    ap.add_argument(
        "--arm_swap_split",
        action="store_true",
        default=ARM_SWAP_DEFAULTS.split_run,
        help="Run arm swap in two passes (Qwen-only then SAM-only, default: enabled)",
    )
    ap.add_argument(
        "--arm_swap_single",
        action="store_false",
        dest="arm_swap_split",
        help="Run arm swap in a single process (disable split mode)",
    )
    ap.add_argument(
        "--arm_swap_qwen_only",
        action="store_true",
        help="Run arm swap in Qwen-only mode to emit bbox JSON",
    )
    ap.add_argument(
        "--arm_swap_sam_only",
        action="store_true",
        help="Run arm swap in SAM-only mode (requires --arm_swap_bbox_json)",
    )
    ap.add_argument(
        "--arm_swap_bbox_json",
        type=str,
        default=None,
        help="Path to bbox JSON for arm swap (output for Qwen-only or input for SAM-only)",
    )
    ap.add_argument(
        "--arm_swap_keep_bbox_json",
        action="store_true",
        help="Keep arm swap bbox JSON after split run completes (default: delete)",
    )

    # Depth cache
    ap.add_argument("--no_cache_control", action="store_true", help="Disable depth control cache (default: enabled)")
    ap.add_argument("--force_regen_control", action="store_true", help="Force regenerate depth control (ignore cache)")
    ap.add_argument("--fingerprint_mode", type=str, default=CACHE_DEFAULTS.fingerprint_mode, choices=["path", "stat", "md5"],
                    help="Fingerprint mode for cache key: stat (fast), md5 (strong), path (weak)")
    ap.add_argument("--lock_timeout_sec", type=int, default=CACHE_DEFAULTS.lock_timeout_sec, help="Lock wait timeout for cache generation")
    ap.add_argument("--stale_lock_sec", type=int, default=CACHE_DEFAULTS.stale_lock_sec, help="Treat lock as stale after this many seconds")

    # Cleanup intermediates after successful jobs
    ap.add_argument("--no_cleanup", action="store_true",
                    help="Disable cleanup of intermediates after successful inference (default: cleanup enabled)")
    ap.add_argument("--keep_prompt", action="store_true", help="Do NOT delete per-job prompt files after success")
    ap.add_argument("--keep_json", action="store_true", help="Do NOT delete per-job json spec files after success")
    ap.add_argument("--keep_control_dir", action="store_true", help="Do NOT delete per-job non-cached control dir after success")
    ap.add_argument("--cleanup_logs", action="store_true", help="Also delete per-job log files after success")

    args = ap.parse_args(argv)

    # Enable console printing across the project via an environment variable.
    if args.console:
        os.environ["ATR_CONSOLE"] = "1"

    if args.variants_per_video < 1:
        raise ValueError("--variants_per_video must be >= 1")
    if args.min_terms < 1:
        raise ValueError("--min_terms must be >= 1")
    if args.max_terms < args.min_terms:
        raise ValueError("--max_terms must be >= --min_terms")
    if args.replacements_per_term < 1:
        raise ValueError("--replacements_per_term must be >= 1")
    if args.pre_width is not None and args.pre_width < 1:
        raise ValueError("--pre_width must be >= 1")
    if args.pre_height is not None and args.pre_height < 1:
        raise ValueError("--pre_height must be >= 1")
    if args.pre_chunk_force_chunks is not None and args.pre_chunk_force_chunks < 1:
        raise ValueError("--pre_chunk_force_chunks must be >= 1")
    if args.post_restore_mode not in ["speed", "minterpolate", "rife"]:
        raise ValueError("--post_restore_mode must be speed/minterpolate/rife")
    if args.rife_scale <= 0:
        raise ValueError("--rife_scale must be > 0")
    if args.post_restore_mode == "rife" and args.rife_backend == "pytorch":
        if not args.rife_pytorch_repo:
            raise ValueError("--rife_pytorch_repo is required when --rife_backend=pytorch")
    if args.resolution not in RESOLUTION_CHOICES:
        raise ValueError(f"--resolution must be one of: {', '.join(RESOLUTION_CHOICES)}")
    if args.arm_swap_qwen_only and args.arm_swap_sam_only:
        raise ValueError("--arm_swap_qwen_only and --arm_swap_sam_only are mutually exclusive")
    if args.arm_swap_sam_only and not args.arm_swap_bbox_json:
        raise ValueError("--arm_swap_bbox_json is required when --arm_swap_sam_only is set")

    cfg = build_config(args)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    cfg.repos.work_dir.mkdir(parents=True, exist_ok=True)

    default_selection_name = selection_file_name(
        input_dir,
        args.num_videos,
        args.shuffle,
        args.shuffle_seed,
        args.selective_task,
    )
    selection_path = (
        Path(args.selection_file)
        if args.selection_file
        else cfg.repos.work_dir / default_selection_name
    )
    if selection_path.exists():
        with selection_path.open("r", encoding="utf-8") as f:
            selection = json.load(f)
        expected = {
            "input_dir": str(input_dir.resolve()),
            "num_videos": args.num_videos,
            "shuffle": args.shuffle,
            "shuffle_seed": args.shuffle_seed,
            "selective_task": args.selective_task,
        }
        mismatches = {
            key: (selection.get(key), value)
            for key, value in expected.items()
            if selection.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "selection_file settings mismatch: "
                + ", ".join(
                    f"{key}={found!r} != {expected_value!r}"
                    for key, (found, expected_value) in mismatches.items()
                )
                + ". Delete the selection file or pick a new --selection_file."
            )
        videos = [Path(p) for p in selection.get("videos", [])]
    else:
        videos = pick_videos(
            input_dir,
            args.num_videos,
            args.shuffle,
            args.selective_task,
            args.shuffle_seed,
        )
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        with selection_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "input_dir": str(input_dir.resolve()),
                    "num_videos": args.num_videos,
                    "shuffle": args.shuffle,
                    "shuffle_seed": args.shuffle_seed,
                    "selective_task": args.selective_task,
                    "videos": [str(vp) for vp in videos],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    if not videos:
        raise RuntimeError("No mp4 files found in input_dir.")

    # Build queue:
    # - First select unique videos (controlled by --num_videos and --shuffle)
    # - Then expand each video into K variants (controlled by --variants_per_video)
    # - Queue is ordered by variant so workers pick different videos first.
    q: mp.Queue = mp.Queue()
    batch_tag = now_tag()
    base_hashes = [short_hash(str(vp.resolve()), 8) for vp in videos]

    job_index = 0
    for v in range(args.variants_per_video):
        for idx, vp in enumerate(videos):
            base_hash = base_hashes[idx]
            job_id = f"{batch_tag}__{idx:04d}__v{v:03d}__{base_hash}"
            q.put(
                {
                    "video_path": str(vp),
                    "job_id": job_id,
                    "variant_idx": v,
                    "job_index": job_index,
                    "input_root": str(input_dir),
                }
            )
            job_index += 1

    # Start one worker per GPU
    procs: List[mp.Process] = []
    for g in (cfg.run.gpus or ["0"]):
        p = mp.Process(target=worker_loop, args=(g, q, cfg), daemon=False)
        p.start()
        procs.append(p)

    # Stop signals
    for _ in procs:
        q.put(None)

    for p in procs:
        p.join()

    expected_jobs = len(videos) * args.variants_per_video
    output_root = cfg.repos.work_dir / "output_video"
    checkpoint_root = cfg.repos.work_dir / CHECKPOINT_DIRNAME
    job_checkpoint_dirs = []
    completed_jobs = 0
    for v in range(args.variants_per_video):
        for vp in videos:
            output_dir = make_output_dir(
                output_root,
                input_root=input_dir,
                video_path=vp,
                variant_idx=v,
            )
            checkpoint_dir = checkpoint_dir_for_job(cfg.repos.work_dir, output_dir)
            job_checkpoint_dirs.append(checkpoint_dir)
            if job_done_marker_path(checkpoint_dir).exists():
                completed_jobs += 1

    if expected_jobs > 0 and completed_jobs == expected_jobs:
        for checkpoint_dir in job_checkpoint_dirs:
            clear_checkpoint_dir(checkpoint_dir, root=checkpoint_root)

    print(format_console_line("All jobs finished."))
    print(format_console_line(f"work_dir={cfg.repos.work_dir}"))
    print(format_console_line("Outputs: output_video/, control_cache/, logs/ (optional), manifest.jsonl"))
