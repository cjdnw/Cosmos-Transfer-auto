from __future__ import annotations

import argparse
import datetime as dt
import multiprocessing as mp
import os
import random
from pathlib import Path
from typing import List

from .config import (
    DEFAULT_AUTOPROMPT_ROOT,
    DEFAULT_DEPTH_ROOT,
    DEFAULT_SEG_ROOT,
    DEFAULT_CV_ROOT,
    DEFAULT_HF_CACHE,
    DEFAULT_TRANSFER_ROOT,
    DEFAULT_WORK_DIR,
    DEFAULT_ARM_SWAP_ROOT,
    DEFAULT_ARM_SWAP_OUTPUT_ROOT,
    DEFAULT_ARM_SWAP_WORKDIR_ROOT,
    CACHE_DEFAULTS,
    DEPTH_DEFAULTS,
    SEG_DEFAULTS,
    CV_DEFAULTS,
    PROMPT_DEFAULTS,
    TRANSFER_DEFAULTS,
    ARM_SWAP_DEFAULTS,
    RepoPaths,
    PromptConfig,
    DepthConfig,
    SegConfig,
    CvControlConfig,
    TransferConfig,
    ArmSwapConfig,
    CacheConfig,
    CleanupConfig,
    RunConfig,
    GlobalConfig,
)
from .pipeline import Job, run_job
from .utils.hashing import short_hash
from .utils.logging_utils import format_console_line


def now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def pick_videos(input_dir: Path, num_videos: int, shuffle: bool) -> List[Path]:
    """Select unique input videos from a folder (non-recursive)."""
    videos = sorted([p for p in input_dir.glob("*.mp4") if p.is_file()])
    if shuffle:
        random.shuffle(videos)
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
        skip_if_output_exists=(not args.force_infer),
    )

    arm_swap = ArmSwapConfig(
        enable=args.enable_arm_swap,
        project_root=Path(args.arm_swap_root),
        conda_env=args.arm_swap_env,
        output_root=Path(args.arm_swap_output_root),
        workdir_root=Path(args.arm_swap_workdir_root),
        fps=args.arm_swap_fps,
        width=args.arm_swap_width,
        qwen_prompt=args.arm_swap_prompt,
        skip_if_output_exists=(not args.arm_swap_force),
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
    )


def worker_loop(gpu_id: str, q: mp.Queue, cfg: GlobalConfig) -> None:
    """One GPU == one worker process. Each worker runs Step2->Step5 for jobs it pulls."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Also update the process environment so console logs include the GPU tag.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    while True:
        item = q.get()
        if item is None:
            return

        job = Job(
            video_path=Path(item["video_path"]),
            job_id=item["job_id"],
            variant_idx=item.get("variant_idx", 0),
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
    ap.add_argument("--input_dir", required=True, help="Directory that contains source .mp4 files")
    ap.add_argument("--num_videos", type=int, default=0, help="How many unique input videos to process; 0 = all")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle videos before selecting the first N")
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

    # Step4/5 parameters
    ap.add_argument("--seed", type=int, default=TRANSFER_DEFAULTS.seed, help="Seed in json spec (default: 2025)")
    ap.add_argument("--num_conditional_frames", type=int, default=TRANSFER_DEFAULTS.num_conditional_frames,
                    help="num_conditional_frames in json spec (default: 1)")
    ap.add_argument("--resolution", type=str, default=TRANSFER_DEFAULTS.resolution, help="resolution in json spec (default: 720)")
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
    ap.add_argument("--arm_swap_root", type=str, default=str(DEFAULT_ARM_SWAP_ROOT),
                    help="Robot arm swap project root (contains robot_arm_swap module)")
    ap.add_argument("--arm_swap_env", type=str, default=ARM_SWAP_DEFAULTS.conda_env,
                    help="Conda env for robot arm swap (default: robot_arm_swap)")
    ap.add_argument("--arm_swap_output_root", type=str, default=str(DEFAULT_ARM_SWAP_OUTPUT_ROOT),
                    help="Output root for arm swap videos")
    ap.add_argument("--arm_swap_workdir_root", type=str, default=str(DEFAULT_ARM_SWAP_WORKDIR_ROOT),
                    help="Workdir root for arm swap cache (per-job dir will be cleaned after output)")
    ap.add_argument("--arm_swap_fps", type=int, default=ARM_SWAP_DEFAULTS.fps,
                    help="FPS for robot arm swap (default: 30)")
    ap.add_argument("--arm_swap_width", type=int, default=ARM_SWAP_DEFAULTS.width,
                    help="Width for robot arm swap (default: 1280)")
    ap.add_argument("--arm_swap_prompt", type=str, default=ARM_SWAP_DEFAULTS.qwen_prompt,
                    help="Qwen prompt for robot arm swap (default: 'robotic arm')")
    ap.add_argument("--arm_swap_force", action="store_true",
                    help="Force rerun arm swap even if output already exists")

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

    cfg = build_config(args)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    cfg.repos.work_dir.mkdir(parents=True, exist_ok=True)

    videos = pick_videos(input_dir, args.num_videos, args.shuffle)
    if not videos:
        raise RuntimeError("No mp4 files found in input_dir.")

    # Build queue:
    # - First select unique videos (controlled by --num_videos and --shuffle)
    # - Then expand each video into K variants (controlled by --variants_per_video)
    # - Queue is ordered by variant so workers pick different videos first.
    q: mp.Queue = mp.Queue()
    batch_tag = now_tag()
    base_hashes = [short_hash(str(vp.resolve()), 8) for vp in videos]

    for v in range(args.variants_per_video):
        for idx, vp in enumerate(videos):
            base_hash = base_hashes[idx]
            job_id = f"{batch_tag}__{idx:04d}__v{v:03d}__{base_hash}"
            q.put({"video_path": str(vp), "job_id": job_id, "variant_idx": v})

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

    print(format_console_line("All jobs finished."))
    print(format_console_line(f"work_dir={cfg.repos.work_dir}"))
    print(format_console_line("Outputs: output_video/, control_cache/, logs/ (optional), manifest.jsonl"))
