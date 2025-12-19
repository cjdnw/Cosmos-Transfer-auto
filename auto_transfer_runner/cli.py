from __future__ import annotations

import argparse
import datetime as dt
import multiprocessing as mp
import os
import random
from pathlib import Path
from typing import List

from .config import (
    RepoPaths,
    PromptConfig,
    DepthConfig,
    TransferConfig,
    CacheConfig,
    CleanupConfig,
    RunConfig,
    GlobalConfig,
)
from .pipeline import Job, run_job
from .utils.hashing import short_hash


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
        transfer_root=Path(args.transfer_root),
        hf_cache=Path(args.hf_cache),
    )

    prompt = PromptConfig(
        conda_env=args.qwen_env,
        fps=args.prompt_fps,
        size=args.prompt_size,
        variant=args.prompt_variant,
        reuse_if_exists=args.reuse_prompt,
    )

    depth = DepthConfig(
        conda_env=args.depth_env,
        encoder=args.depth_encoder,
        grayscale=(not args.no_depth_grayscale),
    )

    transfer = TransferConfig(
        guidance=args.guidance,
        num_steps=args.num_steps,
        control_weight=args.control_weight,
        skip_if_output_exists=(not args.force_infer),
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

    return GlobalConfig(repos=repos, prompt=prompt, depth=depth, transfer=transfer, cache=cache, cleanup=cleanup, run=run)


def worker_loop(gpu_id: str, q: mp.Queue, cfg: GlobalConfig) -> None:
    """One GPU == one worker process. Each worker runs Step2->Step5 for jobs it pulls."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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
        description="Cosmos-Transfer2.5 batch automation (multi-GPU, cached depth control).",
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
    ap.add_argument("--work_dir", type=str, default="/mnt/data/ganrunhe/auto_transfer", help="Work/output directory")
    ap.add_argument("--autoprompt_root", type=str, default="/mnt/data/ganrunhe/AutoPrompt", help="AutoPrompt repo path")
    ap.add_argument("--depth_root", type=str, default="/mnt/data/ganrunhe/Video-Depth-Anything", help="Video-Depth-Anything repo path")
    # Updated default as requested
    ap.add_argument("--transfer_root", type=str, default="/mnt/data/ganrunhe/cosmos-transfer2.5", help="Cosmos-Transfer2.5 repo path (contains .venv/)")
    ap.add_argument("--hf_cache", type=str, default="/mnt/data/ganrunhe/hf_cache", help="HuggingFace cache root")

    # Control type (depth implemented; others reserved)
    ap.add_argument("--control_type", type=str, default="depth", choices=["depth", "edge", "blur", "seg", "combo"],
                    help="Control type; currently only depth is implemented")

    # Step2 (prompt)
    ap.add_argument("--qwen_env", type=str, default="qwen25vl", help="Conda env for Step2 (AutoPrompt)")
    ap.add_argument("--prompt_fps", type=float, default=4.0, help="FPS for robot_video_to_prompt.py")
    ap.add_argument("--prompt_size", type=str, default="7B", help="Model size argument passed to robot_video_to_prompt.py")
    ap.add_argument("--prompt_variant", type=str, default="instruct", help="Variant argument passed to robot_video_to_prompt.py")
    ap.add_argument("--reuse_prompt", dest="reuse_prompt", action="store_true", default=True,
                    help="Reuse cached prompt by fingerprint (default: enabled)")
    ap.add_argument("--no_reuse_prompt", dest="reuse_prompt", action="store_false",
                    help="Force regenerate prompt even if cache exists")

    # Step3 (depth)
    ap.add_argument("--depth_env", type=str, default="depthanything", help="Conda env for Step3 (Depth Anything)")
    ap.add_argument("--depth_encoder", type=str, default="vitb", help="Encoder for depth model (e.g., vitb)")
    ap.add_argument("--no_depth_grayscale", action="store_true", help="Disable --grayscale for depth generation")

    # Step4/5 parameters
    ap.add_argument("--guidance", type=float, default=3.0, help="guidance value in json spec")
    ap.add_argument("--num_steps", type=int, default=35, help="num_steps in json spec (default: 35)")
    ap.add_argument("--control_weight", type=float, default=1.0, help="control_weight in json spec")
    ap.add_argument("--force_infer", action="store_true", help="Force inference even if output directory is not empty")

    # Depth cache
    ap.add_argument("--no_cache_control", action="store_true", help="Disable depth control cache (default: enabled)")
    ap.add_argument("--force_regen_control", action="store_true", help="Force regenerate depth control (ignore cache)")
    ap.add_argument("--fingerprint_mode", type=str, default="stat", choices=["path", "stat", "md5"],
                    help="Fingerprint mode for cache key: stat (fast), md5 (strong), path (weak)")
    ap.add_argument("--lock_timeout_sec", type=int, default=6 * 3600, help="Lock wait timeout for cache generation")
    ap.add_argument("--stale_lock_sec", type=int, default=3 * 3600, help="Treat lock as stale after this many seconds")

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
    q: mp.Queue = mp.Queue()
    batch_tag = now_tag()

    for v in range(args.variants_per_video):
        for idx, vp in enumerate(videos):
            base_hash = short_hash(str(vp.resolve()), 8)
            job_id = f"{batch_tag}__{idx:04d}__v{v:03d}__{base_hash}"
            q.put({"video_path": str(vp), "job_id": job_id, "variant_idx": v})

    # Start one worker per GPU
    gpus = cfg.run.gpus or ["0"]
    worker_count = min(len(gpus), len(videos)) if videos else 0
    procs: List[mp.Process] = []
    for g in gpus[:worker_count]:
        p = mp.Process(target=worker_loop, args=(g, q, cfg), daemon=False)
        p.start()
        procs.append(p)

    # Stop signals
    for _ in procs:
        q.put(None)

    for p in procs:
        p.join()

    print("All jobs finished.")
    print(f"work_dir={cfg.repos.work_dir}")
    print("Outputs: output_video/, control_cache/, logs/ (optional), manifest.jsonl")
