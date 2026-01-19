from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


# Repository root (used for defaults)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Default paths (kept consistent with original CLI defaults)
DEFAULT_WORK_DIR = Path("/mnt/data/ganrunhe/auto_transfer")
DEFAULT_AUTOPROMPT_ROOT = REPO_ROOT
DEFAULT_DEPTH_ROOT = Path("/mnt/data/ganrunhe/Video-Depth-Anything")
DEFAULT_SEG_ROOT = Path("/mnt/data/ganrunhe/AutoSeg-SAM2")
DEFAULT_CV_ROOT = REPO_ROOT
DEFAULT_TRANSFER_ROOT = Path("/mnt/data/ganrunhe/cosmos-transfer2.5")
DEFAULT_HF_CACHE = Path("/mnt/data/ganrunhe/hf_cache")
DEFAULT_ARM_SWAP_ROOT = Path("/mnt/data/ganrunhe/robot_arm_swap_project")
DEFAULT_ARM_SWAP_OUTPUT_ROOT = Path("/mnt/data/ganrunhe/auto_transfer/output_video_mask_arm")
DEFAULT_ARM_SWAP_WORKDIR_ROOT = DEFAULT_WORK_DIR / "qwen_sam_cache"


@dataclass(frozen=True)
class RepoPaths:
    """All repo / working directories used by the pipeline."""
    work_dir: Path
    autoprompt_root: Path
    depth_root: Path
    seg_root: Path
    cv_root: Path
    transfer_root: Path
    hf_cache: Path


@dataclass(frozen=True)
class PromptConfig:
    """Step2: prompt generation config."""
    conda_env: str = "qwen3vl"
    fps: float = 4.0
    size: str = "32B"
    variant: str = "instruct"
    model_type: str = "Qwen3"  # Qwen3 | Qwen2.5
    reuse_if_exists: bool = False  # default: regen (prompt is typically stochastic)
    sim2real_only: bool = False
    min_replace_terms: int = 1
    max_replace_terms: int = 3
    replace_all_terms: bool = False
    replacements_per_term: int = 64
    interactive_replace: bool = False


@dataclass(frozen=True)
class DepthConfig:
    """Step3: depth control generation config."""
    conda_env: str = "depthanything"
    encoder: str = "vitb"
    grayscale: bool = True


@dataclass(frozen=True)
class SegConfig:
    """Step3: seg control generation config (AutoSeg-SAM2)."""
    conda_env: str = "autoseg"
    level: str = "large"  # small | middle | large
    use_other_level: bool = False
    detect_stride: int = 10
    save_intermediate: bool = False
    save_npy: bool = False
    enable_highlight: bool = False
    enable_overlay: bool = True
    overlay_style: str = "mask"


@dataclass(frozen=True)
class CvControlConfig:
    """Step3: edge/blur control generation config (gen_controls_cv_mp4.py)."""
    conda_env: str = "sam"
    blur_sigma: float = 10.0
    canny_low: int = 80
    canny_high: int = 160


@dataclass(frozen=True)
class PreprocessConfig:
    """Video preprocessing config (CFR + resize) before inference."""
    width: int | None = None
    height: int | None = None
    keep_aspect: bool = True
    pre_chunk_mode: str = "none"
    pre_chunk_policy: str = "down"
    pre_chunk_clip_where: str = "tail"
    pre_chunk_max_adjust: int = 40
    pre_chunk_force_chunks: int | None = None
    keyframe_sampling: str = "uniform"  # uniform | parquet_motion
    keyframe_target_frames: int | None = None
    keyframe_smooth_win: int = 5
    keyframe_max_gap: int = 8
    parquet_pose_subset: str = "zarm_pos_vel"
    proxy_preflight_enable: bool = False
    proxy_iou_threshold: float = 0.9
    proxy_iou_high_threshold: float = 0.98
    proxy_motion_low_quantile: float = 0.25
    proxy_max_iters: int = 3
    proxy_restore_mode: str = "rife"
    proxy_rife_backend: str | None = None


@dataclass(frozen=True)
class PostprocessConfig:
    """Postprocess config (restore duration after Transfer)."""
    restore_duration: bool = False
    restore_mode: str = "speed"  # speed | minterpolate | rife
    rife_bin: str = "/mnt/data/ganrunhe/rife/rife-ncnn-vulkan-20221029-ubuntu/rife-ncnn-vulkan"
    rife_model: str | None = "/mnt/data/ganrunhe/rife/rife-ncnn-vulkan-20221029-ubuntu/rife-v4.6"
    rife_backend: str = "ncnn_vulkan"  # ncnn_vulkan | pytorch
    rife_pytorch_repo: str | None = "/mnt/data/ganrunhe/Practical-RIFE"
    rife_pytorch_model_dir: str | None = None
    rife_pytorch_env: str = "rife"
    rife_scale: float = 1.0
    rife_keep_audio: bool = False
    rife_time_steps_mode: str = "sam_zarm_hybrid"
    rife_time_steps_cache: str | None = None
    rife_sam_mask_dir: str | None = None
    rife_time_steps_force_sam: bool = False
    rife_sam_generate: bool = False
    rife_sam_bbox_json: str | None = None
    rife_sam_cleanup_workdir: bool = True
    rife_pose_subset: str = "zarm_pos"
    rife_time_steps_debug: bool = False


@dataclass(frozen=True)
class TransferConfig:
    """Step5: Transfer2.5 inference config."""
    seed: int | None = 2025
    negative_prompt: str = (
        "extra objects, foreign objects, hallucinated items, object duplication, "
        "sudden appearance/disappearance, flicker, jitter, wobble, camera shake, "
        "warped geometry, deformed shapes, texture crawling, inconsistent lighting, "
        "sudden dark shadows, overly dark basket interior, moving/warping background, "
        "motion blur, heavy compression artifacts, noise, low resolution, subtitles, "
        "text, watermark, cartoon, CGI"
    )
    num_conditional_frames: int = 1
    resolution: str = "720"
    sigma_max: float | None = None
    num_video_frames_per_chunk: int = 93
    show_control_condition: bool = False
    show_input: bool = False
    keep_input_resolution: bool = True
    image_context_path: str | None = None
    guidance: float = 3.0
    num_steps: int = 35
    control_weight: float = 1.0
    depth_control_weight: float = 1.0
    edge_control_weight: float = 1.0
    seg_control_weight: float = 1.0
    vis_control_weight: float = 1.0
    disable_guardrails: bool = False
    skip_if_output_exists: bool = True


@dataclass(frozen=True)
class ArmSwapConfig:
    """Step6: robot arm swap config (qwen3vl + sam3)."""
    enable: bool = False
    project_root: Path = DEFAULT_ARM_SWAP_ROOT
    conda_env: str = "arm_swap"
    output_root: Path = DEFAULT_ARM_SWAP_OUTPUT_ROOT
    workdir_root: Path = DEFAULT_ARM_SWAP_WORKDIR_ROOT
    no_workdir_write: bool = True
    workdir_write: bool = False
    qwen_prompt: str = "robotic arm"
    erode: int = 2
    feather: int = 3
    dilate: int = 0
    split_run: bool = True
    cleanup_bbox_json: bool = True
    qwen_only: bool = False
    sam_only: bool = False
    bbox_json: Path | None = None


@dataclass(frozen=True)
class CacheConfig:
    """Cache strategy (control signals + base prompt)."""
    enable: bool = True
    force_regen: bool = False
    fingerprint_mode: str = "stat"  # path | stat | md5
    lock_timeout_sec: int = 6 * 3600
    stale_lock_sec: int = 3 * 3600


@dataclass(frozen=True)
class CleanupConfig:
    """Cleanup strategy after successful inference."""
    enable: bool = True
    delete_prompt: bool = True
    delete_json: bool = True
    delete_log: bool = False
    delete_non_cached_control_dir: bool = True


@dataclass(frozen=True)
class RunConfig:
    """Batch run config."""
    control_type: str = "depth"  # depth | edge | blur | seg | combo
    gpus: List[str] | None = None
    shuffle: bool = False
    num_videos: int = 0  # 0=all


@dataclass(frozen=True)
class GlobalConfig:
    """All configs."""
    repos: RepoPaths
    prompt: PromptConfig
    depth: DepthConfig
    seg: SegConfig
    cv: CvControlConfig
    preprocess: PreprocessConfig
    postprocess: PostprocessConfig
    transfer: TransferConfig
    arm_swap: ArmSwapConfig
    cache: CacheConfig
    cleanup: CleanupConfig
    run: RunConfig


# Default config objects (single source of truth for CLI defaults)
PROMPT_DEFAULTS = PromptConfig()
DEPTH_DEFAULTS = DepthConfig()
SEG_DEFAULTS = SegConfig()
CV_DEFAULTS = CvControlConfig()
PREPROCESS_DEFAULTS = PreprocessConfig()
POSTPROCESS_DEFAULTS = PostprocessConfig()
TRANSFER_DEFAULTS = TransferConfig()
ARM_SWAP_DEFAULTS = ArmSwapConfig()
CACHE_DEFAULTS = CacheConfig()
CLEANUP_DEFAULTS = CleanupConfig()
RUN_DEFAULTS = RunConfig(control_type="depth", gpus=None, shuffle=False, num_videos=0)
