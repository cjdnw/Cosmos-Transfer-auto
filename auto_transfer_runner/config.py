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
DEFAULT_TRANSFER_ROOT = Path("/mnt/data/ganrunhe/cosmos-transfer2.5")
DEFAULT_HF_CACHE = Path("/mnt/data/ganrunhe/hf_cache")


@dataclass(frozen=True)
class RepoPaths:
    """All repo / working directories used by the pipeline."""
    work_dir: Path
    autoprompt_root: Path
    depth_root: Path
    seg_root: Path
    transfer_root: Path
    hf_cache: Path


@dataclass(frozen=True)
class PromptConfig:
    """Step2: prompt generation config."""
    conda_env: str = "qwen3vl"
    fps: float = 4.0
    size: str = "7B"
    variant: str = "instruct"
    model_type: str = "Qwen2.5"  # Qwen3 | Qwen2.5
    reuse_if_exists: bool = False  # default: regen (prompt is typically stochastic)
    min_replace_terms: int = 1
    max_replace_terms: int = 3
    replace_all_terms: bool = False
    replacements_per_term: int = 6


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
class TransferConfig:
    """Step5: Transfer2.5 inference config."""
    seed: int | None = 2025
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
    skip_if_output_exists: bool = True


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
    transfer: TransferConfig
    cache: CacheConfig
    cleanup: CleanupConfig
    run: RunConfig


# Default config objects (single source of truth for CLI defaults)
PROMPT_DEFAULTS = PromptConfig()
DEPTH_DEFAULTS = DepthConfig()
SEG_DEFAULTS = SegConfig()
TRANSFER_DEFAULTS = TransferConfig()
CACHE_DEFAULTS = CacheConfig()
CLEANUP_DEFAULTS = CleanupConfig()
RUN_DEFAULTS = RunConfig(control_type="depth", gpus=None, shuffle=False, num_videos=0)
