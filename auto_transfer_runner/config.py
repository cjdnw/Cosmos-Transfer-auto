from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class RepoPaths:
    """All repo / working directories used by the pipeline."""
    work_dir: Path
    autoprompt_root: Path
    depth_root: Path
    transfer_root: Path
    hf_cache: Path


@dataclass(frozen=True)
class PromptConfig:
    """Step2: prompt generation config."""
    conda_env: str = "qwen25vl"
    fps: float = 4.0
    size: str = "7B"
    variant: str = "instruct"
    reuse_if_exists: bool = True  # 默认命中缓存：同视频直接跳过 Step1 读取


@dataclass(frozen=True)
class DepthConfig:
    """Step3: depth control generation config."""
    conda_env: str = "depthanything"
    encoder: str = "vitb"
    grayscale: bool = True


@dataclass(frozen=True)
class TransferConfig:
    """Step5: Transfer2.5 inference config."""
    guidance: float = 3.0
    num_steps: int = 35
    control_weight: float = 1.0
    skip_if_output_exists: bool = True


@dataclass(frozen=True)
class CacheConfig:
    """Control cache strategy (currently only depth is cached)."""
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
    transfer: TransferConfig
    cache: CacheConfig
    cleanup: CleanupConfig
    run: RunConfig
