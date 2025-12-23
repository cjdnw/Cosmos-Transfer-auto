from __future__ import annotations

import json
from pathlib import Path

from ..config import TransferConfig


def _build_base_spec(
    *,
    name: str,
    prompt_path: Path,
    video_path: Path,
    seed: int,
    transfer_cfg: TransferConfig,
) -> dict:
    return {
        "name": name,
        "prompt_path": str(prompt_path),
        "seed": seed,
        "negative_prompt": transfer_cfg.negative_prompt,
        "guidance": transfer_cfg.guidance,
        "video_path": str(video_path),
        "image_context_path": transfer_cfg.image_context_path,
        "num_conditional_frames": transfer_cfg.num_conditional_frames,
        "resolution": transfer_cfg.resolution,
        "sigma_max": transfer_cfg.sigma_max,
        "num_video_frames_per_chunk": transfer_cfg.num_video_frames_per_chunk,
        "num_steps": transfer_cfg.num_steps,
        "show_control_condition": transfer_cfg.show_control_condition,
        "show_input": transfer_cfg.show_input,
        "keep_input_resolution": transfer_cfg.keep_input_resolution,
        "edge": None,
        "depth": None,
        "vis": None,
        "seg": None,
    }


def write_json_spec_depth(
    *,
    json_path: Path,
    name: str,
    prompt_path: Path,
    video_path: Path,
    seed: int,
    transfer_cfg: TransferConfig,
    control_path: Path,
) -> None:
    """Step4：生成 depth 格式的 json spec。"""
    json_path.parent.mkdir(parents=True, exist_ok=True)

    spec = _build_base_spec(
        name=name,
        prompt_path=prompt_path,
        video_path=video_path,
        seed=seed,
        transfer_cfg=transfer_cfg,
    )
    spec["depth"] = {
        "control_path": str(control_path),
        "control_weight": transfer_cfg.control_weight,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=4)


def write_json_spec_seg(
    *,
    json_path: Path,
    name: str,
    prompt_path: Path,
    video_path: Path,
    seed: int,
    transfer_cfg: TransferConfig,
    control_path: Path,
) -> None:
    """Step4：生成 seg 格式的 json spec。"""
    json_path.parent.mkdir(parents=True, exist_ok=True)

    spec = _build_base_spec(
        name=name,
        prompt_path=prompt_path,
        video_path=video_path,
        seed=seed,
        transfer_cfg=transfer_cfg,
    )
    spec["seg"] = {
        "control_path": str(control_path),
        "control_weight": transfer_cfg.control_weight,
        "mask_path": None,
        "mask_prompt": None,
        "control_prompt": None,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=4)
