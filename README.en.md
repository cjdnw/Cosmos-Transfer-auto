**English | [简体中文](README.md)**

# Cosmos-Transfer2.5 Automated Batch Pipeline

An automation layer around NVIDIA Cosmos-Transfer2.5 that turns a previously manual, multi-step video generation workflow into a reproducible batch pipeline for robotics video generation and sim-to-real data augmentation.

## What it does

Given a batch of raw robot videos, the pipeline runs the full workflow end to end:

1. **Prompt generation (AutoPrompt)** — a vision-language model (Qwen2.5 / Qwen3 VL) describes each video and produces prompts with controlled random texture replacement.
2. **Control-signal generation** — depth (Video-Depth-Anything), segmentation (SAM 2 based AutoSeg), edge and blur (OpenCV) signals, including multi-signal combos. Signals are cached by video fingerprint, so each input video's signals are generated only once.
3. **Spec generation** — per-video JSON specs for Cosmos-Transfer2.5.
4. **Inference** — batched Cosmos-Transfer2.5 runs with multi-GPU scheduling; each GPU processes complete videos in parallel.
5. **Post-processing** — optional RIFE frame interpolation to restore the original duration, and optional robotic-arm replacement (Qwen3-VL + SAM 3).

## Features

- Multi-GPU batch scheduling with per-video parallelism
- Checkpoint / resume via atomic marker files — interrupted runs continue where they left off
- Control-signal caching and reuse across runs
- Multiple output variants per input video (K samples with different replacements)
- Task-based sampling from `tasks.parquet`
- `manifest.jsonl` mapping every input to its outputs
- Automatic cleanup of intermediate files after successful generation

## Quick start

```bash
python scripts/run_batch.py \
  --input_dir <input_dir> \
  --gpus 0 \
  --control_type depth
```

See the [Chinese README](README.md) for the full parameter reference (every CLI option is documented with examples).

## Requirements

- `conda` environments for the individual steps (defaults: `qwen25vl` / `qwen3vl` for prompts, `depthanything` for depth, `autoseg` for segmentation, `sam` for edge/blur)
- NVIDIA Cosmos-Transfer2.5 installed separately (uses its own `uv` `.venv`)
- Optional: RIFE (`rife-ncnn-vulkan` or Practical-RIFE) for duration restoration
- This repository's own dependencies: `numpy`, `pyarrow`, `pillow` (see `requirements.txt`)

Paths to the external projects are configured in `auto_transfer_runner/config.py` (or via CLI options). The upstream projects used by the pipeline are bundled under `Additional_proj/` for reference; install each following its own README.

## Project structure

- `scripts/run_batch.py` — main entry point
- `auto_transfer_runner/` — the pipeline package (`cli.py` argument parsing and GPU scheduling, `pipeline.py` per-video pipeline, `steps/`, `utils/`)
- `gen_controls_cv_mp4.py` — edge/blur control-signal generation
- `robot_video_to_prompt.py` — video-to-prompt generation
- `Additional_proj/` — bundled upstream projects (see NOTICE)

## License

The original code in this repository is released under the [MIT License](LICENSE). Third-party components under `Additional_proj/` remain under their own licenses — see [NOTICE](NOTICE).
