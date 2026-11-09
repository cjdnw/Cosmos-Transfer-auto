# Training-Free Synthetic Video Quality Benchmark

Independent package for **training-free** quality assessment and diversity-aware
selection of Cosmos-Transfer2.5 (and similar) generated robotic videos.

This package does **not** modify `auto_transfer_runner`. It reads an existing
`work_dir/manifest.jsonl` produced by the generation pipeline.

## Hard constraints

- No training/finetuning of Cosmos, VLA, large backbones, or learned quality nets.
- Frozen encoders (e.g. DINOv2) allowed for features; downstream trains only a small head (Pilot+).
- Control signals from Depth-Anything / AutoSeg-SAM2 / OpenCV are **Proxy**, never GT.
- Default score weights `w_c=0.45, w_t=0.35, w_n=0.20` are **hypotheses only** — not tuned on a real test set.
- Do not invent numerical experiment results. Smoke uses clearly labeled **SYNTHETIC** fixtures.

## Install

From the repo root (recommended: a venv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r benchmark/requirements.txt
```

Optional GPU extras for Pilot (not required for smoke): `torch`, DINOv2 weights, `lpips`.

## SYNTHETIC smoke (CPU)

```bash
# from repo root
python benchmark/scripts/run_smoke.py
python -m pytest benchmark/tests -q
```

The default encoder is `pixel_pca`, so this path does not download weights.

Artifacts land in `benchmark/artifacts/smoke/` (gitignored), including `report.json`.
The report records `encoder.backend` (`pixel_pca` or `dinov2`).

## DINOv2 diversity features

Frozen encoder only — nothing in this package trains DINOv2 or a quality network.
Control metrics stay **Proxy**, not ground truth. Smoke numbers stay **synthetic**.

Device order when `diversity.dinov2.device` is `auto`:

1. CUDA, if `torch.cuda.is_available()`
2. else `torch_directml` (`privateuseone:0`) when that package is installed (AMD Windows boxes)
3. else CPU

Forwards run in **float32**. Autocast is left off. CUDA and CPU use
`torch.inference_mode`. DirectML uses `torch.no_grad`: its `privateuseone`
kernels call `set_version_counter`, which raises
`Cannot set version_counter for inference tensor` on inference-mode tensors.
DirectML also has no `scaled_dot_product_attention` kernel, so that device uses
a float32 matmul attention for the forward only. If the DirectML forward still
fails, the loader skips to CPU and records `device_note` (this is a device
skip, not a missing checkpoint). `diversity.encoder` stays `pixel_pca` unless
you opt into `dinov2`.

Weight load order (first success wins):

1. `torch.hub` `facebookresearch/dinov2` at the revision in `configs/metrics.yaml`,
   official `dinov2_*_pretrain.pth`
2. `torchvision`, only when that install exports the DINOv2 builder
   (torchvision 0.29.2 does not)
3. `timm` equivalent `vit_*_patch14_dinov2.lvd142m` if timm is installed

Checkpoints cache under `benchmark/artifacts/weights/dinov2/` (gitignored).
Override with `diversity.dinov2.cache_dir` or the env var `BENCHMARK_DINOV2_CACHE`.

```bash
pip install torch   # CPU or CUDA build; on AMD Windows use torch-directml instead
# configs/metrics.yaml: diversity.encoder: dinov2
python benchmark/scripts/run_smoke.py --encoder dinov2
```

If torch is missing the status is `MISSING_DEP`. If torch is present but the
download or load fails, the status is `MISSING_WEIGHTS`. With
`dinov2.allow_fallback: true` (the default) the run continues on `pixel_pca`.
Set `allow_fallback: false` to make a missing encoder a hard failure.
DINOv2 video vectors are the mean of L2-normalized frame CLS tokens (native
width 384 for `dinov2_vits14`). `feature_dim` applies to `pixel_pca` only.

Fixtures live under `benchmark/testdata/smoke/` and are labeled **SYNTHETIC**.

## Point at real auto_transfer outputs

When lab disk is available (e.g. `/mnt/data/ganrunhe/auto_transfer`):

```bash
python benchmark/scripts/build_manifest.py \
  --work_dir /mnt/data/ganrunhe/auto_transfer \
  --out /mnt/data/ganrunhe/auto_transfer/benchmark_manifest.jsonl

python benchmark/scripts/run_metrics.py \
  --manifest /mnt/data/ganrunhe/auto_transfer/benchmark_manifest.jsonl \
  --out /tmp/metrics.json

# or reuse smoke entrypoint
python benchmark/scripts/run_smoke.py \
  --work_dir /mnt/data/ganrunhe/auto_transfer \
  --artifacts benchmark/artifacts/real_smoke
```

Expected `work_dir` layout:

```
work_dir/
  manifest.jsonl
  output_video/...
  control_cache/{depth,seg,edge,blur}/<fingerprint>/
  prompt/  json/  logs/
```

## Proxy vs GT

| Signal | Source in auto_transfer | Tag |
|--------|-------------------------|-----|
| Depth  | Video-Depth-Anything    | Proxy |
| Seg    | AutoSeg-SAM2            | Proxy |
| Edge/Blur | OpenCV               | Proxy |
| Independent human / sim GT | (not in default pipeline) | GT (optional later) |

Control fidelity always compares **re-extracted** attributes from the *generated* video
to the control signal — never control against itself.

## Quality score

\[
Q_i = w_c C_i + w_t T_i + w_n N_i
\]

Also supports `uniform`, `multiplicative`, and `hard_gate_then_weighted` (see `configs/scoring.yaml`).

## Selection

Two-stage: **Quality Gate** (YAML thresholds) → **Greedy diversity** (or Random / Top-K).
Facility Location is a stub in this PR (falls back to greedy).

## Not implemented yet (by design for this PR)

- Full Pilot / multi-seed downstream E0–E6 on real Cosmos videos
- Temporal LPIPS
- Facility Location (full)
- Claiming downstream gains or tuned weights

## Layout

See package tree under `benchmark/` — configs, datasets, metrics, scoring, downstream stubs, analysis stubs, tests, scripts.
