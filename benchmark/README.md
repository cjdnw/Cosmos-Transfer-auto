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

Artifacts land in `benchmark/artifacts/smoke/` (gitignored), including `report.json`.

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
- DINOv2 feature extraction (returns `MISSING_DEP` / pixel_pca fallback)
- Temporal LPIPS
- Facility Location (full)
- Claiming downstream gains or tuned weights

## Layout

See package tree under `benchmark/` — configs, datasets, metrics, scoring, downstream stubs, analysis stubs, tests, scripts.
