#!/usr/bin/env python3
"""
End-to-end SYNTHETIC smoke:
  fixtures → unified manifest → metrics → quality score → selection → report JSON

No fabricated real-experiment claims. All fixtures are labeled synthetic.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.analysis.visualizations import save_score_histogram
from benchmark.datasets.manifest import build_and_write, load_benchmark_manifest
from benchmark.scoring.quality_score import score_records
from benchmark.scoring.selectors import run_selection
from benchmark.scripts.make_smoke_fixtures import main as make_fixtures
from benchmark.scripts.run_metrics import run_metrics


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SYNTHETIC benchmark smoke")
    parser.add_argument(
        "--work_dir",
        type=Path,
        default=None,
        help="Optional real/synth auto_transfer work_dir. Default: regenerate testdata/smoke/work_dir",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=ROOT / "benchmark" / "artifacts" / "smoke",
    )
    parser.add_argument("--regen-fixtures", action="store_true", default=True)
    parser.add_argument("--no-regen-fixtures", action="store_false", dest="regen_fixtures")
    parser.add_argument(
        "--encoder",
        choices=["pixel_pca", "dinov2"],
        default=None,
        help="Override diversity.encoder from metrics.yaml (dinov2 downloads frozen weights once).",
    )
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    if artifacts.exists():
        # clean previous smoke artifacts only
        for child in artifacts.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
    artifacts.mkdir(parents=True, exist_ok=True)

    if args.work_dir is None:
        if args.regen_fixtures:
            make_fixtures()
        work_dir = ROOT / "benchmark" / "testdata" / "smoke" / "work_dir"
    else:
        work_dir = Path(args.work_dir)

    if not (work_dir / "manifest.jsonl").exists():
        print(f"ERROR: missing {work_dir / 'manifest.jsonl'}", file=sys.stderr)
        return 2

    # 1) Unified manifest
    bmanifest = artifacts / "benchmark_manifest.jsonl"
    meta = build_and_write(work_dir, bmanifest, include_failed=True)
    (artifacts / "manifest_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 2) Metrics
    metrics_cfg = load_yaml(ROOT / "benchmark" / "configs" / "metrics.yaml")
    if args.encoder:
        metrics_cfg.setdefault("diversity", {})["encoder"] = args.encoder
    scoring_cfg = load_yaml(ROOT / "benchmark" / "configs" / "scoring.yaml")
    selection_cfg = load_yaml(ROOT / "benchmark" / "configs" / "selection.yaml")

    metrics_out = run_metrics(bmanifest, metrics_cfg, max_frames=16)
    features = metrics_out.pop("features", None)
    (artifacts / "metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    if features is not None:
        np.save(artifacts / "features.npy", features)

    # Merge C/T/N into scoreable records
    scoreable = []
    for row in metrics_out["per_video"]:
        if row.get("metrics_status") != "ok":
            continue
        scoreable.append(
            {
                "uid": row["uid"],
                "C": row.get("C"),
                "T": row.get("T"),
                "N": row.get("N"),
                "control_source_tag": row.get("control_source_tag", "Proxy"),
            }
        )

    scored = score_records(scoreable, scoring_cfg=scoring_cfg)
    (artifacts / "scores.json").write_text(json.dumps(scored, indent=2), encoding="utf-8")

    # Feature matrix aligned to scoreable order
    uid_to_featrow = {}
    if features is not None:
        # features rows align with full per_video list
        for i, row in enumerate(metrics_out["per_video"]):
            if row.get("uid") and row.get("metrics_status") == "ok":
                uid_to_featrow[row["uid"]] = features[i]
    feat_mat = None
    if uid_to_featrow and scored:
        feat_mat = np.stack([uid_to_featrow[r["uid"]] for r in scored], axis=0)
    else:
        # Deterministic random features so selectors still run
        rng = np.random.RandomState(0)
        feat_mat = rng.randn(len(scored), 16).astype(np.float32)
        feat_mat /= np.linalg.norm(feat_mat, axis=1, keepdims=True) + 1e-8

    sel_cfg = selection_cfg.get("selection", {})
    gate_cfg = selection_cfg.get("quality_gate", {})
    k = int(sel_cfg.get("k", 4))
    strategy = str(sel_cfg.get("strategy", "quality_gate_greedy_diversity"))
    selected, sel_meta = run_selection(
        scored,
        feat_mat,
        strategy=strategy,
        k=k,
        seed=int(sel_cfg.get("seed", 0)),
        lambda_quality=float(sel_cfg.get("lambda_quality", 0.6)),
        gate_cfg=gate_cfg,
    )

    # Also run baselines for the smoke report
    baselines = {}
    for name in ("random", "topk_quality"):
        chosen, meta_b = run_selection(
            scored, feat_mat, strategy=name, k=k, seed=0, gate_cfg=gate_cfg
        )
        baselines[name] = {
            "uids": [c["uid"] for c in chosen],
            "meta": meta_b,
            "mean_Q": float(np.mean([c["Q"] for c in chosen if c.get("Q") is not None])) if chosen else None,
        }

    qs = np.array([r["Q"] for r in scored if r.get("Q") is not None], dtype=np.float64)
    hist_path = artifacts / "score_histogram.csv"
    save_score_histogram(qs, hist_path, title="SYNTHETIC Q distribution")

    report = {
        "title": "SYNTHETIC smoke report — not a real experiment result",
        "synthetic": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "work_dir": str(work_dir),
        "manifest_meta": meta,
        "n_scoreable": len(scored),
        "weights_hypothesis_only": scoring_cfg.get("weights"),
        "weights_note": (
            "wc/wt/wn defaults are hypotheses only; not tuned on a real test set."
        ),
        "proxy_note": (
            "All control metrics are Proxy (Depth-Anything / AutoSeg / OpenCV), never GT."
        ),
        "encoder": metrics_out.get("encoder"),
        "scores_summary": {
            "Q_mean": float(qs.mean()) if qs.size else None,
            "Q_std": float(qs.std()) if qs.size else None,
            "C_mean": float(np.mean([r["C"] for r in scored if r.get("C") is not None])) if scored else None,
            "T_mean": float(np.mean([r["T"] for r in scored if r.get("T") is not None])) if scored else None,
            "N_mean": float(np.mean([r["N"] for r in scored if r.get("N") is not None])) if scored else None,
        },
        "selection": {
            "strategy": strategy,
            "k": k,
            "selected_uids": [s["uid"] for s in selected],
            "selected": selected,
            "meta": sel_meta,
        },
        "baselines": baselines,
        "artifacts": {
            "benchmark_manifest": str(bmanifest),
            "metrics": str(artifacts / "metrics.json"),
            "scores": str(artifacts / "scores.json"),
            "histogram_csv": str(hist_path),
        },
        "encoder_note": (
            "dinov2 is a frozen official (or timm-equivalent) ViT. "
            "pixel_pca is the offline fallback when torch or weights are missing. "
            "Smoke numbers are synthetic, not Cosmos results."
        ),
        "not_implemented_yet": [
            "Full Pilot / multi-seed downstream on real Cosmos outputs",
            "Facility Location (stub → greedy fallback)",
            "Temporal LPIPS",
            "Real GT control fidelity",
        ],
    }
    report_path = artifacts / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Schema sample (committed-friendly small pointer)
    schema = {
        "required_keys": [
            "synthetic",
            "manifest_meta",
            "scores_summary",
            "selection",
            "baselines",
            "proxy_note",
            "weights_hypothesis_only",
            "encoder",
        ],
        "note": "See report.json after smoke; artifacts/ is gitignored.",
    }
    (ROOT / "benchmark" / "artifacts" / "smoke_report_schema.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    schema_path = ROOT / "benchmark" / "testdata" / "smoke" / "report_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    enc = metrics_out.get("encoder") or {}
    enc_detail = enc.get("detail") or {}
    print("=" * 60)
    print("SMOKE OK (SYNTHETIC)")
    print(f"  work_dir:   {work_dir}")
    print(f"  scored:     {len(scored)}")
    print(
        "  encoder:    "
        f"{enc.get('backend')} status={enc.get('status')} "
        f"model={enc_detail.get('model_name')} device={enc_detail.get('device')} "
        f"source={enc_detail.get('source')}"
    )
    print(f"  selected:   {[s['uid'] for s in selected]}")
    print(f"  report:     {report_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
