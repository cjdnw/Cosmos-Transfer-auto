#!/usr/bin/env python3
"""Compute control/temporal/diversity metrics for a benchmark manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.datasets.manifest import load_benchmark_manifest
from benchmark.metrics.control import (
    aggregate_control_score,
    compute_control_metrics_for_video,
)
from benchmark.metrics.diversity import FeatureEncoder, novelty_against_pool
from benchmark.metrics.io_utils import load_rgb_sequence
from benchmark.metrics.temporal import aggregate_temporal_score, compute_temporal_metrics


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_metrics(
    manifest_path: Path,
    metrics_cfg: Dict[str, Any],
    *,
    max_frames: int = 16,
) -> Dict[str, Any]:
    records = load_benchmark_manifest(manifest_path)
    encoder = FeatureEncoder(metrics_cfg.get("diversity", {}))
    per_video: List[Dict[str, Any]] = []
    rgbs = []
    usable_idx = []

    for i, rec in enumerate(records):
        row: Dict[str, Any] = {
            "uid": rec.get("uid"),
            "status": rec.get("status"),
            "missing_fields": rec.get("missing_fields", []),
            "control_source_tag": rec.get("control_source_tag", "Proxy"),
        }
        if rec.get("status") != "ok" or not rec.get("generated_video"):
            row["metrics_status"] = "skipped"
            per_video.append(row)
            continue
        try:
            rgb = load_rgb_sequence(rec["generated_video"], max_frames=max_frames)
        except Exception as e:
            row["metrics_status"] = "error"
            row["error"] = repr(e)
            per_video.append(row)
            continue

        control_res = compute_control_metrics_for_video(
            rgb,
            rec.get("control_paths") or {},
            cfg=metrics_cfg.get("control", {}),
        )
        temporal_res = compute_temporal_metrics(rgb, cfg=metrics_cfg.get("temporal", {}))
        C = aggregate_control_score(control_res)
        T = aggregate_temporal_score(temporal_res)
        row.update(
            {
                "metrics_status": "ok",
                "control_metrics": {k: v.to_dict() for k, v in control_res.items()},
                "temporal_metrics": {k: v.to_dict() for k, v in temporal_res.items()},
                "C": C,
                "T": T,
            }
        )
        per_video.append(row)
        rgbs.append(rgb)
        usable_idx.append(len(per_video) - 1)

    # Diversity / novelty
    features = None
    if usable_idx and encoder.backend != "unavailable":
        encoder.fit_pca(rgbs)
        feats = np.stack([encoder.encode(rgb) for rgb in rgbs], axis=0)
        features = np.zeros((len(per_video), feats.shape[1]), dtype=np.float32)
        for local_j, global_i in enumerate(usable_idx):
            features[global_i] = feats[local_j]
            # Novelty vs other candidates as stand-in when no real pool
            pool = np.concatenate([feats[:local_j], feats[local_j + 1 :]], axis=0) if len(feats) > 1 else feats
            N = novelty_against_pool(feats[local_j], pool)
            per_video[global_i]["N"] = N
            per_video[global_i]["feature_backend"] = encoder.backend
            per_video[global_i]["feature_status"] = encoder.status
    else:
        for i in usable_idx:
            per_video[i]["N"] = None
            per_video[i]["feature_backend"] = encoder.backend
            per_video[i]["feature_status"] = encoder.status

    return {
        "n_records": len(records),
        "n_scored": len(usable_idx),
        "encoder": {"backend": encoder.backend, "status": encoder.status, "detail": encoder.detail},
        "per_video": per_video,
        "features": features,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument(
        "--metrics-config",
        type=Path,
        default=ROOT / "benchmark" / "configs" / "metrics.yaml",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    cfg = load_yaml(args.metrics_config)
    result = run_metrics(args.manifest, cfg)
    # features are numpy — drop from JSON dump
    features = result.pop("features", None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if features is not None:
        np.save(args.out.with_suffix(".features.npy"), features)
    print(f"Wrote {args.out} (scored={result['n_scored']})")


if __name__ == "__main__":
    main()
