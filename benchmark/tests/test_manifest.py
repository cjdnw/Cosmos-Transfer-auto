"""Tests for UnifiedManifest missing-field tagging."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.datasets.manifest import build_unified_records, write_benchmark_manifest


@pytest.fixture()
def tiny_work_dir(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    work.mkdir()
    # done row with missing controls + missing generated
    row_done = {
        "status": "done",
        "input_video": str(work / "missing_input.mp4"),
        "variant_idx": 0,
        "job_id": "j1",
        "control_type": "multicontrol",
        "prompt_path": str(work / "p.txt"),
        "json_path": str(work / "j.json"),
        "output_dir": str(work / "output_video" / "x"),
        "control_paths": {"depth": str(work / "no_depth"), "seg": None, "edge": None, "blur": None},
    }
    row_fail = {
        "status": "failed",
        "input_video": str(work / "y.mp4"),
        "variant_idx": 1,
        "job_id": "j2",
        "control_type": "edge",
        "error": "boom",
    }
    # create prompt so only some fields missing
    (work / "p.txt").write_text("hi", encoding="utf-8")
    (work / "j.json").write_text(json.dumps({"seed": 42}), encoding="utf-8")
    with (work / "manifest.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(row_done) + "\n")
        f.write(json.dumps(row_fail) + "\n")
    return work


def test_missing_fields_tagged(tiny_work_dir: Path):
    recs = build_unified_records(tiny_work_dir, include_failed=True)
    assert len(recs) == 2
    done = recs[0]
    assert done.control_source_tag == "Proxy"
    assert "generated_video" in done.missing_fields
    assert any(m.startswith("control:") for m in done.missing_fields)
    assert done.status == "missing_generated"
    # seed present via json
    assert done.seed == 42
    assert "seed" not in done.missing_fields

    failed = recs[1]
    assert failed.status == "skipped_source_not_done"
    assert "generated_video" in failed.missing_fields


def test_manifest_hash_written(tiny_work_dir: Path, tmp_path: Path):
    recs = build_unified_records(tiny_work_dir, include_failed=True)
    out = tmp_path / "benchmark_manifest.jsonl"
    meta = write_benchmark_manifest(recs, out)
    assert out.exists()
    assert Path(str(out) + ".sha256").exists()
    assert meta["n_records"] == 2
    assert len(meta["sha256"]) == 64
