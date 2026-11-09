#!/usr/bin/env python3
"""Build benchmark_manifest.jsonl from an auto_transfer work_dir."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.datasets.manifest import build_and_write


def main() -> None:
    p = argparse.ArgumentParser(description="Build unified benchmark manifest")
    p.add_argument(
        "--work_dir",
        type=Path,
        required=True,
        help="auto_transfer work_dir containing manifest.jsonl",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output benchmark_manifest.jsonl (default: <work_dir>/benchmark_manifest.jsonl)",
    )
    p.add_argument("--include-failed", action="store_true")
    args = p.parse_args()
    out = args.out or (args.work_dir / "benchmark_manifest.jsonl")
    meta = build_and_write(args.work_dir, out, include_failed=args.include_failed)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
