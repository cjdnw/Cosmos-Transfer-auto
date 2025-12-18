from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .locking import FileLock, FileLockConfig


def append_manifest(work_dir: Path, record: Dict[str, Any]) -> None:
    """
    Append one JSON line to <work_dir>/manifest.jsonl.

    Why:
      - When processing entire folders and/or repeating each video K times, it's easy to lose track.
      - manifest.jsonl provides a machine-readable mapping:
          input_video -> job_id/variant -> prompt/json/output paths

    Concurrency:
      - Multi-GPU means multi-process.
      - We guard the manifest file with a small lock to avoid interleaved writes.
    """
    manifest_path = work_dir / "manifest.jsonl"
    lock_path = work_dir / "manifest.lock"

    # Keep this lock separate from control cache locks.
    lock_cfg = FileLockConfig(timeout_sec=60, stale_lock_sec=10 * 60)

    with FileLock(lock_path, lock_cfg):
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
