from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def append_manifest(work_dir: Path, record: Dict[str, Any]) -> None:
    """
    Append one JSON line to <work_dir>/manifest.jsonl.

    Why:
      - When processing entire folders and/or repeating each video K times, it's easy to lose track.
      - manifest.jsonl provides a machine-readable mapping:
          input_video -> job_id/variant -> prompt/json/output paths

    Concurrency:
      - Multi-GPU means multi-process.
      - Appends rely on the OS for atomicity of the single-line write.
    """
    manifest_path = work_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
