from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

CHECKPOINT_DIRNAME = "checkpoints"


def checkpoint_dir_for_job(work_dir: Path, output_dir: Path) -> Path:
    relative = output_dir.relative_to(work_dir)
    return work_dir / CHECKPOINT_DIRNAME / relative


def marker_path(checkpoint_dir: Path, name: str) -> Path:
    return checkpoint_dir / f"{name}.done"


def job_done_marker_path(checkpoint_dir: Path) -> Path:
    return marker_path(checkpoint_dir, "job")


def marker_exists(marker: Path) -> bool:
    return marker.exists()


def mark_done(marker: Path) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = marker.with_name(f".{marker.name}.tmp")
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    tmp_path.write_text(f"{timestamp}\n", encoding="utf-8")
    tmp_path.replace(marker)


def clear_checkpoint_dir(checkpoint_dir: Path, *, root: Path) -> None:
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    _prune_empty_parents(checkpoint_dir.parent, root)


def _prune_empty_parents(path: Path, root: Path) -> None:
    current = path
    while current != root and current.exists():
        if any(current.iterdir()):
            return
        current.rmdir()
        current = current.parent
