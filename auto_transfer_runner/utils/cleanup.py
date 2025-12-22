from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from .logging_utils import append_log, console_print


def safe_unlink(path: Path, log_path: Optional[Path] = None) -> None:
    """Best-effort delete a file."""
    try:
        if path.exists():
            path.unlink()
            if log_path:
                append_log(log_path, f"[CLEANUP] deleted file: {path}")
    except Exception as e:
        if log_path:
            append_log(log_path, f"[CLEANUP WARN] failed to delete file {path}: {repr(e)}")


def safe_rmtree(path: Path, log_path: Optional[Path] = None) -> None:
    """Best-effort delete a directory tree."""
    try:
        if path.exists() and path.is_dir():
            shutil.rmtree(path, ignore_errors=False)
            if log_path:
                append_log(log_path, f"[CLEANUP] deleted dir: {path}")
    except Exception as e:
        if log_path:
            append_log(log_path, f"[CLEANUP WARN] failed to delete dir {path}: {repr(e)}")


def cleanup_job_intermediates(
    prompt_path: Path,
    prompt_intermediate_paths: Optional[list[Path]],
    json_path: Path,
    non_cached_control_dir: Path,
    log_path: Path,
    *,
    delete_prompt: bool,
    delete_json: bool,
    delete_non_cached_control_dir: bool,
    delete_log: bool,
) -> None:
    """
    Cleanup intermediates after a successful job.

    Keeps:
      - final output directory (not passed here)
      - cached control signals in control_cache/ (not passed here)
      - manifest.jsonl (global)

    Deletes (configurable):
      - per-job prompt file
      - per-job json spec
      - per-job non-cached control output dir (reserved for edge/seg/...)
      - per-job log file (optional; default False to preserve debug trail)
    Always deletes:
      - prompt_intermediate_paths (e.g., bracketed prompt, replacement json)
    """
    console_print("[CLEANUP] removing intermediates (prompt/json/...)")

    for p in prompt_intermediate_paths or []:
        safe_unlink(p, log_path=log_path)

    if delete_prompt:
        safe_unlink(prompt_path, log_path=log_path)
    if delete_json:
        safe_unlink(json_path, log_path=log_path)
    if delete_non_cached_control_dir:
        safe_rmtree(non_cached_control_dir, log_path=log_path)

    # Delete log last; after this point we cannot append to it
    if delete_log:
        safe_unlink(log_path, log_path=None)
