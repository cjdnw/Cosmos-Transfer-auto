from __future__ import annotations

import datetime as dt
import os
from pathlib import Path


def console_enabled() -> bool:
    """Whether to print high-level progress to stdout."""
    return os.environ.get("ATR_CONSOLE", "0") == "1"


def console_print(message: str) -> None:
    """Print a single progress line to stdout when enabled."""
    if console_enabled():
        print(message, flush=True)


def append_log(log_path: Path, message: str) -> None:
    """Append a line to the per-job log file (UTF-8)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip("\n") + "\n")


def log_section(log_path: Path, title: str) -> None:
    """Write a readable section header to the log file; also print a short line to console if enabled."""
    ts = dt.datetime.now().isoformat(timespec="seconds")
    append_log(log_path, "\n" + "=" * 90)
    append_log(log_path, f"[{ts}] {title}")
    append_log(log_path, "=" * 90 + "\n")
    console_print(f"[{ts}] {title}")
