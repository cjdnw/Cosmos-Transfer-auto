from __future__ import annotations

import datetime as dt
import os
import time
from pathlib import Path


def format_duration(seconds: float) -> str:
    """Format a duration (seconds) as HH:MM:SS.mmm for logs/console."""
    milliseconds = int((seconds - int(seconds)) * 1000)
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


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


def log_step_done(log_path: Path, title: str, start_time: float) -> None:
    """Log a completion line with elapsed time for a step."""
    elapsed = time.perf_counter() - start_time
    msg = f"[DONE] {title} in {format_duration(elapsed)}"
    append_log(log_path, msg)
    console_print(msg)
