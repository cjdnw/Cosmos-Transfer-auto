from __future__ import annotations

import datetime as dt
import os
import shutil
from pathlib import Path
from zoneinfo import ZoneInfo

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def _current_timestamp() -> str:
    """Return the current Asia/Shanghai timestamp string, with timezone offset."""
    return dt.datetime.now(SHANGHAI_TZ).isoformat(timespec="seconds")


def _gpu_prefix() -> str:
    """Return GPU tag based on CUDA_VISIBLE_DEVICES, empty if unset."""
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cuda:
        return ""
    return f"[GPU{cuda}] "


def format_console_line(message: str) -> str:
    """Prefix a message with the Asia/Shanghai timestamp and GPU tag."""
    ts = _current_timestamp()
    return f"[{ts}]{_gpu_prefix()}{message}"


def console_enabled() -> bool:
    """Whether to print high-level progress to stdout."""
    return os.environ.get("ATR_CONSOLE", "0") == "1"


def console_print(message: str) -> None:
    """Print a single progress line to stdout when enabled, prefixed with Asia/Shanghai time."""
    if console_enabled():
        print(format_console_line(message), flush=True)


def append_log(log_path: Path, message: str) -> None:
    """Append a line to the per-job log file (UTF-8)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip("\n") + "\n")


def log_section(log_path: Path, title: str) -> None:
    """Write a readable section header to the log file; also print a short line to console if enabled."""
    ts = _current_timestamp()
    append_log(log_path, "\n" + "=" * 90)
    append_log(log_path, f"[{ts}] {title}")
    append_log(log_path, "=" * 90 + "\n")
    console_print(title)


def classify_job_log(log_path: Path, status: str) -> Path | None:
    """Copy a job log into logs/success or logs/fail for easier inspection."""
    if not log_path.exists():
        return None
    normalized = status.lower()
    status_dir = "success" if normalized in {"success", "done"} else "fail"
    target_dir = log_path.parent / status_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / log_path.name
    shutil.copy2(log_path, target_path)
    return target_path
