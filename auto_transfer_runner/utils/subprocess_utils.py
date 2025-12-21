from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Dict

from .logging_utils import append_log, log_section, console_enabled


def run_cmd(
    cmd: List[str],
    cwd: Optional[Path],
    env: Dict[str, str],
    log_path: Path,
    title: str,
    stream_to_console: bool = False,
    force_console: bool = False,
) -> None:
    """
    运行一个外部命令，并将 stdout/stderr 追加写入 log_path。
    出错时抛异常（包含 log 路径方便定位）。
    """
    log_section(log_path, title)
    append_log(log_path, "CWD: " + (str(cwd) if cwd else os.getcwd()))
    append_log(log_path, "CMD: " + " ".join(cmd))
    append_log(log_path, "")

    with log_path.open("a", encoding="utf-8") as f:
        if stream_to_console:
            process = subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd else None,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert process.stdout is not None  # for mypy
            for line in process.stdout:
                f.write(line)
                if force_console or console_enabled():
                    print(line, end="")
            process.wait()
            returncode = process.returncode
        else:
            result = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                env=env,
                stdout=f,
                stderr=f,
                text=True,
                check=False,
            )
            returncode = result.returncode

    if returncode != 0:
        raise RuntimeError(f"Command failed (returncode={returncode}). See log: {log_path}")


def conda_run(conda_env: str, args: List[str]) -> List[str]:
    """生成 `conda run -n <env> ...` 命令行。"""
    return ["conda", "run", "-n", conda_env] + args
