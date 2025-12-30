from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List

from ..utils.subprocess_utils import run_cmd, conda_run
from ..utils.logging_utils import append_log, console_print


def _find_transfer_output_mp4(output_dir: Path) -> Path:
    mp4s: List[Path] = [p for p in output_dir.glob("*.mp4") if p.is_file()]
    if not mp4s:
        raise FileNotFoundError(f"No mp4 outputs found in transfer output_dir: {output_dir}")
    if len(mp4s) == 1:
        return mp4s[0]
    return max(mp4s, key=lambda p: p.stat().st_size)


def _cleanup_workdir(workdir: Path) -> None:
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)


def run_arm_swap(
    *,
    project_root: Path,
    conda_env: str,
    output_dir: Path,
    output_root: Path,
    workdir: Path,
    orig_video: Path,
    fps: int,
    width: int,
    qwen_prompt: str,
    env: Dict[str, str],
    log_path: Path,
    skip_if_output_exists: bool,
) -> Path:
    """
    Step6: run robot arm swap (qwen3vl + sam3).

    Command:
      python -m robot_arm_swap --orig <input> --skin <transfer_output> --workdir <dir>
        --fps <fps> --width <width> --qwen-prompt "<prompt>" --out <output>
    """
    output_root.mkdir(parents=True, exist_ok=True)
    workdir.parent.mkdir(parents=True, exist_ok=True)

    transfer_mp4 = _find_transfer_output_mp4(output_dir)
    output_path = output_root / transfer_mp4.name

    if skip_if_output_exists and output_path.exists() and output_path.stat().st_size > 0:
        append_log(log_path, f"[SKIP] arm swap output exists: {output_path}")
        console_print(f"[SKIP] arm swap (output exists): {output_path.name}")
        _cleanup_workdir(workdir)
        return output_path

    workdir.mkdir(parents=True, exist_ok=True)

    cmd = conda_run(
        conda_env,
        [
            "python",
            "-m",
            "robot_arm_swap",
            "--orig",
            str(orig_video),
            "--skin",
            str(transfer_mp4),
            "--workdir",
            str(workdir),
            "--fps",
            str(fps),
            "--width",
            str(width),
            "--qwen-prompt",
            qwen_prompt,
            "--out",
            str(output_path),
        ],
    )
    run_cmd(cmd, cwd=project_root, env=env, log_path=log_path, title="STEP6: Robot arm swap (qwen3vl + sam3)")

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise FileNotFoundError(f"Arm swap output not found after run: {output_path}")

    _cleanup_workdir(workdir)
    return output_path
