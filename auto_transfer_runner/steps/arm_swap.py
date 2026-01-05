from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

from ..utils.logging_utils import append_log
from ..utils.subprocess_utils import conda_run, run_cmd


def _find_transfer_output_mp4(output_dir: Path) -> Path:
    mp4s: List[Path] = [p for p in output_dir.glob("*.mp4") if p.is_file()]
    if not mp4s:
        raise FileNotFoundError(f"No mp4 outputs found in transfer output_dir: {output_dir}")
    if len(mp4s) > 1:
        names = ", ".join(sorted(p.name for p in mp4s))
        raise FileExistsError(
            "Multiple transfer mp4 outputs found; expected a single file. "
            f"Found: {names}"
        )
    return mp4s[0]


def _cleanup_workdir(workdir: Path) -> None:
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)


def _parse_fraction(value: str) -> float:
    if "/" in value:
        num_str, den_str = value.split("/", 1)
        den = float(den_str)
        if den == 0:
            return 0.0
        return float(num_str) / den
    return float(value)


def _probe_video_info(video_path: Path, log_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    append_log(log_path, f"[STEP6] ffprobe: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        append_log(log_path, result.stdout)
        append_log(log_path, result.stderr)
        raise RuntimeError(f"ffprobe failed for {video_path} (code={result.returncode}). See log: {log_path}")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe returned no video streams for {video_path}")
    stream = streams[0]
    width = int(stream["width"])
    avg_frame_rate = stream.get("avg_frame_rate", "0/0")
    fps_float = _parse_fraction(avg_frame_rate)
    fps = int(round(fps_float))
    if fps <= 0 or width <= 0:
        raise RuntimeError(f"Invalid video info from ffprobe: fps={fps}, width={width}")
    append_log(log_path, f"[STEP6] ffprobe parsed fps={fps} width={width}")
    return fps, width


def run_arm_swap(
    *,
    project_root: Path,
    conda_env: str,
    output_dir: Path,
    output_base_dir: Path,
    workdir: Path,
    orig_video: Path,
    qwen_prompt: str,
    env: Dict[str, str],
    log_path: Path,
) -> Path:
    """
    Step6: run robot arm swap (qwen3vl + sam3).

    Command:
      python -m robot_arm_swap --orig <input> --skin <transfer_output> --workdir <dir>
        --fps <fps> --width <width> --qwen-prompt "<prompt>" --out <output>
    """
    output_base_dir.mkdir(parents=True, exist_ok=True)
    workdir.parent.mkdir(parents=True, exist_ok=True)

    transfer_mp4 = _find_transfer_output_mp4(output_dir)
    output_path = output_base_dir / transfer_mp4.name

    workdir.mkdir(parents=True, exist_ok=True)
    fps, width = _probe_video_info(orig_video, log_path)
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
    try:
        run_cmd(cmd, cwd=project_root, env=env, log_path=log_path, title="STEP6: Robot arm swap")
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise FileNotFoundError(f"Arm swap output not found after run: {output_path}")
        return output_path
    finally:
        _cleanup_workdir(workdir)
