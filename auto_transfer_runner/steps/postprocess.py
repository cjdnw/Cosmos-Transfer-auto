from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict

from ..utils.cleanup import safe_unlink
from ..utils.logging_utils import append_log, console_print
from ..utils.subprocess_utils import run_cmd


def restore_transfer_length(
    *,
    output_dir: Path,
    reference_video: Path,
    fps: int,
    env: Dict[str, str],
    log_path: Path,
) -> Path:
    transfer_mp4 = _find_transfer_output_mp4(output_dir)
    target_frames = _probe_frame_count(transfer_mp4, env=env, log_path=log_path)
    total_frames = _probe_frame_count(reference_video, env=env, log_path=log_path)

    if target_frames == total_frames:
        append_log(
            log_path,
            f"[POST_RESTORE SKIP] already matches T={total_frames}: {transfer_mp4}",
        )
        console_print(f"[POST_RESTORE SKIP] {transfer_mp4.name} already T={total_frames}")
        return transfer_mp4

    tmp_restore = transfer_mp4.with_name(f".{transfer_mp4.stem}__restore_tmp{transfer_mp4.suffix}")
    safe_unlink(tmp_restore, log_path=log_path)
    restore_factor = total_frames / target_frames
    restore_filter = f"setpts=PTS*{restore_factor},minterpolate=fps={fps}"
    restore_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(transfer_mp4),
        "-vf",
        restore_filter,
        "-frames:v",
        str(total_frames),
        "-vsync",
        "cfr",
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_restore),
    ]
    run_cmd(
        restore_cmd,
        cwd=None,
        env=env or dict(os.environ),
        log_path=log_path,
        title="STEP5.5 Restore length (interpolate)",
    )
    tmp_restore.replace(transfer_mp4)
    append_log(
        log_path,
        (
            "[POST_RESTORE DONE] "
            f"T_restore={total_frames} factor={restore_factor} output={transfer_mp4}"
        ),
    )
    console_print(
        "[POST_RESTORE DONE] "
        f"T_restore={total_frames} factor={restore_factor} output={transfer_mp4.name}"
    )
    return transfer_mp4


def _find_transfer_output_mp4(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob("*.mp4"))
    if not candidates:
        raise FileNotFoundError(f"No mp4 outputs found in transfer output_dir: {output_dir}")
    return candidates[0]


def _probe_frame_count(video_path: Path, *, env: Dict[str, str], log_path: Path) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]
    append_log(log_path, "[POST_RESTORE] " + " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env or dict(os.environ),
        check=False,
    )
    if result.returncode != 0:
        append_log(log_path, f"[POST_RESTORE] ffprobe failed: {result.stderr.strip()}")
        raise RuntimeError("ffprobe failed while counting frames.")
    output = result.stdout.strip()
    if not output.isdigit():
        append_log(log_path, f"[POST_RESTORE] ffprobe output not integer: {output}")
        raise RuntimeError("ffprobe returned non-integer frame count.")
    return int(output)
