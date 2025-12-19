from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..utils.subprocess_utils import run_cmd
from ..utils.logging_utils import append_log, console_print


def run_transfer_inference(
    transfer_root: Path,
    json_path: Path,
    output_dir: Path,
    hf_cache: Path,
    env: Dict[str, str],
    log_path: Path,
    skip_if_output_exists: bool,
) -> None:
    """
    Step5: run Cosmos-Transfer2.5 inference.

    Notes:
    - We call <transfer_root>/.venv/bin/python directly.
    - HF cache env vars are set inside this subprocess env.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if skip_if_output_exists and any(output_dir.iterdir()):
        append_log(log_path, f"[SKIP] output_dir not empty, skip inference: {output_dir}")
        console_print(f"[SKIP] inference (output exists): {output_dir.name}")
        return

    venv_python = transfer_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise FileNotFoundError(f"Transfer venv python not found: {venv_python}")

    env = dict(env)
    env["HF_HOME"] = str(hf_cache)
    env["HF_HUB_CACHE"] = str(hf_cache / "hub")
    env["HUGGINGFACE_HUB_CACHE"] = str(hf_cache / "hub")
    env["TRANSFORMERS_CACHE"] = str(hf_cache / "hub")

    cmd = [
        str(venv_python),
        "examples/inference.py",
        "-i",
        str(json_path),
        "-o",
        str(output_dir),
    ]
    run_cmd(
        cmd,
        cwd=transfer_root,
        env=env,
        log_path=log_path,
        title="STEP5: Transfer2.5 inference",
        stream_to_console=True,
        force_console=True,  # 需要实时查看扩散进度条
    )
