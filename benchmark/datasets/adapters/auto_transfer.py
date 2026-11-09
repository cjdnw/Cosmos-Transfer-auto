"""Adapter: auto_transfer_runner work_dir / manifest.jsonl → candidate records."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                rows.append(
                    {
                        "status": "manifest_parse_error",
                        "error": repr(e),
                        "line_no": line_no,
                    }
                )
    return rows


def resolve_generated_video(record: Dict[str, Any]) -> Optional[Path]:
    """
    Prefer post_restore_output, else largest mp4 under output_dir,
    else a frames/ directory if present (smoke fixtures).
    """
    post = record.get("post_restore_output")
    if post:
        p = Path(post)
        if p.exists():
            return p

    arm = record.get("arm_swap_output")
    if arm:
        p = Path(arm)
        if p.exists():
            return p

    out_dir = record.get("output_dir")
    if not out_dir:
        return None
    out = Path(out_dir)
    if not out.exists():
        return None

    # Smoke: frames directory
    frames = out / "frames"
    if frames.is_dir() and any(frames.iterdir()):
        return frames

    mp4s = [p for p in out.glob("*.mp4") if p.is_file() and p.stat().st_size > 0]
    if not mp4s:
        # also accept .npy video tensor
        npys = list(out.glob("*.npy"))
        if npys:
            return npys[0]
        return None
    return max(mp4s, key=lambda p: p.stat().st_size)


def resolve_control_paths(record: Dict[str, Any], work_dir: Optional[Path] = None) -> Dict[str, Optional[str]]:
    """
    Normalize control_paths dict. Values may be files or directories.
    Always treat as Proxy for auto_transfer-generated controls.
    """
    raw = record.get("control_paths") or {}
    out: Dict[str, Optional[str]] = {"depth": None, "seg": None, "edge": None, "blur": None}
    if isinstance(raw, dict):
        for k in out:
            v = raw.get(k)
            if v is None:
                continue
            p = Path(v)
            out[k] = str(p) if p.exists() else str(p)  # keep path even if missing; tagged later
    # control_vis_path is visualization, not a typed control — ignore for fidelity metrics
    return out


def iter_auto_transfer_records(work_dir: Path) -> Iterable[Dict[str, Any]]:
    manifest = work_dir / "manifest.jsonl"
    for row in read_jsonl(manifest):
        yield row
