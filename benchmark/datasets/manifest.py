"""UnifiedManifest: auto_transfer manifest → benchmark_manifest.jsonl + hash."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .adapters.auto_transfer import (
    iter_auto_transfer_records,
    resolve_control_paths,
    resolve_generated_video,
)


PROXY_CONTROL_NOTE = (
    "Control signals from Video-Depth-Anything / AutoSeg-SAM2 / OpenCV are Proxy, "
    "never Ground Truth."
)


@dataclass
class UnifiedRecord:
    uid: str
    status: str
    input_video: Optional[str]
    generated_video: Optional[str]
    variant_idx: Optional[int]
    job_id: Optional[str]
    control_type: Optional[str]
    control_paths: Dict[str, Optional[str]] = field(default_factory=dict)
    control_source_tag: str = "Proxy"
    prompt_path: Optional[str] = None
    json_path: Optional[str] = None
    seed: Optional[int] = None
    missing_fields: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    source_status: Optional[str] = None
    split: str = "candidate"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _extract_seed(json_path: Optional[str]) -> Optional[int]:
    if not json_path:
        return None
    p = Path(json_path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    # Transfer specs often nest seed
    if isinstance(data, dict):
        if "seed" in data:
            return int(data["seed"])
        for v in data.values():
            if isinstance(v, dict) and "seed" in v:
                return int(v["seed"])
    return None


def _uid_for(record: Dict[str, Any], idx: int) -> str:
    job = record.get("job_id") or "nojid"
    var = record.get("variant_idx")
    stem = Path(str(record.get("input_video") or f"row{idx}")).stem
    return f"{stem}__var{var}__{job}__{idx}"


def build_unified_records(
    work_dir: Path,
    *,
    include_failed: bool = False,
) -> List[UnifiedRecord]:
    work_dir = Path(work_dir)
    unified: List[UnifiedRecord] = []
    for idx, row in enumerate(iter_auto_transfer_records(work_dir)):
        src_status = row.get("status")
        if src_status != "done":
            # Emit an auditable record when include_failed; otherwise skip entirely.
            if not include_failed:
                continue
            missing = ["generated_video", "control_paths"]
            notes = [f"source_status={src_status}", PROXY_CONTROL_NOTE]
            if row.get("error"):
                notes.append(f"source_error={row.get('error')}")
            if row.get("skip_reason"):
                notes.append(f"skip_reason={row.get('skip_reason')}")
            unified.append(
                UnifiedRecord(
                    uid=_uid_for(row, idx),
                    status="skipped_source_not_done",
                    input_video=row.get("input_video"),
                    generated_video=None,
                    variant_idx=row.get("variant_idx"),
                    job_id=row.get("job_id"),
                    control_type=row.get("control_type"),
                    control_paths={},
                    missing_fields=missing,
                    notes=notes,
                    source_status=src_status,
                )
            )
            continue

        gen = resolve_generated_video(row)
        controls = resolve_control_paths(row, work_dir=work_dir)
        missing: List[str] = []
        if gen is None:
            missing.append("generated_video")
        for k, v in controls.items():
            if not v or not Path(v).exists():
                missing.append(f"control:{k}")

        seed = _extract_seed(row.get("json_path"))
        if seed is None:
            missing.append("seed")

        for opt in ("prompt_path", "json_path", "input_video"):
            p = row.get(opt)
            if not p or not Path(str(p)).exists():
                # input may live on lab NAS — tag missing but don't fail smoke if generated exists
                if opt != "input_video" or gen is None:
                    if opt not in missing and (not p or not Path(str(p)).exists()):
                        missing.append(opt)

        status = "ok" if "generated_video" not in missing else "missing_generated"
        notes = [PROXY_CONTROL_NOTE]
        if row.get("error"):
            notes.append(f"source_error={row.get('error')}")
        if row.get("skip_reason"):
            notes.append(f"skip_reason={row.get('skip_reason')}")

        unified.append(
            UnifiedRecord(
                uid=_uid_for(row, idx),
                status=status,
                input_video=row.get("input_video"),
                generated_video=str(gen) if gen is not None else None,
                variant_idx=row.get("variant_idx"),
                job_id=row.get("job_id"),
                control_type=row.get("control_type"),
                control_paths=controls,
                control_source_tag="Proxy",
                prompt_path=row.get("prompt_path"),
                json_path=row.get("json_path"),
                seed=seed,
                missing_fields=sorted(set(missing)),
                notes=notes,
                source_status=src_status,
            )
        )
    return unified


def write_benchmark_manifest(
    records: Sequence[UnifiedRecord],
    out_path: Path,
) -> Dict[str, Any]:
    """Write benchmark_manifest.jsonl and a sidecar .sha256 hash file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for rec in records:
        lines.append(json.dumps(rec.to_dict(), ensure_ascii=False, sort_keys=True))
    payload = "\n".join(lines) + ("\n" if lines else "")
    out_path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    meta = {
        "path": str(out_path),
        "sha256": digest,
        "n_records": len(records),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "control_source_tag_default": "Proxy",
        "note": PROXY_CONTROL_NOTE,
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    hash_path = out_path.with_suffix(out_path.suffix + ".sha256")
    hash_path.write_text(digest + "\n", encoding="utf-8")
    return meta


def load_benchmark_manifest(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_and_write(work_dir: Path, out_path: Path, include_failed: bool = False) -> Dict[str, Any]:
    records = build_unified_records(work_dir, include_failed=include_failed)
    meta = write_benchmark_manifest(records, out_path)
    meta["n_ok"] = sum(1 for r in records if r.status == "ok")
    meta["n_missing_generated"] = sum(1 for r in records if r.status == "missing_generated")
    return meta
