"""Visualization stubs (no fabricated plots of real results)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def save_score_histogram(scores: np.ndarray, out_path: Path, title: str = "Q distribution") -> Optional[Path]:
    """
    Minimal histogram writer without matplotlib dependency:
    writes a CSV of bin counts. PNG plotting deferred to Pilot extras.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if scores.size == 0:
        out_path.write_text("bin_left,bin_right,count\n", encoding="utf-8")
        return out_path
    hist, edges = np.histogram(scores.astype(float), bins=10, range=(0.0, 1.0))
    lines = ["bin_left,bin_right,count"]
    for i, c in enumerate(hist):
        lines.append(f"{edges[i]},{edges[i+1]},{int(c)}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
