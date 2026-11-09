"""Downstream train entry (stub for this PR — not required for smoke)."""
from __future__ import annotations

from typing import Any, Dict


def train_downstream(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train a small head on frozen features.
    Not executed in smoke. Pilot requires torch + labels + GPU.
    """
    return {
        "status": "not_implemented_in_this_pr",
        "note": "Smoke does not train downstream. See configs/downstream.yaml.",
        "cfg_keys": sorted(cfg.keys()) if cfg else [],
    }


if __name__ == "__main__":
    print(train_downstream({}))
