from pathlib import Path
import sys
import json

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto_transfer_runner.utils.warp_time import (  # noqa: E402
    compute_time_steps,
    count_guided_output_frames,
    validate_time_steps,
)


def _write_parquet(parquet_path: Path, states: np.ndarray) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"observation.state": states.tolist()})
    pq.write_table(table, parquet_path)


def _write_meta(dataset_root: Path, state_dim: int) -> None:
    meta_root = dataset_root / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    names = [f"position.zarm_{idx}" for idx in range(state_dim)]
    info = {"observation": {"state": {"names": names}}}
    (meta_root / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_guided_frame_count_ok():
    frames = ["I0", "I1"]
    segment_frames = [5]
    time_steps = [[0.1, 0.5, 0.9]]
    validate_time_steps(time_steps, segment_frames)
    assert count_guided_output_frames(frames, segment_frames, time_steps) == 5


def test_time_steps_invalid_raises():
    segment_frames = [5]
    time_steps = [[0.5, 0.4, 0.9]]
    with pytest.raises(ValueError):
        validate_time_steps(time_steps, segment_frames)


def test_compute_time_steps_hybrid_debug(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    data_root = dataset_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    parquet_path = data_root / "sample.parquet"

    state_dim = 4
    states = np.zeros((8, state_dim), dtype=float)
    states[3:, 0] = np.linspace(0.0, 1.0, states.shape[0] - 3)
    _write_meta(dataset_root, state_dim)
    _write_parquet(parquet_path, states)

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(states.shape[0]):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[:, idx % 8 :] = 255
        np.save(mask_dir / f"{idx:08d}.npy", mask)

    frame_map = [0, 4, 7]
    segment_frames = [5, 4]
    time_steps, debug_info = compute_time_steps(
        frame_map=frame_map,
        segment_frames=segment_frames,
        parquet_path=str(parquet_path),
        pose_name_prefix=["position.zarm_"],
        sam_masks=mask_dir,
        image_wh=(8, 8),
        mode="sam_zarm_hybrid",
        cache_path=None,
        debug=True,
    )

    assert len(time_steps) == 2
    assert len(time_steps[0]) == 3
    assert len(time_steps[1]) == 2
    assert debug_info.alpha_stats is not None
    print("segment 0 stats", min(time_steps[0]), np.mean(time_steps[0]), max(time_steps[0]))
    print("segment 1 stats", min(time_steps[1]), np.mean(time_steps[1]), max(time_steps[1]))
    print("alpha stats", debug_info.alpha_stats)
