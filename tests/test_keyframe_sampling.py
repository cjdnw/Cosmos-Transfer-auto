from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto_transfer_runner.steps.preprocess import (  # noqa: E402
    _select_keyframes_from_states,
    _select_unique_by_cumulative,
)


def test_unique_sampling_handles_plateaus(tmp_path: Path):
    video_frames = 447
    diffs = np.zeros(video_frames, dtype=float)
    diffs[::50] = 1.0
    cumulative = np.cumsum(diffs)

    target_frames = 93
    indices, resolved_target = _select_unique_by_cumulative(
        cumulative,
        target_frames=target_frames,
        log_path=tmp_path / "keyframe.log",
    )

    assert resolved_target == target_frames
    assert len(indices) == target_frames
    assert indices == sorted(indices)
    assert len(set(indices)) == target_frames
    assert indices[0] == 0
    assert indices[-1] == video_frames - 1


def test_parquet_motion_sampling_survives_large_max_gap(tmp_path: Path):
    video_frames = 447
    state_dim = 4
    states = np.zeros((video_frames, state_dim), dtype=float)
    states[200:, 0] = np.linspace(0.0, 3.0, video_frames - 200)
    state_names = [f"position.zarm_{idx}" for idx in range(state_dim)]

    indices = _select_keyframes_from_states(
        states,
        state_names=state_names,
        dexhand_names=set(),
        target_frames=93,
        smooth_win=1,
        max_gap=9000,
        pose_subset="full",
        stats_mean=None,
        stats_std=None,
        log_path=tmp_path / "keyframe.log",
    )

    assert len(indices) == 93
    assert indices == sorted(indices)
    assert len(set(indices)) == 93
