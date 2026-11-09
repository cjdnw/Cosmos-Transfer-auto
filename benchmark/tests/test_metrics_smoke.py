"""Metric function smoke tests on tiny arrays."""
from __future__ import annotations

import numpy as np

import pytest

from benchmark.metrics.control import depth_si_rmse, edge_f1, segmentation_miou
from benchmark.metrics.temporal import flickering_score, temporal_warping_error


def test_seg_miou_perfect():
    lab = np.zeros((2, 8, 8), dtype=np.int32)
    lab[:, :4, :] = 1
    assert segmentation_miou(lab, lab) == 1.0


def test_edge_f1_perfect():
    e = np.zeros((2, 8, 8), dtype=np.float32)
    e[:, 4, :] = 1.0
    assert edge_f1(e, e) == pytest.approx(1.0)


def test_depth_si_rmse_scaled():
    d = np.ones((2, 4, 4), dtype=np.float32) * 2.0
    # scale-invariant: 2x depth → SI-RMSE ~ 0
    assert depth_si_rmse(d * 2.0, d) < 1e-5


def test_temporal_on_static():
    rgb = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    rgb[:] = 40
    err, detail = temporal_warping_error(rgb, flow_backend="frame_diff")
    assert err == 0.0
    assert flickering_score(rgb) == 0.0
