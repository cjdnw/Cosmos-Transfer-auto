"""DINOv2 loader policy, fallback, and an optional real encode."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from benchmark.metrics.dinov2_encoder import (
    Dinov2LoadError,
    directml_scaled_dot_product_attention,
    round_to_patch,
    select_device_kind,
)
from benchmark.metrics.diversity import FeatureEncoder


def test_default_config_stays_on_pixel_pca():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "metrics.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert cfg["diversity"]["encoder"] == "pixel_pca"
    assert cfg["diversity"]["dinov2"]["allow_fallback"] is True
    assert cfg["diversity"]["dinov2"]["device"] == "auto"


def test_select_device_kind_order():
    assert select_device_kind("auto", cuda_available=True, directml_available=True) == "cuda"
    assert select_device_kind("auto", cuda_available=False, directml_available=True) == "directml"
    assert select_device_kind("auto", cuda_available=False, directml_available=False) == "cpu"
    assert select_device_kind("cpu", cuda_available=True, directml_available=True) == "cpu"
    assert select_device_kind("dml", cuda_available=False, directml_available=True) == "directml"
    with pytest.raises(RuntimeError):
        select_device_kind("cuda", cuda_available=False, directml_available=True)
    with pytest.raises(RuntimeError):
        select_device_kind("directml", cuda_available=True, directml_available=False)


def test_round_image_size_to_patch():
    assert round_to_patch(224, 14) == 224
    assert round_to_patch(225, 14) == 238
    assert round_to_patch(8, 14) == 14


def test_pixel_pca_encode_without_torch():
    enc = FeatureEncoder({"encoder": "pixel_pca", "feature_dim": 8})
    rgb = np.zeros((3, 16, 16, 3), dtype=np.uint8)
    enc.fit_pca([rgb, rgb + 1])
    feat = enc.encode(rgb)
    assert enc.backend == "pixel_pca"
    assert feat.shape == (enc.feature_dim,)
    assert np.isclose(np.linalg.norm(feat), 1.0)


def test_dinov2_falls_back_when_weights_missing(monkeypatch):
    def _boom(_cfg):
        raise Dinov2LoadError("no checkpoint", reason="MISSING_WEIGHTS", attempts=[{"source": "torch.hub", "error": "404"}])

    monkeypatch.setattr("benchmark.metrics.dinov2_encoder.load_dinov2_encoder", _boom)
    enc = FeatureEncoder({"encoder": "dinov2", "dinov2": {"allow_fallback": True}})
    assert enc.backend == "pixel_pca"
    assert enc.status == "MISSING_WEIGHTS"
    assert enc.detail["fallback"] == "pixel_pca"
    rgb = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    feat = enc.encode(rgb)
    assert feat.ndim == 1
    assert np.isfinite(feat).all()


def test_dinov2_unavailable_without_fallback(monkeypatch):
    def _boom(_cfg):
        raise Dinov2LoadError("torch is not installed", reason="MISSING_DEP", attempts=[])

    monkeypatch.setattr("benchmark.metrics.dinov2_encoder.load_dinov2_encoder", _boom)
    enc = FeatureEncoder({"encoder": "dinov2", "dinov2": {"allow_fallback": False}})
    assert enc.backend == "unavailable"
    assert enc.status == "MISSING_DEP"
    with pytest.raises(RuntimeError, match="MISSING_DEP"):
        enc.encode(np.zeros((2, 8, 8, 3), dtype=np.uint8))


def test_mocked_dinov2_encode_is_l2_normalized(monkeypatch):
    class _Fake:
        info = {
            "model_name": "dinov2_vits14",
            "device": "cpu",
            "source": "mock",
            "embed_dim": 8,
            "frozen": True,
        }

        def encode_video(self, rgb):
            vec = np.arange(8, dtype=np.float32) + float(rgb.mean())
            vec /= np.linalg.norm(vec) + 1e-8
            return vec

    monkeypatch.setattr(
        "benchmark.metrics.dinov2_encoder.load_dinov2_encoder",
        lambda _cfg: _Fake(),
    )
    enc = FeatureEncoder({"encoder": "dinov2", "dinov2": {"allow_fallback": False}})
    assert enc.backend == "dinov2"
    assert enc.status == "ok"
    assert enc.detail["model_name"] == "dinov2_vits14"
    rgb = np.zeros((4, 12, 12, 3), dtype=np.uint8)
    enc.fit_pca([rgb])  # must not replace the DINOv2 vector with pixel PCA
    feat = enc.encode(rgb)
    assert feat.shape == (8,)
    assert np.isclose(np.linalg.norm(feat), 1.0, atol=1e-5)


def test_directml_sdpa_matches_math_backend():
    torch = pytest.importorskip("torch")
    q = torch.randn(2, 3, 5, 4)
    k = torch.randn(2, 3, 5, 4)
    v = torch.randn(2, 3, 5, 4)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    got = directml_scaled_dot_product_attention(q, k, v)
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4)


def test_real_dinov2_encode_when_weights_available():
    pytest.importorskip("torch")
    enc = FeatureEncoder(
        {
            "encoder": "dinov2",
            "dinov2": {
                "allow_fallback": False,
                "device": "cpu",
                "max_frames": 2,
                "image_size": 224,
                "model_name": "dinov2_vits14",
            },
        }
    )
    if enc.backend != "dinov2":
        pytest.skip(f"DINOv2 weights unavailable: {enc.status} {enc.detail.get('error')}")
    assert enc.detail["source"]
    assert enc.detail["device"] == "cpu"
    assert enc.detail["frozen"] is True
    assert enc.detail["autocast"] is False
    rgb_a = np.zeros((3, 32, 40, 3), dtype=np.uint8)
    rgb_b = np.random.RandomState(0).randint(0, 255, size=(3, 32, 40, 3), dtype=np.uint8)
    feat_a = enc.encode(rgb_a)
    feat_a2 = enc.encode(rgb_a)
    feat_b = enc.encode(rgb_b)
    assert feat_a.shape[0] == enc.detail.get("embed_dim", feat_a.shape[0])
    assert feat_a.shape[0] > 16
    assert np.isfinite(feat_a).all()
    assert np.isclose(np.linalg.norm(feat_a), 1.0, atol=1e-4)
    assert np.allclose(feat_a, feat_a2)
    assert not np.allclose(feat_a, feat_b)
    assert all(not p.requires_grad for p in enc._dino.model.parameters())
