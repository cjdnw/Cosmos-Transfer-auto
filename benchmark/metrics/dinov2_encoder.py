"""Frozen DINOv2 features for diversity. No training and no finetuning.

Weight load order (first success wins):

1. ``torch.hub`` — ``facebookresearch/dinov2`` at a pinned revision, official
   ``*_pretrain.pth`` from ``dl.fbaipublicfiles.com``. Verified for
   ``dinov2_vits14`` (ViT-S/14, CLS embedding width 384).
2. ``torchvision`` — only if that install actually exports the DINOv2 builder.
   torchvision 0.29.2 does not.
3. ``timm`` — documented equivalent ``vit_*_patch14_dinov2.lvd142m`` when
   timm is installed.

Device order for ``device: auto``: CUDA, else ``torch_directml``
(``privateuseone``), else CPU. Forwards stay in float32. Autocast is not used.
DirectML often has no ``scaled_dot_product_attention`` kernel, so that device
uses an explicit float32 matmul attention during the forward only.

Checkpoints cache under ``benchmark/artifacts/weights/dinov2`` (gitignored)
unless ``BENCHMARK_DINOV2_CACHE`` or ``cache_dir`` is set.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# facebookresearch/dinov2 main as of the load verified in this repo.
DEFAULT_HUB_REPO = "facebookresearch/dinov2"
DEFAULT_HUB_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DEFAULT_SOURCES = ("torch.hub", "torchvision", "timm")

# timm names are the same LVD-142M DINOv2 checkpoints, not a different pretraining run.
TIMM_NAMES = {
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
    "dinov2_vitg14": "vit_giant_patch14_dinov2.lvd142m",
    "dinov2_vits14_reg": "vit_small_patch14_reg4_dinov2.lvd142m",
    "dinov2_vitb14_reg": "vit_base_patch14_reg4_dinov2.lvd142m",
    "dinov2_vitl14_reg": "vit_large_patch14_reg4_dinov2.lvd142m",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Dinov2LoadError(RuntimeError):
    """Torch missing (MISSING_DEP) or weights/code failed to load (MISSING_WEIGHTS)."""

    def __init__(self, message: str, *, reason: str, attempts: Optional[List[Dict[str, str]]] = None):
        super().__init__(message)
        self.reason = reason
        self.attempts = attempts or []


def default_cache_dir() -> Path:
    env = os.environ.get("BENCHMARK_DINOV2_CACHE")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[1] / "artifacts" / "weights" / "dinov2"


def round_to_patch(size: int, patch: int = 14) -> int:
    size = int(size)
    patch = int(patch)
    if patch <= 0:
        raise ValueError("patch must be positive")
    if size <= patch:
        return patch
    rem = size % patch
    if rem == 0:
        return size
    return size + (patch - rem)


def select_device_kind(
    preference: str,
    *,
    cuda_available: bool,
    directml_available: bool,
) -> str:
    """Return ``cuda``, ``directml``, or ``cpu``. Does not import torch."""
    pref = (preference or "auto").strip().lower()
    pref = {"dml": "directml", "privateuseone": "directml", "gpu": "auto"}.get(pref, pref)
    if pref not in {"auto", "cuda", "directml", "cpu"}:
        raise ValueError(f"Unknown DINOv2 device preference: {preference}")
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return "cuda"
    if pref == "directml":
        if not directml_available:
            raise RuntimeError("DirectML requested but torch_directml is not installed")
        return "directml"
    if cuda_available:
        return "cuda"
    if directml_available:
        return "directml"
    return "cpu"


def directml_scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale=None,
    **_kwargs,
):
    """Float32 attention used when DirectML has no SDPA kernel. Eval dropout is 0."""
    import torch
    import torch.nn.functional as F

    q = query.float()
    k = key.float()
    v = value.float()
    head_dim = q.shape[-1]
    scale_val = float(scale) if scale is not None else head_dim ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale_val
    if is_causal:
        n = q.shape[-2]
        causal = torch.ones(n, n, device=q.device, dtype=torch.bool).triu(1)
        attn = attn.masked_fill(causal, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn = attn.masked_fill(~attn_mask, float("-inf"))
        else:
            attn = attn + attn_mask.to(dtype=attn.dtype, device=attn.device)
    attn = torch.softmax(attn, dim=-1)
    if dropout_p and float(dropout_p) > 0:
        attn = F.dropout(attn, p=float(dropout_p))
    return torch.matmul(attn, v)


@contextmanager
def sdpa_for_device(kind: str) -> Iterator[None]:
    """On DirectML, swap SDPA for the float32 matmul path for the duration of the forward."""
    if kind != "directml":
        yield
        return
    import torch.nn.functional as F

    original = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = directml_scaled_dot_product_attention
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original


def resolve_torch_device(preference: str = "auto"):
    """Return ``(torch.device, kind)`` with kind in cuda/directml/cpu."""
    import torch

    directml_mod = None
    try:
        import torch_directml as directml_mod  # type: ignore
    except Exception:
        directml_mod = None
    kind = select_device_kind(
        preference,
        cuda_available=bool(torch.cuda.is_available()),
        directml_available=directml_mod is not None,
    )
    if kind == "cuda":
        return torch.device("cuda"), "cuda"
    if kind == "directml":
        return directml_mod.device(), "directml"
    return torch.device("cpu"), "cpu"


def _patch_size(model_name: str) -> int:
    if "16" in model_name:
        return 16
    return 14


def _subsample_indices(n_frames: int, max_frames: int) -> np.ndarray:
    n_take = max(1, min(int(max_frames), int(n_frames)))
    return np.linspace(0, n_frames - 1, num=n_take).astype(int)


def _preprocess_rgb(frames_uint8: np.ndarray, image_size: int) -> Any:
    """RGB uint8 (T,H,W,3) → float32 NCHW on CPU, ImageNet-normalized. No autocast."""
    import torch
    import torch.nn.functional as F

    arr = np.ascontiguousarray(frames_uint8)
    x = torch.from_numpy(arr).to(dtype=torch.float32)
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"expected RGB frames (T,H,W,3), got {tuple(x.shape)}")
    if float(x.detach().max()) > 1.5:
        x = x / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()
    x = F.interpolate(x, size=(image_size, image_size), mode="bicubic", align_corners=False)
    x = x.clamp(0.0, 1.0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return (x - mean) / std


def _as_frame_matrix(out: Any) -> np.ndarray:
    """Model output → (T, C) float32 numpy. Prefers the CLS token."""
    import torch

    if isinstance(out, dict):
        if "x_norm_clstoken" in out:
            out = out["x_norm_clstoken"]
        elif "x_prenorm" in out and isinstance(out["x_prenorm"], torch.Tensor) and out["x_prenorm"].ndim == 3:
            out = out["x_prenorm"][:, 0]
        else:
            out = next(iter(out.values()))
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"DINOv2 forward returned {type(out)}")
    if out.ndim == 3:
        out = out[:, 0]
    if out.ndim != 2:
        raise RuntimeError(f"unexpected DINOv2 embedding shape {tuple(out.shape)}")
    return out.detach().float().cpu().numpy()


def _freeze(model: Any) -> Any:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _load_torch_hub(model_name: str, cache_dir: Path, hub_repo: str, hub_revision: str) -> Tuple[Any, Dict[str, Any]]:
    import warnings

    import torch

    cache_dir.mkdir(parents=True, exist_ok=True)
    prev = torch.hub.get_dir()
    torch.hub.set_dir(str(cache_dir / "hub"))
    try:
        repo = f"{hub_repo}:{hub_revision}" if hub_revision else hub_repo
        # Upstream prints this when xformers is absent. The math attention path still runs.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="xFormers is not available")
            model = torch.hub.load(repo, model_name, trust_repo=True, verbose=False)
    finally:
        torch.hub.set_dir(prev)
    return model, {
        "source": "torch.hub",
        "hub_repo": hub_repo,
        "hub_revision": hub_revision or None,
    }


def _load_torchvision(model_name: str, cache_dir: Path) -> Tuple[Any, Dict[str, Any]]:
    import torch
    from torchvision.models import get_model, get_model_weights

    cache_dir.mkdir(parents=True, exist_ok=True)
    prev = torch.hub.get_dir()
    torch.hub.set_dir(str(cache_dir / "hub"))
    try:
        weights_enum = get_model_weights(model_name)
        weights = weights_enum.DEFAULT
        model = get_model(model_name, weights=weights)
    finally:
        torch.hub.set_dir(prev)
    return model, {"source": "torchvision", "url": getattr(weights, "url", None)}


def _load_timm(model_name: str, cache_dir: Path) -> Tuple[Any, Dict[str, Any]]:
    timm_name = TIMM_NAMES.get(model_name)
    if not timm_name:
        raise RuntimeError(f"no timm equivalent registered for {model_name}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hf"))
    os.environ.setdefault("TORCH_HOME", str(cache_dir / "hub"))
    import timm

    model = timm.create_model(timm_name, pretrained=True, num_classes=0)
    return model, {"source": "timm", "timm_name": timm_name}


def _place_model(model: Any, preference: str, image_size: int) -> Tuple[Any, Any, str, Optional[str]]:
    """Move to the requested device. ``auto`` + DirectML failure falls back to CPU."""
    import torch

    device, kind = resolve_torch_device(preference)
    note = None
    try:
        model = model.float().to(device)
        if kind == "directml":
            _probe_forward(model, device, kind, image_size)
        return model, device, kind, note
    except Exception as exc:
        pref = (preference or "auto").strip().lower()
        if kind == "directml" and pref in {"auto", "gpu", ""}:
            cpu = torch.device("cpu")
            model = model.float().to(cpu)
            note = f"DirectML forward failed ({exc}); using CPU"
            return model, cpu, "cpu", note
        raise


def _probe_forward(model: Any, device: Any, kind: str, image_size: int) -> None:
    import torch

    dummy = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32, device=device)
    with torch.inference_mode():
        with sdpa_for_device(kind):
            model(dummy)


class FrozenDinov2:
    """Eval-only DINOv2. ``encode_video`` mean-pools L2-normalized frame CLS tokens."""

    def __init__(self, model: Any, device: Any, kind: str, info: Dict[str, Any]):
        self.model = model
        self.device = device
        self.kind = kind
        self.info = info

    def encode_video(self, rgb: np.ndarray) -> np.ndarray:
        import torch

        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError(f"expected RGB (T,H,W,3), got {rgb.shape}")
        if rgb.shape[0] < 1:
            raise ValueError("expected at least one frame")
        idx = _subsample_indices(rgb.shape[0], int(self.info["max_frames"]))
        frames = rgb[idx]
        batch = _preprocess_rgb(frames, int(self.info["image_size"]))
        batch = batch.to(device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            with sdpa_for_device(self.kind):
                out = self.model(batch)
        emb = _as_frame_matrix(out)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        video = emb.mean(axis=0)
        video = video / (np.linalg.norm(video) + 1e-8)
        return np.ascontiguousarray(video, dtype=np.float32)


def load_dinov2_encoder(cfg: Optional[Dict[str, Any]] = None) -> FrozenDinov2:
    """Load a frozen DINOv2 (or timm equivalent) and return an encoder."""
    cfg = cfg or {}
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise Dinov2LoadError(
            "torch is not installed",
            reason="MISSING_DEP",
            attempts=[{"source": "torch", "error": str(exc)}],
        ) from exc

    model_name = str(cfg.get("model_name") or "dinov2_vits14")
    cache_dir = Path(cfg["cache_dir"]).expanduser() if cfg.get("cache_dir") else default_cache_dir()
    preference = str(cfg.get("device") or "auto")
    max_frames = int(cfg.get("max_frames") or 8)
    image_size = round_to_patch(int(cfg.get("image_size") or 224), _patch_size(model_name))
    sources: Sequence[str] = cfg.get("sources") or DEFAULT_SOURCES
    hub_repo = str(cfg.get("hub_repo") or DEFAULT_HUB_REPO)
    hub_revision = str(cfg.get("hub_revision") or DEFAULT_HUB_REVISION)

    attempts: List[Dict[str, str]] = []
    model = None
    source_info: Dict[str, Any] = {}
    for source in sources:
        try:
            if source == "torch.hub":
                model, source_info = _load_torch_hub(model_name, cache_dir, hub_repo, hub_revision)
            elif source == "torchvision":
                model, source_info = _load_torchvision(model_name, cache_dir)
            elif source == "timm":
                model, source_info = _load_timm(model_name, cache_dir)
            else:
                raise RuntimeError(f"unknown DINOv2 source {source}")
            break
        except Exception as exc:
            attempts.append({"source": str(source), "error": f"{type(exc).__name__}: {exc}"})
            model = None

    if model is None:
        summary = "; ".join(f"{a['source']}: {a['error']}" for a in attempts) or "no sources configured"
        raise Dinov2LoadError(
            f"DINOv2 weights could not be loaded ({summary})",
            reason="MISSING_WEIGHTS",
            attempts=attempts,
        )

    model = _freeze(model)
    try:
        model, device, kind, device_note = _place_model(model, preference, image_size)
    except Exception as exc:
        attempts.append({"source": source_info.get("source", "device"), "error": f"{type(exc).__name__}: {exc}"})
        summary = "; ".join(f"{a['source']}: {a['error']}" for a in attempts)
        raise Dinov2LoadError(
            f"DINOv2 loaded but could not run on the selected device ({summary})",
            reason="MISSING_WEIGHTS",
            attempts=attempts,
        ) from exc

    embed_dim = getattr(model, "embed_dim", None) or getattr(model, "num_features", None)
    info: Dict[str, Any] = {
        "model_name": model_name,
        "device": kind,
        "image_size": image_size,
        "max_frames": max_frames,
        "cache_dir": str(cache_dir),
        "embed_dim": int(embed_dim) if embed_dim is not None else None,
        "dtype": "float32",
        "autocast": False,
        "frozen": True,
        "failed_sources": attempts,
        **source_info,
    }
    if device_note:
        info["device_note"] = device_note
    return FrozenDinov2(model, device, kind, info)
