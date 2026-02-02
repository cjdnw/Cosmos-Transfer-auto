import importlib.util
import os
from types import ModuleType
from typing import Type

from robot_arm_swap.config import QWEN3_VL_WRAPPER_FILE, SAM3_VIDEO_TRACKER_FILE


def _load_module_from_path(module_path: str, module_name: str) -> ModuleType:
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Wrapper module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_qwen3_vl_class(wrapper_dir: str) -> Type:
    module_path = os.path.join(wrapper_dir, QWEN3_VL_WRAPPER_FILE)
    module = _load_module_from_path(module_path, "qwen3_vl_wrapper")
    if not hasattr(module, "QwenVL3"):
        raise AttributeError("Qwen wrapper is missing class `QwenVL3`")
    return module.QwenVL3


def load_sam3_video_tracker_class(wrapper_dir: str) -> Type:
    module_path = os.path.join(wrapper_dir, SAM3_VIDEO_TRACKER_FILE)
    module = _load_module_from_path(module_path, "sam3_video_tracker_wrapper")
    if not hasattr(module, "SAM3VideoTracker"):
        raise AttributeError("SAM3 wrapper is missing class `SAM3VideoTracker`")
    return module.SAM3VideoTracker
