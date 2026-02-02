import os

QWEN3_VL_WRAPPER_DIR = os.getenv(
    "QWEN3_VL_WRAPPER_DIR",
    "/mnt/data/ganrunhe/robot_arm_swap_project",
)
SAM3_VIDEO_TRACKER_DIR = os.getenv(
    "SAM3_VIDEO_TRACKER_DIR",
    "/mnt/data/ganrunhe/robot_arm_swap_project",
)

QWEN3_VL_WRAPPER_FILE = "get_bbox.py"
SAM3_VIDEO_TRACKER_FILE = "sam3_video_tracker.py"
