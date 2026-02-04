import os
import shutil
import subprocess
from typing import List, Optional

class CmdError(RuntimeError):
    pass

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def run_cmd(args: List[str], check: bool = True) -> None:
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and proc.returncode != 0:
        raise CmdError(
            f"Command failed ({proc.returncode}): {' '.join(args)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

def assert_ffmpeg() -> None:
    if which("ffmpeg") is None:
        raise CmdError("ffmpeg not found in PATH. Please install ffmpeg and ensure `ffmpeg` is callable.")

def sorted_glob(pattern: str):
    import glob
    paths = glob.glob(pattern)
    paths.sort()
    return paths

def read_image(path: str):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def read_mask(path: str):
    import cv2
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m
