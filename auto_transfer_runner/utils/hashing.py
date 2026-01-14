from __future__ import annotations

import hashlib
from pathlib import Path


def short_hash(s: str, n: int = 12) -> str:
    """短 hash：用于命名，避免过长。"""
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:n]


def text_md5(text: str, n: int | None = 16) -> str:
    """MD5 hash of a string (optionally shortened)."""
    h = hashlib.md5(text.encode("utf-8"))
    digest = h.hexdigest()
    return digest if n is None else digest[:n]


def video_fingerprint(video_path: Path, mode: str = "stat") -> str:
    """
    计算“视频指纹”，用于缓存复用（同一输入视频 => 同一控制信号缓存目录）。

    mode:
      - path: 仅基于绝对路径（最快，但移动/改名会失效）
      - stat: 基于 绝对路径 + 文件大小 + 修改时间（推荐默认：快且足够稳）
      - md5 : 基于文件内容（最稳，但需要读完整视频，IO 更慢）
    """
    vp = str(video_path.resolve())

    if mode == "path":
        return short_hash(vp, 12)

    if mode == "stat":
        st = video_path.stat()
        key = f"{vp}|{st.st_size}|{st.st_mtime_ns}"
        return short_hash(key, 12)

    if mode == "md5":
        h = hashlib.md5()
        with video_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()[:12]

    raise ValueError(f"Unknown fingerprint mode: {mode}")


def video_cache_key(video_path: Path, mode: str = "stat") -> str:
    """缓存 key：mp4 文件名 + 指纹。"""
    fingerprint = video_fingerprint(video_path, mode=mode)
    return f"{video_path.name}__{fingerprint}"
