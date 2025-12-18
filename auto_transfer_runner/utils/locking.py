from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileLockConfig:
    """锁文件配置：用于并发时避免重复生成同一缓存。"""
    timeout_sec: int = 6 * 3600
    stale_lock_sec: int = 3 * 3600  # 锁文件过旧可视为异常（可抢占）


class FileLock:
    """
    简单文件锁（基于原子创建 .lock 文件）：
    - acquire()：抢锁
    - release()：释放锁
    """

    def __init__(self, lock_path: Path, config: FileLockConfig):
        self.lock_path = lock_path
        self.config = config
        self.acquired = False

    def _is_stale(self) -> bool:
        if not self.lock_path.exists():
            return False
        age = time.time() - self.lock_path.stat().st_mtime
        return age > self.config.stale_lock_sec

    def acquire(self) -> None:
        start = time.time()
        while True:
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"pid={os.getpid()}\n".encode("utf-8"))
                os.close(fd)
                self.acquired = True
                return
            except FileExistsError:
                if self._is_stale():
                    try:
                        self.lock_path.unlink()
                    except Exception:
                        pass
                    continue

                if time.time() - start > self.config.timeout_sec:
                    raise TimeoutError(f"Waited too long for lock: {self.lock_path}")

                time.sleep(2)

    def release(self) -> None:
        if self.acquired and self.lock_path.exists():
            try:
                self.lock_path.unlink()
            finally:
                self.acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False
