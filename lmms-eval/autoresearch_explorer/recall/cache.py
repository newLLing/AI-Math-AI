"""
召回结果缓存模块
支持内存缓存 + 磁盘缓存（JSON），带 TTL
"""
import os
import json
import hashlib
import time
from typing import Optional, Any
from pathlib import Path


class RecallCache:
    """简单的召回结果缓存，支持内存和磁盘两级缓存"""

    def __init__(self, cache_dir: str = ".autoresearch_cache", ttl_seconds: int = 604800):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self._memory = {}

    def _make_key(self, *parts) -> str:
        raw = "|".join(str(p) for p in parts)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _disk_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, *parts) -> Optional[Any]:
        key = self._make_key(*parts)
        # 1. 查内存
        if key in self._memory:
            entry = self._memory[key]
            if time.time() - entry["ts"] < self.ttl_seconds:
                return entry["data"]
            del self._memory[key]

        # 2. 查磁盘
        path = self._disk_path(key)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                if time.time() - entry["ts"] < self.ttl_seconds:
                    self._memory[key] = entry
                    return entry["data"]
                else:
                    path.unlink(missing_ok=True)
            except Exception:
                pass
        return None

    def set(self, data: Any, *parts):
        key = self._make_key(*parts)
        entry = {"ts": time.time(), "data": data}
        self._memory[key] = entry
        path = self._disk_path(key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False)
        except Exception:
            pass
