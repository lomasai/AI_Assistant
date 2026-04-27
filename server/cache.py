"""Lightweight async TTL cache utilities for server optimization."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar


K = TypeVar("K")
V = TypeVar("V")


@dataclass(slots=True)
class CacheStats:
    """Basic cache statistics."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0


class AsyncTTLCache(Generic[K, V]):
    """Small async-safe TTL cache with bounded size."""

    def __init__(self, ttl_seconds: float = 60.0, max_items: int = 256) -> None:
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self.max_items = max(1, int(max_items))
        self._data: OrderedDict[K, tuple[float, V]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = CacheStats()

    async def get(self, key: K) -> V | None:
        """Get value for key if it exists and is not expired."""
        now = time.monotonic()
        async with self._lock:
            item = self._data.get(key)
            if item is None:
                self._stats.misses += 1
                return None

            expires_at, value = item
            if now >= expires_at:
                self._data.pop(key, None)
                self._stats.misses += 1
                return None

            # LRU touch
            self._data.move_to_end(key)
            self._stats.hits += 1
            return value

    async def set(self, key: K, value: V) -> None:
        """Set value with TTL, evicting oldest item if needed."""
        expires_at = time.monotonic() + self.ttl_seconds
        async with self._lock:
            if key in self._data:
                self._data.pop(key, None)
            self._data[key] = (expires_at, value)
            self._data.move_to_end(key)
            self._stats.sets += 1

            while len(self._data) > self.max_items:
                self._data.popitem(last=False)
                self._stats.evictions += 1

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._data.clear()

    async def stats(self) -> CacheStats:
        """Get a copy of current stats."""
        async with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                sets=self._stats.sets,
                evictions=self._stats.evictions,
            )
