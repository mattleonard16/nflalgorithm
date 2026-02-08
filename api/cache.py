"""Simple in-memory TTL cache for hot API endpoints."""

from __future__ import annotations

import time
import threading
from typing import Any, Dict, Optional, Tuple


class EndpointCache:
    """Thread-safe in-memory cache with TTL and max size, keyed by arbitrary string keys."""

    def __init__(self, default_ttl: int = 300, max_size: int = 500):
        self._store: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl
        self._max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if it exists and hasn't expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.time() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Cache a value with optional TTL override."""
        with self._lock:
            # Evict expired entries if at capacity
            if len(self._store) >= self._max_size:
                self._evict_expired()
            # If still at capacity, evict oldest entry
            if len(self._store) >= self._max_size:
                oldest_key = min(self._store, key=lambda k: self._store[k][1])
                del self._store[oldest_key]

            expires_at = time.time() + (ttl if ttl is not None else self._default_ttl)
            self._store[key] = (value, expires_at)

    def _evict_expired(self) -> None:
        """Remove all expired entries (must be called under lock)."""
        now = time.time()
        expired = [k for k, (_, exp) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]

    def invalidate(self, prefix: str) -> int:
        """Invalidate all cache entries whose key starts with prefix."""
        with self._lock:
            keys_to_remove = [k for k in self._store if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._store[k]
            return len(keys_to_remove)

    def invalidate_all(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        """Return number of cached entries (including potentially expired)."""
        with self._lock:
            return len(self._store)


def make_cache_key(endpoint: str, **kwargs: Any) -> str:
    """Build a deterministic cache key from endpoint name and params."""
    parts = [endpoint]
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if v is not None:
            parts.append(f"{k}={v}")
    return ":".join(parts)


# Global cache instance for hot endpoints (5-minute default TTL, 500 entry max)
value_bets_cache = EndpointCache(default_ttl=300, max_size=500)
