"""Per-client-IP token-bucket rate limiting for the public API surface.

Two tiers protect different costs: the auth prefix guards bcrypt CPU (each
login/register burns hundreds of milliseconds of unauthenticated work), while
the global tier bounds full-table export and read traffic.

Buckets live in this process only. Under multi-worker uvicorn each worker keeps
its own buckets, so the effective limit is `workers x configured limit`. That
is accepted for launch; a shared store (Redis) is the fix when one process is
no longer enough.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

logger = logging.getLogger(__name__)

AUTH_PATH_PREFIX = "/api/auth/"
GLOBAL_PATH_PREFIX = "/api/"

DEFAULT_AUTH_PER_MIN = 10
DEFAULT_GLOBAL_PER_MIN = 120

AUTH_LIMIT_ENV = "RATE_LIMIT_AUTH_PER_MIN"
GLOBAL_LIMIT_ENV = "RATE_LIMIT_GLOBAL_PER_MIN"
TRUSTED_PROXY_ENV = "TRUSTED_PROXY"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

_WINDOW_SECONDS = 60.0


def _read_limit(name: str, default: int) -> int:
    """Read a per-minute limit from the environment, failing loud on garbage."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"Invalid {name}={raw!r}. Set a non-negative integer of requests per minute "
            "(0 disables this tier)."
        ) from exc
    if value < 0:
        raise ValueError(
            f"Invalid {name}={raw!r}. Set a non-negative integer of requests per minute "
            "(0 disables this tier)."
        )
    return value


def trusted_proxy_enabled() -> bool:
    return os.getenv(TRUSTED_PROXY_ENV, "").strip().lower() in _TRUTHY


def client_identifier(scope: dict[str, Any], *, trust_proxy: bool) -> str:
    """Resolve the client address, honoring X-Forwarded-For only behind a proxy.

    Trusting the header unconditionally would let any caller mint a fresh
    identity per request and bypass the limiter entirely.
    """
    if trust_proxy:
        for raw_name, raw_value in scope.get("headers") or ():
            if raw_name == b"x-forwarded-for":
                forwarded = raw_value.decode("latin-1").split(",")[0].strip()
                if forwarded:
                    return forwarded
    client = scope.get("client")
    if client:
        return str(client[0])
    return "unknown"


def _hash_identifier(identifier: str) -> str:
    """Hash the client address so logs carry no raw IP."""
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class _Decision:
    allowed: bool
    tier: str
    retry_after: int


class _TokenBucketTier:
    """Fixed per-minute token budget refilled continuously, keyed by client."""

    def __init__(self, name: str, limit_per_minute: int) -> None:
        self.name = name
        self.limit = limit_per_minute
        self._refill_per_second = limit_per_minute / _WINDOW_SECONDS if limit_per_minute else 0.0
        self._state: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.limit > 0

    def consume(self, key: str, now: float) -> _Decision:
        if not self.enabled:
            return _Decision(allowed=True, tier=self.name, retry_after=0)

        with self._lock:
            tokens, last_seen = self._state.get(key, (float(self.limit), now))
            tokens = min(float(self.limit), tokens + (now - last_seen) * self._refill_per_second)
            if tokens >= 1.0:
                self._state[key] = (tokens - 1.0, now)
                return _Decision(allowed=True, tier=self.name, retry_after=0)
            self._state[key] = (tokens, now)
            deficit = 1.0 - tokens

        retry_after = max(1, int(deficit / self._refill_per_second) + 1)
        return _Decision(allowed=False, tier=self.name, retry_after=retry_after)

    def reset(self) -> None:
        with self._lock:
            self._state.clear()


class RateLimitMiddleware:
    """Pure-ASGI limiter applied to `/api/` traffic ahead of route handling."""

    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        *,
        auth_per_minute: int | None = None,
        global_per_minute: int | None = None,
        time_source: Callable[[], float] | None = None,
    ) -> None:
        self.app = app
        self._time = time_source or time.monotonic
        auth_limit = (
            _read_limit(AUTH_LIMIT_ENV, DEFAULT_AUTH_PER_MIN)
            if auth_per_minute is None
            else auth_per_minute
        )
        global_limit = (
            _read_limit(GLOBAL_LIMIT_ENV, DEFAULT_GLOBAL_PER_MIN)
            if global_per_minute is None
            else global_per_minute
        )
        self.auth_tier = _TokenBucketTier("auth", auth_limit)
        self.global_tier = _TokenBucketTier("global", global_limit)

    def _tiers_for(self, path: str) -> Iterable[_TokenBucketTier]:
        if not path.startswith(GLOBAL_PATH_PREFIX):
            return ()
        if path.startswith(AUTH_PATH_PREFIX):
            # The auth bucket is the binding constraint; the global bucket still
            # counts these requests so a login flood cannot crowd out reads.
            return (self.auth_tier, self.global_tier)
        return (self.global_tier,)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        tiers = self._tiers_for(scope.get("path", ""))
        if tiers:
            key = client_identifier(scope, trust_proxy=trusted_proxy_enabled())
            now = self._time()
            for tier in tiers:
                decision = tier.consume(key, now)
                if not decision.allowed:
                    logger.warning(
                        "rate limit exceeded: tier=%s client=%s path=%s retry_after=%ss",
                        decision.tier,
                        _hash_identifier(key),
                        scope.get("path", ""),
                        decision.retry_after,
                    )
                    await _send_429(send, decision.retry_after)
                    return

        await self.app(scope, receive, send)


async def _send_429(send: Any, retry_after: int) -> None:
    body = b'{"detail":"Rate limit exceeded"}'
    await send(
        {
            "type": "http.response.start",
            "status": 429,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"retry-after", str(retry_after).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


__all__ = [
    "AUTH_LIMIT_ENV",
    "GLOBAL_LIMIT_ENV",
    "TRUSTED_PROXY_ENV",
    "RateLimitMiddleware",
    "client_identifier",
    "trusted_proxy_enabled",
]
