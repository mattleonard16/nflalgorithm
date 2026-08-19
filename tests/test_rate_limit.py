"""Tests for per-client-IP API rate limiting."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.rate_limit import (
    AUTH_LIMIT_ENV,
    GLOBAL_LIMIT_ENV,
    TRUSTED_PROXY_ENV,
    RateLimitMiddleware,
    client_identifier,
)


class _FrozenClock:
    """Manual clock so limit tests never sleep through a refill window."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_client(
    *,
    auth_per_minute: int | None = None,
    global_per_minute: int | None = None,
    clock: _FrozenClock | None = None,
) -> TestClient:
    app = FastAPI()

    @app.post("/api/auth/login")
    def login() -> dict:
        return {"ok": True}

    @app.get("/api/value-bets")
    def value_bets() -> dict:
        return {"ok": True}

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    app.add_middleware(
        RateLimitMiddleware,
        auth_per_minute=auth_per_minute,
        global_per_minute=global_per_minute,
        time_source=clock or _FrozenClock(),
    )
    return TestClient(app)


def test_auth_tier_allows_tenth_and_rejects_eleventh_request() -> None:
    client = _make_client(auth_per_minute=10, global_per_minute=0)

    for attempt in range(10):
        response = client.post("/api/auth/login")
        assert response.status_code == 200, f"request {attempt + 1} should pass"

    blocked = client.post("/api/auth/login")

    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) >= 1


def test_auth_bucket_refills_after_the_window() -> None:
    clock = _FrozenClock()
    client = _make_client(auth_per_minute=10, global_per_minute=0, clock=clock)

    for _ in range(10):
        assert client.post("/api/auth/login").status_code == 200
    assert client.post("/api/auth/login").status_code == 429

    clock.advance(60.0)

    assert client.post("/api/auth/login").status_code == 200


def test_global_tier_is_independent_of_the_auth_tier() -> None:
    client = _make_client(auth_per_minute=2, global_per_minute=120)

    for _ in range(2):
        assert client.post("/api/auth/login").status_code == 200
    assert client.post("/api/auth/login").status_code == 429

    # Exhausting the strict auth bucket must not spend the read budget.
    assert client.get("/api/value-bets").status_code == 200


def test_global_tier_limits_non_auth_api_paths() -> None:
    client = _make_client(auth_per_minute=0, global_per_minute=3)

    for _ in range(3):
        assert client.get("/api/value-bets").status_code == 200

    blocked = client.get("/api/value-bets")

    assert blocked.status_code == 429
    assert "retry-after" in blocked.headers


def test_paths_outside_the_api_prefix_are_not_limited() -> None:
    client = _make_client(auth_per_minute=1, global_per_minute=1)

    for _ in range(5):
        assert client.get("/healthz").status_code == 200


def test_zero_disables_the_auth_tier() -> None:
    client = _make_client(auth_per_minute=0, global_per_minute=0)

    for _ in range(25):
        assert client.post("/api/auth/login").status_code == 200


def test_forwarded_for_is_ignored_without_trusted_proxy(monkeypatch) -> None:
    monkeypatch.delenv(TRUSTED_PROXY_ENV, raising=False)
    client = _make_client(auth_per_minute=2, global_per_minute=0)

    # A spoofed per-request identity must not buy extra budget.
    assert client.post("/api/auth/login", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert client.post("/api/auth/login", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200
    blocked = client.post("/api/auth/login", headers={"X-Forwarded-For": "3.3.3.3"})

    assert blocked.status_code == 429


def test_forwarded_for_is_honored_with_trusted_proxy(monkeypatch) -> None:
    monkeypatch.setenv(TRUSTED_PROXY_ENV, "true")
    client = _make_client(auth_per_minute=1, global_per_minute=0)

    assert client.post("/api/auth/login", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    # Separate upstream client, so it gets its own bucket.
    assert client.post("/api/auth/login", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200
    repeat = client.post("/api/auth/login", headers={"X-Forwarded-For": "1.1.1.1"})

    assert repeat.status_code == 429


def test_client_identifier_uses_leftmost_forwarded_hop() -> None:
    scope = {
        "client": ("10.0.0.9", 5000),
        "headers": [(b"x-forwarded-for", b"203.0.113.7, 70.41.3.18")],
    }

    assert client_identifier(scope, trust_proxy=True) == "203.0.113.7"
    assert client_identifier(scope, trust_proxy=False) == "10.0.0.9"


def test_non_integer_limit_fails_loud(monkeypatch) -> None:
    monkeypatch.setenv(AUTH_LIMIT_ENV, "ten")

    with pytest.raises(ValueError, match=AUTH_LIMIT_ENV):
        RateLimitMiddleware(lambda scope, receive, send: None)


def test_negative_limit_fails_loud(monkeypatch) -> None:
    monkeypatch.delenv(AUTH_LIMIT_ENV, raising=False)
    monkeypatch.setenv(GLOBAL_LIMIT_ENV, "-5")

    with pytest.raises(ValueError, match=GLOBAL_LIMIT_ENV):
        RateLimitMiddleware(lambda scope, receive, send: None)


def test_env_defaults_apply_when_unset(monkeypatch) -> None:
    monkeypatch.delenv(AUTH_LIMIT_ENV, raising=False)
    monkeypatch.delenv(GLOBAL_LIMIT_ENV, raising=False)

    middleware = RateLimitMiddleware(lambda scope, receive, send: None)

    assert middleware.auth_tier.limit == 10
    assert middleware.global_tier.limit == 120
