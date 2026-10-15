"""Live NBA schedule fetch: negative caching and single-flight behavior."""

from __future__ import annotations

import threading

import pytest

import api.nba_router as router_mod

GAME = {
    "game_id": "001",
    "game_date": "2026-03-10",
    "home_team": "LAL",
    "away_team": "BOS",
    "status": "Final",
    "home_score": 100,
    "away_score": 99,
}


class _Clock:
    def __init__(self) -> None:
        self.now = 10_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(router_mod.time, "time", clock)
    monkeypatch.setattr(
        router_mod, "_schedule_cache", {"ts": 0.0, "data": None, "failed": False}
    )
    return clock


def test_failure_is_negative_cached(clock, monkeypatch):
    calls = []

    def failing_load():
        calls.append(1)
        raise TimeoutError("nba.com down")

    monkeypatch.setattr(router_mod, "_load_live_scoreboard", failing_load)

    assert router_mod._fetch_live_schedule() == []
    assert router_mod._fetch_live_schedule() == []
    assert len(calls) == 1

    clock.now += router_mod._FAILURE_TTL + 1
    assert router_mod._fetch_live_schedule() == []
    assert len(calls) == 2


def test_failure_serves_stale_schedule(clock, monkeypatch):
    monkeypatch.setattr(router_mod, "_load_live_scoreboard", lambda: [GAME])
    assert router_mod._fetch_live_schedule() == [GAME]

    clock.now += router_mod._CACHE_TTL + 1

    def failing_load():
        raise TimeoutError("nba.com down")

    monkeypatch.setattr(router_mod, "_load_live_scoreboard", failing_load)

    assert router_mod._fetch_live_schedule() == [GAME]


def test_success_after_failure_is_cached_for_full_ttl(clock, monkeypatch):
    def failing_load():
        raise TimeoutError("nba.com down")

    monkeypatch.setattr(router_mod, "_load_live_scoreboard", failing_load)
    router_mod._fetch_live_schedule()

    clock.now += router_mod._FAILURE_TTL + 1
    calls = []
    monkeypatch.setattr(router_mod, "_load_live_scoreboard", lambda: calls.append(1) or [GAME])

    assert router_mod._fetch_live_schedule() == [GAME]
    clock.now += router_mod._FAILURE_TTL + 1
    assert router_mod._fetch_live_schedule() == [GAME]
    assert len(calls) == 1


def test_concurrent_caller_does_not_wait_behind_upstream_when_stale_exists(clock, monkeypatch):
    monkeypatch.setattr(router_mod, "_load_live_scoreboard", lambda: [GAME])
    router_mod._fetch_live_schedule()
    clock.now += router_mod._CACHE_TTL + 1

    started = threading.Event()
    release = threading.Event()

    def slow_load():
        started.set()
        release.wait(timeout=5)
        return [GAME]

    monkeypatch.setattr(router_mod, "_load_live_scoreboard", slow_load)
    worker = threading.Thread(target=router_mod._fetch_live_schedule)
    worker.start()
    try:
        assert started.wait(timeout=5)
        # The upstream call is in flight; this caller must not block on it.
        assert router_mod._fetch_live_schedule() == [GAME]
        # And the cache lock itself is not held across the network call.
        assert router_mod._schedule_lock.acquire(timeout=1)
        router_mod._schedule_lock.release()
    finally:
        release.set()
        worker.join(timeout=5)


def test_scoreboard_uses_short_timeout(monkeypatch):
    seen = {}

    class FakeBoard:
        def __init__(self, timeout=30, **kwargs):
            seen["timeout"] = timeout

        def get_dict(self):
            return {"scoreboard": {"games": []}}

    from nba_api.live.nba.endpoints import scoreboard

    monkeypatch.setattr(scoreboard, "ScoreBoard", FakeBoard)

    assert router_mod._load_live_scoreboard() == []
    assert seen["timeout"] == router_mod._SCOREBOARD_TIMEOUT <= 10
