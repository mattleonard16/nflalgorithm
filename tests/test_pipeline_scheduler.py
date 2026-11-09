"""Scheduler coverage for the canonical NFL pregame workflow."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from scripts import pipeline_scheduler
from utils.current_week import SeasonOverError

T0 = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


class FakeJob:
    run_id = "run-1"
    job_id = "job-1"
    status = "queued"


@pytest.fixture
def harness(monkeypatch):
    """Stub the clock, schedule, and DB boundary; expose the recorded enqueue calls."""
    state = {"now": T0, "active": False, "week": (2026, 1), "calls": []}

    def resolve():
        if isinstance(state["week"], Exception):
            raise state["week"]
        return state["week"]

    class FakeService:
        def create_pipeline_job(self, **kwargs):
            state["calls"].append(kwargs)
            return FakeJob()

    monkeypatch.setattr(pipeline_scheduler, "_utcnow", lambda: state["now"])
    monkeypatch.setattr(pipeline_scheduler, "resolve_current_week", resolve)
    monkeypatch.setattr(
        pipeline_scheduler, "fetchone", lambda *a, **k: (1,) if state["active"] else None
    )
    monkeypatch.setattr(pipeline_scheduler, "JobService", FakeService)
    monkeypatch.setattr(pipeline_scheduler.config.pipeline, "update_interval_minutes", 360)
    scheduler = pipeline_scheduler.PipelineScheduler.__new__(pipeline_scheduler.PipelineScheduler)
    return scheduler, state


def test_scheduler_routes_current_week_through_production_runner(harness) -> None:
    scheduler, state = harness
    result = scheduler.run_pregame_pipeline()

    call = state["calls"][0]
    assert (call["season"], call["week"], call["source"]) == (2026, 1, "scheduler")
    assert result == {"success": True, "status": "queued", "run_id": "run-1", "job_id": "job-1"}


def test_ticks_in_one_window_share_an_idempotency_key(harness) -> None:
    scheduler, state = harness
    scheduler.run_pregame_pipeline()
    state["now"] = T0 + timedelta(minutes=15)
    scheduler.run_pregame_pipeline()

    keys = [c["idempotency_key"] for c in state["calls"]]
    assert keys[0] == keys[1]


def test_tick_in_the_next_window_gets_a_new_idempotency_key(harness) -> None:
    scheduler, state = harness
    scheduler.run_pregame_pipeline()
    state["now"] = T0 + timedelta(minutes=360)
    scheduler.run_pregame_pipeline()

    keys = [c["idempotency_key"] for c in state["calls"]]
    assert keys[0] != keys[1]


def test_active_job_for_the_week_blocks_enqueue(harness) -> None:
    scheduler, state = harness
    state["active"] = True
    result = scheduler.run_pregame_pipeline()

    assert state["calls"] == []
    assert result["status"] == "skipped"
    assert result["reason"] == "active_job"


def test_offseason_blocks_enqueue(harness) -> None:
    scheduler, state = harness
    state["week"] = SeasonOverError("every scheduled game has already kicked off")
    result = scheduler.run_pregame_pipeline()

    assert state["calls"] == []
    assert result["status"] == "skipped"
    assert result["reason"] == "offseason"


def test_missing_schedule_fails_instead_of_passing_as_offseason(harness) -> None:
    scheduler, state = harness
    state["week"] = ValueError("the schedule frame is empty")

    with pytest.raises(ValueError, match="schedule frame is empty"):
        scheduler.run_pregame_pipeline()
    assert state["calls"] == []


def test_scheduler_surfaces_enqueue_failure(harness, monkeypatch) -> None:
    scheduler, _ = harness

    class BrokenService:
        def create_pipeline_job(self, **kwargs):
            raise RuntimeError("database unavailable")

    monkeypatch.setattr(pipeline_scheduler, "JobService", BrokenService)

    with pytest.raises(RuntimeError, match="database unavailable"):
        scheduler.run_pregame_pipeline()
