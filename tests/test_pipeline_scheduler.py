"""Scheduler coverage for the canonical NFL pregame workflow."""

from __future__ import annotations

import pytest

from scripts import pipeline_scheduler


def test_scheduler_routes_current_week_through_production_runner(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    scheduler = pipeline_scheduler.PipelineScheduler.__new__(pipeline_scheduler.PipelineScheduler)

    monkeypatch.setattr(pipeline_scheduler.nfl, "get_current_season", lambda roster=True: 2026)
    monkeypatch.setattr(pipeline_scheduler.nfl, "get_current_week", lambda: 1)

    class FakeJob:
        run_id = "run-1"
        job_id = "job-1"
        status = "queued"

    class FakeService:
        def create_pipeline_job(self, **kwargs):
            calls.append(kwargs)
            return FakeJob()

    monkeypatch.setattr(pipeline_scheduler, "JobService", FakeService)

    result = scheduler.run_pregame_pipeline()

    assert calls[0]["season"] == 2026
    assert calls[0]["week"] == 1
    assert calls[0]["source"] == "scheduler"
    assert result["success"] is True
    assert result["status"] == "queued"


def test_scheduler_surfaces_enqueue_failure(monkeypatch) -> None:
    scheduler = pipeline_scheduler.PipelineScheduler.__new__(pipeline_scheduler.PipelineScheduler)
    monkeypatch.setattr(pipeline_scheduler.nfl, "get_current_season", lambda roster=True: 2026)
    monkeypatch.setattr(pipeline_scheduler.nfl, "get_current_week", lambda: 1)

    class BrokenService:
        def create_pipeline_job(self, **kwargs):
            raise RuntimeError("database unavailable")

    monkeypatch.setattr(pipeline_scheduler, "JobService", BrokenService)

    with pytest.raises(RuntimeError, match="database unavailable"):
        scheduler.run_pregame_pipeline()
