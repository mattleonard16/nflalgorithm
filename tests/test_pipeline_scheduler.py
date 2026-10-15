"""Scheduler coverage for the canonical NFL pregame workflow."""

from __future__ import annotations

import pytest

from config import config
from pipeline_jobs.service import JobService
from schema_migrations import MigrationManager
from scripts import pipeline_scheduler


@pytest.fixture()
def no_active_job(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_scheduler, "active_job_for_week", lambda season, week: None)


def test_scheduler_routes_current_week_through_production_runner(
    monkeypatch, no_active_job
) -> None:
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


def test_scheduler_surfaces_enqueue_failure(monkeypatch, no_active_job) -> None:
    scheduler = pipeline_scheduler.PipelineScheduler.__new__(pipeline_scheduler.PipelineScheduler)
    monkeypatch.setattr(pipeline_scheduler.nfl, "get_current_season", lambda roster=True: 2026)
    monkeypatch.setattr(pipeline_scheduler.nfl, "get_current_week", lambda: 1)

    class BrokenService:
        def create_pipeline_job(self, **kwargs):
            raise RuntimeError("database unavailable")

    monkeypatch.setattr(pipeline_scheduler, "JobService", BrokenService)

    with pytest.raises(RuntimeError, match="database unavailable"):
        scheduler.run_pregame_pipeline()


@pytest.fixture()
def job_database(tmp_path, monkeypatch) -> str:
    db_path = str(tmp_path / "scheduler.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)
    monkeypatch.setattr(config.database, "backend", "sqlite")
    monkeypatch.setattr(config.database, "path", db_path)
    MigrationManager(db_path).run()
    return db_path


def _scheduler_for(monkeypatch, season: int, week: int):
    scheduler = pipeline_scheduler.PipelineScheduler.__new__(pipeline_scheduler.PipelineScheduler)
    monkeypatch.setattr(pipeline_scheduler.nfl, "get_current_season", lambda roster=True: season)
    monkeypatch.setattr(pipeline_scheduler.nfl, "get_current_week", lambda: week)
    return scheduler


def _job_count() -> int:
    from utils.db import fetchone

    row = fetchone("SELECT COUNT(*) FROM pipeline_jobs")
    return int(row[0]) if row else 0


def test_scheduler_skips_while_a_job_for_the_week_is_queued(monkeypatch, job_database) -> None:
    scheduler = _scheduler_for(monkeypatch, 2026, 3)

    first = scheduler.run_pregame_pipeline()
    # A later tick (different minute bucket, so a different idempotency key)
    # must not stack a second run behind the queued one.
    second = scheduler.run_pregame_pipeline()

    assert first["status"] == "queued"
    assert second["status"] == "skipped"
    assert second["run_id"] == first["run_id"]
    assert _job_count() == 1


def test_scheduler_skips_while_a_job_is_running(monkeypatch, job_database) -> None:
    job = JobService().create_pipeline_job(season=2026, week=3, source="api")
    claimed = JobService().claim_next("worker-1")
    assert claimed is not None and claimed.job_id == job.job_id

    result = _scheduler_for(monkeypatch, 2026, 3).run_pregame_pipeline()

    assert result["status"] == "skipped"
    assert result["job_id"] == job.job_id
    assert _job_count() == 1


def test_scheduler_enqueues_when_only_other_weeks_or_finished_jobs_exist(
    monkeypatch, job_database
) -> None:
    from utils.db import execute

    JobService().create_pipeline_job(season=2026, week=2, source="api")
    done = JobService().create_pipeline_job(season=2026, week=3, source="api")
    execute("UPDATE pipeline_jobs SET status = 'completed' WHERE job_id = ?", (done.job_id,))

    result = _scheduler_for(monkeypatch, 2026, 3).run_pregame_pipeline()

    assert result["status"] == "queued"
    assert _job_count() == 3
