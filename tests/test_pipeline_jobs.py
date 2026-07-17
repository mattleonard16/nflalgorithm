"""Database-backed pipeline job control and worker coverage."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from pipeline_jobs.service import JobService
from pipeline_jobs.worker import PipelineWorker
from schema_migrations import MigrationManager
from utils.db import execute, fetchall, fetchone, get_table_columns, table_exists


@pytest.fixture()
def job_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "jobs.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    import config as cfg

    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")
    MigrationManager(db_path).run()
    return db_path


def test_job_schema_is_part_of_normal_migrations(job_db) -> None:
    assert table_exists("pipeline_jobs")
    assert table_exists("pipeline_stage_runs")
    assert table_exists("pipeline_artifacts")
    assert "claim_token" in get_table_columns("pipeline_jobs")


def test_stage_history_primary_key_is_attempt_specific(job_db) -> None:
    columns = fetchall("PRAGMA table_info(pipeline_stage_runs)")
    primary_key = [
        row[1] for row in sorted(columns, key=lambda row: row[5]) if row[5] > 0
    ]

    assert primary_key == ["run_id", "attempt", "stage_name"]


def test_create_job_is_idempotent_and_starts_queued(job_db) -> None:
    service = JobService()

    first = service.create_pipeline_job(
        season=2026,
        week=1,
        source="api",
        idempotency_key="2026-w1-refresh",
    )
    second = service.create_pipeline_job(
        season=2026,
        week=1,
        source="api",
        idempotency_key="2026-w1-refresh",
    )

    assert first.job_id == second.job_id
    assert first.run_id == second.run_id
    assert first.status == "queued"
    assert fetchone("SELECT status FROM pipeline_runs WHERE run_id = ?", (first.run_id,)) == (
        "queued",
    )


def test_idempotency_is_principal_scoped_and_payload_bound(job_db) -> None:
    service = JobService()
    first = service.create_pipeline_job(
        season=2026,
        week=1,
        source="api",
        requested_by="operator-a",
        idempotency_key="refresh",
    )
    second_principal = service.create_pipeline_job(
        season=2026,
        week=1,
        source="api",
        requested_by="operator-b",
        idempotency_key="refresh",
    )

    assert first.job_id != second_principal.job_id
    with pytest.raises(ValueError, match="different request"):
        service.create_pipeline_job(
            season=2026,
            week=2,
            source="api",
            requested_by="operator-a",
            idempotency_key="refresh",
        )


@pytest.mark.parametrize("season,week", [(1999, 1), (2101, 1), (2026, 0), (2026, 23)])
def test_create_job_rejects_invalid_period(job_db, season: int, week: int) -> None:
    with pytest.raises(ValueError):
        JobService().create_pipeline_job(season=season, week=week, source="api")


def test_worker_claims_fifo_and_records_stage_timeline(job_db) -> None:
    service = JobService()
    first = service.create_pipeline_job(season=2026, week=1, source="scheduler")
    service.create_pipeline_job(season=2026, week=2, source="scheduler")

    def runner(season, week, **kwargs):
        kwargs["on_stage_start"]("prepare_week", 0)
        kwargs["on_stage_result"](
            "prepare_week", 0, {"stage": "prepare_week", "status": "ok", "players": 42}
        )
        return {
            "season": season,
            "week": week,
            "success": True,
            "stages": [{"stage": "prepare_week", "status": "ok", "players": 42}],
            "errors": [],
        }

    worker = PipelineWorker(worker_id="test-worker", service=service, runner=runner)

    assert worker.process_once() is True
    assert fetchone("SELECT status FROM pipeline_runs WHERE run_id = ?", (first.run_id,)) == (
        "completed",
    )
    stage = fetchone(
        "SELECT stage_name, ordinal, status, result_json FROM pipeline_stage_runs WHERE run_id = ?",
        (first.run_id,),
    )
    assert stage[:3] == ("prepare_week", 0, "completed")
    assert json.loads(stage[3])["players"] == 42
    queued = fetchall("SELECT status FROM pipeline_jobs ORDER BY created_at, job_id")
    assert queued == [("completed",), ("queued",)]


def test_queued_job_can_be_cancelled_without_execution(job_db) -> None:
    service = JobService()
    job = service.create_pipeline_job(season=2026, week=1, source="cli")

    cancelled = service.request_cancel(job.run_id)

    assert cancelled.status == "cancelled"
    assert service.claim_next("worker-1") is None
    assert fetchone("SELECT status FROM pipeline_runs WHERE run_id = ?", (job.run_id,)) == (
        "cancelled",
    )


def test_failed_job_is_retried_then_terminal(job_db) -> None:
    service = JobService(retry_base_seconds=0)
    job = service.create_pipeline_job(
        season=2026,
        week=1,
        source="api",
        max_attempts=2,
    )

    def failing_runner(*args, **kwargs):
        return {
            "success": False,
            "stages": [{"stage": "odds", "status": "error", "error": "provider down"}],
            "errors": [{"stage": "odds", "error": "provider down"}],
        }

    worker = PipelineWorker(worker_id="retry-worker", service=service, runner=failing_runner)

    assert worker.process_once() is True
    assert fetchone("SELECT status FROM pipeline_jobs WHERE job_id = ?", (job.job_id,)) == (
        "retry_scheduled",
    )
    assert worker.process_once() is True
    assert fetchone("SELECT status FROM pipeline_jobs WHERE job_id = ?", (job.job_id,)) == (
        "failed",
    )
    assert fetchone("SELECT status FROM pipeline_runs WHERE run_id = ?", (job.run_id,)) == (
        "failed",
    )


def test_expired_worker_lease_is_reclaimed_and_fenced(job_db) -> None:
    service = JobService(retry_base_seconds=0, stale_after_seconds=30)
    queued = service.create_pipeline_job(season=2026, week=1, source="scheduler")
    first_claim = service.claim_next("dead-worker")
    assert first_claim is not None
    expired = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    execute(
        "UPDATE pipeline_jobs SET heartbeat_at = ? WHERE job_id = ?",
        (expired, queued.job_id),
    )

    second_claim = service.claim_next("replacement-worker")

    assert second_claim is not None
    assert second_claim.job_id == queued.job_id
    assert second_claim.worker_id == "replacement-worker"
    assert service.complete(first_claim, {"success": True}) is False
    assert service.get_job(queued.job_id).status == "running"


def test_explicit_retry_clears_prior_attempt_read_models(job_db) -> None:
    service = JobService(retry_base_seconds=0)
    queued = service.create_pipeline_job(
        season=2026,
        week=1,
        source="api",
        max_attempts=1,
    )
    claimed = service.claim_next("worker")
    assert claimed is not None
    service.record_stage_started(claimed.run_id, "prepare_week", 0)
    service.record_stage_result(
        claimed.run_id,
        "prepare_week",
        0,
        {"status": "ok", "stage": "prepare_week"},
    )
    service.fail(claimed, "odds unavailable", {"success": False})

    service.retry(queued.run_id)

    assert fetchone(
        "SELECT status, result_json FROM pipeline_stage_runs WHERE run_id = ?",
        (queued.run_id,),
    ) == ("queued", None)
    assert fetchone(
        "SELECT stages_completed, report_json, data_health_json FROM pipeline_runs WHERE run_id = ?",
        (queued.run_id,),
    ) == (0, None, None)
