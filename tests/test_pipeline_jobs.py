"""Database-backed pipeline job control and worker coverage."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
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
    assert table_exists("pipeline_odds_validations")
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


@pytest.mark.parametrize(
    "override",
    [
        {"priority": 2, "max_attempts": 3},
        {"priority": 1, "max_attempts": 4},
    ],
)
def test_idempotency_binds_priority_and_retry_budget(job_db, override) -> None:
    service = JobService()
    service.create_pipeline_job(
        season=2026,
        week=1,
        source="api",
        requested_by="operator",
        priority=1,
        max_attempts=3,
        idempotency_key="refresh",
    )

    with pytest.raises(ValueError, match="different request"):
        service.create_pipeline_job(
            season=2026,
            week=1,
            source="api",
            requested_by="operator",
            idempotency_key="refresh",
            **override,
        )


def test_concurrent_duplicate_idempotent_creation_returns_one_job(job_db) -> None:
    def create(_index: int):
        return JobService().create_pipeline_job(
            season=2026,
            week=1,
            source="api",
            requested_by="operator",
            priority=2,
            max_attempts=4,
            idempotency_key="same-request",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = list(pool.map(create, range(8)))

    assert len({job.job_id for job in jobs}) == 1
    assert fetchone("SELECT COUNT(*) FROM pipeline_jobs") == (1,)
    assert fetchone("SELECT COUNT(*) FROM pipeline_runs") == (1,)


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


def test_every_claim_receives_a_unique_attempt_token(job_db) -> None:
    service = JobService(retry_base_seconds=0)
    service.create_pipeline_job(season=2026, week=1, source="scheduler", max_attempts=2)

    first = service.claim_next("worker")
    assert first is not None and first.claim_token
    service.fail(first, "retry")
    second = service.claim_next("worker")

    assert second is not None and second.claim_token
    assert second.attempts == 2
    assert second.claim_token != first.claim_token


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


def test_reclaimed_attempt_is_fenced_even_with_same_worker_id(job_db) -> None:
    service = JobService(retry_base_seconds=0, stale_after_seconds=30)
    queued = service.create_pipeline_job(season=2026, week=1, source="scheduler")
    stale = service.claim_next("reused-worker")
    assert stale is not None
    execute(
        "UPDATE pipeline_jobs SET heartbeat_at = ? WHERE job_id = ?",
        ((datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(), queued.job_id),
    )

    current = service.claim_next("reused-worker")

    assert current is not None and current.claim_token != stale.claim_token
    assert service.complete(stale, {"success": True}) is False
    assert service.get_job(queued.job_id).status == "running"


def test_worker_identity_comparison_is_case_sensitive(job_db) -> None:
    service = JobService()
    service.create_pipeline_job(season=2026, week=1, source="scheduler")
    claimed = service.claim_next("Worker-A")
    assert claimed is not None

    case_equivalent = replace(claimed, worker_id="worker-a")

    assert service.heartbeat(case_equivalent) is False
    assert service.heartbeat(claimed) is True


def test_cancellation_wins_race_with_complete(job_db) -> None:
    service = JobService()
    queued = service.create_pipeline_job(season=2026, week=1, source="api")
    claimed = service.claim_next("worker")
    assert claimed is not None
    service.request_cancel(queued.run_id)

    assert service.complete(claimed, {"success": True}) is False
    current = service.get_job(queued.job_id)
    assert current is not None and current.status == "cancelled"
    assert fetchone("SELECT status FROM pipeline_runs WHERE run_id = ?", (queued.run_id,)) == (
        "cancelled",
    )


def test_cancellation_wins_race_with_fail(job_db) -> None:
    service = JobService(retry_base_seconds=0)
    queued = service.create_pipeline_job(
        season=2026, week=1, source="api", max_attempts=3
    )
    claimed = service.claim_next("worker")
    assert claimed is not None
    service.request_cancel(queued.run_id)

    assert service.fail(claimed, "provider unavailable", retryable=True) == "cancelled"
    current = service.get_job(queued.job_id)
    assert current is not None and current.status == "cancelled"
    assert current.status != "retry_scheduled"


def test_explicit_retry_preserves_prior_attempt_history(job_db) -> None:
    service = JobService(retry_base_seconds=0)
    queued = service.create_pipeline_job(
        season=2026,
        week=1,
        source="api",
        max_attempts=1,
    )
    claimed = service.claim_next("worker")
    assert claimed is not None
    service.record_stage_started(claimed, "prepare_week", 0)
    service.record_stage_result(
        claimed,
        "prepare_week",
        0,
        {"status": "ok", "stage": "prepare_week"},
    )
    service.fail(claimed, "odds unavailable", {"success": False})

    service.retry(queued.run_id)

    stage = fetchone(
        """
        SELECT attempt, status, result_json, finished_at
        FROM pipeline_stage_runs WHERE run_id = ?
        """,
        (queued.run_id,),
    )
    assert stage is not None
    assert stage[0:2] == (1, "completed")
    assert json.loads(stage[2])["status"] == "ok"
    assert stage[3] is not None
    assert fetchone(
        "SELECT stages_completed, report_json, data_health_json FROM pipeline_runs WHERE run_id = ?",
        (queued.run_id,),
    ) == (0, None, None)


def test_retry_records_a_new_stage_attempt_without_overwriting_history(job_db) -> None:
    service = JobService(retry_base_seconds=0)
    service.create_pipeline_job(
        season=2026, week=1, source="api", max_attempts=2
    )
    first = service.claim_next("worker")
    assert first is not None
    service.record_stage_started(first, "odds", 1)
    service.record_stage_result(
        first, "odds", 1, {"status": "error", "stage": "odds", "error": "down"}
    )
    service.fail(first, "down")
    second = service.claim_next("worker")
    assert second is not None
    service.record_stage_started(second, "odds", 1)

    rows = fetchall(
        """
        SELECT attempt, status, finished_at FROM pipeline_stage_runs
        WHERE run_id = ? AND stage_name = 'odds' ORDER BY attempt
        """,
        (first.run_id,),
    )
    assert rows[0][0:2] == (1, "failed")
    assert rows[0][2] is not None
    assert rows[1][0:2] == (2, "running")


def test_odds_validation_reason_and_metrics_are_persisted(job_db) -> None:
    service = JobService()
    service.create_pipeline_job(season=2026, week=1, source="scheduler")
    claimed = service.claim_next("worker")
    assert claimed is not None
    validation = {
        "valid": False,
        "reason_code": "market_coverage",
        "reason": "Odds cover 2/3 required event-market pairs",
        "market_coverage": 2 / 3,
        "odds_rows": 18,
    }

    service.record_stage_result(
        claimed,
        "odds",
        1,
        {
            "status": "error",
            "stage": "odds",
            "error": validation["reason"],
            "odds_validation": validation,
        },
    )

    row = fetchone(
        """
        SELECT attempt, valid, reason_code, reason, metrics_json
        FROM pipeline_odds_validations WHERE run_id = ?
        """,
        (claimed.run_id,),
    )
    assert row is not None
    assert row[0:4] == (
        1,
        0,
        "market_coverage",
        "Odds cover 2/3 required event-market pairs",
    )
    assert json.loads(row[4])["odds_rows"] == 18


def test_terminal_stale_recovery_closes_running_stage(job_db) -> None:
    service = JobService(retry_base_seconds=0, stale_after_seconds=30)
    service.create_pipeline_job(
        season=2026, week=1, source="scheduler", max_attempts=1
    )
    claimed = service.claim_next("dead-worker")
    assert claimed is not None
    service.record_stage_started(claimed, "materialize", 5)
    execute(
        "UPDATE pipeline_jobs SET heartbeat_at = ? WHERE job_id = ?",
        ((datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(), claimed.job_id),
    )

    assert service.recover_stale_jobs() == 1
    stage = fetchone(
        """
        SELECT status, finished_at, error_message FROM pipeline_stage_runs
        WHERE run_id = ? AND attempt = ? AND stage_name = 'materialize'
        """,
        (claimed.run_id, claimed.attempts),
    )
    assert stage is not None
    assert stage[0] == "failed"
    assert stage[1] is not None
    assert stage[2] == "worker heartbeat lease expired"


@pytest.mark.parametrize("heartbeat_mode", ["exception", "zero-row"])
def test_heartbeat_failure_stops_execution_before_completion(job_db, heartbeat_mode) -> None:
    service = JobService()
    queued = service.create_pipeline_job(season=2026, week=1, source="scheduler")
    original_heartbeat = service.heartbeat

    def broken_heartbeat(job):
        if heartbeat_mode == "exception":
            raise RuntimeError("database unavailable")
        return False

    service.heartbeat = broken_heartbeat  # type: ignore[method-assign]

    def cooperative_runner(*args, **kwargs):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not kwargs["cancellation_requested"]():
            time.sleep(0.01)
        return {"success": True, "stages": [], "errors": []}

    worker = PipelineWorker(
        worker_id="heartbeat-worker",
        service=service,
        runner=cooperative_runner,
        heartbeat_seconds=0.01,
    )

    assert worker.process_once() is True
    service.heartbeat = original_heartbeat  # type: ignore[method-assign]
    current = service.get_job(queued.job_id)
    assert current is not None and current.status == "running"


def test_lease_loss_after_runner_prevents_completion(job_db) -> None:
    service = JobService()
    queued = service.create_pipeline_job(season=2026, week=1, source="scheduler")

    def losing_runner(*args, **kwargs):
        execute(
            "UPDATE pipeline_jobs SET claim_token = ? WHERE job_id = ?",
            (b"replacement-token", queued.job_id),
        )
        return {"success": True, "stages": [], "errors": []}

    worker = PipelineWorker(
        worker_id="lease-worker",
        service=service,
        runner=losing_runner,
        heartbeat_seconds=60,
    )

    assert worker.process_once() is True
    current = service.get_job(queued.job_id)
    assert current is not None and current.status == "running"
