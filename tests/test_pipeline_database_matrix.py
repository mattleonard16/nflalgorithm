"""Backend-neutral failure-safety checks for the durable pipeline runtime."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from threading import Barrier

import pytest

from config import config
from pipeline_jobs.service import JobService
from pipeline_jobs.worker import PipelineWorker
from schema_migrations import MigrationManager
from utils.db import execute, fetchone, get_connection


@pytest.fixture()
def runtime_database(tmp_path, monkeypatch) -> str:
    backend = os.getenv("TEST_DB_BACKEND", "sqlite").lower()
    if backend == "sqlite":
        db_path = str(tmp_path / "pipeline-matrix.db")
        monkeypatch.setenv("DB_BACKEND", "sqlite")
        monkeypatch.setenv("SQLITE_DB_PATH", db_path)
        monkeypatch.setattr(config.database, "backend", "sqlite")
        monkeypatch.setattr(config.database, "path", db_path)
        MigrationManager(db_path).run()
    else:
        monkeypatch.setenv("DB_BACKEND", "mysql")
        monkeypatch.setenv("DB_URL", os.environ["TEST_DB_URL"])
        monkeypatch.setattr(config.database, "backend", "mysql")
        monkeypatch.setattr(config.database, "db_url", os.environ["TEST_DB_URL"])
        MigrationManager("unused-mysql-path").run()

    with get_connection() as conn:
        for table in (
            "pipeline_artifacts",
            "pipeline_card_staging",
            "pipeline_odds_validations",
            "pipeline_stage_runs",
            "pipeline_jobs",
            "pipeline_runs",
        ):
            execute(f"DELETE FROM {table}", conn=conn)
        conn.commit()
    return backend


def test_concurrent_workers_never_double_claim(runtime_database) -> None:
    service = JobService()
    queued = [
        service.create_pipeline_job(season=2026, week=week, source="scheduler")
        for week in range(1, 5)
    ]
    contender_count = 8
    barrier = Barrier(contender_count)

    def claim(index: int):
        barrier.wait()
        return JobService().claim_next(f"worker-{index}")

    with ThreadPoolExecutor(max_workers=contender_count) as pool:
        claims = list(pool.map(claim, range(contender_count)))

    winners = [claim for claim in claims if claim is not None]
    claimed_ids = {winner.job_id for winner in winners}
    assert len(claimed_ids) == len(winners)

    # SKIP LOCKED may produce a transient no-work result while all visible rows
    # are locked. A normal worker polls again; drain the remaining claims and
    # prove every job is eventually claimed exactly once.
    while remaining := JobService().claim_next(f"drain-worker-{len(claimed_ids)}"):
        assert remaining.job_id not in claimed_ids
        claimed_ids.add(remaining.job_id)
    assert claimed_ids == {job.job_id for job in queued}


def test_worker_crash_recovers_and_fences_stale_attempt(runtime_database) -> None:
    service = JobService(retry_base_seconds=0, stale_after_seconds=30)
    queued = service.create_pipeline_job(season=2026, week=1, source="scheduler")

    def crash_runner(*args, **kwargs):
        raise SystemExit("simulated worker process crash")

    crashed_worker = PipelineWorker(
        worker_id="crashed-worker",
        service=service,
        runner=crash_runner,
        heartbeat_seconds=1,
    )
    with pytest.raises(SystemExit, match="simulated worker process crash"):
        crashed_worker.process_once()

    stale_claim = service.get_job(queued.job_id)
    assert stale_claim is not None and stale_claim.status == "running"
    expired = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    execute(
        "UPDATE pipeline_jobs SET heartbeat_at = ? WHERE job_id = ?",
        (expired, queued.job_id),
    )

    replacement = PipelineWorker(
        worker_id="replacement-worker",
        service=service,
        runner=lambda *args, **kwargs: {"success": True, "stages": [], "errors": []},
    )
    assert replacement.process_once() is True

    recovered = service.get_job(queued.job_id)
    assert recovered is not None
    assert recovered.status == "completed"
    assert recovered.worker_id == "replacement-worker"
    assert service.complete(stale_claim, {"success": True}) is False
    with pytest.raises(RuntimeError, match="lease is no longer active"):
        service.record_stage_result(
            stale_claim,
            "materialize",
            5,
            {"status": "ok", "card_size": 99},
        )


def test_reclaim_fences_same_and_case_equivalent_worker_ids(runtime_database) -> None:
    service = JobService(retry_base_seconds=0, stale_after_seconds=30)
    queued = service.create_pipeline_job(season=2026, week=1, source="scheduler")
    stale = service.claim_next("Worker-A")
    assert stale is not None
    expired = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    execute(
        "UPDATE pipeline_jobs SET heartbeat_at = ? WHERE job_id = ?",
        (expired, queued.job_id),
    )

    replacement = service.claim_next("Worker-A")

    assert replacement is not None
    assert replacement.attempts == stale.attempts + 1
    assert replacement.claim_token != stale.claim_token
    assert service.complete(stale, {"success": True}) is False
    assert service.heartbeat(replace(replacement, worker_id="worker-a")) is False
    assert service.heartbeat(replacement) is True


def test_mixed_version_tokenless_lease_recovers_on_real_backend(runtime_database) -> None:
    service = JobService(retry_base_seconds=0, stale_after_seconds=30)
    queued = service.create_pipeline_job(season=2026, week=1, source="scheduler")
    legacy_claim = service.claim_next("legacy-worker")
    assert legacy_claim is not None
    service.record_stage_started(legacy_claim, "odds", 1)
    expired = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    execute(
        "UPDATE pipeline_jobs SET claim_token = NULL, heartbeat_at = ? WHERE job_id = ?",
        (expired, queued.job_id),
    )

    assert service.recover_stale_jobs() == 1
    recovered = service.get_job(queued.job_id)
    assert recovered is not None
    assert recovered.status == "retry_scheduled"
    assert recovered.worker_id is None
    assert recovered.claim_token is None

    replacement = service.claim_next("replacement-worker")
    assert replacement is not None
    assert replacement.attempts == legacy_claim.attempts + 1
    assert replacement.claim_token is not None
    stage = fetchone(
        """
        SELECT status, error_message FROM pipeline_stage_runs
        WHERE run_id = ? AND attempt = ? AND stage_name = 'odds'
        """,
        (legacy_claim.run_id, legacy_claim.attempts),
    )
    assert stage == ("failed", "worker heartbeat lease expired")


def test_backend_terminal_transitions_honor_cancellation(runtime_database) -> None:
    service = JobService(retry_base_seconds=0)
    completion_job = service.create_pipeline_job(season=2026, week=1, source="api", max_attempts=3)
    completion_claim = service.claim_next("complete-worker")
    assert completion_claim is not None
    service.request_cancel(completion_job.run_id)
    assert service.complete(completion_claim, {"success": True}) is False
    assert service.get_job(completion_job.job_id).status == "cancelled"

    failure_job = service.create_pipeline_job(season=2026, week=2, source="api", max_attempts=3)
    failure_claim = service.claim_next("fail-worker")
    assert failure_claim is not None
    service.request_cancel(failure_job.run_id)
    assert service.fail(failure_claim, "cancel race") == "cancelled"
    assert service.get_job(failure_job.job_id).status == "cancelled"


def test_inline_and_worker_paths_produce_equivalent_reports(
    runtime_database, tmp_path, monkeypatch
) -> None:
    from scripts import production_runner

    monkeypatch.setattr(production_runner.config, "logs_dir", tmp_path)
    monkeypatch.setattr(
        production_runner,
        "stage_prepare_week",
        lambda season, week, refresh_history=None: {
            "status": "ok",
            "stage": "prepare_week",
            "predictions": 12,
        },
    )
    monkeypatch.setattr(
        production_runner,
        "POST_PREPARE_STAGES",
        [
            (
                "odds",
                lambda season, week: {"status": "ok", "stage": "odds", "odds_count": 8},
            ),
            (
                "materialize",
                lambda season, week, **_kwargs: {
                    "status": "ok",
                    "stage": "materialize",
                    "card_size": 3,
                },
            ),
        ],
    )

    inline_report = production_runner.run_production_pipeline(2026, 1)
    job = JobService().create_pipeline_job(season=2026, week=1, source="scheduler")
    worker = PipelineWorker(
        worker_id="equivalence-worker",
        service=JobService(),
        runner=production_runner.run_production_pipeline,
    )
    assert worker.process_once() is True

    stored = fetchone("SELECT report_json FROM pipeline_runs WHERE run_id = ?", (job.run_id,))
    assert stored is not None
    worker_report = json.loads(stored[0])
    comparable_keys = {
        "season",
        "week",
        "stages",
        "errors",
        "success",
        "cancelled",
        "incomplete",
    }
    assert {key: inline_report[key] for key in comparable_keys} == {
        key: worker_report[key] for key in comparable_keys
    }


def test_operational_metrics_cover_queue_failures_and_stage_duration(runtime_database) -> None:
    service = JobService(retry_base_seconds=0)
    service.create_pipeline_job(season=2026, week=1, source="scheduler")
    failing = service.create_pipeline_job(
        season=2026,
        week=2,
        source="scheduler",
        priority=1,
        max_attempts=1,
    )
    claimed = service.claim_next("metrics-worker")
    assert claimed is not None and claimed.job_id == failing.job_id
    service.record_stage_started(
        claimed,
        "odds",
        1,
    )
    service.record_stage_result(
        claimed,
        "odds",
        1,
        {"status": "error", "error": "provider unavailable"},
    )
    service.fail(claimed, "provider unavailable", retryable=False)

    metrics = service.operational_metrics()

    assert metrics["queue"]["queued"] == 1
    assert metrics["queue"]["failed"] == 1
    assert metrics["failures_total"] == 1
    assert metrics["stale_running"] == 0
    assert metrics["oldest_queued_seconds"] >= 0
    assert metrics["stage_durations"]["odds"]["samples"] == 1
