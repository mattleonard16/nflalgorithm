"""Create, claim, track, retry, and cancel durable pipeline jobs."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from pipelines.nfl_contract import NFL_STAGE_COUNT
from utils.db import execute, fetchall, fetchone, get_connection, is_sqlite_connection

QUEUED_STATUSES = ("queued", "retry_scheduled")
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
LEGAL_TRANSITIONS = {
    "queued": frozenset({"running", "cancelled"}),
    "retry_scheduled": frozenset({"running", "cancelled"}),
    "running": frozenset({"completed", "failed", "retry_scheduled", "cancelled"}),
    "failed": frozenset({"queued"}),
    "completed": frozenset(),
    "cancelled": frozenset(),
}
_JOB_COLUMNS = (
    "job_id, run_id, job_type, payload_json, status, priority, attempts, max_attempts, "
    "available_at, claimed_at, heartbeat_at, worker_id, claim_token, cancel_requested, "
    "idempotency_key, source, requested_by, created_at, updated_at, last_error"
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or _utcnow()).isoformat()


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("ascii")
    return str(value)


class LeaseLostError(RuntimeError):
    """Raised when an execution attempt no longer owns its database lease."""


def _lease_clause(conn: Any) -> str:
    worker_match = (
        "worker_id = ? COLLATE BINARY" if is_sqlite_connection(conn) else "BINARY worker_id = BINARY ?"
    )
    return (
        f"job_id = ? AND {worker_match} AND attempts = ? AND claim_token = ? "
        "AND status = 'running'"
    )


def _lease_values(job: "PipelineJob") -> tuple[Any, ...]:
    if not job.worker_id or not job.claim_token or job.attempts < 1:
        raise ValueError("job does not contain an active attempt lease")
    return (job.job_id, job.worker_id, job.attempts, job.claim_token.encode("ascii"))


def _lock_active_lease(conn: Any, job: "PipelineJob", *, allow_cancel: bool = False) -> bool:
    if is_sqlite_connection(conn):
        conn.execute("BEGIN IMMEDIATE")
        suffix = ""
    else:
        suffix = " FOR UPDATE"
    owned = fetchone(
        "SELECT cancel_requested FROM pipeline_jobs "
        f"WHERE {_lease_clause(conn)}{suffix}",
        _lease_values(job),
        conn=conn,
    )
    if not owned:
        conn.rollback()
        raise LeaseLostError("worker attempt lease is no longer active")
    cancelled = bool(owned[0])
    if cancelled and not allow_cancel:
        conn.rollback()
        raise LeaseLostError("worker attempt was cancelled")
    return cancelled


@dataclass(frozen=True, slots=True)
class PipelineJob:
    job_id: str
    run_id: str
    job_type: str
    payload: dict[str, Any]
    status: str
    priority: int
    attempts: int
    max_attempts: int
    available_at: str
    claimed_at: str | None
    heartbeat_at: str | None
    worker_id: str | None
    claim_token: str | None
    cancel_requested: bool
    idempotency_key: str | None
    source: str
    requested_by: str | None
    created_at: str
    updated_at: str
    last_error: str | None

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> "PipelineJob":
        return cls(
            job_id=str(row[0]),
            run_id=str(row[1]),
            job_type=str(row[2]),
            payload=json.loads(row[3]) if row[3] else {},
            status=str(row[4]),
            priority=int(row[5]),
            attempts=int(row[6]),
            max_attempts=int(row[7]),
            available_at=str(row[8]),
            claimed_at=row[9],
            heartbeat_at=row[10],
            worker_id=row[11],
            claim_token=_text(row[12]),
            cancel_requested=bool(row[13]),
            idempotency_key=row[14],
            source=str(row[15]),
            requested_by=row[16],
            created_at=str(row[17]),
            updated_at=str(row[18]),
            last_error=row[19],
        )


class JobService:
    """Durable pipeline-job state machine backed by the application database."""

    def __init__(self, *, retry_base_seconds: int = 30, stale_after_seconds: int = 300) -> None:
        self.retry_base_seconds = max(0, retry_base_seconds)
        self.stale_after_seconds = max(30, stale_after_seconds)

    def create_pipeline_job(
        self,
        *,
        season: int,
        week: int,
        source: str,
        requested_by: str | None = None,
        skip_ingest: bool = False,
        skip_odds: bool = False,
        priority: int = 0,
        max_attempts: int = 3,
        idempotency_key: str | None = None,
    ) -> PipelineJob:
        """Persist a queue item and its read-model run atomically."""
        if not 2000 <= season <= 2100:
            raise ValueError("season must be between 2000 and 2100")
        if not 1 <= week <= 22:
            raise ValueError("week must be between 1 and 22")
        if source not in {"api", "cli", "scheduler"}:
            raise ValueError("source must be api, cli, or scheduler")
        if not 1 <= max_attempts <= 10:
            raise ValueError("max_attempts must be between 1 and 10")

        payload = json.dumps(
            {
                "season": season,
                "week": week,
                "skip_ingest": bool(skip_ingest),
                "skip_odds": bool(skip_odds),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        if idempotency_key and len(idempotency_key) > 128:
            raise ValueError("idempotency key must be at most 128 characters")
        scoped_key = (
            f"{source}:{requested_by or 'system'}:{idempotency_key}" if idempotency_key else None
        )

        if scoped_key:
            existing = fetchone(
                f"SELECT {_JOB_COLUMNS} FROM pipeline_jobs WHERE idempotency_key = ?",
                (scoped_key,),
            )
            if existing:
                job = PipelineJob.from_row(existing)
                if job.payload != json.loads(payload):
                    raise ValueError("idempotency key was already used for a different request")
                return job

        job_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        now = _iso()
        try:
            with get_connection() as conn:
                execute(
                    """
                    INSERT INTO pipeline_runs
                        (run_id, season, week, status, stages_requested, stages_completed,
                         started_at, source, requested_by, updated_at)
                    VALUES (?, ?, ?, 'queued', ?, 0, ?, ?, ?, ?)
                    """,
                    (run_id, season, week, NFL_STAGE_COUNT, now, source, requested_by, now),
                    conn=conn,
                )
                execute(
                    """
                    INSERT INTO pipeline_jobs
                        (job_id, run_id, job_type, payload_json, status, priority, attempts,
                         max_attempts, available_at, cancel_requested, idempotency_key, source,
                         requested_by, created_at, updated_at)
                    VALUES (?, ?, 'nfl_weekly', ?, 'queued', ?, 0, ?, ?, 0, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        run_id,
                        payload,
                        priority,
                        max_attempts,
                        now,
                        scoped_key,
                        source,
                        requested_by,
                        now,
                        now,
                    ),
                    conn=conn,
                )
                conn.commit()
        except Exception:
            if scoped_key:
                existing = fetchone(
                    f"SELECT {_JOB_COLUMNS} FROM pipeline_jobs WHERE idempotency_key = ?",
                    (scoped_key,),
                )
                if existing:
                    job = PipelineJob.from_row(existing)
                    if job.payload != json.loads(payload):
                        raise ValueError("idempotency key was already used for a different request")
                    return job
            raise

        created = self.get_job(job_id)
        assert created is not None
        return created

    def get_job(self, job_id: str) -> PipelineJob | None:
        row = fetchone(f"SELECT {_JOB_COLUMNS} FROM pipeline_jobs WHERE job_id = ?", (job_id,))
        return PipelineJob.from_row(row) if row else None

    def get_job_for_run(self, run_id: str) -> PipelineJob | None:
        row = fetchone(f"SELECT {_JOB_COLUMNS} FROM pipeline_jobs WHERE run_id = ?", (run_id,))
        return PipelineJob.from_row(row) if row else None

    def get_jobs_for_runs(self, run_ids: list[str]) -> dict[str, PipelineJob]:
        """Load job metadata for several run read models in one query."""
        if not run_ids:
            return {}
        placeholders = ", ".join("?" for _ in run_ids)
        rows = fetchall(
            f"SELECT {_JOB_COLUMNS} FROM pipeline_jobs WHERE run_id IN ({placeholders})",
            tuple(run_ids),
        )
        jobs = (PipelineJob.from_row(row) for row in rows)
        return {job.run_id: job for job in jobs}

    def claim_next(self, worker_id: str) -> PipelineJob | None:
        """Atomically claim the highest-priority available job."""
        self.recover_stale_jobs()
        now = _iso()
        with get_connection() as conn:
            if is_sqlite_connection(conn):
                conn.execute("BEGIN IMMEDIATE")
                suffix = ""
            else:
                suffix = " FOR UPDATE SKIP LOCKED"

            row = fetchone(
                f"""
                SELECT {_JOB_COLUMNS}
                FROM pipeline_jobs
                WHERE status IN (?, ?) AND available_at <= ? AND cancel_requested = 0
                ORDER BY priority DESC, created_at ASC, job_id ASC
                LIMIT 1{suffix}
                """,
                (QUEUED_STATUSES[0], QUEUED_STATUSES[1], now),
                conn=conn,
            )
            if not row:
                conn.commit()
                return None

            job = PipelineJob.from_row(row)
            claim_token = uuid.uuid4().hex
            affected = execute(
                """
                UPDATE pipeline_jobs
                SET status = 'running', attempts = attempts + 1, claimed_at = ?, heartbeat_at = ?,
                    worker_id = ?, claim_token = ?, updated_at = ?
                WHERE job_id = ? AND status IN (?, ?) AND cancel_requested = 0
                """,
                (
                    now,
                    now,
                    worker_id,
                    claim_token.encode("ascii"),
                    now,
                    job.job_id,
                    *QUEUED_STATUSES,
                ),
                conn=conn,
            )
            if affected != 1:
                conn.rollback()
                return None
            execute(
                "UPDATE pipeline_runs SET status = 'running', updated_at = ? WHERE run_id = ?",
                (now, job.run_id),
                conn=conn,
            )
            conn.commit()

        return self.get_job(job.job_id)

    def recover_stale_jobs(self) -> int:
        """Reclaim expired worker leases while honoring each job's retry budget."""
        cutoff = _iso(_utcnow() - timedelta(seconds=self.stale_after_seconds))
        now = _iso()
        recovered = 0
        with get_connection() as conn:
            if is_sqlite_connection(conn):
                conn.execute("BEGIN IMMEDIATE")
                suffix = ""
            else:
                suffix = " FOR UPDATE SKIP LOCKED"
            rows = fetchall(
                f"""
                SELECT {_JOB_COLUMNS} FROM pipeline_jobs
                WHERE status = 'running' AND heartbeat_at < ?{suffix}
                """,
                (cutoff,),
                conn=conn,
            )
            for row in rows:
                job = PipelineJob.from_row(row)
                if job.cancel_requested:
                    status = run_status = "cancelled"
                elif job.attempts < job.max_attempts:
                    status, run_status = "retry_scheduled", "queued"
                else:
                    status = run_status = "failed"
                affected = execute(
                    """
                    UPDATE pipeline_jobs
                    SET status = ?, available_at = ?, claimed_at = NULL, heartbeat_at = NULL,
                        worker_id = NULL, claim_token = NULL, updated_at = ?, last_error = ?
                    WHERE job_id = ? AND status = 'running' AND worker_id = ?
                    """,
                    (
                        status,
                        now,
                        now,
                        "worker heartbeat lease expired",
                        job.job_id,
                        job.worker_id,
                    ),
                    conn=conn,
                )
                if affected != 1:
                    continue
                execute(
                    """
                    UPDATE pipeline_runs
                    SET status = ?, stages_completed = 0, error_message = ?,
                        finished_at = ?, updated_at = ? WHERE run_id = ?
                    """,
                    (
                        run_status,
                        "worker heartbeat lease expired",
                        now if run_status in TERMINAL_STATUSES else None,
                        now,
                        job.run_id,
                    ),
                    conn=conn,
                )
                if status == "retry_scheduled":
                    execute(
                        """
                        UPDATE pipeline_stage_runs
                        SET status = 'queued', finished_at = NULL, result_json = NULL,
                            error_message = NULL WHERE run_id = ?
                        """,
                        (job.run_id,),
                        conn=conn,
                    )
                recovered += 1
            conn.commit()
        return recovered

    def heartbeat(self, job: PipelineJob) -> bool:
        now = _iso()
        with get_connection() as conn:
            affected = execute(
                f"""
            UPDATE pipeline_jobs SET heartbeat_at = ?, updated_at = ?
            WHERE {_lease_clause(conn)} AND cancel_requested = 0
            """,
                (now, now, *_lease_values(job)),
                conn=conn,
            )
            conn.commit()
        return affected == 1

    def cancellation_requested(self, job: PipelineJob) -> bool:
        with get_connection() as conn:
            row = fetchone(
                f"SELECT cancel_requested FROM pipeline_jobs WHERE {_lease_clause(conn)}",
                _lease_values(job),
                conn=conn,
            )
        if not row:
            raise LeaseLostError("worker attempt lease is no longer active")
        return bool(row[0])

    def request_cancel(self, run_id: str) -> PipelineJob:
        now = _iso()
        with get_connection() as conn:
            if is_sqlite_connection(conn):
                conn.execute("BEGIN IMMEDIATE")
                suffix = ""
            else:
                suffix = " FOR UPDATE"
            row = fetchone(
                f"SELECT {_JOB_COLUMNS} FROM pipeline_jobs WHERE run_id = ?{suffix}",
                (run_id,),
                conn=conn,
            )
            if not row:
                raise KeyError(run_id)
            job = PipelineJob.from_row(row)
            if job.status in TERMINAL_STATUSES:
                conn.commit()
                return job
            if job.status in QUEUED_STATUSES:
                execute(
                    """
                    UPDATE pipeline_jobs
                    SET cancel_requested = 1, status = 'cancelled', updated_at = ?
                    WHERE job_id = ? AND status IN (?, ?)
                    """,
                    (now, job.job_id, *QUEUED_STATUSES),
                    conn=conn,
                )
                execute(
                    """
                    UPDATE pipeline_runs
                    SET status = 'cancelled', finished_at = ?, updated_at = ?
                    WHERE run_id = ?
                    """,
                    (now, now, run_id),
                    conn=conn,
                )
            else:
                execute(
                    """
                    UPDATE pipeline_jobs SET cancel_requested = 1, updated_at = ?
                    WHERE job_id = ? AND status = 'running'
                    """,
                    (now, job.job_id),
                    conn=conn,
                )
                execute(
                    "UPDATE pipeline_runs SET status = 'cancelling', updated_at = ? WHERE run_id = ?",
                    (now, run_id),
                    conn=conn,
                )
            conn.commit()
        updated = self.get_job(job.job_id)
        assert updated is not None
        return updated

    def retry(self, run_id: str) -> PipelineJob:
        now = _iso()
        with get_connection() as conn:
            if is_sqlite_connection(conn):
                conn.execute("BEGIN IMMEDIATE")
                suffix = ""
            else:
                suffix = " FOR UPDATE"
            row = fetchone(
                f"SELECT {_JOB_COLUMNS} FROM pipeline_jobs WHERE run_id = ?{suffix}",
                (run_id,),
                conn=conn,
            )
            if not row:
                raise KeyError(run_id)
            job = PipelineJob.from_row(row)
            if job.status != "failed":
                raise ValueError("only failed jobs can be retried")
            execute(
                """
                UPDATE pipeline_jobs
                SET status = 'queued', attempts = 0, available_at = ?, claimed_at = NULL,
                    heartbeat_at = NULL, worker_id = NULL, claim_token = NULL, cancel_requested = 0,
                    last_error = NULL, updated_at = ?
                WHERE job_id = ? AND status = 'failed'
                """,
                (now, now, job.job_id),
                conn=conn,
            )
            execute(
                """
                UPDATE pipeline_runs
                SET status = 'queued', stages_completed = 0, error_message = NULL,
                    finished_at = NULL, report_json = NULL, data_health_json = NULL, updated_at = ?
                WHERE run_id = ?
                """,
                (now, run_id),
                conn=conn,
            )
            execute(
                """
                UPDATE pipeline_stage_runs
                SET status = 'queued', finished_at = NULL, result_json = NULL,
                    error_message = NULL
                WHERE run_id = ?
                """,
                (run_id,),
                conn=conn,
            )
            conn.commit()
        updated = self.get_job(job.job_id)
        assert updated is not None
        return updated

    def record_stage_started(
        self,
        job: PipelineJob,
        stage_name: str,
        ordinal: int,
    ) -> None:
        now = _iso()
        with get_connection() as conn:
            _lock_active_lease(conn, job)
            existing = fetchone(
                """
                SELECT 1 FROM pipeline_stage_runs
                WHERE run_id = ? AND attempt = ? AND stage_name = ?
                """,
                (job.run_id, job.attempts, stage_name),
                conn=conn,
            )
            if existing:
                execute(
                    """
                    UPDATE pipeline_stage_runs
                    SET ordinal = ?, status = 'running', started_at = ?,
                        finished_at = NULL, result_json = NULL, error_message = NULL
                    WHERE run_id = ? AND attempt = ? AND stage_name = ?
                    """,
                    (ordinal, now, job.run_id, job.attempts, stage_name),
                    conn=conn,
                )
            else:
                execute(
                    """
                    INSERT INTO pipeline_stage_runs
                        (run_id, stage_name, ordinal, status, attempt, started_at)
                    VALUES (?, ?, ?, 'running', ?, ?)
                    """,
                    (job.run_id, stage_name, ordinal, job.attempts, now),
                    conn=conn,
                )
            affected = execute(
                f"""
                UPDATE pipeline_jobs SET heartbeat_at = ?, updated_at = ?
                WHERE {_lease_clause(conn)} AND cancel_requested = 0
                """,
                (now, now, *_lease_values(job)),
                conn=conn,
            )
            if affected != 1:
                conn.rollback()
                raise LeaseLostError("worker attempt lease was lost during stage start")
            conn.commit()

    def record_stage_result(
        self,
        job: PipelineJob,
        stage_name: str,
        ordinal: int,
        result: Mapping[str, Any],
    ) -> None:
        now = _iso()
        status = "completed" if result.get("status") in {"ok", "skipped"} else "failed"
        error = result.get("error") or result.get("detail")
        with get_connection() as conn:
            _lock_active_lease(conn, job)
            existing = fetchone(
                """
                SELECT 1 FROM pipeline_stage_runs
                WHERE run_id = ? AND attempt = ? AND stage_name = ?
                """,
                (job.run_id, job.attempts, stage_name),
                conn=conn,
            )
            if not existing:
                execute(
                    """
                    INSERT INTO pipeline_stage_runs
                        (run_id, stage_name, ordinal, status, attempt, started_at)
                    VALUES (?, ?, ?, 'running', ?, ?)
                    """,
                    (job.run_id, stage_name, ordinal, job.attempts, now),
                    conn=conn,
                )
            execute(
                """
                UPDATE pipeline_stage_runs
                SET ordinal = ?, status = ?, finished_at = ?, result_json = ?, error_message = ?
                WHERE run_id = ? AND attempt = ? AND stage_name = ?
                """,
                (
                    ordinal,
                    status,
                    now,
                    json.dumps(dict(result), default=str, separators=(",", ":")),
                    str(error) if error else None,
                    job.run_id,
                    job.attempts,
                    stage_name,
                ),
                conn=conn,
            )
            completed = fetchone(
                """
                SELECT COUNT(*) FROM pipeline_stage_runs
                WHERE run_id = ? AND attempt = ? AND status = 'completed'
                """,
                (job.run_id, job.attempts),
                conn=conn,
            )
            execute(
                "UPDATE pipeline_runs SET stages_completed = ?, updated_at = ? WHERE run_id = ?",
                (int(completed[0]) if completed else 0, now, job.run_id),
                conn=conn,
            )
            affected = execute(
                f"""
                UPDATE pipeline_jobs SET heartbeat_at = ?, updated_at = ?
                WHERE {_lease_clause(conn)} AND cancel_requested = 0
                """,
                (now, now, *_lease_values(job)),
                conn=conn,
            )
            if affected != 1:
                conn.rollback()
                raise LeaseLostError("worker attempt lease was lost during stage result")
            conn.commit()

    def complete(
        self,
        job: PipelineJob,
        report: Mapping[str, Any],
        *,
        data_health: Mapping[str, Any] | None = None,
    ) -> bool:
        now = _iso()
        with get_connection() as conn:
            affected = execute(
                """
                UPDATE pipeline_jobs
                SET status = 'completed', heartbeat_at = ?, updated_at = ?, last_error = NULL
                WHERE job_id = ? AND worker_id = ? AND status = 'running'
                """,
                (now, now, job.job_id, job.worker_id),
                conn=conn,
            )
            if affected != 1:
                conn.rollback()
                return False
            execute(
                """
                UPDATE pipeline_runs
                SET status = 'completed', error_message = NULL, finished_at = ?, report_json = ?,
                    data_health_json = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (
                    now,
                    json.dumps(dict(report), default=str),
                    json.dumps(dict(data_health), default=str) if data_health is not None else None,
                    now,
                    job.run_id,
                ),
                conn=conn,
            )
            conn.commit()
        return True

    def cancel_running(self, job: PipelineJob, report: Mapping[str, Any] | None = None) -> bool:
        now = _iso()
        with get_connection() as conn:
            affected = execute(
                """
                UPDATE pipeline_jobs SET status = 'cancelled', updated_at = ?
                WHERE job_id = ? AND worker_id = ? AND status = 'running'
                """,
                (now, job.job_id, job.worker_id),
                conn=conn,
            )
            if affected != 1:
                conn.rollback()
                return False
            execute(
                """
                UPDATE pipeline_runs SET status = 'cancelled', finished_at = ?, report_json = ?,
                    updated_at = ? WHERE run_id = ?
                """,
                (now, json.dumps(dict(report or {}), default=str), now, job.run_id),
                conn=conn,
            )
            conn.commit()
        return True

    def fail(
        self,
        job: PipelineJob,
        error: str,
        report: Mapping[str, Any] | None = None,
        *,
        retryable: bool = True,
    ) -> str:
        now = _utcnow()
        current = self.get_job(job.job_id) or job
        retrying = (
            retryable and current.attempts < current.max_attempts and not current.cancel_requested
        )
        status = "retry_scheduled" if retrying else "failed"
        delay = self.retry_base_seconds * (2 ** max(0, current.attempts - 1))
        available_at = _iso(now + timedelta(seconds=delay))
        run_status = "queued" if retrying else "failed"
        with get_connection() as conn:
            affected = execute(
                """
                UPDATE pipeline_jobs
                SET status = ?, available_at = ?, claimed_at = NULL, heartbeat_at = NULL,
                    worker_id = NULL, claim_token = NULL, updated_at = ?, last_error = ?
                WHERE job_id = ? AND worker_id = ? AND status = 'running'
                """,
                (
                    status,
                    available_at,
                    _iso(now),
                    error[:2000],
                    job.job_id,
                    job.worker_id,
                ),
                conn=conn,
            )
            if affected != 1:
                conn.rollback()
                return current.status
            completed = fetchone(
                "SELECT COUNT(*) FROM pipeline_stage_runs WHERE run_id = ? AND status = 'completed'",
                (job.run_id,),
                conn=conn,
            )
            execute(
                """
                UPDATE pipeline_runs
                SET status = ?, stages_completed = ?, error_message = ?, finished_at = ?,
                    report_json = ?, updated_at = ? WHERE run_id = ?
                """,
                (
                    run_status,
                    0 if retrying else int(completed[0]) if completed else 0,
                    error[:2000],
                    None if retrying else _iso(now),
                    json.dumps(dict(report or {}), default=str),
                    _iso(now),
                    job.run_id,
                ),
                conn=conn,
            )
            if retrying:
                execute(
                    """
                    UPDATE pipeline_stage_runs
                    SET status = 'queued', finished_at = NULL, result_json = NULL,
                        error_message = NULL WHERE run_id = ?
                    """,
                    (job.run_id,),
                    conn=conn,
                )
            conn.commit()
        return status

    def register_artifact(
        self,
        *,
        run_id: str,
        kind: str,
        uri: str,
        checksum: str | None = None,
        size_bytes: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        artifact_id = str(uuid.uuid4())
        execute(
            """
            INSERT INTO pipeline_artifacts
                (artifact_id, run_id, kind, uri, checksum, size_bytes, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                run_id,
                kind,
                uri,
                checksum,
                size_bytes,
                json.dumps(dict(metadata or {}), default=str),
                _iso(),
            ),
        )
        return artifact_id

    def queue_summary(self) -> dict[str, int]:
        rows = fetchall("SELECT status, COUNT(*) FROM pipeline_jobs GROUP BY status")
        return {str(status): int(count) for status, count in rows}

    def operational_metrics(self) -> dict[str, Any]:
        """Return queue, lease, retry, failure, and stage-duration telemetry."""
        now = _utcnow()

        def age_seconds(value: Any) -> float | None:
            if not value:
                return None
            try:
                parsed = datetime.fromisoformat(str(value))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return max(0.0, (now - parsed.astimezone(timezone.utc)).total_seconds())
            except (TypeError, ValueError):
                return None

        jobs = fetchall("SELECT status, attempts, heartbeat_at, created_at FROM pipeline_jobs")
        queue = {status: 0 for status in (*QUEUED_STATUSES, "running", *TERMINAL_STATUSES)}
        retries_total = 0
        lease_ages: list[float] = []
        queued_ages: list[float] = []
        for status, attempts, heartbeat_at, created_at in jobs:
            normalized_status = str(status)
            queue[normalized_status] = queue.get(normalized_status, 0) + 1
            retries_total += max(0, int(attempts) - 1)
            if normalized_status == "running":
                lease_age = age_seconds(heartbeat_at)
                if lease_age is not None:
                    lease_ages.append(lease_age)
            if normalized_status in QUEUED_STATUSES:
                queued_age = age_seconds(created_at)
                if queued_age is not None:
                    queued_ages.append(queued_age)

        durations: dict[str, list[float]] = {}
        stage_rows = fetchall("""
            SELECT stage_name, started_at, finished_at
            FROM pipeline_stage_runs
            WHERE started_at IS NOT NULL AND finished_at IS NOT NULL
            ORDER BY finished_at DESC LIMIT 1000
            """)
        for stage_name, started_at, finished_at in stage_rows:
            try:
                started = datetime.fromisoformat(str(started_at))
                finished = datetime.fromisoformat(str(finished_at))
                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                if finished.tzinfo is None:
                    finished = finished.replace(tzinfo=timezone.utc)
                duration = max(0.0, (finished - started).total_seconds())
            except (TypeError, ValueError):
                continue
            durations.setdefault(str(stage_name), []).append(duration)

        stage_durations: dict[str, dict[str, float | int]] = {}
        for stage_name, values in durations.items():
            ordered = sorted(values)
            p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
            stage_durations[stage_name] = {
                "samples": len(ordered),
                "average_seconds": round(sum(ordered) / len(ordered), 3),
                "p95_seconds": round(ordered[p95_index], 3),
                "max_seconds": round(ordered[-1], 3),
            }

        stale_running = sum(age > self.stale_after_seconds for age in lease_ages)
        return {
            "generated_at": _iso(now),
            "queue": queue,
            "retries_total": retries_total,
            "failures_total": queue.get("failed", 0),
            "stale_running": stale_running,
            "oldest_queued_seconds": round(max(queued_ages), 3) if queued_ages else 0.0,
            "oldest_lease_seconds": round(max(lease_ages), 3) if lease_ages else 0.0,
            "stage_durations": stage_durations,
        }
