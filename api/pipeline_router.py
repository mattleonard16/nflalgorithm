"""FastAPI routes for durable pipeline job control and architecture read models."""

from __future__ import annotations

import json
import os
import secrets
from collections import defaultdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from api.auth import validate_session
from pipeline_jobs.service import JobService, PipelineJob
from pipelines.nfl_contract import NFL_STAGE_COUNT
from utils.db import fetchall, fetchone, get_backend


class PipelineRunResponse(BaseModel):
    run_id: str
    season: int
    week: int
    status: str
    stages_requested: int = NFL_STAGE_COUNT
    stages_completed: int = 0
    error_message: Optional[str] = None
    started_at: str
    finished_at: Optional[str] = None
    report_json: Optional[Dict[str, Any]] = None
    data_health: Optional[Dict[str, Any]] = None
    job_id: Optional[str] = None
    source: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 0
    worker_id: Optional[str] = None
    cancel_requested: bool = False
    available_at: Optional[str] = None
    stages: List[Dict[str, Any]] = Field(default_factory=list)


def _pipeline_principal(request: Request, *, operator_only: bool) -> str:
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.removeprefix("Bearer ") if auth_header.startswith("Bearer ") else ""
    control_token = os.getenv("PIPELINE_CONTROL_TOKEN", "")
    if control_token and token and secrets.compare_digest(token, control_token):
        return "control-token"
    if token:
        user = validate_session(token)
        if user:
            allowed_tiers = {
                tier.strip().lower()
                for tier in os.getenv("PIPELINE_OPERATOR_TIERS", "operator,admin").split(",")
                if tier.strip()
            }
            if operator_only and user.subscription_tier.lower() not in allowed_tiers:
                raise HTTPException(status_code=403, detail="Pipeline operator access required")
            return user.user_id
    raise HTTPException(status_code=401, detail="Pipeline control requires authentication")


def require_pipeline_reader(request: Request) -> str:
    """Require authentication before returning operational run diagnostics."""
    return _pipeline_principal(request, operator_only=False)


def require_pipeline_operator(request: Request) -> str:
    """Require a privileged account or dedicated control token for mutations."""
    return _pipeline_principal(request, operator_only=True)


_PIPELINE_RUN_COLS = (
    "run_id, season, week, status, stages_requested, stages_completed, error_message, "
    "started_at, finished_at, report_json, data_health_json"
)


def _parse_pipeline_run_row(
    row: tuple[Any, ...],
    *,
    job: PipelineJob | None = None,
    stage_rows: list[tuple[Any, ...]] | None = None,
) -> PipelineRunResponse:
    report = None
    if row[9]:
        try:
            report = json.loads(row[9])
        except (json.JSONDecodeError, TypeError):
            pass

    health = None
    if len(row) > 10 and row[10]:
        try:
            health = json.loads(row[10])
        except (json.JSONDecodeError, TypeError):
            pass

    if job is None:
        job = JobService().get_job_for_run(str(row[0]))
    if stage_rows is None:
        stage_rows = fetchall(
            """
            SELECT stage_name, ordinal, status, attempt, started_at, finished_at,
                   result_json, error_message
            FROM pipeline_stage_runs WHERE run_id = ? ORDER BY ordinal, stage_name
            """,
            (row[0],),
        )
    stages: list[dict[str, Any]] = []
    for stage_row in stage_rows:
        result = None
        if stage_row[6]:
            try:
                result = json.loads(stage_row[6])
            except (json.JSONDecodeError, TypeError):
                result = None
        stages.append(
            {
                "name": stage_row[0],
                "ordinal": stage_row[1],
                "status": stage_row[2],
                "attempt": stage_row[3],
                "started_at": stage_row[4],
                "finished_at": stage_row[5],
                "result": result,
                "error_message": stage_row[7],
            }
        )

    return PipelineRunResponse(
        run_id=row[0],
        season=row[1],
        week=row[2],
        status=row[3],
        stages_requested=row[4],
        stages_completed=row[5],
        error_message=row[6],
        started_at=row[7],
        finished_at=row[8],
        report_json=report,
        data_health=health,
        job_id=job.job_id if job else None,
        source=job.source if job else None,
        attempts=job.attempts if job else 0,
        max_attempts=job.max_attempts if job else 0,
        worker_id=job.worker_id if job else None,
        cancel_requested=job.cancel_requested if job else False,
        available_at=job.available_at if job else None,
        stages=stages,
    )


def _get_pipeline_run(run_id: str) -> PipelineRunResponse:
    row = fetchone(f"SELECT {_PIPELINE_RUN_COLS} FROM pipeline_runs WHERE run_id = ?", (run_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    return _parse_pipeline_run_row(row)


router = APIRouter(tags=["pipeline-control"])


@router.post("/api/run", response_model=PipelineRunResponse)
def trigger_pipeline_run(
    request: Request,
    season: int = Query(..., ge=2000, le=2100, description="NFL season year"),
    week: int = Query(..., ge=1, le=22, description="Week number"),
    skip_ingest: bool = Query(False, description="Skip data ingestion"),
    skip_odds: bool = Query(False, description="Skip odds refresh"),
    operator_id: str = Depends(require_pipeline_operator),
):
    """Create a durable job; a dedicated worker owns long-running execution."""
    try:
        job = JobService().create_pipeline_job(
            season=season,
            week=week,
            source="api",
            requested_by=operator_id,
            skip_ingest=skip_ingest,
            skip_odds=skip_odds,
            idempotency_key=request.headers.get("Idempotency-Key"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _get_pipeline_run(job.run_id)


@router.get("/api/run/latest", response_model=Optional[PipelineRunResponse])
def get_latest_pipeline_run(
    season: int = Query(..., description="NFL season year"),
    week: int = Query(..., description="Week number"),
    reader_id: str = Depends(require_pipeline_reader),
):
    del reader_id
    row = fetchone(
        f"SELECT {_PIPELINE_RUN_COLS} FROM pipeline_runs "
        "WHERE season = ? AND week = ? ORDER BY started_at DESC LIMIT 1",
        (season, week),
    )
    return _parse_pipeline_run_row(row) if row else None


@router.get("/api/run/{run_id}", response_model=PipelineRunResponse)
def get_pipeline_run(run_id: str, reader_id: str = Depends(require_pipeline_reader)):
    del reader_id
    return _get_pipeline_run(run_id)


@router.post("/api/run/{run_id}/cancel", response_model=PipelineRunResponse)
def cancel_pipeline_run(
    run_id: str,
    operator_id: str = Depends(require_pipeline_operator),
):
    del operator_id
    try:
        JobService().request_cancel(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")
    return _get_pipeline_run(run_id)


@router.post("/api/run/{run_id}/retry", response_model=PipelineRunResponse)
def retry_pipeline_run(
    run_id: str,
    operator_id: str = Depends(require_pipeline_operator),
):
    del operator_id
    try:
        JobService().retry(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return _get_pipeline_run(run_id)


@router.get("/api/system/architecture")
def get_architecture_status(reader_id: str = Depends(require_pipeline_reader)):
    del reader_id
    job_service = JobService()
    queue = job_service.queue_summary()
    run_rows = fetchall(
        f"SELECT {_PIPELINE_RUN_COLS} FROM pipeline_runs ORDER BY started_at DESC LIMIT 8"
    )
    run_ids = [str(row[0]) for row in run_rows]
    jobs = job_service.get_jobs_for_runs(run_ids)
    stages_by_run: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    if run_ids:
        placeholders = ", ".join("?" for _ in run_ids)
        stage_rows = fetchall(
            f"""
            SELECT run_id, stage_name, ordinal, status, attempt, started_at, finished_at,
                   result_json, error_message
            FROM pipeline_stage_runs WHERE run_id IN ({placeholders})
            ORDER BY run_id, ordinal, stage_name
            """,
            tuple(run_ids),
        )
        for stage_row in stage_rows:
            stages_by_run[str(stage_row[0])].append(stage_row[1:])
    counts = fetchone("""
        SELECT
            (SELECT COUNT(*) FROM pipeline_artifacts),
            (SELECT COUNT(*) FROM agent_decisions),
            (SELECT COUNT(*) FROM materialized_value_view)
        """)
    recent_runs = [
        _parse_pipeline_run_row(
            row,
            job=jobs.get(str(row[0])),
            stage_rows=stages_by_run.get(str(row[0]), []),
        ).model_dump()
        for row in run_rows
    ]
    return {
        "database_backend": get_backend(),
        "queue": queue,
        "metrics": job_service.operational_metrics(),
        "workers_active": queue.get("running", 0),
        "artifact_count": int(counts[0]) if counts else 0,
        "decision_count": int(counts[1]) if counts else 0,
        "read_model_rows": int(counts[2]) if counts else 0,
        "recent_runs": recent_runs,
        "levels": [
            {
                "id": "entry",
                "level": 1,
                "title": "Entry Points",
                "tone": "blue",
                "nodes": ["Next.js Dashboard", "CLI", "Scheduler"],
            },
            {
                "id": "control",
                "level": 2,
                "title": "API + Job Control",
                "tone": "blue",
                "nodes": ["FastAPI", "Pipeline Job Service"],
            },
            {
                "id": "execution",
                "level": 3,
                "title": "Execution",
                "tone": "amber",
                "nodes": ["Job Queue", "NFL Worker", "Shared Orchestrator"],
            },
            {
                "id": "pipeline",
                "level": 4,
                "title": "NFL Pipeline",
                "tone": "green",
                "nodes": [
                    "Prepare Data",
                    "Validate Pregame",
                    "Generate Projections",
                    "Fetch Live Odds",
                ],
            },
            {
                "id": "decision",
                "level": 5,
                "title": "Betting Decision",
                "tone": "purple",
                "nodes": [
                    "Value Engine",
                    "Confidence + Risk",
                    "Specialist Agents",
                    "Final Betting Card",
                ],
            },
            {
                "id": "persistence",
                "level": 6,
                "title": "Persistence",
                "tone": "amber",
                "nodes": ["Operational Database", "Artifact Storage", "API Read Models"],
            },
        ],
    }


@router.get("/api/system/pipeline-metrics")
def get_pipeline_metrics(reader_id: str = Depends(require_pipeline_reader)):
    """Expose authenticated operational telemetry for scraping and alerting."""
    del reader_id
    return JobService().operational_metrics()
