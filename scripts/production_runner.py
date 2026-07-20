"""Production runner: orchestrates the full weekly pipeline.

Calls each pipeline stage in order:
    1. Canonical pregame preparation (migrations, causal data, roster checks, projections)
    2. Live odds refresh
    3. Value ranking + confidence scoring
    4. Risk assessment
    5. Agent evaluation
    6. Final card persistence

Can be driven by APScheduler (see pipeline_scheduler.py) or run
manually via CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Dict

from config import config
from pipelines.nfl_contract import NFL_AUTOMATIC_RETRY_SAFE_STAGES
from pipelines.orchestrator import PipelineStage, run_stages
from utils.db import fetchone, read_dataframe

logger = logging.getLogger(__name__)
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _runtime_commit_sha() -> str:
    """Resolve the worker build identity before any pipeline side effect."""
    configured = os.getenv("APP_COMMIT_SHA", "").strip().lower()
    if configured:
        if not SHA_PATTERN.fullmatch(configured):
            raise RuntimeError("APP_COMMIT_SHA must be a full 40-character Git SHA")
        return configured
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    commit_sha = result.stdout.strip().lower()
    if not SHA_PATTERN.fullmatch(commit_sha):
        raise RuntimeError("worker runtime could not resolve a full Git commit SHA")
    return commit_sha


def _validate_odds_observation(observed: Mapping[str, Any]) -> Dict[str, Any]:
    """Turn malformed observation metrics into persisted fail-closed evidence."""
    from pipelines.odds_validation import validate_odds_snapshot

    try:
        return dict(validate_odds_snapshot(observed))
    except Exception as exc:
        logger.error("Odds validation failed: %s", exc)
        return {
            "valid": False,
            "reason_code": "validation_error",
            "reason": f"Odds validation could not evaluate snapshot metrics: {exc}",
            "validation_error": str(exc),
            "snapshot_at": observed.get("snapshot_at"),
            "odds_rows": observed.get("odds_rows", 0),
        }


# ── Stage functions ──────────────────────────────────────────────────


def stage_prepare_week(
    season: int, week: int, refresh_history: bool | None = None
) -> Dict[str, Any]:
    """Run the sole validated pregame data/projection workflow."""
    try:
        from scripts.prepare_nfl_week import prepare_week

        summary = dict(prepare_week(season, week, refresh_history=refresh_history))
        return {"status": "ok", "stage": "prepare_week", **summary}
    except Exception as exc:
        logger.error("Canonical pregame preparation failed: %s", exc)
        return {"status": "error", "stage": "prepare_week", "error": str(exc)}


def stage_odds(season: int, week: int) -> Dict[str, Any]:
    """Refresh live odds; synthetic/demo lines are forbidden here."""
    from scripts.prop_line_scraper import NFLPropScraper

    scraper = None
    try:
        scraper = NFLPropScraper()
        odds = scraper.run_weekly_update(
            week,
            season,
            allow_synthetic=False,
        )
    except Exception as exc:
        observed = dict(getattr(scraper, "last_weekly_audit", {}) or {})
        observed.setdefault("odds_rows", 0)
        validation = _validate_odds_observation(observed)
        validation["snapshot_reason_code"] = validation["reason_code"]
        validation["snapshot_reason"] = validation["reason"]
        validation["valid"] = False
        validation["reason_code"] = "provider_error"
        validation["reason"] = f"Live odds refresh failed: {exc}"
        validation["provider_error"] = str(exc)
        logger.error("Odds refresh failed: %s", exc)
        return {
            "status": "error",
            "stage": "odds",
            "error": str(exc),
            "odds_count": 0,
            "odds_validation": validation,
        }

    observed = dict(odds.attrs.get("odds_audit", {}))
    observed["odds_rows"] = len(odds)
    validation = _validate_odds_observation(observed)
    if odds.empty and validation["valid"]:
        validation["valid"] = False
        validation["reason_code"] = "empty_snapshot"
        validation["reason"] = "Live odds refresh returned no rows"
    if not validation["valid"]:
        logger.error("Odds snapshot rejected: %s", validation["reason"])
        return {
            "status": "error",
            "stage": "odds",
            "error": validation["reason"],
            "odds_count": len(odds),
            "odds_validation": validation,
        }
    return {
        "status": "ok",
        "stage": "odds",
        "odds_count": len(odds),
        "odds_validation": validation,
    }


def stage_value_ranking(season: int, week: int) -> Dict[str, Any]:
    """Rank value opportunities with confidence scoring."""
    try:
        from value_betting_engine import rank_weekly_value

        from confidence_engine import score_plays

        raw = rank_weekly_value(season, week)
        if raw.empty:
            return {"status": "ok", "stage": "value_ranking", "picks": 0}

        scored = score_plays(raw)
        return {
            "status": "ok",
            "stage": "value_ranking",
            "picks": len(scored),
            "avg_edge": round(float(scored["edge_percentage"].mean()), 4),
        }
    except Exception as exc:
        logger.error("Value ranking failed: %s", exc)
        return {"status": "error", "stage": "value_ranking", "error": str(exc)}


def stage_risk_assessment(season: int, week: int) -> Dict[str, Any]:
    """Run risk checks on value opportunities."""
    try:
        from risk_manager import run_risk_check

        assessed = run_risk_check(season, week)
        warnings = 0
        if not assessed.empty and "exposure_warning" in assessed.columns:
            warnings = int(assessed["exposure_warning"].notna().sum())
        return {"status": "ok", "stage": "risk_assessment", "warnings": warnings}
    except Exception as exc:
        logger.error("Risk assessment failed: %s", exc)
        return {"status": "error", "stage": "risk_assessment", "error": str(exc)}


def stage_agents(season: int, week: int) -> Dict[str, Any]:
    """Run agent coordinator for consensus decisions."""
    try:
        from agents.coordinator import run_all_agents

        decisions = run_all_agents(season, week)
        approved = sum(1 for d in decisions if d.get("decision") == "APPROVED")
        rejected = sum(1 for d in decisions if d.get("decision") == "REJECTED")
        return {
            "status": "ok",
            "stage": "agents",
            "total": len(decisions),
            "approved": approved,
            "rejected": rejected,
        }
    except Exception as exc:
        logger.error("Agent evaluation failed: %s", exc)
        return {"status": "error", "stage": "agents", "error": str(exc)}


def stage_materialize(
    season: int,
    week: int,
    *,
    run_id: str | None = None,
    attempt: int | None = None,
) -> Dict[str, Any]:
    """Stage a durable card, or publish directly for an explicit inline debug run."""
    try:
        from materialized_value_view import materialize_week

        if run_id is not None and attempt is not None:
            validation = fetchone(
                """
                SELECT valid FROM pipeline_odds_validations
                WHERE run_id = ? AND attempt = ?
                """,
                (run_id, attempt),
            )
            if not validation or not bool(validation[0]):
                raise RuntimeError(
                    "Final card staging requires a valid odds snapshot for this attempt"
                )
        materialize_week(season, week, run_id=run_id, attempt=attempt)
        count_params: tuple[Any, ...]
        if run_id is not None and attempt is not None:
            count_query = (
                "SELECT COUNT(*) as n FROM pipeline_card_staging "
                "WHERE run_id = ? AND attempt = ?"
            )
            count_params = (run_id, attempt)
        else:
            count_query = (
                "SELECT COUNT(*) as n FROM materialized_value_view " "WHERE season = ? AND week = ?"
            )
            count_params = (season, week)
        df = read_dataframe(count_query, params=count_params)
        n = int(df.iloc[0]["n"]) if not df.empty else 0
        return {
            "status": "ok",
            "stage": "materialize",
            "card_size": n,
            "publication": "staged" if run_id is not None else "active",
        }
    except Exception as exc:
        logger.error("Materialization failed: %s", exc)
        return {"status": "error", "stage": "materialize", "error": str(exc)}


# ── Orchestrator ─────────────────────────────────────────────────────


POST_PREPARE_STAGES = [
    ("odds", stage_odds),
    ("value_ranking", stage_value_ranking),
    ("risk_assessment", stage_risk_assessment),
    ("agents", stage_agents),
    ("materialize", stage_materialize),
]


def run_production_pipeline(
    season: int,
    week: int,
    skip_ingest: bool = False,
    skip_odds: bool = False,
    *,
    on_stage_start: Callable[[str, int], None] | None = None,
    on_stage_result: Callable[[str, int, Mapping[str, Any]], None] | None = None,
    cancellation_requested: Callable[[], bool] | None = None,
    run_id: str | None = None,
    attempt: int | None = None,
) -> Dict[str, Any]:
    """Execute the full production pipeline for a season/week.

    Returns a run report with per-stage results.
    """
    commit_sha = _runtime_commit_sha()
    started_at = datetime.now(timezone.utc).isoformat()
    stages = [
        PipelineStage(
            "prepare_week",
            partial(
                stage_prepare_week,
                season,
                week,
                refresh_history=False if skip_ingest else None,
            ),
            retry_safe="prepare_week" in NFL_AUTOMATIC_RETRY_SAFE_STAGES,
        ),
        *[
            PipelineStage(
                stage_name,
                partial(
                    stage_fn,
                    season,
                    week,
                    **(
                        {"run_id": run_id, "attempt": attempt}
                        if stage_name == "materialize" and run_id is not None
                        else {}
                    ),
                ),
                retry_safe=stage_name in NFL_AUTOMATIC_RETRY_SAFE_STAGES,
            )
            for stage_name, stage_fn in POST_PREPARE_STAGES
        ],
    ]
    stage_results = run_stages(
        stages,
        skip={"odds": "skip_odds"} if skip_odds else None,
        stop_on_error=True,
        # Every remaining stage consumes weekly_odds. Skipping the refresh
        # must never authorize a card from stale cached lines.
        stop_after_skip={"odds"},
        on_stage_start=on_stage_start,
        on_stage_result=on_stage_result,
        cancellation_requested=cancellation_requested,
    )

    finished_at = datetime.now(timezone.utc).isoformat()

    cancelled = bool(cancellation_requested and cancellation_requested())
    final_stage = POST_PREPARE_STAGES[-1][0]
    finalized = any(
        stage.get("stage") == final_stage and stage.get("status") == "ok" for stage in stage_results
    )
    run_report = {
        "commit_sha": commit_sha,
        "season": season,
        "week": week,
        "started_at": started_at,
        "finished_at": finished_at,
        "stages": stage_results,
        "errors": [s for s in stage_results if s.get("status") == "error"],
        "success": not cancelled
        and finalized
        and all(s.get("status") != "error" for s in stage_results),
        "cancelled": cancelled,
        "incomplete": not cancelled
        and not finalized
        and all(s.get("status") != "error" for s in stage_results),
    }

    artifact_path = _persist_run_report(run_report)
    if artifact_path:
        run_report["artifact_uri"] = str(artifact_path)
    return run_report


def _persist_run_report(report: Dict[str, Any]) -> Path | None:
    """Save run report to logs directory."""
    logs_dir = Path(config.logs_dir) / "production_runs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.fromisoformat(str(report["started_at"]))
    timestamp = started.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    unique = uuid.uuid4().hex[:8]
    path = logs_dir / f"run_{report['season']}_w{report['week']}_{timestamp}_{unique}.json"
    temporary_path = path.with_suffix(".tmp")
    try:
        with open(temporary_path, "x") as f:
            json.dump(report, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
        logger.info("Run report saved to %s", path)
        return path
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        logger.error("Failed to save run report: %s", exc)
        return None


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Production pipeline runner")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Reuse historical data; current-week roster preparation still runs",
    )
    parser.add_argument("--skip-odds", action="store_true", help="Skip odds refresh stage")
    parser.add_argument(
        "--inline",
        action="store_true",
        help="Run in this process for debugging instead of enqueueing a worker job",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not args.inline:
        from pipeline_jobs.service import JobService

        job = JobService().create_pipeline_job(
            season=args.season,
            week=args.week,
            source="cli",
            skip_ingest=args.skip_ingest,
            skip_odds=args.skip_odds,
        )
        print(
            json.dumps(
                {
                    "job_id": job.job_id,
                    "run_id": job.run_id,
                    "status": job.status,
                    "season": args.season,
                    "week": args.week,
                },
                indent=2,
            )
        )
        return

    print(f"Production pipeline (inline): season={args.season} week={args.week}")
    report = run_production_pipeline(
        args.season,
        args.week,
        skip_ingest=args.skip_ingest,
        skip_odds=args.skip_odds,
    )

    print(f"\nPipeline {'SUCCEEDED' if report['success'] else 'HAD ERRORS'}")
    for stage in report["stages"]:
        status = stage.get("status", "?")
        name = stage.get("stage", "?")
        marker = "[OK]" if status == "ok" else "[SKIP]" if status == "skipped" else "[ERR]"
        print(f"  {marker} {name}")

    if report["errors"]:
        print(f"\n{len(report['errors'])} stage(s) had errors:")
        for err in report["errors"]:
            print(f"  - {err['stage']}: {err.get('error', 'unknown')}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
