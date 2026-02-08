"""Production runner: orchestrates the full weekly pipeline.

Calls each pipeline stage in order:
    1. Data ingestion (refresh player stats)
    2. Projection generation
    3. Odds refresh
    4. Value ranking + confidence scoring
    5. Risk assessment
    6. Agent evaluation
    7. Final card persistence

Can be driven by APScheduler (see pipeline_scheduler.py) or run
manually via CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config import config
from utils.db import execute, get_connection, read_dataframe

logger = logging.getLogger(__name__)


# ── Stage functions ──────────────────────────────────────────────────


def stage_ingest(season: int, week: int) -> Dict[str, Any]:
    """Stage 1: Refresh player stats from nflverse."""
    try:
        from scripts.ingest_real_nfl_data import ingest_seasons
        ingest_seasons([season], through_week=week)
        return {"status": "ok", "stage": "ingest"}
    except ImportError:
        logger.warning("ingest_real_nfl_data not available; skipping ingestion")
        return {"status": "skipped", "stage": "ingest", "reason": "module not found"}
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        return {"status": "error", "stage": "ingest", "error": str(exc)}


def stage_player_dim(season: int, week: int) -> Dict[str, Any]:
    """Stage 1b: Populate canonical player dimension table."""
    try:
        from scripts.populate_player_dim import populate_player_dim
        count = populate_player_dim()
        return {"status": "ok", "stage": "player_dim", "players": count}
    except Exception as exc:
        logger.error("player_dim population failed: %s", exc)
        return {"status": "error", "stage": "player_dim", "error": str(exc)}


def stage_projections(season: int, week: int) -> Dict[str, Any]:
    """Stage 2: Generate weekly projections."""
    try:
        from models.position_specific import predict_week
        predict_week(season, week)
        count_query = (
            "SELECT COUNT(*) as n FROM weekly_projections "
            "WHERE season = ? AND week = ?"
        )
        df = read_dataframe(count_query, params=(season, week))
        n = int(df.iloc[0]["n"]) if not df.empty else 0
        return {"status": "ok", "stage": "projections", "projection_count": n}
    except ImportError:
        logger.warning("models.position_specific not available")
        return {"status": "skipped", "stage": "projections", "reason": "module not found"}
    except Exception as exc:
        logger.error("Projections failed: %s", exc)
        return {"status": "error", "stage": "projections", "error": str(exc)}


def stage_odds(season: int, week: int) -> Dict[str, Any]:
    """Stage 3: Refresh odds from external sources."""
    try:
        from scripts.prop_line_scraper import scrape_props
        scrape_props(season, week)
        count_query = (
            "SELECT COUNT(*) as n FROM weekly_odds "
            "WHERE season = ? AND week = ?"
        )
        df = read_dataframe(count_query, params=(season, week))
        n = int(df.iloc[0]["n"]) if not df.empty else 0
        return {"status": "ok", "stage": "odds", "odds_count": n}
    except ImportError:
        logger.warning("prop_line_scraper not available; using existing odds")
        return {"status": "skipped", "stage": "odds", "reason": "module not found"}
    except Exception as exc:
        logger.error("Odds refresh failed: %s", exc)
        return {"status": "error", "stage": "odds", "error": str(exc)}


def stage_value_ranking(season: int, week: int) -> Dict[str, Any]:
    """Stage 4: Rank value opportunities with confidence scoring."""
    try:
        from confidence_engine import score_plays
        from value_betting_engine import rank_weekly_value

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
    """Stage 5: Run risk checks on value opportunities."""
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
    """Stage 6: Run agent coordinator for consensus decisions."""
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


def stage_materialize(season: int, week: int) -> Dict[str, Any]:
    """Stage 7: Materialize final card to value view."""
    try:
        from scripts.materialize_value_view import materialize
        materialize(season, week)
        count_query = (
            "SELECT COUNT(*) as n FROM materialized_value_view "
            "WHERE season = ? AND week = ?"
        )
        df = read_dataframe(count_query, params=(season, week))
        n = int(df.iloc[0]["n"]) if not df.empty else 0
        return {"status": "ok", "stage": "materialize", "card_size": n}
    except ImportError:
        logger.warning("materialize_value_view not available")
        return {"status": "skipped", "stage": "materialize", "reason": "module not found"}
    except Exception as exc:
        logger.error("Materialization failed: %s", exc)
        return {"status": "error", "stage": "materialize", "error": str(exc)}


# ── Orchestrator ─────────────────────────────────────────────────────


STAGES = [
    ("ingest", stage_ingest),
    ("player_dim", stage_player_dim),
    ("projections", stage_projections),
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
) -> Dict[str, Any]:
    """Execute the full production pipeline for a season/week.

    Returns a run report with per-stage results.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    stage_results: List[Dict[str, Any]] = []

    for stage_name, stage_fn in STAGES:
        if skip_ingest and stage_name == "ingest":
            stage_results.append({"status": "skipped", "stage": stage_name, "reason": "skip_ingest"})
            continue
        if skip_odds and stage_name == "odds":
            stage_results.append({"status": "skipped", "stage": stage_name, "reason": "skip_odds"})
            continue

        logger.info("Running stage: %s", stage_name)
        result = stage_fn(season, week)
        stage_results.append(result)
        logger.info("Stage %s: %s", stage_name, result.get("status", "unknown"))

        if result.get("status") == "error":
            logger.warning("Stage %s failed; continuing with remaining stages", stage_name)

    finished_at = datetime.now(timezone.utc).isoformat()

    run_report = {
        "season": season,
        "week": week,
        "started_at": started_at,
        "finished_at": finished_at,
        "stages": stage_results,
        "errors": [s for s in stage_results if s.get("status") == "error"],
        "success": all(s.get("status") != "error" for s in stage_results),
    }

    _persist_run_report(run_report)
    return run_report


def _persist_run_report(report: Dict[str, Any]) -> None:
    """Save run report to logs directory."""
    logs_dir = config.logs_dir / "production_runs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"run_{report['season']}_w{report['week']}_{report['started_at'][:10]}.json"
    try:
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Run report saved to %s", path)
    except Exception as exc:
        logger.error("Failed to save run report: %s", exc)


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Production pipeline runner")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip data ingestion stage")
    parser.add_argument("--skip-odds", action="store_true",
                        help="Skip odds refresh stage")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print(f"Production pipeline: season={args.season} week={args.week}")
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


if __name__ == "__main__":
    main()
