"""NBA Production Runner: orchestrates the full daily NBA pipeline.

Calls each pipeline stage in order:
    1. ingest   — refresh player game logs from NBA.com
    2. injuries — ingest injury / DNP data for the game date
    3. predict  — train + predict all 4 markets (pts, reb, ast, fg3m)
    4. odds     — scrape player prop odds and upsert to nba_odds
    5. value    — compute Kelly value bets and materialise view
    6. risk     — run correlation / exposure risk checks
    7. agents   — run multi-agent coordinator for consensus decisions

Can be driven by a scheduler or run manually via CLI:

    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python scripts/nba_production_runner.py --date 2026-02-21
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

MARKETS = ["pts", "reb", "ast", "fg3m"]

# Ordered stage names used to filter via --stages CLI flag
ALL_STAGE_NAMES = ["ingest", "injuries", "predict", "odds", "value", "risk", "agents", "drift"]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _auto_season(game_date: str) -> int:
    """Derive NBA season year from a date string (YYYY-MM-DD).

    Convention: season = year if month >= October, else year - 1.
    E.g. 2026-02-21 -> season 2025, 2025-11-01 -> season 2025.
    """
    dt = datetime.strptime(game_date, "%Y-%m-%d")
    return dt.year if dt.month >= 10 else dt.year - 1


def _result(stage: str, status: str, detail: str = "") -> Dict[str, Any]:
    return {"stage": stage, "status": status, "detail": detail}


# ── Stage functions ───────────────────────────────────────────────────────────


def stage_ingest(game_date: str, season: int) -> Dict[str, Any]:
    """Stage 1: Refresh player game logs from NBA.com."""
    t0 = time.monotonic()
    try:
        from scripts.ingest_nba_data import ingest, ingest_defensive_stats
    except ImportError as exc:
        return _result("ingest", "skipped", f"module not found: {exc}")

    try:
        logger.info("[ingest] starting")
        ingest(seasons=[season])
        logger.info("[ingest] ingesting defensive stats season=%s", season)
        ingest_defensive_stats(seasons=[season])
        elapsed = round(time.monotonic() - t0, 1)
        logger.info("[ingest] done in %ss", elapsed)
        return _result("ingest", "ok", f"season={season} elapsed={elapsed}s")
    except Exception as exc:
        logger.error("[ingest] failed: %s", exc)
        return _result("ingest", "error", str(exc))


def stage_injuries(game_date: str, season: int) -> Dict[str, Any]:
    """Stage 2: Ingest injury / DNP data for game_date."""
    t0 = time.monotonic()
    try:
        from scripts.ingest_nba_injuries import detect_dnps, save_injuries
    except ImportError as exc:
        logger.warning("[injuries] module not available: %s", exc)
        return _result("injuries", "skipped", f"module not found: {exc}")

    try:
        logger.info("[injuries] starting")
        injuries = detect_dnps(game_date)
        saved = save_injuries(injuries) if injuries else 0
        elapsed = round(time.monotonic() - t0, 1)
        logger.info("[injuries] done in %ss — %d records", elapsed, saved)
        return _result("injuries", "ok", f"records={saved} elapsed={elapsed}s")
    except Exception as exc:
        logger.error("[injuries] failed: %s", exc)
        return _result("injuries", "error", str(exc))


def stage_predict(game_date: str, season: int) -> Dict[str, Any]:
    """Stage 3: Train MinutesModel then train + predict all stat markets."""
    t0 = time.monotonic()
    try:
        from models.nba.stat_model import predict, train, _load_minutes_predictions
    except ImportError as exc:
        return _result("predict", "skipped", f"module not found: {exc}")

    # Train (or load) MinutesModel once before running stat markets so that
    # the minutes lookup is shared across all predict() calls.
    minutes_lookup: Dict[str, Any] = {}
    try:
        from models.nba.minutes_model import MinutesModel
        import os
        from pathlib import Path

        db_path = os.environ.get("SQLITE_DB_PATH", "nfl_data.db")
        minutes_model_path = Path(__file__).parent.parent / "models" / "nba" / "minutes_model.joblib"

        mm = MinutesModel()
        logger.info("[predict] training MinutesModel …")
        result_mm = mm.train(db_path)
        logger.info(
            "[predict] MinutesModel cv_mae=%.3f n_samples=%d",
            result_mm["cv_mae"],
            result_mm["n_samples"],
        )
        try:
            mm.save(str(minutes_model_path))
        except Exception as exc:
            logger.warning("[predict] could not save MinutesModel: %s", exc)

        predictions = mm.predict(db_path, target_date=game_date)
        minutes_lookup = {
            p["player_id"]: {
                "predicted_minutes": p["predicted_minutes"],
                "minutes_sigma": p["minutes_sigma"],
            }
            for p in predictions
        }
        logger.info("[predict] MinutesModel produced %d player predictions", len(minutes_lookup))
    except Exception as exc:
        logger.warning("[predict] MinutesModel stage failed (%s); stat models will use fallback minutes", exc)

    errors: List[str] = []
    for market in MARKETS:
        try:
            logger.info("[predict] training market=%s", market)
            train(market)
            logger.info("[predict] predicting market=%s date=%s", market, game_date)
            predict(market, game_date, _minutes_lookup=minutes_lookup)
        except Exception as exc:
            logger.error("[predict] market=%s failed: %s", market, exc)
            errors.append(f"{market}: {exc}")

    elapsed = round(time.monotonic() - t0, 1)
    if errors:
        logger.warning("[predict] completed with %d market error(s)", len(errors))
        return _result("predict", "error", "; ".join(errors))

    logger.info("[predict] all markets done in %ss", elapsed)
    return _result("predict", "ok", f"markets={MARKETS} elapsed={elapsed}s")


def stage_odds(game_date: str, season: int) -> Dict[str, Any]:
    """Stage 4: Scrape player prop odds and upsert to nba_odds."""
    t0 = time.monotonic()
    try:
        from scripts.scrape_nba_odds import scrape_nba_odds, upsert_odds_rows
    except ImportError as exc:
        logger.warning("[odds] module not available: %s", exc)
        return _result("odds", "skipped", f"module not found: {exc}")

    try:
        logger.info("[odds] scraping date=%s season=%d", game_date, season)
        rows = scrape_nba_odds(game_date, season)
        n = upsert_odds_rows(rows) if rows else 0
        elapsed = round(time.monotonic() - t0, 1)
        logger.info("[odds] upserted %d rows in %ss", n, elapsed)
        return _result("odds", "ok", f"rows={n} elapsed={elapsed}s")
    except Exception as exc:
        logger.error("[odds] failed: %s", exc)
        return _result("odds", "error", str(exc))


def stage_value(
    game_date: str,
    season: int,
    use_monte_carlo: bool = False,
    calibrated: bool = False,
) -> Dict[str, Any]:
    """Stage 5: Compute Kelly value bets and materialise view."""
    t0 = time.monotonic()
    try:
        from nba_value_engine import materialize_nba_value
    except ImportError as exc:
        return _result("value", "skipped", f"module not found: {exc}")

    try:
        logger.info(
            "[value] materialising date=%s season=%d monte_carlo=%s calibrated=%s",
            game_date, season, use_monte_carlo, calibrated,
        )
        n = materialize_nba_value(game_date, season, use_monte_carlo=use_monte_carlo, calibrated=calibrated)
        elapsed = round(time.monotonic() - t0, 1)
        logger.info("[value] %d value bets written in %ss", n, elapsed)
        return _result("value", "ok", f"value_bets={n} elapsed={elapsed}s")
    except Exception as exc:
        logger.error("[value] failed: %s", exc)
        return _result("value", "error", str(exc))


def stage_risk(game_date: str, season: int) -> Dict[str, Any]:
    """Stage 6: Run correlation / exposure risk checks."""
    t0 = time.monotonic()
    try:
        from nba_risk_manager import run_risk_check
    except ImportError as exc:
        logger.warning("[risk] module not available: %s", exc)
        return _result("risk", "skipped", f"module not found: {exc}")

    try:
        logger.info("[risk] assessing date=%s", game_date)
        df = run_risk_check(game_date)
        warnings = 0
        if df is not None and not df.empty and "exposure_warning" in df.columns:
            warnings = int(df["exposure_warning"].notna().sum())
        elapsed = round(time.monotonic() - t0, 1)
        logger.info("[risk] done — %d warnings in %ss", warnings, elapsed)
        return _result("risk", "ok", f"warnings={warnings} elapsed={elapsed}s")
    except Exception as exc:
        logger.error("[risk] failed: %s", exc)
        return _result("risk", "error", str(exc))


def stage_agents(game_date: str, season: int) -> Dict[str, Any]:
    """Stage 7: Run multi-agent coordinator for consensus decisions."""
    t0 = time.monotonic()
    try:
        from agents.nba_coordinator import run_nba_agents
    except ImportError as exc:
        logger.warning("[agents] module not available: %s", exc)
        return _result("agents", "skipped", f"module not found: {exc}")

    try:
        logger.info("[agents] running date=%s", game_date)
        decisions = run_nba_agents(game_date)
        approved = sum(1 for d in decisions if d.get("decision") == "APPROVED")
        rejected = sum(1 for d in decisions if d.get("decision") == "REJECTED")
        elapsed = round(time.monotonic() - t0, 1)
        logger.info(
            "[agents] %d decisions (%d approved / %d rejected) in %ss",
            len(decisions),
            approved,
            rejected,
            elapsed,
        )
        return _result(
            "agents",
            "ok",
            f"total={len(decisions)} approved={approved} rejected={rejected} elapsed={elapsed}s",
        )
    except Exception as exc:
        logger.error("[agents] failed: %s", exc)
        return _result("agents", "error", str(exc))


def stage_drift(game_date: str, season: int) -> Dict[str, Any]:
    """Stage 8: Non-blocking drift detection checks."""
    try:
        from utils.nba_drift_detector import run_drift_checks
    except ImportError as exc:
        return _result("drift", "skipped", f"module not found: {exc}")

    try:
        alerts = run_drift_checks(game_date)
        n_alerts = sum(1 for a in alerts if a.get("alert_level") == "alert")
        return _result("drift", "ok", f"checks={len(alerts)} alerts={n_alerts}")
    except Exception as exc:
        logger.error("[drift] failed: %s", exc)
        return _result("drift", "ok", f"non-blocking error: {exc}")  # Never fails pipeline


# ── Stage registry ────────────────────────────────────────────────────────────

STAGES: List[tuple[str, Any]] = [
    ("ingest", stage_ingest),
    ("injuries", stage_injuries),
    ("predict", stage_predict),
    ("odds", stage_odds),
    ("value", stage_value),
    ("risk", stage_risk),
    ("agents", stage_agents),
    ("drift", stage_drift),
]


# ── Orchestrator ──────────────────────────────────────────────────────────────


def run_nba_pipeline(
    game_date: str,
    season: Optional[int] = None,
    stages: Optional[List[str]] = None,
    skip_ingest: bool = False,
    skip_odds: bool = False,
    use_monte_carlo: bool = False,
    calibrated: bool = False,
) -> List[Dict[str, Any]]:
    """Execute the NBA production pipeline for a given game date.

    Args:
        game_date: ISO date string (YYYY-MM-DD) for the slate to process.
        season:    NBA season year (defaults to auto-detection from game_date).
        stages:    If provided, run only these named stages in order.
        skip_ingest: Shorthand to skip the 'ingest' stage.
        skip_odds:   Shorthand to skip the 'odds' stage.
        calibrated:  Use probability calibration in the value stage.

    Returns:
        List of per-stage result dicts.
    """
    if season is None:
        season = _auto_season(game_date)

    active_names = set(stages) if stages else None

    started_at = datetime.now(timezone.utc).isoformat()
    results: List[Dict[str, Any]] = []

    for stage_name, stage_fn in STAGES:
        if active_names is not None and stage_name not in active_names:
            continue
        if skip_ingest and stage_name == "ingest":
            results.append(_result("ingest", "skipped", "skip_ingest flag"))
            continue
        if skip_odds and stage_name == "odds":
            results.append(_result("odds", "skipped", "skip_odds flag"))
            continue

        logger.info("=== Stage: %s ===", stage_name)
        if stage_name == "value":
            result = stage_fn(game_date, season, use_monte_carlo=use_monte_carlo, calibrated=calibrated)
        else:
            result = stage_fn(game_date, season)
        results.append(result)
        logger.info("Stage %s -> %s", stage_name, result["status"])

        if result["status"] == "error":
            logger.warning("Stage %s had errors; continuing pipeline", stage_name)

    finished_at = datetime.now(timezone.utc).isoformat()
    _write_run_report(game_date, season, started_at, finished_at, results)
    return results


def _write_run_report(
    game_date: str,
    season: int,
    started_at: str,
    finished_at: str,
    results: List[Dict[str, Any]],
) -> None:
    """Persist a JSON run report to logs/nba_production_runs/."""
    logs_dir = Path(__file__).parent.parent / "logs" / "nba_production_runs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename = f"{game_date}_{timestamp}.json"
    report = {
        "game_date": game_date,
        "season": season,
        "started_at": started_at,
        "finished_at": finished_at,
        "stages": results,
        "errors": [r for r in results if r["status"] == "error"],
        "success": all(r["status"] != "error" for r in results),
    }

    path = logs_dir / filename
    try:
        with open(path, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info("Run report saved: %s", path)
    except Exception as exc:
        logger.error("Failed to save run report: %s", exc)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NBA production pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python scripts/nba_production_runner.py --date 2026-02-21\n"
            "  uv run python scripts/nba_production_runner.py --date 2026-02-21 --skip-ingest\n"
            "  uv run python scripts/nba_production_runner.py --date 2026-02-21 --stages predict,odds,value\n"
        ),
    )
    parser.add_argument("--date", required=True, metavar="YYYY-MM-DD", help="Game slate date")
    parser.add_argument("--season", type=int, default=None, help="NBA season year (auto-detected if omitted)")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip data ingestion stage")
    parser.add_argument("--skip-odds", action="store_true", help="Skip odds scrape stage")
    parser.add_argument(
        "--stages",
        default=None,
        metavar="STAGE[,STAGE,...]",
        help=f"Comma-separated subset of stages to run. Valid: {','.join(ALL_STAGE_NAMES)}",
    )
    parser.add_argument(
        "--monte-carlo",
        action="store_true",
        dest="monte_carlo",
        help="Use Monte Carlo simulation for probability estimates in the value stage",
    )
    parser.add_argument(
        "--calibrated",
        action="store_true",
        dest="calibrated",
        help="Use probability calibration in the value stage (requires trained calibration model)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    stages_list: Optional[List[str]] = None
    if args.stages:
        stages_list = [s.strip() for s in args.stages.split(",") if s.strip()]
        invalid = [s for s in stages_list if s not in ALL_STAGE_NAMES]
        if invalid:
            parser.error(f"Unknown stage(s): {', '.join(invalid)}. Valid: {', '.join(ALL_STAGE_NAMES)}")

    season = args.season if args.season is not None else _auto_season(args.date)
    print(f"NBA production pipeline: date={args.date} season={season}")

    results = run_nba_pipeline(
        game_date=args.date,
        season=season,
        stages=stages_list,
        skip_ingest=args.skip_ingest,
        skip_odds=args.skip_odds,
        use_monte_carlo=args.monte_carlo,
        calibrated=args.calibrated,
    )

    success = all(r["status"] != "error" for r in results)
    print(f"\nPipeline {'SUCCEEDED' if success else 'HAD ERRORS'}")
    for r in results:
        marker = "[OK  ]" if r["status"] == "ok" else "[SKIP]" if r["status"] == "skipped" else "[ERR ]"
        detail = f" — {r['detail']}" if r.get("detail") else ""
        print(f"  {marker} {r['stage']}{detail}")

    errors = [r for r in results if r["status"] == "error"]
    if errors:
        print(f"\n{len(errors)} stage(s) had errors:")
        for r in errors:
            print(f"  - {r['stage']}: {r['detail']}")


if __name__ == "__main__":
    main()
