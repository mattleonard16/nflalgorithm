"""Evaluate persisted NFL projections using only point-in-time eligible rows."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from utils.db import read_dataframe
from utils.nfl_markets import MARKET_TO_STAT, melt_actuals

MAX_PROJECTION_AGE = pd.Timedelta(days=7)
PROJECTION_KEYS = ("season", "week", "player_id", "market")
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _timestamps(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True)


def _team_kickoffs(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(columns=["season", "week", "team", "kickoff_utc"])
    home = games[["season", "week", "home_team", "kickoff_utc"]].rename(
        columns={"home_team": "team"}
    )
    away = games[["season", "week", "away_team", "kickoff_utc"]].rename(
        columns={"away_team": "team"}
    )
    return pd.concat([home, away], ignore_index=True).drop_duplicates(
        ["season", "week", "team"], keep="last"
    )


def _freshness_failure(row: pd.Series) -> str | None:
    kickoff = row.get("kickoff_utc")
    generated = row.get("generated_at")
    if pd.isna(kickoff):
        return "missing_kickoff"
    if pd.isna(generated):
        return "missing_projection_timestamp"
    if generated >= kickoff:
        return "projection_after_kickoff"
    if kickoff - generated > MAX_PROJECTION_AGE:
        return "stale_projection"
    return None


def _metric_group(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"projection_count": 0, "mae": None, "rmse": None, "mean_bias": None}
    return {
        "projection_count": int(len(rows)),
        "mae": float(rows["abs_error"].mean()),
        "rmse": float(np.sqrt(np.mean(np.square(rows["signed_error"])))),
        "mean_bias": float(rows["signed_error"].mean()),
    }


def _evaluation_scope(*frames: pd.DataFrame) -> dict[str, Any]:
    season_weeks: set[tuple[int, int]] = set()
    for frame in frames:
        if frame.empty or not {"season", "week"}.issubset(frame.columns):
            continue
        for season, week in frame[["season", "week"]].dropna().itertuples(index=False):
            season_weeks.add((int(season), int(week)))
    return {
        "season_weeks": [{"season": season, "week": week} for season, week in sorted(season_weeks)]
    }


def evaluate_projections(
    projections: pd.DataFrame,
    actuals: pd.DataFrame,
    games: pd.DataFrame,
    runs: pd.DataFrame,
    *,
    candidate_sha: str,
) -> dict[str, Any]:
    """Score production projection rows without training a surrogate model."""
    candidate_sha = candidate_sha.lower()
    scope = _evaluation_scope(projections, actuals, games)
    blockers: list[str] = []
    provenance_failures: Counter[str] = Counter()
    if not SHA_PATTERN.fullmatch(candidate_sha):
        blockers.append("candidate SHA is not a full 40-character Git SHA")
    required_run_columns = {
        "run_id",
        "season",
        "week",
        "status",
        "started_at",
        "finished_at",
        "report_json",
    }
    if scope["season_weeks"] and not required_run_columns.issubset(runs.columns):
        provenance_failures["invalid_run_report"] = len(scope["season_weeks"])
    for item in scope["season_weeks"]:
        if provenance_failures.get("invalid_run_report"):
            break
        season = int(item["season"])
        week = int(item["week"])
        matches = runs[
            (runs["season"] == season) & (runs["week"] == week) & (runs["status"] == "completed")
        ].copy()
        if matches.empty:
            provenance_failures["missing_completed_run"] += 1
            continue
        matches["finished_at"] = _timestamps(matches["finished_at"])
        latest = matches.sort_values(["finished_at", "run_id"]).iloc[-1]
        try:
            report = json.loads(str(latest["report_json"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            provenance_failures["invalid_run_report"] += 1
            continue
        if str(report.get("commit_sha", "")).lower() != candidate_sha:
            provenance_failures["producer_sha_mismatch"] += 1
        started_at = pd.to_datetime(latest["started_at"], errors="coerce", utc=True)
        finished_at = latest["finished_at"]
        generated = _timestamps(
            projections.loc[
                (projections["season"] == season) & (projections["week"] == week),
                "generated_at",
            ]
        )
        if (
            pd.isna(started_at)
            or pd.isna(finished_at)
            or generated.empty
            or generated.isna().any()
            or (generated < started_at).any()
            or (generated > finished_at).any()
        ):
            provenance_failures["outside_run_window"] += 1
    if provenance_failures.get("producer_sha_mismatch"):
        blockers.append("completed run producer SHA does not match candidate SHA")
    if provenance_failures.get("outside_run_window"):
        blockers.append("projections are not bound to the completed producer run")
    if provenance_failures.get("missing_completed_run"):
        blockers.append("evaluation scope is missing a completed producer run")
    if provenance_failures.get("invalid_run_report"):
        blockers.append("completed producer run report is invalid")
    if projections.empty:
        blockers.append("no persisted projections were found")
        eligible = pd.DataFrame()
        failures: Counter[str] = Counter()
        outcome_failures: Counter[str] = Counter()
    elif not scope["season_weeks"]:
        blockers.append("evaluation scope is empty")
        eligible = pd.DataFrame()
        failures = Counter()
        outcome_failures = Counter()
    else:
        frame = projections.copy()
        frame["generated_at"] = _timestamps(frame["generated_at"])
        kickoffs = _team_kickoffs(games.copy())
        if not kickoffs.empty:
            kickoffs["kickoff_utc"] = _timestamps(kickoffs["kickoff_utc"])
        frame = frame.merge(kickoffs, on=["season", "week", "team"], how="left")
        frame = frame.merge(melt_actuals(actuals), on=list(PROJECTION_KEYS), how="left")
        frame["freshness_failure"] = frame.apply(_freshness_failure, axis=1)
        failures = Counter(frame["freshness_failure"].dropna().astype(str))
        if failures:
            blockers.append("projection freshness violations are present")
        missing_actuals = int(frame["actual"].isna().sum())
        outcome_failures = Counter({"missing_actual": missing_actuals} if missing_actuals else {})
        if outcome_failures:
            blockers.append("projection outcome coverage is incomplete")
        eligible = frame[frame["freshness_failure"].isna() & frame["actual"].notna()].copy()
        if not eligible.empty:
            eligible["signed_error"] = eligible["mu"].astype(float) - eligible["actual"].astype(
                float
            )
            eligible["abs_error"] = eligible["signed_error"].abs()

    if eligible.empty:
        blockers.append("no eligible projections with actual outcomes")
        overall = _metric_group(eligible)
        by_market: dict[str, Any] = {}
        by_model_version: dict[str, Any] = {}
    else:
        overall = _metric_group(eligible)
        by_market = {
            str(name): _metric_group(group)
            for name, group in eligible.groupby("market", dropna=False)
        }
        by_model_version = {
            str(name): _metric_group(group)
            for name, group in eligible.groupby("model_version", dropna=False)
        }

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_sha": candidate_sha,
        "scope": scope,
        "passed": not blockers,
        "blockers": blockers,
        "freshness_failures": dict(sorted(failures.items())),
        "provenance_failures": dict(sorted(provenance_failures.items())),
        "outcome_failures": dict(sorted(outcome_failures.items())),
        "metrics": {
            **overall,
            "by_market": by_market,
            "by_model_version": by_model_version,
        },
    }


def _improvement_pct(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0.0 if candidate == 0 else float("-inf")
    return (baseline - candidate) / baseline * 100.0


def compare_reports(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    min_improvement_pct: float,
    max_market_regression_pct: float,
) -> dict[str, Any]:
    """Require overall improvement without hiding material market regressions."""
    blockers: list[str] = []
    if min_improvement_pct <= 0:
        blockers.append("minimum improvement must be greater than zero")
    if max_market_regression_pct < 0:
        blockers.append("maximum market regression cannot be negative")
    if baseline.get("passed") is not True:
        blockers.append("baseline evaluation did not pass")
    if candidate.get("passed") is not True:
        blockers.append("candidate evaluation did not pass")
    baseline_sha = str(baseline.get("candidate_sha", "")).lower()
    candidate_sha = str(candidate.get("candidate_sha", "")).lower()
    if not SHA_PATTERN.fullmatch(baseline_sha):
        blockers.append("baseline evaluation is not bound to a full Git SHA")
    if not SHA_PATTERN.fullmatch(candidate_sha):
        blockers.append("candidate evaluation is not bound to a full Git SHA")
    if baseline_sha == candidate_sha and SHA_PATTERN.fullmatch(baseline_sha):
        blockers.append("baseline and candidate SHAs are identical")
    baseline_scope = baseline.get("scope")
    candidate_scope = candidate.get("scope")
    if baseline_scope != candidate_scope:
        blockers.append("evaluation scope differs between baseline and candidate")
    elif not isinstance(baseline_scope, Mapping) or not baseline_scope.get("season_weeks"):
        blockers.append("evaluation scope is empty")
    baseline_metrics = baseline.get("metrics", {})
    candidate_metrics = candidate.get("metrics", {})
    baseline_mae = baseline_metrics.get("mae")
    candidate_mae = candidate_metrics.get("mae")
    if baseline_mae is None or candidate_mae is None:
        improvement = None
        blockers.append("overall MAE is missing")
    else:
        improvement = _improvement_pct(float(baseline_mae), float(candidate_mae))
        if improvement < min_improvement_pct:
            blockers.append(
                f"overall MAE improvement {improvement:.2f}% is below "
                f"required {min_improvement_pct:.2f}%"
            )

    baseline_count = int(baseline_metrics.get("projection_count", 0))
    candidate_count = int(candidate_metrics.get("projection_count", 0))
    if candidate_count < baseline_count:
        blockers.append(f"projection coverage regressed from {baseline_count} to {candidate_count}")

    market_results: dict[str, Any] = {}
    baseline_markets = baseline_metrics.get("by_market", {})
    candidate_markets = candidate_metrics.get("by_market", {})
    for market in sorted(set(baseline_markets) | set(candidate_markets)):
        before = baseline_markets.get(market)
        after = candidate_markets.get(market)
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            blockers.append(f"{market} market coverage is missing from one evaluation")
            continue
        before_mae = before.get("mae")
        after_mae = after.get("mae")
        if before_mae is None or after_mae is None:
            blockers.append(f"{market} MAE is missing")
            continue
        market_improvement = _improvement_pct(float(before_mae), float(after_mae))
        market_results[market] = {
            "baseline_mae": float(before_mae),
            "candidate_mae": float(after_mae),
            "mae_improvement_pct": market_improvement,
        }
        if market_improvement < -max_market_regression_pct:
            blockers.append(
                f"{market} MAE regressed by {-market_improvement:.2f}%, above "
                f"allowed {max_market_regression_pct:.2f}%"
            )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_sha": baseline_sha,
        "candidate_sha": candidate_sha,
        "scope": candidate_scope,
        "passed": not blockers,
        "blockers": blockers,
        "overall": {
            "baseline_mae": baseline_mae,
            "candidate_mae": candidate_mae,
            "mae_improvement_pct": improvement,
            "baseline_projection_count": baseline_count,
            "candidate_projection_count": candidate_count,
        },
        "by_market": market_results,
    }


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_inputs(season: int, weeks: Iterable[int]) -> tuple[pd.DataFrame, ...]:
    requested = sorted(set(int(week) for week in weeks))
    if not requested:
        raise ValueError("At least one week is required")
    placeholders = ",".join("?" for _ in requested)
    params = (season, *requested)
    where = f"season = ? AND week IN ({placeholders})"
    projections = read_dataframe(
        "SELECT season, week, player_id, team, market, mu, model_version, "
        f"featureset_hash, generated_at FROM weekly_projections WHERE {where}",
        params=params,
    )
    actual_columns = ", ".join(sorted(set(MARKET_TO_STAT.values())))
    actuals = read_dataframe(
        f"SELECT season, week, player_id, {actual_columns} "
        f"FROM player_stats_enhanced WHERE {where}",
        params=params,
    )
    games = read_dataframe(
        f"SELECT season, week, home_team, away_team, kickoff_utc FROM games WHERE {where}",
        params=params,
    )
    runs = read_dataframe(
        "SELECT run_id, season, week, status, started_at, finished_at, report_json "
        f"FROM pipeline_runs WHERE {where} AND status = 'completed'",
        params=params,
    )
    return projections, actuals, games, runs


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--season", type=int, required=True)
    evaluate.add_argument("--weeks", type=int, nargs="+", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--min-improvement-pct", type=float, default=1.0)
    compare.add_argument("--max-market-regression-pct", type=float, default=5.0)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "evaluate":
        report = evaluate_projections(
            *_load_inputs(args.season, args.weeks),
            candidate_sha=_git_sha(),
        )
    else:
        report = compare_reports(
            json.loads(args.baseline.read_text(encoding="utf-8")),
            json.loads(args.candidate.read_text(encoding="utf-8")),
            min_improvement_pct=args.min_improvement_pct,
            max_market_regression_pct=args.max_market_regression_pct,
        )
    _write(args.output, report)
    print(json.dumps(report, indent=2, default=str))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
