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

import pandas as pd

from utils.db import read_dataframe
from utils.nfl_markets import MARKET_TO_STAT, error_summary, melt_actuals, player_positions

MAX_PROJECTION_AGE = pd.Timedelta(days=7)
PROJECTION_KEYS = ("season", "week", "player_id", "market")
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")

# Absolute per-position MAE ceilings, in the units of the projected stat.
# These are yardage-dominated markets, so the ceilings sit far above
# config.model.target_mae; a position without an entry falls back to it.
#
# Calibrated from the 2025 walk-forward backtest (5,117 predictions,
# reports/nfl_backtest_2025_baseline.json): each ceiling is ~10% above that
# position's WORST single-week MAE (QB 59.5, RB 24.0, WR 26.0, TE 24.6), so
# a normal bad week passes and a genuinely broken model trips the gate.
# Re-derive from the latest walk-forward rows CSV before tightening.
POSITION_MAE_THRESHOLDS = {
    "QB": 65.0,
    "RB": 26.0,
    "WR": 29.0,
    "TE": 27.0,
}

# Below this many eligible projections a position MAE is noise. Such positions
# are reported as skipped, never counted as passing.
MIN_POSITION_SAMPLE = 30

UNKNOWN_POSITION = "UNKNOWN"


def _default_position_threshold() -> float:
    """Fallback ceiling for a position with no explicit threshold.

    `config` is a gitignored proprietary module, so it is imported lazily:
    this file's gate logic must remain importable in CI, where config.py is
    absent.
    """
    try:
        from config import config
    except ImportError:
        return 3.0
    return float(config.model.target_mae)


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
    return {"projection_count": int(len(rows)), **error_summary(rows)}


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
        # weekly_projections carries no position, so attach it from the same
        # actuals rows rather than migrating the projections table.
        frame = frame.merge(
            player_positions(actuals, fill_missing=UNKNOWN_POSITION),
            on=["season", "week", "player_id"],
            how="left",
        )
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
        by_position: dict[str, Any] = {}
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
        positions = (
            eligible["position"]
            if "position" in eligible.columns
            else pd.Series(pd.NA, index=eligible.index)
        )
        by_position = {
            str(name): _metric_group(group)
            for name, group in eligible.groupby(positions.fillna(UNKNOWN_POSITION), dropna=False)
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
            "by_position": by_position,
        },
    }


def thresholds_from_backtest(
    backtest_report: Mapping[str, Any],
    *,
    tolerance_pct: float = 10.0,
) -> dict[str, float]:
    """Derive per-position ceilings from a walk-forward backtest report.

    The absolute ceilings in POSITION_MAE_THRESHOLDS are hand-calibrated from
    one baseline run and go stale as the model or slate changes. This turns
    the gate into a regression check against a named baseline instead, with
    the same rule the absolute table was set by: each position's ceiling is
    its worst single-week MAE in the baseline plus `tolerance_pct`. A ceiling
    on the season MEAN would fire on every normal bad week (2025: TE's worst
    week ran 29% above its season MAE). Reports written before
    `worst_week_mae` existed fall back to the season MAE, so pass a wider
    tolerance with those.

    Positions the backtest flagged as `small_sample` are left out — a ceiling
    derived from noise is noise — so they fall back to the absolute table (or
    the config default) exactly as an unlisted position would.

    Raises ValueError when the report carries no usable per-position MAE, so
    an empty or mis-shaped baseline can never quietly become "no ceilings".
    """
    if tolerance_pct < 0:
        raise ValueError("tolerance_pct must be non-negative")
    by_position = backtest_report.get("by_position")
    if not isinstance(by_position, Mapping) or not by_position:
        raise ValueError("backtest report has no by_position metrics")

    ceilings: dict[str, float] = {}
    for position, group in by_position.items():
        if not isinstance(group, Mapping) or group.get("mae") is None:
            continue
        if group.get("small_sample"):
            continue
        anchor = float(group["mae"])
        worst_week = group.get("worst_week_mae")
        if worst_week is not None:
            anchor = max(anchor, float(worst_week))
        ceilings[str(position)] = anchor * (1.0 + tolerance_pct / 100.0)
    if not ceilings:
        raise ValueError("backtest report has no position with a usable MAE")
    return ceilings


def check_position_mae(
    report: Mapping[str, Any],
    thresholds: Mapping[str, float] | None = None,
    *,
    min_sample: int = MIN_POSITION_SAMPLE,
) -> dict[str, Any]:
    """Fail a candidate whose per-position MAE exceeds its absolute ceiling.

    The `compare` path only asks whether a candidate beats a baseline, so a
    model can regress for years while still "improving". This is the absolute
    floor: each position must project within its own ceiling.

    Positions with fewer than `min_sample` eligible projections are reported in
    `skipped` — too little data to judge, and never silently counted as a pass.
    A position present with a missing MAE is a blocker, not a skip.

    Args:
        report: An `evaluate_projections` report.
        thresholds: Per-position MAE ceilings. Defaults to
            POSITION_MAE_THRESHOLDS; positions absent from it fall back to
            `config.model.target_mae`.
        min_sample: Minimum eligible projections for a position to be judged.

    Returns:
        `{"passed", "blockers", "skipped", "by_position"}`.
    """
    ceilings = dict(POSITION_MAE_THRESHOLDS if thresholds is None else thresholds)
    fallback = _default_position_threshold()

    metrics = report.get("metrics")
    by_position = metrics.get("by_position") if isinstance(metrics, Mapping) else None

    blockers: list[str] = []
    skipped: list[dict[str, Any]] = []
    results: dict[str, Any] = {}

    if not isinstance(by_position, Mapping) or not by_position:
        return {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "passed": False,
            "blockers": ["evaluation report has no per-position metrics"],
            "skipped": skipped,
            "by_position": results,
        }

    for position in sorted(by_position):
        group = by_position[position]
        if not isinstance(group, Mapping):
            blockers.append(f"{position} metrics are malformed")
            continue

        threshold = float(ceilings.get(position, fallback))
        count = int(group.get("projection_count") or 0)
        mae = group.get("mae")

        if count < min_sample:
            skipped.append(
                {
                    "position": position,
                    "projection_count": count,
                    "reason": f"below minimum sample of {min_sample}",
                }
            )
            continue

        if mae is None:
            blockers.append(f"{position} MAE is missing")
            continue

        results[position] = {
            "mae": float(mae),
            "threshold": threshold,
            "projection_count": count,
        }
        if float(mae) > threshold:
            blockers.append(
                f"{position} MAE {float(mae):.2f} exceeds threshold {threshold:.2f} "
                f"over {count} projections"
            )

    if not results and not blockers:
        blockers.append("no position met the minimum sample size to be evaluated")

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": not blockers,
        "blockers": blockers,
        "skipped": skipped,
        "by_position": results,
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
        f"SELECT season, week, player_id, position, {actual_columns} "
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
    mae_gate = subparsers.add_parser("mae-gate")
    mae_gate.add_argument("--season", type=int, required=True)
    mae_gate.add_argument("--week", type=int, required=True)
    mae_gate.add_argument("--output", type=Path, default=None)
    mae_gate.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help=(
            "walk-forward backtest report (scripts.run_nfl_backtest run --output); "
            "per-position ceilings become its MAE plus --tolerance-pct instead of "
            "the absolute POSITION_MAE_THRESHOLDS table"
        ),
    )
    mae_gate.add_argument(
        "--tolerance-pct",
        type=float,
        default=10.0,
        help="headroom above the baseline MAE before a position blocks (with --baseline)",
    )
    args = parser.parse_args()

    if args.command == "evaluate":
        report = evaluate_projections(
            *_load_inputs(args.season, args.weeks),
            candidate_sha=_git_sha(),
        )
    elif args.command == "mae-gate":
        evaluation = evaluate_projections(
            *_load_inputs(args.season, [args.week]),
            candidate_sha=_git_sha(),
        )
        thresholds = None
        threshold_source: dict[str, Any] = {"kind": "absolute"}
        if args.baseline is not None:
            baseline_report = json.loads(args.baseline.read_text(encoding="utf-8"))
            thresholds = thresholds_from_backtest(baseline_report, tolerance_pct=args.tolerance_pct)
            threshold_source = {
                "kind": "backtest",
                "path": str(args.baseline),
                "label": baseline_report.get("label"),
                "season": baseline_report.get("season"),
                "tolerance_pct": args.tolerance_pct,
            }
        report = check_position_mae(evaluation, thresholds)
        report["threshold_source"] = threshold_source
        # The evaluation's own blockers (provenance, freshness, coverage) gate
        # the numbers the MAE check reads, so they must fail the gate too.
        if evaluation["passed"] is not True:
            report["blockers"] = [
                *(f"evaluation blocker: {item}" for item in evaluation["blockers"]),
                *report["blockers"],
            ]
            report["passed"] = False
    else:
        report = compare_reports(
            json.loads(args.baseline.read_text(encoding="utf-8")),
            json.loads(args.candidate.read_text(encoding="utf-8")),
            min_improvement_pct=args.min_improvement_pct,
            max_market_regression_pct=args.max_market_regression_pct,
        )
    if args.output is not None:
        _write(args.output, report)
    print(json.dumps(report, indent=2, default=str))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
