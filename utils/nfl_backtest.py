"""NFL walk-forward backtest harness.

Retrain-per-week evaluation of weekly mu/sigma projections against actual
outcomes. The model is injected as ``predict_fn(season, week)`` so this module
stays tracked and CI-testable while the proprietary model wires in through
``scripts/run_nfl_backtest.py``.

The harness owns the honesty rules: each week's predictions are joined only to
that week's actuals, weeks that produce too few evaluable rows are excluded
from aggregates and reported (never silently averaged in), and a run in which
no week survives raises instead of returning an empty "pass".
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import pandas as pd

from utils.nfl_markets import error_summary, melt_actuals, player_positions

REQUIRED_PREDICTION_COLUMNS = ("player_id", "market", "mu")
JOIN_KEYS = ["season", "week", "player_id", "market"]

# Below this many evaluable rows a group's MAE is noise; flagged, never hidden.
MIN_GROUP_SAMPLE = 30

PredictFn = Callable[[int, int], pd.DataFrame]


@dataclass(frozen=True)
class WalkForwardConfig:
    season: int
    weeks: tuple[int, ...]
    label: str = "walk_forward"
    min_week_rows: int = 20
    # Feature flags in effect for the run (name -> bool), recorded on the
    # report so a compare can say what actually differed between two runs.
    features: Mapping[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class WalkForwardResult:
    report: dict[str, Any]
    evaluated: pd.DataFrame = field(repr=False)


def _validate_predictions(predictions: pd.DataFrame, season: int, week: int) -> list[str]:
    problems: list[str] = []
    missing = [col for col in REQUIRED_PREDICTION_COLUMNS if col not in predictions.columns]
    if missing:
        problems.append(f"week {week}: predictions missing columns {missing}")
        return problems
    for column in ("season", "week"):
        expected = season if column == "season" else week
        if column in predictions.columns and not (predictions[column] == expected).all():
            problems.append(f"week {week}: predictions carry mismatched {column} values")
    return problems


def evaluate_week(
    predictions: pd.DataFrame,
    actuals_melted: pd.DataFrame,
    positions: pd.DataFrame,
    *,
    season: int,
    week: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Join one week's predictions to that week's actuals and score errors.

    Returns the evaluated frame plus a list of problems (duplicate prediction
    keys, unmatched rows). The frame carries signed_error/abs_error and, when
    sigma is present and positive, z = signed_error / sigma.
    """
    problems: list[str] = []
    preds = predictions.copy()
    preds["season"] = season
    preds["week"] = week
    preds["player_id"] = preds["player_id"].astype(str)

    duplicates = int(preds.duplicated(subset=JOIN_KEYS).sum())
    if duplicates:
        problems.append(f"week {week}: {duplicates} duplicate prediction keys (kept last)")
        preds = preds.drop_duplicates(subset=JOIN_KEYS, keep="last")

    week_actuals = actuals_melted[
        (actuals_melted["season"] == season) & (actuals_melted["week"] == week)
    ].copy()
    week_actuals["player_id"] = week_actuals["player_id"].astype(str)
    joined = preds.merge(week_actuals, on=JOIN_KEYS, how="inner")

    unmatched = len(preds) - len(joined)
    if unmatched:
        problems.append(f"week {week}: {unmatched} of {len(preds)} predictions had no actual")

    if joined.empty:
        return joined, problems

    if "position" not in joined.columns and not positions.empty:
        joined = joined.merge(positions, on=["season", "week", "player_id"], how="left")

    joined["mu"] = joined["mu"].astype(float)
    joined["actual"] = joined["actual"].astype(float)
    joined["signed_error"] = joined["mu"] - joined["actual"]
    joined["abs_error"] = joined["signed_error"].abs()
    if "sigma" in joined.columns:
        sigma = pd.to_numeric(joined["sigma"], errors="coerce")
        joined["z"] = np.where(sigma > 0, joined["signed_error"] / sigma, np.nan)
    else:
        joined["z"] = np.nan
    return joined, problems


def _worst_week(rows: pd.DataFrame) -> dict[str, Any]:
    """The single evaluated week with the highest MAE inside this group.

    A gate ceiling set from the season MAE alone fires on every normal bad
    week (2025: TE's worst week ran 29% above its season MAE), so the
    per-position ceilings are calibrated from the worst week instead; this
    is what `thresholds_from_backtest` reads.
    """
    if "week" not in rows.columns:
        return {}
    weekly = rows.groupby("week")["abs_error"].mean().dropna()
    if weekly.empty:
        return {}
    worst = weekly.idxmax()
    return {"worst_week": int(worst), "worst_week_mae": float(weekly.loc[worst])}


def _metric_group(rows: pd.DataFrame, *, with_worst_week: bool = True) -> dict[str, Any]:
    group: dict[str, Any] = {
        "count": int(len(rows)),
        **error_summary(rows),
        "small_sample": bool(len(rows) < MIN_GROUP_SAMPLE),
    }
    if with_worst_week:
        group.update(_worst_week(rows))
    z = rows["z"].dropna()
    if not z.empty:
        group["sigma_count"] = int(len(z))
        group["coverage_1sigma"] = float((z.abs() <= 1.0).mean())
        group["z_std"] = float(z.std(ddof=1)) if len(z) > 1 else None
        group["mean_sigma"] = float(pd.to_numeric(rows["sigma"], errors="coerce").dropna().mean())
    return group


def _grouped_metrics(rows: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in rows.columns:
        return {}
    grouped: dict[str, Any] = {}
    for value, group in rows.groupby(rows[column].fillna("UNKNOWN"), dropna=False):
        grouped[str(value)] = _metric_group(group)
    return grouped


def _market_position_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    """Cross of market x position, keyed "market|position".

    Sigma calibration lives in (market, position) buckets
    (utils/nfl_sigma.py), so per-market or per-position views alone
    cannot say which bucket is miscalibrated.
    """
    if "market" not in rows.columns or "position" not in rows.columns:
        return {}
    keys = (
        rows["market"].fillna("UNKNOWN").astype(str)
        + "|"
        + rows["position"].fillna("UNKNOWN").astype(str)
    )
    return {str(key): _metric_group(group) for key, group in rows.groupby(keys)}


def run_walk_forward(
    predict_fn: PredictFn,
    actuals: pd.DataFrame,
    config: WalkForwardConfig,
) -> WalkForwardResult:
    """Run predict_fn once per week and score every week against its actuals.

    Raises ValueError when the requested weeks are empty or when no week
    yields enough evaluable rows — an empty backtest must never look like a
    clean one.
    """
    if not config.weeks:
        raise ValueError("At least one week is required")

    melted = melt_actuals(actuals)
    positions = player_positions(actuals)
    problems: list[str] = []
    evaluated_frames: list[pd.DataFrame] = []
    weeks_evaluated: list[int] = []

    for week in sorted(set(config.weeks)):
        predictions = predict_fn(config.season, week)
        if predictions is None or predictions.empty:
            problems.append(f"week {week}: predict_fn returned no predictions")
            continue
        validation = _validate_predictions(predictions, config.season, week)
        if validation:
            problems.extend(validation)
            continue
        joined, week_problems = evaluate_week(
            predictions, melted, positions, season=config.season, week=week
        )
        problems.extend(week_problems)
        if len(joined) < config.min_week_rows:
            problems.append(
                f"week {week}: only {len(joined)} evaluable rows "
                f"(minimum {config.min_week_rows}); excluded from aggregates"
            )
            continue
        evaluated_frames.append(joined)
        weeks_evaluated.append(week)

    if not evaluated_frames:
        raise ValueError(f"No week produced enough evaluable rows; problems: {problems}")

    evaluated = pd.concat(evaluated_frames, ignore_index=True)
    report = {
        "schema_version": 1,
        "label": config.label,
        "season": config.season,
        "weeks_requested": sorted(set(config.weeks)),
        "weeks_evaluated": weeks_evaluated,
        "features": dict(config.features),
        "problems": problems,
        "overall": _metric_group(evaluated),
        "by_market": _grouped_metrics(evaluated, "market"),
        "by_position": _grouped_metrics(evaluated, "position"),
        "by_market_position": _market_position_metrics(evaluated),
        "by_week": {
            str(week): _metric_group(group, with_worst_week=False)
            for week, group in evaluated.groupby("week")
        },
    }
    return WalkForwardResult(report=report, evaluated=evaluated)


@contextmanager
def feature_overrides(features: Any, **overrides: bool) -> Iterator[dict[str, bool]]:
    """Pin feature flags on a ``config.features`` namespace for one run.

    Yields the full flag state in effect (every boolean attribute, overrides
    applied) so the caller can stamp it on the report, and restores the
    original values on exit — including on error — so a backtest never leaks
    a flag into whatever runs next in the same process.

    An override naming an attribute the namespace does not have raises rather
    than silently creating one the model would never read.
    """
    missing = [name for name in overrides if not hasattr(features, name)]
    if missing:
        raise AttributeError(f"unknown feature flag(s): {', '.join(sorted(missing))}")
    saved = {name: getattr(features, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(features, name, bool(value))
        yield {
            name: bool(value) for name, value in vars(features).items() if isinstance(value, bool)
        }
    finally:
        for name, value in saved.items():
            setattr(features, name, value)


def _improvement_pct(baseline: float, candidate: float) -> float | None:
    if baseline == 0:
        return None
    return (baseline - candidate) / baseline * 100.0


def compare_walk_forward(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare two walk-forward reports over the identical evaluated scope.

    Refuses (via blockers) to compare runs that evaluated different seasons or
    weeks — a candidate must not look better because it skipped a bad week.
    """
    blockers: list[str] = []
    if baseline.get("season") != candidate.get("season"):
        blockers.append("season differs between baseline and candidate")
    if baseline.get("weeks_evaluated") != candidate.get("weeks_evaluated"):
        blockers.append("evaluated weeks differ between baseline and candidate")

    def _delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "baseline_mae": before.get("mae"),
            "candidate_mae": after.get("mae"),
        }
        if before.get("mae") is not None and after.get("mae") is not None:
            result["mae_improvement_pct"] = _improvement_pct(
                float(before["mae"]), float(after["mae"])
            )
        if before.get("coverage_1sigma") is not None and after.get("coverage_1sigma") is not None:
            result["coverage_1sigma_shift"] = float(after["coverage_1sigma"]) - float(
                before["coverage_1sigma"]
            )
        return result

    by_market: dict[str, Any] = {}
    baseline_markets = baseline.get("by_market", {})
    candidate_markets = candidate.get("by_market", {})
    for market in sorted(set(baseline_markets) | set(candidate_markets)):
        before = baseline_markets.get(market)
        after = candidate_markets.get(market)
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            blockers.append(f"{market} is missing from one report")
            continue
        by_market[market] = _delta(before, after)

    return {
        "schema_version": 1,
        "baseline_label": baseline.get("label"),
        "candidate_label": candidate.get("label"),
        "baseline_features": dict(baseline.get("features") or {}),
        "candidate_features": dict(candidate.get("features") or {}),
        "season": candidate.get("season"),
        "weeks_evaluated": candidate.get("weeks_evaluated"),
        "passed": not blockers,
        "blockers": blockers,
        "overall": _delta(baseline.get("overall", {}), candidate.get("overall", {})),
        "by_market": by_market,
    }
