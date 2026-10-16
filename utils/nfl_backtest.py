"""NFL walk-forward backtest harness.

Retrain-per-week evaluation of weekly mu/sigma projections against actual
outcomes. The model is injected as ``predict_fn(season, week)`` so this module
stays tracked and CI-testable while the proprietary model wires in through
``scripts/run_nfl_backtest.py``.

The harness owns the honesty rules: each week's predictions are joined only to
that week's actuals, weeks that produce too few evaluable rows or leave too
many predictions without an actual are excluded from aggregates and reported
(never silently averaged in), and a run in which no week survives raises
instead of returning an empty "pass".

Every report fingerprints what it scored (per-market row counts and a sha256
of the sorted ``(week, player_id, market)`` keys), and ``compare_walk_forward``
refuses two reports whose fingerprints differ: a candidate must not look
better by scoring fewer or different rows.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, NamedTuple

import numpy as np
import pandas as pd

from scripts.evaluate_nfl_projections import YARDAGE_MARKETS
from utils.nfl_markets import error_summary, melt_actuals, player_positions

REQUIRED_PREDICTION_COLUMNS = ("player_id", "market", "mu")
JOIN_KEYS = ["season", "week", "player_id", "market"]

# Below this many evaluable rows a group's MAE is noise; flagged, never hidden.
MIN_GROUP_SAMPLE = 30

# A prediction with no actual (no stat row: a bust, an inactive player, an id
# mismatch) cannot be scored, and dropping it flatters MAE by exclusion. A
# week leaving more than this share unscored is excluded from aggregates.
MAX_UNMATCHED_FRACTION = 0.10

REPORT_SCHEMA_VERSION = 2
# Scope fingerprint fields compare_walk_forward requires on both reports.
SCOPE_FIELDS = ("market_counts", "evaluated_keys_sha256")

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


class WeekEvaluation(NamedTuple):
    """One week's scored rows plus how many predictions went unscored."""

    frame: pd.DataFrame
    problems: list[str]
    predicted: int
    unmatched: int


def evaluate_week(
    predictions: pd.DataFrame,
    actuals_melted: pd.DataFrame,
    positions: pd.DataFrame,
    *,
    season: int,
    week: int,
) -> WeekEvaluation:
    """Join one week's predictions to that week's actuals and score errors.

    Returns the evaluated frame, a list of problems (duplicate prediction
    keys, unmatched rows), and the deduplicated prediction count with how
    many of them had no actual (no stat row, or a null stat). The frame
    carries signed_error/abs_error and, when sigma is present and positive,
    z = signed_error / sigma.
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
    # A null stat is no actual either: scoring it would yield a NaN error
    # that counts as a row but drops out of every mean.
    joined = joined[pd.to_numeric(joined["actual"], errors="coerce").notna()]

    unmatched = len(preds) - len(joined)
    if unmatched:
        problems.append(f"week {week}: {unmatched} of {len(preds)} predictions had no actual")

    if joined.empty:
        return WeekEvaluation(joined, problems, len(preds), unmatched)

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
    return WeekEvaluation(joined, problems, len(preds), unmatched)


def _metric_group(rows: pd.DataFrame) -> dict[str, Any]:
    group: dict[str, Any] = {
        "count": int(len(rows)),
        **error_summary(rows),
        "small_sample": bool(len(rows) < MIN_GROUP_SAMPLE),
    }
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


def evaluated_keys_sha256(evaluated: pd.DataFrame) -> str:
    """sha256 over the sorted ``(week, player_id, market)`` keys scored.

    Two runs with the same fingerprint scored exactly the same rows, so an
    MAE difference between them is the model, not the sample.
    """
    keys = sorted(
        f"{int(week)}|{player_id}|{market}"
        for week, player_id, market in zip(
            evaluated["week"], evaluated["player_id"].astype(str), evaluated["market"].astype(str)
        )
    )
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def run_walk_forward(
    predict_fn: PredictFn,
    actuals: pd.DataFrame,
    config: WalkForwardConfig,
) -> WalkForwardResult:
    """Run predict_fn once per week and score every week against its actuals.

    Raises ValueError when the requested weeks are empty or when no week
    yields enough evaluable rows — an empty backtest must never look like a
    clean one. A week leaving more than ``MAX_UNMATCHED_FRACTION`` of its
    predictions without an actual is excluded and reported like a thin week.

    ``by_position`` aggregates ``YARDAGE_MARKETS`` only, matching the
    per-position MAE gate in scripts/evaluate_nfl_projections.py (count
    props err on a far smaller scale and would deflate a yardage MAE);
    ``by_market_position`` keeps every market.
    """
    if not config.weeks:
        raise ValueError("At least one week is required")

    melted = melt_actuals(actuals)
    positions = player_positions(actuals)
    problems: list[str] = []
    evaluated_frames: list[pd.DataFrame] = []
    weeks_evaluated: list[int] = []
    predicted_total = 0
    unmatched_total = 0

    for week in sorted(set(config.weeks)):
        predictions = predict_fn(config.season, week)
        if predictions is None or predictions.empty:
            problems.append(f"week {week}: predict_fn returned no predictions")
            continue
        validation = _validate_predictions(predictions, config.season, week)
        if validation:
            problems.extend(validation)
            continue
        joined, week_problems, predicted, unmatched = evaluate_week(
            predictions, melted, positions, season=config.season, week=week
        )
        problems.extend(week_problems)
        if predicted and unmatched / predicted > MAX_UNMATCHED_FRACTION:
            problems.append(
                f"week {week}: {unmatched / predicted:.1%} of predictions had no actual "
                f"(maximum {MAX_UNMATCHED_FRACTION:.0%}); excluded from aggregates"
            )
            continue
        if len(joined) < config.min_week_rows:
            problems.append(
                f"week {week}: only {len(joined)} evaluable rows "
                f"(minimum {config.min_week_rows}); excluded from aggregates"
            )
            continue
        evaluated_frames.append(joined)
        weeks_evaluated.append(week)
        predicted_total += predicted
        unmatched_total += unmatched

    if not evaluated_frames:
        raise ValueError(f"No week produced enough evaluable rows; problems: {problems}")

    evaluated = pd.concat(evaluated_frames, ignore_index=True)
    yardage = evaluated[evaluated["market"].isin(YARDAGE_MARKETS)]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "label": config.label,
        "season": config.season,
        "weeks_requested": sorted(set(config.weeks)),
        "weeks_evaluated": weeks_evaluated,
        "features": dict(config.features),
        "problems": problems,
        "market_counts": {
            str(market): int(count)
            for market, count in evaluated["market"].value_counts().sort_index().items()
        },
        "evaluated_keys_sha256": evaluated_keys_sha256(evaluated),
        "match_coverage": {
            "predictions": predicted_total,
            "unmatched": unmatched_total,
            "unmatched_fraction": unmatched_total / predicted_total if predicted_total else 0.0,
            "max_unmatched_fraction": MAX_UNMATCHED_FRACTION,
        },
        "overall": _metric_group(evaluated),
        "by_market": _grouped_metrics(evaluated, "market"),
        "by_position": _grouped_metrics(yardage, "position"),
        "by_market_position": _market_position_metrics(evaluated),
        "by_week": {str(week): _metric_group(group) for week, group in evaluated.groupby("week")},
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


def _scope_blockers(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[str]:
    """Blockers for two reports that did not score the same rows."""
    blockers: list[str] = []
    reports = (("baseline", baseline), ("candidate", candidate))
    stale = False
    for name, report in reports:
        if any(report.get(field_name) is None for field_name in SCOPE_FIELDS):
            stale = True
            blockers.append(
                f"{name} report (label {report.get('label')!r}, schema_version "
                f"{report.get('schema_version', 1)}) has no evaluated-scope fingerprint "
                f"({', '.join(SCOPE_FIELDS)}); re-run it with the current "
                "scripts.run_nfl_backtest before comparing"
            )
    if not stale:
        if dict(baseline["market_counts"]) != dict(candidate["market_counts"]):
            blockers.append(
                "evaluated row counts per market differ between baseline and candidate: "
                f"{dict(baseline['market_counts'])} vs {dict(candidate['market_counts'])}"
            )
        elif baseline["evaluated_keys_sha256"] != candidate["evaluated_keys_sha256"]:
            blockers.append(
                "evaluated (week, player_id, market) keys differ between baseline and candidate"
            )
    for name, report in reports:
        fraction = (report.get("match_coverage") or {}).get("unmatched_fraction")
        if fraction is not None and float(fraction) > MAX_UNMATCHED_FRACTION:
            blockers.append(
                f"{name} left {float(fraction):.1%} of predictions without an actual "
                f"(maximum {MAX_UNMATCHED_FRACTION:.0%})"
            )
    return blockers


def _improvement_pct(baseline: float, candidate: float) -> float | None:
    if baseline == 0:
        return None
    return (baseline - candidate) / baseline * 100.0


def compare_walk_forward(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare two walk-forward reports over the identical evaluated scope.

    Refuses (via blockers) to compare runs that evaluated different seasons,
    weeks, per-market row counts or ``(week, player_id, market)`` keys — a
    candidate must not look better because it skipped a bad week or scored
    fewer rows. Reports predating the scope fingerprint (schema_version 1)
    are refused with a request to re-run them.
    """
    blockers: list[str] = []
    if baseline.get("season") != candidate.get("season"):
        blockers.append("season differs between baseline and candidate")
    if baseline.get("weeks_evaluated") != candidate.get("weeks_evaluated"):
        blockers.append("evaluated weeks differ between baseline and candidate")
    blockers.extend(_scope_blockers(baseline, candidate))

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
        "schema_version": REPORT_SCHEMA_VERSION,
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
