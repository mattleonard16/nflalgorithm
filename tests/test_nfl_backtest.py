"""Tests for the NFL walk-forward backtest harness (utils.nfl_backtest).

The model is a stub predict_fn, so these run in CI without the proprietary
weekly model. The harness owns the honesty rules under test: week isolation,
loud exclusion of thin weeks, refusal to compare mismatched scopes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from utils.nfl_backtest import (
    WalkForwardConfig,
    compare_walk_forward,
    run_walk_forward,
)

SEASON = 2025
MARKET = "receiving_yards"


def _actuals(rows: list[tuple[str, int, float]]) -> pd.DataFrame:
    """rows: (player_id, week, receiving_yards)."""
    return pd.DataFrame(
        [
            {
                "season": SEASON,
                "week": week,
                "player_id": player,
                "position": "WR",
                "receiving_yards": value,
            }
            for player, week, value in rows
        ]
    )


def _predictions(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    """rows: (player_id, mu, sigma)."""
    return pd.DataFrame(
        [
            {"player_id": player, "market": MARKET, "mu": mu, "sigma": sigma}
            for player, mu, sigma in rows
        ]
    )


def _config(weeks: tuple[int, ...], *, min_week_rows: int = 1) -> WalkForwardConfig:
    return WalkForwardConfig(season=SEASON, weeks=weeks, min_week_rows=min_week_rows)


def test_each_week_scores_against_its_own_actuals() -> None:
    actuals = _actuals([("wr_a", 1, 50.0), ("wr_a", 2, 90.0)])

    def predict_fn(season: int, week: int) -> pd.DataFrame:
        return _predictions([("wr_a", 60.0, 10.0)])

    result = run_walk_forward(predict_fn, actuals, _config((1, 2)))

    by_week = result.report["by_week"]
    assert by_week["1"]["mae"] == pytest.approx(10.0)
    assert by_week["2"]["mae"] == pytest.approx(30.0)
    assert result.report["overall"]["mae"] == pytest.approx(20.0)
    assert result.report["overall"]["mean_bias"] == pytest.approx(-10.0)


def test_predict_fn_called_once_per_unique_sorted_week() -> None:
    actuals = _actuals([("wr_a", 1, 50.0), ("wr_a", 3, 50.0)])
    calls: list[int] = []

    def predict_fn(season: int, week: int) -> pd.DataFrame:
        calls.append(week)
        return _predictions([("wr_a", 55.0, 10.0)])

    run_walk_forward(predict_fn, actuals, _config((3, 1, 3)))

    assert calls == [1, 3]


def test_sigma_coverage_and_z_std_reported() -> None:
    actuals = _actuals(
        [("wr_a", 1, 55.0), ("wr_b", 1, 80.0), ("wr_c", 1, 10.0)]
    )

    def predict_fn(season: int, week: int) -> pd.DataFrame:
        # errors: -5 (z=-0.5), 0 (z=0), +30 (z=3) -> coverage 2/3
        return _predictions([("wr_a", 50.0, 10.0), ("wr_b", 80.0, 10.0), ("wr_c", 40.0, 10.0)])

    result = run_walk_forward(predict_fn, actuals, _config((1,)))

    overall = result.report["overall"]
    assert overall["coverage_1sigma"] == pytest.approx(2 / 3)
    assert overall["sigma_count"] == 3
    assert overall["small_sample"] is True


def test_missing_sigma_column_omits_coverage_metrics() -> None:
    actuals = _actuals([("wr_a", 1, 50.0)])

    def predict_fn(season: int, week: int) -> pd.DataFrame:
        return pd.DataFrame([{"player_id": "wr_a", "market": MARKET, "mu": 60.0}])

    result = run_walk_forward(predict_fn, actuals, _config((1,)))

    assert "coverage_1sigma" not in result.report["overall"]
    assert result.report["overall"]["mae"] == pytest.approx(10.0)


def test_empty_week_is_a_problem_but_other_weeks_still_evaluate() -> None:
    actuals = _actuals([("wr_a", 1, 50.0), ("wr_a", 2, 50.0)])

    def predict_fn(season: int, week: int) -> pd.DataFrame:
        if week == 1:
            return pd.DataFrame()
        return _predictions([("wr_a", 55.0, 10.0)])

    result = run_walk_forward(predict_fn, actuals, _config((1, 2)))

    assert result.report["weeks_evaluated"] == [2]
    assert any("week 1" in problem for problem in result.report["problems"])


def test_all_weeks_failing_raises_instead_of_returning_empty_pass() -> None:
    actuals = _actuals([("wr_a", 1, 50.0)])

    def predict_fn(season: int, week: int) -> pd.DataFrame:
        return pd.DataFrame()

    with pytest.raises(ValueError, match="No week produced"):
        run_walk_forward(predict_fn, actuals, _config((1, 2)))


def test_thin_week_is_excluded_from_aggregates_and_reported() -> None:
    actuals = _actuals([("wr_a", 1, 50.0), ("wr_a", 2, 50.0), ("wr_b", 2, 70.0)])

    def predict_fn(season: int, week: int) -> pd.DataFrame:
        if week == 1:
            return _predictions([("wr_a", 90.0, 10.0)])  # 1 row, huge error
        return _predictions([("wr_a", 55.0, 10.0), ("wr_b", 75.0, 10.0)])

    result = run_walk_forward(predict_fn, actuals, _config((1, 2), min_week_rows=2))

    assert result.report["weeks_evaluated"] == [2]
    assert result.report["overall"]["mae"] == pytest.approx(5.0)
    assert any("excluded from aggregates" in problem for problem in result.report["problems"])


def test_duplicate_prediction_keys_dedupe_keep_last_and_report() -> None:
    actuals = _actuals([("wr_a", 1, 50.0)])

    def predict_fn(season: int, week: int) -> pd.DataFrame:
        return _predictions([("wr_a", 10.0, 10.0), ("wr_a", 55.0, 10.0)])

    result = run_walk_forward(predict_fn, actuals, _config((1,)))

    assert result.report["overall"]["mae"] == pytest.approx(5.0)
    assert any("duplicate prediction keys" in problem for problem in result.report["problems"])


def test_mismatched_week_column_skips_week_with_problem() -> None:
    actuals = _actuals([("wr_a", 1, 50.0), ("wr_a", 2, 50.0)])

    def predict_fn(season: int, week: int) -> pd.DataFrame:
        preds = _predictions([("wr_a", 55.0, 10.0)])
        preds["week"] = 99  # claims a different week than requested
        if week == 2:
            preds["week"] = 2
        return preds

    result = run_walk_forward(predict_fn, actuals, _config((1, 2)))

    assert result.report["weeks_evaluated"] == [2]
    assert any("mismatched week" in problem for problem in result.report["problems"])


def test_position_grouping_attaches_from_actuals() -> None:
    actuals = _actuals([("wr_a", 1, 50.0)])

    def predict_fn(season: int, week: int) -> pd.DataFrame:
        return _predictions([("wr_a", 55.0, 10.0)])

    result = run_walk_forward(predict_fn, actuals, _config((1,)))

    assert "WR" in result.report["by_position"]
    assert result.report["by_position"]["WR"]["count"] == 1


def _report_for_compare(mae: float, coverage: float, weeks: list[int]) -> dict:
    group = {"mae": mae, "coverage_1sigma": coverage}
    return {
        "label": f"run_mae_{mae}",
        "season": SEASON,
        "weeks_evaluated": weeks,
        "overall": group,
        "by_market": {MARKET: group},
    }


def test_compare_reports_deltas_on_identical_scope() -> None:
    baseline = _report_for_compare(10.0, 0.60, [1, 2])
    candidate = _report_for_compare(9.0, 0.68, [1, 2])

    comparison = compare_walk_forward(baseline, candidate)

    assert comparison["passed"] is True
    assert comparison["overall"]["mae_improvement_pct"] == pytest.approx(10.0)
    assert comparison["overall"]["coverage_1sigma_shift"] == pytest.approx(0.08)
    assert comparison["by_market"][MARKET]["mae_improvement_pct"] == pytest.approx(10.0)


def test_compare_refuses_mismatched_weeks() -> None:
    baseline = _report_for_compare(10.0, 0.60, [1, 2])
    candidate = _report_for_compare(5.0, 0.68, [2])

    comparison = compare_walk_forward(baseline, candidate)

    assert comparison["passed"] is False
    assert any("weeks differ" in blocker for blocker in comparison["blockers"])


def test_no_weeks_requested_raises() -> None:
    actuals = _actuals([("wr_a", 1, 50.0)])

    with pytest.raises(ValueError, match="At least one week"):
        run_walk_forward(lambda s, w: pd.DataFrame(), actuals, _config(()))
