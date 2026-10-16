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
    feature_overrides,
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
    actuals = _actuals([("wr_a", 1, 55.0), ("wr_b", 1, 80.0), ("wr_c", 1, 10.0)])

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


def test_market_position_cross_grouping() -> None:
    # Sigma calibration is bucketed by (market, position); the report must
    # split the cross, not just each axis alone.
    actuals = pd.DataFrame(
        [
            {
                "season": SEASON,
                "week": 1,
                "player_id": "wr_a",
                "position": "WR",
                "receiving_yards": 50.0,
                "rushing_yards": 5.0,
            },
            {
                "season": SEASON,
                "week": 1,
                "player_id": "rb_a",
                "position": "RB",
                "receiving_yards": 20.0,
                "rushing_yards": 80.0,
            },
        ]
    )

    def predict_fn(season: int, week: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"player_id": "wr_a", "market": "receiving_yards", "mu": 60.0, "sigma": 10.0},
                {"player_id": "rb_a", "market": "receiving_yards", "mu": 25.0, "sigma": 10.0},
                {"player_id": "rb_a", "market": "rushing_yards", "mu": 70.0, "sigma": 10.0},
            ]
        )

    result = run_walk_forward(predict_fn, actuals, _config((1,)))

    cross = result.report["by_market_position"]
    assert set(cross) == {
        "receiving_yards|WR",
        "receiving_yards|RB",
        "rushing_yards|RB",
    }
    assert cross["receiving_yards|WR"]["mae"] == pytest.approx(10.0)
    assert cross["receiving_yards|RB"]["mae"] == pytest.approx(5.0)
    assert cross["rushing_yards|RB"]["mean_bias"] == pytest.approx(-10.0)


def _report_for_compare(
    mae: float,
    coverage: float,
    weeks: list[int],
    *,
    count: int = 100,
    keys_sha: str = "same-rows",
) -> dict:
    group = {"mae": mae, "coverage_1sigma": coverage}
    return {
        "schema_version": 2,
        "label": f"run_mae_{mae}",
        "season": SEASON,
        "weeks_evaluated": weeks,
        "market_counts": {MARKET: count},
        "evaluated_keys_sha256": keys_sha,
        "match_coverage": {"predictions": count, "unmatched": 0, "unmatched_fraction": 0.0},
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


def test_report_records_features_in_effect() -> None:
    actuals = _actuals([("wr_a", 1, 50.0), ("wr_b", 1, 60.0)])
    predictions = pd.DataFrame(
        [
            {"season": SEASON, "week": 1, "player_id": "wr_a", "market": MARKET, "mu": 55.0},
            {"season": SEASON, "week": 1, "player_id": "wr_b", "market": MARKET, "mu": 58.0},
        ]
    )
    config = WalkForwardConfig(
        season=SEASON, weeks=(1,), min_week_rows=1, features={"context_factors_enabled": True}
    )

    result = run_walk_forward(lambda s, w: predictions, actuals, config)

    assert result.report["features"] == {"context_factors_enabled": True}


def test_compare_reports_which_features_differed() -> None:
    baseline = {**_report_for_compare(10.0, 0.60, [1, 2]), "features": {"ctx": False}}
    candidate = {**_report_for_compare(9.0, 0.68, [1, 2]), "features": {"ctx": True}}

    comparison = compare_walk_forward(baseline, candidate)

    assert comparison["passed"] is True
    assert comparison["baseline_features"] == {"ctx": False}
    assert comparison["candidate_features"] == {"ctx": True}


def test_feature_overrides_pin_then_restore_flags() -> None:
    from types import SimpleNamespace

    features = SimpleNamespace(context_factors_enabled=False, kelly_cap_enabled=True, label="x")

    with feature_overrides(features, context_factors_enabled=True) as in_effect:
        assert features.context_factors_enabled is True
        assert in_effect == {"context_factors_enabled": True, "kelly_cap_enabled": True}

    assert features.context_factors_enabled is False


def test_feature_overrides_restore_on_error_and_reject_unknown_flags() -> None:
    from types import SimpleNamespace

    features = SimpleNamespace(context_factors_enabled=False)

    with pytest.raises(RuntimeError):
        with feature_overrides(features, context_factors_enabled=True):
            raise RuntimeError("model blew up")
    assert features.context_factors_enabled is False

    with pytest.raises(AttributeError, match="unknown feature flag"):
        with feature_overrides(features, no_such_flag=True):
            pass
    assert not hasattr(features, "no_such_flag")


# ---------------------------------------------------------------------------
# Evaluated-scope fingerprint, unmatched predictions, yardage-only positions
# ---------------------------------------------------------------------------


def test_compare_refuses_fewer_rows_on_the_same_weeks() -> None:
    """The verified defect: 5,117 rows at MAE 26.88 vs 2,000 rows at MAE 20
    on the same weeks used to pass."""
    baseline = _report_for_compare(26.88, 0.68, [1, 2], count=5117, keys_sha="full")
    candidate = _report_for_compare(20.0, 0.68, [1, 2], count=2000, keys_sha="subset")

    comparison = compare_walk_forward(baseline, candidate)

    assert comparison["passed"] is False
    assert any("row counts per market differ" in b for b in comparison["blockers"])


def test_compare_refuses_same_counts_but_different_rows() -> None:
    baseline = _report_for_compare(10.0, 0.68, [1, 2], keys_sha="rows-a")
    candidate = _report_for_compare(9.0, 0.68, [1, 2], keys_sha="rows-b")

    comparison = compare_walk_forward(baseline, candidate)

    assert comparison["passed"] is False
    assert any("player_id, market) keys differ" in b for b in comparison["blockers"])


def test_compare_refuses_reports_without_a_fingerprint() -> None:
    baseline = _report_for_compare(10.0, 0.68, [1, 2])
    for field_name in ("schema_version", "market_counts", "evaluated_keys_sha256"):
        baseline.pop(field_name)
    candidate = _report_for_compare(9.0, 0.68, [1, 2])

    comparison = compare_walk_forward(baseline, candidate)

    assert comparison["passed"] is False
    assert any(
        b.startswith("baseline report") and "re-run" in b for b in comparison["blockers"]
    )


def test_compare_refuses_a_report_with_too_many_unscored_predictions() -> None:
    baseline = _report_for_compare(10.0, 0.68, [1, 2])
    candidate = _report_for_compare(9.0, 0.68, [1, 2])
    candidate["match_coverage"]["unmatched_fraction"] = 0.25

    comparison = compare_walk_forward(baseline, candidate)

    assert comparison["passed"] is False
    assert any("without an actual" in b for b in comparison["blockers"])


def test_identical_runs_share_a_fingerprint_and_compare_cleanly() -> None:
    actuals = _actuals([("wr_a", 1, 50.0), ("wr_b", 1, 60.0), ("wr_a", 2, 70.0)])

    def predict_fn(season: int, week: int) -> pd.DataFrame:
        if week == 1:
            return _predictions([("wr_b", 58.0, 10.0), ("wr_a", 55.0, 10.0)])
        return _predictions([("wr_a", 66.0, 10.0)])

    first = run_walk_forward(predict_fn, actuals, _config((1, 2))).report
    second = run_walk_forward(predict_fn, actuals, _config((1, 2))).report

    assert first["schema_version"] == 2
    assert first["market_counts"] == {MARKET: 3}
    assert len(first["evaluated_keys_sha256"]) == 64
    assert first["evaluated_keys_sha256"] == second["evaluated_keys_sha256"]
    assert compare_walk_forward(first, second)["passed"] is True


def test_week_with_too_many_unscored_predictions_is_excluded() -> None:
    # Week 1: 1 of 2 predictions has no stat row (50% unmatched) -> excluded,
    # even though it clears min_week_rows. Week 2 is clean.
    actuals = _actuals([("wr_a", 1, 50.0), ("wr_a", 2, 50.0), ("wr_b", 2, 40.0)])

    def predict_fn(season: int, week: int) -> pd.DataFrame:
        return _predictions([("wr_a", 52.0, 10.0), ("wr_b", 45.0, 10.0)])

    result = run_walk_forward(predict_fn, actuals, _config((1, 2)))

    assert result.report["weeks_evaluated"] == [2]
    assert any(
        "week 1" in p and "excluded from aggregates" in p for p in result.report["problems"]
    )
    assert result.report["match_coverage"]["unmatched"] == 0


def test_null_stat_counts_as_unmatched_not_as_a_scored_row() -> None:
    actuals = _actuals([(f"wr_{i}", 1, 50.0) for i in range(20)] + [("wr_null", 1, None)])

    def predict_fn(season: int, week: int) -> pd.DataFrame:
        return _predictions([(f"wr_{i}", 50.0, 10.0) for i in range(20)] + [("wr_null", 0.0, 1.0)])

    report = run_walk_forward(predict_fn, actuals, _config((1,))).report

    assert report["overall"]["count"] == 20
    assert report["match_coverage"] == {
        "predictions": 21,
        "unmatched": 1,
        "unmatched_fraction": pytest.approx(1 / 21),
        "max_unmatched_fraction": 0.10,
    }


def test_by_position_is_yardage_only_while_market_position_keeps_all() -> None:
    actuals = pd.DataFrame(
        [
            {
                "season": SEASON,
                "week": 1,
                "player_id": "wr_a",
                "position": "WR",
                "receiving_yards": 50.0,
                "receptions": 5.0,
            }
        ]
    )

    def predict_fn(season: int, week: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"player_id": "wr_a", "market": "receiving_yards", "mu": 60.0, "sigma": 10.0},
                {"player_id": "wr_a", "market": "receptions", "mu": 5.5, "sigma": 1.0},
            ]
        )

    report = run_walk_forward(predict_fn, actuals, _config((1,))).report

    assert report["by_position"]["WR"]["count"] == 1
    assert report["by_position"]["WR"]["mae"] == pytest.approx(10.0)
    assert set(report["by_market_position"]) == {"receiving_yards|WR", "receptions|WR"}
    assert report["market_counts"] == {"receiving_yards": 1, "receptions": 1}
