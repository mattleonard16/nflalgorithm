"""Outcome-based replay coverage for the NFL prediction pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.backtest_replay import build_replay_dataset, compute_metrics


def _replay_inputs() -> tuple[pd.DataFrame, ...]:
    projections = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "player_id": "BUF_wr1",
                "team": "BUF",
                "market": "receiving_yards",
                "mu": 65.0,
                "sigma": 10.0,
                "generated_at": "2026-09-08T16:00:00Z",
            }
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "player_id": "BUF_wr1",
                "market": "receiving_yards",
                "sportsbook": "book",
                "line": 50.5,
                "price": -110,
                "under_price": -110,
                "as_of": "2026-09-09T16:00:00Z",
            },
            {
                "season": 2026,
                "week": 1,
                "player_id": "BUF_wr1",
                "market": "receiving_yards",
                "sportsbook": "book",
                "line": 52.5,
                "price": -110,
                "under_price": -110,
                "as_of": "2026-09-10T15:00:00Z",
            },
            # This post-kickoff line must never become the closing line.
            {
                "season": 2026,
                "week": 1,
                "player_id": "BUF_wr1",
                "market": "receiving_yards",
                "sportsbook": "book",
                "line": 70.5,
                "price": -110,
                "under_price": -110,
                "as_of": "2026-09-10T18:00:00Z",
            },
        ]
    )
    actuals = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "player_id": "BUF_wr1",
                "receiving_yards": 70.0,
            }
        ]
    )
    context = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "player_id": "BUF_wr1",
                "team": "BUF",
                "position": "WR",
                "is_starter": 1,
                "depth_rank": 1,
                "is_rookie": 0,
                "is_new_team": 0,
                "source_updated_at": "2026-09-08T12:00:00Z",
                "captured_at": "2026-09-08T12:00:00Z",
            }
        ]
    )
    games = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "home_team": "BUF",
                "away_team": "MIA",
                "kickoff_utc": "2026-09-10T17:00:00Z",
            }
        ]
    )
    return projections, odds, actuals, context, games


def test_replay_uses_only_information_available_before_kickoff() -> None:
    replay = build_replay_dataset(*_replay_inputs(), min_edge=0.05)

    row = replay.iloc[0]
    assert row["entry_line"] == 50.5
    assert row["close_line"] == 52.5
    assert row["side"] == "over"
    assert row["actual"] == 70.0
    assert row["result"] == "win"
    assert row["clv_line"] == 2.0
    assert bool(row["freshness_pass"]) is True
    assert row["role"] == "starter"


def test_replay_fails_freshness_when_projection_is_post_kickoff() -> None:
    projections, odds, actuals, context, games = _replay_inputs()
    projections.loc[0, "generated_at"] = "2026-09-10T18:00:00Z"

    replay = build_replay_dataset(projections, odds, actuals, context, games, min_edge=0.0)

    assert bool(replay.iloc[0]["freshness_pass"]) is False
    assert "projection_after_kickoff" in replay.iloc[0]["freshness_failures"]


def test_replay_fails_when_context_source_is_post_kickoff() -> None:
    projections, odds, actuals, context, games = _replay_inputs()
    context.loc[0, "source_updated_at"] = "2026-09-10T18:00:00Z"

    replay = build_replay_dataset(projections, odds, actuals, context, games, min_edge=0.0)

    assert bool(replay.iloc[0]["freshness_pass"]) is False
    assert "context_source_after_kickoff" in replay.iloc[0]["freshness_failures"]


def test_metrics_use_actual_results_and_report_market_role_calibration() -> None:
    replay = build_replay_dataset(*_replay_inputs(), min_edge=0.05)

    metrics = compute_metrics(replay)

    assert metrics["bets_placed"] == 1
    assert metrics["wins"] == 1
    assert metrics["roi"] == pytest.approx(100 / 110)
    assert metrics["mae"] == 5.0
    assert metrics["avg_clv_line"] == 2.0
    assert metrics["brier"] == pytest.approx((replay.iloc[0]["p_win"] - 1.0) ** 2)
    assert metrics["by_market"]["receiving_yards"]["bets"] == 1
    assert metrics["by_role"]["starter"]["bets"] == 1
    assert metrics["calibration"][0]["observed_win_rate"] == 1.0


def test_metrics_exclude_rows_that_fail_freshness_gate() -> None:
    projections, odds, actuals, context, games = _replay_inputs()
    context.loc[0, "captured_at"] = "2026-09-10T18:00:00Z"
    replay = build_replay_dataset(projections, odds, actuals, context, games, min_edge=0.0)

    metrics = compute_metrics(replay)

    assert metrics["rows_seen"] == 1
    assert metrics["fresh_rows"] == 0
    assert metrics["bets_placed"] == 0
    assert metrics["freshness_failures"]["context_after_kickoff"] == 1
    assert metrics["freshness_by_market"]["receiving_yards"]["fresh_rate"] == 0.0
    assert metrics["freshness_by_role"]["starter"]["failures"] == {"context_after_kickoff": 1}


def test_accuracy_and_calibration_include_fresh_rows_below_betting_threshold() -> None:
    replay = build_replay_dataset(*_replay_inputs(), min_edge=1.0)

    metrics = compute_metrics(replay)

    assert metrics["bets_placed"] == 0
    assert metrics["mae"] == 5.0
    assert metrics["brier"] == pytest.approx((replay.iloc[0]["p_win"] - 1.0) ** 2)
    assert metrics["by_market"]["receiving_yards"]["projections"] == 1
    assert metrics["by_market"]["receiving_yards"]["bets"] == 0
    assert metrics["calibration"][0]["observed_win_rate"] == 1.0


def test_missing_context_is_reported_instead_of_crashing() -> None:
    projections, odds, actuals, _, games = _replay_inputs()

    replay = build_replay_dataset(projections, odds, actuals, pd.DataFrame(), games, min_edge=0.0)

    assert replay.iloc[0]["role"] == "unknown"
    assert "missing_context_snapshot" in replay.iloc[0]["freshness_failures"]
