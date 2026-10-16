"""The core-player context factor evaluation harness.

What matters is that it measures the right population on pregame-only inputs:
starters by trailing usage, a baseline that never sees the graded game, and
deltas signed so that negative means the factor helped.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.evaluate_context_factors import (
    COMPONENTS,
    EvalConfig,
    build_rows,
    summarize,
)

SEASON = 2025


def _games(weeks=range(1, 9)):
    return pd.DataFrame(
        [
            {
                "season": SEASON,
                "week": w,
                "home_team": "KC",
                "away_team": "DEN",
                "spread_line": 7.0,
                "total_line": 45.0,
            }
            for w in weeks
        ]
    )


def _row(gsis, team, position, week, *, targets=0.0, rec=0.0, carries=0.0, rush=0.0):
    return {
        "player_id": f"{team}_{gsis}",
        "gsis_id": gsis,
        "season": SEASON,
        "week": week,
        "team": team,
        "position": position,
        "passing_yards": 0.0,
        "rushing_yards": rush,
        "receiving_yards": rec,
        "passing_attempts": 0.0,
        "rushing_attempts": carries,
        "targets": targets,
    }


def _stats():
    rows = []
    for week in range(1, 9):
        # A starter: 8 targets every week, 60 yards except a 200-yard week 6.
        rows.append(_row("wr1", "KC", "WR", week, targets=8.0, rec=200.0 if week == 6 else 60.0))
        # A depth receiver: 2 targets, below the WR starter floor.
        rows.append(_row("wr4", "KC", "WR", week, targets=2.0, rec=15.0))
        # An RB1 on the other side.
        rows.append(_row("rb1", "DEN", "RB", week, carries=16.0, rush=70.0))
    return pd.DataFrame(rows)


CONFIG = EvalConfig(first_week=4, last_week=8, bootstrap_draws=50)


def test_only_starters_by_trailing_usage_are_scored():
    rows = build_rows(_stats(), _games(), [SEASON], config=CONFIG)
    assert set(rows["gsis_id"]) == {"wr1", "rb1"}
    assert set(rows["market"]) == {"receiving_yards", "rushing_yards"}


def test_baseline_never_sees_the_graded_game():
    rows = build_rows(_stats(), _games(), [SEASON], config=CONFIG)
    big_week = rows[(rows["gsis_id"] == "wr1") & (rows["week"] == 6)].iloc[0]
    # Every prior game was 60 yards, so the pregame mean is exactly 60 even
    # though the graded game was 200.
    assert big_week["mu"] == pytest.approx(60.0)
    assert big_week["actual"] == pytest.approx(200.0)
    # The week after, the 200 is history and pulls the mean up.
    after = rows[(rows["gsis_id"] == "wr1") & (rows["week"] == 7)].iloc[0]
    assert after["mu"] > 60.0


def test_every_component_is_reported_and_finite():
    rows = build_rows(_stats(), _games(), [SEASON], config=CONFIG)
    for column in COMPONENTS:
        assert rows[column].notna().all()
        assert (rows[column] > 0).all()


def test_summary_signs_a_helping_factor_negative():
    rows = pd.DataFrame(
        {
            "season": [SEASON] * 4,
            "week": [4, 4, 5, 5],
            "market": ["rushing_yards"] * 4,
            "mu": [50.0, 50.0, 50.0, 50.0],
            "actual": [60.0, 60.0, 60.0, 60.0],
            # Composite nudges toward the actual; usage trend pushes away.
            "composite_factor": [1.1] * 4,
            "game_script_factor": [1.0] * 4,
            "matchup_history_factor": [1.0] * 4,
            "usage_trend_factor": [0.9] * 4,
        }
    )
    report = summarize(rows, config=CONFIG)
    assert report["composite_factor"]["ALL"]["delta_mae"] == pytest.approx(-5.0)
    assert report["usage_trend_factor"]["ALL"]["delta_mae"] == pytest.approx(5.0)
    assert report["game_script_factor"]["ALL"]["delta_mae"] == pytest.approx(0.0)
    assert report["composite_factor"]["rushing_yards"]["n"] == 4
