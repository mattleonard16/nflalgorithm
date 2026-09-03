"""Unit tests for empirical game script computation and snapshot expected script (U2)."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.ingest_real_nfl_data import (
    _compute_actual_game_script,
    build_player_context_snapshots,
    transform_to_enhanced_stats,
)


def _sample_weekly_stats() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "position": "QB",
                "player_id": "00-0036389",
                "player_name": "Jalen Hurts",
                "player_display_name": "Jalen Hurts",
                "team": "PHI",
                "opponent_team": "DAL",
                "season": 2025,
                "week": 1,
                "rushing_yards": 40.0,
                "rushing_attempts": 8.0,
                "passing_yards": 240.0,
                "attempts": 28.0,
                "receiving_yards": 0.0,
                "receptions": 0.0,
                "targets": 0.0,
                "passing_tds": 2.0,
                "rushing_tds": 1.0,
                "receiving_tds": 0.0,
            },
            {
                "position": "QB",
                "player_id": "00-0033077",
                "player_name": "Dak Prescott",
                "player_display_name": "Dak Prescott",
                "team": "DAL",
                "opponent_team": "PHI",
                "season": 2025,
                "week": 1,
                "rushing_yards": 10.0,
                "rushing_attempts": 2.0,
                "passing_yards": 280.0,
                "attempts": 38.0,
                "receiving_yards": 0.0,
                "receptions": 0.0,
                "targets": 0.0,
                "passing_tds": 1.0,
                "rushing_tds": 0.0,
                "receiving_tds": 0.0,
            },
        ]
    )


class TestComputeActualGameScript:
    def test_schedule_final_scores_assign_margins(self):
        df = pd.DataFrame(
            [
                {"player_id": "P1", "season": 2025, "week": 1, "team": "PHI"},
                {"player_id": "P2", "season": 2025, "week": 1, "team": "DAL"},
            ]
        )
        schedule = pd.DataFrame(
            [
                {
                    "season": 2025,
                    "week": 1,
                    "home_team": "PHI",
                    "away_team": "DAL",
                    "home_score": 28.0,
                    "away_score": 14.0,
                }
            ]
        )
        result = _compute_actual_game_script(df, schedule=schedule)
        by_team = result.set_index("team")

        # Home team won 28-14: margin is (28 - 14) / 2 = +7.0
        assert by_team.loc["PHI", "game_script"] == pytest.approx(7.0)
        # Away team lost: (14 - 28) / 2 = -7.0
        assert by_team.loc["DAL", "game_script"] == pytest.approx(-7.0)

    def test_tie_game_produces_neutral_script(self):
        df = pd.DataFrame([{"player_id": "P1", "season": 2025, "week": 1, "team": "KC"}])
        schedule = pd.DataFrame(
            [
                {
                    "season": 2025,
                    "week": 1,
                    "home_team": "KC",
                    "away_team": "LAC",
                    "home_score": 20.0,
                    "away_score": 20.0,
                }
            ]
        )
        result = _compute_actual_game_script(df, schedule=schedule)
        assert result.iloc[0]["game_script"] == pytest.approx(0.0)

    def test_pbp_average_score_differential_takes_priority(self):
        df = pd.DataFrame([{"player_id": "P1", "season": 2025, "week": 1, "team": "SF"}])
        pbp = pd.DataFrame(
            [
                {"season": 2025, "week": 1, "posteam": "SF", "score_differential": 3.0},
                {"season": 2025, "week": 1, "posteam": "SF", "score_differential": 7.0},
                {"season": 2025, "week": 1, "posteam": "SF", "score_differential": 11.0},
            ]
        )
        # Final score might be different (e.g. 24-21), but pbp mean is (3+7+11)/3 = 7.0
        schedule = pd.DataFrame(
            [
                {
                    "season": 2025,
                    "week": 1,
                    "home_team": "SF",
                    "away_team": "LAR",
                    "home_score": 24.0,
                    "away_score": 21.0,
                }
            ]
        )
        result = _compute_actual_game_script(df, schedule=schedule, pbp=pbp)
        assert result.iloc[0]["game_script"] == pytest.approx(7.0)

    def test_missing_schedule_falls_back_to_zero(self):
        df = pd.DataFrame([{"player_id": "P1", "season": 2025, "week": 1, "team": "DET"}])
        result = _compute_actual_game_script(df, schedule=None, pbp=None)
        assert result.iloc[0]["game_script"] == 0.0

    def test_transform_integrates_game_script(self):
        weekly = _sample_weekly_stats()
        schedule = pd.DataFrame(
            [
                {
                    "season": 2025,
                    "week": 1,
                    "home_team": "PHI",
                    "away_team": "DAL",
                    "home_score": 28.0,
                    "away_score": 14.0,
                    "gameday": "2025-09-07",
                }
            ]
        )
        result = transform_to_enhanced_stats(weekly, pd.DataFrame(), schedule=schedule)
        by_team = result.set_index("team")
        assert by_team.loc["PHI", "game_script"] == pytest.approx(7.0)
        assert by_team.loc["DAL", "game_script"] == pytest.approx(-7.0)


class TestSnapshotExpectedGameScript:
    def test_favored_and_underdog_expected_script(self):
        rosters = pd.DataFrame(
            [
                {
                    "season": 2025,
                    "week": 1,
                    "gsis_id": "00-001",
                    "player_name": "Player One",
                    "team": "KC",
                    "position": "QB",
                    "roster_status": "ACT",
                },
                {
                    "season": 2025,
                    "week": 1,
                    "gsis_id": "00-002",
                    "player_name": "Player Two",
                    "team": "LV",
                    "position": "QB",
                    "roster_status": "ACT",
                },
            ]
        )
        schedule = pd.DataFrame(
            [
                {
                    "season": 2025,
                    "week": 1,
                    "home_team": "KC",
                    "away_team": "LV",
                    "spread_line": 6.0,  # KC favored by 6.0
                }
            ]
        )
        snapshots = build_player_context_snapshots(
            rosters,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            target_week=1,
            schedule=schedule,
        )
        by_team = snapshots.set_index("team")
        # KC favored by 6.0: expected_game_script = +3.0
        assert by_team.loc["KC", "expected_game_script"] == pytest.approx(3.0)
        # LV underdog by 6.0: expected_game_script = -3.0
        assert by_team.loc["LV", "expected_game_script"] == pytest.approx(-3.0)

    def test_missing_spread_defaults_to_zero(self):
        rosters = pd.DataFrame(
            [
                {
                    "season": 2025,
                    "week": 1,
                    "gsis_id": "00-001",
                    "player_name": "Player One",
                    "team": "NYG",
                    "position": "QB",
                    "roster_status": "ACT",
                }
            ]
        )
        snapshots = build_player_context_snapshots(
            rosters,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            target_week=1,
            schedule=None,
        )
        assert snapshots.iloc[0]["expected_game_script"] == 0.0
