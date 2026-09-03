"""Behavior of schedule game-context extraction."""

from __future__ import annotations

import pandas as pd
import pytest

from utils.game_context import (
    CONTEXT_COLUMNS,
    GAME_CONTEXT_COLUMNS,
    GAME_CONTEXT_DEFAULTS,
    attach_game_context_to_player_frame,
    extract_game_context,
    home_favored_by,
    implied_team_totals,
    is_indoor_roof,
)

SCHEDULE = pd.DataFrame(
    [
        {
            "game_id": "2025_01_DAL_PHI",
            "spread_line": 8.5,
            "total_line": 47.5,
            "temp": 75.0,
            "wind": 11.0,
            "roof": "outdoors",
            "surface": "grass",
            "div_game": 1,
        },
        {
            "game_id": "2025_01_KC_LAC",
            "spread_line": -3.0,
            "total_line": 47.5,
            "temp": None,
            "wind": None,
            "roof": "dome",
            "surface": "",
            "div_game": 0,
        },
        {
            "game_id": "2025_01_TB_ATL",
            "spread_line": -1.5,
            "total_line": 44.0,
            "temp": None,
            "wind": None,
            "roof": "closed",
            "surface": "fieldturf",
            "div_game": 1,
        },
    ]
)


class TestIsIndoorRoof:
    @pytest.mark.parametrize("roof", ["dome", "closed", "DOME", " Closed "])
    def test_indoor_values(self, roof):
        assert is_indoor_roof(roof) is True

    @pytest.mark.parametrize("roof", ["outdoors", "open", "Outdoors"])
    def test_outdoor_values(self, roof):
        assert is_indoor_roof(roof) is False

    @pytest.mark.parametrize("roof", [None, "", "retractable-ish", 3, float("nan")])
    def test_unknown_values_are_none_not_guessed(self, roof):
        # Guessing here silently decides whether weather features apply.
        assert is_indoor_roof(roof) is None


class TestExtractGameContext:
    def test_returns_every_context_column(self):
        assert list(extract_game_context(SCHEDULE).columns) == list(CONTEXT_COLUMNS)

    def test_preserves_row_alignment(self):
        context = extract_game_context(SCHEDULE)
        assert len(context) == len(SCHEDULE)
        assert list(context.index) == list(SCHEDULE.index)

    def test_carries_market_numbers_through(self):
        context = extract_game_context(SCHEDULE)
        assert context.loc[0, "spread_line"] == 8.5
        assert context.loc[0, "total_line"] == 47.5

    def test_indoor_games_keep_null_weather_rather_than_imputed(self):
        """Imputing a league-average temperature into a dome teaches the model
        that domes are 60 degrees and windy."""
        context = extract_game_context(SCHEDULE)
        assert pd.isna(context.loc[1, "temp"])
        assert pd.isna(context.loc[1, "wind"])
        assert bool(context.loc[1, "is_indoor"]) is True

    def test_closed_retractable_roof_counts_as_indoor(self):
        assert bool(extract_game_context(SCHEDULE).loc[2, "is_indoor"]) is True

    def test_outdoor_game_keeps_its_weather(self):
        context = extract_game_context(SCHEDULE)
        assert context.loc[0, "temp"] == 75.0
        assert context.loc[0, "wind"] == 11.0
        assert bool(context.loc[0, "is_indoor"]) is False

    def test_div_game_is_normalized_to_a_flag(self):
        context = extract_game_context(SCHEDULE)
        assert context.loc[0, "div_game"] == 1
        assert context.loc[1, "div_game"] == 0

    def test_blank_surface_becomes_null_not_empty_string(self):
        assert extract_game_context(SCHEDULE).loc[1, "surface"] is None

    def test_surface_is_case_normalized(self):
        frame = pd.DataFrame([{"roof": "outdoors", "surface": "  GRASS "}])
        assert extract_game_context(frame).loc[0, "surface"] == "grass"

    def test_absent_column_degrades_to_null_not_an_error(self):
        # Older seasons omit columns; failing the whole ingest over one
        # missing optional feature is worse than a null column.
        frame = pd.DataFrame([{"roof": "outdoors", "spread_line": 3.0}])
        context = extract_game_context(frame)
        assert pd.isna(context.loc[0, "wind"])
        assert context.loc[0, "spread_line"] == 3.0

    def test_unparseable_number_becomes_null(self):
        frame = pd.DataFrame([{"roof": "outdoors", "total_line": "forty"}])
        assert pd.isna(extract_game_context(frame).loc[0, "total_line"])

    def test_empty_schedule_returns_empty_frame_with_columns(self):
        context = extract_game_context(pd.DataFrame())
        assert context.empty
        assert list(context.columns) == list(CONTEXT_COLUMNS)

    def test_none_schedule_is_tolerated(self):
        assert extract_game_context(None).empty


class TestHomeFavoredBy:
    def test_positive_means_the_home_team_is_favored(self):
        # The convention the whole feature depends on; a flipped sign here is
        # a bug a model will learn around rather than surface.
        assert home_favored_by(8.5) == 8.5

    def test_negative_means_the_home_team_is_the_underdog(self):
        assert home_favored_by(-3.0) == -3.0

    def test_missing_is_none(self):
        assert home_favored_by(None) is None
        assert home_favored_by(float("nan")) is None

    def test_unparseable_is_none(self):
        assert home_favored_by("pick-em") is None


class TestImpliedTeamTotals:
    def test_splits_the_total_by_the_spread(self):
        # Home favored by 8.5 in a 47.5-point game: 28.0 / 19.5.
        assert implied_team_totals(8.5, 47.5) == (28.0, 19.5)

    def test_underdog_home_team_gets_the_smaller_share(self):
        home, away = implied_team_totals(-3.0, 47.5)
        assert home < away
        assert (home, away) == (22.25, 25.25)

    def test_pick_em_splits_evenly(self):
        assert implied_team_totals(0.0, 44.0) == (22.0, 22.0)

    def test_halves_always_sum_to_the_total(self):
        home, away = implied_team_totals(6.5, 51.0)
        assert home + away == pytest.approx(51.0)

    def test_missing_either_input_returns_none(self):
        assert implied_team_totals(None, 47.5) is None
        assert implied_team_totals(8.5, None) is None


class TestAttachGameContextToPlayerFrame:
    def test_home_and_away_perspectives(self):
        games = pd.DataFrame(
            [
                {
                    "game_id": "2025_01_DAL_PHI",
                    "season": 2025,
                    "week": 1,
                    "home_team": "PHI",
                    "away_team": "DAL",
                    "spread_line": 6.0,
                    "total_line": 48.0,
                    "temp": 72.0,
                    "wind": 10.0,
                    "roof": "outdoors",
                    "div_game": 1,
                }
            ]
        )
        players = pd.DataFrame(
            [
                {"player_id": "P1", "season": 2025, "week": 1, "team": "PHI"},
                {"player_id": "P2", "season": 2025, "week": 1, "team": "DAL"},
            ]
        )
        result = attach_game_context_to_player_frame(players, games)

        phi = result[result["team"] == "PHI"].iloc[0]
        dal = result[result["team"] == "DAL"].iloc[0]

        assert phi["spread_margin"] == 6.0
        assert phi["implied_team_total"] == 27.0
        assert phi["game_total"] == 48.0
        assert phi["wind_speed"] == 10.0
        assert phi["temperature"] == 72.0
        assert phi["is_indoor"] == 0
        assert phi["div_game"] == 1

        assert dal["spread_margin"] == -6.0
        assert dal["implied_team_total"] == 21.0
        assert dal["game_total"] == 48.0
        assert dal["wind_speed"] == 10.0
        assert dal["temperature"] == 72.0
        assert dal["is_indoor"] == 0
        assert dal["div_game"] == 1

    def test_indoor_dome_imputes_clean_weather(self):
        games = pd.DataFrame(
            [
                {
                    "game_id": "2025_01_DET_MIN",
                    "season": 2025,
                    "week": 1,
                    "home_team": "MIN",
                    "away_team": "DET",
                    "spread_line": -2.5,
                    "total_line": 51.0,
                    "temp": None,
                    "wind": None,
                    "roof": "dome",
                    "div_game": 1,
                }
            ]
        )
        players = pd.DataFrame(
            [{"player_id": "P1", "season": 2025, "week": 1, "team": "MIN"}]
        )
        result = attach_game_context_to_player_frame(players, games)
        row = result.iloc[0]
        assert row["is_indoor"] == 1
        assert row["wind_speed"] == 0.0
        assert row["temperature"] == 70.0
        assert row["spread_margin"] == -2.5
        assert row["implied_team_total"] == pytest.approx(24.25)

    def test_bye_week_defaults(self):
        games = pd.DataFrame(
            [
                {
                    "game_id": "2025_05_KC_DEN",
                    "season": 2025,
                    "week": 5,
                    "home_team": "DEN",
                    "away_team": "KC",
                    "spread_line": -4.0,
                    "total_line": 46.0,
                }
            ]
        )
        players = pd.DataFrame(
            [{"player_id": "P1", "season": 2025, "week": 5, "team": "MIA"}]  # MIA on bye
        )
        result = attach_game_context_to_player_frame(players, games)
        row = result.iloc[0]
        assert row["spread_margin"] == GAME_CONTEXT_DEFAULTS["spread_margin"]
        assert row["implied_team_total"] == GAME_CONTEXT_DEFAULTS["implied_team_total"]
        assert row["game_total"] == GAME_CONTEXT_DEFAULTS["game_total"]
        assert row["wind_speed"] == GAME_CONTEXT_DEFAULTS["wind_speed"]
        assert row["temperature"] == GAME_CONTEXT_DEFAULTS["temperature"]
        assert row["is_indoor"] == GAME_CONTEXT_DEFAULTS["is_indoor"]
        assert row["div_game"] == GAME_CONTEXT_DEFAULTS["div_game"]

    def test_corrupt_data_degrades_gracefully(self):
        games = pd.DataFrame(
            [
                {
                    "game_id": "corrupt",
                    "season": 2025,
                    "week": 1,
                    "home_team": "BUF",
                    "away_team": "NYJ",
                    "spread_line": "invalid",
                    "total_line": "bad",
                    "temp": "none",
                    "wind": "nan",
                    "roof": 12345,
                }
            ]
        )
        players = pd.DataFrame(
            [{"player_id": "P1", "season": 2025, "week": 1, "team": "BUF"}]
        )
        result = attach_game_context_to_player_frame(players, games)
        row = result.iloc[0]
        assert row["spread_margin"] == 0.0
        assert row["implied_team_total"] == 22.5
        assert row["game_total"] == 45.0

