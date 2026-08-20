"""Behavior of the weekly context adjustment factors.

Two things carry most of the risk here and are tested hardest: the *sign* of the
game-script factor (a flipped sign quietly makes every projection worse in a way
that looks like model noise) and the *bounds* (these multipliers ride on top of a
fitted model, so an unbounded one is a projection-destroying bug).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from utils.context_factors import (
    COMPONENT_GAME_SCRIPT,
    COMPONENT_MATCHUP_HISTORY,
    COMPONENT_USAGE_TREND,
    DEFAULT_PARAMS,
    MARKET_OPPORTUNITY_COLUMN,
    OUTPUT_COLUMNS,
    ContextParams,
    compute_context_factors,
    context_factor_lookup,
    context_factors_for_week,
    game_script_factor,
    matchup_history_factor,
    usage_trend_factor,
)

# One game, home favored by 7 in a 45-point game. Implied totals: 26 / 19.
GAMES = pd.DataFrame(
    [
        {
            "season": 2025,
            "week": 10,
            "home_team": "KC",
            "away_team": "DEN",
            "spread_line": 7.0,
            "total_line": 45.0,
        }
    ]
)


def _history_row(**overrides):
    row = {
        "player_id": "p1",
        "name": "Test Player",
        "season": 2025,
        "week": 1,
        "team": "KC",
        "position": "WR",
        "receiving_yards": 60.0,
        "targets": 6.0,
        "rushing_yards": 0.0,
        "rushing_attempts": 0.0,
        "passing_yards": 0.0,
        "passing_attempts": 0.0,
    }
    row.update(overrides)
    return row


def _player(player_id="p1", team="KC", position="WR"):
    return pd.DataFrame([{"player_id": player_id, "team": team, "position": position}])


class TestGameScriptSign:
    """A favored offense runs; a trailing one throws. Getting this backwards is
    the failure mode that looks like noise instead of a bug."""

    def test_favored_team_helps_its_running_back(self):
        factor, neutral = game_script_factor(
            market="rushing_yards",
            position="RB",
            team_favored_by=7.0,
            implied_team_total=26.0,
        )
        assert neutral is False
        assert factor > 1.0

    def test_underdog_helps_its_receivers(self):
        factor, neutral = game_script_factor(
            market="receiving_yards",
            position="WR",
            team_favored_by=-7.0,
            implied_team_total=19.0,
        )
        assert neutral is False
        assert factor > 1.0

    def test_favored_team_hurts_its_receivers(self):
        factor, _ = game_script_factor(
            market="receiving_yards",
            position="WR",
            team_favored_by=7.0,
            implied_team_total=26.0,
        )
        assert factor < 1.0

    def test_underdog_hurts_its_running_back(self):
        factor, _ = game_script_factor(
            market="rushing_yards",
            position="RB",
            team_favored_by=-7.0,
            implied_team_total=19.0,
        )
        assert factor < 1.0

    def test_underdog_quarterback_gains_passing_volume(self):
        factor, _ = game_script_factor(
            market="passing_yards",
            position="QB",
            team_favored_by=-7.0,
            implied_team_total=19.0,
        )
        assert factor > 1.0

    def test_quarterback_rushing_carries_no_spread_signal(self):
        """Scrambles rise when trailing and kneel-downs when leading; the sign
        is genuinely ambiguous, so only the total moves it."""
        favored, _ = game_script_factor(
            market="rushing_yards", position="QB", team_favored_by=10.0, implied_team_total=22.5
        )
        trailing, _ = game_script_factor(
            market="rushing_yards", position="QB", team_favored_by=-10.0, implied_team_total=22.5
        )
        assert favored == pytest.approx(1.0)
        assert trailing == pytest.approx(1.0)

    def test_high_total_lifts_everyone(self):
        factor, _ = game_script_factor(
            market="receiving_yards",
            position="WR",
            team_favored_by=0.0,
            implied_team_total=30.0,
        )
        assert factor > 1.0

    def test_missing_both_inputs_is_neutral(self):
        factor, neutral = game_script_factor(
            market="receiving_yards",
            position="WR",
            team_favored_by=None,
            implied_team_total=None,
        )
        assert factor == 1.0
        assert neutral is True

    def test_spread_alone_still_counts_as_signal(self):
        factor, neutral = game_script_factor(
            market="receiving_yards",
            position="WR",
            team_favored_by=-7.0,
            implied_team_total=None,
        )
        assert neutral is False
        assert factor > 1.0

    @pytest.mark.parametrize("spread", [-60.0, 60.0])
    @pytest.mark.parametrize("total", [0.5, 400.0])
    @pytest.mark.parametrize("market", sorted(MARKET_OPPORTUNITY_COLUMN))
    def test_absurd_market_numbers_stay_inside_the_bounds(self, spread, total, market):
        factor, _ = game_script_factor(
            market=market,
            position="RB",
            team_favored_by=spread,
            implied_team_total=total,
        )
        lo, hi = DEFAULT_PARAMS.script_bounds
        assert lo <= factor <= hi


class TestMatchupHistory:
    def test_no_prior_meetings_is_exactly_neutral(self):
        history = pd.DataFrame(
            [
                _history_row(week=1, opponent="LV", receiving_yards=60.0),
                _history_row(week=2, opponent="LAC", receiving_yards=80.0),
            ]
        )
        factor, neutral, n = matchup_history_factor(
            history, opponent="DEN", stat_column="receiving_yards"
        )
        assert factor == 1.0
        assert neutral is True
        assert n == 0

    def test_unknown_opponent_is_exactly_neutral(self):
        history = pd.DataFrame([_history_row(week=1, opponent="DEN")])
        factor, neutral, _ = matchup_history_factor(
            history, opponent=None, stat_column="receiving_yards"
        )
        assert factor == 1.0
        assert neutral is True

    def test_one_big_game_is_shrunk_hard_toward_neutral(self):
        rows = [_history_row(week=w, opponent="LV", receiving_yards=50.0) for w in range(1, 7)]
        rows.append(_history_row(week=7, opponent="DEN", receiving_yards=150.0))
        factor, neutral, n = matchup_history_factor(
            pd.DataFrame(rows), opponent="DEN", stat_column="receiving_yards"
        )
        assert neutral is False
        assert n == 1
        # Raw signal is ~2.3x; one meeting must not buy anything like that.
        assert 1.0 < factor <= DEFAULT_PARAMS.matchup_bounds[1]
        assert factor < 1.06

    def test_more_meetings_move_further_from_neutral(self):
        base = [_history_row(week=w, opponent="LV", receiving_yards=50.0) for w in range(1, 7)]
        one = base + [_history_row(week=7, opponent="DEN", receiving_yards=60.0)]
        many = base + [
            _history_row(season=2024, week=w, opponent="DEN", receiving_yards=60.0)
            for w in range(1, 6)
        ]
        one_factor, _, _ = matchup_history_factor(
            pd.DataFrame(one), opponent="DEN", stat_column="receiving_yards"
        )
        many_factor, _, n = matchup_history_factor(
            pd.DataFrame(many), opponent="DEN", stat_column="receiving_yards"
        )
        assert n == 5
        assert many_factor > one_factor > 1.0

    def test_player_with_no_baseline_in_this_stat_is_neutral(self):
        """A receiver's rushing line is a rounding error, not a matchup read."""
        rows = [
            _history_row(week=w, opponent="DEN", rushing_yards=0.0, rushing_attempts=0.0)
            for w in range(1, 6)
        ]
        factor, neutral, _ = matchup_history_factor(
            pd.DataFrame(rows), opponent="DEN", stat_column="rushing_yards"
        )
        assert factor == 1.0
        assert neutral is True

    def test_catastrophic_history_stays_inside_the_bounds(self):
        rows = [_history_row(week=w, opponent="LV", receiving_yards=100.0) for w in range(1, 10)]
        rows += [
            _history_row(season=2024, week=w, opponent="DEN", receiving_yards=0.0)
            for w in range(1, 12)
        ]
        factor, _, _ = matchup_history_factor(
            pd.DataFrame(rows), opponent="DEN", stat_column="receiving_yards"
        )
        lo, hi = DEFAULT_PARAMS.matchup_bounds
        assert lo <= factor <= hi


class TestUsageTrend:
    def _frame(self, targets):
        return pd.DataFrame(
            [_history_row(week=i + 1, targets=value) for i, value in enumerate(targets)]
        )

    def test_rising_role_lifts_the_factor(self):
        factor, neutral, recent, baseline = usage_trend_factor(
            self._frame([4, 4, 4, 4, 4, 9, 9, 9]), opportunity_column="targets"
        )
        assert neutral is False
        assert (recent, baseline) == (3, 5)
        assert factor > 1.0

    def test_shrinking_role_lowers_the_factor(self):
        factor, _, _, _ = usage_trend_factor(
            self._frame([9, 9, 9, 9, 9, 4, 4, 4]), opportunity_column="targets"
        )
        assert factor < 1.0

    def test_flat_role_is_neutral_valued_but_not_flagged_neutral(self):
        factor, neutral, _, _ = usage_trend_factor(
            self._frame([6, 6, 6, 6, 6, 6, 6, 6]), opportunity_column="targets"
        )
        assert factor == pytest.approx(1.0)
        assert neutral is False

    def test_too_little_history_is_exactly_neutral(self):
        factor, neutral, recent, baseline = usage_trend_factor(
            self._frame([5, 6]), opportunity_column="targets"
        )
        assert factor == 1.0
        assert neutral is True
        assert baseline == 0

    def test_tiny_sample_is_shrunk_closer_to_neutral_than_a_full_one(self):
        small, _, _, _ = usage_trend_factor(
            self._frame([5, 6, 6, 6]), opportunity_column="targets"
        )
        large, _, _, _ = usage_trend_factor(
            self._frame([5, 5, 5, 5, 5, 6, 6, 6]), opportunity_column="targets"
        )
        assert 1.0 < small < large

    def test_gaps_in_the_schedule_count_played_games_not_weeks(self):
        """A bye plus an injury absence must not read as a collapsed role."""
        played = pd.DataFrame(
            [
                _history_row(week=1, targets=6.0),
                _history_row(week=2, targets=6.0),
                _history_row(week=3, targets=6.0),
                _history_row(week=4, targets=6.0),
                _history_row(week=5, targets=6.0),
                # Weeks 6-9 missed entirely; no rows exist for them.
                _history_row(week=10, targets=6.0),
                _history_row(week=11, targets=6.0),
                _history_row(week=12, targets=6.0),
            ]
        )
        factor, neutral, recent, baseline = usage_trend_factor(
            played, opportunity_column="targets"
        )
        assert (recent, baseline) == (3, 5)
        assert neutral is False
        assert factor == pytest.approx(1.0)

    def test_zero_baseline_opportunity_is_neutral_not_infinite(self):
        factor, neutral, _, _ = usage_trend_factor(
            self._frame([0, 0, 0, 0, 0, 8, 8, 8]), opportunity_column="targets"
        )
        assert factor == 1.0
        assert neutral is True

    def test_explosive_growth_stays_inside_the_bounds(self):
        factor, _, _, _ = usage_trend_factor(
            self._frame([1, 1, 1, 1, 1, 40, 40, 40]), opportunity_column="targets"
        )
        lo, hi = DEFAULT_PARAMS.trend_bounds
        assert lo <= factor <= hi


class TestComputeContextFactors:
    def test_returns_every_output_column(self):
        frame = compute_context_factors(
            season=2025,
            week=10,
            market="receiving_yards",
            players=_player(),
            games=GAMES,
            history=pd.DataFrame([_history_row()]),
        )
        assert list(frame.columns) == list(OUTPUT_COLUMNS)
        assert len(frame) == 1

    def test_missing_schedule_row_degrades_to_all_neutral_with_a_flag(self):
        frame = compute_context_factors(
            season=2025,
            week=10,
            market="receiving_yards",
            players=_player(team="XXX"),
            games=GAMES,
            history=pd.DataFrame(),
        )
        row = frame.iloc[0]
        assert bool(row["has_game_row"]) is False
        assert row["game_script_factor"] == 1.0
        assert row["matchup_history_factor"] == 1.0
        assert row["usage_trend_factor"] == 1.0
        assert row["composite_factor"] == 1.0
        flags = set(str(row["neutral_components"]).split(","))
        assert flags == {
            COMPONENT_GAME_SCRIPT,
            COMPONENT_MATCHUP_HISTORY,
            COMPONENT_USAGE_TREND,
        }

    def test_missing_schedule_row_still_lets_usage_trend_speak(self):
        """Deliberate: role trajectory does not depend on the schedule, so a
        missing game row neutralizes the opponent-derived components only."""
        history = pd.DataFrame(
            [_history_row(team="XXX", week=i + 1, targets=v) for i, v in enumerate([4] * 5 + [9] * 3)]
        )
        frame = compute_context_factors(
            season=2025,
            week=10,
            market="receiving_yards",
            players=_player(team="XXX"),
            games=GAMES,
            history=history,
        )
        row = frame.iloc[0]
        assert bool(row["has_game_row"]) is False
        assert row["usage_trend_factor"] > 1.0
        assert COMPONENT_USAGE_TREND not in str(row["neutral_components"])

    def test_empty_players_returns_an_empty_shaped_frame(self):
        frame = compute_context_factors(
            season=2025,
            week=10,
            market="receiving_yards",
            players=pd.DataFrame(columns=["player_id", "team", "position"]),
            games=GAMES,
            history=pd.DataFrame(),
        )
        assert frame.empty
        assert list(frame.columns) == list(OUTPUT_COLUMNS)

    def test_history_at_or_after_the_target_week_is_ignored(self):
        """Pregame safety net: week 10's own result must never inform week 10."""
        leaky = pd.DataFrame(
            [_history_row(week=10, targets=40.0, receiving_yards=250.0)]
        )
        frame = compute_context_factors(
            season=2025,
            week=10,
            market="receiving_yards",
            players=_player(),
            games=GAMES,
            history=leaky,
        )
        assert int(frame.iloc[0]["context_n_games"]) == 0

    def test_opponent_is_derived_from_the_schedule(self):
        frame = compute_context_factors(
            season=2025,
            week=10,
            market="receiving_yards",
            players=_player(team="DEN"),
            games=GAMES,
            history=pd.DataFrame(),
        )
        assert frame.iloc[0]["opponent"] == "KC"

    def test_composite_is_the_product_of_the_components(self):
        history = pd.DataFrame(
            [_history_row(week=i + 1, targets=v, receiving_yards=v * 10.0)
             for i, v in enumerate([4] * 5 + [9] * 3)]
        )
        row = compute_context_factors(
            season=2025,
            week=10,
            market="receiving_yards",
            players=_player(team="DEN"),
            games=GAMES,
            history=history,
        ).iloc[0]
        expected = (
            row["game_script_factor"]
            * row["matchup_history_factor"]
            * row["usage_trend_factor"]
        )
        assert row["composite_factor"] == pytest.approx(expected)

    def test_composite_respects_its_bounds_under_extreme_inputs(self):
        wild_games = pd.DataFrame(
            [
                {
                    "season": 2025,
                    "week": 10,
                    "home_team": "KC",
                    "away_team": "DEN",
                    "spread_line": -45.0,
                    "total_line": 120.0,
                }
            ]
        )
        history = pd.DataFrame(
            [_history_row(week=i + 1, targets=v, receiving_yards=v * 20.0)
             for i, v in enumerate([1] * 5 + [30] * 3)]
        )
        row = compute_context_factors(
            season=2025,
            week=10,
            market="receiving_yards",
            players=_player(),
            games=wild_games,
            history=history,
        ).iloc[0]
        lo, hi = DEFAULT_PARAMS.composite_bounds
        assert lo <= row["composite_factor"] <= hi

    def test_null_market_numbers_never_produce_nan(self):
        blank_games = pd.DataFrame(
            [
                {
                    "season": 2025,
                    "week": 10,
                    "home_team": "KC",
                    "away_team": "DEN",
                    "spread_line": None,
                    "total_line": None,
                }
            ]
        )
        history = pd.DataFrame(
            [_history_row(week=1, targets=float("nan"), receiving_yards=float("nan"))]
        )
        frame = compute_context_factors(
            season=2025,
            week=10,
            market="receiving_yards",
            players=_player(),
            games=blank_games,
            history=history,
        )
        factors = frame[
            [
                "game_script_factor",
                "matchup_history_factor",
                "usage_trend_factor",
                "composite_factor",
            ]
        ]
        assert factors.notna().all().all()
        assert (factors == 1.0).all().all()

    def test_schedule_without_market_columns_degrades_rather_than_crashes(self):
        bare = pd.DataFrame(
            [{"season": 2025, "week": 10, "home_team": "KC", "away_team": "DEN"}]
        )
        row = compute_context_factors(
            season=2025,
            week=10,
            market="receiving_yards",
            players=_player(),
            games=bare,
            history=pd.DataFrame(),
        ).iloc[0]
        assert bool(row["has_game_row"]) is True
        assert row["opponent"] == "DEN"
        assert row["game_script_factor"] == 1.0
        assert COMPONENT_GAME_SCRIPT in str(row["neutral_components"])

    def test_inputs_are_not_mutated(self):
        games = GAMES.copy()
        history = pd.DataFrame([_history_row()])
        before_games = games.copy()
        before_history = history.copy()
        compute_context_factors(
            season=2025,
            week=10,
            market="receiving_yards",
            players=_player(),
            games=games,
            history=history,
        )
        pd.testing.assert_frame_equal(games, before_games)
        pd.testing.assert_frame_equal(history, before_history)


class TestWeekOneFallsBackToThePriorSeason:
    def test_trend_and_matchup_use_last_seasons_games(self):
        games = pd.DataFrame(
            [
                {
                    "season": 2024,
                    "week": w,
                    "home_team": "KC",
                    "away_team": "DEN",
                    "spread_line": 3.0,
                    "total_line": 44.0,
                }
                for w in range(1, 9)
            ]
            + [
                {
                    "season": 2025,
                    "week": 1,
                    "home_team": "KC",
                    "away_team": "DEN",
                    "spread_line": 7.0,
                    "total_line": 45.0,
                }
            ]
        )
        history = pd.DataFrame(
            [
                _history_row(season=2024, week=w, targets=v, receiving_yards=v * 10.0)
                for w, v in zip(range(1, 9), [4, 4, 4, 4, 4, 9, 9, 9])
            ]
        )
        row = compute_context_factors(
            season=2025,
            week=1,
            market="receiving_yards",
            players=_player(),
            games=games,
            history=history,
        ).iloc[0]
        # Every 2024 game was against DEN, which is also the week-1 opponent.
        assert int(row["context_n_games"]) == 8
        assert int(row["matchup_n_games"]) == 8
        assert row["usage_trend_factor"] > 1.0
        assert COMPONENT_USAGE_TREND not in str(row["neutral_components"])
        assert row["game_script_factor"] != 1.0


class TestFailLoudAtTheBoundary:
    def test_unsupported_market_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported market"):
            compute_context_factors(
                season=2025,
                week=10,
                market="player_anytime_td",
                players=_player(),
                games=GAMES,
                history=pd.DataFrame(),
            )

    def test_missing_player_columns_are_rejected(self):
        with pytest.raises(ValueError, match="players is missing required columns"):
            compute_context_factors(
                season=2025,
                week=10,
                market="receiving_yards",
                players=pd.DataFrame([{"player_id": "p1"}]),
                games=GAMES,
                history=pd.DataFrame(),
            )

    def test_missing_history_columns_are_rejected(self):
        with pytest.raises(ValueError, match="history is missing required columns"):
            compute_context_factors(
                season=2025,
                week=10,
                market="receiving_yards",
                players=_player(),
                games=GAMES,
                history=pd.DataFrame([{"player_id": "p1", "season": 2025, "week": 1}]),
            )

    def test_duplicate_players_are_rejected(self):
        players = pd.concat([_player(), _player()], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate player_id"):
            compute_context_factors(
                season=2025,
                week=10,
                market="receiving_yards",
                players=players,
                games=GAMES,
                history=pd.DataFrame(),
            )

    def test_a_team_scheduled_twice_in_one_week_is_rejected(self):
        doubled = pd.concat(
            [
                GAMES,
                pd.DataFrame(
                    [
                        {
                            "season": 2025,
                            "week": 10,
                            "home_team": "KC",
                            "away_team": "LV",
                            "spread_line": 3.0,
                            "total_line": 44.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        with pytest.raises(ValueError, match="more than one game"):
            compute_context_factors(
                season=2025,
                week=10,
                market="receiving_yards",
                players=_player(),
                games=doubled,
                history=pd.DataFrame(),
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"script_bounds": (1.05, 1.10)},
            {"composite_bounds": (0.5, 0.9)},
            {"baseline_team_total": 0.0},
            {"matchup_shrinkage_k": 0.0},
            {"trend_recent_games": 0},
            {"history_seasons": 0},
            {"trend_raw_ratio_bounds": (1.5, 2.0)},
        ],
    )
    def test_nonsense_params_are_rejected_at_construction(self, kwargs):
        with pytest.raises(ValueError):
            ContextParams(**kwargs)


def _seed_database(db_path: Path) -> None:
    """Minimal `games` and `player_stats_enhanced`, shaped like the real schema."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                spread_line REAL,
                total_line REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE player_stats_enhanced (
                player_id TEXT NOT NULL,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                name TEXT NOT NULL,
                team TEXT NOT NULL,
                position TEXT NOT NULL,
                receiving_yards REAL NOT NULL DEFAULT 0,
                targets REAL NOT NULL DEFAULT 0,
                rushing_yards REAL NOT NULL DEFAULT 0,
                rushing_attempts REAL NOT NULL DEFAULT 0,
                passing_yards REAL NOT NULL DEFAULT 0,
                passing_attempts REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (player_id, season, week)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE weekly_projections (
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                player_id TEXT NOT NULL,
                team TEXT NOT NULL,
                opponent TEXT NOT NULL,
                market TEXT NOT NULL,
                mu REAL NOT NULL,
                sigma REAL NOT NULL,
                PRIMARY KEY (season, week, player_id, market)
            )
            """
        )
        # Plain INSERT, not INSERT OR IGNORE: a constraint failure here means
        # the fixture is wrong, and seeding nothing would make every later
        # assertion pass vacuously.
        for week in range(1, 11):
            conn.execute(
                "INSERT INTO games "
                "(game_id, season, week, home_team, away_team, spread_line, total_line) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"2025_{week:02d}_DEN_KC", 2025, week, "KC", "DEN", 7.0, 45.0),
            )
        for week, targets in zip(range(1, 10), [4, 4, 4, 4, 4, 4, 9, 9, 9]):
            conn.execute(
                "INSERT INTO player_stats_enhanced "
                "(player_id, season, week, name, team, position, receiving_yards, targets) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("p1", 2025, week, "Rising Receiver", "KC", "WR", targets * 10.0, float(targets)),
            )
        conn.execute(
            "INSERT INTO weekly_projections "
            "(season, week, player_id, team, opponent, market, mu, sigma) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (2025, 10, "p1", "KC", "DEN", "receiving_yards", 70.0, 25.0),
        )
        conn.commit()


class TestDatabaseBackedEntryPoints:
    def test_reads_players_games_and_history_from_the_database(self, tmp_path: Path):
        db_path = tmp_path / "context.db"
        _seed_database(db_path)
        with sqlite3.connect(db_path) as conn:
            frame = context_factors_for_week(2025, 10, "receiving_yards", conn=conn)

        assert len(frame) == 1
        row = frame.iloc[0]
        assert row["player_id"] == "p1"
        assert row["name"] == "Rising Receiver"
        # Position came from the player's most recent completed game, since
        # weekly_projections does not carry one.
        assert row["position"] == "WR"
        assert row["opponent"] == "DEN"
        assert int(row["context_n_games"]) == 9
        assert int(row["matchup_n_games"]) == 9
        assert row["usage_trend_factor"] > 1.0
        lo, hi = DEFAULT_PARAMS.composite_bounds
        assert lo <= row["composite_factor"] <= hi

    def test_supplied_players_override_the_projection_roster(self, tmp_path: Path):
        db_path = tmp_path / "context.db"
        _seed_database(db_path)
        with sqlite3.connect(db_path) as conn:
            frame = context_factors_for_week(
                2025,
                10,
                "receiving_yards",
                players=_player(player_id="p1", team="DEN", position="WR"),
                conn=conn,
            )
        assert frame.iloc[0]["opponent"] == "KC"

    def test_lookup_returns_a_defaultable_player_map(self, tmp_path: Path):
        db_path = tmp_path / "context.db"
        _seed_database(db_path)
        with sqlite3.connect(db_path) as conn:
            lookup = context_factor_lookup(2025, 10, "receiving_yards", conn=conn)
        assert set(lookup) == {"p1"}
        assert isinstance(lookup["p1"], float)
        # The contract the mu site depends on: an unknown player is 1.0, never 0.
        assert lookup.get("nobody", 1.0) == 1.0

    def test_history_query_excludes_the_target_week(self, tmp_path: Path):
        db_path = tmp_path / "context.db"
        _seed_database(db_path)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO player_stats_enhanced "
                "(player_id, season, week, name, team, position, receiving_yards, targets) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("p1", 2025, 10, "Rising Receiver", "KC", "WR", 999.0, 40.0),
            )
            conn.commit()
            frame = context_factors_for_week(2025, 10, "receiving_yards", conn=conn)
        assert int(frame.iloc[0]["context_n_games"]) == 9
