"""Tests for the weekly research memo generator.

Every test builds its own SQLite file, so nothing here depends on the shape of
the developer's ``nfl_data.db``. Only tracked modules are imported: the memo
never touches the gitignored projection code.

The recurring theme is degradation. A memo generated in August, before a snap
has been played, must still be a valid memo -- every section that has nothing
to say has to say exactly that, and the run has to exit 0.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import pytest

from config import config
from scripts.research_outlook import data_freshness, next_week_game_scripts, usage_trends
from scripts.research_review import grading_recap, projection_accuracy
from scripts.weekly_research import MemoInputError, build_memo, main, memo_paths


@contextmanager
def use_database(db_path: Path) -> Iterator[None]:
    """Point the process-wide database config at one temporary SQLite file."""
    original_path = config.database.path
    original_backend = config.database.backend
    env_backend = os.environ.get("DB_BACKEND")
    env_sqlite_path = os.environ.get("SQLITE_DB_PATH")
    os.environ["DB_BACKEND"] = "sqlite"
    os.environ["SQLITE_DB_PATH"] = str(db_path)
    config.database.backend = "sqlite"
    config.database.path = str(db_path)
    try:
        yield
    finally:
        config.database.path = original_path
        config.database.backend = original_backend
        if env_backend is not None:
            os.environ["DB_BACKEND"] = env_backend
        else:
            os.environ.pop("DB_BACKEND", None)
        if env_sqlite_path is not None:
            os.environ["SQLITE_DB_PATH"] = env_sqlite_path
        else:
            os.environ.pop("SQLITE_DB_PATH", None)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

_SCHEMA = (
    """
    CREATE TABLE player_stats_enhanced (
        player_id TEXT, gsis_id TEXT, season INTEGER, week INTEGER, name TEXT,
        team TEXT, position TEXT, rushing_yards REAL, rushing_attempts REAL,
        receiving_yards REAL, receptions REAL, targets REAL, passing_yards REAL,
        passing_attempts REAL, created_at TEXT, updated_at TEXT
    )
    """,
    """
    CREATE TABLE games (
        game_id TEXT, season INTEGER, week INTEGER, home_team TEXT, away_team TEXT,
        kickoff_utc TEXT, game_date TEXT, spread_line REAL, total_line REAL,
        div_game INTEGER, created_at TEXT
    )
    """,
    """
    CREATE TABLE weekly_projections (
        season INTEGER, week INTEGER, player_id TEXT, team TEXT, market TEXT,
        mu REAL, sigma REAL, model_version TEXT, generated_at TEXT
    )
    """,
    """
    CREATE TABLE weekly_odds (
        event_id TEXT, season INTEGER, week INTEGER, player_id TEXT, market TEXT,
        sportsbook TEXT, line REAL, price INTEGER, as_of TEXT
    )
    """,
    """
    CREATE TABLE bet_outcomes (
        bet_id TEXT, season INTEGER, week INTEGER, player_id TEXT, player_name TEXT,
        market TEXT, sportsbook TEXT, side TEXT, line REAL, price INTEGER,
        actual_result REAL, result TEXT, profit_units REAL, confidence_tier TEXT,
        edge_at_placement REAL, recorded_at TEXT
    )
    """,
    """
    CREATE TABLE weekly_performance (
        season INTEGER, week INTEGER, total_bets INTEGER, wins INTEGER, losses INTEGER,
        pushes INTEGER, profit_units REAL, roi_pct REAL, avg_edge REAL, clv_avg REAL,
        best_bet TEXT, worst_bet TEXT, updated_at TEXT
    )
    """,
    """
    CREATE TABLE clv_weekly (
        bet_id TEXT, close_line REAL, close_price INTEGER, clv_bp REAL, closed_at TEXT
    )
    """,
    """
    CREATE TABLE nfl_roster_players (
        season INTEGER, roster_week INTEGER, player_id TEXT, gsis_id TEXT
    )
    """,
)

# Which opportunity column each position's trend is read from, mirroring
# POSITION_PRIMARY_MARKET in the module under test.
_OPPORTUNITY_COLUMN = {
    "QB": "passing_attempts",
    "RB": "rushing_attempts",
    "WR": "targets",
    "TE": "targets",
}
_STAT_COLUMN = {
    "QB": "passing_yards",
    "RB": "rushing_yards",
    "WR": "receiving_yards",
    "TE": "receiving_yards",
}


def _new_db(tmp_path: Path, name: str = "research.db") -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / name)
    for statement in _SCHEMA:
        conn.execute(statement)
    conn.commit()
    return conn


def _insert(conn: sqlite3.Connection, table: str, **values: Any) -> None:
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    # Plain INSERT, not INSERT OR IGNORE: a constraint failure here means the
    # fixture is wrong, and swallowing it would make the assertions vacuous.
    conn.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", tuple(values.values()))


def _seed_player_weeks(
    conn: sqlite3.Connection,
    *,
    player_id: str,
    name: str,
    position: str,
    team: str = "SEA",
    season: int = 2025,
    opportunities: Sequence[float],
    first_week: int = 1,
    gsis_id: str | None = None,
) -> None:
    """One played game per entry in ``opportunities``, starting at ``first_week``."""
    opportunity_column = _OPPORTUNITY_COLUMN[position]
    stat_column = _STAT_COLUMN[position]
    for offset, value in enumerate(opportunities):
        _insert(
            conn,
            "player_stats_enhanced",
            player_id=player_id,
            gsis_id=gsis_id,
            season=season,
            week=first_week + offset,
            name=name,
            team=team,
            position=position,
            **{opportunity_column: float(value), stat_column: float(value) * 8.0},
            created_at="2025-11-01T00:00:00+00:00",
            updated_at="2025-11-01T00:00:00+00:00",
        )


def _seed_game(
    conn: sqlite3.Connection,
    *,
    season: int,
    week: int,
    home: str,
    away: str,
    spread: float | None = None,
    total: float | None = None,
    div: int = 0,
) -> None:
    _insert(
        conn,
        "games",
        game_id=f"{season}_{week:02d}_{away}_{home}",
        season=season,
        week=week,
        home_team=home,
        away_team=away,
        kickoff_utc=f"2025-11-{10 + week:02d}T18:00:00+00:00",
        game_date=f"2025-11-{10 + week:02d}",
        spread_line=spread,
        total_line=total,
        div_game=div,
        created_at="2025-08-01T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# section 1: grading recap
# ---------------------------------------------------------------------------


class TestGradingRecap:
    def test_says_no_data_when_both_grading_tables_are_empty(self, tmp_path: Path) -> None:
        conn = _new_db(tmp_path)
        markdown, payload = grading_recap(2025, 10, conn=conn)
        assert payload["status"] == "no_data"
        assert "bet_outcomes and weekly_performance are both empty" in payload["note"]
        assert "_no data:" in markdown

    def test_says_no_data_when_the_grading_tables_do_not_exist(self, tmp_path: Path) -> None:
        """A database that never ran the grading migration is not a crash."""
        conn = _new_db(tmp_path)
        conn.execute("DROP TABLE bet_outcomes")
        conn.execute("DROP TABLE weekly_performance")
        conn.commit()
        _markdown, payload = grading_recap(2025, 10, conn=conn)
        assert payload["status"] == "no_data"

    def test_summarizes_the_week_from_weekly_performance(self, tmp_path: Path) -> None:
        conn = _new_db(tmp_path)
        _insert(
            conn,
            "weekly_performance",
            season=2025,
            week=10,
            total_bets=3,
            wins=2,
            losses=1,
            pushes=0,
            profit_units=0.91,
            roi_pct=30.3,
            avg_edge=0.12,
            clv_avg=45.0,
            best_bet="Alpha Receiver over 60.5",
            worst_bet="Gamma Back over 40.5",
            updated_at="2025-11-12T12:00:00+00:00",
        )
        for index, (name, result, units) in enumerate(
            [("Alpha Receiver", "win", 0.91), ("Beta End", "win", 0.87), ("Gamma Back", "loss", -1.0)]
        ):
            _insert(
                conn,
                "bet_outcomes",
                bet_id=f"bet-{index}",
                season=2025,
                week=10,
                player_id=f"SEA_p{index}",
                player_name=name,
                market="receiving_yards",
                sportsbook="DraftKings",
                side="over",
                line=60.5,
                price=-110,
                actual_result=70.0,
                result=result,
                profit_units=units,
                confidence_tier="high",
                edge_at_placement=0.12,
                recorded_at="2025-11-12T12:00:00+00:00",
            )
        conn.commit()

        markdown, payload = grading_recap(2025, 10, conn=conn)
        assert payload["status"] == "ok"
        assert payload["summary"]["wins"] == 2
        assert payload["summary"]["roi_pct"] == pytest.approx(30.3)
        assert payload["result_counts"] == {"win": 2, "loss": 1}
        assert payload["best_bets"][0]["player_name"] == "Alpha Receiver"
        assert payload["worst_bets"][0]["player_name"] == "Gamma Back"
        assert "Alpha Receiver" in markdown
        assert payload["notes"] == []

    def test_flags_an_aggregate_that_disagrees_with_the_per_bet_table(
        self, tmp_path: Path
    ) -> None:
        """A stale weekly_performance row is the failure this catches."""
        conn = _new_db(tmp_path)
        _insert(
            conn,
            "weekly_performance",
            season=2025,
            week=10,
            total_bets=1,
            wins=1,
            losses=0,
            pushes=0,
            profit_units=0.91,
            roi_pct=91.0,
            avg_edge=0.1,
            clv_avg=0.0,
            best_bet=None,
            worst_bet=None,
            updated_at="2025-11-12T12:00:00+00:00",
        )
        for index in range(3):
            _insert(
                conn,
                "bet_outcomes",
                bet_id=f"bet-{index}",
                season=2025,
                week=10,
                player_id=f"SEA_p{index}",
                player_name=f"Player {index}",
                market="receiving_yards",
                sportsbook="DraftKings",
                side="over",
                line=50.5,
                price=-110,
                actual_result=55.0,
                result="win",
                profit_units=0.91,
                confidence_tier="high",
                edge_at_placement=0.1,
                recorded_at="2025-11-12T12:00:00+00:00",
            )
        conn.commit()

        _markdown, payload = grading_recap(2025, 10, conn=conn)
        assert any("looks stale" in note for note in payload["notes"])

    def test_falls_back_to_bet_outcomes_when_the_aggregate_is_missing(
        self, tmp_path: Path
    ) -> None:
        conn = _new_db(tmp_path)
        _insert(
            conn,
            "bet_outcomes",
            bet_id="bet-0",
            season=2025,
            week=10,
            player_id="SEA_p0",
            player_name="Alpha Receiver",
            market="receiving_yards",
            sportsbook="DraftKings",
            side="over",
            line=60.5,
            price=-110,
            actual_result=70.0,
            result="win",
            profit_units=0.91,
            confidence_tier="high",
            edge_at_placement=0.12,
            recorded_at="2025-11-12T12:00:00+00:00",
        )
        conn.commit()
        markdown, payload = grading_recap(2025, 10, conn=conn)
        assert payload["status"] == "ok"
        assert payload["summary"] == {}
        assert payload["result_counts"] == {"win": 1}
        assert "derived from bet_outcomes" in markdown

    def test_reports_clv_when_clv_weekly_has_rows(self, tmp_path: Path) -> None:
        conn = _new_db(tmp_path)
        for index, clv_bp in enumerate([120.0, -30.0]):
            _insert(
                conn,
                "bet_outcomes",
                bet_id=f"bet-{index}",
                season=2025,
                week=10,
                player_id=f"SEA_p{index}",
                player_name=f"Player {index}",
                market="receiving_yards",
                sportsbook="DraftKings",
                side="over",
                line=50.5,
                price=-110,
                actual_result=55.0,
                result="win",
                profit_units=0.91,
                confidence_tier="high",
                edge_at_placement=0.1,
                recorded_at="2025-11-12T12:00:00+00:00",
            )
            _insert(
                conn,
                "clv_weekly",
                bet_id=f"bet-{index}",
                close_line=52.5,
                close_price=-120,
                clv_bp=clv_bp,
                closed_at="2025-11-13T17:55:00+00:00",
            )
        conn.commit()

        _markdown, payload = grading_recap(2025, 10, conn=conn)
        assert payload["clv"]["status"] == "ok"
        assert payload["clv"]["bets_with_clv"] == 2
        assert payload["clv"]["mean_clv_bp"] == pytest.approx(45.0)
        assert payload["clv"]["beat_close_count"] == 1

    def test_notes_bets_that_have_not_settled(self, tmp_path: Path) -> None:
        conn = _new_db(tmp_path)
        _insert(
            conn,
            "bet_outcomes",
            bet_id="bet-0",
            season=2025,
            week=10,
            player_id="SEA_p0",
            player_name="Alpha Receiver",
            market="receiving_yards",
            sportsbook="DraftKings",
            side="over",
            line=60.5,
            price=-110,
            actual_result=None,
            result=None,
            profit_units=None,
            confidence_tier="high",
            edge_at_placement=0.12,
            recorded_at="2025-11-12T12:00:00+00:00",
        )
        conn.commit()
        _markdown, payload = grading_recap(2025, 10, conn=conn)
        assert payload["result_counts"] == {"ungraded": 1}
        assert any("null profit_units" in note for note in payload["notes"])
        assert "best_bets" not in payload


# ---------------------------------------------------------------------------
# section 2: projection accuracy
# ---------------------------------------------------------------------------


class TestProjectionAccuracy:
    def test_says_no_data_without_projections(self, tmp_path: Path) -> None:
        conn = _new_db(tmp_path)
        _markdown, payload = projection_accuracy(2025, 10, conn=conn)
        assert payload["status"] == "no_data"
        assert "weekly_projections has no rows" in payload["note"]

    def test_says_no_data_when_actuals_have_not_landed(self, tmp_path: Path) -> None:
        conn = _new_db(tmp_path)
        _insert(
            conn,
            "weekly_projections",
            season=2025,
            week=10,
            player_id="SEA_a_receiver",
            team="SEA",
            market="receiving_yards",
            mu=62.0,
            sigma=18.0,
            model_version="v1",
            generated_at="2025-11-12T12:00:00+00:00",
        )
        conn.commit()
        _markdown, payload = projection_accuracy(2025, 10, conn=conn)
        assert payload["status"] == "no_data"
        assert "no actuals" in payload["note"]

    def test_computes_per_position_mae_against_the_gate_ceilings(self, tmp_path: Path) -> None:
        conn = _new_db(tmp_path)
        # WR projected 10 yards high every time; TE projected 20 high, which is
        # over the TE ceiling of 9.0.
        for index in range(3):
            _insert(
                conn,
                "player_stats_enhanced",
                player_id=f"SEA_wr{index}",
                season=2025,
                week=10,
                name=f"Receiver {index}",
                team="SEA",
                position="WR",
                receiving_yards=50.0,
                targets=7.0,
                created_at="2025-11-14T00:00:00+00:00",
                updated_at="2025-11-14T00:00:00+00:00",
            )
            _insert(
                conn,
                "weekly_projections",
                season=2025,
                week=10,
                player_id=f"SEA_wr{index}",
                team="SEA",
                market="receiving_yards",
                mu=60.0,
                sigma=18.0,
                model_version="v1",
                generated_at="2025-11-12T12:00:00+00:00",
            )
        _insert(
            conn,
            "player_stats_enhanced",
            player_id="SEA_te0",
            season=2025,
            week=10,
            name="Tight End",
            team="SEA",
            position="TE",
            receiving_yards=30.0,
            targets=5.0,
            created_at="2025-11-14T00:00:00+00:00",
            updated_at="2025-11-14T00:00:00+00:00",
        )
        _insert(
            conn,
            "weekly_projections",
            season=2025,
            week=10,
            player_id="SEA_te0",
            team="SEA",
            market="receiving_yards",
            mu=50.0,
            sigma=12.0,
            model_version="v1",
            generated_at="2025-11-12T12:00:00+00:00",
        )
        conn.commit()

        markdown, payload = projection_accuracy(2025, 10, conn=conn)
        assert payload["status"] == "ok"
        assert payload["scored_rows"] == 4
        assert payload["unmatched_rows"] == 0
        by_position = {row["position"]: row for row in payload["by_position"]}
        assert by_position["WR"]["mae"] == pytest.approx(10.0)
        assert by_position["WR"]["over_threshold"] is False
        assert by_position["TE"]["mae"] == pytest.approx(20.0)
        assert by_position["TE"]["over_threshold"] is True
        # Four projections is nowhere near the gate's minimum sample.
        assert all(row["below_min_sample"] for row in payload["by_position"])
        assert "OVER CEILING" in markdown
        # Positive bias means the model projected too high, which it did.
        assert payload["overall_bias"] > 0

    def test_counts_projections_that_never_matched_an_actual(self, tmp_path: Path) -> None:
        conn = _new_db(tmp_path)
        _insert(
            conn,
            "player_stats_enhanced",
            player_id="SEA_wr0",
            season=2025,
            week=10,
            name="Receiver 0",
            team="SEA",
            position="WR",
            receiving_yards=50.0,
            targets=7.0,
            created_at="2025-11-14T00:00:00+00:00",
            updated_at="2025-11-14T00:00:00+00:00",
        )
        for player_id in ("SEA_wr0", "SEA_wr_never_played"):
            _insert(
                conn,
                "weekly_projections",
                season=2025,
                week=10,
                player_id=player_id,
                team="SEA",
                market="receiving_yards",
                mu=60.0,
                sigma=18.0,
                model_version="v1",
                generated_at="2025-11-12T12:00:00+00:00",
            )
        conn.commit()
        _markdown, payload = projection_accuracy(2025, 10, conn=conn)
        assert payload["projection_rows"] == 2
        assert payload["scored_rows"] == 1
        assert payload["unmatched_rows"] == 1

    def test_bridges_the_two_player_id_namespaces_through_gsis(self, tmp_path: Path) -> None:
        """Projections key on roster ids; stats key on an older abbreviated id."""
        conn = _new_db(tmp_path)
        _insert(
            conn,
            "player_stats_enhanced",
            player_id="SEA_a_receiver",
            gsis_id="00-0011111",
            season=2025,
            week=10,
            name="Alpha Receiver",
            team="SEA",
            position="WR",
            receiving_yards=50.0,
            targets=7.0,
            created_at="2025-11-14T00:00:00+00:00",
            updated_at="2025-11-14T00:00:00+00:00",
        )
        _insert(
            conn,
            "nfl_roster_players",
            season=2025,
            roster_week=10,
            player_id="SEA_alpha_receiver",
            gsis_id="00-0011111",
        )
        _insert(
            conn,
            "weekly_projections",
            season=2025,
            week=10,
            player_id="SEA_alpha_receiver",
            team="SEA",
            market="receiving_yards",
            mu=60.0,
            sigma=18.0,
            model_version="v1",
            generated_at="2025-11-12T12:00:00+00:00",
        )
        conn.commit()

        _markdown, payload = projection_accuracy(2025, 10, conn=conn)
        assert payload["scored_rows"] == 1, "the gsis bridge should reconcile the two id forms"
        assert payload["by_position"][0]["position"] == "WR"
        assert payload["overall_mae"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# section 3: usage trends
# ---------------------------------------------------------------------------


class TestUsageTrends:
    def test_says_no_data_when_there_is_no_history(self, tmp_path: Path) -> None:
        conn = _new_db(tmp_path)
        _markdown, payload = usage_trends(2025, 10, conn=conn)
        assert payload["status"] == "no_data"
        assert "no rows at or before 2025 W10" in payload["note"]

    def test_ranks_risers_above_fallers(self, tmp_path: Path) -> None:
        conn = _new_db(tmp_path)
        # Eight played weeks: the last three are the recent window, the five
        # before them the baseline.
        _seed_player_weeks(
            conn,
            player_id="SEA_riser",
            name="Rising Receiver",
            position="WR",
            opportunities=[3, 3, 3, 3, 3, 9, 9, 9],
        )
        _seed_player_weeks(
            conn,
            player_id="SEA_faller",
            name="Falling Receiver",
            position="WR",
            opportunities=[10, 10, 10, 10, 10, 3, 3, 3],
        )
        _seed_player_weeks(
            conn,
            player_id="SEA_steady",
            name="Steady Receiver",
            position="WR",
            opportunities=[6, 6, 6, 6, 6, 6, 6, 6],
        )
        conn.commit()

        markdown, payload = usage_trends(2025, 8, conn=conn)
        assert payload["status"] == "ok"
        assert payload["into_week"] == 9
        riser_ids = [row["player_id"] for row in payload["risers"]]
        faller_ids = [row["player_id"] for row in payload["fallers"]]
        assert riser_ids == ["SEA_riser"]
        assert faller_ids == ["SEA_faller"]
        # A flat usage line is neither, and never appears in either list.
        assert "SEA_steady" not in riser_ids + faller_ids
        assert payload["risers"][0]["recent_opportunities"] == pytest.approx(9.0)
        assert payload["risers"][0]["baseline_opportunities"] == pytest.approx(3.0)
        assert payload["risers"][0]["recent_games"] == 3
        assert payload["risers"][0]["baseline_games"] == 5
        assert "Rising Receiver" in markdown
        assert "Falling Receiver" in markdown

    def test_ties_break_on_volume_then_player_id(self, tmp_path: Path) -> None:
        """Both players saturate the +10% bound, so the tiebreak decides."""
        conn = _new_db(tmp_path)
        _seed_player_weeks(
            conn,
            player_id="SEA_small",
            name="Small Role",
            position="WR",
            opportunities=[2, 2, 2, 2, 2, 6, 6, 6],
        )
        _seed_player_weeks(
            conn,
            player_id="SEA_big",
            name="Big Role",
            position="WR",
            opportunities=[5, 5, 5, 5, 5, 15, 15, 15],
        )
        conn.commit()

        _markdown, payload = usage_trends(2025, 8, conn=conn)
        factors = {row["player_id"]: row["trend_factor"] for row in payload["risers"]}
        assert factors["SEA_big"] == factors["SEA_small"], "both should hit the bound"
        assert [row["player_id"] for row in payload["risers"]] == ["SEA_big", "SEA_small"]

    def test_excludes_players_who_have_not_played_recently(self, tmp_path: Path) -> None:
        conn = _new_db(tmp_path)
        # Last game was week 5; the memo is for week 8, so he is out, not falling.
        _seed_player_weeks(
            conn,
            player_id="SEA_injured",
            name="Injured Receiver",
            position="WR",
            opportunities=[10, 10, 10, 10, 2],
            first_week=1,
        )
        _seed_player_weeks(
            conn,
            player_id="SEA_active",
            name="Active Receiver",
            position="WR",
            opportunities=[10, 10, 10, 10, 10, 3, 3, 3],
        )
        conn.commit()

        _markdown, payload = usage_trends(2025, 8, conn=conn)
        listed = {row["player_id"] for row in payload["risers"] + payload["fallers"]}
        assert listed == {"SEA_active"}

    def test_reads_each_position_from_its_own_opportunity_column(self, tmp_path: Path) -> None:
        conn = _new_db(tmp_path)
        _seed_player_weeks(
            conn,
            player_id="SEA_qb",
            name="Rising Passer",
            position="QB",
            opportunities=[10, 10, 10, 10, 10, 35, 35, 35],
        )
        _seed_player_weeks(
            conn,
            player_id="SEA_rb",
            name="Rising Runner",
            position="RB",
            opportunities=[4, 4, 4, 4, 4, 14, 14, 14],
        )
        conn.commit()

        _markdown, payload = usage_trends(2025, 8, conn=conn)
        metrics = {row["player_id"]: row["opportunity_column"] for row in payload["risers"]}
        assert metrics == {"SEA_qb": "passing_attempts", "SEA_rb": "rushing_attempts"}

    def test_never_reads_the_week_being_projected(self, tmp_path: Path) -> None:
        """Week 9 usage must not inform the trend going into week 9."""
        conn = _new_db(tmp_path)
        _seed_player_weeks(
            conn,
            player_id="SEA_wr",
            name="Some Receiver",
            position="WR",
            opportunities=[3, 3, 3, 3, 3, 9, 9, 9, 40],
        )
        conn.commit()
        _markdown, payload = usage_trends(2025, 8, conn=conn)
        assert payload["risers"][0]["recent_opportunities"] == pytest.approx(9.0)
        assert payload["risers"][0]["last_played_week"] == 8

    def test_limit_caps_each_list(self, tmp_path: Path) -> None:
        conn = _new_db(tmp_path)
        for index in range(5):
            _seed_player_weeks(
                conn,
                player_id=f"SEA_riser{index}",
                name=f"Riser {index}",
                position="WR",
                opportunities=[3, 3, 3, 3, 3, 8 + index, 8 + index, 8 + index],
            )
        conn.commit()
        _markdown, payload = usage_trends(2025, 8, conn=conn, limit=2)
        assert len(payload["risers"]) == 2
        assert payload["ranked_players"] == 5


# ---------------------------------------------------------------------------
# section 4: next week game scripts
# ---------------------------------------------------------------------------


class TestNextWeekGameScripts:
    def test_rolls_over_to_the_following_week(self, tmp_path: Path) -> None:
        conn = _new_db(tmp_path)
        _seed_game(conn, season=2025, week=10, home="SEA", away="ARI", spread=3.0, total=44.0)
        _seed_game(conn, season=2025, week=11, home="LAR", away="SF", spread=-2.5, total=48.5)
        conn.commit()
        _markdown, payload = next_week_game_scripts(2025, 10, conn=conn)
        assert payload["week"] == 11
        assert payload["game_count"] == 1
        assert payload["games"][0]["matchup"] == "SF @ LAR"

    def test_says_no_data_at_the_end_of_the_season(self, tmp_path: Path) -> None:
        """Week 18 with nothing scheduled after it is the rollover edge."""
        conn = _new_db(tmp_path)
        _seed_game(conn, season=2025, week=18, home="SEA", away="ARI", spread=3.0, total=44.0)
        conn.commit()
        markdown, payload = next_week_game_scripts(2025, 18, conn=conn)
        assert payload["status"] == "no_data"
        assert "no rows for 2025 W19" in payload["note"]
        assert "_no data:" in markdown

    def test_names_the_extremes_and_divisional_games(self, tmp_path: Path) -> None:
        conn = _new_db(tmp_path)
        _seed_game(conn, season=2025, week=11, home="NE", away="NYJ", spread=12.5, total=43.5, div=1)
        _seed_game(conn, season=2025, week=11, home="LAR", away="SEA", spread=-3.0, total=49.5)
        _seed_game(conn, season=2025, week=11, home="CLE", away="BAL", spread=-7.5, total=37.5, div=1)
        conn.commit()

        markdown, payload = next_week_game_scripts(2025, 10, conn=conn)
        assert payload["biggest_spread"]["matchup"] == "NYJ @ NE"
        assert payload["biggest_spread"]["favorite"] == "NE"
        assert payload["biggest_spread"]["favored_by"] == pytest.approx(12.5)
        assert payload["highest_total"]["matchup"] == "SEA @ LAR"
        assert payload["lowest_total"]["matchup"] == "BAL @ CLE"
        assert payload["divisional_count"] == 2
        # The road favorite is named correctly, not flipped by the home-side
        # spread convention.
        cleveland = next(g for g in payload["games"] if g["home_team"] == "CLE")
        assert cleveland["favorite"] == "BAL"
        assert cleveland["implied_home_total"] == pytest.approx(15.0)
        assert cleveland["implied_away_total"] == pytest.approx(22.5)
        assert "Biggest spread" in markdown

    def test_survives_a_week_whose_lines_are_not_published(self, tmp_path: Path) -> None:
        conn = _new_db(tmp_path)
        _seed_game(conn, season=2026, week=1, home="SEA", away="ARI", spread=None, total=None)
        conn.commit()
        markdown, payload = next_week_game_scripts(2025, 22, conn=conn)
        # 2025 W23 does not exist; a 2026 row must not be picked up for it.
        assert payload["status"] == "no_data"
        assert "_no data:" in markdown

    def test_reports_an_unpriced_week_without_crashing(self, tmp_path: Path) -> None:
        conn = _new_db(tmp_path)
        _seed_game(conn, season=2026, week=2, home="SEA", away="ARI", spread=None, total=None)
        conn.commit()
        markdown, payload = next_week_game_scripts(2026, 1, conn=conn)
        assert payload["status"] == "ok"
        assert payload["priced_game_count"] == 0
        assert payload["biggest_spread"] is None
        assert "nothing here is script-informed" in markdown


# ---------------------------------------------------------------------------
# section 5: data freshness
# ---------------------------------------------------------------------------


class TestDataFreshness:
    def test_flags_every_empty_feed(self, tmp_path: Path) -> None:
        conn = _new_db(tmp_path)
        markdown, payload = data_freshness(2025, 10, conn=conn)
        assert payload["status"] == "ok"
        assert len(payload["empty_feeds"]) == 3
        assert all(feed["flag"] == "EMPTY" for feed in payload["feeds"])
        assert "**EMPTY**" in markdown

    def test_counts_rows_and_finds_the_newest_timestamp(self, tmp_path: Path) -> None:
        conn = _new_db(tmp_path)
        _seed_player_weeks(
            conn, player_id="SEA_wr", name="Receiver", position="WR", opportunities=[5, 6]
        )
        _seed_game(conn, season=2025, week=11, home="SEA", away="ARI", spread=3.0, total=44.0)
        conn.commit()
        _markdown, payload = data_freshness(2025, 10, conn=conn)
        feeds = {feed["table"]: feed for feed in payload["feeds"]}
        assert feeds["player_stats_enhanced"]["rows"] == 2
        assert feeds["games"]["rows"] == 1
        assert feeds["weekly_odds"]["rows"] == 0
        assert payload["empty_feeds"] == ["weekly_odds (2025 W11)"]

    def test_newest_timestamp_is_chronological_not_lexical(self, tmp_path: Path) -> None:
        """SQLite stores these as TEXT, where MAX() would pick the wrong row."""
        # 09:00+09:00 is 00:00 UTC, three hours EARLIER than 04:00+00:00, but it
        # sorts later as a string.
        conn = _new_db(tmp_path)
        for index, as_of in enumerate(
            ("2025-11-13T04:00:00+00:00", "2025-11-13T09:00:00+09:00")
        ):
            _insert(
                conn,
                "weekly_odds",
                event_id="2025_11_ARI_SEA",
                season=2025,
                week=11,
                player_id=f"SEA_p{index}",
                market="receiving_yards",
                sportsbook="DraftKings",
                line=50.5,
                price=-110,
                as_of=as_of,
            )
        conn.commit()
        _markdown, payload = data_freshness(2025, 10, conn=conn)
        odds = next(feed for feed in payload["feeds"] if feed["table"] == "weekly_odds")
        assert odds["rows"] == 2
        assert odds["newest"].startswith("2025-11-13T04:00:00")

    def test_counts_unparseable_timestamps(self, tmp_path: Path) -> None:
        conn = _new_db(tmp_path)
        for index, as_of in enumerate(("2025-11-13T04:00:00+00:00", "not-a-timestamp")):
            _insert(
                conn,
                "weekly_odds",
                event_id="2025_11_ARI_SEA",
                season=2025,
                week=11,
                player_id=f"SEA_p{index}",
                market="receiving_yards",
                sportsbook="DraftKings",
                line=50.5,
                price=-110,
                as_of=as_of,
            )
        conn.commit()
        _markdown, payload = data_freshness(2025, 10, conn=conn)
        odds = next(feed for feed in payload["feeds"] if feed["table"] == "weekly_odds")
        assert odds["unparseable_timestamps"] == 1
        assert odds["newest"].startswith("2025-11-13T04:00:00")


# ---------------------------------------------------------------------------
# assembly and CLI
# ---------------------------------------------------------------------------


class TestBuildMemo:
    def test_an_empty_database_still_produces_all_five_sections(self, tmp_path: Path) -> None:
        """The preseason case: nothing has happened, and the memo says so."""
        conn = _new_db(tmp_path)
        markdown, payload = build_memo(2026, 1, conn=conn)
        assert set(payload["sections"]) == {
            "grading_recap",
            "projection_accuracy",
            "usage_trends",
            "next_week_game_scripts",
            "data_freshness",
        }
        no_data = [
            name
            for name, section in payload["sections"].items()
            if section["status"] == "no_data"
        ]
        assert sorted(no_data) == [
            "grading_recap",
            "next_week_game_scripts",
            "projection_accuracy",
            "usage_trends",
        ]
        for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5."):
            assert heading in markdown
        assert payload["next_week"] == 2

    def test_fails_loud_when_a_core_table_is_missing(self, tmp_path: Path) -> None:
        conn = _new_db(tmp_path)
        conn.execute("DROP TABLE games")
        conn.commit()
        with pytest.raises(MemoInputError, match="games"):
            build_memo(2025, 10, conn=conn)


class TestCommandLine:
    def test_writes_both_the_markdown_and_the_json(self, tmp_path: Path) -> None:
        db_path = tmp_path / "cli.db"
        conn = sqlite3.connect(db_path)
        for statement in _SCHEMA:
            conn.execute(statement)
        _seed_player_weeks(
            conn,
            player_id="SEA_riser",
            name="Rising Receiver",
            position="WR",
            opportunities=[3, 3, 3, 3, 3, 9, 9, 9],
        )
        _seed_game(conn, season=2025, week=9, home="SEA", away="ARI", spread=3.0, total=44.0)
        conn.commit()
        conn.close()

        output_dir = tmp_path / "reports"
        with use_database(db_path):
            exit_code = main(
                ["--season", "2025", "--week", "8", "--output-dir", str(output_dir)]
            )
        assert exit_code == 0

        markdown_path, json_path = memo_paths(output_dir, 2025, 8)
        assert markdown_path.name == "weekly_research_2025_W08.md"
        assert markdown_path.is_file()
        assert json_path.is_file()

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert payload["season"] == 2025
        assert payload["week"] == 8
        assert payload["sections"]["usage_trends"]["risers"][0]["name"] == "Rising Receiver"
        assert markdown_path.read_text(encoding="utf-8").startswith(
            "# Weekly Research Memo - 2025 Week 08"
        )

    def test_prints_only_the_written_path(self, tmp_path: Path, capsys) -> None:
        db_path = tmp_path / "quiet.db"
        conn = sqlite3.connect(db_path)
        for statement in _SCHEMA:
            conn.execute(statement)
        conn.commit()
        conn.close()

        output_dir = tmp_path / "reports"
        with use_database(db_path):
            main(["--season", "2026", "--week", "1", "--output-dir", str(output_dir)])
        stdout = capsys.readouterr().out.strip().splitlines()
        assert stdout == [str(output_dir / "weekly_research_2026_W01.md")]

    @pytest.mark.parametrize("week", ["0", "23", "-1"])
    def test_rejects_a_week_outside_the_season(self, week: str, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["--season", "2025", "--week", week, "--output-dir", str(tmp_path)])
        assert excinfo.value.code == 2

    def test_rejects_a_season_that_predates_the_data(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["--season", "1900", "--week", "1", "--output-dir", str(tmp_path)])
        assert excinfo.value.code == 2

    def test_rejects_an_unparseable_week(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["--season", "2025", "--week", "ten", "--output-dir", str(tmp_path)])
        assert excinfo.value.code == 2
