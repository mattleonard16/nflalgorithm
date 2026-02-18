"""End-to-end NBA pipeline integration tests with synthetic data.

Covers:
- Grading NBA bets against actual game log stats
- Multi-market grading (pts, reb, ast, fg3m)
- Data flow integrity between tables
"""

from __future__ import annotations

import pytest

from schema_migrations import MigrationManager
from utils.db import execute, executemany, fetchall, read_dataframe


GAME_DATE = "2026-02-17"
SEASON = 2025
PLAYER_ID = 1234
PLAYER_NAME = "Test Player"
TEAM = "LAL"


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)
    import config as cfg
    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")
    from schema_migrations import MigrationManager
    MigrationManager(db_path).run()
    return db_path


def _seed_game_log(
    player_id: int = PLAYER_ID,
    player_name: str = PLAYER_NAME,
    game_date: str = GAME_DATE,
    pts: int = 30,
    reb: int = 8,
    ast: int = 5,
    fg3m: int = 3,
) -> None:
    execute(
        "INSERT INTO nba_player_game_logs "
        "(player_id, player_name, team_abbreviation, season, game_id, game_date, matchup, wl, min, pts, reb, ast, fg3m, fgm, fga, ftm, fta, stl, blk, tov, plus_minus) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params=(player_id, player_name, TEAM, SEASON, "0022500100", game_date, "LAL vs BOS", "W", 35.0, pts, reb, ast, fg3m, 12, 22, 3, 4, 1, 1, 2, 10.0),
    )


def _seed_value_bet(
    player_id: int = PLAYER_ID,
    player_name: str = PLAYER_NAME,
    game_date: str = GAME_DATE,
    market: str = "pts",
    line: float = 25.5,
    over_price: int = -110,
    edge_percentage: float = 12.0,
    event_id: str = "evt1",
    sportsbook: str = "draftkings",
) -> None:
    execute(
        "INSERT INTO nba_materialized_value_view "
        "(season, game_date, player_id, player_name, team, event_id, market, sportsbook, "
        "line, over_price, under_price, mu, sigma, p_win, edge_percentage, expected_roi, "
        "kelly_fraction, confidence, generated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params=(SEASON, game_date, player_id, player_name, TEAM, event_id, market, sportsbook,
                line, over_price, 110, 28.0, 3.0, 0.65, edge_percentage, 0.10, 0.02, 0.85, "2026-02-17T00:00:00"),
    )


class TestNbaPipelineIntegration:
    def test_grading_end_to_end(self, db):
        """Seed game logs and value bet, grade, verify win result, then save outcomes."""
        _seed_game_log(pts=30)
        _seed_value_bet(market="pts", line=25.5, over_price=-110)

        from scripts.record_nba_outcomes import grade_nba_bets, save_nba_outcomes

        outcomes = grade_nba_bets(GAME_DATE)

        assert len(outcomes) == 1, f"Expected 1 outcome, got {len(outcomes)}"
        outcome = outcomes[0]
        assert outcome["result"] == "win", (
            f"Expected 'win' (30 > 25.5) but got '{outcome['result']}'"
        )
        assert outcome["market"] == "pts"
        assert outcome["actual_result"] == pytest.approx(30.0)
        assert outcome["player_id"] == PLAYER_ID

        save_nba_outcomes(outcomes)

        bet_rows = fetchall("SELECT * FROM nba_bet_outcomes WHERE game_date = ?", params=(GAME_DATE,))
        assert len(bet_rows) == 1, f"Expected 1 row in nba_bet_outcomes, got {len(bet_rows)}"

        perf_rows = fetchall("SELECT * FROM nba_daily_performance WHERE game_date = ?", params=(GAME_DATE,))
        assert len(perf_rows) == 1, f"Expected 1 row in nba_daily_performance, got {len(perf_rows)}"

    def test_multiple_markets_grading(self, db):
        """Seed all 4 markets for one player, verify each market grades against its stat column."""
        _seed_game_log(pts=30, reb=8, ast=5, fg3m=3)

        market_configs = [
            ("pts",  25.5, "evt_pts",  "win"),
            ("reb",  10.5, "evt_reb",  "loss"),
            ("ast",   4.5, "evt_ast",  "win"),
            ("fg3m",  2.5, "evt_fg3m", "win"),
        ]

        for market, line, event_id, _expected_result in market_configs:
            _seed_value_bet(
                market=market,
                line=line,
                event_id=event_id,
                over_price=-110,
            )

        from scripts.record_nba_outcomes import grade_nba_bets

        outcomes = grade_nba_bets(GAME_DATE)

        assert len(outcomes) == 4, f"Expected 4 outcomes (one per market), got {len(outcomes)}"

        outcome_by_market = {o["market"]: o for o in outcomes}

        assert outcome_by_market["pts"]["result"] == "win", (
            "pts: 30 > 25.5 should be win"
        )
        assert outcome_by_market["reb"]["result"] == "loss", (
            "reb: 8 < 10.5 should be loss"
        )
        assert outcome_by_market["ast"]["result"] == "win", (
            "ast: 5 > 4.5 should be win"
        )
        assert outcome_by_market["fg3m"]["result"] == "win", (
            "fg3m: 3 > 2.5 should be win"
        )

        for market, _, _, expected_result in market_configs:
            assert outcome_by_market[market]["result"] == expected_result, (
                f"Market '{market}': expected '{expected_result}', "
                f"got '{outcome_by_market[market]['result']}'"
            )

    def test_data_flow_integrity(self, db):
        """After grading, player_ids in nba_bet_outcomes must match nba_materialized_value_view and game_dates must be consistent."""
        _seed_game_log()
        _seed_value_bet()

        from scripts.record_nba_outcomes import grade_nba_bets, save_nba_outcomes

        outcomes = grade_nba_bets(GAME_DATE)
        save_nba_outcomes(outcomes)

        value_view_ids = {
            row[0]
            for row in fetchall(
                "SELECT player_id FROM nba_materialized_value_view WHERE game_date = ?",
                params=(GAME_DATE,),
            )
        }
        bet_outcome_ids = {
            row[0]
            for row in fetchall(
                "SELECT player_id FROM nba_bet_outcomes WHERE game_date = ?",
                params=(GAME_DATE,),
            )
        }

        assert bet_outcome_ids == value_view_ids, (
            f"player_ids mismatch: value_view={value_view_ids}, bet_outcomes={bet_outcome_ids}"
        )

        bet_dates = {
            row[0]
            for row in fetchall(
                "SELECT DISTINCT game_date FROM nba_bet_outcomes",
            )
        }
        perf_dates = {
            row[0]
            for row in fetchall(
                "SELECT DISTINCT game_date FROM nba_daily_performance",
            )
        }

        assert bet_dates == {GAME_DATE}, (
            f"nba_bet_outcomes contains unexpected dates: {bet_dates}"
        )
        assert perf_dates == {GAME_DATE}, (
            f"nba_daily_performance contains unexpected dates: {perf_dates}"
        )

        assert bet_dates == perf_dates, (
            "game_dates in nba_bet_outcomes and nba_daily_performance must match"
        )
