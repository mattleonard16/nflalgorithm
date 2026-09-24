"""grade_bets counts each bet once, no matter how many books priced it."""

from __future__ import annotations

import pytest

from config import config
from schema_migrations import MigrationManager
from scripts.record_outcomes import grade_bets
from utils.db import execute

GAME_ID = "2026_01_KC_BUF"


@pytest.fixture()
def card_database(tmp_path, monkeypatch) -> str:
    db_path = str(tmp_path / "record-outcomes.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)
    monkeypatch.setattr(config.database, "backend", "sqlite")
    monkeypatch.setattr(config.database, "path", db_path)
    MigrationManager(db_path).run()

    execute(
        "INSERT INTO games (game_id, season, week, home_team, away_team, kickoff_utc, game_date) "
        "VALUES (?, 2026, 1, 'BUF', 'KC', '2026-09-13T17:00:00Z', '2026-09-13')",
        (GAME_ID,),
    )
    execute(
        "INSERT INTO nfl_roster_players (season, gsis_id, player_id, player_name, team, "
        "position, updated_at) VALUES (2026, 'G1', 'p1', 'Player One', 'KC', 'WR', "
        "'2026-09-01T00:00:00Z')"
    )
    execute(
        "INSERT INTO player_stats_enhanced (player_id, gsis_id, season, week, name, team, "
        "position, receiving_yards) VALUES ('p1', 'G1', 2026, 1, 'Player One', 'KC', 'WR', 80)"
    )
    return db_path


def _card_row(sportsbook: str, line: float, edge: float, side: str = "over") -> None:
    execute(
        """
        INSERT INTO materialized_value_view (
            season, week, player_id, event_id, team, market, sportsbook,
            line, price, side, mu, sigma, p_win, edge_percentage, expected_roi,
            kelly_fraction, stake, generated_at
        ) VALUES (2026, 1, 'p1', ?, 'KC', 'receiving_yards', ?, ?, -110, ?,
                  72.0, 10.0, 0.6, ?, 0.1, 0.02, 5.0, '2026-09-10T12:00:00Z')
        """,
        (GAME_ID, sportsbook, line, side, edge),
    )


def test_a_bet_priced_at_several_books_is_graded_once_at_the_best_one(card_database) -> None:
    _card_row("DraftKings", 64.5, 0.08)
    _card_row("FanDuel", 63.5, 0.12)
    _card_row("Bovada", 65.5, 0.06)
    _card_row("FanDuel", 90.5, 0.09, side="under")

    outcomes = grade_bets(2026, 1)

    graded = sorted((o["side"], o["sportsbook"], o["line"]) for o in outcomes)
    assert graded == [("over", "FanDuel", 63.5), ("under", "FanDuel", 90.5)]
