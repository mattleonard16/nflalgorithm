"""Opening/closing line derivation from append-only weekly_odds history.

load_opening_lines picks MIN(as_of) and load_closing_lines picks MAX(as_of)
per (season, week, player_id, market, sportsbook) key. Both compare as_of as
SQLite TEXT — lexical, not chronological — which is correct only while all
writers store consistently-formatted UTC timestamps (they do today).
"""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from config import config
from scripts.backfill_line_accuracy import load_closing_lines, load_opening_lines


@contextmanager
def use_database(db_path: Path):
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


def _init_odds_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS weekly_odds (
            event_id TEXT,
            season INTEGER,
            week INTEGER,
            player_id TEXT,
            market TEXT,
            sportsbook TEXT,
            line REAL,
            price INTEGER,
            as_of TEXT
        )
        """
    )


def _insert_odds(conn, *, season, week, player_id, market, sportsbook, line, price, as_of, event_id="evt1"):
    conn.execute(
        "INSERT INTO weekly_odds (event_id, season, week, player_id, market, sportsbook, line, price, as_of) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (event_id, season, week, player_id, market, sportsbook, line, price, as_of),
    )


def test_load_closing_lines_picks_max_as_of_per_key(tmp_path: Path) -> None:
    db_path = tmp_path / "closing.db"
    with sqlite3.connect(db_path) as conn:
        _init_odds_table(conn)
        _insert_odds(
            conn, season=2024, week=1, player_id="HOU_cj_stroud", market="passing_yards",
            sportsbook="Book", line=250.5, price=-110, as_of="2024-09-01T10:00:00Z",
        )
        _insert_odds(
            conn, season=2024, week=1, player_id="HOU_cj_stroud", market="passing_yards",
            sportsbook="Book", line=255.5, price=-108, as_of="2024-09-01T18:00:00Z",
        )
        _insert_odds(
            conn, season=2024, week=1, player_id="HOU_cj_stroud", market="passing_yards",
            sportsbook="Book", line=248.5, price=-112, as_of="2024-09-01T12:00:00Z",
        )
        conn.commit()

    with use_database(db_path):
        result = load_closing_lines([2024])

    assert len(result) == 1
    row = result.iloc[0]
    assert row["line"] == 255.5
    assert row["as_of"] == "2024-09-01T18:00:00Z"


def test_load_opening_lines_picks_min_as_of_per_key(tmp_path: Path) -> None:
    db_path = tmp_path / "opening.db"
    with sqlite3.connect(db_path) as conn:
        _init_odds_table(conn)
        _insert_odds(
            conn, season=2024, week=1, player_id="HOU_cj_stroud", market="passing_yards",
            sportsbook="Book", line=250.5, price=-110, as_of="2024-09-01T10:00:00Z",
        )
        _insert_odds(
            conn, season=2024, week=1, player_id="HOU_cj_stroud", market="passing_yards",
            sportsbook="Book", line=255.5, price=-108, as_of="2024-09-01T18:00:00Z",
        )
        _insert_odds(
            conn, season=2024, week=1, player_id="HOU_cj_stroud", market="passing_yards",
            sportsbook="Book", line=248.5, price=-112, as_of="2024-09-01T12:00:00Z",
        )
        conn.commit()

    with use_database(db_path):
        result = load_opening_lines([2024])

    assert len(result) == 1
    row = result.iloc[0]
    assert row["open_line"] == 250.5
    assert row["open_as_of"] == "2024-09-01T10:00:00Z"


def test_open_and_close_are_independent_per_sportsbook_key(tmp_path: Path) -> None:
    """Each (player, market, sportsbook) key opens/closes independently."""
    db_path = tmp_path / "per_book.db"
    with sqlite3.connect(db_path) as conn:
        _init_odds_table(conn)
        _insert_odds(
            conn, season=2024, week=1, player_id="KC_skyy_moore", market="receiving_yards",
            sportsbook="BookA", line=40.5, price=-110, as_of="2024-09-01T09:00:00Z",
        )
        _insert_odds(
            conn, season=2024, week=1, player_id="KC_skyy_moore", market="receiving_yards",
            sportsbook="BookA", line=44.5, price=-105, as_of="2024-09-01T17:00:00Z",
        )
        _insert_odds(
            conn, season=2024, week=1, player_id="KC_skyy_moore", market="receiving_yards",
            sportsbook="BookB", line=38.5, price=-115, as_of="2024-09-01T20:00:00Z",
        )
        conn.commit()

    with use_database(db_path):
        closing = load_closing_lines([2024]).set_index("sportsbook")
        opening = load_opening_lines([2024]).set_index("sportsbook")

    assert closing.loc["BookA", "line"] == 44.5
    assert opening.loc["BookA", "open_line"] == 40.5
    # BookB has one snapshot: it is simultaneously its own open and close.
    assert closing.loc["BookB", "line"] == 38.5
    assert opening.loc["BookB", "open_line"] == 38.5


def test_load_closing_lines_scopes_by_requested_seasons(tmp_path: Path) -> None:
    db_path = tmp_path / "seasons.db"
    with sqlite3.connect(db_path) as conn:
        _init_odds_table(conn)
        _insert_odds(
            conn, season=2023, week=1, player_id="X_player", market="rushing_yards",
            sportsbook="Book", line=60.5, price=-110, as_of="2023-09-01T10:00:00Z",
        )
        _insert_odds(
            conn, season=2024, week=1, player_id="X_player", market="rushing_yards",
            sportsbook="Book", line=65.5, price=-110, as_of="2024-09-01T10:00:00Z",
        )
        conn.commit()

    with use_database(db_path):
        result = load_closing_lines([2024])

    assert len(result) == 1
    assert result.iloc[0]["season"] == 2024


def test_load_closing_lines_empty_table_returns_empty_frame(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.db"
    with sqlite3.connect(db_path) as conn:
        _init_odds_table(conn)
        conn.commit()

    with use_database(db_path):
        result = load_closing_lines([2024])

    assert result.empty
