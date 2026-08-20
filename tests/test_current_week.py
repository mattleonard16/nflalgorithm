"""Upcoming-week resolution from the schedule.

The rule under test: the current week is the ``(season, week)`` of the earliest
game that has not kicked off yet. The reason it is worth testing at all is that
the naive implementation — ``MIN(kickoff_utc)`` compared as SQLite TEXT — gives
a different answer whenever two rows carry different UTC offsets, and gives it
silently.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from utils.current_week import as_utc, next_week_from_schedule, resolve_current_week

WEEK1_KICKOFF = "2026-09-10T00:20:00+00:00"
WEEK1_SUNDAY = "2026-09-13T17:00:00+00:00"
WEEK2_KICKOFF = "2026-09-17T00:15:00+00:00"


def _schedule(*rows: tuple[str, int, int, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "home_team": "SEA",
                "away_team": "NE",
                "kickoff_utc": kickoff,
            }
            for game_id, season, week, kickoff in rows
        ]
    )


def _seed_games(db_path: Path, frame: pd.DataFrame) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            kickoff_utc TEXT
        )
        """)
    conn.executemany(
        "INSERT INTO games (game_id, season, week, home_team, away_team, kickoff_utc) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        list(
            frame[
                ["game_id", "season", "week", "home_team", "away_team", "kickoff_utc"]
            ].itertuples(index=False, name=None)
        ),
    )
    conn.commit()
    return conn


def test_picks_week_of_earliest_future_kickoff() -> None:
    schedule = _schedule(
        ("2026_01_NE_SEA", 2026, 1, WEEK1_KICKOFF),
        ("2026_02_NE_SEA", 2026, 2, WEEK2_KICKOFF),
    )
    upcoming = next_week_from_schedule(schedule, now="2026-08-19T12:00:00+00:00")
    assert (upcoming.season, upcoming.week) == (2026, 1)
    assert upcoming.game_id == "2026_01_NE_SEA"


def test_week_already_underway_still_resolves_to_that_week() -> None:
    # Thursday night is played; Sunday is not. The week is not over.
    schedule = _schedule(
        ("2026_01_NE_SEA", 2026, 1, WEEK1_KICKOFF),
        ("2026_01_CHI_CAR", 2026, 1, WEEK1_SUNDAY),
        ("2026_02_NE_SEA", 2026, 2, WEEK2_KICKOFF),
    )
    upcoming = next_week_from_schedule(schedule, now="2026-09-11T12:00:00+00:00")
    assert (upcoming.season, upcoming.week) == (2026, 1)


def test_rolls_forward_once_every_game_of_the_week_has_started() -> None:
    schedule = _schedule(
        ("2026_01_NE_SEA", 2026, 1, WEEK1_KICKOFF),
        ("2026_01_CHI_CAR", 2026, 1, WEEK1_SUNDAY),
        ("2026_02_NE_SEA", 2026, 2, WEEK2_KICKOFF),
    )
    upcoming = next_week_from_schedule(schedule, now="2026-09-13T20:00:00+00:00")
    assert (upcoming.season, upcoming.week) == (2026, 2)


def test_offsets_are_compared_chronologically_not_lexically() -> None:
    # "2026-09-13T12:30:00-05:00" is 17:30 UTC, later than "2026-09-13T17:00:00+00:00",
    # but it sorts FIRST as text. A TEXT MIN would answer week 2 here.
    schedule = _schedule(
        ("2026_01_NE_SEA", 2026, 1, "2026-09-13T17:00:00+00:00"),
        ("2026_02_NE_SEA", 2026, 2, "2026-09-13T12:30:00-05:00"),
    )
    assert min(schedule["kickoff_utc"]) == "2026-09-13T12:30:00-05:00"
    upcoming = next_week_from_schedule(schedule, now="2026-09-13T10:00:00+00:00")
    assert (upcoming.season, upcoming.week) == (2026, 1)


def test_unparseable_kickoffs_are_skipped_not_fatal() -> None:
    schedule = _schedule(
        ("2026_01_BAD", 2026, 1, "not-a-timestamp"),
        ("2026_02_NE_SEA", 2026, 2, WEEK2_KICKOFF),
    )
    upcoming = next_week_from_schedule(schedule, now="2026-08-19T12:00:00+00:00")
    assert (upcoming.season, upcoming.week) == (2026, 2)


def test_all_kickoffs_unparseable_fails_loud() -> None:
    schedule = _schedule(("2026_01_BAD", 2026, 1, "not-a-timestamp"))
    with pytest.raises(ValueError, match="no games.kickoff_utc value parsed"):
        next_week_from_schedule(schedule, now="2026-08-19T12:00:00+00:00")


def test_season_fully_played_fails_loud() -> None:
    schedule = _schedule(("2026_01_NE_SEA", 2026, 1, WEEK1_KICKOFF))
    with pytest.raises(ValueError, match="every scheduled game has already kicked off"):
        next_week_from_schedule(schedule, now="2027-02-01T00:00:00+00:00")


def test_empty_schedule_fails_loud() -> None:
    with pytest.raises(ValueError, match="schedule frame is empty"):
        next_week_from_schedule(pd.DataFrame(), now="2026-08-19T12:00:00+00:00")


def test_missing_columns_fail_loud() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        next_week_from_schedule(pd.DataFrame({"kickoff_utc": [WEEK1_KICKOFF]}))


def test_naive_now_is_read_as_utc_not_local() -> None:
    naive = as_utc(datetime(2026, 8, 19, 12, 0, 0))
    assert naive == pd.Timestamp("2026-08-19T12:00:00+00:00")


def test_as_utc_converts_an_aware_clock() -> None:
    aware = as_utc(datetime(2026, 8, 19, 7, 0, 0, tzinfo=timezone.utc))
    assert aware == pd.Timestamp("2026-08-19T07:00:00+00:00")


def test_as_utc_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        as_utc("halfway through the third quarter")


def test_resolve_current_week_reads_the_database(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    schedule = _schedule(
        ("2026_01_NE_SEA", 2026, 1, WEEK1_KICKOFF),
        ("2026_02_NE_SEA", 2026, 2, WEEK2_KICKOFF),
    )
    conn = _seed_games(tmp_path / "schedule.db", schedule)
    try:
        assert resolve_current_week(now="2026-08-19T12:00:00+00:00", conn=conn) == (2026, 1)
    finally:
        conn.close()


def test_resolve_current_week_ignores_rows_with_null_kickoff(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    schedule = _schedule(
        ("2026_01_NE_SEA", 2026, 1, None),
        ("2026_02_NE_SEA", 2026, 2, WEEK2_KICKOFF),
    )
    conn = _seed_games(tmp_path / "partial.db", schedule)
    try:
        assert resolve_current_week(now="2026-08-19T12:00:00+00:00", conn=conn) == (2026, 2)
    finally:
        conn.close()
