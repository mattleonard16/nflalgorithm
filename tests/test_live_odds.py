"""Live-odds selection: stale filter first, then latest remaining snapshot."""

from __future__ import annotations

import pandas as pd
import pytest

from utils.clv import resolve_closing_lines
from utils.live_odds import kickoffs_from_games, select_live_odds

KICKOFF = "2026-09-13T17:00:00+00:00"
KEY = {
    "event_id": "2026_01_NE_SEA",
    "player_id": "p1",
    "market": "receiving_yards",
    "sportsbook": "BookA",
}


def _odds(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame([{**KEY, "under_price": -110, **row} for row in rows])


def _games(game_id: str = KEY["event_id"], kickoff_utc: str = KICKOFF) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [game_id],
            "kickoff_utc": [kickoff_utc],
        }
    )


def test_kickoffs_from_games_renames_game_id() -> None:
    frame = kickoffs_from_games(_games())
    assert list(frame.columns) == ["event_id", "kickoff_utc"]
    assert frame.iloc[0]["event_id"] == KEY["event_id"]


def test_kickoffs_from_games_accepts_event_id() -> None:
    frame = kickoffs_from_games(pd.DataFrame({"event_id": ["e1"], "kickoff_utc": [KICKOFF]}))
    assert frame.iloc[0]["event_id"] == "e1"


def test_kickoffs_from_games_empty() -> None:
    frame = kickoffs_from_games(pd.DataFrame())
    assert frame.empty
    assert list(frame.columns) == ["event_id", "kickoff_utc"]


def test_kickoffs_from_games_drops_null_kickoffs() -> None:
    games = pd.DataFrame(
        {
            "game_id": [KEY["event_id"], "2026_01_KC_BUF"],
            "kickoff_utc": [KICKOFF, None],
        }
    )
    frame = kickoffs_from_games(games)
    assert list(frame["event_id"]) == [KEY["event_id"]]


def test_select_live_odds_null_kickoff_keeps_latest() -> None:
    # A scheduled game whose kickoff is not yet known must not crash the
    # selection; its odds pass through unjoinable and the latest one wins.
    odds = _odds(
        {"line": 70.5, "price": -110, "as_of": "2026-09-13T15:00:00+00:00"},
        {"line": 99.5, "price": -105, "as_of": "2026-09-13T18:00:00+00:00"},
    )
    live = select_live_odds(odds, kickoffs_from_games(_games(kickoff_utc=None)))
    assert len(live) == 1
    assert live.iloc[0]["line"] == 99.5


def test_kickoffs_from_games_missing_id_fails_loud() -> None:
    with pytest.raises(ValueError, match="game_id or event_id"):
        kickoffs_from_games(pd.DataFrame({"kickoff_utc": [KICKOFF]}))


def test_select_live_odds_keeps_pre_kickoff_when_later_post_kickoff_exists() -> None:
    odds = _odds(
        {"line": 70.5, "price": -110, "as_of": "2026-09-13T15:00:00+00:00"},
        {"line": 99.5, "price": -105, "as_of": "2026-09-13T18:00:00+00:00"},
    )
    live = select_live_odds(odds, kickoffs_from_games(_games()))
    assert len(live) == 1
    assert live.iloc[0]["line"] == 70.5


def test_select_live_odds_drops_quotes_older_than_freshness_window() -> None:
    odds = _odds(
        {"line": 60.5, "price": -110, "as_of": "2026-09-10T12:00:00+00:00"},
        {"line": 71.5, "price": -110, "as_of": "2026-09-13T15:00:00+00:00"},
    )
    live = select_live_odds(odds, kickoffs_from_games(_games()), max_age_hours=48)
    assert list(live["line"]) == [71.5]


def test_select_live_odds_without_kickoffs_keeps_latest() -> None:
    odds = _odds(
        {"line": 70.5, "price": -110, "as_of": "2026-09-13T15:00:00+00:00"},
        {"line": 99.5, "price": -105, "as_of": "2026-09-13T18:00:00+00:00"},
    )
    live = select_live_odds(odds, kickoffs_from_games(pd.DataFrame()))
    assert live.iloc[0]["line"] == 99.5


def test_kickoff_aware_close_matches_live_selection_on_pre_kickoff_quotes() -> None:
    odds = _odds(
        {"line": 70.5, "price": -110, "as_of": "2026-09-13T15:00:00+00:00"},
        {"line": 99.5, "price": -105, "as_of": "2026-09-13T18:00:00+00:00"},
    )
    kickoffs = kickoffs_from_games(_games())
    closing = resolve_closing_lines(odds, kickoffs)
    assert closing.iloc[0]["close_line"] == 70.5
    assert select_live_odds(odds, kickoffs).iloc[0]["line"] == 70.5
