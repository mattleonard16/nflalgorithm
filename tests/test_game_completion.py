"""Rules for deciding a game is finished and its bets can be graded.

The bug these guard: grading a week mid-slate used to mark every bet whose
player had no stats row as a push with zero profit, which is indistinguishable
from a real settled push and flows into weekly_performance and the CLV average.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from utils.game_completion import (
    PENDING_IN_PROGRESS,
    PENDING_NO_KICKOFF,
    PENDING_NO_STATS,
    PENDING_NOT_KICKED_OFF,
    SETTLE_AFTER_KICKOFF,
    classify_games,
    gradeable_event_ids,
)

# 2026 week 1's real shape: a Thursday opener, a Friday game, the Sunday block,
# and a Monday nighter. Grading on Sunday night has to handle all four states.
THURSDAY = "2026-09-10T00:20:00+00:00"
FRIDAY = "2026-09-11T00:35:00+00:00"
SUNDAY_EARLY = "2026-09-13T17:00:00+00:00"
MONDAY = "2026-09-15T00:15:00+00:00"


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"game_id": "2026_01_NE_SEA", "away_team": "NE", "home_team": "SEA",
             "kickoff_utc": THURSDAY},
            {"game_id": "2026_01_SF_LA", "away_team": "SF", "home_team": "LAR",
             "kickoff_utc": FRIDAY},
            {"game_id": "2026_01_TB_CIN", "away_team": "TB", "home_team": "CIN",
             "kickoff_utc": SUNDAY_EARLY},
            {"game_id": "2026_01_DEN_KC", "away_team": "DEN", "home_team": "KC",
             "kickoff_utc": MONDAY},
        ]
    )


def _actuals(teams: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"player_id": f"p{i}", "team": team, "receptions": 4} for i, team in enumerate(teams)]
    )


def test_a_finished_game_with_published_stats_is_gradeable() -> None:
    report = classify_games(
        _games(),
        _actuals(["SEA", "NE"]),
        now=datetime(2026, 9, 13, 23, 0, tzinfo=timezone.utc),
    )
    assert "2026_01_NE_SEA" in report.final


def test_a_game_that_has_not_kicked_off_is_pending_not_pushed() -> None:
    """The whole point. Monday night's bets must not settle on Sunday."""
    report = classify_games(
        _games(),
        _actuals(["SEA", "NE", "SF", "LAR", "TB", "CIN"]),
        now=datetime(2026, 9, 13, 23, 0, tzinfo=timezone.utc),
    )
    assert report.pending["2026_01_DEN_KC"] == PENDING_NOT_KICKED_OFF
    assert "2026_01_DEN_KC" not in report.final


def test_a_game_still_being_played_is_pending() -> None:
    """One hour after kickoff the game is live, and a live game has no result."""
    report = classify_games(
        _games(),
        _actuals(["TB", "CIN"]),
        now=datetime(2026, 9, 13, 18, 0, tzinfo=timezone.utc),
    )
    assert report.pending["2026_01_TB_CIN"] == PENDING_IN_PROGRESS


def test_the_settle_margin_covers_overtime() -> None:
    """A game is not final at the three-hour mark; overtime and delays run past
    it. SETTLE_AFTER_KICKOFF is the guard, so a change to it fails here."""
    assert SETTLE_AFTER_KICKOFF == timedelta(hours=4)
    kickoff = datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc)
    stats = _actuals(["TB", "CIN"])

    at_three = classify_games(_games(), stats, now=kickoff + timedelta(hours=3, minutes=30))
    assert at_three.pending["2026_01_TB_CIN"] == PENDING_IN_PROGRESS

    at_four = classify_games(_games(), stats, now=kickoff + timedelta(hours=4, minutes=1))
    assert "2026_01_TB_CIN" in at_four.final


def test_a_finished_game_whose_stats_have_not_published_is_pending() -> None:
    """The feed lags the final whistle. An empty stats table is a wait, not a
    week of pushes, which is exactly how the old grading run lost a week."""
    report = classify_games(
        _games(),
        pd.DataFrame(columns=["player_id", "team", "receptions"]),
        now=datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc),
    )
    assert report.final == ()
    assert set(report.pending.values()) == {PENDING_NO_STATS}


def test_either_side_of_a_game_having_stats_is_enough() -> None:
    """Holding a game back because only one club published would strand it."""
    report = classify_games(
        _games(),
        _actuals(["SEA"]),
        now=datetime(2026, 9, 13, 23, 0, tzinfo=timezone.utc),
    )
    assert "2026_01_NE_SEA" in report.final


def test_an_unreadable_kickoff_is_pending_rather_than_assumed_over() -> None:
    games = _games()
    games.loc[games["game_id"] == "2026_01_TB_CIN", "kickoff_utc"] = None
    report = classify_games(
        games, _actuals(["TB", "CIN"]), now=datetime(2026, 9, 20, tzinfo=timezone.utc)
    )
    assert report.pending["2026_01_TB_CIN"] == PENDING_NO_KICKOFF

    games.loc[games["game_id"] == "2026_01_TB_CIN", "kickoff_utc"] = "not-a-timestamp"
    report = classify_games(
        games, _actuals(["TB", "CIN"]), now=datetime(2026, 9, 20, tzinfo=timezone.utc)
    )
    assert report.pending["2026_01_TB_CIN"] == PENDING_NO_KICKOFF


def test_week_is_complete_only_when_nothing_is_pending() -> None:
    stats = _actuals(["SEA", "NE", "SF", "LAR", "TB", "CIN", "DEN", "KC"])
    mid = classify_games(_games(), stats, now=datetime(2026, 9, 13, 23, tzinfo=timezone.utc))
    assert mid.is_week_complete is False

    done = classify_games(_games(), stats, now=datetime(2026, 9, 15, 6, tzinfo=timezone.utc))
    assert done.is_week_complete is True
    assert len(done.final) == 4


def test_summary_names_every_pending_game_and_its_reason() -> None:
    report = classify_games(
        _games(), _actuals(["SEA", "NE"]), now=datetime(2026, 9, 13, 23, tzinfo=timezone.utc)
    )
    text = report.summary()
    assert "1 game(s) final, 3 pending" in text
    assert "2026_01_DEN_KC: not_kicked_off" in text


def test_gradeable_event_ids_matches_the_report() -> None:
    now = datetime(2026, 9, 13, 23, 0, tzinfo=timezone.utc)
    stats = _actuals(["SEA", "NE", "SF", "LAR"])
    assert gradeable_event_ids(_games(), stats, now=now) == set(
        classify_games(_games(), stats, now=now).final
    )


def test_an_empty_schedule_grades_nothing_rather_than_raising() -> None:
    report = classify_games(pd.DataFrame(), _actuals(["SEA"]))
    assert report.final == ()
    assert report.pending == {}
    assert report.is_week_complete is False


def test_a_schedule_missing_its_columns_fails_loud() -> None:
    with pytest.raises(ValueError, match="games missing required column: kickoff_utc"):
        classify_games(
            pd.DataFrame({"game_id": ["x"], "home_team": ["A"], "away_team": ["B"]}),
            _actuals(["A"]),
        )
    with pytest.raises(ValueError, match="home_team, away_team"):
        classify_games(pd.DataFrame({"game_id": ["x"], "kickoff_utc": [MONDAY]}), _actuals(["A"]))


def test_an_event_id_column_is_accepted_like_the_odds_path_does() -> None:
    games = _games().rename(columns={"game_id": "event_id"})
    report = classify_games(
        games, _actuals(["SEA"]), now=datetime(2026, 9, 13, 23, tzinfo=timezone.utc)
    )
    assert "2026_01_NE_SEA" in report.final
