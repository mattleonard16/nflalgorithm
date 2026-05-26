"""Tests for real player/game field ingestion (T0 #6)."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.ingest_real_nfl_data import (
    _merge_age_from_rosters,
    _merge_game_date_from_schedule,
    _merge_red_zone_from_pbp,
    create_games_from_stats,
)


def _stat_row(gsis_id: str, team: str, season: int, week: int) -> dict:
    return {
        "gsis_id": gsis_id,
        "player_id": gsis_id,
        "player_name": "Test Player",
        "team": team,
        "opponent": "BUF",
        "season": season,
        "week": week,
        "position": "WR",
        "rushing_attempts": 0.0,
        "receptions": 0.0,
    }


def test_game_date_from_schedule_uses_real_kickoff():
    df = pd.DataFrame([_stat_row("00-001", "MIA", 2024, 4)])
    sched = pd.DataFrame(
        [{"season": 2024, "week": 4, "home_team": "NE", "away_team": "MIA", "gameday": "2024-09-29"}]
    )
    merged = _merge_game_date_from_schedule(df, sched)
    assert merged.loc[0, "game_date"] == "2024-09-29"


def test_game_date_falls_back_to_season_start_when_no_match():
    df = pd.DataFrame([_stat_row("00-001", "MIA", 2024, 99)])
    sched = pd.DataFrame(
        [{"season": 2024, "week": 4, "home_team": "NE", "away_team": "MIA", "gameday": "2024-09-29"}]
    )
    merged = _merge_game_date_from_schedule(df, sched)
    assert merged.loc[0, "game_date"] == "2024-09-01"


def test_game_date_handles_empty_schedule():
    df = pd.DataFrame([_stat_row("00-001", "MIA", 2024, 1)])
    merged = _merge_game_date_from_schedule(df, pd.DataFrame())
    assert merged.loc[0, "game_date"] == "2024-09-01"


def test_age_from_rosters_computes_calendar_age():
    df = pd.DataFrame([_stat_row("00-001", "MIA", 2024, 4)])
    df["game_date"] = "2024-09-29"
    rosters = pd.DataFrame(
        [{"gsis_id": "00-001", "season": 2024, "birth_date": "2000-01-01"}]
    )
    merged = _merge_age_from_rosters(df, rosters)
    assert merged.loc[0, "age"] == 25  # ~24.75 → rounds to 25


def test_age_falls_back_to_26_when_birth_date_missing():
    df = pd.DataFrame([_stat_row("00-001", "MIA", 2024, 4)])
    df["game_date"] = "2024-09-29"
    rosters = pd.DataFrame(
        [{"gsis_id": "00-002", "season": 2024, "birth_date": "2000-01-01"}]
    )
    merged = _merge_age_from_rosters(df, rosters)
    assert merged.loc[0, "age"] == 26


def test_age_falls_back_when_rosters_empty():
    df = pd.DataFrame([_stat_row("00-001", "MIA", 2024, 4)])
    df["game_date"] = "2024-09-29"
    merged = _merge_age_from_rosters(df, pd.DataFrame())
    assert merged.loc[0, "age"] == 26


def test_red_zone_uses_pbp_count_when_available():
    df = pd.DataFrame([_stat_row("00-001", "MIA", 2024, 4)])
    df["red_zone_touches"] = 0.5  # synthetic fallback
    pbp_rz = pd.DataFrame(
        [{"player_gsis_id": "00-001", "season": 2024, "week": 4, "red_zone_touches": 7}]
    )
    merged = _merge_red_zone_from_pbp(df, pbp_rz)
    assert merged.loc[0, "red_zone_touches"] == 7.0


def test_red_zone_keeps_synthetic_when_pbp_empty():
    df = pd.DataFrame([_stat_row("00-001", "MIA", 2024, 4)])
    df["red_zone_touches"] = 0.5
    merged = _merge_red_zone_from_pbp(df, pd.DataFrame())
    assert merged.loc[0, "red_zone_touches"] == 0.5


def test_red_zone_falls_back_to_synthetic_when_no_pbp_row_for_player():
    df = pd.DataFrame([_stat_row("00-001", "MIA", 2024, 4)])
    df["red_zone_touches"] = 0.5
    pbp_rz = pd.DataFrame(
        [{"player_gsis_id": "00-999", "season": 2024, "week": 4, "red_zone_touches": 7}]
    )
    merged = _merge_red_zone_from_pbp(df, pbp_rz)
    assert merged.loc[0, "red_zone_touches"] == 0.5


def test_create_games_from_stats_uses_real_schedule_gameday():
    stats = pd.DataFrame([
        {"season": 2024, "week": 4, "team": "MIA", "opponent": "NE"},
        {"season": 2024, "week": 4, "team": "NE", "opponent": "MIA"},
    ])
    sched = pd.DataFrame(
        [{"season": 2024, "week": 4, "home_team": "NE", "away_team": "MIA", "gameday": "2024-09-29"}]
    )
    games = create_games_from_stats(stats, schedule=sched)
    assert len(games) == 1
    assert games.iloc[0]["game_date"] == "2024-09-29"
    assert games.iloc[0]["home_team"] == "NE"
    assert games.iloc[0]["away_team"] == "MIA"


def test_create_games_from_stats_no_more_jan_01_placeholder():
    """Regression: never emit f'{season}-01-01' (the old hardcoded value)."""
    stats = pd.DataFrame([
        {"season": 2024, "week": 1, "team": "MIA", "opponent": "BUF"},
        {"season": 2024, "week": 1, "team": "BUF", "opponent": "MIA"},
    ])
    games = create_games_from_stats(stats, schedule=pd.DataFrame())
    assert all(d != "2024-01-01" for d in games["game_date"])
