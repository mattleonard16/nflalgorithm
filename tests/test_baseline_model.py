"""The public baseline projector a clone runs when the private weekly model is absent."""

from __future__ import annotations

import pandas as pd
import pytest

from models.position_specific import baseline
from utils.db import execute, read_dataframe

SEASON = 2025


@pytest.fixture()
def db(matrix_database):
    for table in (
        "player_stats_enhanced",
        "games",
        "nfl_roster_players",
        "nfl_player_context_snapshots",
        "weekly_projections",
    ):
        execute(f"DELETE FROM {table}")
    return matrix_database


def _game(week: int, home: str, away: str) -> None:
    execute(
        "INSERT INTO games (game_id, season, week, home_team, away_team, game_date) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (f"{SEASON}_{week:02d}_{away}_{home}", SEASON, week, home, away, "2025-10-01"),
    )


def _stats(player_id: str, position: str, team: str, week: int, **stats: float) -> None:
    columns = ["player_id", "gsis_id", "season", "week", "name", "team", "position", *stats]
    values = [player_id, f"gsis-{player_id}", SEASON, week, player_id, team, position, *stats.values()]
    execute(
        f"INSERT INTO player_stats_enhanced ({', '.join(columns)}) "
        f"VALUES ({', '.join('?' for _ in values)})",
        tuple(values),
    )


def _roster(player_id: str, position: str, team: str, **context: object) -> None:
    gsis = f"gsis-{player_id}"
    execute(
        "INSERT INTO nfl_roster_players "
        "(season, gsis_id, player_id, player_name, team, position, roster_status, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, 'ACT', '2025-10-01')",
        (SEASON, gsis, player_id, player_id, team, position),
    )
    columns = ["season", "week", "gsis_id", "player_id", "team", "position", "prior_source",
               "captured_at", *context]
    values = [SEASON, 4, gsis, player_id, team, position, "test", "2025-10-01", *context.values()]
    execute(
        f"INSERT INTO nfl_player_context_snapshots ({', '.join(columns)}) "
        f"VALUES ({', '.join('?' for _ in values)})",
        tuple(values),
    )


def _three_weeks(player_id: str, position: str, team: str, **stats: float) -> None:
    for week in (1, 2, 3):
        _stats(player_id, position, team, week, **stats)


def test_projects_only_from_games_before_the_target_week(db) -> None:
    _game(4, "BUF", "MIA")
    for week, yards in ((1, 50.0), (2, 60.0), (3, 70.0), (4, 500.0)):
        _stats("wr1", "WR", "BUF", week, receiving_yards=yards, targets=6, receptions=4)

    predictions = baseline.predict_week(SEASON, 4)

    mu = predictions.set_index("market").loc["receiving_yards", "mu"]
    assert mu == pytest.approx(pd.Series([50.0, 60.0, 70.0]).ewm(span=3).mean().iloc[-1])
    stored = read_dataframe(
        "SELECT team, opponent, model_version FROM weekly_projections WHERE market = 'receiving_yards'"
    )
    assert stored.values.tolist() == [["BUF", "MIA", baseline.MODEL_VERSION]]


def test_a_player_under_a_markets_volume_floor_gets_no_projection_there(db) -> None:
    _game(4, "BUF", "MIA")
    _three_weeks("wr1", "WR", "BUF", receiving_yards=40.0, targets=1, receptions=1)

    predictions = baseline.predict_week(SEASON, 4)

    assert "receiving_yards" not in set(predictions["market"])


def test_the_roster_path_drops_ruled_out_players_and_backup_qbs(db) -> None:
    _game(4, "BUF", "MIA")
    _roster("qb1", "QB", "BUF", depth_rank=1, is_starter=1)
    _roster("qb2", "QB", "BUF", depth_rank=2, is_starter=0)
    _roster("wr_out", "WR", "BUF", injury_status="Out", injury_report_week=4)
    # Ruled out on last week's report only. He may be back, so he keeps a projection.
    _roster("wr_back", "WR", "BUF", injury_status="Out", injury_report_week=3)
    for qb in ("qb1", "qb2"):
        _three_weeks(qb, "QB", "BUF", passing_yards=250.0, passing_attempts=34)
    for wr in ("wr_out", "wr_back"):
        _three_weeks(wr, "WR", "BUF", receiving_yards=60.0, targets=7, receptions=5)

    predictions = baseline.predict_week(SEASON, 4, roster_backed=True)

    assert set(predictions["player_id"]) == {"qb1", "wr_back"}


def test_a_rerun_keeps_the_stored_rows_of_teams_already_playing(db) -> None:
    _game(4, "BUF", "MIA")
    for player_id, team in (("buf_wr", "BUF"), ("mia_wr", "MIA")):
        _roster(player_id, "WR", team)
        _three_weeks(player_id, "WR", team, receiving_yards=60.0, targets=7, receptions=5)
    baseline.predict_week(SEASON, 4, roster_backed=True)
    execute("UPDATE weekly_projections SET mu = 1.0")

    baseline.predict_week(SEASON, 4, roster_backed=True, exclude_teams=frozenset({"BUF"}))

    stored = read_dataframe(
        "SELECT DISTINCT player_id, mu = 1.0 AS kept FROM weekly_projections ORDER BY player_id"
    )
    assert stored.values.tolist() == [["buf_wr", 1], ["mia_wr", 0]]
