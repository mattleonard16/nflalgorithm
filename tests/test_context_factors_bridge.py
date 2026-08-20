"""Tests for the gsis_id bridge in utils.context_factors.load_context_inputs.

2026 ``weekly_projections`` ids are minted from the season roster
(``ARI_james_conner``) while ``player_stats_enhanced`` keeps the legacy
abbreviated form (``ARI_j_conner``). The only shared key is ``gsis_id`` via
``nfl_roster_players``. The bridge remaps history rows onto caller ids that
have no direct history — and must degrade to unchanged history (neutral
factors) when the bridge tables are missing, never crash.
"""

import sqlite3
from pathlib import Path

import pandas as pd

from utils.context_factors import load_context_inputs

SEASON = 2026
WEEK = 1
MARKET = "receiving_yards"
GSIS = "00-0034796"


def _make_db(db_path: Path, *, with_roster: bool = True, with_gsis_column: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE games ("
        "season INTEGER, week INTEGER, home_team TEXT, away_team TEXT, "
        "spread_line REAL, total_line REAL)"
    )
    gsis_col = ", gsis_id TEXT" if with_gsis_column else ""
    conn.execute(
        "CREATE TABLE player_stats_enhanced ("
        "player_id TEXT, name TEXT, season INTEGER, week INTEGER, team TEXT, "
        f"position TEXT, receiving_yards REAL, targets REAL{gsis_col})"
    )
    if with_roster:
        conn.execute(
            "CREATE TABLE nfl_roster_players ("
            "season INTEGER, gsis_id TEXT, player_id TEXT, team TEXT, "
            "position TEXT, roster_week INTEGER)"
        )
    conn.execute(
        "CREATE TABLE weekly_projections ("
        "season INTEGER, week INTEGER, player_id TEXT, team TEXT, market TEXT)"
    )
    return conn


def _insert_history(conn: sqlite3.Connection, player_id: str, *, gsis: str = GSIS, weeks: int = 4) -> None:
    for wk in range(1, weeks + 1):
        conn.execute(
            "INSERT INTO player_stats_enhanced "
            "(player_id, name, season, week, team, position, receiving_yards, targets, gsis_id) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (player_id, "James Conner", SEASON - 1, wk, "ARI", "RB", 40.0 + wk, 5.0, gsis),
        )


def _players(player_id: str, *, with_position: bool = True) -> pd.DataFrame:
    row = {"player_id": player_id, "team": "ARI"}
    if with_position:
        row["position"] = "RB"
    return pd.DataFrame([row])


def test_roster_id_gains_stats_history_via_gsis(tmp_path: Path) -> None:
    conn = _make_db(tmp_path / "bridge.db")
    _insert_history(conn, "ARI_j_conner")
    conn.execute(
        "INSERT INTO nfl_roster_players VALUES (?,?,?,?,?,?)",
        (SEASON, GSIS, "ARI_james_conner", "ARI", "RB", 1),
    )
    conn.commit()

    inputs = load_context_inputs(
        SEASON, WEEK, MARKET, players=_players("ARI_james_conner"), conn=conn
    )

    ids = set(inputs["history"]["player_id"])
    assert "ARI_james_conner" in ids, "history should be remapped onto the roster id"
    assert "ARI_j_conner" not in ids, "the legacy id should be replaced, not duplicated"
    assert len(inputs["history"]) == 4


def test_direct_match_ids_are_left_alone(tmp_path: Path) -> None:
    conn = _make_db(tmp_path / "direct.db")
    _insert_history(conn, "ARI_j_conner")
    # Roster maps the same human to a different id, but the caller's id already
    # has direct history, so the bridge must not fire.
    conn.execute(
        "INSERT INTO nfl_roster_players VALUES (?,?,?,?,?,?)",
        (SEASON, GSIS, "ARI_james_conner", "ARI", "RB", 1),
    )
    conn.commit()

    inputs = load_context_inputs(
        SEASON, WEEK, MARKET, players=_players("ARI_j_conner"), conn=conn
    )

    assert set(inputs["history"]["player_id"]) == {"ARI_j_conner"}


def test_missing_roster_table_degrades_to_unchanged_history(tmp_path: Path) -> None:
    conn = _make_db(tmp_path / "noroster.db", with_roster=False)
    _insert_history(conn, "ARI_j_conner")
    conn.commit()

    inputs = load_context_inputs(
        SEASON, WEEK, MARKET, players=_players("ARI_james_conner"), conn=conn
    )

    assert set(inputs["history"]["player_id"]) == {"ARI_j_conner"}


def test_missing_gsis_column_degrades_to_unchanged_history(tmp_path: Path) -> None:
    conn = _make_db(tmp_path / "nogsis.db", with_gsis_column=False)
    for wk in range(1, 5):
        conn.execute(
            "INSERT INTO player_stats_enhanced "
            "(player_id, name, season, week, team, position, receiving_yards, targets) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("ARI_j_conner", "James Conner", SEASON - 1, wk, "ARI", "RB", 42.0, 5.0),
        )
    conn.execute(
        "INSERT INTO nfl_roster_players VALUES (?,?,?,?,?,?)",
        (SEASON, GSIS, "ARI_james_conner", "ARI", "RB", 1),
    )
    conn.commit()

    inputs = load_context_inputs(
        SEASON, WEEK, MARKET, players=_players("ARI_james_conner"), conn=conn
    )

    assert set(inputs["history"]["player_id"]) == {"ARI_j_conner"}


def test_latest_roster_stint_wins_for_midseason_mover(tmp_path: Path) -> None:
    conn = _make_db(tmp_path / "mover.db")
    _insert_history(conn, "ARI_j_conner")
    conn.execute(
        "INSERT INTO nfl_roster_players VALUES (?,?,?,?,?,?)",
        (SEASON, GSIS, "ARI_james_conner", "ARI", "RB", 1),
    )
    conn.execute(
        "INSERT INTO nfl_roster_players VALUES (?,?,?,?,?,?)",
        (SEASON, GSIS, "MIN_james_conner", "MIN", "RB", 5),
    )
    conn.commit()

    players = pd.DataFrame(
        [
            {"player_id": "ARI_james_conner", "team": "ARI", "position": "RB"},
            {"player_id": "MIN_james_conner", "team": "MIN", "position": "RB"},
        ]
    )
    inputs = load_context_inputs(SEASON, WEEK, MARKET, players=players, conn=conn)

    ids = set(inputs["history"]["player_id"])
    assert ids == {"MIN_james_conner"}, "the latest roster stint should own the history"


def test_bridge_runs_before_position_attach(tmp_path: Path) -> None:
    conn = _make_db(tmp_path / "position.db")
    _insert_history(conn, "ARI_j_conner")
    conn.execute(
        "INSERT INTO nfl_roster_players VALUES (?,?,?,?,?,?)",
        (SEASON, GSIS, "ARI_james_conner", "ARI", "RB", 1),
    )
    conn.commit()

    inputs = load_context_inputs(
        SEASON,
        WEEK,
        MARKET,
        players=_players("ARI_james_conner", with_position=False),
        conn=conn,
    )

    players = inputs["players"]
    assert "position" in players.columns
    assert players.loc[players["player_id"] == "ARI_james_conner", "position"].iloc[0] == "RB"


def test_no_missing_ids_skips_bridge_even_without_roster_table(tmp_path: Path) -> None:
    conn = _make_db(tmp_path / "skip.db", with_roster=False)
    _insert_history(conn, "ARI_james_conner")
    conn.commit()

    inputs = load_context_inputs(
        SEASON, WEEK, MARKET, players=_players("ARI_james_conner"), conn=conn
    )

    assert set(inputs["history"]["player_id"]) == {"ARI_james_conner"}
