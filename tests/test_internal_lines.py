"""Internal line derivation and storage.

The DDL below is the proposed ``internal_lines`` table, written in the same
SQLite dialect ``schema_migrations._ddl_statements`` uses so that
``_mysql_compatible_ddl`` can translate the ``TEXT`` key columns to bounded
``VARCHAR`` on MySQL. The test owns its copy on purpose: this module must not
depend on the migration runner having been extended yet.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from utils.internal_lines import (
    INTERNAL_LINE_COLUMNS,
    build_internal_lines,
    load_internal_lines,
    persist_internal_lines,
    round_to_line,
)

INTERNAL_LINES_DDL = """
CREATE TABLE IF NOT EXISTS internal_lines (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    player_id TEXT NOT NULL,
    market TEXT NOT NULL,
    name TEXT NOT NULL,
    team TEXT NOT NULL,
    position TEXT NOT NULL,
    line REAL NOT NULL,
    mu REAL NOT NULL,
    sigma REAL NOT NULL,
    universe_rank INTEGER NOT NULL,
    generated_at TEXT NOT NULL,
    computed_at TEXT NOT NULL,
    PRIMARY KEY (season, week, player_id, market)
)
"""

GENERATED_AT = "2026-09-08T12:00:00+00:00"
COMPUTED_AT = "2026-09-09T12:00:00+00:00"


def _universe(*rows: tuple[int, str, str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "universe_rank": rank,
                "player_id": player_id,
                "gsis_id": f"gsis_{player_id}",
                "stats_player_id": f"OLD_{player_id}",
                "name": name,
                "team": team,
                "position": position,
                "usage_score": 100.0 - rank,
                "total_opportunities": 600.0 - rank,
                "games_in_window": 6,
            }
            for rank, player_id, name, team, position in rows
        ]
    )


def _projections(*rows: tuple[str, str, float, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "player_id": player_id,
                "team": "SEA",
                "opponent": "NE",
                "market": market,
                "mu": mu,
                "sigma": sigma,
                "model_version": "causal_asof_v1",
                "featureset_hash": "abc123",
                "generated_at": GENERATED_AT,
            }
            for player_id, market, mu, sigma in rows
        ]
    )


DEFAULT_UNIVERSE = _universe(
    (1, "SEA_alpha_receiver", "Alpha Receiver", "SEA", "WR"),
    (2, "SEA_bravo_back", "Bravo Back", "SEA", "RB"),
)
DEFAULT_PROJECTIONS = _projections(
    ("SEA_alpha_receiver", "receiving_yards", 72.4, 25.0),
    ("SEA_alpha_receiver", "rushing_yards", 2.2, 4.0),
    ("SEA_bravo_back", "rushing_yards", 61.7, 20.0),
)


@pytest.mark.parametrize(
    ("mu", "expected"),
    [
        (72.4, 72.5),
        (72.2, 72.0),
        (16.25, 16.5),
        (16.75, 17.0),
        (17.0, 17.0),
        (0.2, 0.0),
        (0.25, 0.5),
    ],
)
def test_round_to_line_rounds_half_up(mu: float, expected: float) -> None:
    assert round_to_line(mu) == expected


def test_round_to_line_avoids_bankers_rounding() -> None:
    # Python's round() sends both of these to an even multiple, which would put
    # 16.25 at 16.0 and 16.75 at 17.0 — inconsistent by eye.
    assert round(16.25 * 2) / 2 == 16.0
    assert round_to_line(16.25) == 16.5


def test_round_to_line_rejects_a_non_positive_increment() -> None:
    with pytest.raises(ValueError, match="increment must be positive"):
        round_to_line(10.0, 0.0)


def test_build_produces_one_row_per_player_per_market() -> None:
    lines = build_internal_lines(DEFAULT_UNIVERSE, DEFAULT_PROJECTIONS, computed_at=COMPUTED_AT)
    assert list(lines.columns) == INTERNAL_LINE_COLUMNS
    assert len(lines) == 3
    assert set(zip(lines["player_id"], lines["market"])) == {
        ("SEA_alpha_receiver", "receiving_yards"),
        ("SEA_alpha_receiver", "rushing_yards"),
        ("SEA_bravo_back", "rushing_yards"),
    }
    alpha = lines[
        (lines["player_id"] == "SEA_alpha_receiver") & (lines["market"] == "receiving_yards")
    ].iloc[0]
    assert alpha["line"] == 72.5
    assert alpha["mu"] == pytest.approx(72.4)
    assert alpha["universe_rank"] == 1
    assert alpha["generated_at"] == GENERATED_AT
    assert alpha["computed_at"] == COMPUTED_AT


def test_rows_are_ordered_by_universe_rank_then_market() -> None:
    lines = build_internal_lines(DEFAULT_UNIVERSE, DEFAULT_PROJECTIONS, computed_at=COMPUTED_AT)
    assert list(lines["universe_rank"]) == [1, 1, 2]
    assert list(lines["market"][:2]) == ["receiving_yards", "rushing_yards"]


def test_players_outside_the_universe_are_not_priced() -> None:
    projections = pd.concat(
        [DEFAULT_PROJECTIONS, _projections(("SEA_charlie_depth", "receiving_yards", 9.0, 6.0))]
    )
    lines = build_internal_lines(DEFAULT_UNIVERSE, projections, computed_at=COMPUTED_AT)
    assert "SEA_charlie_depth" not in set(lines["player_id"])


def test_a_universe_player_with_no_projection_simply_has_no_line() -> None:
    universe = pd.concat(
        [DEFAULT_UNIVERSE, _universe((3, "SEA_delta_end", "Delta End", "SEA", "TE"))]
    )
    lines = build_internal_lines(universe, DEFAULT_PROJECTIONS, computed_at=COMPUTED_AT)
    assert "SEA_delta_end" not in set(lines["player_id"])
    assert len(lines) == 3


def test_blank_projection_team_falls_back_to_the_roster_team() -> None:
    projections = DEFAULT_PROJECTIONS.copy()
    projections["team"] = ["", None, "SEA"]
    lines = build_internal_lines(DEFAULT_UNIVERSE, projections, computed_at=COMPUTED_AT)
    assert set(lines["team"]) == {"SEA"}


def test_negative_mu_fails_loud() -> None:
    projections = _projections(("SEA_alpha_receiver", "receiving_yards", -3.0, 20.0))
    with pytest.raises(ValueError, match="mu is negative"):
        build_internal_lines(DEFAULT_UNIVERSE, projections)


def test_non_positive_sigma_fails_loud() -> None:
    projections = _projections(("SEA_alpha_receiver", "receiving_yards", 50.0, 0.0))
    with pytest.raises(ValueError, match=r"sigma is missing or <= 0"):
        build_internal_lines(DEFAULT_UNIVERSE, projections)


def test_an_unregistered_market_fails_loud() -> None:
    projections = _projections(("SEA_alpha_receiver", "field_goals", 0.6, 0.4))
    with pytest.raises(ValueError, match="sports.markets does not define"):
        build_internal_lines(DEFAULT_UNIVERSE, projections)


def test_no_overlap_between_universe_and_projections_fails_loud() -> None:
    projections = _projections(("SOMEONE_ELSE", "receiving_yards", 50.0, 20.0))
    with pytest.raises(ValueError, match="has a weekly_projections row"):
        build_internal_lines(DEFAULT_UNIVERSE, projections)


def test_missing_universe_columns_fail_loud() -> None:
    with pytest.raises(ValueError, match="universe frame missing required columns"):
        build_internal_lines(pd.DataFrame({"player_id": ["x"]}), DEFAULT_PROJECTIONS)


def test_missing_projection_columns_fail_loud() -> None:
    with pytest.raises(ValueError, match="projections frame missing required columns"):
        build_internal_lines(DEFAULT_UNIVERSE, pd.DataFrame({"player_id": ["x"]}))


def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute(INTERNAL_LINES_DDL)
    conn.execute("""
        CREATE TABLE weekly_odds (
            event_id TEXT, season INTEGER, week INTEGER, player_id TEXT, market TEXT,
            sportsbook TEXT, line REAL, price INTEGER, as_of TEXT, under_price INTEGER
        )
        """)
    conn.execute("""
        CREATE TABLE weekly_projections (
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            player_id TEXT NOT NULL,
            team TEXT NOT NULL,
            opponent TEXT NOT NULL,
            market TEXT NOT NULL,
            mu REAL NOT NULL,
            sigma REAL NOT NULL,
            model_version TEXT NOT NULL,
            featureset_hash TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            PRIMARY KEY (season, week, player_id, market)
        )
        """)
    conn.commit()
    return conn


def test_persist_round_trips_every_column(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    lines = build_internal_lines(DEFAULT_UNIVERSE, DEFAULT_PROJECTIONS, computed_at=COMPUTED_AT)
    conn = _open_db(tmp_path / "lines.db")
    try:
        assert persist_internal_lines(lines, conn) == 3
        stored = pd.read_sql_query(
            "SELECT * FROM internal_lines ORDER BY universe_rank, market", conn
        )
    finally:
        conn.close()

    assert len(stored) == 3
    assert set(stored.columns) == set(INTERNAL_LINE_COLUMNS)
    assert list(stored["line"]) == [72.5, 2.0, 61.5]
    assert set(stored["computed_at"]) == {COMPUTED_AT}


def test_persist_never_writes_to_weekly_odds(tmp_path: Path, monkeypatch) -> None:
    # Synthetic rows in weekly_odds are what utils.odds_quality exists to screen
    # back out; internal lines must never land there.
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    lines = build_internal_lines(DEFAULT_UNIVERSE, DEFAULT_PROJECTIONS, computed_at=COMPUTED_AT)
    conn = _open_db(tmp_path / "isolation.db")
    try:
        persist_internal_lines(lines, conn)
        odds_rows = conn.execute("SELECT COUNT(*) FROM weekly_odds").fetchone()[0]
    finally:
        conn.close()
    assert odds_rows == 0


def test_republishing_a_week_replaces_the_slice(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    first = build_internal_lines(DEFAULT_UNIVERSE, DEFAULT_PROJECTIONS, computed_at=COMPUTED_AT)
    revised_universe = _universe((1, "SEA_bravo_back", "Bravo Back", "SEA", "RB"))
    revised = build_internal_lines(
        revised_universe,
        _projections(("SEA_bravo_back", "rushing_yards", 70.4, 20.0)),
        computed_at="2026-09-10T12:00:00+00:00",
    )

    conn = _open_db(tmp_path / "republish.db")
    try:
        persist_internal_lines(first, conn)
        assert persist_internal_lines(revised, conn) == 1
        stored = pd.read_sql_query("SELECT * FROM internal_lines", conn)
    finally:
        conn.close()

    # The dropped player is gone, not stranded on last run's card.
    assert list(stored["player_id"]) == ["SEA_bravo_back"]
    assert stored.iloc[0]["line"] == 70.5


def test_persist_rejects_an_empty_frame(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    conn = _open_db(tmp_path / "empty.db")
    try:
        with pytest.raises(ValueError, match="refusing to persist an empty"):
            persist_internal_lines(pd.DataFrame(columns=INTERNAL_LINE_COLUMNS), conn)
    finally:
        conn.close()


def test_persist_rejects_a_frame_missing_columns(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    conn = _open_db(tmp_path / "short.db")
    try:
        with pytest.raises(ValueError, match="missing required columns"):
            persist_internal_lines(pd.DataFrame({"season": [2026]}), conn)
    finally:
        conn.close()


def test_load_internal_lines_reads_projections_for_a_supplied_universe(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    conn = _open_db(tmp_path / "load.db")
    try:
        DEFAULT_PROJECTIONS.drop(columns=[]).to_sql(
            "weekly_projections", conn, if_exists="append", index=False
        )
        conn.commit()
        lines = load_internal_lines(
            2026, 1, universe=DEFAULT_UNIVERSE, conn=conn, computed_at=COMPUTED_AT
        )
    finally:
        conn.close()

    assert len(lines) == 3
    assert list(lines.columns) == INTERNAL_LINE_COLUMNS


def test_load_internal_lines_without_projections_fails_loud(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    conn = _open_db(tmp_path / "no_projections.db")
    try:
        with pytest.raises(ValueError, match="no weekly_projections rows for season 2026 week 1"):
            load_internal_lines(2026, 1, universe=DEFAULT_UNIVERSE, conn=conn)
    finally:
        conn.close()
