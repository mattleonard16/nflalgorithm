"""Top-N player universe selection.

The behaviours worth pinning down:

- the join between ``player_stats_enhanced`` and ``nfl_roster_players`` runs on
  ``gsis_id``, because the two tables mint ``player_id`` differently and a
  ``player_id`` join silently matches almost nobody;
- week 1 sources its trailing usage from the prior season instead of coming
  back short;
- position floors keep the card from collapsing into one position group;
- ordering is deterministic, so two runs of the same week publish the same card.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from utils.player_universe import (
    DEFAULT_POSITION_FLOORS,
    DEFAULT_UNIVERSE_SIZE,
    eligible_roster,
    load_player_universe,
    select_universe,
    trailing_usage,
    validate_mix,
)

SMALL_FLOORS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
SMALL_SIZE = 6


def _stats(
    gsis_id: str,
    season: int,
    week: int,
    *,
    targets: float = 0.0,
    carries: float = 0.0,
    pass_attempts: float = 0.0,
    stats_player_id: str | None = None,
) -> dict:
    return {
        "player_id": stats_player_id or f"OLD_{gsis_id}",
        "gsis_id": gsis_id,
        "name": "stale name",
        "team": "OLD",
        "position": "XX",
        "season": season,
        "week": week,
        "targets": targets,
        "rushing_attempts": carries,
        "passing_attempts": pass_attempts,
    }


def _roster(gsis_id: str, position: str, *, team: str = "SEA", status: str = "ACT") -> dict:
    return {
        "season": 2026,
        "gsis_id": gsis_id,
        "player_id": f"{team}_{gsis_id.lower()}",
        "player_name": gsis_id.replace("_", " ").title(),
        "team": team,
        "position": position,
        "roster_status": status,
    }


def _pool(
    counts: dict[str, int],
    season: int = 2025,
    weeks: int = 3,
    bases: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a stats/roster pair where usage descends within each position."""
    stats_rows: list[dict] = []
    roster_rows: list[dict] = []
    for position, count in counts.items():
        base = (bases or {}).get(position, 100.0)
        for index in range(count):
            gsis = f"{position}{index:03d}"
            roster_rows.append(_roster(gsis, position))
            volume = float(base - index)
            for week in range(1, weeks + 1):
                if position == "QB":
                    stats_rows.append(_stats(gsis, season, week, pass_attempts=volume))
                else:
                    stats_rows.append(_stats(gsis, season, week, targets=volume, carries=1.0))
    return pd.DataFrame(stats_rows), pd.DataFrame(roster_rows)


def _universe(
    counts: dict[str, int], bases: dict[str, float] | None = None, **kwargs
) -> pd.DataFrame:
    stats, roster = _pool(counts, bases=bases)
    scored = trailing_usage(stats, roster, 2026, 1)
    return select_universe(scored, **kwargs)


def test_roster_join_uses_gsis_id_not_player_id() -> None:
    # This is the real shape of the data: the stats table abbreviates the first
    # name and carries last season's club; the roster spells it out. A
    # player_id join would match nothing.
    stats = pd.DataFrame(
        [
            _stats("00-0039041", 2025, w, targets=8.0, stats_player_id="ARI_e_higgins")
            for w in (1, 2)
        ]
    )
    roster = pd.DataFrame(
        [{**_roster("00-0039041", "TE", team="ARI"), "player_id": "ARI_elijah_higgins"}]
    )

    scored = trailing_usage(stats, roster, 2026, 1)
    assert len(scored) == 1
    row = scored.iloc[0]
    assert row["player_id"] == "ARI_elijah_higgins"
    assert row["stats_player_id"] == "ARI_e_higgins"
    assert row["position"] == "TE"
    assert row["team"] == "ARI"


def test_identity_comes_from_the_roster_not_the_stale_stats_row() -> None:
    stats = pd.DataFrame([_stats("g1", 2025, 1, targets=10.0)])
    roster = pd.DataFrame([_roster("g1", "WR", team="BUF")])
    row = trailing_usage(stats, roster, 2026, 1).iloc[0]
    assert (row["team"], row["position"]) == ("BUF", "WR")
    assert row["name"] == "G1"


def test_week_one_sources_trailing_usage_from_the_prior_season() -> None:
    stats = pd.DataFrame([_stats("g1", 2025, w, targets=6.0) for w in (16, 17, 18)])
    roster = pd.DataFrame([_roster("g1", "WR")])
    scored = trailing_usage(stats, roster, 2026, 1)
    assert scored.iloc[0]["games_in_window"] == 3
    assert scored.iloc[0]["usage_score"] == pytest.approx(6.0)


def test_window_walks_back_across_the_season_boundary() -> None:
    stats = pd.DataFrame(
        [_stats("g1", 2026, 1, targets=10.0)]
        + [_stats("g1", 2025, w, targets=4.0) for w in (16, 17, 18)]
    )
    roster = pd.DataFrame([_roster("g1", "WR")])
    scored = trailing_usage(stats, roster, 2026, 2, window=3)
    # Week 1 of 2026 plus the two most recent 2025 weeks: (10 + 4 + 4) / 3.
    assert scored.iloc[0]["games_in_window"] == 3
    assert scored.iloc[0]["usage_score"] == pytest.approx(6.0)


def test_window_drops_weeks_beyond_the_trailing_limit() -> None:
    stats = pd.DataFrame(
        [_stats("g1", 2025, 1, targets=100.0)]
        + [_stats("g1", 2025, w, targets=2.0) for w in (2, 3)]
    )
    roster = pd.DataFrame([_roster("g1", "WR")])
    scored = trailing_usage(stats, roster, 2026, 1, window=2)
    assert scored.iloc[0]["usage_score"] == pytest.approx(2.0)


def test_target_week_and_later_weeks_are_excluded() -> None:
    stats = pd.DataFrame([_stats("g1", 2026, 1, targets=3.0), _stats("g1", 2026, 5, targets=99.0)])
    roster = pd.DataFrame([_roster("g1", "WR")])
    scored = trailing_usage(stats, roster, 2026, 5)
    assert scored.iloc[0]["usage_score"] == pytest.approx(3.0)


def test_quarterbacks_are_ranked_on_pass_attempts() -> None:
    stats = pd.DataFrame([_stats("qb1", 2025, 1, pass_attempts=35.0, carries=4.0, targets=0.0)])
    roster = pd.DataFrame([_roster("qb1", "QB")])
    assert trailing_usage(stats, roster, 2026, 1).iloc[0]["usage_score"] == pytest.approx(35.0)


def test_skill_players_are_ranked_on_carries_plus_targets() -> None:
    stats = pd.DataFrame([_stats("rb1", 2025, 1, carries=15.0, targets=5.0, pass_attempts=1.0)])
    roster = pd.DataFrame([_roster("rb1", "RB")])
    assert trailing_usage(stats, roster, 2026, 1).iloc[0]["usage_score"] == pytest.approx(20.0)


def test_a_mid_season_trade_collapses_to_one_player() -> None:
    stats = pd.DataFrame(
        [
            _stats("g1", 2025, 1, targets=6.0, stats_player_id="NYJ_g1"),
            _stats("g1", 2025, 2, targets=10.0, stats_player_id="PIT_g1"),
        ]
    )
    roster = pd.DataFrame([_roster("g1", "WR", team="PIT")])
    scored = trailing_usage(stats, roster, 2026, 1)
    assert len(scored) == 1
    assert scored.iloc[0]["usage_score"] == pytest.approx(8.0)
    assert scored.iloc[0]["team"] == "PIT"


def test_a_player_missing_from_the_roster_is_excluded() -> None:
    stats = pd.DataFrame(
        [_stats("retired", 2025, 1, targets=20.0), _stats("g1", 2025, 1, targets=5.0)]
    )
    roster = pd.DataFrame([_roster("g1", "WR")])
    scored = trailing_usage(stats, roster, 2026, 1)
    assert list(scored["gsis_id"]) == ["g1"]


def test_cut_and_retired_roster_rows_are_excluded() -> None:
    roster = pd.DataFrame(
        [
            _roster("active", "WR", status="ACT"),
            _roster("gone", "WR", status="CUT"),
            _roster("done", "WR", status="RET"),
            _roster("hurt", "WR", status="RES"),
            _roster("unknown", "WR", status=None),
        ]
    )
    kept = set(eligible_roster(roster)["gsis_id"])
    assert kept == {"active", "hurt", "unknown"}


def test_non_prop_positions_are_excluded() -> None:
    roster = pd.DataFrame([_roster("wr", "WR"), _roster("guard", "OL"), _roster("kicker", "K")])
    assert list(eligible_roster(roster)["gsis_id"]) == ["wr"]


def test_position_floors_are_honoured_against_a_lopsided_pool() -> None:
    # 40 high-usage WRs would take every slot without the floors.
    universe = _universe(
        {"WR": 40, "QB": 5, "RB": 5, "TE": 5}, size=SMALL_SIZE, floors=SMALL_FLOORS
    )
    mix = universe["position"].value_counts().to_dict()
    assert mix == SMALL_FLOORS
    assert len(universe) == SMALL_SIZE


# WRs out-target everyone here, so every slot the floors do not reserve is
# theirs on merit.
_WR_HEAVY = {"WR": 200.0, "QB": 20.0, "RB": 20.0, "TE": 20.0}


def test_spare_slots_go_to_the_best_players_regardless_of_position() -> None:
    floors = {"QB": 1, "RB": 1, "WR": 1, "TE": 1}
    universe = _universe(
        {"WR": 20, "QB": 5, "RB": 5, "TE": 5}, bases=_WR_HEAVY, size=8, floors=floors
    )
    assert universe["position"].value_counts()["WR"] == 5


def test_caps_bound_a_position_even_when_slots_remain() -> None:
    floors = {"QB": 1, "RB": 1, "WR": 1, "TE": 1}
    universe = _universe(
        {"WR": 20, "QB": 5, "RB": 5, "TE": 5},
        bases=_WR_HEAVY,
        size=8,
        floors=floors,
        caps={"WR": 2},
    )
    assert universe["position"].value_counts()["WR"] == 2
    assert len(universe) == 8


def test_universe_rank_follows_usage_not_selection_order() -> None:
    universe = _universe(
        {"WR": 10, "QB": 3, "RB": 3, "TE": 3}, size=SMALL_SIZE, floors=SMALL_FLOORS
    )
    assert list(universe["universe_rank"]) == list(range(1, SMALL_SIZE + 1))
    assert universe["usage_score"].is_monotonic_decreasing


def test_ordering_is_deterministic_when_scores_tie() -> None:
    stats = pd.DataFrame([_stats(gsis, 2025, 1, targets=7.0) for gsis in ("wr_c", "wr_a", "wr_b")])
    roster = pd.DataFrame([_roster(gsis, "WR") for gsis in ("wr_c", "wr_a", "wr_b")])
    scored = trailing_usage(stats, roster, 2026, 1)
    picked = select_universe(scored, size=3, floors={"WR": 3})
    assert list(picked["player_id"]) == ["SEA_wr_a", "SEA_wr_b", "SEA_wr_c"]


def test_unmet_position_floor_fails_loud_naming_the_position() -> None:
    stats, roster = _pool({"WR": 20, "QB": 0, "RB": 5, "TE": 5})
    scored = trailing_usage(stats, roster, 2026, 1)
    with pytest.raises(ValueError, match=r"QB: need 1, have 0"):
        select_universe(scored, size=SMALL_SIZE, floors=SMALL_FLOORS)


def test_a_pool_too_small_to_fill_the_universe_fails_loud() -> None:
    stats, roster = _pool({"WR": 2, "QB": 1, "RB": 2, "TE": 1})
    scored = trailing_usage(stats, roster, 2026, 1)
    with pytest.raises(ValueError, match="only 6 players qualify"):
        select_universe(scored, size=20, floors=SMALL_FLOORS)


def test_default_floors_fit_the_default_universe_size() -> None:
    assert sum(DEFAULT_POSITION_FLOORS.values()) <= DEFAULT_UNIVERSE_SIZE
    universe = _universe({"WR": 80, "QB": 30, "RB": 60, "TE": 40})
    assert len(universe) == DEFAULT_UNIVERSE_SIZE
    mix = universe["position"].value_counts().to_dict()
    for position, floor in DEFAULT_POSITION_FLOORS.items():
        assert mix[position] >= floor


def test_validate_mix_rejects_floors_that_exceed_the_size() -> None:
    with pytest.raises(ValueError, match="exceeds the universe size"):
        validate_mix(5, {"QB": 3, "RB": 3}, None, ("QB", "RB", "WR", "TE"))


def test_validate_mix_rejects_a_floor_above_its_cap() -> None:
    with pytest.raises(ValueError, match="floor exceeds its cap"):
        validate_mix(10, {"WR": 5}, {"WR": 3}, ("QB", "RB", "WR", "TE"))


def test_validate_mix_rejects_an_unknown_position() -> None:
    with pytest.raises(ValueError, match="outside"):
        validate_mix(10, {"K": 1}, None, ("QB", "RB", "WR", "TE"))


def _seed_universe_db(
    db_path: Path, stats: pd.DataFrame, roster: pd.DataFrame
) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE player_stats_enhanced (
            player_id TEXT NOT NULL,
            gsis_id TEXT,
            name TEXT,
            team TEXT,
            position TEXT,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            targets REAL,
            rushing_attempts REAL,
            passing_attempts REAL
        )
        """)
    conn.execute("""
        CREATE TABLE nfl_roster_players (
            season INTEGER NOT NULL,
            gsis_id TEXT NOT NULL,
            player_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            team TEXT NOT NULL,
            position TEXT NOT NULL,
            roster_status TEXT
        )
        """)
    stats.to_sql("player_stats_enhanced", conn, if_exists="append", index=False)
    roster.to_sql("nfl_roster_players", conn, if_exists="append", index=False)
    conn.commit()
    return conn


def test_load_player_universe_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    stats, roster = _pool({"WR": 10, "QB": 3, "RB": 4, "TE": 3})
    conn = _seed_universe_db(tmp_path / "universe.db", stats, roster)
    try:
        universe = load_player_universe(2026, 1, conn=conn, size=SMALL_SIZE, floors=SMALL_FLOORS)
    finally:
        conn.close()

    assert len(universe) == SMALL_SIZE
    assert universe["position"].value_counts().to_dict() == SMALL_FLOORS
    assert universe["player_id"].is_unique
    assert list(universe["universe_rank"]) == list(range(1, SMALL_SIZE + 1))


def test_load_player_universe_without_a_roster_fails_loud(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    stats, roster = _pool({"WR": 3})
    conn = _seed_universe_db(tmp_path / "no_roster.db", stats, roster.iloc[0:0])
    try:
        with pytest.raises(ValueError, match="no rows in nfl_roster_players for season 2026"):
            load_player_universe(2026, 1, conn=conn, size=1, floors={"WR": 1})
    finally:
        conn.close()


def test_load_player_universe_without_history_fails_loud(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    stats, roster = _pool({"WR": 3})
    conn = _seed_universe_db(tmp_path / "no_history.db", stats.iloc[0:0], roster)
    try:
        with pytest.raises(ValueError, match="no player_stats_enhanced rows before 2026 week 1"):
            load_player_universe(2026, 1, conn=conn, size=1, floors={"WR": 1})
    finally:
        conn.close()
