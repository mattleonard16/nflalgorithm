"""Select the consistent top-N player universe we publish internal lines for.

A lines product needs a *stable, defensible* universe. If the set of players we
price is whatever The Odds API happened to return this week, coverage swings
with a vendor's mood and week-over-week comparisons stop meaning anything.
This module picks the universe from our own data instead: trailing opportunity
volume out of ``player_stats_enhanced``, verified against the current season's
roster.

Three decisions are encoded here rather than left to the caller.

**Opportunity, not production.** A player is ranked on chances (pass attempts
for a QB, carries plus targets for a RB/WR/TE), not on yards. Yardage is
opportunity times efficiency, and efficiency is the noisy half. The volume half
is what actually persists week to week, and it is what makes a prop liquid.

**The identifier bridge is ``gsis_id``, not ``player_id``.** These two tables
mint ``player_id`` differently — ``player_stats_enhanced`` abbreviates the
first name (``ARI_e_higgins``) while ``nfl_roster_players`` spells it out
(``ARI_elijah_higgins``) — and the team prefix changes when a player moves. On
the live database, joining 2025 stats to the 2026 roster on ``player_id``
matches **1** player; joining on ``gsis_id`` matches 563. Only ``gsis_id``
survives both a rename scheme and a trade. The roster's ``player_id`` is what
comes back out, because that is the key ``weekly_projections`` uses.

**Trailing window walks backwards across the season boundary.** The window is
the last ``window`` weeks the player actually played, taken from the target
season first and topped up from the prior season. Week 1 therefore sources
entirely from last season — automatically, not as a special case — and week 3
blends two current weeks with four from last year. A universe that came back
short in week 1 because "the season has no stats yet" would be a silent
failure; this module raises instead.

Position floors keep the card from collapsing into one position group. The
defaults (QB 10, RB 25, WR 35, TE 12) reserve 82 of 100 slots and leave 18 to
the best players regardless of position, which also bounds any single position
at its floor plus those 18.
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional, Sequence

import pandas as pd

from utils.db import DBConnection, read_dataframe

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE_SIZE = 100

# Weeks of *played* football behind the target week that feed the ranking.
DEFAULT_TRAILING_WEEKS = 6

# Positions eligible for a prop card. Everything else on an NFL roster
# (OL/DL/DB/LB/K/P/LS) has no player-prop market in `sports/markets.py`.
DEFAULT_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE")

# Minimum slots reserved per position. Sum must not exceed the universe size;
# the remainder goes to the best players regardless of position.
DEFAULT_POSITION_FLOORS: Mapping[str, int] = {"QB": 10, "RB": 25, "WR": 35, "TE": 12}

# Maximum slots per position. QBs dominate raw opportunity counts (30+ dropbacks
# a game), so an uncapped top-100 fills with ~28 quarterbacks — one per team is
# 32, and a prop card doesn't need backups. 14 keeps every likely starter with
# a real passing market while returning the freed slots to skill positions.
DEFAULT_POSITION_CAPS: Mapping[str, int] = {"QB": 14}

# Roster rows that describe a player who will not take a snap. Kept
# configurable: "RES" (reserve/injured) is deliberately *not* excluded by
# default because preseason reserve designations churn, and dropping those
# players would shrink the universe for a reason that resolves itself.
DEFAULT_EXCLUDED_ROSTER_STATUSES: frozenset[str] = frozenset({"CUT", "RET"})

# What counts as an opportunity, per position.
OPPORTUNITY_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "QB": ("passing_attempts",),
    "RB": ("rushing_attempts", "targets"),
    "WR": ("rushing_attempts", "targets"),
    "TE": ("rushing_attempts", "targets"),
}

# Ranking order. Score first; the rest exist so that two players with an
# identical mean never swap places between runs.
_RANK_COLUMNS = ["usage_score", "total_opportunities", "games_in_window", "player_id"]
_RANK_ASCENDING = [False, False, False, True]

UNIVERSE_COLUMNS = [
    "universe_rank",
    "player_id",
    "gsis_id",
    "stats_player_id",
    "name",
    "team",
    "position",
    "usage_score",
    "total_opportunities",
    "games_in_window",
]

_STATS_QUERY = """
    SELECT player_id, gsis_id, name, team, position, season, week,
           targets, rushing_attempts, passing_attempts
    FROM player_stats_enhanced
    WHERE (season = ? AND week < ?) OR season = ?
"""

_ROSTER_QUERY = """
    SELECT season, gsis_id, player_id, player_name, team, position, roster_status
    FROM nfl_roster_players
    WHERE season = ?
"""


def validate_mix(
    size: int,
    floors: Mapping[str, int],
    caps: Optional[Mapping[str, int]],
    positions: Sequence[str],
) -> None:
    """Reject an impossible position mix before any data is read."""
    if size <= 0:
        raise ValueError(f"universe size must be positive, got {size}")

    unknown = sorted(set(floors) - set(positions))
    if unknown:
        raise ValueError(
            f"position floors reference positions outside {list(positions)}: {unknown}"
        )

    total_floor = sum(floors.values())
    if total_floor > size:
        raise ValueError(
            f"position floors sum to {total_floor}, which exceeds the universe size {size}: {dict(floors)}"
        )

    if caps is None:
        return
    unknown_caps = sorted(set(caps) - set(positions))
    if unknown_caps:
        raise ValueError(
            f"position caps reference positions outside {list(positions)}: {unknown_caps}"
        )
    conflicting = {
        pos: (floors[pos], caps[pos]) for pos in set(floors) & set(caps) if floors[pos] > caps[pos]
    }
    if conflicting:
        raise ValueError(f"position floor exceeds its cap for {conflicting}")
    if sum(caps.get(pos, size) for pos in positions) < size:
        raise ValueError(f"position caps {dict(caps)} cannot fill a universe of {size}")


def eligible_roster(
    roster: pd.DataFrame,
    *,
    positions: Sequence[str] = DEFAULT_POSITIONS,
    excluded_statuses: frozenset[str] = DEFAULT_EXCLUDED_ROSTER_STATUSES,
) -> pd.DataFrame:
    """Reduce a season's roster to one row per prop-eligible player.

    The roster is the authority on both position and team for the target
    season: a stats row from last year may have a player at a different club,
    or listed at a position he has since moved off.

    A null ``roster_status`` is kept. Some historical roster feeds do not carry
    the column at all, and dropping every such player would silently empty the
    universe for those seasons.
    """
    required = ("gsis_id", "player_id", "player_name", "team", "position")
    missing = [column for column in required if column not in roster.columns]
    if missing:
        raise ValueError(f"roster frame missing required columns: {missing}")

    frame = roster.copy()
    frame["position"] = frame["position"].astype("string").str.strip().str.upper()
    frame = frame.loc[frame["position"].isin(list(positions))]

    if "roster_status" in frame.columns and excluded_statuses:
        status = frame["roster_status"].astype("string").str.strip().str.upper()
        frame = frame.loc[status.isna() | ~status.isin(list(excluded_statuses))]

    frame = frame.loc[frame["gsis_id"].notna() & (frame["gsis_id"].astype("string") != "")]
    # PRIMARY KEY (season, gsis_id) makes duplicates impossible today; the sort
    # keeps the outcome deterministic if that ever stops being true.
    frame = frame.sort_values(["gsis_id", "player_id"]).drop_duplicates("gsis_id", keep="first")

    return frame[["gsis_id", "player_id", "player_name", "team", "position"]].reset_index(drop=True)


def _opportunity(frame: pd.DataFrame) -> pd.Series:
    """Opportunities in one player-week, by the row's roster position."""
    needed = sorted({column for columns in OPPORTUNITY_COLUMNS.values() for column in columns})
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise ValueError(f"stats frame missing opportunity columns: {missing}")

    opportunity = pd.Series(0.0, index=frame.index, dtype="float64")
    for position, columns in OPPORTUNITY_COLUMNS.items():
        mask = frame["position"] == position
        if not mask.any():
            continue
        total = pd.Series(0.0, index=frame.index[mask], dtype="float64")
        for column in columns:
            total = total + pd.to_numeric(frame.loc[mask, column], errors="coerce").fillna(0.0)
        opportunity.loc[mask] = total
    return opportunity


def trailing_usage(
    stats: pd.DataFrame,
    roster: pd.DataFrame,
    season: int,
    week: int,
    *,
    window: int = DEFAULT_TRAILING_WEEKS,
    positions: Sequence[str] = DEFAULT_POSITIONS,
    excluded_statuses: frozenset[str] = DEFAULT_EXCLUDED_ROSTER_STATUSES,
) -> pd.DataFrame:
    """Score every roster-verified player on trailing opportunity per game.

    Pure. ``stats`` may contain any weeks; only player-weeks strictly before
    ``(season, week)`` are used, newest first, capped at ``window`` per player.

    Args:
        stats: ``player_stats_enhanced`` rows.
        roster: ``nfl_roster_players`` rows for the target season; reduced by
            :func:`eligible_roster`. The join is on ``gsis_id``, and
            team/position/name are taken from the roster — see the module
            docstring for why the ``player_id`` columns in these two tables
            cannot be joined to each other.
        season: Target season.
        week: Target week; the window ends at ``week - 1``.
        window: Maximum played weeks per player.
        positions: Prop-eligible positions.
        excluded_statuses: Roster statuses treated as "will not play".

    Returns:
        One row per player with ``usage_score`` (mean opportunities per played
        game in the window), ``total_opportunities``, ``games_in_window``, and
        the roster identity columns. Empty when nothing qualifies — the caller
        decides whether that is fatal.
    """
    if window <= 0:
        raise ValueError(f"trailing window must be positive, got {window}")

    verified = eligible_roster(roster, positions=positions, excluded_statuses=excluded_statuses)
    if stats is None or stats.empty or verified.empty:
        return pd.DataFrame(columns=[*UNIVERSE_COLUMNS[1:]])

    for column in ("player_id", "gsis_id", "season", "week"):
        if column not in stats.columns:
            raise ValueError(f"stats frame missing required column: {column}")

    frame = stats.loc[stats["gsis_id"].notna() & (stats["gsis_id"].astype("string") != "")].copy()
    # Identity for the target week is the roster's, not the stats row's: a
    # stats row from last season carries the club the player has since left.
    frame = frame.drop(columns=[c for c in ("position", "team", "name") if c in frame.columns])
    frame = frame.rename(columns={"player_id": "stats_player_id"})
    frame = frame.merge(verified, on="gsis_id", how="inner")
    frame = frame.rename(columns={"player_name": "name"})

    frame["season"] = pd.to_numeric(frame["season"], errors="coerce").astype("Int64")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce").astype("Int64")
    frame = frame.loc[frame["season"].notna() & frame["week"].notna()]
    before_target = (frame["season"] < season) | (
        (frame["season"] == season) & (frame["week"] < week)
    )
    frame = frame.loc[before_target]
    if frame.empty:
        return pd.DataFrame(columns=[*UNIVERSE_COLUMNS[1:]])

    frame["opportunity"] = _opportunity(frame)
    return _score_window(frame, window)


def _score_window(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Keep each player's most recent ``window`` player-weeks and aggregate."""
    # season*100 + week orders periods across the season boundary in one key.
    frame = frame.assign(
        _period=frame["season"].astype("int64") * 100 + frame["week"].astype("int64")
    )
    frame = frame.sort_values(["gsis_id", "_period"], ascending=[True, False])
    frame = frame.loc[frame.groupby("gsis_id").cumcount() < window]

    grouped = frame.groupby("gsis_id", as_index=False).agg(
        usage_score=("opportunity", "mean"),
        total_opportunities=("opportunity", "sum"),
        games_in_window=("opportunity", "size"),
    )
    # Sorted newest-first above, so "first" is the most recent stats identity.
    identity = frame.drop_duplicates("gsis_id", keep="first")[
        ["gsis_id", "player_id", "stats_player_id", "name", "team", "position"]
    ]

    scored = grouped.merge(identity, on="gsis_id", how="left")
    scored["usage_score"] = scored["usage_score"].round(6)
    scored["total_opportunities"] = scored["total_opportunities"].round(6)
    return scored[[*UNIVERSE_COLUMNS[1:]]].reset_index(drop=True)


def select_universe(
    scored: pd.DataFrame,
    *,
    size: int = DEFAULT_UNIVERSE_SIZE,
    floors: Mapping[str, int] = DEFAULT_POSITION_FLOORS,
    caps: Optional[Mapping[str, int]] = DEFAULT_POSITION_CAPS,
    positions: Sequence[str] = DEFAULT_POSITIONS,
) -> pd.DataFrame:
    """Take the top ``size`` players, honouring position floors and caps.

    Floors are filled first from each position's own ranking, then the
    remaining slots go to the best unselected players overall. ``universe_rank``
    is assigned by score afterwards, so the rank column reflects usage rather
    than the order the mix rules happened to pick players in.

    Raises:
        ValueError: When a position cannot meet its floor, or the pool cannot
            fill ``size``. The message carries per-position availability, since
            "the universe came back short" is useless without knowing where.
    """
    validate_mix(size, floors, caps, positions)

    ranked = scored.sort_values(_RANK_COLUMNS, ascending=_RANK_ASCENDING).reset_index(drop=True)
    available = ranked["position"].value_counts().to_dict() if not ranked.empty else {}

    short = {
        pos: (floor, int(available.get(pos, 0)))
        for pos, floor in floors.items()
        if int(available.get(pos, 0)) < floor
    }
    if short:
        detail = ", ".join(
            f"{pos}: need {need}, have {have}" for pos, (need, have) in sorted(short.items())
        )
        raise ValueError(
            f"cannot build a {size}-player universe: position floor unmet ({detail}). "
            f"Pool holds {len(ranked)} roster-verified players with trailing usage."
        )

    selected: list[int] = []
    counts: dict[str, int] = {pos: 0 for pos in positions}
    for position in sorted(floors):
        pool = ranked.index[ranked["position"] == position][: floors[position]]
        selected.extend(int(i) for i in pool)
        counts[position] += len(pool)

    chosen = set(selected)
    for index, position in ranked["position"].items():
        if len(selected) >= size:
            break
        if index in chosen:
            continue
        cap = caps.get(position) if caps else None
        if cap is not None and counts.get(position, 0) >= cap:
            continue
        selected.append(int(index))
        chosen.add(index)
        counts[position] = counts.get(position, 0) + 1

    if len(selected) < size:
        raise ValueError(
            f"cannot build a {size}-player universe: only {len(selected)} players qualify "
            f"(pool {len(ranked)}, by position {available}, caps {dict(caps) if caps else 'none'})"
        )

    universe = ranked.loc[selected].sort_values(_RANK_COLUMNS, ascending=_RANK_ASCENDING)
    universe = universe.reset_index(drop=True)
    universe.insert(0, "universe_rank", range(1, len(universe) + 1))
    return universe[UNIVERSE_COLUMNS]


def load_player_universe(
    season: int,
    week: int,
    *,
    conn: Optional[DBConnection] = None,
    size: int = DEFAULT_UNIVERSE_SIZE,
    window: int = DEFAULT_TRAILING_WEEKS,
    floors: Mapping[str, int] = DEFAULT_POSITION_FLOORS,
    caps: Optional[Mapping[str, int]] = DEFAULT_POSITION_CAPS,
    positions: Sequence[str] = DEFAULT_POSITIONS,
) -> pd.DataFrame:
    """Read the database and return the top-``size`` universe for a week."""
    if int(week) < 1:
        raise ValueError(f"week must be >= 1, got {week}")
    validate_mix(size, floors, caps, positions)

    roster = read_dataframe(_ROSTER_QUERY, (int(season),), conn=conn)
    if roster.empty:
        raise ValueError(
            f"no rows in nfl_roster_players for season {season}; run `make ingest-nfl` "
            f"before selecting a player universe"
        )

    stats = read_dataframe(_STATS_QUERY, (int(season), int(week), int(season) - 1), conn=conn)
    if stats.empty:
        raise ValueError(
            f"no player_stats_enhanced rows before {season} week {week} (checked season "
            f"{season} weeks < {week} and all of season {season - 1}); the trailing-usage "
            f"window has nothing to rank"
        )

    scored = trailing_usage(
        stats, roster, int(season), int(week), window=window, positions=positions
    )
    universe = select_universe(scored, size=size, floors=floors, caps=caps, positions=positions)

    logger.info(
        "selected %d-player universe for %s week %s from %d roster-verified candidates "
        "(trailing %d weeks); position mix %s",
        len(universe),
        season,
        week,
        len(scored),
        window,
        universe["position"].value_counts().to_dict(),
    )
    return universe
