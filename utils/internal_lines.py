"""Derive our own published prop lines from the weekly projections.

The odds pipeline can only price what a sportsbook happens to post. An
internal line has no such dependency: for every player in the week's universe
(``utils.player_universe``) we publish the number our model would hang, whether
or not any book has a market up. That gives a stable weekly product, a record
of what we thought *before* the market moved, and — once books do post — a
like-for-like comparison instead of a coverage-shaped one.

**These lines never go into ``weekly_odds``.** That table is scrape history of
what real books quoted, and ``utils.odds_quality`` exists to screen synthetic
``SimBook`` rows back out of grading precisely because writing our own numbers
there once poisoned the value path. Internal lines get their own table,
``internal_lines``, with its own primary key; nothing in the CLV or
value-ranking path reads it.

The line itself is deliberately the plainest possible transform: the
projection's mean, rounded to the nearest half unit. No shading, no vig, no
market-aware nudge. A line that is just ``mu`` is auditable — when it is wrong,
the projection is wrong, and there is no second model in between to blame.

Rounding is half-up, not Python's banker's rounding: ``round()`` would send
16.25 to 16.0 and 16.75 to 17.0 in a way nobody reading the number would
predict.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from sports.markets import get_sport
from utils.db import DBConnection, execute, executemany, read_dataframe

logger = logging.getLogger(__name__)

INTERNAL_LINES_TABLE = "internal_lines"

# Half a unit: the standard prop increment, and the smallest step that keeps a
# yardage line off a push.
LINE_INCREMENT = 0.5

SUPPORTED_MARKETS = frozenset(get_sport("nfl").markets)

INTERNAL_LINE_COLUMNS = [
    "season",
    "week",
    "player_id",
    "name",
    "team",
    "position",
    "market",
    "line",
    "mu",
    "sigma",
    "universe_rank",
    "generated_at",
    "computed_at",
]

_INT_COLUMNS = ("season", "week", "universe_rank")
_FLOAT_COLUMNS = ("line", "mu", "sigma")
_TEXT_COLUMNS = (
    "player_id",
    "name",
    "team",
    "position",
    "market",
    "generated_at",
    "computed_at",
)

_UNIVERSE_REQUIRED = ("player_id", "name", "team", "position", "universe_rank")
_PROJECTION_REQUIRED = ("season", "week", "player_id", "market", "mu", "sigma", "generated_at")

_PROJECTIONS_QUERY = """
    SELECT season, week, player_id, team, market, mu, sigma, generated_at
    FROM weekly_projections
    WHERE season = ? AND week = ?
"""

_INSERT_SQL = f"""
    INSERT INTO {INTERNAL_LINES_TABLE}
        (season, week, player_id, name, team, position, market, line, mu, sigma,
         universe_rank, generated_at, computed_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def round_to_line(mu: Any, increment: float = LINE_INCREMENT) -> float:
    """Round a projection mean to the nearest posted increment, ties upward."""
    if increment <= 0:
        raise ValueError(f"line increment must be positive, got {increment}")
    value = float(mu)
    if not math.isfinite(value):
        raise ValueError(f"cannot round a non-finite projection mean: {mu!r}")
    return round(math.floor(value / increment + 0.5) * increment, 6)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject(frame: pd.DataFrame, mask: pd.Series, problem: str) -> None:
    """Raise naming the offending rows, or return quietly when there are none."""
    if not mask.any():
        return
    offenders = frame.loc[mask, ["player_id", "market"]].head(5).to_dict("records")
    raise ValueError(
        f"weekly_projections rows are unusable as internal lines ({problem}): "
        f"{int(mask.sum())} row(s), first {offenders}"
    )


def _validate_projections(frame: pd.DataFrame) -> pd.DataFrame:
    """Fail loud on projections that cannot become a publishable line."""
    unknown = sorted(set(frame["market"].dropna().unique()) - SUPPORTED_MARKETS)
    if unknown:
        raise ValueError(
            f"weekly_projections carries markets that sports.markets does not define: "
            f"{unknown}. Register the market before publishing lines for it."
        )

    checked = frame.copy()
    checked["mu"] = pd.to_numeric(checked["mu"], errors="coerce")
    checked["sigma"] = pd.to_numeric(checked["sigma"], errors="coerce")

    usable_mu = checked["mu"].map(lambda value: pd.notna(value) and math.isfinite(float(value)))
    _reject(checked, ~usable_mu, "mu is missing or non-finite")
    # A negative projected total is not a low line, it is a broken model run;
    # publishing it would put an unbettable number on the card.
    _reject(checked, checked["mu"] < 0, "mu is negative")
    _reject(checked, checked["sigma"].isna() | (checked["sigma"] <= 0), "sigma is missing or <= 0")
    return checked


def build_internal_lines(
    universe: pd.DataFrame,
    projections: pd.DataFrame,
    *,
    increment: float = LINE_INCREMENT,
    computed_at: Optional[str] = None,
) -> pd.DataFrame:
    """Join the week's universe to its projections and mint one line per market.

    Pure: no database access.

    Args:
        universe: Output of :func:`utils.player_universe.load_player_universe`.
            ``player_id`` here is the roster identifier, which is the key
            ``weekly_projections`` is written under.
        projections: ``weekly_projections`` rows for one ``(season, week)``.
        increment: Posted line granularity.
        computed_at: ISO timestamp stamped on every row; defaults to now. The
            projection's own ``generated_at`` is carried through separately so
            a line can always be traced back to the model run behind it.

    Returns:
        A frame with :data:`INTERNAL_LINE_COLUMNS`, ordered by universe rank
        then market.

    Raises:
        ValueError: On missing columns, unregistered markets, unusable ``mu``
            or ``sigma``, or when no universe player has a projection at all.
    """
    missing_universe = [c for c in _UNIVERSE_REQUIRED if c not in universe.columns]
    if missing_universe:
        raise ValueError(f"universe frame missing required columns: {missing_universe}")
    missing_projections = [c for c in _PROJECTION_REQUIRED if c not in projections.columns]
    if missing_projections:
        raise ValueError(f"projections frame missing required columns: {missing_projections}")

    if projections.empty:
        raise ValueError(
            "no weekly_projections rows supplied; run `make week-predict` before publishing lines"
        )

    checked = _validate_projections(projections)
    roster_side = universe[list(_UNIVERSE_REQUIRED)].rename(columns={"team": "roster_team"})

    joined = checked.merge(roster_side, on="player_id", how="inner")
    if joined.empty:
        raise ValueError(
            f"no player in the {len(universe)}-player universe has a weekly_projections row "
            f"(projections cover {checked['player_id'].nunique()} players). The universe and "
            f"the projections must be keyed on the same player_id."
        )

    return _assemble(joined, universe, increment, computed_at)


def _assemble(
    joined: pd.DataFrame, universe: pd.DataFrame, increment: float, computed_at: Optional[str]
) -> pd.DataFrame:
    """Shape the joined frame into the published line rows."""
    lines = joined.copy()
    lines["line"] = [round_to_line(value, increment) for value in lines["mu"]]
    # The projection's team is the club for *this* week's game; fall back to
    # the roster's when a legacy projection row left it blank (see CLAUDE.md's
    # note on empty weekly_projections.team on 2025 rows).
    if "team" in lines.columns:
        projection_team = lines["team"].astype("string")
    else:
        projection_team = pd.Series(pd.NA, index=lines.index, dtype="string")
    usable_team = projection_team.notna() & (projection_team.str.strip() != "")
    lines["team"] = projection_team.where(usable_team, lines["roster_team"])
    lines["computed_at"] = computed_at or _utc_now_iso()
    lines["mu"] = lines["mu"].astype(float).round(6)
    lines["sigma"] = lines["sigma"].astype(float).round(6)
    lines["universe_rank"] = lines["universe_rank"].astype(int)

    ordered = lines.sort_values(["universe_rank", "market"]).reset_index(drop=True)
    logger.info(
        "built %d internal lines covering %d of %d universe players across markets %s",
        len(ordered),
        ordered["player_id"].nunique(),
        len(universe),
        sorted(ordered["market"].unique()),
    )
    return ordered[INTERNAL_LINE_COLUMNS]


def load_internal_lines(
    season: int,
    week: int,
    *,
    universe: Optional[pd.DataFrame] = None,
    conn: Optional[DBConnection] = None,
    increment: float = LINE_INCREMENT,
    computed_at: Optional[str] = None,
    **universe_kwargs: Any,
) -> pd.DataFrame:
    """Read the database and return the week's internal lines.

    ``universe`` is accepted so a caller that already selected one (or wants a
    pinned one) does not pay for the selection twice; otherwise it is built
    here with :func:`utils.player_universe.load_player_universe`.
    """
    if universe is None:
        from utils.player_universe import load_player_universe

        universe = load_player_universe(season, week, conn=conn, **universe_kwargs)
    elif universe_kwargs:
        raise ValueError(
            f"universe was supplied, so selection arguments cannot be honoured: "
            f"{sorted(universe_kwargs)}"
        )

    projections = read_dataframe(_PROJECTIONS_QUERY, (int(season), int(week)), conn=conn)
    if projections.empty:
        raise ValueError(
            f"no weekly_projections rows for season {season} week {week}; run "
            f"`make week-predict SEASON={season} WEEK={week}` first"
        )
    return build_internal_lines(universe, projections, increment=increment, computed_at=computed_at)


def _typed_rows(df: pd.DataFrame) -> list[tuple]:
    """Coerce the frame to the column types the table declares."""
    typed = df[INTERNAL_LINE_COLUMNS].copy()
    for column in _INT_COLUMNS:
        typed[column] = typed[column].astype(int)
    for column in _FLOAT_COLUMNS:
        typed[column] = typed[column].astype(float)
    for column in _TEXT_COLUMNS:
        typed[column] = typed[column].astype(str)
    return list(typed.itertuples(index=False, name=None))


def persist_internal_lines(df: pd.DataFrame, conn: DBConnection) -> int:
    """Replace the stored lines for every ``(season, week)`` present in ``df``.

    Delete-then-insert rather than an upsert: SQLite spells conflict handling
    ``ON CONFLICT`` and MySQL spells it ``ON DUPLICATE KEY UPDATE`` (punch-list
    item 16), and a weekly republication rewrites the whole slice anyway. A
    player who dropped out of the universe must not linger on last run's card.

    Args:
        df: Rows shaped by :func:`build_internal_lines`.
        conn: An open connection. The write is committed here, matching
            ``materialized_value_view``.

    Returns:
        Number of rows inserted.
    """
    missing = [c for c in INTERNAL_LINE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"internal-lines frame missing required columns: {missing}")
    if df.empty:
        raise ValueError("refusing to persist an empty internal-lines frame")

    rows = _typed_rows(df)
    slices = df[["season", "week"]].drop_duplicates().itertuples(index=False, name=None)
    for season, week in slices:
        execute(
            f"DELETE FROM {INTERNAL_LINES_TABLE} WHERE season = ? AND week = ?",
            (int(season), int(week)),
            conn=conn,
        )

    executemany(_INSERT_SQL, rows, conn=conn)
    conn.commit()

    logger.info("persisted %d internal lines to %s", len(rows), INTERNAL_LINES_TABLE)
    return len(rows)
