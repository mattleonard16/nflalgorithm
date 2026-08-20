"""Resolve which NFL (season, week) the pipeline should currently target.

Every weekly command in this repo takes ``SEASON`` and ``WEEK`` from a human
(``make week-predict SEASON=2026 WEEK=1``). A scheduled job has nobody to ask,
and a hardcoded week is exactly the defect Tier-0 #5 was raised for. This
module derives the answer from the schedule that is already in the database.

Definition: the upcoming week is the ``(season, week)`` of the *earliest game
whose kickoff has not happened yet*. During a week that is already underway —
Thursday night played, Sunday not — that still resolves to the current week,
because Sunday's kickoffs are still in the future. It rolls to the next week
only once every game of the current one has kicked off.

``kickoff_utc`` is stored as ISO-8601 TEXT. It is parsed with pandas rather
than compared as a string: lexical ordering equals chronological ordering only
while every writer uses the same offset and width, and
``scripts/backfill_line_accuracy.py`` is already carrying that latent bug
(punch-list item 19). One module making that assumption is a known gap; a
scheduler making it would silently run the wrong week.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, NamedTuple, Optional

import pandas as pd

from utils.db import DBConnection, read_dataframe

logger = logging.getLogger(__name__)

# Only identity and kickoff are needed; the caller that wants spread/total
# reads `games` itself through utils.game_context.
_SCHEDULE_QUERY = """
    SELECT game_id, season, week, home_team, away_team, kickoff_utc
    FROM games
    WHERE kickoff_utc IS NOT NULL
"""

_REQUIRED_COLUMNS = ("season", "week", "kickoff_utc")


class UpcomingWeek(NamedTuple):
    """The next unplayed week, plus the game that pins it down."""

    season: int
    week: int
    kickoff_utc: pd.Timestamp
    game_id: Optional[str]


def as_utc(now: Any = None) -> pd.Timestamp:
    """Normalize a caller-supplied clock to an aware UTC timestamp.

    ``None`` means "right now". A naive datetime is read as UTC rather than as
    local time: the whole schedule is stored in UTC, and quietly shifting by
    the host's timezone would move the answer by a week for a Monday-night
    game.
    """
    if now is None:
        return pd.Timestamp(datetime.now(timezone.utc))

    stamp = pd.Timestamp(now)
    if pd.isna(stamp):
        raise ValueError(f"could not read {now!r} as a timestamp")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def next_week_from_schedule(games: pd.DataFrame, now: Any = None) -> UpcomingWeek:
    """Return the upcoming week from an in-memory schedule frame.

    Pure: no database access, so the resolution rule stays testable without a
    fixture database.

    Args:
        games: Rows carrying at least ``season``, ``week`` and ``kickoff_utc``.
            ``game_id`` is reported when present.
        now: Reference clock; see :func:`as_utc`.

    Returns:
        The :class:`UpcomingWeek` of the earliest kickoff at or after ``now``.

    Raises:
        ValueError: When the frame is empty, is missing required columns, has
            no parseable kickoff, or every kickoff is already in the past. Each
            case names what was actually found — a scheduler that cannot
            resolve a week must stop with a reason, not fall back to a guess.
    """
    reference = as_utc(now)

    if games is None or games.empty:
        raise ValueError(
            "cannot resolve the upcoming NFL week: the schedule frame is empty "
            "(run `make ingest-nfl` so the `games` table is populated)"
        )

    missing = [column for column in _REQUIRED_COLUMNS if column not in games.columns]
    if missing:
        raise ValueError(f"schedule frame missing required columns: {missing}")

    frame = games.copy()
    frame["_kickoff_ts"] = pd.to_datetime(
        frame["kickoff_utc"], errors="coerce", utc=True, format="mixed"
    )
    parseable = frame.loc[frame["_kickoff_ts"].notna()]
    if parseable.empty:
        samples = list(frame["kickoff_utc"].dropna().unique()[:5])
        raise ValueError(
            f"cannot resolve the upcoming NFL week: no games.kickoff_utc value parsed as a "
            f"timestamp (saw {samples})"
        )

    dropped = len(frame) - len(parseable)
    if dropped:
        logger.warning(
            "ignoring %d schedule row(s) with an unparseable kickoff_utc while resolving "
            "the upcoming week",
            dropped,
        )

    upcoming = parseable.loc[parseable["_kickoff_ts"] >= reference]
    if upcoming.empty:
        latest = parseable["_kickoff_ts"].max()
        raise ValueError(
            f"cannot resolve the upcoming NFL week: every scheduled game has already kicked "
            f"off (latest kickoff {latest.isoformat()}, now {reference.isoformat()}). Ingest "
            f"the next season's schedule before scheduling a run."
        )

    row = upcoming.loc[upcoming["_kickoff_ts"].idxmin()]
    return UpcomingWeek(
        season=int(row["season"]),
        week=int(row["week"]),
        kickoff_utc=row["_kickoff_ts"],
        game_id=str(row["game_id"]) if "game_id" in upcoming.columns else None,
    )


def resolve_current_week(now: Any = None, conn: Optional[DBConnection] = None) -> tuple[int, int]:
    """Return ``(season, week)`` of the upcoming NFL week from the database.

    The thin wrapper the Makefile and cron entries call. Detail (which game
    pins the week, when it kicks off) is available from
    :func:`resolve_upcoming_week`.
    """
    upcoming = resolve_upcoming_week(now=now, conn=conn)
    return (upcoming.season, upcoming.week)


def resolve_upcoming_week(now: Any = None, conn: Optional[DBConnection] = None) -> UpcomingWeek:
    """Load the schedule and resolve the upcoming week, with its pinning game."""
    schedule = read_dataframe(_SCHEDULE_QUERY, conn=conn)
    upcoming = next_week_from_schedule(schedule, now=now)
    logger.info(
        "resolved upcoming NFL week: %s week %s (first kickoff %s, game %s)",
        upcoming.season,
        upcoming.week,
        upcoming.kickoff_utc.isoformat(),
        upcoming.game_id,
    )
    return upcoming
