"""Normalize the schedule's game-context columns.

nflverse ships spread, total, weather and roof on every schedule row the
ingest already downloads, and the ingest then drops them on the floor: the
``games`` table keeps only identity and kickoff. These are the cheapest
features available — no extra request, no extra provider, no leakage, since
the closing spread and total are known before kickoff.

Two conventions in the feed are easy to get backwards, and both are encoded
here rather than left to each caller:

``spread_line`` is quoted **from the home team's perspective**, so a positive
number means the home side is favored. Storing it without that convention
recorded invites a sign flip that a model will happily learn around.

``temp`` and ``wind`` are null for indoor games. That is not missing data —
it means climate controlled. Imputing a league-average temperature into a
dome teaches the model that domes are 60 degrees and windy. ``is_indoor``
carries the distinction so a caller can fill deliberately.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

# Roof values nflverse uses for a game played out of the weather. "closed"
# is a retractable roof that was shut, which is indoors for our purposes.
INDOOR_ROOFS = frozenset({"dome", "closed"})
OUTDOOR_ROOFS = frozenset({"outdoors", "open"})

CONTEXT_COLUMNS = (
    "spread_line",
    "total_line",
    "temp",
    "wind",
    "roof",
    "surface",
    "div_game",
    "is_indoor",
)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Coerce a schedule column to float, or all-null when it is absent."""
    if column not in frame.columns:
        return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[column], errors="coerce").astype("Float64")


def _text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([None] * len(frame), index=frame.index, dtype="object")
    # Build the Series from a list rather than via .map()/.where(): both
    # coerce None back to NaN on an object column, so a blank surface would
    # read as nan and every `is None` check downstream would miss it.
    cleaned = [
        v.strip().lower() if isinstance(v, str) and v.strip() else None for v in frame[column]
    ]
    return pd.Series(cleaned, index=frame.index, dtype="object")


def is_indoor_roof(roof: Any) -> Optional[bool]:
    """Whether a roof value means the game is played out of the weather.

    Returns ``None`` for an unrecognized or absent value rather than guessing.
    A wrong guess here silently decides whether weather features apply.
    """
    if not isinstance(roof, str):
        return None
    normalized = roof.strip().lower()
    if normalized in INDOOR_ROOFS:
        return True
    if normalized in OUTDOOR_ROOFS:
        return False
    return None


def extract_game_context(schedule: pd.DataFrame) -> pd.DataFrame:
    """Pull the context columns out of a raw nflverse schedule frame.

    Args:
        schedule: A schedule frame as published by ``nflreadpy.load_schedules``.
            Missing context columns are tolerated — the feed's shape varies by
            season, and an older season lacking ``wind`` should degrade to a
            null column rather than fail the whole ingest.

    Returns:
        A frame indexed like ``schedule`` with every column in
        ``CONTEXT_COLUMNS``. ``is_indoor`` is a nullable boolean derived from
        ``roof``; ``temp`` and ``wind`` are left null for indoor games rather
        than imputed.
    """
    if schedule is None or schedule.empty:
        return pd.DataFrame(columns=list(CONTEXT_COLUMNS))

    context = pd.DataFrame(index=schedule.index)
    context["spread_line"] = _numeric(schedule, "spread_line")
    context["total_line"] = _numeric(schedule, "total_line")
    context["temp"] = _numeric(schedule, "temp")
    context["wind"] = _numeric(schedule, "wind")
    context["roof"] = _text(schedule, "roof")
    context["surface"] = _text(schedule, "surface")

    div = _numeric(schedule, "div_game")
    context["div_game"] = div.map(lambda v: None if pd.isna(v) else int(bool(v))).astype("Int64")

    context["is_indoor"] = context["roof"].map(is_indoor_roof).astype("boolean")
    return context[list(CONTEXT_COLUMNS)]


def home_favored_by(spread_line: Any) -> Optional[float]:
    """Points the *home* team is favored by, negative when it is the underdog.

    A pass-through with the convention named. nflverse quotes ``spread_line``
    from the home side already; this exists so call sites read as the
    convention rather than assuming one.
    """
    if spread_line is None or (isinstance(spread_line, float) and pd.isna(spread_line)):
        return None
    try:
        return float(spread_line)
    except (TypeError, ValueError):
        return None


def implied_team_totals(spread_line: Any, total_line: Any) -> Optional[tuple[float, float]]:
    """Split a game total into ``(home_points, away_points)``.

    The market's own view of how many points each side scores, which is a
    far better prior for volume-driven props than a team's season average.
    Given a total ``T`` and a home spread ``S``, the implied scores are
    ``T/2 + S/2`` and ``T/2 - S/2``.

    Returns ``None`` when either input is missing, since a half-known market
    is not a market.
    """
    spread = home_favored_by(spread_line)
    total = home_favored_by(total_line)
    if spread is None or total is None:
        return None
    return (total / 2.0 + spread / 2.0, total / 2.0 - spread / 2.0)
