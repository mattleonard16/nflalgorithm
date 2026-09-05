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

GAME_CONTEXT_COLUMNS = (
    "spread_margin",
    "implied_team_total",
    "game_total",
    "wind_speed",
    "temperature",
    "is_indoor",
    "div_game",
)

GAME_CONTEXT_DEFAULTS: dict[str, float | int] = {
    "spread_margin": 0.0,
    "implied_team_total": 22.5,
    "game_total": 45.0,
    "wind_speed": 0.0,
    "temperature": 70.0,
    "is_indoor": 0,
    "div_game": 0,
}


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


def _safe_float(val: Any, default: float) -> float:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    try:
        f = float(val)
        return default if pd.isna(f) else f
    except (TypeError, ValueError):
        return default


def attach_game_context_to_player_frame(
    player_df: pd.DataFrame, games_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Attach pregame contest context to player rows matching (season, week, team).

    Extracts team-specific perspective:
    - ``spread_margin``: points favored by (>0 favored, <0 underdog)
    - ``implied_team_total``: Vegas implied points scored
    - ``game_total``: Vegas over/under line
    - ``wind_speed``: 0.0 if indoor, else wind (or outdoor default)
    - ``temperature``: 70.0 if indoor, else temp (or outdoor default)
    - ``is_indoor``: 1 if dome/closed, else 0
    - ``div_game``: 1 if divisional, else 0
    """
    df = player_df.copy()
    if df.empty:
        for col, default in GAME_CONTEXT_DEFAULTS.items():
            if col not in df.columns:
                df[col] = pd.Series(dtype=type(default))
        return df

    if games_df is None or games_df.empty:
        for col, default in GAME_CONTEXT_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        return df

    records: list[dict[str, Any]] = []
    for row in games_df.itertuples(index=False):
        row_dict = row._asdict() if hasattr(row, "_asdict") else dict(zip(games_df.columns, row))
        season = row_dict.get("season")
        week = row_dict.get("week")
        home = row_dict.get("home_team")
        away = row_dict.get("away_team")
        if season is None or week is None or not home or not away:
            continue

        try:
            s_val = int(season)
            w_val = int(week)
        except (TypeError, ValueError):
            continue

        spread = home_favored_by(row_dict.get("spread_line"))
        total = home_favored_by(row_dict.get("total_line"))
        roof = row_dict.get("roof")
        indoor = is_indoor_roof(roof)
        indoor_flag = 1 if indoor is True else 0

        temp = row_dict.get("temp")
        wind = row_dict.get("wind")
        div = row_dict.get("div_game")
        div_flag = 1 if (div is not None and not pd.isna(div) and int(bool(div)) == 1) else 0

        if indoor_flag == 1:
            eff_temp = 70.0
            eff_wind = 0.0
        else:
            eff_temp = _safe_float(temp, 65.0)
            eff_wind = _safe_float(wind, 5.0)

        game_tot = total if total is not None else 45.0

        # Home team perspective
        home_spread = spread if spread is not None else 0.0
        home_implied = game_tot / 2.0 + home_spread / 2.0
        records.append({
            "season": s_val,
            "week": w_val,
            "team": str(home).strip().upper(),
            "spread_margin": float(home_spread),
            "implied_team_total": float(home_implied),
            "game_total": float(game_tot),
            "wind_speed": float(eff_wind),
            "temperature": float(eff_temp),
            "is_indoor": int(indoor_flag),
            "div_game": int(div_flag),
        })

        # Away team perspective
        away_spread = -spread if spread is not None else 0.0
        away_implied = game_tot / 2.0 + away_spread / 2.0
        records.append({
            "season": s_val,
            "week": w_val,
            "team": str(away).strip().upper(),
            "spread_margin": float(away_spread),
            "implied_team_total": float(away_implied),
            "game_total": float(game_tot),
            "wind_speed": float(eff_wind),
            "temperature": float(eff_temp),
            "is_indoor": int(indoor_flag),
            "div_game": int(div_flag),
        })

    if not records:
        for col, default in GAME_CONTEXT_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        return df

    context_df = pd.DataFrame(records).drop_duplicates(subset=["season", "week", "team"], keep="last")

    if "team" in df.columns and "season" in df.columns and "week" in df.columns:
        # The merge joins on the _clean_* helpers (not on "team" itself), so
        # the player's own "team" column survives untouched and the schedule's
        # copy lands as "team_context_dup" for disposal below. Do NOT restore
        # via `df["team"] = <saved series>`: after the merge the frame carries
        # a fresh RangeIndex, and assigning a series saved under the caller's
        # original (e.g. non-default) index realigns by label — wiping every
        # team to NaN and silently breaking the odds join downstream.
        df["_clean_team"] = df["team"].astype(str).str.strip().str.upper()
        df["_clean_season"] = pd.to_numeric(df["season"], errors="coerce").fillna(0).astype(int)
        df["_clean_week"] = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)

        cols_to_drop = [c for c in GAME_CONTEXT_COLUMNS if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        df = df.merge(
            context_df,
            left_on=["_clean_season", "_clean_week", "_clean_team"],
            right_on=["season", "week", "team"],
            how="left",
            suffixes=("", "_context_dup"),
        )
        df = df.drop(columns=["_clean_team", "_clean_season", "_clean_week"], errors="ignore")
        if "team_context_dup" in df.columns:
            df = df.drop(columns=["team_context_dup"], errors="ignore")
        if "season_context_dup" in df.columns:
            df = df.drop(columns=["season_context_dup"], errors="ignore")
        if "week_context_dup" in df.columns:
            df = df.drop(columns=["week_context_dup"], errors="ignore")

    for col, default in GAME_CONTEXT_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    return df
