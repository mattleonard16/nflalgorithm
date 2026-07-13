#!/usr/bin/env python3
"""
Ingest Real NFL Data from nfl_data_py
=====================================

Fetches actual NFL weekly stats and populates the database for accurate predictions.
This replaces synthetic data with real player performance data.

Usage:
    python scripts/ingest_real_nfl_data.py --season 2024 --through-week 12
"""

import argparse
import logging

# Add project root to path
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import nflreadpy as nfl
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from utils.db import (
    execute,
    executemany,
    fetchone,
    get_connection,
    is_sqlite_connection,
    read_dataframe,
)
from utils.player_id_utils import canonicalize_team, make_player_id

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

GAME_COLUMNS = (
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "kickoff_utc",
    "game_date",
    "venue",
)

# A season feed must represent essentially the whole league before it can
# authoritatively remove players that disappeared from the source snapshot.
NFL_TEAM_COUNT = 32
MIN_PLAYERS_PER_TEAM_FOR_AUTHORITATIVE_ROSTER = 40
MIN_EXISTING_ROSTER_RETENTION = 0.90

ROLE_PRIORS: dict[str, dict[str, float]] = {
    "QB": {
        "snap_percentage": 75.0,
        "rushing_attempts": 3.0,
        "targets": 0.0,
        "passing_attempts": 28.0,
        "target_share": 0.0,
        "air_yards": 0.0,
        "yac_yards": 0.0,
        "red_zone_touches": 0.5,
    },
    "RB": {
        "snap_percentage": 38.0,
        "rushing_attempts": 8.0,
        "targets": 2.5,
        "passing_attempts": 0.0,
        "target_share": 0.08,
        "air_yards": 5.0,
        "yac_yards": 18.0,
        "red_zone_touches": 1.5,
    },
    "WR": {
        "snap_percentage": 55.0,
        "rushing_attempts": 0.3,
        "targets": 4.5,
        "passing_attempts": 0.0,
        "target_share": 0.14,
        "air_yards": 52.0,
        "yac_yards": 18.0,
        "red_zone_touches": 0.8,
    },
    "TE": {
        "snap_percentage": 52.0,
        "rushing_attempts": 0.0,
        "targets": 3.5,
        "passing_attempts": 0.0,
        "target_share": 0.11,
        "air_yards": 30.0,
        "yac_yards": 14.0,
        "red_zone_touches": 0.8,
    },
    "FB": {
        "snap_percentage": 22.0,
        "rushing_attempts": 1.0,
        "targets": 0.8,
        "passing_attempts": 0.0,
        "target_share": 0.03,
        "air_yards": 3.0,
        "yac_yards": 5.0,
        "red_zone_touches": 0.4,
    },
}


def default_history_seasons(target_season: int) -> list[int]:
    """Return the configured number of immediately preceding NFL seasons."""
    history_depth = max(1, len(config.pipeline.default_seasons))
    return list(range(target_season - history_depth, target_season))


def _load_nflverse_by_season(
    loader: Callable[[List[int]], Any],
    seasons: Sequence[int],
    dataset_name: str,
    *,
    optional_missing_seasons: Sequence[int] = (),
    suppress_errors: bool = False,
) -> pd.DataFrame:
    """Load each season independently so an unpublished feed does not hide history."""
    frames: list[pd.DataFrame] = []
    optional_missing = set(optional_missing_seasons)
    for season in dict.fromkeys(seasons):
        try:
            frame = loader([season]).to_pandas()
        except Exception as exc:
            missing_optional_feed = season in optional_missing and _is_missing_feed_error(exc)
            if suppress_errors or missing_optional_feed:
                logger.warning("Could not fetch %s for %s: %s", dataset_name, season, exc)
                continue
            raise
        if not frame.empty:
            if "season" not in frame.columns:
                frame = frame.copy()
                frame["season"] = season
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _is_missing_feed_error(exc: Exception) -> bool:
    """Return whether nflreadpy failed because a season artifact is not published yet."""
    if isinstance(exc, FileNotFoundError):
        return True

    current: BaseException | None = exc
    while current is not None:
        response = getattr(current, "response", None)
        if getattr(response, "status_code", None) == 404:
            return True
        current = current.__cause__
    return False


def fetch_weekly_stats(seasons: List[int]) -> pd.DataFrame:
    """Fetch weekly player stats from nflreadpy (nflverse)."""
    logger.info(f"Fetching weekly data for seasons: {seasons}")
    current_season = nfl.get_current_season(roster=True)
    weekly = _load_nflverse_by_season(
        nfl.load_player_stats,
        seasons,
        "weekly player stats",
        optional_missing_seasons=[current_season],
    )
    logger.info(f"Fetched {len(weekly)} total player-week rows")
    return weekly


def fetch_snap_counts(seasons: List[int]) -> pd.DataFrame:
    """Fetch snap count data for usage metrics."""
    logger.info(f"Fetching snap counts for seasons: {seasons}")
    snaps = _load_nflverse_by_season(
        nfl.load_snap_counts, seasons, "snap counts", suppress_errors=True
    )
    logger.info(f"Fetched {len(snaps)} snap count rows")
    return snaps


def fetch_rosters(seasons: List[int]) -> pd.DataFrame:
    """Fetch rosters from nflreadpy.

    Returns DataFrame with (at least) gsis_id, full_name/player_name, team,
    season, birth_date. An unpublished current-season feed is treated as empty;
    other failures propagate so production does not silently lose roster context.
    """
    logger.info(f"Fetching rosters for seasons: {seasons}")
    current_season = nfl.get_current_season(roster=True)
    rosters = _load_nflverse_by_season(
        nfl.load_rosters,
        seasons,
        "rosters",
        optional_missing_seasons=[current_season],
    )
    logger.info(f"Fetched {len(rosters)} roster rows")
    return rosters


def fetch_weekly_rosters(seasons: List[int]) -> pd.DataFrame:
    """Fetch week-versioned roster status and experience fields."""
    current_season = nfl.get_current_season(roster=True)
    return _load_nflverse_by_season(
        nfl.load_rosters_weekly,
        seasons,
        "weekly rosters",
        optional_missing_seasons=[current_season],
    )


def fetch_depth_charts(seasons: List[int]) -> pd.DataFrame:
    """Fetch depth position/rank context and retain the requested season."""
    current_season = nfl.get_current_season(roster=True)
    return _load_nflverse_by_season(
        nfl.load_depth_charts,
        seasons,
        "depth charts",
        optional_missing_seasons=[current_season],
    )


def fetch_injuries(seasons: List[int]) -> pd.DataFrame:
    """Fetch report and practice availability context."""
    current_season = nfl.get_current_season(roster=True)
    return _load_nflverse_by_season(
        nfl.load_injuries,
        seasons,
        "injuries",
        optional_missing_seasons=[current_season],
    )


def fetch_schedules(seasons: List[int]) -> pd.DataFrame:
    """Fetch game schedules from nflreadpy.

    Returns DataFrame with season, week, home_team, away_team, gameday.
    An unpublished current-season feed is treated as empty; other failures
    propagate so production does not silently lose schedule context.
    """
    logger.info(f"Fetching schedules for seasons: {seasons}")
    current_season = nfl.get_current_season(roster=True)
    schedule = _load_nflverse_by_season(
        nfl.load_schedules,
        seasons,
        "schedules",
        optional_missing_seasons=[current_season],
    )
    logger.info(f"Fetched {len(schedule)} schedule rows")
    return schedule


def fetch_pbp_red_zone(seasons: List[int]) -> pd.DataFrame:
    """Aggregate per-player red-zone touches from play-by-play.

    Counts plays inside the 20-yard line (yardline_100 <= 20) where the player
    was either the rusher (rush_attempt == 1) or a pass target
    (pass_attempt == 1 and receiver_player_id == player). Returns a
    DataFrame with columns (player_id, season, week, red_zone_touches).
    Empty DataFrame on failure.
    """
    logger.info(f"Fetching pbp for seasons: {seasons} (red-zone touches)")
    needed = ["season", "week", "yardline_100", "rusher_player_id", "receiver_player_id"]
    aggregates: list[pd.DataFrame] = []

    for season in dict.fromkeys(seasons):
        try:
            pbp = nfl.load_pbp([season])
        except Exception as exc:
            logger.warning("Could not fetch play-by-play for %s: %s", season, exc)
            continue

        missing = set(needed) - set(pbp.columns)
        if missing:
            logger.warning(
                "pbp for %s missing columns %s — red_zone_touches falls back to derived formula.",
                season,
                sorted(missing),
            )
            continue

        # PBP is the largest nflverse feed. Trim it in Polars before converting
        # the small red-zone slice to pandas.
        red_zone = pbp.filter(pbp["yardline_100"] <= 20).select(needed).to_pandas()
        rushers = red_zone.dropna(subset=["rusher_player_id"])[
            ["season", "week", "rusher_player_id"]
        ].rename(columns={"rusher_player_id": "player_gsis_id"})
        receivers = red_zone.dropna(subset=["receiver_player_id"])[
            ["season", "week", "receiver_player_id"]
        ].rename(columns={"receiver_player_id": "player_gsis_id"})
        touches = pd.concat([rushers, receivers], ignore_index=True)
        if touches.empty:
            continue
        aggregates.append(
            touches.groupby(["player_gsis_id", "season", "week"], as_index=False)
            .size()
            .rename(columns={"size": "red_zone_touches"})
        )

    if not aggregates:
        return pd.DataFrame()
    result = pd.concat(aggregates, ignore_index=True)
    logger.info("Aggregated %d player-week red-zone touch rows.", len(result))
    return result


def _normalize_name_for_merge(name: str) -> str:
    """Strip punctuation, lowercase, collapse whitespace.

    nflreadpy weekly stats expose `player_name` as "A.Rodgers" and
    `player_display_name` as "Aaron Rodgers". snap_counts exposes
    `player` as "Aaron Rodgers". Normalizing both sides to lowercased
    alphanumerics lets the join survive punctuation/suffix drift
    (Jr/Sr/III, hyphenation, apostrophes).
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = str(name).lower()
    # Drop common suffixes after a space
    for suf in (" jr", " sr", " ii", " iii", " iv", " v"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    # Keep only letters/digits
    return "".join(ch for ch in s if ch.isalnum())


def _merge_snap_counts(df: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """Left-join snap counts into stats df.

    Keys on (normalized full name, season, week, team) using
    canonicalized team and a punctuation-stripped lowercased name.
    Stats `player_display_name` (preferred) or `player_name` is matched
    against snap_counts `player`. Aggregates per key with
    sum(offense_snaps) and max(offense_pct) to absorb duplicate rows.

    Returns df with ``snap_count`` and ``snap_percentage`` (0–100).
    Unmatched rows fall back to 0 — the schema declares both NOT NULL
    DEFAULT 0, and NULL binds would otherwise fail insert.
    """
    df = df.copy()
    df["snap_count"] = 0.0
    df["snap_percentage"] = 0.0

    if snaps is None or snaps.empty:
        logger.info("No snap_counts data — snap fields default to 0.")
        return df

    needed = {"player", "season", "week", "team", "offense_snaps", "offense_pct"}
    missing = needed - set(snaps.columns)
    if missing:
        logger.warning(
            "load_snap_counts missing columns %s — snap fields default to 0.", sorted(missing)
        )
        return df

    snaps_norm = snaps[["player", "season", "week", "team", "offense_snaps", "offense_pct"]].copy()
    snaps_norm["team"] = snaps_norm["team"].apply(canonicalize_team)
    snaps_norm["_name_key"] = snaps_norm["player"].apply(_normalize_name_for_merge)
    # offense_pct from nflreadpy is a 0–1 fraction; rescale to 0–100 if so.
    pct_max = snaps_norm["offense_pct"].dropna().max()
    if pd.notna(pct_max) and pct_max <= 1.5:
        snaps_norm["offense_pct"] = snaps_norm["offense_pct"] * 100.0

    agg = snaps_norm.groupby(["_name_key", "season", "week", "team"], as_index=False).agg(
        offense_snaps=("offense_snaps", "sum"), offense_pct=("offense_pct", "max")
    )

    # Prefer player_display_name ("Aaron Rodgers") over player_name ("A.Rodgers")
    df_name_source = df.get("player_display_name")
    if df_name_source is None or df_name_source.isna().all():
        df_name_source = df["player_name"]
    df["_name_key"] = df_name_source.fillna(df["player_name"]).apply(_normalize_name_for_merge)

    merged = df.merge(
        agg,
        how="left",
        on=["_name_key", "season", "week", "team"],
    )
    merged["snap_count"] = merged["offense_snaps"].fillna(0.0)
    merged["snap_percentage"] = merged["offense_pct"].fillna(0.0)
    merged = merged.drop(columns=["offense_snaps", "offense_pct", "_name_key"])

    matched = (merged["snap_percentage"] > 0).sum()
    logger.info(
        "Snap merge: matched %d / %d player-weeks (%.1f%%).",
        int(matched),
        len(merged),
        100.0 * matched / max(1, len(merged)),
    )
    return merged


def _merge_age_from_rosters(
    df: pd.DataFrame, rosters: pd.DataFrame, game_date_col: str = "game_date"
) -> pd.DataFrame:
    """Compute player age at game_date by joining rosters on gsis_id+season.

    df must already have `gsis_id` (the nflverse player_id) and `game_date`.
    Falls back to 26 only when birth_date is unknown (existing schema requires
    NOT NULL). Logs the fallback rate so it can be monitored against the ≤2%
    acceptance criterion in FR-T0-6.
    """
    df = df.copy()
    if rosters is None or rosters.empty or "gsis_id" not in rosters.columns:
        logger.warning("No rosters or rosters missing gsis_id — age falls back to 26 for all rows.")
        df["age"] = 26
        return df
    needed = {"gsis_id", "season", "birth_date"}
    missing = needed - set(rosters.columns)
    if missing:
        logger.warning("Rosters missing columns %s — age falls back to 26.", sorted(missing))
        df["age"] = 26
        return df
    # One birth_date per gsis_id (per-season rows can repeat the same DoB).
    bd = rosters.dropna(subset=["gsis_id", "birth_date"]).drop_duplicates(
        subset=["gsis_id"], keep="last"
    )[["gsis_id", "birth_date"]]
    merged = df.merge(bd, how="left", on="gsis_id")
    game_date = pd.to_datetime(merged[game_date_col], errors="coerce")
    birth = pd.to_datetime(merged["birth_date"], errors="coerce")
    # Calendar-age at game_date (no leap-day adjustment — close enough).
    delta_days = (game_date - birth).dt.days
    age = (delta_days / 365.25).round().astype("Int64")
    # Fallback for unknown birth_date OR game_date.
    fallback_mask = age.isna()
    fallback_rate = float(fallback_mask.mean()) if len(merged) else 0.0
    logger.info(
        "Age fallback rate (real → 26): %.2f%% (%d / %d rows).",
        100.0 * fallback_rate,
        int(fallback_mask.sum()),
        len(merged),
    )
    age = age.fillna(26).astype(int)
    merged["age"] = age
    return merged.drop(columns=["birth_date"], errors="ignore")


def _merge_game_date_from_schedule(df: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Attach real `game_date` to player-week rows from schedules.

    df keys: season, week, team. Schedules: season, week, home_team, away_team,
    gameday. Player team matches either home or away. Falls back to
    f"{season}-09-01" (regular-season start) if no schedule match — same
    NOT NULL constraint applies.
    """
    df = df.copy()
    if schedule is None or schedule.empty:
        logger.warning("No schedules — game_date falls back to {season}-09-01.")
        df["game_date"] = df["season"].astype(str) + "-09-01"
        return df
    needed = {"season", "week", "home_team", "away_team", "gameday"}
    missing = needed - set(schedule.columns)
    if missing:
        logger.warning(
            "Schedule missing columns %s — game_date falls back to {season}-09-01.", sorted(missing)
        )
        df["game_date"] = df["season"].astype(str) + "-09-01"
        return df
    sched = schedule[["season", "week", "home_team", "away_team", "gameday"]].copy()
    sched["home_team"] = sched["home_team"].apply(canonicalize_team)
    sched["away_team"] = sched["away_team"].apply(canonicalize_team)
    home = sched[["season", "week", "home_team", "gameday"]].rename(columns={"home_team": "team"})
    away = sched[["season", "week", "away_team", "gameday"]].rename(columns={"away_team": "team"})
    all_teams = pd.concat([home, away], ignore_index=True).drop_duplicates(
        subset=["season", "week", "team"]
    )
    merged = df.merge(all_teams, how="left", on=["season", "week", "team"])
    fallback_mask = merged["gameday"].isna()
    if fallback_mask.any():
        logger.info(
            "game_date fallback rate: %.2f%% (%d / %d rows).",
            100.0 * float(fallback_mask.mean()),
            int(fallback_mask.sum()),
            len(merged),
        )
    merged["game_date"] = merged["gameday"].fillna(merged["season"].astype(str) + "-09-01")
    return merged.drop(columns=["gameday"])


def _merge_red_zone_from_pbp(df: pd.DataFrame, pbp_rz: pd.DataFrame) -> pd.DataFrame:
    """Replace synthetic red_zone_touches with pbp-derived counts when available.

    df must have gsis_id, season, week. Falls back to the existing derived
    formula (rushing_attempts * 0.15 + receptions * 0.1) when pbp empty.
    """
    df = df.copy()
    if pbp_rz is None or pbp_rz.empty:
        return df  # caller already populated synthetic value
    merged = df.merge(
        pbp_rz,
        how="left",
        left_on=["gsis_id", "season", "week"],
        right_on=["player_gsis_id", "season", "week"],
    )
    real_rz = (
        merged["red_zone_touches_y"]
        if "red_zone_touches_y" in merged.columns
        else merged.get("red_zone_touches")
    )
    if real_rz is None:
        return df
    # Prefer real value; fall back to synthetic where no pbp row.
    synthetic = merged.get("red_zone_touches_x", merged.get("red_zone_touches"))
    merged["red_zone_touches"] = real_rz.fillna(synthetic).astype(float)
    drop_cols = [
        c
        for c in ("red_zone_touches_x", "red_zone_touches_y", "player_gsis_id")
        if c in merged.columns
    ]
    return merged.drop(columns=drop_cols)


def transform_to_enhanced_stats(
    weekly: pd.DataFrame,
    snaps: pd.DataFrame,
    rosters: Optional[pd.DataFrame] = None,
    schedule: Optional[pd.DataFrame] = None,
    pbp_rz: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Transform nflreadpy format to player_stats_enhanced format."""
    logger.info("Transforming data to player_stats_enhanced format...")

    # Filter to skill positions
    positions = ["QB", "RB", "WR", "TE", "FB"]
    df = weekly[weekly["position"].isin(positions)].copy()

    # nflreadpy uses 'team' instead of 'recent_team'
    team_col = "team" if "team" in df.columns else "recent_team"

    # Preserve nflverse gsis_id BEFORE we overwrite player_id with our local
    # canonical ID (needed for roster/pbp joins).
    df["gsis_id"] = df["player_id"]

    # Create player_id using our convention
    df["player_id"] = df.apply(lambda r: make_player_id(r["player_name"], r[team_col]), axis=1)

    # Map columns to our schema
    df["name"] = df["player_display_name"].fillna(df["player_name"])
    df["team"] = df[team_col].apply(canonicalize_team)
    df["opponent"] = df["opponent_team"].apply(canonicalize_team)

    # Stats mapping - handle both nflreadpy and nfl_data_py column names
    df["rushing_yards"] = df["rushing_yards"].fillna(0).astype(float)
    carries_col = "carries" if "carries" in df.columns else "rushing_attempts"
    df["rushing_attempts"] = (
        df[carries_col].fillna(0).astype(float) if carries_col in df.columns else 0.0
    )
    df["receiving_yards"] = df["receiving_yards"].fillna(0).astype(float)
    df["receptions"] = df["receptions"].fillna(0).astype(float)
    df["targets"] = df["targets"].fillna(0).astype(float)
    df["passing_yards"] = df["passing_yards"].fillna(0).astype(float)
    att_col = "attempts" if "attempts" in df.columns else "passing_attempts"
    df["passing_attempts"] = df[att_col].fillna(0).astype(float) if att_col in df.columns else 0.0

    # Derived metrics
    df["target_share"] = (
        df["target_share"].fillna(0).astype(float) if "target_share" in df.columns else 0.0
    )
    air_yards_col = "receiving_air_yards" if "receiving_air_yards" in df.columns else "air_yards"
    df["air_yards"] = (
        df[air_yards_col].fillna(0).astype(float) if air_yards_col in df.columns else 0.0
    )
    yac_col = (
        "receiving_yards_after_catch"
        if "receiving_yards_after_catch" in df.columns
        else "yac_yards"
    )
    df["yac_yards"] = df[yac_col].fillna(0).astype(float) if yac_col in df.columns else 0.0

    df = _merge_snap_counts(df, snaps)

    # Real game_date from schedules (T0 #6) — must precede age so the age
    # calculation uses the actual kickoff day.
    df = _merge_game_date_from_schedule(df, schedule if schedule is not None else pd.DataFrame())

    # Real age from rosters (T0 #6) — falls back to 26 only when birth_date
    # missing (schema is NOT NULL).
    df = _merge_age_from_rosters(df, rosters if rosters is not None else pd.DataFrame())

    # Compute games played per player up to each week
    df = df.sort_values(["player_id", "season", "week"])
    df["games_played"] = df.groupby(["player_id", "season"]).cumcount() + 1

    # Synthetic red_zone_touches stays as the fallback. If pbp_rz provided,
    # _merge_red_zone_from_pbp overrides it with the real count.
    df["red_zone_touches"] = (df["rushing_attempts"] * 0.15 + df["receptions"] * 0.1).round(2)
    df = _merge_red_zone_from_pbp(df, pbp_rz if pbp_rz is not None else pd.DataFrame())
    df["game_script"] = 0.0
    df["usage_delta"] = 0.02
    df["age_curve"] = 1.0
    df["oc_change"] = 0
    df["injury_recovery"] = 0
    df["preseason_buzz"] = 0.5
    df["age_squared"] = df["age"] ** 2
    df["injury_games_missed"] = 0.0
    df["team_context_flag"] = df["targets"].apply(lambda t: "HIGH" if t >= 8 else "NEUTRAL")
    df["breakout_percentile"] = (df["target_share"] * 2).clip(0, 1)

    timestamp = datetime.now(timezone.utc).isoformat()
    df["created_at"] = timestamp
    df["updated_at"] = timestamp

    # Select final columns - match MySQL schema exactly
    final_cols = [
        "player_id",
        "gsis_id",
        "season",
        "week",
        "name",
        "team",
        "position",
        "age",
        "games_played",
        "snap_count",
        "snap_percentage",
        "rushing_yards",
        "rushing_attempts",
        "passing_yards",
        "passing_attempts",
        "receiving_yards",
        "receptions",
        "targets",
        "red_zone_touches",
        "target_share",
        "air_yards",
        "yac_yards",
        "game_script",
        "created_at",
        "updated_at",
    ]

    result = df[final_cols].copy()
    logger.info(f"Transformed {len(result)} rows for player_stats_enhanced")
    return result


def upsert_player_stats(df: pd.DataFrame) -> int:
    """Upsert player stats to database (SQLite or MySQL)."""
    if df.empty:
        return 0

    logger.info(f"Upserting {len(df)} rows to player_stats_enhanced...")

    with get_connection() as conn:
        cols = list(df.columns)
        col_list = ", ".join(cols)

        is_sqlite = is_sqlite_connection(conn)
        placeholders = ", ".join(["?"] * len(cols))

        if is_sqlite:
            # SQLite: use INSERT OR REPLACE
            sql = (
                f"INSERT OR REPLACE INTO player_stats_enhanced ({col_list}) VALUES ({placeholders})"
            )
        else:
            # MySQL: use ON DUPLICATE KEY UPDATE
            update_clause = ", ".join(
                [f"{c} = VALUES({c})" for c in cols if c not in ("player_id", "season", "week")]
            )
            sql = f"""
                INSERT INTO player_stats_enhanced ({col_list})
                VALUES ({placeholders})
                ON DUPLICATE KEY UPDATE {update_clause}
            """

        # Convert pandas NA / NaN to None so DB drivers bind NULL.
        df_for_insert = df.astype(object).where(df.notna(), None)
        rows = [tuple(row) for row in df_for_insert.values]
        executemany(sql, rows, conn=conn)
        conn.commit()

        inserted = len(rows)
        logger.info(f"Upserted {inserted} rows")
        return inserted


def create_games_from_schedule(schedule: pd.DataFrame, through_week: int) -> pd.DataFrame:
    """Normalize published nflverse schedule rows for the ``games`` table."""
    required = {"season", "week", "home_team", "away_team", "gameday"}
    if schedule is None or schedule.empty:
        return pd.DataFrame(columns=GAME_COLUMNS)
    missing = required - set(schedule.columns)
    if missing:
        raise ValueError(f"Schedule data is missing required columns: {sorted(missing)}")

    games = schedule.copy()
    if "game_type" in games.columns:
        games = games[games["game_type"].isin(["REG", "POST"])].copy()
    games["week"] = pd.to_numeric(games["week"], errors="coerce")
    games = games[games["week"].between(1, through_week)].copy()
    games = games.dropna(subset=["season", "week", "home_team", "away_team", "gameday"])
    if games.empty:
        return pd.DataFrame(columns=GAME_COLUMNS)

    games["season"] = pd.to_numeric(games["season"], errors="raise").astype(int)
    games["week"] = games["week"].astype(int)
    games["home_team"] = games["home_team"].apply(canonicalize_team)
    games["away_team"] = games["away_team"].apply(canonicalize_team)
    invalid_teams = (games["home_team"] == "") | (games["away_team"] == "")
    if invalid_teams.any():
        raise ValueError("Schedule data contains invalid home or away team codes")
    game_dates = pd.to_datetime(games["gameday"], errors="coerce")
    if game_dates.isna().any():
        raise ValueError("Schedule data contains invalid gameday values")
    games["game_date"] = game_dates.dt.strftime("%Y-%m-%d")

    if "game_id" not in games.columns:
        games["game_id"] = pd.NA
    generated_ids = games.apply(
        lambda row: (f"{row['season']}_W{row['week']}_{row['away_team']}_at_{row['home_team']}"),
        axis=1,
    )
    games["game_id"] = games["game_id"].where(games["game_id"].notna(), generated_ids)

    datetime_col = next(
        (name for name in ("start_time", "game_datetime", "datetime") if name in games.columns),
        None,
    )
    if datetime_col:
        kickoff = pd.to_datetime(games[datetime_col], errors="coerce", utc=True)
        games["kickoff_utc"] = kickoff.map(
            lambda value: value.isoformat() if pd.notna(value) else None
        )
    elif "gametime" in games.columns:
        gametimes = games["gametime"].fillna("").astype(str).str.strip()
        kickoff = pd.to_datetime(
            (games["game_date"] + " " + gametimes).where(gametimes != ""), errors="coerce"
        )
        kickoff = kickoff.dt.tz_localize(
            "America/New_York", ambiguous="NaT", nonexistent="NaT"
        ).dt.tz_convert("UTC")
        games["kickoff_utc"] = kickoff.map(
            lambda value: value.isoformat() if pd.notna(value) else None
        )
    else:
        games["kickoff_utc"] = None

    venue_col = next((name for name in ("stadium", "venue") if name in games.columns), None)
    games["venue"] = games[venue_col] if venue_col else None
    return (
        games[list(GAME_COLUMNS)]
        .drop_duplicates(subset=["game_id"], keep="last")
        .reset_index(drop=True)
    )


def upsert_games(games: pd.DataFrame) -> int:
    """Upsert normalized schedule context into the cross-backend games table."""
    if games.empty:
        return 0

    missing = set(GAME_COLUMNS) - set(games.columns)
    if missing:
        raise ValueError(f"Games data is missing required columns: {sorted(missing)}")
    games = games[list(GAME_COLUMNS)]
    columns = list(GAME_COLUMNS)
    placeholders = ", ".join(["?"] * len(columns))
    update_columns = [column for column in columns if column != "game_id"]
    with get_connection() as conn:
        is_sqlite = is_sqlite_connection(conn)
        if is_sqlite:
            updates = ", ".join(f"{column} = excluded.{column}" for column in update_columns)
            sql = (
                f"INSERT INTO games ({', '.join(columns)}) VALUES ({placeholders}) "
                f"ON CONFLICT(game_id) DO UPDATE SET {updates}"
            )
        else:
            updates = ", ".join(f"{column} = VALUES({column})" for column in update_columns)
            sql = (
                f"INSERT INTO games ({', '.join(columns)}) VALUES ({placeholders}) "
                f"ON DUPLICATE KEY UPDATE {updates}"
            )
        rows = list(
            games.astype(object).where(games.notna(), None).itertuples(index=False, name=None)
        )
        executemany(sql, rows, conn=conn)
        conn.commit()
    return len(rows)


def upsert_roster_players(rosters: pd.DataFrame) -> int:
    """Seed ``player_dim`` from a published roster before weekly stats exist."""
    if rosters is None or rosters.empty:
        return 0

    required = {"gsis_id", "position", "team", "season"}
    name_col = next(
        (name for name in ("full_name", "player_name", "football_name") if name in rosters.columns),
        None,
    )
    missing = required - set(rosters.columns)
    if missing or not name_col:
        missing_columns = sorted(missing | ({"player_name"} if not name_col else set()))
        raise ValueError(f"Roster data is missing required columns: {missing_columns}")

    source_columns = ["gsis_id", name_col, "position", "team", "season"]
    if "status" in rosters.columns:
        source_columns.append("status")
    if "week" in rosters.columns:
        source_columns.append("week")
    players = rosters[source_columns].copy()
    players = players.rename(
        columns={
            name_col: "player_name",
            "season": "last_season",
            "status": "roster_status",
            "week": "roster_week",
        }
    )
    if "roster_status" not in players.columns:
        players["roster_status"] = None
    if "roster_week" not in players.columns:
        players["roster_week"] = 0
    players = players.dropna(subset=["gsis_id", "player_name", "position", "team", "last_season"])
    players["team"] = players["team"].apply(canonicalize_team)
    players = players[players["team"] != ""].copy()
    players["player_id"] = players.apply(
        lambda row: make_player_id(row["player_name"], row["team"]), axis=1
    )
    players = players[players["player_id"] != ""].copy()
    if players.empty:
        raise ValueError("Roster data contains no usable player identities")
    players["last_season"] = pd.to_numeric(players["last_season"], errors="coerce")
    players = players.dropna(subset=["last_season"])
    players["last_season"] = players["last_season"].astype(int)
    players["roster_week"] = (
        pd.to_numeric(players["roster_week"], errors="coerce").fillna(0).astype(int)
    )
    players = players.sort_values(["last_season", "roster_week"]).drop_duplicates(
        ["last_season", "gsis_id"], keep="last"
    )
    players["updated_at"] = datetime.now(timezone.utc).isoformat()
    dim_players = players.sort_values(["last_season", "roster_week"]).drop_duplicates(
        "gsis_id", keep="last"
    )
    dim_players = dim_players.copy()
    dim_players["last_week"] = 0
    columns = [
        "player_id",
        "gsis_id",
        "player_name",
        "position",
        "team",
        "last_season",
        "last_week",
        "updated_at",
    ]
    with get_connection() as conn:
        is_sqlite = is_sqlite_connection(conn)
        placeholders = ", ".join(["?"] * len(columns))
        if is_sqlite:
            sql = f"""
                INSERT INTO player_dim ({', '.join(columns)}) VALUES ({placeholders})
                ON CONFLICT(player_id) DO UPDATE SET
                    gsis_id = CASE WHEN excluded.last_season >= player_dim.last_season
                        THEN excluded.gsis_id ELSE player_dim.gsis_id END,
                    player_name = CASE WHEN excluded.last_season >= player_dim.last_season
                        THEN excluded.player_name ELSE player_dim.player_name END,
                    position = CASE WHEN excluded.last_season >= player_dim.last_season
                        THEN excluded.position ELSE player_dim.position END,
                    team = CASE WHEN excluded.last_season >= player_dim.last_season
                        THEN excluded.team ELSE player_dim.team END,
                    last_week = CASE WHEN excluded.last_season > player_dim.last_season
                        THEN 0 ELSE player_dim.last_week END,
                    last_season = MAX(player_dim.last_season, excluded.last_season),
                    updated_at = CASE WHEN excluded.last_season >= player_dim.last_season
                        THEN excluded.updated_at ELSE player_dim.updated_at END
            """
        else:
            sql = f"""
                INSERT INTO player_dim ({', '.join(columns)}) VALUES ({placeholders})
                ON DUPLICATE KEY UPDATE
                    gsis_id = IF(VALUES(last_season) >= last_season,
                        VALUES(gsis_id), gsis_id),
                    player_name = IF(VALUES(last_season) >= last_season,
                        VALUES(player_name), player_name),
                    position = IF(VALUES(last_season) >= last_season,
                        VALUES(position), position),
                    team = IF(VALUES(last_season) >= last_season, VALUES(team), team),
                    last_week = IF(VALUES(last_season) > last_season, 0, last_week),
                    updated_at = IF(VALUES(last_season) >= last_season,
                        VALUES(updated_at), updated_at),
                    last_season = GREATEST(last_season, VALUES(last_season))
            """
        rows = list(
            dim_players[columns]
            .astype(object)
            .where(dim_players[columns].notna(), None)
            .itertuples(index=False, name=None)
        )
        executemany(sql, rows, conn=conn)

        roster_columns = [
            "last_season",
            "gsis_id",
            "player_id",
            "player_name",
            "team",
            "position",
            "roster_status",
            "roster_week",
            "updated_at",
        ]
        roster_rows = list(
            players[roster_columns]
            .astype(object)
            .where(players[roster_columns].notna(), None)
            .itertuples(index=False, name=None)
        )
        roster_placeholders = ", ".join(["?"] * len(roster_columns))
        if is_sqlite:
            roster_sql = f"""
                INSERT INTO nfl_roster_players
                    (season, gsis_id, player_id, player_name, team, position, roster_status,
                     roster_week, updated_at)
                VALUES ({roster_placeholders})
                ON CONFLICT(season, gsis_id) DO UPDATE SET
                    player_id = excluded.player_id,
                    player_name = excluded.player_name,
                    team = excluded.team,
                    position = excluded.position,
                    roster_status = excluded.roster_status,
                    roster_week = excluded.roster_week,
                    updated_at = excluded.updated_at
            """
        else:
            roster_sql = f"""
                INSERT INTO nfl_roster_players
                    (season, gsis_id, player_id, player_name, team, position, roster_status,
                     roster_week, updated_at)
                VALUES ({roster_placeholders})
                ON DUPLICATE KEY UPDATE
                    player_id = VALUES(player_id),
                    player_name = VALUES(player_name),
                    team = VALUES(team),
                    position = VALUES(position),
                    roster_status = VALUES(roster_status),
                    roster_week = VALUES(roster_week),
                    updated_at = VALUES(updated_at)
            """
        executemany(roster_sql, roster_rows, conn=conn)
        for roster_season, season_players in players.groupby("last_season"):
            team_count = season_players["team"].nunique()
            minimum_team_size = int(season_players.groupby("team")["gsis_id"].nunique().min())
            existing_row = fetchone(
                "SELECT COUNT(*) FROM nfl_roster_players WHERE season = ?",
                params=(int(roster_season),),
                conn=conn,
            )
            existing_count = int(existing_row[0]) if existing_row else 0
            incoming_count = int(season_players["gsis_id"].nunique())
            retains_existing_snapshot = (
                existing_count == 0
                or incoming_count >= existing_count * MIN_EXISTING_ROSTER_RETENTION
            )
            authoritative = (
                team_count == NFL_TEAM_COUNT
                and minimum_team_size >= MIN_PLAYERS_PER_TEAM_FOR_AUTHORITATIVE_ROSTER
                and retains_existing_snapshot
            )
            if not authoritative:
                logger.warning(
                    "Skipping stale-roster pruning for %s: teams=%s, minimum_team_size=%s, "
                    "incoming=%s, existing=%s",
                    roster_season,
                    team_count,
                    minimum_team_size,
                    incoming_count,
                    existing_count,
                )
                continue
            current_gsis_ids = season_players["gsis_id"].astype(str).unique().tolist()
            id_placeholders = ", ".join(["?"] * len(current_gsis_ids))
            execute(
                f"""
                DELETE FROM nfl_roster_players
                WHERE season = ? AND gsis_id NOT IN ({id_placeholders})
                """,
                params=(int(roster_season), *current_gsis_ids),
                conn=conn,
            )
        conn.commit()
    return len(rows)


CONTEXT_SNAPSHOT_COLUMNS = (
    "season",
    "week",
    "gsis_id",
    "player_id",
    "team",
    "position",
    "roster_status",
    "depth_position",
    "depth_rank",
    "is_starter",
    "injury_status",
    "practice_status",
    "primary_injury",
    "expected_snap_count",
    "expected_snap_percentage",
    "expected_rushing_attempts",
    "expected_targets",
    "expected_passing_attempts",
    "expected_target_share",
    "expected_air_yards",
    "expected_yac_yards",
    "expected_red_zone_touches",
    "expected_game_script",
    "is_rookie",
    "is_new_team",
    "uncertainty_multiplier",
    "prior_source",
    "source_updated_at",
    "captured_at",
)


def _latest_ewm(history: pd.DataFrame, column: str, default: float) -> float:
    if history.empty or column not in history.columns:
        return default
    values = pd.to_numeric(history[column], errors="coerce").dropna().tail(6)
    if values.empty:
        return default
    return float(values.ewm(span=3, min_periods=1).mean().iloc[-1])


def _depth_factor(depth_rank: Optional[int]) -> float:
    if depth_rank is None:
        return 1.0
    return {1: 1.0, 2: 0.70, 3: 0.45}.get(depth_rank, 0.25)


def _availability_adjustment(
    injury_status: Optional[str], practice_status: Optional[str]
) -> tuple[float, float]:
    report = (injury_status or "").strip().upper()
    practice = (practice_status or "").strip().upper()
    if report in {"OUT", "INJURED RESERVE", "IR"}:
        return 0.0, 2.0
    if report == "DOUBTFUL":
        return 0.25, 1.65
    if report == "QUESTIONABLE":
        return 0.75, 1.30
    if "DID NOT PARTICIPATE" in practice:
        return 0.65, 1.35
    if "LIMITED" in practice:
        return 0.90, 1.15
    return 1.0, 1.0


def build_player_context_snapshots(
    rosters: pd.DataFrame,
    depth_charts: pd.DataFrame,
    injuries: pd.DataFrame,
    history: pd.DataFrame,
    *,
    target_week: int,
    target_cutoffs: Optional[dict[int, str]] = None,
    captured_at: Optional[str] = None,
) -> pd.DataFrame:
    """Build causal roster-role and availability snapshots for one week."""
    if rosters is None or rosters.empty:
        return pd.DataFrame(columns=CONTEXT_SNAPSHOT_COLUMNS)
    if not 1 <= target_week <= 22:
        raise ValueError("target_week must be between 1 and 22")

    name_col = next(
        (name for name in ("full_name", "player_name", "football_name") if name in rosters),
        None,
    )
    required = {"season", "gsis_id", "team", "position"}
    if not name_col or not required.issubset(rosters.columns):
        raise ValueError("Roster context is missing identity columns")

    base = rosters.copy()
    if "week" in base.columns:
        roster_weeks = pd.to_numeric(base["week"], errors="coerce").fillna(0).astype(int)
        base = base[roster_weeks <= target_week].copy()
        base["_roster_week"] = roster_weeks[roster_weeks <= target_week]
    else:
        base["_roster_week"] = 0
    base = base.dropna(subset=["gsis_id", name_col, "team", "position", "season"])
    base["gsis_id"] = base["gsis_id"].astype(str).str.strip()
    base = base[base["gsis_id"] != ""].copy()
    base["team"] = base["team"].apply(canonicalize_team)
    base = base[base["team"] != ""].copy()
    base = base.sort_values(["season", "_roster_week"]).drop_duplicates(
        ["season", "gsis_id"], keep="last"
    )

    depth_by_player: dict[tuple[int, str], dict[str, Any]] = {}
    if depth_charts is not None and not depth_charts.empty and "gsis_id" in depth_charts:
        depth = depth_charts.copy()
        if "season" not in depth.columns:
            raise ValueError("Depth chart context must include season")
        if "dt" in depth.columns:
            depth["_depth_as_of"] = pd.to_datetime(depth["dt"], errors="coerce", utc=True)
            if target_cutoffs:
                cutoff_by_row = pd.to_datetime(
                    depth["season"].map(target_cutoffs), errors="coerce", utc=True
                )
                depth = depth[
                    cutoff_by_row.isna()
                    | (depth["_depth_as_of"].notna() & (depth["_depth_as_of"] < cutoff_by_row))
                ]
            depth = depth.sort_values("_depth_as_of")
        depth = depth.dropna(subset=["season", "gsis_id"]).drop_duplicates(
            ["season", "gsis_id"], keep="last"
        )
        depth_by_player = {
            (int(row["season"]), str(row["gsis_id"])): row.to_dict() for _, row in depth.iterrows()
        }

    injury_by_player: dict[tuple[int, str], dict[str, Any]] = {}
    if injuries is not None and not injuries.empty and "gsis_id" in injuries:
        injury = injuries.copy()
        injury["week"] = pd.to_numeric(injury["week"], errors="coerce").fillna(0).astype(int)
        injury = injury[injury["week"] <= target_week]
        injury = injury.dropna(subset=["season", "gsis_id"]).sort_values("week")
        injury = injury.drop_duplicates(["season", "gsis_id"], keep="last")
        injury_by_player = {
            (int(row["season"]), str(row["gsis_id"])): row.to_dict() for _, row in injury.iterrows()
        }

    history = history.copy() if history is not None else pd.DataFrame()
    history_by_player: dict[str, pd.DataFrame] = {}
    if not history.empty:
        history = history.dropna(subset=["gsis_id", "season", "week"])
        history["gsis_id"] = history["gsis_id"].astype(str)
        history["season"] = pd.to_numeric(history["season"], errors="raise").astype(int)
        history["week"] = pd.to_numeric(history["week"], errors="raise").astype(int)
        history = history.sort_values(["gsis_id", "season", "week"])
        history_by_player = {
            str(player_id): group for player_id, group in history.groupby("gsis_id", sort=False)
        }

    captured = captured_at or datetime.now(timezone.utc).isoformat()
    status_col = "status" if "status" in base.columns else "roster_status"
    records: list[dict[str, Any]] = []
    for _, roster_row in base.iterrows():
        season = int(roster_row["season"])
        gsis_id = str(roster_row["gsis_id"])
        team = str(roster_row["team"])
        position = str(roster_row["position"]).upper()
        prior = ROLE_PRIORS.get(position, ROLE_PRIORS["WR"])

        player_history = history_by_player.get(gsis_id, pd.DataFrame())
        if not player_history.empty:
            player_history = player_history[
                (player_history["season"] < season)
                | ((player_history["season"] == season) & (player_history["week"] < target_week))
            ].copy()

        latest_team = None
        if not player_history.empty and "team" in player_history.columns:
            latest_team = canonicalize_team(str(player_history.iloc[-1]["team"]))
        is_new_team = int(bool(latest_team and latest_team != team))
        years_exp = pd.to_numeric(pd.Series([roster_row.get("years_exp")]), errors="coerce").iloc[0]
        rookie_year = pd.to_numeric(
            pd.Series([roster_row.get("rookie_year")]), errors="coerce"
        ).iloc[0]
        is_rookie = int(
            (pd.notna(years_exp) and int(years_exp) == 0)
            or (pd.notna(rookie_year) and int(rookie_year) == season)
        )

        depth_row = depth_by_player.get((season, gsis_id), {})
        raw_depth_rank = pd.to_numeric(
            pd.Series([depth_row.get("pos_rank")]), errors="coerce"
        ).iloc[0]
        depth_rank = int(raw_depth_rank) if pd.notna(raw_depth_rank) else None
        depth_position = depth_row.get("pos_abb") or depth_row.get("pos_name")
        role_factor = _depth_factor(depth_rank)

        injury_row = injury_by_player.get((season, gsis_id), {})
        injury_status = injury_row.get("report_status")
        practice_status = injury_row.get("practice_status")
        primary_injury = injury_row.get("report_primary_injury") or injury_row.get(
            "practice_primary_injury"
        )
        availability_factor, availability_uncertainty = _availability_adjustment(
            None if pd.isna(injury_status) else str(injury_status),
            None if pd.isna(practice_status) else str(practice_status),
        )

        new_team_factor = 0.85 if is_new_team else 1.0
        prior_source = "history"
        if player_history.empty:
            prior_source = "rookie_prior" if is_rookie else "position_prior"
        elif is_new_team:
            prior_source = "new_team_history"

        def expected(column: str) -> float:
            value = _latest_ewm(player_history, column, prior[column])
            return max(0.0, value * role_factor * new_team_factor * availability_factor)

        expected_snap_percentage = expected("snap_percentage")
        uncertainty = availability_uncertainty
        if is_rookie:
            uncertainty *= 1.50
        elif player_history.empty:
            uncertainty *= 1.35
        if is_new_team:
            uncertainty *= 1.25

        roster_status = roster_row.get(status_col)
        source_updated_at = depth_row.get("dt") or captured
        records.append(
            {
                "season": season,
                "week": target_week,
                "gsis_id": gsis_id,
                "player_id": make_player_id(str(roster_row[name_col]), team),
                "team": team,
                "position": position,
                "roster_status": None if pd.isna(roster_status) else str(roster_status),
                "depth_position": None if pd.isna(depth_position) else str(depth_position),
                "depth_rank": depth_rank,
                "is_starter": int(depth_rank == 1),
                "injury_status": None if pd.isna(injury_status) else str(injury_status),
                "practice_status": None if pd.isna(practice_status) else str(practice_status),
                "primary_injury": None if pd.isna(primary_injury) else str(primary_injury),
                "expected_snap_count": expected_snap_percentage * 0.65,
                "expected_snap_percentage": expected_snap_percentage,
                "expected_rushing_attempts": expected("rushing_attempts"),
                "expected_targets": expected("targets"),
                "expected_passing_attempts": expected("passing_attempts"),
                "expected_target_share": expected("target_share"),
                "expected_air_yards": expected("air_yards"),
                "expected_yac_yards": expected("yac_yards"),
                "expected_red_zone_touches": expected("red_zone_touches"),
                "expected_game_script": 0.0,
                "is_rookie": is_rookie,
                "is_new_team": is_new_team,
                "uncertainty_multiplier": max(1.0, uncertainty),
                "prior_source": prior_source,
                "source_updated_at": str(source_updated_at),
                "captured_at": captured,
            }
        )

    return pd.DataFrame.from_records(records, columns=CONTEXT_SNAPSHOT_COLUMNS)


def upsert_player_context_snapshots(snapshots: pd.DataFrame) -> int:
    """Persist week-versioned role and availability snapshots."""
    if snapshots is None or snapshots.empty:
        return 0
    missing = set(CONTEXT_SNAPSHOT_COLUMNS) - set(snapshots.columns)
    if missing:
        raise ValueError(f"Player context snapshots are missing columns: {sorted(missing)}")

    placeholders = ", ".join(["?"] * len(CONTEXT_SNAPSHOT_COLUMNS))
    update_columns = [
        column for column in CONTEXT_SNAPSHOT_COLUMNS if column not in {"season", "week", "gsis_id"}
    ]
    with get_connection() as conn:
        if is_sqlite_connection(conn):
            updates = ", ".join(f"{column} = excluded.{column}" for column in update_columns)
            sql = (
                f"INSERT INTO nfl_player_context_snapshots "
                f"({', '.join(CONTEXT_SNAPSHOT_COLUMNS)}) VALUES ({placeholders}) "
                f"ON CONFLICT(season, week, gsis_id) DO UPDATE SET {updates}"
            )
        else:
            updates = ", ".join(f"{column} = VALUES({column})" for column in update_columns)
            sql = (
                f"INSERT INTO nfl_player_context_snapshots "
                f"({', '.join(CONTEXT_SNAPSHOT_COLUMNS)}) VALUES ({placeholders}) "
                f"ON DUPLICATE KEY UPDATE {updates}"
            )
        values = snapshots[list(CONTEXT_SNAPSHOT_COLUMNS)].astype(object)
        rows = list(values.where(values.notna(), None).itertuples(index=False, name=None))
        executemany(sql, rows, conn=conn)
        conn.commit()
    return len(rows)


def refresh_player_context_snapshots(
    rosters: pd.DataFrame,
    depth_charts: pd.DataFrame,
    injuries: pd.DataFrame,
    *,
    through_week: int,
) -> int:
    """Persist causal roster context for the requested week only.

    Prior weeks are never reconstructed with today's capture timestamp; they
    remain immutable evidence for historical replay.
    """
    if rosters is None or rosters.empty:
        return 0
    history = read_dataframe("""
        SELECT gsis_id, season, week, team, snap_count, snap_percentage,
               rushing_attempts, targets, passing_attempts, target_share,
               air_yards, yac_yards, red_zone_touches
        FROM player_stats_enhanced
        WHERE gsis_id IS NOT NULL
        ORDER BY gsis_id, season, week
        """)
    games = read_dataframe(
        """
        SELECT season, kickoff_utc
        FROM games
        WHERE week = ? AND kickoff_utc IS NOT NULL
        """,
        params=(through_week,),
    )
    target_cutoffs: dict[int, str] = {}
    if not games.empty:
        games["kickoff_utc"] = pd.to_datetime(games["kickoff_utc"], errors="coerce", utc=True)
        earliest = games.dropna(subset=["kickoff_utc"]).groupby("season")["kickoff_utc"].min()
        target_cutoffs = {int(season): cutoff.isoformat() for season, cutoff in earliest.items()}
    captured_at = datetime.now(timezone.utc).isoformat()
    snapshots = build_player_context_snapshots(
        rosters,
        depth_charts,
        injuries,
        history,
        target_week=through_week,
        target_cutoffs=target_cutoffs,
        captured_at=captured_at,
    )
    captured_timestamp = pd.Timestamp(captured_at)
    blocked_seasons = {
        season
        for season, cutoff in target_cutoffs.items()
        if captured_timestamp >= pd.Timestamp(cutoff)
    }
    if blocked_seasons:
        snapshots = snapshots[~snapshots["season"].isin(blocked_seasons)].copy()
    if snapshots.empty:
        logger.warning(
            "Skipping context refresh after kickoff for seasons %s",
            sorted(blocked_seasons),
        )
        return 0
    return upsert_player_context_snapshots(snapshots)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    current_season = nfl.get_current_season(roster=True)
    default_seasons = ",".join(
        str(season) for season in [*default_history_seasons(current_season), current_season]
    )
    parser = argparse.ArgumentParser(description="Ingest real NFL data")
    parser.add_argument(
        "--season", type=int, help="Single season to fetch (use --seasons for multiple)"
    )
    parser.add_argument("--through-week", type=int, default=18, help="Fetch through this week")
    parser.add_argument(
        "--seasons",
        type=str,
        default=default_seasons,
        help=f"Comma-separated seasons (default: {default_seasons})",
    )
    return parser.parse_args(argv)


def _refresh_context_and_log(
    rosters: pd.DataFrame,
    depth_charts: pd.DataFrame,
    injuries: pd.DataFrame,
    through_week: int,
) -> None:
    context_count = refresh_player_context_snapshots(
        rosters,
        depth_charts,
        injuries,
        through_week=through_week,
    )
    logger.info("Persisted %s player context snapshots", context_count)


def ingest_seasons(
    seasons: List[int], through_week: int = 18, stats_through_week: Optional[int] = None
) -> int:
    """Fetch, transform, and persist NFL player-week data for the requested seasons."""
    if not seasons:
        raise ValueError("At least one NFL season is required")
    if not 1 <= through_week <= 22:
        raise ValueError("through_week must be between 1 and 22")
    if stats_through_week is None:
        stats_through_week = through_week
    if not 0 <= stats_through_week <= through_week:
        raise ValueError("stats_through_week must be between 0 and through_week")

    logger.info("=== NFL Data Ingestion ===")
    logger.info("Seasons: %s, Through Week: %s", seasons, through_week)

    # Schedule and roster feeds are useful before the first weekly stat is published.
    rosters = fetch_rosters(seasons)
    weekly_rosters = fetch_weekly_rosters(seasons)
    depth_charts = fetch_depth_charts(seasons)
    injuries = fetch_injuries(seasons)
    schedule = fetch_schedules(seasons)
    games = create_games_from_schedule(schedule, through_week=through_week)
    game_count = upsert_games(games)
    roster_context = (
        pd.concat([rosters, weekly_rosters], ignore_index=True, sort=False)
        if not weekly_rosters.empty
        else rosters
    )
    roster_count = upsert_roster_players(roster_context)
    logger.info(
        "Persisted preseason context: %s games, %s roster players", game_count, roster_count
    )

    if stats_through_week == 0:
        logger.info("Skipping player-week ingestion before Week 1")
        _refresh_context_and_log(roster_context, depth_charts, injuries, through_week)
        return 0

    weekly = fetch_weekly_stats(seasons)
    if weekly.empty:
        logger.warning("No weekly player data is available for seasons %s yet", seasons)
        _refresh_context_and_log(roster_context, depth_charts, injuries, through_week)
        return 0
    if "week" not in weekly.columns:
        raise ValueError("NFL weekly player data is missing the required 'week' column")

    weekly = weekly[weekly["week"] <= stats_through_week]
    logger.info("Filtered player stats to weeks 1-%s: %s rows", stats_through_week, len(weekly))
    if weekly.empty:
        logger.warning("No weekly player data exists through week %s", stats_through_week)
        _refresh_context_and_log(roster_context, depth_charts, injuries, through_week)
        return 0

    # Pull high-volume secondary feeds only after confirming player-week data exists.
    snaps = fetch_snap_counts(seasons)
    pbp_rz = fetch_pbp_red_zone(seasons)

    enhanced = transform_to_enhanced_stats(
        weekly,
        snaps,
        rosters=rosters,
        schedule=schedule,
        pbp_rz=pbp_rz,
    )
    if enhanced.empty:
        logger.warning("NFL transformation produced no player-week rows")
        _refresh_context_and_log(roster_context, depth_charts, injuries, through_week)
        return 0
    logger.info("=== Data Summary ===")
    logger.info("Total player-weeks: %s", len(enhanced))
    logger.info("Unique players: %s", enhanced["player_id"].nunique())
    logger.info("Positions: %s", enhanced["position"].value_counts().to_dict())
    logger.info("Weeks: %s", sorted(enhanced["week"].unique()))

    count = upsert_player_stats(enhanced)
    _refresh_context_and_log(roster_context, depth_charts, injuries, through_week)
    logger.info("=== Ingestion Complete: %s player-weeks ===", count)
    return count


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if args.season:
        seasons = [args.season]
    else:
        seasons = [int(value.strip()) for value in args.seasons.split(",") if value.strip()]

    count = ingest_seasons(seasons, through_week=args.through_week)
    logger.info(f"Ingested {count} player stat rows")
    logger.info(f"Ready for Week {args.through_week + 1} predictions")


if __name__ == "__main__":
    main()
