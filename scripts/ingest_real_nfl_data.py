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
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import nflreadpy as nfl

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from utils.db import get_connection, execute, executemany, read_dataframe
from utils.player_id_utils import make_player_id, canonicalize_team

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_weekly_stats(seasons: List[int]) -> pd.DataFrame:
    """Fetch weekly player stats from nflreadpy (nflverse)."""
    logger.info(f"Fetching weekly data for seasons: {seasons}")
    # nflreadpy returns Polars DataFrame, convert to pandas
    weekly_polars = nfl.load_player_stats(seasons)
    weekly = weekly_polars.to_pandas()
    logger.info(f"Fetched {len(weekly)} total player-week rows")
    return weekly


def fetch_snap_counts(seasons: List[int]) -> pd.DataFrame:
    """Fetch snap count data for usage metrics."""
    try:
        logger.info(f"Fetching snap counts for seasons: {seasons}")
        snaps_polars = nfl.load_snap_counts(seasons)
        snaps = snaps_polars.to_pandas()
        logger.info(f"Fetched {len(snaps)} snap count rows")
        return snaps
    except Exception as e:
        logger.warning(f"Could not fetch snap counts: {e}")
        return pd.DataFrame()


def fetch_rosters(seasons: List[int]) -> pd.DataFrame:
    """Fetch rosters from nflreadpy.

    Returns DataFrame with (at least) gsis_id, full_name/player_name, team,
    season, birth_date. Empty DataFrame on failure (callers fall back to NULL).
    """
    try:
        logger.info(f"Fetching rosters for seasons: {seasons}")
        rosters_polars = nfl.load_rosters(seasons)
        rosters = rosters_polars.to_pandas()
        logger.info(f"Fetched {len(rosters)} roster rows")
        return rosters
    except Exception as e:
        logger.warning(f"Could not fetch rosters: {e}")
        return pd.DataFrame()


def fetch_schedules(seasons: List[int]) -> pd.DataFrame:
    """Fetch game schedules from nflreadpy.

    Returns DataFrame with season, week, home_team, away_team, gameday.
    Empty DataFrame on failure.
    """
    try:
        logger.info(f"Fetching schedules for seasons: {seasons}")
        sched_polars = nfl.load_schedules(seasons)
        sched = sched_polars.to_pandas()
        logger.info(f"Fetched {len(sched)} schedule rows")
        return sched
    except Exception as e:
        logger.warning(f"Could not fetch schedules: {e}")
        return pd.DataFrame()


def fetch_pbp_red_zone(seasons: List[int]) -> pd.DataFrame:
    """Aggregate per-player red-zone touches from play-by-play.

    Counts plays inside the 20-yard line (yardline_100 <= 20) where the player
    was either the rusher (rush_attempt == 1) or a pass target
    (pass_attempt == 1 and receiver_player_id == player). Returns a
    DataFrame with columns (player_id, season, week, red_zone_touches).
    Empty DataFrame on failure.
    """
    try:
        logger.info(f"Fetching pbp for seasons: {seasons} (red-zone touches)")
        pbp_polars = nfl.load_pbp(seasons)
        pbp = pbp_polars.to_pandas()
    except Exception as e:
        logger.warning(f"Could not fetch pbp: {e}")
        return pd.DataFrame()

    needed = {"season", "week", "yardline_100", "rusher_player_id", "receiver_player_id"}
    missing = needed - set(pbp.columns)
    if missing:
        logger.warning("pbp missing columns %s — red_zone_touches falls back to derived formula.", sorted(missing))
        return pd.DataFrame()

    rz = pbp[pbp["yardline_100"] <= 20].copy()
    rushers = rz.dropna(subset=["rusher_player_id"])[
        ["season", "week", "rusher_player_id"]
    ].rename(columns={"rusher_player_id": "player_gsis_id"})
    receivers = rz.dropna(subset=["receiver_player_id"])[
        ["season", "week", "receiver_player_id"]
    ].rename(columns={"receiver_player_id": "player_gsis_id"})
    touches = pd.concat([rushers, receivers], ignore_index=True)
    agg = (
        touches.groupby(["player_gsis_id", "season", "week"], as_index=False)
        .size()
        .rename(columns={"size": "red_zone_touches"})
    )
    logger.info("Aggregated %d player-week red-zone touch rows.", len(agg))
    return agg


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
    df['snap_count'] = 0.0
    df['snap_percentage'] = 0.0

    if snaps is None or snaps.empty:
        logger.info("No snap_counts data — snap fields default to 0.")
        return df

    needed = {'player', 'season', 'week', 'team', 'offense_snaps', 'offense_pct'}
    missing = needed - set(snaps.columns)
    if missing:
        logger.warning("load_snap_counts missing columns %s — snap fields default to 0.", sorted(missing))
        return df

    snaps_norm = snaps[['player', 'season', 'week', 'team', 'offense_snaps', 'offense_pct']].copy()
    snaps_norm['team'] = snaps_norm['team'].apply(canonicalize_team)
    snaps_norm['_name_key'] = snaps_norm['player'].apply(_normalize_name_for_merge)
    # offense_pct from nflreadpy is a 0–1 fraction; rescale to 0–100 if so.
    pct_max = snaps_norm['offense_pct'].dropna().max()
    if pd.notna(pct_max) and pct_max <= 1.5:
        snaps_norm['offense_pct'] = snaps_norm['offense_pct'] * 100.0

    agg = (
        snaps_norm
        .groupby(['_name_key', 'season', 'week', 'team'], as_index=False)
        .agg(offense_snaps=('offense_snaps', 'sum'), offense_pct=('offense_pct', 'max'))
    )

    # Prefer player_display_name ("Aaron Rodgers") over player_name ("A.Rodgers")
    df_name_source = df.get('player_display_name')
    if df_name_source is None or df_name_source.isna().all():
        df_name_source = df['player_name']
    df['_name_key'] = df_name_source.fillna(df['player_name']).apply(_normalize_name_for_merge)

    merged = df.merge(
        agg, how='left', on=['_name_key', 'season', 'week', 'team'],
    )
    merged['snap_count'] = merged['offense_snaps'].fillna(0.0)
    merged['snap_percentage'] = merged['offense_pct'].fillna(0.0)
    merged = merged.drop(columns=['offense_snaps', 'offense_pct', '_name_key'])

    matched = (merged['snap_percentage'] > 0).sum()
    logger.info("Snap merge: matched %d / %d player-weeks (%.1f%%).",
                int(matched), len(merged),
                100.0 * matched / max(1, len(merged)))
    return merged


def _merge_age_from_rosters(df: pd.DataFrame, rosters: pd.DataFrame, game_date_col: str = "game_date") -> pd.DataFrame:
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
    bd = (
        rosters.dropna(subset=["gsis_id", "birth_date"])
        .drop_duplicates(subset=["gsis_id"], keep="last")[["gsis_id", "birth_date"]]
    )
    merged = df.merge(bd, how="left", on="gsis_id")
    game_date = pd.to_datetime(merged[game_date_col], errors="coerce")
    birth = pd.to_datetime(merged["birth_date"], errors="coerce")
    # Calendar-age at game_date (no leap-day adjustment — close enough).
    delta_days = (game_date - birth).dt.days
    age = (delta_days / 365.25).round().astype("Int64")
    # Fallback for unknown birth_date OR game_date.
    fallback_mask = age.isna()
    fallback_rate = float(fallback_mask.mean()) if len(merged) else 0.0
    logger.info("Age fallback rate (real → 26): %.2f%% (%d / %d rows).",
                100.0 * fallback_rate, int(fallback_mask.sum()), len(merged))
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
        logger.warning("Schedule missing columns %s — game_date falls back to {season}-09-01.", sorted(missing))
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
        logger.info("game_date fallback rate: %.2f%% (%d / %d rows).",
                    100.0 * float(fallback_mask.mean()),
                    int(fallback_mask.sum()), len(merged))
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
        pbp_rz, how="left",
        left_on=["gsis_id", "season", "week"],
        right_on=["player_gsis_id", "season", "week"],
    )
    real_rz = merged["red_zone_touches_y"] if "red_zone_touches_y" in merged.columns else merged.get("red_zone_touches")
    if real_rz is None:
        return df
    # Prefer real value; fall back to synthetic where no pbp row.
    synthetic = merged.get("red_zone_touches_x", merged.get("red_zone_touches"))
    merged["red_zone_touches"] = real_rz.fillna(synthetic).astype(float)
    drop_cols = [c for c in ("red_zone_touches_x", "red_zone_touches_y", "player_gsis_id") if c in merged.columns]
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
    positions = ['QB', 'RB', 'WR', 'TE', 'FB']
    df = weekly[weekly['position'].isin(positions)].copy()
    
    # nflreadpy uses 'team' instead of 'recent_team'
    team_col = 'team' if 'team' in df.columns else 'recent_team'

    # Preserve nflverse gsis_id BEFORE we overwrite player_id with our local
    # canonical ID (needed for roster/pbp joins).
    df['gsis_id'] = df['player_id']

    # Create player_id using our convention
    df['player_id'] = df.apply(
        lambda r: make_player_id(r['player_name'], r[team_col]), axis=1
    )

    # Map columns to our schema
    df['name'] = df['player_display_name'].fillna(df['player_name'])
    df['team'] = df[team_col].apply(canonicalize_team)
    df['opponent'] = df['opponent_team'].apply(canonicalize_team)
    
    # Stats mapping - handle both nflreadpy and nfl_data_py column names
    df['rushing_yards'] = df['rushing_yards'].fillna(0).astype(float)
    carries_col = 'carries' if 'carries' in df.columns else 'rushing_attempts'
    df['rushing_attempts'] = df[carries_col].fillna(0).astype(float) if carries_col in df.columns else 0.0
    df['receiving_yards'] = df['receiving_yards'].fillna(0).astype(float)
    df['receptions'] = df['receptions'].fillna(0).astype(float)
    df['targets'] = df['targets'].fillna(0).astype(float)
    df['passing_yards'] = df['passing_yards'].fillna(0).astype(float)
    att_col = 'attempts' if 'attempts' in df.columns else 'passing_attempts'
    df['passing_attempts'] = df[att_col].fillna(0).astype(float) if att_col in df.columns else 0.0

    # Derived metrics
    df['target_share'] = df['target_share'].fillna(0).astype(float) if 'target_share' in df.columns else 0.0
    air_yards_col = 'receiving_air_yards' if 'receiving_air_yards' in df.columns else 'air_yards'
    df['air_yards'] = df[air_yards_col].fillna(0).astype(float) if air_yards_col in df.columns else 0.0
    yac_col = 'receiving_yards_after_catch' if 'receiving_yards_after_catch' in df.columns else 'yac_yards'
    df['yac_yards'] = df[yac_col].fillna(0).astype(float) if yac_col in df.columns else 0.0
    
    df = _merge_snap_counts(df, snaps)

    # Real game_date from schedules (T0 #6) — must precede age so the age
    # calculation uses the actual kickoff day.
    df = _merge_game_date_from_schedule(df, schedule if schedule is not None else pd.DataFrame())

    # Real age from rosters (T0 #6) — falls back to 26 only when birth_date
    # missing (schema is NOT NULL).
    df = _merge_age_from_rosters(df, rosters if rosters is not None else pd.DataFrame())

    # Compute games played per player up to each week
    df = df.sort_values(['player_id', 'season', 'week'])
    df['games_played'] = df.groupby(['player_id', 'season']).cumcount() + 1

    # Synthetic red_zone_touches stays as the fallback. If pbp_rz provided,
    # _merge_red_zone_from_pbp overrides it with the real count.
    df['red_zone_touches'] = (df['rushing_attempts'] * 0.15 + df['receptions'] * 0.1).round(2)
    df = _merge_red_zone_from_pbp(df, pbp_rz if pbp_rz is not None else pd.DataFrame())
    df['game_script'] = 0.0
    df['usage_delta'] = 0.02
    df['age_curve'] = 1.0
    df['oc_change'] = 0
    df['injury_recovery'] = 0
    df['preseason_buzz'] = 0.5
    df['age_squared'] = df['age'] ** 2
    df['injury_games_missed'] = 0.0
    df['team_context_flag'] = df['targets'].apply(lambda t: 'HIGH' if t >= 8 else 'NEUTRAL')
    df['breakout_percentile'] = (df['target_share'] * 2).clip(0, 1)
    
    timestamp = datetime.now(timezone.utc).isoformat()
    df['created_at'] = timestamp
    df['updated_at'] = timestamp
    
    # Select final columns - match MySQL schema exactly
    final_cols = [
        'player_id', 'season', 'week', 'name', 'team', 'position', 'age',
        'games_played', 'snap_count', 'snap_percentage',
        'rushing_yards', 'rushing_attempts', 'passing_yards', 'passing_attempts',
        'receiving_yards', 'receptions', 'targets',
        'red_zone_touches', 'target_share', 'air_yards', 'yac_yards', 'game_script',
        'created_at', 'updated_at'
    ]
    
    result = df[final_cols].copy()
    logger.info(f"Transformed {len(result)} rows for player_stats_enhanced")
    return result


def compute_rolling_features(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Compute rolling averages for key metrics - stores in existing columns."""
    logger.info(f"Computing rolling {window}-game features...")
    
    df = df.sort_values(['player_id', 'season', 'week'])
    
    # Just return sorted df - MySQL schema doesn't have rolling columns
    return df


def upsert_player_stats(df: pd.DataFrame) -> int:
    """Upsert player stats to database (SQLite or MySQL)."""
    if df.empty:
        return 0
    
    logger.info(f"Upserting {len(df)} rows to player_stats_enhanced...")
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cols = list(df.columns)
        col_list = ', '.join(cols)
        
        # Detect backend by checking connection type
        is_sqlite = 'sqlite' in str(type(conn)).lower()
        
        if is_sqlite:
            # SQLite: use INSERT OR REPLACE
            placeholders = ', '.join(['?'] * len(cols))
            sql = f"INSERT OR REPLACE INTO player_stats_enhanced ({col_list}) VALUES ({placeholders})"
        else:
            # MySQL: use ON DUPLICATE KEY UPDATE
            placeholders = ', '.join(['%s'] * len(cols))
            update_clause = ', '.join([f"{c} = VALUES({c})" for c in cols if c not in ('player_id', 'season', 'week')])
            sql = f"""
                INSERT INTO player_stats_enhanced ({col_list})
                VALUES ({placeholders})
                ON DUPLICATE KEY UPDATE {update_clause}
            """
        
        # Convert pandas NA / NaN to None so DB drivers bind NULL.
        df_for_insert = df.astype(object).where(df.notna(), None)
        rows = [tuple(row) for row in df_for_insert.values]
        cursor.executemany(sql, rows)
        conn.commit()
        
        inserted = len(rows)
        logger.info(f"Upserted {inserted} rows")
        return inserted


def create_games_from_stats(df: pd.DataFrame, schedule: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Generate games table entries from player stats joined with real schedules.

    Uses nflverse schedules for game_date, home/away orientation, and venue
    when available. Falls back to f"{season}-09-01" only for games the
    schedule doesn't cover (T0 #6 — no more {season}-01-01 placeholders).
    """
    logger.info("Generating games table entries...")

    if schedule is not None and not schedule.empty and {
        "season", "week", "home_team", "away_team", "gameday"
    }.issubset(schedule.columns):
        sched = schedule.copy()
        sched["home_team"] = sched["home_team"].apply(canonicalize_team)
        sched["away_team"] = sched["away_team"].apply(canonicalize_team)
        # Pick the columns we always need; carry venue/stadium when present.
        venue_col = next((c for c in ("stadium", "venue") if c in sched.columns), None)
        cols = ["season", "week", "home_team", "away_team", "gameday"]
        if venue_col:
            cols.append(venue_col)
        sched = sched[cols].copy()
        # Filter to weeks/seasons present in df to avoid emitting non-played games.
        keys = df[["season", "week"]].drop_duplicates()
        sched = sched.merge(keys, on=["season", "week"], how="inner")
        records = []
        for _, row in sched.iterrows():
            game_id = f"{row['season']}_W{int(row['week'])}_{row['away_team']}_at_{row['home_team']}"
            records.append({
                "game_id": game_id,
                "season": int(row["season"]),
                "week": int(row["week"]),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "game_date": str(row["gameday"])[:10],
                "venue": row[venue_col] if venue_col else None,
            })
        if records:
            return pd.DataFrame(records)
        logger.warning("Schedule join produced 0 game rows — falling back to derived games.")

    # Fallback: derive games from player_stats opponent column.
    games = df.groupby(['season', 'week', 'team']).agg({'opponent': 'first'}).reset_index()
    game_records = []
    seen = set()
    for _, row in games.iterrows():
        season, week = row['season'], row['week']
        team1, team2 = sorted([row['team'], row['opponent']])
        key = (season, week, team1, team2)
        if key not in seen:
            seen.add(key)
            game_id = f"{season}_W{week}_{team2}_at_{team1}"
            game_records.append({
                'game_id': game_id,
                'season': season,
                'week': week,
                'home_team': team1,
                'away_team': team2,
                'game_date': f"{season}-09-01",
                'venue': None,
            })
    return pd.DataFrame(game_records)


def main():
    parser = argparse.ArgumentParser(description='Ingest real NFL data')
    parser.add_argument('--season', type=int, help='Single season to fetch (use --seasons for multiple)')
    parser.add_argument('--through-week', type=int, default=18, help='Fetch through this week')
    parser.add_argument('--seasons', type=str, default='2024,2025', help='Comma-separated seasons (default: 2024,2025)')
    args = parser.parse_args()
    
    if args.season:
        # Single season mode
        seasons = [args.season]
    else:
        # Multi-season mode (default: 2024,2025)
        seasons = [int(s) for s in args.seasons.split(',')]
    
    logger.info(f"=== NFL Data Ingestion ===")
    logger.info(f"Seasons: {seasons}, Through Week: {args.through_week}")
    
    # Fetch data (T0 #6 — pull rosters, schedules, pbp for real field values)
    weekly = fetch_weekly_stats(seasons)
    snaps = fetch_snap_counts(seasons)
    rosters = fetch_rosters(seasons)
    schedule = fetch_schedules(seasons)
    pbp_rz = fetch_pbp_red_zone(seasons)

    # Filter to through_week
    weekly = weekly[weekly['week'] <= args.through_week]
    logger.info(f"Filtered to weeks 1-{args.through_week}: {len(weekly)} rows")

    # Transform
    enhanced = transform_to_enhanced_stats(
        weekly, snaps, rosters=rosters, schedule=schedule, pbp_rz=pbp_rz,
    )
    enhanced = compute_rolling_features(enhanced)
    
    # Summary stats
    logger.info(f"\n=== Data Summary ===")
    logger.info(f"Total player-weeks: {len(enhanced)}")
    logger.info(f"Unique players: {enhanced['player_id'].nunique()}")
    logger.info(f"Positions: {enhanced['position'].value_counts().to_dict()}")
    logger.info(f"Weeks: {sorted(enhanced['week'].unique())}")
    
    # Upsert to database
    count = upsert_player_stats(enhanced)
    
    logger.info(f"\n=== Ingestion Complete ===")
    logger.info(f"Ingested {count} player stat rows")
    logger.info(f"Ready for Week {args.through_week + 1} predictions")


if __name__ == "__main__":
    main()
