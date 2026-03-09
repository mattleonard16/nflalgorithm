"""Ingest NBA player game logs from NBA.com via nba_api.

Pulls all player game logs for the current and prior season and upserts
into the nba_player_game_logs table.  Rate-limited to avoid NBA.com bans.

Usage:
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python scripts/ingest_nba_data.py
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python scripts/ingest_nba_data.py --seasons 2025
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from utils.db import executemany, read_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# NBA.com needs courtesy delays between requests to avoid rate limiting.
# Increase to 2.0 if you see 429s or connection resets.
REQUEST_DELAY_SECONDS = 1.0

# NBA season identifiers (season_year = year season ENDS).
# 2024 => "2024-25" season, 2025 => "2025-26" season.
DEFAULT_SEASONS = [2024, 2025]

# Retry configuration for NBA.com requests.
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 2


def _season_str(season_year: int) -> str:
    """Convert 2024 -> '2024-25'."""
    return f"{season_year}-{str(season_year + 1)[-2:]}"


def _fetch_with_retry(fetch_fn: Any, label: str) -> pd.DataFrame:
    """Call fetch_fn() with exponential-backoff retry on transient errors.

    Retries up to MAX_RETRIES times on requests.exceptions.Timeout and
    requests.exceptions.ConnectionError.  All other exceptions are re-raised
    immediately.  Returns an empty DataFrame after all retries are exhausted.

    Args:
        fetch_fn: Zero-argument callable that returns a pd.DataFrame.
        label: Human-readable label for log messages (e.g. season string).
    """
    import requests.exceptions

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fetch_fn()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY_SECONDS ** (attempt + 1)
                log.warning(
                    "Transient error fetching %s (attempt %d/%d): %s — retrying in %ds",
                    label,
                    attempt + 1,
                    MAX_RETRIES,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                log.warning(
                    "All %d retries exhausted for %s: %s — returning empty DataFrame",
                    MAX_RETRIES,
                    label,
                    exc,
                )
        except Exception:
            raise

    return pd.DataFrame()


def _fetch_season_logs(season_year: int) -> pd.DataFrame:
    """Fetch all player game logs for one NBA season."""
    from nba_api.stats.endpoints import playergamelogs

    season = _season_str(season_year)
    log.info("Fetching game logs for NBA season %s …", season)

    def _call() -> pd.DataFrame:
        endpoint = playergamelogs.PlayerGameLogs(
            season_nullable=season,
            season_type_nullable="Regular Season",
            timeout=30,
        )
        df = endpoint.get_data_frames()[0]
        log.info("  Fetched %d rows for %s", len(df), season)
        return df

    try:
        return _fetch_with_retry(_call, season)
    except Exception as exc:
        log.warning("  Failed to fetch %s: %s", season, exc)
        return pd.DataFrame()


def _transform(df: pd.DataFrame, season_year: int) -> list[dict[str, Any]]:
    """Map NBA.com column names to our table schema."""
    if df.empty:
        return []

    col_map = {
        "PLAYER_ID": "player_id",
        "PLAYER_NAME": "player_name",
        "TEAM_ABBREVIATION": "team_abbreviation",
        "GAME_ID": "game_id",
        "GAME_DATE": "game_date",
        "MATCHUP": "matchup",
        "WL": "wl",
        "MIN": "min",
        "PTS": "pts",
        "REB": "reb",
        "AST": "ast",
        "FG3M": "fg3m",
        "FGM": "fgm",
        "FGA": "fga",
        "FTM": "ftm",
        "FTA": "fta",
        "STL": "stl",
        "BLK": "blk",
        "TOV": "tov",
        "PLUS_MINUS": "plus_minus",
    }

    available = {k: v for k, v in col_map.items() if k in df.columns}
    renamed = df[list(available.keys())].rename(columns=available)
    renamed["season"] = season_year

    # Normalize game_date to ISO format (NBA.com returns 'MMM DD, YYYY')
    if "game_date" in renamed.columns:
        try:
            renamed["game_date"] = pd.to_datetime(renamed["game_date"]).dt.strftime("%Y-%m-%d")
        except Exception:
            pass  # leave as-is if parsing fails

    return renamed.to_dict("records")


def _upsert_rows(rows: list[dict[str, Any]]) -> int:
    """Upsert rows into nba_player_game_logs. Returns count inserted."""
    if not rows:
        return 0

    sql = """
        INSERT OR REPLACE INTO nba_player_game_logs (
            player_id, player_name, team_abbreviation, season,
            game_id, game_date, matchup, wl, min,
            pts, reb, ast, fg3m, fgm, fga, ftm, fta,
            stl, blk, tov, plus_minus
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?
        )
    """

    params = [
        (
            r.get("player_id"),
            r.get("player_name"),
            r.get("team_abbreviation"),
            r.get("season"),
            r.get("game_id"),
            r.get("game_date"),
            r.get("matchup"),
            r.get("wl"),
            r.get("min"),
            r.get("pts"),
            r.get("reb"),
            r.get("ast"),
            r.get("fg3m"),
            r.get("fgm"),
            r.get("fga"),
            r.get("ftm"),
            r.get("fta"),
            r.get("stl"),
            r.get("blk"),
            r.get("tov"),
            r.get("plus_minus"),
        )
        for r in rows
    ]

    executemany(sql, params)
    return len(rows)


def _fetch_team_defensive_stats(season_year: int) -> pd.DataFrame:
    """Fetch team defensive ratings and pace for a season from nba_api."""
    from nba_api.stats.endpoints import LeagueDashTeamStats

    season_str = _season_str(season_year)
    log.info("Fetching team defensive stats for %s ...", season_str)

    # Per-100 possessions stats for defensive ratings
    def _fetch_per100() -> pd.DataFrame:
        stats = LeagueDashTeamStats(
            season=season_str,
            per_mode_detailed="Per100Possessions",
        )
        df = stats.get_data_frames()[0]
        time.sleep(REQUEST_DELAY_SECONDS)
        return df

    try:
        df_per100 = _fetch_with_retry(_fetch_per100, f"{season_str} per100")
    except Exception as exc:
        log.warning("Could not fetch per100 defensive stats for %s: %s", season_str, exc)
        return pd.DataFrame()

    # Base stats for PACE (pace lives in Base measure type)
    def _fetch_pace() -> pd.DataFrame:
        stats = LeagueDashTeamStats(
            season=season_str,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Advanced",
        )
        df = stats.get_data_frames()[0]
        time.sleep(REQUEST_DELAY_SECONDS)
        return df

    df_pace = pd.DataFrame()
    try:
        df_pace = _fetch_with_retry(_fetch_pace, f"{season_str} pace")
    except Exception as exc:
        log.warning("Could not fetch pace stats for %s: %s", season_str, exc)

    if df_per100.empty:
        return pd.DataFrame()

    today = date.today().isoformat()
    result = pd.DataFrame({
        "team_abbreviation": df_per100["TEAM_ABBREVIATION"],
        "season": season_year,
        "as_of_date": today,
        "def_rating": df_per100.get("DEF_RATING", pd.Series([None] * len(df_per100))),
        "opp_pts_per100": df_per100.get("OPP_PTS", pd.Series([None] * len(df_per100))),
        "opp_reb_per100": df_per100.get("OPP_REB", pd.Series([None] * len(df_per100))),
        "opp_ast_per100": df_per100.get("OPP_AST", pd.Series([None] * len(df_per100))),
        "opp_fg3m_per100": df_per100.get("OPP_FG3M", pd.Series([None] * len(df_per100))),
        "games_played": df_per100.get("GP", pd.Series([0] * len(df_per100))).fillna(0).astype(int),
    })

    # Merge PACE from advanced stats if available
    if not df_pace.empty and "PACE" in df_pace.columns:
        pace_map = dict(zip(df_pace["TEAM_ABBREVIATION"], df_pace["PACE"]))
        result["opp_pace"] = result["team_abbreviation"].map(pace_map)
    else:
        result["opp_pace"] = None

    return result


def ingest_defensive_stats(seasons: list[int] = DEFAULT_SEASONS) -> None:
    """Ingest team defensive stats for the given seasons."""
    for season in seasons:
        df = _fetch_team_defensive_stats(season)
        if df.empty:
            log.warning("No defensive stats for season %d", season)
            continue
        cols = [
            "team_abbreviation", "season", "as_of_date",
            "def_rating", "opp_pts_per100", "opp_reb_per100",
            "opp_ast_per100", "opp_fg3m_per100", "games_played", "opp_pace",
        ]
        rows = [tuple(row) for row in df[cols].values]
        executemany(
            "INSERT OR REPLACE INTO nba_team_defensive_stats "
            "(team_abbreviation, season, as_of_date, def_rating, opp_pts_per100, "
            "opp_reb_per100, opp_ast_per100, opp_fg3m_per100, games_played, opp_pace) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        log.info("Stored defensive stats for %d teams (season %d)", len(rows), season)


def data_freshness_check(db_path: str, max_stale_days: int = 3) -> None:
    """Warn if the most recent game_date in nba_player_game_logs is stale.

    Args:
        db_path: Path to the SQLite database (used only for logging context).
        max_stale_days: Number of days before the absence of new data is a warning.
    """
    try:
        result = read_dataframe("SELECT MAX(game_date) as latest FROM nba_player_game_logs")
        if result.empty or result.iloc[0]["latest"] is None:
            log.warning("data_freshness_check: nba_player_game_logs has no data in %s", db_path)
            return

        latest_str = result.iloc[0]["latest"]
        latest_date = date.fromisoformat(str(latest_str)[:10])
        gap = (date.today() - latest_date).days

        if gap > max_stale_days:
            log.warning(
                "data_freshness_check: latest game_date is %s (%d days ago) — data may be stale (db=%s)",
                latest_str,
                gap,
                db_path,
            )
        else:
            log.info(
                "data_freshness_check: latest game_date is %s (%d days ago) — OK",
                latest_str,
                gap,
            )
    except Exception as exc:
        log.warning("data_freshness_check failed: %s", exc)


def ingest(seasons: list[int] = DEFAULT_SEASONS) -> None:
    total = 0
    for i, season_year in enumerate(seasons):
        df = _fetch_season_logs(season_year)
        rows = _transform(df, season_year)
        n = _upsert_rows(rows)
        total += n
        log.info("  Upserted %d rows for %s", n, _season_str(season_year))

        if i < len(seasons) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    # Summary
    summary = read_dataframe(
        "SELECT season, COUNT(*) as games, COUNT(DISTINCT player_id) as players "
        "FROM nba_player_game_logs GROUP BY season ORDER BY season"
    )
    log.info("Ingestion complete. %d total rows upserted.", total)
    log.info("\n%s", summary.to_string(index=False))

    import os
    db_path = os.environ.get("SQLITE_DB_PATH", "nfl_data.db")
    data_freshness_check(db_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NBA player game logs")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=DEFAULT_SEASONS,
        help="Season end-years to ingest (e.g. 2024 2025)",
    )
    args = parser.parse_args()
    ingest(args.seasons)
    ingest_defensive_stats(args.seasons)


if __name__ == "__main__":
    main()
