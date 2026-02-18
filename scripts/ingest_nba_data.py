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


def _season_str(season_year: int) -> str:
    """Convert 2024 -> '2024-25'."""
    return f"{season_year}-{str(season_year + 1)[-2:]}"


def _fetch_season_logs(season_year: int) -> pd.DataFrame:
    """Fetch all player game logs for one NBA season."""
    from nba_api.stats.endpoints import playergamelogs

    season = _season_str(season_year)
    log.info("Fetching game logs for NBA season %s …", season)

    try:
        endpoint = playergamelogs.PlayerGameLogs(
            season_nullable=season,
            season_type_nullable="Regular Season",
            timeout=30,
        )
        df = endpoint.get_data_frames()[0]
        log.info("  Fetched %d rows for %s", len(df), season)
        return df
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


if __name__ == "__main__":
    main()
