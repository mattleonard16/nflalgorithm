#!/usr/bin/env python3
"""Real-time data updater for the NFL algorithm.

This script demonstrates how new game data could be pulled from a live API and
stored in the existing SQLite database. The API calls are placeholders to be
replaced with an actual provider.
"""

import requests
import sqlite3
from datetime import datetime
import argparse
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCORES_API = "https://example.com/nfl/scores"  # Placeholder URL


def _ensure_table(db_path: str) -> None:
    """Create player_stats table if missing."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_stats (
            player_id TEXT,
            season INTEGER,
            name TEXT,
            team TEXT,
            position TEXT,
            age INTEGER,
            games_played INTEGER,
            rushing_yards INTEGER,
            rushing_attempts INTEGER,
            receiving_yards INTEGER,
            receptions INTEGER,
            targets INTEGER,
            snap_count INTEGER,
            injury_games_missed INTEGER,
            PRIMARY KEY (player_id, season)
        )
        """
    )
    conn.commit()
    conn.close()


def fetch_daily_stats(date: str) -> List[Dict[str, str]]:
    """Fetch stats for a given date (YYYY-MM-DD)."""
    try:
        resp = requests.get(SCORES_API, params={"date": date}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.error("Failed to fetch stats: %s", exc)
        return []


def update_database(stats: List[Dict[str, str]], db_path: str = "nfl_data.db") -> None:
    """Insert live stats into the database."""
    if not stats:
        logger.info("No stats to update")
        return

    _ensure_table(db_path)
    conn = sqlite3.connect(db_path)
    try:
        for row in stats:
            conn.execute(
                """
                INSERT OR IGNORE INTO player_stats
                (player_id, season, name, team, position, games_played,
                 rushing_yards, rushing_attempts, receiving_yards, receptions, targets,
                 snap_count, injury_games_missed)
                VALUES (:player_id, :season, :name, :team, :position, 1,
                        :rush_yds, :rush_att, :rec_yds, :rec, :tgt, 0, 0)
                """,
                {
                    "player_id": row.get("player_id"),
                    "season": datetime.utcnow().year,
                    "name": row.get("name"),
                    "team": row.get("team"),
                    "position": row.get("position"),
                    "rush_yds": row.get("rushing_yards", 0),
                    "rush_att": row.get("rushing_attempts", 0),
                    "rec_yds": row.get("receiving_yards", 0),
                    "rec": row.get("receptions", 0),
                    "tgt": row.get("targets", 0),
                },
            )
        conn.commit()
        logger.info("Inserted %d live stat rows", len(stats))
    except Exception as exc:
        logger.error("Failed to update database: %s", exc)
        conn.rollback()
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> None:
    """Run the real-time update process."""
    parser = argparse.ArgumentParser(description="Update player stats in real time")
    parser.add_argument(
        "--date",
        default=datetime.utcnow().strftime("%Y-%m-%d"),
        help="Date to fetch stats for (YYYY-MM-DD)",
    )
    args = parser.parse_args(argv)

    print("Real-Time NFL Updater")
    stats = fetch_daily_stats(args.date)
    update_database(stats)


if __name__ == "__main__":
    main()

