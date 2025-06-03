#!/usr/bin/env python3
"""Sportsbook API integration for NFL Algorithm."""

import requests
import sqlite3
import logging
from typing import List, Dict, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Placeholder API URL - Replace with actual odds provider
API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"

class SportsbookAPI:
    """Collect sportsbook lines and store in the database."""

    def __init__(self, api_key: str, db_path: str = "nfl_data.db"):
        self.api_key = api_key
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create betting_lines table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS betting_lines (
                player_id TEXT,
                season INTEGER,
                sportsbook TEXT,
                rushing_over_under REAL,
                receiving_over_under REAL,
                date_scraped TEXT,
                PRIMARY KEY (player_id, season, sportsbook)
            )
            """
        )
        conn.commit()
        conn.close()

    def fetch_player_props(self) -> List[Dict[str, Optional[float]]]:
        """Fetch player prop lines from the odds API."""
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "player_pass_yards,player_rush_yards,player_rec_yards",
        }
        try:
            response = requests.get(API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.error("Failed to fetch odds: %s", exc)
            return []

        lines = []
        for event in data:
            for bookmaker in event.get("bookmakers", []):
                sportsbook = bookmaker.get("title")
                for market in bookmaker.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        lines.append({
                            "player_id": outcome.get("description"),
                            "season": datetime.now().year,
                            "sportsbook": sportsbook,
                            "prop": market.get("key"),
                            "line": outcome.get("point"),
                            "date_scraped": datetime.utcnow().isoformat(),
                        })
        logger.info("Fetched %d betting lines", len(lines))
        return lines

    def save_lines(self, lines: List[Dict[str, Optional[float]]]):
        """Save betting lines to SQLite database."""
        if not lines:
            logger.warning("No lines to save")
            return
        conn = sqlite3.connect(self.db_path)
        try:
            for entry in lines:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO betting_lines
                    (player_id, season, sportsbook, rushing_over_under, receiving_over_under, date_scraped)
                    VALUES (:player_id, :season, :sportsbook, :rush, :rec, :date_scraped)
                    """,
                    {
                        "player_id": entry.get("player_id"),
                        "season": entry.get("season"),
                        "sportsbook": entry.get("sportsbook"),
                        "rush": entry.get("line") if "rush" in entry.get("prop", "") else None,
                        "rec": entry.get("line") if "rec" in entry.get("prop", "") else None,
                        "date_scraped": entry.get("date_scraped"),
                    },
                )
            conn.commit()
            logger.info("Saved %d betting lines", len(lines))
        except Exception as exc:
            logger.error("Failed to save betting lines: %s", exc)
            conn.rollback()
        finally:
            conn.close()

if __name__ == "__main__":
    print("Sportsbook API Integration")
    api_key = "YOUR_API_KEY"  # Replace with real API key
    api = SportsbookAPI(api_key)
    lines = api.fetch_player_props()
    api.save_lines(lines)
