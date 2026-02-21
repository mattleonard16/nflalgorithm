#!/usr/bin/env python3
"""Ingest NBA injury / DNP data.

Retrospective mode: detects DNPs from nba_player_game_logs by finding
players with min=0 or absent from games their team played.

Writes rows to the ``nba_injuries`` table created by schema_migrations.

Usage:
    python scripts/ingest_nba_injuries.py --date 2026-02-17
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.db import executemany, read_dataframe

logger = logging.getLogger(__name__)


def _season_from_date(game_date: str) -> int:
    """NBA season starts in October: Oct 2024 -> season 2024."""
    dt = datetime.strptime(game_date, "%Y-%m-%d")
    return dt.year if dt.month >= 10 else dt.year - 1


def detect_dnps(game_date: str) -> List[Dict]:
    """Detect players who did not play on *game_date*.

    Algorithm:
    1. Find all teams that played on game_date from nba_player_game_logs.
    2. For each team, find the roster (all players who have appeared that season).
    3. Players on that roster who either have min=0 or are absent from the
       game_date logs are marked as DNP (status='OUT').
    """
    season = _season_from_date(game_date)

    # Teams that played on game_date
    teams_df = read_dataframe(
        "SELECT DISTINCT team_abbreviation FROM nba_player_game_logs "
        "WHERE game_date = ?",
        params=(game_date,),
    )
    if teams_df.empty:
        logger.info("No games found for %s — no DNPs to detect.", game_date)
        return []

    teams_playing = list(teams_df["team_abbreviation"])
    logger.info("Teams playing on %s: %s", game_date, teams_playing)

    # Players who actually played (min > 0)
    played_df = read_dataframe(
        "SELECT player_id, player_name, team_abbreviation, min "
        "FROM nba_player_game_logs WHERE game_date = ?",
        params=(game_date,),
    )

    played_ids = set()
    zero_min_rows: List[Dict] = []

    for row in played_df.to_dict("records"):
        minutes = row.get("min") or 0
        if float(minutes) > 0:
            played_ids.add(int(row["player_id"]))
        else:
            zero_min_rows.append(row)

    # Build per-team rosters for this season (players who appeared at least once)
    roster_df = read_dataframe(
        "SELECT DISTINCT player_id, player_name, team_abbreviation "
        "FROM nba_player_game_logs "
        "WHERE season = ? AND team_abbreviation IN ({}) ".format(
            ",".join("?" for _ in teams_playing)
        ),
        params=(season, *teams_playing),
    )

    scraped_at = datetime.now(timezone.utc).isoformat()
    injuries: List[Dict] = []

    # Players with 0 minutes logged
    for row in zero_min_rows:
        injuries.append({
            "player_id": int(row["player_id"]),
            "player_name": row["player_name"],
            "team": row.get("team_abbreviation"),
            "game_date": game_date,
            "status": "OUT",
            "reason": "DNP — 0 minutes",
            "source": "game_logs",
            "scraped_at": scraped_at,
        })

    # Players on roster but absent from game_date entirely
    for row in roster_df.to_dict("records"):
        pid = int(row["player_id"])
        team = row["team_abbreviation"]
        if team not in teams_playing:
            continue
        if pid in played_ids:
            continue
        # Check not already captured via zero-min
        if any(i["player_id"] == pid for i in injuries):
            continue

        injuries.append({
            "player_id": pid,
            "player_name": row["player_name"],
            "team": team,
            "game_date": game_date,
            "status": "OUT",
            "reason": "DNP — absent from game log",
            "source": "game_logs",
            "scraped_at": scraped_at,
        })

    logger.info("Detected %d DNPs for %s", len(injuries), game_date)
    return injuries


def save_injuries(injuries: List[Dict]) -> int:
    """Persist injury rows to nba_injuries. Returns count written."""
    if not injuries:
        return 0

    sql = """
        INSERT OR IGNORE INTO nba_injuries
            (player_id, player_name, team, game_date, status, reason, source, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = [
        (
            r["player_id"],
            r["player_name"],
            r.get("team"),
            r["game_date"],
            r["status"],
            r.get("reason"),
            r.get("source"),
            r["scraped_at"],
        )
        for r in injuries
    ]

    executemany(sql, params)
    logger.info("Saved %d injury rows to nba_injuries.", len(params))
    return len(params)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NBA injury / DNP data.")
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        metavar="YYYY-MM-DD",
        help="Game date to process",
    )
    args = parser.parse_args()

    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        logger.error("Invalid date format: %s. Use YYYY-MM-DD.", args.date)
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    injuries = detect_dnps(args.date)
    count = save_injuries(injuries)
    logger.info("Done. %d injury rows written for %s.", count, args.date)


if __name__ == "__main__":
    main()
