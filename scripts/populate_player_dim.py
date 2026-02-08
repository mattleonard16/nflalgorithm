"""Populate the player_dim canonical dimension table.

Reads the latest row per player_id from player_stats_enhanced
(ranked by season DESC, week DESC) and upserts into player_dim.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from utils.db import execute, read_dataframe

logger = logging.getLogger(__name__)


def populate_player_dim() -> int:
    """Upsert player_dim from player_stats_enhanced.

    Returns the number of players written.
    """
    query = """
        SELECT
            player_id,
            name AS player_name,
            position,
            team,
            season AS last_season,
            week AS last_week
        FROM (
            SELECT
                player_id, name, position, team, season, week,
                ROW_NUMBER() OVER (
                    PARTITION BY player_id
                    ORDER BY season DESC, week DESC
                ) AS rn
            FROM player_stats_enhanced
        )
        WHERE rn = 1
    """
    df = read_dataframe(query)

    if df.empty:
        logger.info("No player stats found; player_dim not populated")
        return 0

    now = datetime.now(timezone.utc).isoformat()

    for _, row in df.iterrows():
        execute(
            """
            INSERT OR REPLACE INTO player_dim
                (player_id, player_name, position, team, last_season, last_week, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            params=(
                row["player_id"],
                row["player_name"],
                row["position"],
                row["team"],
                int(row["last_season"]),
                int(row["last_week"]),
                now,
            ),
        )

    logger.info("player_dim populated with %d players", len(df))
    return len(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = populate_player_dim()
    print(f"Populated player_dim with {count} players")
