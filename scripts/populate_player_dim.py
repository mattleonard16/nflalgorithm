"""Populate the player_dim canonical dimension table.

Reads the latest row per player_id from player_stats_enhanced
(ranked by season DESC, week DESC) and upserts into player_dim.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from utils.db import executemany, get_backend, read_dataframe

logger = logging.getLogger(__name__)


def populate_player_dim() -> int:
    """Upsert player_dim from player_stats_enhanced.

    Returns the number of players written.
    """
    query = """
        SELECT
            latest.player_id,
            latest.gsis_id,
            latest.name AS player_name,
            latest.position,
            latest.team,
            latest.season AS last_season,
            latest.week AS last_week
        FROM (
            SELECT
                player_id, gsis_id, name, position, team, season, week,
                ROW_NUMBER() OVER (
                    PARTITION BY player_id
                    ORDER BY season DESC, week DESC
                ) AS rn
            FROM player_stats_enhanced
        ) latest
        LEFT JOIN player_dim existing ON existing.player_id = latest.player_id
        WHERE rn = 1
          AND (
              existing.player_id IS NULL
              OR latest.season > existing.last_season
              OR (latest.season = existing.last_season AND latest.week >= existing.last_week)
          )
    """
    df = read_dataframe(query)

    if df.empty:
        logger.info("No player stats found; player_dim not populated")
        return 0

    now = datetime.now(timezone.utc).isoformat()
    columns = "player_id, gsis_id, player_name, position, team, last_season, last_week, updated_at"
    if get_backend() == "sqlite":
        upsert_sql = f"""
            INSERT INTO player_dim ({columns})
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(player_id) DO UPDATE SET
                gsis_id = excluded.gsis_id,
                player_name = excluded.player_name,
                position = excluded.position,
                team = excluded.team,
                last_season = excluded.last_season,
                last_week = excluded.last_week,
                updated_at = excluded.updated_at
        """
    else:
        upsert_sql = f"""
            INSERT INTO player_dim ({columns})
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON DUPLICATE KEY UPDATE
                gsis_id = VALUES(gsis_id),
                player_name = VALUES(player_name),
                position = VALUES(position),
                team = VALUES(team),
                last_season = VALUES(last_season),
                last_week = VALUES(last_week),
                updated_at = VALUES(updated_at)
        """

    rows = [
        (
            row["player_id"],
            row["gsis_id"],
            row["player_name"],
            row["position"],
            row["team"],
            int(row["last_season"]),
            int(row["last_week"]),
            now,
        )
        for _, row in df.iterrows()
    ]
    executemany(upsert_sql, rows)

    logger.info("player_dim populated with %d players", len(df))
    return len(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = populate_player_dim()
    print(f"Populated player_dim with {count} players")
