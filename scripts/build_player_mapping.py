#!/usr/bin/env python3
"""Utility to populate player_mappings table using current joins."""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from prop_integration import join_odds_projections
from schema_migrations import MigrationManager


logger = logging.getLogger(__name__)


def _prepare_mapping(df: pd.DataFrame, min_confidence: float) -> pd.DataFrame:
    if df.empty:
        return df
    required = {
        'player_id',
        'player_id_odds',
        'player_name',
        'team',
        'team_odds',
        'match_type',
        'match_confidence',
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"join_odds_projections missing required columns: {missing}")

    df = df[df['match_confidence'] >= min_confidence].copy()
    if df.empty:
        return df

    df['player_id_projections'] = df['player_id']

    df = df.sort_values('match_confidence', ascending=False)
    df = df.drop_duplicates(subset=['player_id', 'player_id_odds'])

    return df[
        [
            'player_id',
            'player_id_odds',
            'player_id_projections',
            'player_name',
            'team',
            'team_odds',
            'match_type',
            'match_confidence',
        ]
    ].rename(
        columns={
            'player_id': 'player_id_canonical',
            'team': 'team_projections',
        }
    )


def _upsert_mappings(rows: Iterable[tuple]) -> None:
    with sqlite3.connect(config.database.path) as conn:
        conn.executemany(
            """
            INSERT INTO player_mappings (
                player_id_canonical,
                player_id_odds,
                player_id_projections,
                player_name,
                team_projections,
                team_odds,
                match_type,
                confidence_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(player_id_canonical, player_id_odds) DO UPDATE SET
                player_id_projections=excluded.player_id_projections,
                player_name=excluded.player_name,
                team_projections=excluded.team_projections,
                team_odds=excluded.team_odds,
                match_type=excluded.match_type,
                confidence_score=excluded.confidence_score,
                created_at=CURRENT_TIMESTAMP
            """,
            rows,
        )
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build player_id mapping table from joins")
    parser.add_argument("season", type=int, help="Season to sample")
    parser.add_argument("week", type=int, help="Week to sample")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.9,
        help="Minimum match confidence to persist (default: 0.9)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print mappings instead of writing")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    # Ensure schema includes player_mappings table
    MigrationManager(config.database.path).run()

    logger.info("Building mappings for season=%s week=%s", args.season, args.week)
    joined = join_odds_projections(args.season, args.week)

    if joined.empty:
        logger.warning("No joined data available; aborting")
        return

    mapping_df = _prepare_mapping(joined, args.min_confidence)
    if mapping_df.empty:
        logger.warning("No matches met confidence >= %.3f", args.min_confidence)
        return

    if args.dry_run:
        print(mapping_df.to_string(index=False))
        return

    _upsert_mappings(mapping_df.itertuples(index=False, name=None))
    logger.info("Upserted %d player mappings", len(mapping_df))


if __name__ == "__main__":
    main()
