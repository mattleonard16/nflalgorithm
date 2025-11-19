"""Materialize weekly value betting view."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config import config
from utils.db import get_connection
from value_betting_engine import rank_weekly_value


def materialize_week(season: int, week: int, min_edge: Optional[float] = None) -> pd.DataFrame:
    """Persist materialized value view for dashboard consumption."""

    threshold = min_edge if min_edge is not None else config.betting.min_edge_threshold
    ranked = rank_weekly_value(season, week, threshold, place=False)

    with get_connection() as conn:
        conn.execute(
            "DELETE FROM materialized_value_view WHERE season = ? AND week = ?",
            (season, week),
        )

        if ranked.empty:
            conn.commit()
            return ranked

        payload = ranked.copy()
        # Ensure season and week are explicitly set and non-null (required for NOT NULL constraint)
        payload['season'] = payload['season'].fillna(season).astype(int)
        payload['week'] = payload['week'].fillna(week).astype(int)
        
        # Drop any rows with missing required columns to avoid constraint errors
        required_cols = ['season', 'week', 'player_id', 'event_id', 'market', 'sportsbook']
        payload = payload.dropna(subset=required_cols).copy()
        
        if payload.empty:
            conn.commit()
            return ranked
        
        payload['generated_at'] = datetime.utcnow().isoformat()

        sql = (
            """
            INSERT INTO materialized_value_view (
                season, week, player_id, event_id, team, team_odds, market, sportsbook, line, price,
                mu, sigma, p_win, edge_percentage, expected_roi, kelly_fraction, stake, generated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(season, week, player_id, market, sportsbook, event_id)
            DO UPDATE SET
                team=excluded.team,
                team_odds=excluded.team_odds,
                line=excluded.line,
                price=excluded.price,
                mu=excluded.mu,
                sigma=excluded.sigma,
                p_win=excluded.p_win,
                edge_percentage=excluded.edge_percentage,
                expected_roi=excluded.expected_roi,
                kelly_fraction=excluded.kelly_fraction,
                stake=excluded.stake,
                generated_at=excluded.generated_at
            """
        )

        conn.executemany(
            sql,
            payload[
                [
                    'season', 'week', 'player_id', 'event_id', 'team', 'team_odds', 'market',
                    'sportsbook', 'line', 'price', 'mu', 'sigma', 'p_win', 'edge_percentage',
                    'expected_roi', 'kelly_fraction', 'stake', 'generated_at',
                ]
            ].itertuples(index=False, name=None)
        )
        conn.commit()

    return ranked


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize weekly value betting view")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--min-edge", type=float, default=None, help="Minimum edge to include")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    materialize_week(args.season, args.week, min_edge=args.min_edge)


if __name__ == "__main__":
    main()
