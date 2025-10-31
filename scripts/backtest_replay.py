"""Weekly backtest replay for NFL value engine."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from config import config
from value_betting_engine import rank_weekly_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay weekly odds snapshots and compute metrics")
    parser.add_argument("--season", type=int, required=True, help="Season to replay")
    parser.add_argument("--weeks", nargs="+", type=int, required=True, help="List of weeks to replay")
    parser.add_argument("--min-edge", type=float, default=None, help="Minimum edge to filter bets")
    parser.add_argument("--kelly", type=float, default=None, help="Override Kelly fraction cap")
    parser.add_argument("--max-frac", type=float, default=None, help="Override max bankroll fraction")
    parser.add_argument("--dry-run", action="store_true", help="Do not place bets; metrics only")
    return parser.parse_args()


def replay_weeks(
    season: int,
    weeks: Iterable[int],
    min_edge: float | None,
    kelly_cap: float | None,
    max_fraction: float | None,
    dry_run: bool
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    threshold = min_edge if min_edge is not None else config.betting.min_edge_threshold
    bankroll = 1000.0

    for week in weeks:
        ranked = rank_weekly_value(season, week, threshold, place=not dry_run, bankroll=bankroll)
        if ranked.empty:
            continue

        df = ranked.copy()
        if kelly_cap is not None:
            df['kelly_fraction'] = np.minimum(df['kelly_fraction'], kelly_cap)
        if max_fraction is not None:
            df['kelly_fraction'] = np.minimum(df['kelly_fraction'], max_fraction)
        df['stake'] = df['kelly_fraction'] * bankroll
        df['season'] = season
        df['week'] = week
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def compute_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            'bets_placed': 0,
            'total_stake': 0,
            'expected_return': 0,
            'expected_roi': 0,
            'avg_edge': 0,
            'avg_clv_bp': 0,
            'hit_rate': 0,
            'max_drawdown': 0
        }

    total_stake = float(df['stake'].sum())
    expected_return = float((df['stake'] * df['expected_roi']).sum())
    expected_roi = (expected_return / total_stake) if total_stake else 0
    avg_edge = float(df['edge_percentage'].mean())
    avg_clv_bp = float(((df['mu'] - df['line']) * 100).mean())
    hit_rate = float(df['p_win'].mean())
    max_drawdown = float(-df['stake'].max())

    return {
        'bets_placed': int((df['recommendation'] == 'BET').sum()),
        'total_stake': total_stake,
        'expected_return': expected_return,
        'expected_roi': expected_roi,
        'avg_edge': avg_edge,
        'avg_clv_bp': avg_clv_bp,
        'hit_rate': hit_rate,
        'max_drawdown': max_drawdown
    }


def save_metrics(season: int, weeks: List[int], metrics: dict) -> None:
    logs_dir = config.logs_dir / 'metrics'
    logs_dir.mkdir(parents=True, exist_ok=True)
    week_label = f"{min(weeks)}-{max(weeks)}" if len(weeks) > 1 else str(weeks[0])
    path = logs_dir / f"season-{season}-weeks-{week_label}.json"
    payload = {
        'season': season,
        'weeks': weeks,
        'generated_at': datetime.utcnow().isoformat(),
        **metrics
    }
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    weeks = sorted(set(args.weeks))
    df = replay_weeks(
        season=args.season,
        weeks=weeks,
        min_edge=args.min_edge,
        kelly_cap=args.kelly,
        max_fraction=args.max_frac,
        dry_run=args.dry_run
    )
    metrics = compute_metrics(df)
    save_metrics(args.season, weeks, metrics)


if __name__ == "__main__":
    main()
