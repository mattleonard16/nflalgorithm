"""Export deduplicated weekly value bets for a season/week."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import config
from utils.db import read_dataframe


EXPORT_COLUMNS = [
    "player_name",
    "position",
    "team_display",
    "opponent",
    "prop_type",
    "sportsbook",
    "line",
    "price",
    "model_prediction",
    "sigma",
    "p_win_pct",
    "edge_pct",
    "roi_pct",
    "kelly_pct",
    "stake",
    "recommendation",
]


def _fetch_materialized(season: int, week: int) -> pd.DataFrame:
    query = """
        SELECT
            mv.season,
            mv.week,
            mv.player_id,
            mv.market,
            mv.sportsbook,
            mv.line,
            mv.price,
            mv.mu,
            mv.sigma,
            mv.p_win,
            mv.edge_percentage,
            mv.expected_roi,
            mv.kelly_fraction,
            mv.stake,
            wp.opponent,
            wp.team AS team_projection,
            ps.name AS player_name,
            ps.position,
            ps.team AS team_stats
        FROM materialized_value_view mv
        LEFT JOIN weekly_projections wp
            ON mv.season = wp.season
            AND mv.week = wp.week
            AND mv.player_id = wp.player_id
            AND mv.market = wp.market
        LEFT JOIN player_stats_enhanced ps
            ON mv.player_id = ps.player_id
            AND mv.season = ps.season
            AND mv.week = ps.week
        WHERE mv.season = ? AND mv.week = ?
    """
    return read_dataframe(query, params=(season, week))


def _format_export(df: pd.DataFrame, min_edge: float) -> pd.DataFrame:
    if df.empty:
        return df

    working = df.copy()
    working["player_name"] = working["player_name"].fillna(working["player_id"])
    working["position"] = working["position"].fillna("FLEX")
    working["team_display"] = working["team_projection"].fillna(working["team_stats"])
    working["team_display"] = working["team_display"].fillna("-")
    working["opponent"] = working["opponent"].fillna("-")
    working["prop_type"] = working["market"]
    working["model_prediction"] = working["mu"]
    working["p_win_pct"] = (working["p_win"] * 100).round(1)
    working["edge_pct"] = (working["edge_percentage"] * 100).round(2)
    working["roi_pct"] = (working["expected_roi"] * 100).round(2)
    working["kelly_pct"] = (working["kelly_fraction"] * 100).round(2)
    working["recommendation"] = np.where(
        working["edge_percentage"] >= min_edge,
        "BET",
        "PASS",
    )

    working = working.sort_values(
        ["edge_percentage", "expected_roi", "p_win", "price"],
        ascending=[False, False, False, False],
    )
    # Prefer real books over SimBook when both exist for the same player/market
    working["is_simbook"] = working["sportsbook"].str.lower().eq("simbook")
    def pick_best(group: pd.DataFrame) -> pd.DataFrame:
        # If any real book present, drop the simbook rows
        real = group[~group["is_simbook"]]
        if not real.empty:
            group = real
        # Take the top edge / ROI combo
        return group.sort_values(
            ["edge_percentage", "expected_roi", "p_win", "price"],
            ascending=[False, False, False, False],
        ).head(1)

    deduped = (
        working.groupby(["player_id", "market"], as_index=False, group_keys=False)
        .apply(pick_best)
        .reset_index(drop=True)
    )
    # Secondary guardrail: collapse residual duplicates that share the same display
    # name and market (often synthetic/alias rows) by keeping the best edge.
    deduped = (
        deduped.sort_values(
            ["edge_percentage", "expected_roi", "p_win", "price"],
            ascending=[False, False, False, False],
        )
        .drop_duplicates(subset=["player_name", "market"], keep="first")
        .reset_index(drop=True)
    )

    return deduped[EXPORT_COLUMNS].reset_index(drop=True)


def export_weekly_values(
    season: int,
    week: int,
    output_path: Path,
    mirror_path: Optional[Path] = None,
    min_edge: Optional[float] = None,
) -> Path:
    min_edge_threshold = min_edge if min_edge is not None else config.betting.min_edge_threshold
    df = _fetch_materialized(season, week)

    formatted = _format_export(df, min_edge_threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    formatted.to_csv(output_path, index=False)

    if mirror_path is not None:
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        formatted.to_csv(mirror_path, index=False)

    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export deduplicated weekly value bets")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination CSV file",
    )
    parser.add_argument(
        "--mirror",
        type=Path,
        default=None,
        help="Optional secondary path (e.g., ~/Documents/week10.csv)",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=None,
        help="Override min edge threshold",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = args.output or Path("reports") / f"week_{args.week}_dedup.csv"
    mirror_path = Path(args.mirror).expanduser() if args.mirror else None
    export_weekly_values(
        args.season,
        args.week,
        output_path,
        mirror_path=mirror_path,
        min_edge=args.min_edge,
    )


if __name__ == "__main__":
    main()
