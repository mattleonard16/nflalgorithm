#!/usr/bin/env python3
"""Utility script to inspect projection ↔ odds matching quality."""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config


logger = logging.getLogger(__name__)


@dataclass
class Snapshot:
    projections: pd.DataFrame
    odds: pd.DataFrame
    players: pd.DataFrame


def _load_snapshot(season: int, week: int) -> Snapshot:
    with sqlite3.connect(config.database.path) as conn:
        projections = pd.read_sql_query(
            """
            SELECT season, week, player_id, team, opponent, market, mu, sigma
            FROM weekly_projections
            WHERE season = ? AND week = ?
            """,
            conn,
            params=(season, week),
        )

        odds = pd.read_sql_query(
            """
            SELECT event_id, season, week, player_id, market, sportsbook, line, price
            FROM weekly_odds
            WHERE season = ? AND week = ?
            """,
            conn,
            params=(season, week),
        )

        players = pd.read_sql_query(
            """
            SELECT DISTINCT player_id, name AS player_name
            FROM player_stats_enhanced
            WHERE season = ? AND week = ?
            """,
            conn,
            params=(season, week),
        )

    return Snapshot(projections=projections, odds=odds, players=players)


def _safe_nunique(series: pd.Series) -> int:
    return series.dropna().nunique()


def _summarise(snapshot: Snapshot) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    projections = snapshot.projections
    odds = snapshot.odds

    matched = projections.merge(
        odds,
        on=["season", "week", "player_id", "market"],
        how="inner",
        suffixes=("_proj", "_odds"),
    )

    matched_ids = set(matched["player_id"].dropna())

    proj_unmatched = (
        projections[~projections["player_id"].isin(matched_ids)].copy()
        if not projections.empty
        else projections.copy()
    )
    odds_unmatched = (
        odds[~odds["player_id"].isin(matched_ids)].copy()
        if not odds.empty
        else odds.copy()
    )

    def _rank_unmatched(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        grouped = (
            df.dropna(subset=["player_id"])
            .groupby("player_id")
            .agg(
                markets=("market", _safe_nunique),
                entries=("market", "count"),
            )
            .sort_values("entries", ascending=False)
        )
        return grouped.reset_index()

    return matched, _rank_unmatched(proj_unmatched), _rank_unmatched(odds_unmatched)


def _attach_names(df: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    name_map = players.set_index("player_id")["player_name"].to_dict()
    df = df.copy()
    df["player_name"] = df["player_id"].map(name_map)
    return df


def _print_header(title: str) -> None:
    print("\n" + title)
    print("=" * len(title))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect projection/odds matching quality")
    parser.add_argument("season", type=int, help="Season to inspect")
    parser.add_argument("week", type=int, help="Week to inspect")
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Maximum unmatched players to display per side",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    snapshot = _load_snapshot(args.season, args.week)

    projections = snapshot.projections
    odds = snapshot.odds

    if projections.empty:
        logger.warning("No projections found for season=%s week=%s", args.season, args.week)
    if odds.empty:
        logger.warning("No odds found for season=%s week=%s", args.season, args.week)

    matched, proj_unmatched, odds_unmatched = _summarise(snapshot)

    _print_header("Summary")
    print(
        "Projections rows: {rows} | players: {players} | markets: {markets}".format(
            rows=len(projections),
            players=_safe_nunique(projections.get("player_id", pd.Series(dtype=object))),
            markets=_safe_nunique(projections.get("market", pd.Series(dtype=object))),
        )
    )
    print(
        "Odds rows      : {rows} | players: {players} | markets: {markets}".format(
            rows=len(odds),
            players=_safe_nunique(odds.get("player_id", pd.Series(dtype=object))),
            markets=_safe_nunique(odds.get("market", pd.Series(dtype=object))),
        )
    )
    print(
        "Matched rows   : {rows} | players: {players}".format(
            rows=len(matched),
            players=_safe_nunique(matched.get("player_id", pd.Series(dtype=object))),
        )
    )

    proj_unmatched = _attach_names(proj_unmatched, snapshot.players)
    odds_unmatched = _attach_names(odds_unmatched, snapshot.players)

    if not proj_unmatched.empty:
        _print_header("Top unmatched projections")
        print(
            proj_unmatched.head(args.limit)[
                ["player_id", "player_name", "markets", "entries"]
            ].to_string(index=False)
        )
    else:
        print("\nNo unmatched projections by player_id.")

    if not odds_unmatched.empty:
        _print_header("Top unmatched odds")
        print(
            odds_unmatched.head(args.limit)[
                ["player_id", "player_name", "markets", "entries"]
            ].to_string(index=False)
        )
    else:
        print("\nNo unmatched odds by player_id.")


if __name__ == "__main__":
    main()
