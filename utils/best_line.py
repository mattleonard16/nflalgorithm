"""Collapse a value card to one row per bet.

The card stores one row per sportsbook for each (player, market, side), so the
same bet can appear five times. The dashboard shows the best book by default
and grading should count the bet once, so both go through this function.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

BET_KEY = ("player_id", "market", "side")


def best_line_per_bet(card: pd.DataFrame, keys: Sequence[str] = BET_KEY) -> pd.DataFrame:
    """Keep the highest-edge row for each bet, ordered by edge descending.

    Edge already prices in the line and the odds, so a tie goes to the book
    name only to make reruns agree. Each kept row is whole:
    ``groupby().first()`` fills a null column from another book's row, which
    mixed two books into one bet.
    """
    ordered = card.sort_values(
        ["edge_percentage", "sportsbook"],
        ascending=[False, True],
        na_position="last",
        kind="mergesort",
    )
    return ordered.drop_duplicates(subset=list(keys), keep="first").reset_index(drop=True)
