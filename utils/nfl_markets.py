"""Canonical NFL prop-market mappings and outcome reshaping."""

from __future__ import annotations

import pandas as pd

MARKET_TO_STAT = {
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "passing_yards": "passing_yards",
    "receptions": "receptions",
    "targets": "targets",
}


def melt_actuals(actuals: pd.DataFrame) -> pd.DataFrame:
    """Reshape player-week actuals to one row per supported prop market."""
    keys = ["season", "week", "player_id"]
    rows: list[pd.DataFrame] = []
    for market, stat in MARKET_TO_STAT.items():
        if stat not in actuals.columns:
            continue
        part = actuals[keys + [stat]].rename(columns={stat: "actual"}).copy()
        part["market"] = market
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=keys + ["actual", "market"])
    return pd.concat(rows, ignore_index=True)
