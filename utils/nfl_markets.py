"""Canonical NFL prop-market mappings and shared projection-scoring helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sports.markets import get_sport

MARKET_TO_STAT = {market: spec.stat_column for market, spec in get_sport("nfl").markets.items()}


def player_positions(
    actuals: pd.DataFrame, fill_missing: str | None = None
) -> pd.DataFrame:
    """One uppercased position per player-week, for grouping projection errors.

    ``fill_missing`` replaces NaN positions (e.g. with "UNKNOWN"); None keeps
    them NaN for the caller to handle at grouping time.
    """
    keys = ["season", "week", "player_id"]
    if actuals.empty or "position" not in actuals.columns:
        return pd.DataFrame(columns=keys + ["position"])
    positions = actuals[keys + ["position"]].drop_duplicates(keys, keep="last").copy()
    upper = positions["position"].astype("string").str.upper()
    positions["position"] = upper if fill_missing is None else upper.fillna(fill_missing)
    return positions


def error_summary(rows: pd.DataFrame) -> dict[str, float]:
    """MAE / RMSE / mean bias from a frame carrying signed_error and abs_error.

    Callers guard against empty frames; NaN-on-empty here would leak into
    reports as null metrics that look computed.
    """
    return {
        "mae": float(rows["abs_error"].mean()),
        "rmse": float(np.sqrt(np.mean(np.square(rows["signed_error"])))),
        "mean_bias": float(rows["signed_error"].mean()),
    }


def melt_actuals(actuals: pd.DataFrame) -> pd.DataFrame:
    """Reshape player-week actuals to one row per supported prop market."""
    keys = ["season", "week", "player_id"]
    rows: list[pd.DataFrame] = []
    df_actuals = actuals.copy()
    if not df_actuals.empty and "anytime_td" not in df_actuals.columns:
        if "rushing_tds" in df_actuals.columns or "receiving_tds" in df_actuals.columns:
            rush = pd.to_numeric(df_actuals.get("rushing_tds", 0), errors="coerce").fillna(0.0)
            rec = pd.to_numeric(df_actuals.get("receiving_tds", 0), errors="coerce").fillna(0.0)
            df_actuals["anytime_td"] = (rush + rec > 0).astype(int)

    for market, stat in MARKET_TO_STAT.items():
        if stat not in df_actuals.columns:
            continue
        part = df_actuals[keys + [stat]].rename(columns={stat: "actual"}).copy()
        part["market"] = market
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=keys + ["actual", "market"])
    return pd.concat(rows, ignore_index=True)
