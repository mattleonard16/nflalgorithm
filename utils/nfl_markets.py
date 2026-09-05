"""Canonical NFL prop-market mappings and shared projection-scoring helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

from sports.markets import get_sport

MARKET_TO_STAT = {market: spec.stat_column for market, spec in get_sport("nfl").markets.items()}

# Physical stat columns present in `player_stats_enhanced` that cover every
# market. `anytime_touchdown` maps to the virtual `anytime_td` column, which
# does not exist in the table — it is synthesized at read time from
# `rushing_tds` + `receiving_tds` (see `synthesize_anytime_td`). SQL loaders
# must select these columns, never `MARKET_TO_STAT.values()` directly.
DATABASE_STAT_COLUMNS = sorted(
    {stat for market, stat in MARKET_TO_STAT.items() if market != "anytime_touchdown"}
    | {"rushing_tds", "receiving_tds"}
)


def synthesize_anytime_td(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the virtual `anytime_td` count column to a player-stat frame.

    Anytime touchdowns are stored as two physical columns (`rushing_tds`,
    `receiving_tds`); the count is their sum. Either column may be absent
    (e.g. a partial select) — the missing side is treated as zeros. A frame
    with neither column is returned unchanged. An already-present `anytime_td`
    column is coerced to integer counts rather than recomputed.

    Pure: never mutates the input.
    """
    out = df.copy()
    if "anytime_td" in out.columns:
        out["anytime_td"] = (
            pd.to_numeric(out["anytime_td"], errors="coerce").fillna(0).astype(int)
        )
        return out
    if "rushing_tds" not in out.columns and "receiving_tds" not in out.columns:
        return out
    if "rushing_tds" in out.columns:
        rush = pd.to_numeric(out["rushing_tds"], errors="coerce").fillna(0.0)
    else:
        rush = pd.Series(0.0, index=out.index)
    if "receiving_tds" in out.columns:
        rec = pd.to_numeric(out["receiving_tds"], errors="coerce").fillna(0.0)
    else:
        rec = pd.Series(0.0, index=out.index)
    out["anytime_td"] = (rush + rec).astype(int)
    return out


def prob_over(mu: float, sigma: float, line: float, market: str | None = None) -> float:
    """Probability of going OVER a line.

    Continuous yardage/reception props use the Gaussian normal CDF
    ``1 - Phi((line - mu) / sigma)``. Touchdown props are a small count, not a
    continuous quantity, so they price off Poisson survival instead, which
    avoids Gaussian tail distortion. The line still chooses the threshold:
    ``P(X > floor(line))``, so 0.5 asks for 1+ and 1.5 asks for 2+. Answering
    1+ for every touchdown line regardless would roughly double the price of a
    1.5 line.

    Lives in this tracked module (rather than gitignored
    ``value_betting_engine.py``) so public market tests run in clean CI
    checkouts; the engine re-exports it for backward compatibility.
    """
    if market is not None and "touchdown" in market:
        return float(poisson.sf(math.floor(line), max(0.0, mu)))
    return float(1 - norm.cdf(line, loc=mu, scale=sigma))


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
    df_actuals = synthesize_anytime_td(actuals)

    for market, stat in MARKET_TO_STAT.items():
        if stat not in df_actuals.columns:
            continue
        part = df_actuals[keys + [stat]].rename(columns={stat: "actual"}).copy()
        part["market"] = market
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=keys + ["actual", "market"])
    return pd.concat(rows, ignore_index=True)
