"""Canonical NFL prop-market mappings and shared projection-scoring helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm, poisson

from sports.markets import get_sport

MARKET_TO_STAT = {market: spec.stat_column for market, spec in get_sport("nfl").markets.items()}

# Markets priced off a gamma distribution instead of a normal one. Weekly
# yardage is right-skewed: a receiver clears his own season mean about 39% of
# the time, so a normal centred on mu calls every over a coin flip and
# overstates it. Measured on the 2025 walk-forward rows
# (`reports/nfl_backtest_2025_sigma_v2_rows.csv`, 25,585 priced lines) the
# normal curve ran +10.84pp long on receiving_yards, +7.53pp on rushing_yards
# and +4.43pp on passing_yards. Gamma matched to the same mu and sigma cut
# those to +1.46pp, -0.92pp and +0.69pp, and won on Brier score in all six
# market-position buckets.
#
# `receptions` and `targets` are deliberately absent. They are counts, and the
# walk-forward output carries no rows for either, so there is nothing to
# calibrate a replacement against yet. `anytime_touchdown` keeps the Poisson
# branch in `prob_over`.
GAMMA_MARKETS = frozenset({"passing_yards", "rushing_yards", "receiving_yards"})

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

    Three curves, chosen by market:

    Yardage markets (``GAMMA_MARKETS``) use gamma survival, matched to the
    caller's mu and sigma by method of moments: shape ``(mu / sigma) ** 2``,
    scale ``sigma ** 2 / mu``. Mean and variance come out equal to mu and
    sigma squared, so sigma keeps the meaning ``utils/nfl_sigma.py`` calibrated
    it to, and only the shape changes. Gamma is bounded at zero and
    right-skewed, which is what a weekly yardage line looks like. See
    ``GAMMA_MARKETS`` for the measured bias it removes.

    Touchdown props are a small count, not a continuous quantity, so they price
    off Poisson survival, which avoids Gaussian tail distortion. The line still
    chooses the threshold: ``P(X > floor(line))``, so 0.5 asks for 1+ and 1.5
    asks for 2+. Answering 1+ for every touchdown line regardless would roughly
    double the price of a 1.5 line.

    Everything else, and any yardage row with a non-positive mu where gamma is
    undefined, falls back to the normal CDF ``1 - Phi((line - mu) / sigma)``.

    A non-positive sigma is a degenerate distribution, a point mass at mu, so it
    prices 1.0 above the line and 0.0 at or below it. ``scipy`` divides by the
    scale and would return NaN, which a caller then averages into a report or
    compares against an edge threshold, where it silently drops the row instead
    of failing. ``nba_value_engine.prob_over`` already uses this convention.
    ``compute_player_sigma`` floors every bucket, so production never reaches
    here; bad input and direct callers do.

    Lives in this tracked module (rather than gitignored
    ``value_betting_engine.py``) so public market tests run in clean CI
    checkouts; the engine re-exports it for backward compatibility.
    """
    if market is not None and "touchdown" in market:
        return float(poisson.sf(math.floor(line), max(0.0, mu)))
    if sigma <= 0:
        return 1.0 if mu > line else 0.0
    if market in GAMMA_MARKETS and mu > 0:
        return float(gamma.sf(line, a=(mu / sigma) ** 2, scale=sigma**2 / mu))
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
