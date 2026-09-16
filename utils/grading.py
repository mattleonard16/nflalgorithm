"""Shared grading utilities for bet outcome evaluation.

Used by both NFL (scripts/record_outcomes.py) and NBA (scripts/record_nba_outcomes.py)
grading pipelines.
"""

from __future__ import annotations

import pandas as pd


def grade_bet(actual: float, line: float, side: str) -> str:
    """Grade a single bet based on actual result vs line.

    Args:
        actual: Actual stat value
        line: Bet line
        side: Bet side ('over' or 'under')

    Returns:
        Result: 'win', 'loss', or 'push'
    """
    if pd.isna(actual):
        return "push"

    if actual == line:
        return "push"

    if side.lower() == "over":
        return "win" if actual > line else "loss"
    else:
        return "win" if actual < line else "loss"


def calculate_profit_units(result: str, price: int) -> float:
    """Calculate profit in units for a bet.

    Args:
        result: Bet result ('win', 'loss', or 'push')
        price: American odds (e.g., -110, +150)

    Returns:
        Profit in units (1 unit = stake)
    """
    if result == "push":
        return 0.0
    elif result == "loss":
        return -1.0
    elif result == "win":
        if price < 0:
            return 100.0 / abs(price)
        else:
            return price / 100.0
    else:
        return 0.0


def get_confidence_tier(edge_percentage: float) -> str:
    """Determine confidence tier based on edge percentage.

    Args:
        edge_percentage: Edge percentage at time of placement

    Returns:
        Confidence tier: HIGH, MEDIUM, LOW, or MINIMAL
    """
    if edge_percentage >= 15.0:
        return "HIGH"
    elif edge_percentage >= 8.0:
        return "MEDIUM"
    elif edge_percentage >= 3.0:
        return "LOW"
    else:
        return "MINIMAL"


def align_actuals_to_bets(actuals: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Re-key actual stat rows to the player ids the bets are stored under.

    Two tables mint ``player_id`` from different spellings of the same name and
    the two never match. ``scripts/ingest_real_nfl_data.transform_to_enhanced_stats``
    builds it from nflverse's abbreviated ``player_name`` ("M.Stafford"), giving
    ``LAR_m_stafford``, while ``upsert_roster_players`` builds it from the full
    ``player_name`` on the roster feed, giving ``LAR_matthew_stafford``. Bets
    carry the roster form, so grading matched nothing and recorded every bet as
    a push with zero profit. That is indistinguishable from a settled push once
    it lands in ``bet_outcomes`` and the CLV average.

    Both tables also carry nflverse's ``gsis_id``, which is stable across name
    spellings, suffixes and mid-season trades, so that is what this joins on.
    It is unique per season on the roster and per player-week in the stats, so
    the mapping is unambiguous.

    A stat row whose ``gsis_id`` is missing from the roster keeps the id it
    already has rather than being dropped: some feeds carry players the roster
    snapshot does not, and an id that was already aligned still matches.
    """
    for column in ("player_id", "gsis_id"):
        if column not in actuals.columns:
            raise ValueError(f"actuals missing required column: {column}")
        if column not in roster.columns:
            raise ValueError(f"roster missing required column: {column}")

    if actuals.empty or roster.empty:
        return actuals

    usable = roster[["gsis_id", "player_id"]].copy()
    usable["gsis_id"] = usable["gsis_id"].fillna("").astype(str).str.strip()
    usable = usable[usable["gsis_id"] != ""].drop_duplicates("gsis_id")
    bet_id_by_gsis = dict(zip(usable["gsis_id"], usable["player_id"]))

    out = actuals.copy()
    keys = out["gsis_id"].fillna("").astype(str).str.strip()
    out["player_id"] = [
        bet_id_by_gsis.get(key, current) for key, current in zip(keys, out["player_id"])
    ]
    return out
