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
