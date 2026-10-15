"""American-odds conversions shared by the NFL and NBA pricing paths.

Tracked so CI can test the no-vig math. ``utils/clv.py`` grades closing line
value with it, ``nba_value_engine`` prices with it, and the gitignored
``value_betting_engine`` should import it from here rather than keep a copy.
"""

from __future__ import annotations


def _check_american(odds: float) -> None:
    """Reject values that are not American odds.

    American odds are at or below -100 or at or above +100; nothing lies
    strictly between. A 0 or 50 is a bad cell, and converting it anyway
    returns a plausible-looking probability (50 -> 0.667) that prices a bet.
    The negated test also rejects NaN.
    """
    if not (odds <= -100 or odds >= 100):
        raise ValueError(f"not valid American odds (must be <= -100 or >= +100): {odds!r}")


def implied_probability(odds: int) -> float:
    """Convert American odds to implied win probability (no vig removed).

    Raises ``ValueError`` for values strictly between -100 and +100, or NaN.
    """
    _check_american(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def implied_probability_no_vig(over_odds: int, under_odds: int) -> tuple[float, float]:
    """Return (p_over, p_under) with vig removed by normalizing to sum=1.0.

    Raw book probabilities sum to more than 1.0 because of the bookmaker's
    margin (vig). Dividing each by their sum gives fair-market probabilities
    that sum to exactly 1.0.
    """
    raw_over = implied_probability(over_odds)
    raw_under = implied_probability(under_odds)
    total = raw_over + raw_under
    if total <= 0:
        raise ValueError("Implied probabilities summed to non-positive value")
    return raw_over / total, raw_under / total


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal (European) odds.

    Raises ``ValueError`` for values strictly between -100 and +100, or NaN.
    """
    _check_american(odds)
    if odds < 0:
        return 1 + 100 / abs(odds)
    return 1 + odds / 100
