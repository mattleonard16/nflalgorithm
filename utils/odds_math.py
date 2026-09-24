"""American-odds conversions shared by the NFL and NBA engines and CLV grading.

This lives in a tracked module so CI can import it. It used to sit in gitignored
``value_betting_engine``, which made ``utils/clv.py`` import it lazily and every
CLV test that reached the two-sided path skip in a clean checkout.
"""

from __future__ import annotations


def implied_probability(odds: int) -> float:
    """Convert American odds to implied win probability, vig included."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def implied_probability_no_vig(over_odds: int, under_odds: int) -> tuple[float, float]:
    """Return (p_over, p_under) with vig removed by normalizing to sum=1.0.

    Two-sided book quotes sum to more than 1.0 because of the bookmaker margin.
    Dividing each raw implied probability by their sum removes it and yields
    fair probabilities that sum to exactly 1.0.

    >>> p_o, p_u = implied_probability_no_vig(-110, -110)
    >>> abs(p_o - 0.5) < 1e-9
    True
    """
    raw_over = implied_probability(over_odds)
    raw_under = implied_probability(under_odds)
    total = raw_over + raw_under
    return raw_over / total, raw_under / total
