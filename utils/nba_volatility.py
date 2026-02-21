"""NBA-calibrated volatility scoring.

Wraps the core ``utils/volatility_scoring.py`` functions with NBA-specific
normalisation denominators per market. NBA stats have different scale and
variance characteristics than NFL yardage.

CV denominator by market:
  pts=0.8, reb=0.7, ast=0.8, fg3m=1.2
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from utils.volatility_scoring import (
    coefficient_of_variation,
    max_week_contribution,
    range_ratio,
)


# NBA-calibrated normalisation denominators per market
# Lower denom = more sensitive to CV (flags volatility sooner)
_CV_DENOM: dict[str, float] = {
    "pts": 0.8,
    "reb": 0.7,
    "ast": 0.8,
    "fg3m": 1.2,
}

# max-week-contribution and range-ratio normalisation constants
_MAX_CONTRIB_DENOM = 0.5
_RANGE_RATIO_DENOM = 3.0


def compute_nba_volatility_score(
    game_values: Sequence[float],
    market: str = "pts",
    *,
    cv_weight: float = 0.45,
    max_contrib_weight: float = 0.30,
    range_weight: float = 0.25,
) -> float:
    """Compute a 0-100 volatility score for an NBA player/market.

    0 = very stable (consistent game-to-game production)
    100 = extremely volatile (boom-bust)

    Parameters
    ----------
    game_values : sequence of floats
        Historical game stat values (at least 2 needed).
    market : str
        NBA market for calibrated normalisation (pts, reb, ast, fg3m).
    cv_weight, max_contrib_weight, range_weight : float
        Component weights (should sum to 1.0).

    Returns
    -------
    float
        Volatility score between 0 and 100.
    """
    arr = np.asarray(game_values, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) < 2:
        return 50.0  # insufficient data = neutral score

    cv = coefficient_of_variation(arr)
    max_contrib = max_week_contribution(arr)
    rr = range_ratio(arr)

    cv_denom = _CV_DENOM.get(market, 0.8)
    cv_norm = min(cv / cv_denom, 1.0)
    contrib_norm = min(max_contrib / _MAX_CONTRIB_DENOM, 1.0)
    rr_norm = min(rr / _RANGE_RATIO_DENOM, 1.0)

    raw = (
        cv_weight * cv_norm
        + max_contrib_weight * contrib_norm
        + range_weight * rr_norm
    )

    return round(min(max(raw * 100, 0.0), 100.0), 2)
