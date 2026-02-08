"""Volatility scoring for NFL player projections.

Computes a boom-bust volatility score (0-100) from historical weekly
yardage data.  High scores indicate players whose production is driven
by a few explosive plays, making projections less reliable.

Inputs
------
* Weekly yardage totals (from ``player_stats_enhanced``)
* Optional per-play yardage breakdown for single-play concentration

Scoring components
------------------
1. Coefficient of variation (CV) of weekly yardage
2. Max single-week contribution percentage
3. Range ratio  (max - min) / mean
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from config import config


def coefficient_of_variation(values: Sequence[float]) -> float:
    """Return CV (stddev / mean). Returns 0 when mean is zero."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return 0.0
    mean = float(np.mean(arr))
    if mean == 0:
        return 0.0
    return float(np.std(arr, ddof=1) / abs(mean))


def max_week_contribution(values: Sequence[float]) -> float:
    """Fraction of total yardage contributed by the single best week.

    Returns 0 when total is zero.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0
    total = float(np.sum(arr))
    if total <= 0:
        return 0.0
    return float(np.max(arr) / total)


def range_ratio(values: Sequence[float]) -> float:
    """(max - min) / mean.  Returns 0 when mean is zero."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return 0.0
    mean = float(np.mean(arr))
    if mean == 0:
        return 0.0
    return float((np.max(arr) - np.min(arr)) / abs(mean))


def compute_volatility_score(
    weekly_yards: Sequence[float],
    *,
    cv_weight: float = 0.45,
    max_contrib_weight: float = 0.30,
    range_weight: float = 0.25,
) -> float:
    """Compute a 0-100 volatility score from weekly yardage totals.

    0 = very stable (consistent week-to-week production)
    100 = extremely volatile (boom-bust)

    Parameters
    ----------
    weekly_yards : sequence of floats
        Historical weekly yardage totals (at least 2 needed).
    cv_weight, max_contrib_weight, range_weight : float
        Component weights (should sum to 1.0).
    """
    arr = np.asarray(weekly_yards, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) < 2:
        return 50.0  # insufficient data = neutral score

    cv = coefficient_of_variation(arr)
    max_contrib = max_week_contribution(arr)
    rr = range_ratio(arr)

    # Normalize each component to roughly 0-1 via sigmoid-like clamping
    cv_norm = min(cv / 1.0, 1.0)  # CV of 1.0+ is extreme
    contrib_norm = min(max_contrib / 0.5, 1.0)  # 50%+ in one week is extreme
    rr_norm = min(rr / 3.0, 1.0)  # range_ratio of 3.0+ is extreme

    raw = (
        cv_weight * cv_norm
        + max_contrib_weight * contrib_norm
        + range_weight * rr_norm
    )

    return round(min(max(raw * 100, 0.0), 100.0), 2)


def widen_sigma_for_volatility(
    sigma: float,
    volatility_score: float,
    penalty_weight: float | None = None,
) -> float:
    """Widen sigma proportionally to the volatility score.

    A volatility_score of 0 leaves sigma untouched.
    A score of 100 increases sigma by ``penalty_weight`` (default from config).

    Formula: sigma_adj = sigma * (1 + penalty_weight * volatility_score / 100)
    """
    if penalty_weight is None:
        penalty_weight = getattr(
            config.betting, "volatility_penalty_weight", 0.15
        )
    multiplier = 1.0 + penalty_weight * (volatility_score / 100.0)
    return sigma * multiplier
