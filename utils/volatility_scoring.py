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
import pandas as pd

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


def apply_volatility_widening(
    sigma: pd.Series,
    volatility_score: pd.Series | None,
    *,
    penalty_weight: float | None = None,
) -> tuple[pd.Series, int]:
    """Widen a column of sigmas, treating an absent score as no information.

    Returns the widened sigmas and the number of rows that had no score.

    The distinction this encodes: a *measured* score of 50 means the player
    really is middling-volatile and has earned a 7.5% penalty. A *missing*
    score means nothing was measured, and inflating sigma for it prices
    ignorance as if it were risk. Both cases previously collapsed onto the
    same ``fillna(50.0)``.

    That mattered more than it looks: ``weekly_projections.volatility_score``
    has never been written by the NFL model (0 of 1964 rows), so every NFL
    sigma was being multiplied by a constant 1.075 — a uniform inflation of
    every p_win and edge, dressed up as risk sensitivity. Rows with no score
    are now left untouched, and the caller is told how many there were so a
    silently unpopulated column cannot masquerade as a calibrated one.
    """
    sigma_numeric = pd.to_numeric(sigma, errors="coerce")

    if volatility_score is None:
        return sigma_numeric, len(sigma_numeric)

    scores = pd.to_numeric(volatility_score, errors="coerce")
    missing = scores.isna()

    widened = sigma_numeric * (
        1.0
        + _resolve_penalty_weight(penalty_weight) * (scores.fillna(0.0) / 100.0)
    )
    return widened, int(missing.sum())


def _resolve_penalty_weight(penalty_weight: float | None) -> float:
    if penalty_weight is not None:
        return penalty_weight
    return getattr(config.betting, "volatility_penalty_weight", 0.15)
