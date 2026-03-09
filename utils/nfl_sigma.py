"""Player-specific sigma (standard deviation) from historical variance for NFL markets.

Computes EWMA-weighted variance for each player/market combination to
produce a more accurate sigma than the flat 30% rule. Used by the prediction
layer and value engine.

Market-specific floors prevent unrealistically tight distributions.
Fallback defaults are used when a player has fewer than MIN_GAMES_FOR_SIGMA games.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# Market-specific sigma floors (minimum sigma regardless of history)
SIGMA_FLOORS: dict[str, float] = {
    "rushing_yards": 15.0,
    "receiving_yards": 12.0,
    "passing_yards": 30.0,
}

# Fallback defaults when player has fewer than MIN_GAMES_FOR_SIGMA games
SIGMA_DEFAULTS: dict[str, float] = {
    "rushing_yards": 25.0,
    "receiving_yards": 20.0,
    "passing_yards": 50.0,
}

MIN_GAMES_FOR_SIGMA = 6


def compute_player_sigma(
    game_values: Sequence[float],
    market: str = "rushing_yards",
    decay: float = 0.65,
) -> float:
    """Compute EWMA-weighted standard deviation for a player's historical values.

    Parameters
    ----------
    game_values : sequence of floats
        Historical game stat values (most recent last).
    market : str
        Market type for floor/default lookup (rushing_yards, receiving_yards, passing_yards).
    decay : float
        EWMA decay factor. Higher values weight recent games more.

    Returns
    -------
    float
        Sigma value, floored by market minimum.
    """
    arr = np.asarray(game_values, dtype=float)
    arr = arr[~np.isnan(arr)]

    floor = SIGMA_FLOORS.get(market, 10.0)
    default = SIGMA_DEFAULTS.get(market, 20.0)

    if len(arr) < MIN_GAMES_FOR_SIGMA:
        return default

    return _ewma_sigma(arr, decay=decay, floor=floor)


def _ewma_sigma(
    values: np.ndarray,
    decay: float,
    floor: float,
) -> float:
    """Compute EWMA-weighted standard deviation.

    Weights are exponentially decaying from most recent (last element)
    backwards, then normalised to sum to 1.
    """
    n = len(values)
    if n < 2:
        return floor

    # Build weights: most recent game gets highest weight
    raw_weights = np.array([decay ** i for i in range(n - 1, -1, -1)])
    weights = raw_weights / raw_weights.sum()

    weighted_mean = float(np.dot(weights, values))
    squared_diffs = (values - weighted_mean) ** 2
    weighted_var = float(np.dot(weights, squared_diffs))

    # Bessel-like correction for weighted variance
    sum_w2 = float(np.dot(weights, weights))
    correction = 1.0 / (1.0 - sum_w2) if sum_w2 < 1.0 else 1.0
    sigma = float(np.sqrt(weighted_var * correction))

    return max(sigma, floor)


def get_sigma_or_default(
    sigma_value: float | None,
    projected_value: float,
    market: str = "rushing_yards",
) -> float:
    """Return sigma from projections table, falling back to market default.

    Used by the value engine and explainability layer when sigma may be NULL.
    """
    if sigma_value is not None and not (isinstance(sigma_value, float) and np.isnan(sigma_value)):
        return float(sigma_value)

    # Fallback: market default
    return SIGMA_DEFAULTS.get(market, max(projected_value * 0.25, 10.0))
