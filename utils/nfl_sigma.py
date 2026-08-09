"""Player-specific sigma (standard deviation) from historical variance for NFL markets.

Computes EWMA-weighted variance for each player/market combination to
produce a more accurate sigma than the flat 30% rule. Used by the prediction
layer and value engine.

Floors and fallback defaults are keyed by (market, position_bucket).
The ``None`` bucket carries the legacy per-market values, so callers that do
not pass a position get exactly the pre-recalibration behavior.

Position-specific values derive from a held-out 2025 coverage study
(1,933 predictions; target mu +/- 1 sigma coverage 68.3%):

- rushing_yards:  RB 71.5% (near nominal, keep floor), QB 80.5% (~1.3x wide),
  WR 95.6% / TE 95.1% (>=2x wide; these players' rushing values cluster
  0-5 yards, so the legacy 15.0 floor dominated and was far too large).
- receiving_yards: TE 68.3% (nominal), RB 79.4% (~1.27x wide),
  WR 66.6% but z-score std 1.25-1.30 (tails fatter than Gaussian, sigma
  modestly too small) -> floor raised slightly and a 1.20 multiplier is
  applied to the EWMA estimate.
- passing_yards: no coverage measurements exist; values are UNVALIDATED
  copies of the legacy per-market numbers.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# (market, position_bucket) -> value. Bucket ``None`` = legacy per-market
# behavior, used when position is not supplied or not a known bucket.
_Bucket = tuple[str, str | None]

# Market/position sigma floors (minimum sigma regardless of history)
SIGMA_FLOORS: dict[_Bucket, float] = {
    # Legacy per-market floors (position=None path) — unchanged.
    ("rushing_yards", None): 15.0,
    ("receiving_yards", None): 12.0,
    ("passing_yards", None): 30.0,
    # rushing_yards (2025 coverage: RB 71.5%, QB 80.5%, WR 95.6%, TE 95.1%)
    ("rushing_yards", "RB"): 15.0,  # near nominal — keep legacy floor
    ("rushing_yards", "QB"): 12.0,  # ~1.3x too wide -> 15.0 / 1.3
    ("rushing_yards", "WR"): 5.0,   # floor-dominated, >=2x too wide; values cluster 0-5
    ("rushing_yards", "TE"): 5.0,   # same regime as WR rushing
    # receiving_yards (2025 coverage: WR 66.6% z-std 1.25-1.30, RB 79.4%, TE 68.3%)
    ("receiving_yards", "WR"): 13.0,  # slight raise for fat tails (see multiplier)
    ("receiving_yards", "RB"): 10.0,  # ~1.27x too wide -> 12.0 / 1.27
    ("receiving_yards", "TE"): 12.0,  # nominal — keep legacy floor
    # passing_yards: UNVALIDATED — no coverage measurements; legacy value kept.
    ("passing_yards", "QB"): 30.0,
}

# Fallback defaults when player has fewer than MIN_GAMES_FOR_SIGMA games,
# scaled by the same coverage-implied ratios as the floors above.
SIGMA_DEFAULTS: dict[_Bucket, float] = {
    # Legacy per-market defaults (position=None path) — unchanged.
    ("rushing_yards", None): 25.0,
    ("receiving_yards", None): 20.0,
    ("passing_yards", None): 50.0,
    # rushing_yards
    ("rushing_yards", "RB"): 25.0,
    ("rushing_yards", "QB"): 20.0,  # 25.0 / 1.3
    ("rushing_yards", "WR"): 10.0,  # scaled ~2.5x down; kept at 2x floor for no-history uncertainty
    ("rushing_yards", "TE"): 10.0,
    # receiving_yards
    ("receiving_yards", "WR"): 24.0,  # 20.0 * 1.20 fat-tail multiplier
    ("receiving_yards", "RB"): 16.0,  # 20.0 / 1.27
    ("receiving_yards", "TE"): 20.0,
    # passing_yards: UNVALIDATED — no coverage measurements; legacy value kept.
    ("passing_yards", "QB"): 50.0,
}

# Multiplier applied to the EWMA estimate (before flooring). WR receiving
# z-score std of 1.25-1.30 on held-out data means the EWMA sigma is modestly
# too small in the tails; 1.20 is the midpoint of the measured band.
SIGMA_EWMA_MULTIPLIERS: dict[_Bucket, float] = {
    ("receiving_yards", "WR"): 1.20,
}

_GENERIC_FLOOR = 10.0
_GENERIC_DEFAULT = 20.0

MIN_GAMES_FOR_SIGMA = 6

_KNOWN_BUCKETS = frozenset({"QB", "RB", "WR", "TE"})


def _position_bucket(position: str | None) -> str | None:
    """Normalise a position string to a known bucket, or None (legacy path)."""
    if position is None:
        return None
    pos = position.strip().upper()
    return pos if pos in _KNOWN_BUCKETS else None


def _lookup(table: dict[_Bucket, float], market: str, bucket: str | None, generic: float) -> float:
    """Bucket-specific value, falling back to (market, None), then generic."""
    if bucket is not None and (market, bucket) in table:
        return table[(market, bucket)]
    return table.get((market, None), generic)


def compute_player_sigma(
    game_values: Sequence[float],
    market: str = "rushing_yards",
    position: str | None = None,
    decay: float = 0.65,
) -> float:
    """Compute EWMA-weighted standard deviation for a player's historical values.

    Parameters
    ----------
    game_values : sequence of floats
        Historical game stat values (most recent last).
    market : str
        Market type for floor/default lookup (rushing_yards, receiving_yards, passing_yards).
    position : str or None
        Player position (QB/RB/WR/TE) for position-bucketed calibration.
        None (or an unknown position) reproduces the legacy per-market behavior.
    decay : float
        EWMA decay factor. Higher values weight recent games more.

    Returns
    -------
    float
        Sigma value, floored by the (market, position) minimum.
    """
    arr = np.asarray(game_values, dtype=float)
    arr = arr[~np.isnan(arr)]

    bucket = _position_bucket(position)
    floor = _lookup(SIGMA_FLOORS, market, bucket, _GENERIC_FLOOR)
    default = _lookup(SIGMA_DEFAULTS, market, bucket, _GENERIC_DEFAULT)
    multiplier = _lookup(SIGMA_EWMA_MULTIPLIERS, market, bucket, 1.0)

    if len(arr) < MIN_GAMES_FOR_SIGMA:
        return default

    return max(_ewma_sigma(arr, decay=decay) * multiplier, floor)


def _ewma_sigma(values: np.ndarray, decay: float) -> float:
    """Compute the raw EWMA-weighted standard deviation (no floor applied).

    Weights are exponentially decaying from most recent (last element)
    backwards, then normalised to sum to 1.
    """
    n = len(values)
    if n < 2:
        return 0.0

    # Build weights: most recent game gets highest weight
    raw_weights = np.array([decay ** i for i in range(n - 1, -1, -1)])
    weights = raw_weights / raw_weights.sum()

    weighted_mean = float(np.dot(weights, values))
    squared_diffs = (values - weighted_mean) ** 2
    weighted_var = float(np.dot(weights, squared_diffs))

    # Bessel-like correction for weighted variance
    sum_w2 = float(np.dot(weights, weights))
    correction = 1.0 / (1.0 - sum_w2) if sum_w2 < 1.0 else 1.0
    return float(np.sqrt(weighted_var * correction))
