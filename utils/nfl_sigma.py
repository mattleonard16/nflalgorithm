"""Player-specific sigma (standard deviation) from historical variance for NFL markets.

Computes EWMA-weighted variance for each player/market combination to
produce a more accurate sigma than the flat 30% rule. Used by the prediction
layer and value engine.

Floors and fallback defaults are keyed by (market, position_bucket).
The ``None`` bucket carries the legacy per-market values, so callers that do
not pass a position get exactly the pre-recalibration behavior.

Position-specific values derive from the full-season 2025 walk-forward
backtest (5,117 predictions, retrain-per-week; see
``reports/nfl_backtest_2025_baseline.json`` ``by_market_position`` and
``make nfl-backtest``). Per bucket, the applied scale factor is the 68.27th
percentile of |z| — the multiplier that puts mu +/- 1 sigma coverage exactly
at nominal. Only buckets whose measured coverage deviated from 68.3% by at
least two standard errors were changed; floor and default scale by the same
factor as the EWMA multiplier so all three paths widen together.

- passing_yards QB: 58.2% coverage (5 SE below nominal) -> 1.29x. This
  bucket was previously an unvalidated copy of the legacy numbers; it is
  now measured.
- receiving_yards: TE 63.0% -> 1.14x; RB 63.6% -> 1.08x; WR 71.7%
  (slightly over-wide) -> 0.94x, taking the former 1.20 fat-tail
  multiplier to 1.13.
- rushing_yards: QB 63.1% -> 1.13x; RB 67.0% is within noise of nominal
  (kept); WR/TE rushing kept (TE has 11 samples, WR none with sigma).

Caveat: these factors are fit on the same 2025 season the backtest scores,
so they are in-sample for 2025. They are one-dimensional scale constants
per bucket (low overfit risk), but 2026 walk-forward results are the real
validation. Re-run ``make nfl-backtest`` before re-tuning.
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
    # rushing_yards (2025 walk-forward coverage: RB 67.0%, QB 63.1%)
    ("rushing_yards", "RB"): 15.0,   # within noise of nominal — kept
    ("rushing_yards", "QB"): 13.5,   # 12.0 * 1.13
    ("rushing_yards", "WR"): 5.0,    # kept — no sigma'd samples in the backtest
    ("rushing_yards", "TE"): 5.0,    # kept — 11 samples, too few to move
    # receiving_yards (2025 walk-forward: WR 71.7%, RB 63.6%, TE 63.0%)
    ("receiving_yards", "WR"): 12.0,  # 13.0 * 0.94
    ("receiving_yards", "RB"): 11.0,  # 10.0 * 1.08
    ("receiving_yards", "TE"): 13.5,  # 12.0 * 1.14
    # passing_yards (2025 walk-forward: QB 58.2% — first measured calibration)
    ("passing_yards", "QB"): 39.0,   # 30.0 * 1.29
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
    ("rushing_yards", "QB"): 22.5,  # 20.0 * 1.13
    ("rushing_yards", "WR"): 10.0,  # kept at 2x floor for no-history uncertainty
    ("rushing_yards", "TE"): 10.0,
    # receiving_yards
    ("receiving_yards", "WR"): 22.5,  # 24.0 * 0.94
    ("receiving_yards", "RB"): 17.5,  # 16.0 * 1.08
    ("receiving_yards", "TE"): 23.0,  # 20.0 * 1.14
    # passing_yards
    ("passing_yards", "QB"): 64.5,   # 50.0 * 1.29
}

# Multiplier applied to the EWMA estimate (before flooring), per bucket the
# 68.27th percentile of |z| from the 2025 walk-forward backtest — the factor
# that lands mu +/- 1 sigma coverage on nominal. WR receiving folds the
# former 1.20 fat-tail multiplier and the measured 0.94 into one number.
SIGMA_EWMA_MULTIPLIERS: dict[_Bucket, float] = {
    ("passing_yards", "QB"): 1.29,
    ("receiving_yards", "WR"): 1.13,
    ("receiving_yards", "RB"): 1.08,
    ("receiving_yards", "TE"): 1.14,
    ("rushing_yards", "QB"): 1.13,
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
