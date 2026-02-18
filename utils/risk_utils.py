"""Sport-agnostic risk utility functions.

Shared math helpers used by both NFL ``risk_manager`` and NBA
``nba_risk_manager``.  Extracting these avoids duplicating Monte Carlo
simulation, Kelly adjustment, and warning-append logic.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from config import config


def monte_carlo_drawdown(
    kelly_fracs: np.ndarray,
    win_probs: np.ndarray,
    odds: np.ndarray,
    iterations: Optional[int] = None,
) -> Dict[str, float]:
    """Simulate correlated bankroll paths and compute max drawdown stats.

    Parameters
    ----------
    kelly_fracs : array of Kelly fractions per bet
    win_probs   : array of win probabilities per bet
    odds        : array of American odds per bet
    iterations  : number of Monte Carlo iterations (default from config)

    Returns
    -------
    dict with keys ``mean_drawdown``, ``max_drawdown``, ``p95_drawdown``.
    """
    n_iters = iterations or config.risk.monte_carlo_iterations
    n_bets = len(kelly_fracs)

    if n_bets == 0:
        return {"mean_drawdown": 0.0, "max_drawdown": 0.0, "p95_drawdown": 0.0}

    rng = np.random.default_rng(42)
    outcomes = rng.random((n_iters, n_bets)) < win_probs

    safe_odds = np.where(odds == 0, 1, odds)
    payouts = np.where(
        safe_odds < 0,
        100.0 / np.abs(safe_odds),
        safe_odds / 100.0,
    )
    payouts = np.where(odds == 0, 0.0, payouts)

    pnl = np.where(outcomes, kelly_fracs * payouts, -kelly_fracs)
    cumulative = np.cumsum(pnl, axis=1)
    running_max = np.maximum.accumulate(cumulative, axis=1)
    drawdowns = running_max - cumulative
    max_dd_per_path = drawdowns.max(axis=1)

    return {
        "mean_drawdown": float(np.mean(max_dd_per_path)),
        "max_drawdown": float(np.max(max_dd_per_path)),
        "p95_drawdown": float(np.percentile(max_dd_per_path, 95)),
    }


def risk_adjusted_kelly(
    kelly_frac: float,
    drawdown_stats: Dict[str, float],
) -> float:
    """Scale Kelly fraction down when max drawdown exceeds threshold."""
    threshold = config.risk.max_drawdown_threshold
    p95 = drawdown_stats.get("p95_drawdown", 0.0)
    if p95 <= 0 or threshold <= 0:
        return kelly_frac
    if p95 > threshold:
        scale = threshold / p95
        return kelly_frac * scale
    return kelly_frac


def append_warning(existing: Optional[str], new: str) -> str:
    """Append a warning string, joining with semicolons."""
    if existing is None:
        return new
    return f"{existing}; {new}"
