"""Monte Carlo simulation utilities for NBA player prop probabilities.

Provides simulation-based alternatives to analytic CDF approximations,
enabling richer uncertainty modelling (e.g. minutes × rate decomposition,
correlated multi-stat portfolios, Gamma-Poisson for fg3m).

All functions are pure (no DB access) and scipy-free.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Rate markets that support minutes × rate decomposition
_RATE_MARKETS = {"pts", "reb", "ast"}

# Minutes are clipped to this range before simulation
_MINUTES_MIN = 0.0
_MINUTES_MAX = 48.0

# Regularisation added to correlation matrix before Cholesky
_CHOL_EPS = 1e-6

# Seed used when the caller passes neither ``rng`` nor ``seed``, so the same
# inputs give the same probabilities run to run (re-materializing a slate must
# not flip bets across the edge threshold on simulation noise). Matches the
# fixed seed in ``utils/risk_utils.monte_carlo_drawdown``.
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Core simulation: single-variable and two-variable (minutes × rate)
# ---------------------------------------------------------------------------


def monte_carlo_prob(
    mu: float,
    sigma: float,
    line: float,
    market: str,
    n_sims: int = 10_000,
    rng: np.random.Generator | None = None,
    minutes_mu: float | None = None,
    minutes_sigma: float | None = None,
    seed: int | None = DEFAULT_SEED,
) -> tuple[float, float]:
    """Simulate P(stat > line) and P(stat < line) via Monte Carlo.

    Parameters
    ----------
    mu : float
        Model projection (mean of the stat).
    sigma : float
        Model uncertainty (std-dev of the stat).
    line : float
        Prop line to evaluate.
    market : str
        One of pts/reb/ast/fg3m.
    n_sims : int
        Number of simulation draws.
    rng : numpy Generator or None
        RNG to draw from. When None, ``np.random.default_rng(seed)`` is used.
    minutes_mu : float or None
        Mean predicted minutes.  Enables minutes × rate decomposition for
        rate markets (pts/reb/ast) when provided together with minutes_sigma.
    minutes_sigma : float or None
        Std-dev of predicted minutes.
    seed : int or None
        Seed for the RNG built when ``rng`` is None. Defaults to
        ``DEFAULT_SEED`` (deterministic); pass None for fresh entropy.

    Returns
    -------
    (p_over, p_under) : tuple[float, float]
        Probabilities that always sum to exactly 1.0.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    use_minutes = (
        market in _RATE_MARKETS
        and minutes_mu is not None
        and minutes_sigma is not None
        and minutes_mu > 0
    )

    if market == "fg3m":
        # Simple Poisson simulation to match the analytic prob_over_poisson baseline.
        # (The Gamma-Poisson compound is reserved for portfolio correlation simulation.)
        samples = _simulate_fg3m_simple(mu, n_sims, rng)
    elif use_minutes:
        samples = _simulate_rate_market(
            mu=mu,
            sigma=sigma,
            minutes_mu=float(minutes_mu),
            minutes_sigma=float(minutes_sigma),
            n_sims=n_sims,
            rng=rng,
        )
    else:
        if sigma <= 0:
            val = mu
            p_over = 0.0 if val <= line else 1.0
            return p_over, 1.0 - p_over
        samples = rng.normal(loc=mu, scale=sigma, size=n_sims)

    p_over = float(np.mean(samples > line))
    p_under = 1.0 - p_over
    return p_over, p_under


def _simulate_fg3m_simple(mu: float, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """Simple Poisson simulation for fg3m — matches analytic prob_over_poisson baseline."""
    if mu <= 0:
        return np.zeros(n_sims)
    return rng.poisson(lam=mu, size=n_sims).astype(float)


def _simulate_rate_market(
    mu: float,
    sigma: float,
    minutes_mu: float,
    minutes_sigma: float,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate stat = rate × minutes with joint uncertainty.

    rate_mu  = mu / minutes_mu
    rate_sigma = sigma / minutes_mu   (propagate uncertainty from overall sigma)
    """
    rate_mu = mu / minutes_mu
    rate_sigma = max(sigma / minutes_mu, 1e-8) if minutes_mu > 0 else 1e-8

    rate_samples = rng.normal(loc=rate_mu, scale=rate_sigma, size=n_sims)
    min_samples = np.clip(
        rng.normal(loc=minutes_mu, scale=max(minutes_sigma, 1e-8), size=n_sims),
        _MINUTES_MIN,
        _MINUTES_MAX,
    )
    return rate_samples * min_samples


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------


