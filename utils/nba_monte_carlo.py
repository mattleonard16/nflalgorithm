"""Monte Carlo simulation utilities for NBA player prop probabilities.

Provides simulation-based alternatives to analytic CDF approximations,
enabling richer uncertainty modelling (e.g. minutes × rate decomposition,
correlated multi-stat portfolios, Gamma-Poisson for fg3m).

All functions are pure (no DB access) and scipy-free.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Rate markets that support minutes × rate decomposition
_RATE_MARKETS = {"pts", "reb", "ast"}

# Minutes are clipped to this range before simulation
_MINUTES_MIN = 0.0
_MINUTES_MAX = 48.0

# Regularisation added to correlation matrix before Cholesky
_CHOL_EPS = 1e-6


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
        Seeded RNG for reproducibility. Uses default_rng() if None.
    minutes_mu : float or None
        Mean predicted minutes.  Enables minutes × rate decomposition for
        rate markets (pts/reb/ast) when provided together with minutes_sigma.
    minutes_sigma : float or None
        Std-dev of predicted minutes.

    Returns
    -------
    (p_over, p_under) : tuple[float, float]
        Probabilities that always sum to exactly 1.0.
    """
    if rng is None:
        rng = np.random.default_rng()

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


def _simulate_fg3m(mu: float, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """Gamma-Poisson compound for fg3m (negative binomial approximation).

    Marginalises over a Gamma prior on the Poisson rate to capture
    over-dispersion in 3-point makes.  Shape k = mu, scale = 1 gives
    Var[rate] = mu so total Var[X] = mu + mu = 2*mu.
    Used in portfolio simulations where inter-game variance matters.
    """
    if mu <= 0:
        return np.zeros(n_sims)
    # Draw Poisson rates from Gamma(shape=mu, scale=1)
    rates = rng.gamma(shape=max(mu, 1e-6), scale=1.0, size=n_sims)
    return rng.poisson(lam=rates).astype(float)


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


def build_correlation_matrix(
    game_logs_df: pd.DataFrame,
    markets: list[str] | None = None,
    min_games: int = 20,
) -> np.ndarray:
    """Compute a regularised correlation matrix from player game logs.

    Parameters
    ----------
    game_logs_df : pd.DataFrame
        Must contain one column per market (e.g. pts, reb, ast).
    markets : list[str] or None
        Markets to correlate.  Defaults to ["pts", "reb", "ast"].
    min_games : int
        Minimum number of rows required to use empirical correlation;
        falls back to identity matrix otherwise.

    Returns
    -------
    np.ndarray
        (len(markets) × len(markets)) correlation matrix, PSD.
    """
    if markets is None:
        markets = ["pts", "reb", "ast"]

    n = len(markets)

    present = [m for m in markets if m in game_logs_df.columns]
    if len(present) < n or len(game_logs_df) < min_games:
        return np.eye(n)

    sub = game_logs_df[present].dropna()
    if len(sub) < min_games:
        return np.eye(n)

    corr = sub.corr().values.astype(float)

    # Regularise: add small diagonal to ensure positive definiteness
    corr_reg = corr + np.eye(n) * _CHOL_EPS

    # Validate via Cholesky; fall back to identity on failure
    try:
        np.linalg.cholesky(corr_reg)
    except np.linalg.LinAlgError:
        log.warning("Correlation matrix failed Cholesky; returning identity.")
        corr_reg = np.eye(n)

    return corr_reg


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------


def monte_carlo_portfolio(
    projections: list[dict],
    correlation_matrix: np.ndarray,
    n_sims: int = 10_000,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Simulate correlated multi-stat outcomes for a single player.

    Parameters
    ----------
    projections : list[dict]
        Each dict must have keys: market, mu, sigma, line.
        Optional keys: minutes_mu, minutes_sigma.
    correlation_matrix : np.ndarray
        Square PSD matrix of size len(projections).
    n_sims : int
        Number of simulation draws.
    rng : numpy Generator or None
        Seeded RNG for reproducibility.

    Returns
    -------
    list[dict]
        One dict per projection with added keys:
        mc_p_over, mc_p_under.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(projections)
    if n == 0:
        return []

    # Draw correlated standard normals via Cholesky decomposition
    try:
        L = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        L = np.eye(n)

    z = rng.standard_normal((n_sims, n))
    correlated_z = z @ L.T  # shape (n_sims, n)

    results = []
    for i, proj in enumerate(projections):
        mu = float(proj["mu"])
        sigma = float(proj.get("sigma", 0.0))
        line = float(proj["line"])
        market = str(proj.get("market", "pts"))

        zi = correlated_z[:, i]

        if market == "fg3m":
            samples = _simulate_fg3m(mu, n_sims, rng)
        elif market in _RATE_MARKETS and proj.get("minutes_mu") and sigma > 0:
            min_mu = float(proj["minutes_mu"])
            min_sigma = float(proj.get("minutes_sigma", min_mu * 0.15))
            rate_mu = mu / max(min_mu, 1e-8)
            rate_sigma = sigma / max(min_mu, 1e-8)
            # Use correlated z for rate; independent for minutes
            rate_samples = rate_mu + rate_sigma * zi
            min_samples = np.clip(
                rng.normal(loc=min_mu, scale=max(min_sigma, 1e-8), size=n_sims),
                _MINUTES_MIN,
                _MINUTES_MAX,
            )
            samples = rate_samples * min_samples
        elif sigma > 0:
            samples = mu + sigma * zi
        else:
            samples = np.full(n_sims, mu)

        p_over = float(np.mean(samples > line))
        p_under = 1.0 - p_over

        results.append(
            {
                **proj,
                "mc_p_over": p_over,
                "mc_p_under": p_under,
            }
        )

    return results
