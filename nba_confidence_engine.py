"""NBA confidence scoring and tiered play selection.

Computes a 0-100 confidence score for each NBA value betting opportunity
by combining edge size, projection stability, volume/usage certainty, and
volatility scoring. Assigns plays to one of four tiers:

- Premium (90-100): high edge + stable + high volume
- Strong  (75-89):  good edge + moderate stability
- Marginal (60-74): some edge + some uncertainty
- Pass    (<60):    not recommended

IMPORTANT: Does NOT import from ``config`` — uses hardcoded defaults
since NBA does not share the NFL config system.
"""

from __future__ import annotations

from typing import Any, Dict, Union

import pandas as pd


# Component weights
EDGE_WEIGHT = 0.35
STABILITY_WEIGHT = 0.20
VOLUME_WEIGHT = 0.25
VOLATILITY_WEIGHT = 0.20

# Tier thresholds
TIERS: Dict[str, Dict[str, Any]] = {
    "Premium":  {"min": 90, "max": 100, "rank": 1},
    "Strong":   {"min": 75, "max": 89,  "rank": 2},
    "Marginal": {"min": 60, "max": 74,  "rank": 3},
    "Pass":     {"min": 0,  "max": 59,  "rank": 4},
}

# Usage spike penalty: caps confidence at Marginal tier ceiling
USAGE_SPIKE_CAP = 74


def _tier_for_score(score: float) -> str:
    """Return the tier label for a given confidence score."""
    if score >= 90:
        return "Premium"
    if score >= 75:
        return "Strong"
    if score >= 60:
        return "Marginal"
    return "Pass"


def _tier_rank(tier: str) -> int:
    """Return the numeric rank for a tier (lower = better)."""
    return TIERS.get(tier, TIERS["Pass"])["rank"]


# -- Component scoring -------------------------------------------------------


def _score_edge(edge_percentage: float) -> float:
    """Score edge size on a 0-100 scale.

    Edge of 0 -> 0, edge of 0.25+ -> 100, linear in between.
    """
    if edge_percentage <= 0:
        return 0.0
    return min(edge_percentage / 0.25, 1.0) * 100.0


def _score_stability(mu: float, sigma: float) -> float:
    """Score projection stability on a 0-100 scale.

    Low sigma relative to mu is more stable.
    CV of 0 -> 100, CV of 1.0+ -> 0.
    """
    if mu <= 0 or sigma < 0:
        return 50.0
    cv = sigma / mu
    return max(0.0, min((1.0 - cv) * 100.0, 100.0))


def _score_volume(fga_share: float) -> float:
    """Score volume certainty on a 0-100 scale using FGA share.

    fga_share of 0.25+ -> 100, 0 -> 0.
    NBA's equivalent of NFL's target_share.
    """
    if fga_share <= 0:
        return 0.0
    return min(fga_share / 0.25, 1.0) * 100.0


def _score_volatility(volatility_score: float) -> float:
    """Convert volatility score (0-100, high=bad) to confidence component.

    Inverted: low volatility = high confidence contribution.
    """
    clamped = max(0.0, min(volatility_score, 100.0))
    return 100.0 - clamped


# -- Main scoring function ---------------------------------------------------


def compute_nba_confidence_score(
    row: Union[Dict[str, Any], pd.Series],
) -> float:
    """Compute a 0-100 confidence score for an NBA value betting opportunity.

    Parameters
    ----------
    row : dict or pd.Series
        Must contain: edge_percentage, mu, sigma.
        Optional: fga_share, volatility_score, usage_spike.

    Returns
    -------
    float
        Confidence score between 0 and 100.
    """
    edge_pct = _safe_float(row, "edge_percentage", 0.0)
    mu = _safe_float(row, "mu", 0.0)
    sigma = _safe_float(row, "sigma", 0.0)
    fga_share = _safe_float(row, "fga_share", 0.0)
    volatility = _safe_float(row, "volatility_score", 50.0)
    usage_spike = _safe_bool(row, "usage_spike", False)

    edge_score = _score_edge(edge_pct)
    stability_score = _score_stability(mu, sigma)
    volume_score = _score_volume(fga_share)
    volatility_component = _score_volatility(volatility)

    raw = (
        EDGE_WEIGHT * edge_score
        + STABILITY_WEIGHT * stability_score
        + VOLUME_WEIGHT * volume_score
        + VOLATILITY_WEIGHT * volatility_component
    )

    score = max(0.0, min(raw, 100.0))

    # Usage spike penalty: cap at Marginal tier ceiling
    if usage_spike:
        score = min(score, USAGE_SPIKE_CAP)

    return round(score, 2)


def assign_nba_tier(score: float) -> str:
    """Assign a confidence tier label based on score."""
    return _tier_for_score(score)


# -- Filtering ---------------------------------------------------------------


def filter_by_nba_confidence(
    plays_df: pd.DataFrame,
    min_tier: int = 2,
) -> pd.DataFrame:
    """Filter plays by minimum confidence tier.

    Parameters
    ----------
    plays_df : DataFrame
        Must contain ``confidence_score`` and ``confidence_tier`` columns.
    min_tier : int
        Minimum tier rank to include (1=Premium, 2=Strong, 3=Marginal).

    Returns
    -------
    DataFrame
        Filtered copy -- original is never mutated.
    """
    if plays_df.empty:
        return plays_df.copy()

    if "confidence_tier" not in plays_df.columns:
        return plays_df.copy()

    ranks = plays_df["confidence_tier"].map(_tier_rank)
    return plays_df[ranks <= min_tier].copy()


# -- Bulk scoring -------------------------------------------------------------


def score_nba_plays(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``confidence_score`` and ``confidence_tier`` columns to a DataFrame.

    Returns a new DataFrame -- the original is never mutated.
    """
    if df.empty:
        result = df.copy()
        result["confidence_score"] = pd.Series(dtype="float64")
        result["confidence_tier"] = pd.Series(dtype="object")
        return result

    scores = df.apply(
        lambda r: compute_nba_confidence_score(r), axis=1
    )
    tiers = scores.map(assign_nba_tier)

    return df.assign(confidence_score=scores, confidence_tier=tiers)


# -- Helpers ------------------------------------------------------------------


def _safe_float(
    row: Union[Dict[str, Any], pd.Series],
    key: str,
    default: float,
) -> float:
    """Extract a float from a dict/Series, falling back to *default*."""
    if isinstance(row, dict):
        val = row.get(key)
    else:
        val = row.get(key) if key in row.index else None

    if val is None:
        return default
    try:
        result = float(val)
        if result != result:  # NaN check
            return default
        return result
    except (TypeError, ValueError):
        return default


def _safe_bool(
    row: Union[Dict[str, Any], pd.Series],
    key: str,
    default: bool,
) -> bool:
    """Extract a bool from a dict/Series, falling back to *default*."""
    if isinstance(row, dict):
        val = row.get(key)
    else:
        val = row.get(key) if key in row.index else None

    if val is None:
        return default
    return bool(val)
