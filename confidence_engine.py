"""Confidence scoring and tiered play selection.

Computes a 0-100 confidence score for each value betting opportunity
by combining edge size, projection stability, volume certainty, and
volatility scoring.  Assigns plays to one of four tiers:

- Tier 1 (90-100): "Premium" -- high edge + stable + high volume
- Tier 2 (75-89):  "Strong"  -- good edge + moderate stability
- Tier 3 (60-74):  "Marginal"-- some edge + some uncertainty
- Below 60:        "Pass"    -- not recommended
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import pandas as pd

from config import config


# ── Tier definitions ──────────────────────────────────────────────────

TIERS: Dict[str, Dict[str, Any]] = {
    "Premium":  {"min": 90, "max": 100, "rank": 1},
    "Strong":   {"min": 75, "max": 89,  "rank": 2},
    "Marginal": {"min": 60, "max": 74,  "rank": 3},
    "Pass":     {"min": 0,  "max": 59,  "rank": 4},
}


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


# ── Component scoring ────────────────────────────────────────────────

def _score_edge(edge_percentage: float) -> float:
    """Score edge size on a 0-100 scale.

    An edge of 0 maps to 0; an edge of 0.25+ maps to 100.
    Linear interpolation in between.
    """
    if edge_percentage <= 0:
        return 0.0
    max_edge = 0.25
    return min(edge_percentage / max_edge, 1.0) * 100.0


def _score_stability(mu: float, sigma: float) -> float:
    """Score projection stability on a 0-100 scale.

    Low sigma relative to mu is more stable.
    A CV (sigma/mu) of 0 maps to 100; CV of 1.0+ maps to 0.
    """
    if mu <= 0 or sigma < 0:
        return 50.0
    cv = sigma / mu
    return max(0.0, min((1.0 - cv) * 100.0, 100.0))


def _score_volume(target_share: float) -> float:
    """Score volume certainty on a 0-100 scale.

    A target_share of 0.30+ maps to 100; 0 maps to 0.
    """
    if target_share <= 0:
        return 0.0
    max_share = 0.30
    return min(target_share / max_share, 1.0) * 100.0


def _score_volatility(volatility_score: float) -> float:
    """Convert volatility score (0-100, high=bad) to confidence component.

    Invert: low volatility = high confidence contribution.
    """
    clamped = max(0.0, min(volatility_score, 100.0))
    return 100.0 - clamped


# ── Main scoring function ────────────────────────────────────────────

def compute_confidence_score(row: Union[Dict[str, Any], pd.Series]) -> float:
    """Compute a 0-100 confidence score for a value betting opportunity.

    Parameters
    ----------
    row : dict or pd.Series
        Must contain keys: ``edge_percentage``, ``mu``, ``sigma``.
        Optional keys: ``target_share``, ``volatility_score``.

    Returns
    -------
    float
        Confidence score between 0 and 100.
    """
    weights = _get_weights()

    edge_pct = _safe_float(row, "edge_percentage", 0.0)
    mu = _safe_float(row, "mu", 0.0)
    sigma = _safe_float(row, "sigma", 0.0)
    target_share = _safe_float(row, "target_share", 0.0)
    volatility = _safe_float(row, "volatility_score", 50.0)

    edge_score = _score_edge(edge_pct)
    stability_score = _score_stability(mu, sigma)
    volume_score = _score_volume(target_share)
    volatility_component = _score_volatility(volatility)

    raw = (
        weights["edge"] * edge_score
        + weights["stability"] * stability_score
        + weights["volume"] * volume_score
        + weights["volatility"] * volatility_component
    )

    return round(max(0.0, min(raw, 100.0)), 2)


def assign_tier(score: float) -> str:
    """Assign a confidence tier label based on score."""
    return _tier_for_score(score)


# ── Filtering ─────────────────────────────────────────────────────────

def filter_by_confidence(
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
        Default 2 includes Premium and Strong.

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


# ── Bulk scoring ──────────────────────────────────────────────────────

def score_plays(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``confidence_score`` and ``confidence_tier`` columns to a DataFrame.

    Returns a new DataFrame -- the original is never mutated.
    """
    if df.empty:
        result = df.copy()
        result["confidence_score"] = pd.Series(dtype="float64")
        result["confidence_tier"] = pd.Series(dtype="object")
        return result

    scores = df.apply(
        lambda r: compute_confidence_score(r), axis=1
    )
    tiers = scores.map(assign_tier)

    return df.assign(confidence_score=scores, confidence_tier=tiers)


# ── Helpers ───────────────────────────────────────────────────────────

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


def _get_weights() -> Dict[str, float]:
    """Read confidence weights from config, with sensible defaults."""
    conf = getattr(config, "confidence", None)
    if conf is not None:
        return {
            "edge": getattr(conf, "edge_weight", 0.35),
            "stability": getattr(conf, "stability_weight", 0.25),
            "volume": getattr(conf, "volume_weight", 0.20),
            "volatility": getattr(conf, "volatility_weight", 0.20),
        }
    return {
        "edge": 0.35,
        "stability": 0.25,
        "volume": 0.20,
        "volatility": 0.20,
    }
