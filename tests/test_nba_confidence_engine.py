"""Tests for nba_confidence_engine.py — 4-component score, tiers, spike penalty.

Covers:
- 4-component score weighting (edge=35%, stability=20%, volume=25%, volatility=20%)
- Tier assignment (Premium/Strong/Marginal/Pass)
- Usage spike penalty caps confidence at 74
- Edge cases: zero edge, very high edge, missing fields
- score_nba_plays() DataFrame batch processing
- filter_by_nba_confidence() tier filtering
- NBA confidence engine does NOT import from config
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Constants matching expected implementation
# ---------------------------------------------------------------------------

TIER_CUTOFFS = {
    "Premium": 90,
    "Strong": 75,
    "Marginal": 60,
    "Pass": 0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(
    edge_percentage: float = 0.12,
    mu: float = 28.0,
    sigma: float = 5.0,
    fga_share: float = 0.18,
    volatility_score: float = 30.0,
    usage_spike: bool = False,
) -> dict:
    """Build a mock value row for confidence scoring.

    Uses the actual input fields the implementation expects:
    edge_percentage (decimal), mu, sigma, fga_share, volatility_score.
    """
    return {
        "player_id": 1628369,
        "player_name": "Jayson Tatum",
        "market": "pts",
        "edge_percentage": edge_percentage,
        "mu": mu,
        "sigma": sigma,
        "fga_share": fga_share,
        "volatility_score": volatility_score,
        "usage_spike": usage_spike,
    }


def _make_df(rows: list[dict] | None = None) -> pd.DataFrame:
    """Convert row dicts to DataFrame for batch processing."""
    if rows is None:
        rows = [_make_row()]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_nba_confidence_score — component weighting
# ---------------------------------------------------------------------------


class TestConfidenceScore:
    """Test the 4-component weighted confidence score."""

    def test_returns_float(self):
        from nba_confidence_engine import compute_nba_confidence_score

        result = compute_nba_confidence_score(_make_row())
        assert isinstance(result, float)

    def test_score_in_0_100_range(self):
        from nba_confidence_engine import compute_nba_confidence_score

        result = compute_nba_confidence_score(_make_row())
        assert 0.0 <= result <= 100.0

    def test_high_edge_produces_high_score(self):
        from nba_confidence_engine import compute_nba_confidence_score

        row = _make_row(
            edge_percentage=0.25,
            mu=30.0,
            sigma=3.0,
            fga_share=0.25,
            volatility_score=10.0,
        )
        score = compute_nba_confidence_score(row)
        assert score >= 80.0

    def test_zero_edge_produces_low_score(self):
        from nba_confidence_engine import compute_nba_confidence_score

        row = _make_row(
            edge_percentage=0.0,
            mu=25.0,
            sigma=8.0,
            fga_share=0.10,
            volatility_score=50.0,
        )
        score = compute_nba_confidence_score(row)
        assert score < 60.0

    def test_high_volatility_reduces_score(self):
        """High volatility (inverted) should lower the score."""
        from nba_confidence_engine import compute_nba_confidence_score

        stable_row = _make_row(volatility_score=10.0)
        volatile_row = _make_row(volatility_score=90.0)
        stable_score = compute_nba_confidence_score(stable_row)
        volatile_score = compute_nba_confidence_score(volatile_row)
        assert stable_score > volatile_score

    def test_edge_weight_is_dominant(self):
        """Edge component (35%) should be the most influential."""
        from nba_confidence_engine import compute_nba_confidence_score

        # Only edge is high, others are medium
        high_edge = _make_row(edge_percentage=0.20, mu=25.0, sigma=10.0, fga_share=0.10, volatility_score=50.0)
        # Only stability is high (low sigma/mu), others are medium
        high_stab = _make_row(edge_percentage=0.05, mu=25.0, sigma=2.0, fga_share=0.10, volatility_score=50.0)

        edge_score = compute_nba_confidence_score(high_edge)
        stab_score = compute_nba_confidence_score(high_stab)
        assert edge_score > stab_score


# ---------------------------------------------------------------------------
# Usage spike penalty
# ---------------------------------------------------------------------------


class TestUsageSpikePenalty:
    """Usage spike should cap confidence at 74 (below Premium tier)."""

    def test_spike_caps_at_74(self):
        from nba_confidence_engine import compute_nba_confidence_score

        # This row would normally score > 74 without spike
        row = _make_row(
            edge_percentage=0.25,
            mu=30.0,
            sigma=3.0,
            fga_share=0.25,
            volatility_score=10.0,
            usage_spike=True,
        )
        score = compute_nba_confidence_score(row)
        assert score <= 74.0

    def test_no_spike_allows_full_score(self):
        from nba_confidence_engine import compute_nba_confidence_score

        row = _make_row(
            edge_percentage=0.25,
            mu=30.0,
            sigma=3.0,
            fga_share=0.25,
            volatility_score=10.0,
            usage_spike=False,
        )
        score = compute_nba_confidence_score(row)
        assert score > 74.0


# ---------------------------------------------------------------------------
# assign_nba_tier
# ---------------------------------------------------------------------------


class TestAssignTier:
    """Tier assignment based on score thresholds."""

    def test_premium_tier(self):
        from nba_confidence_engine import assign_nba_tier

        assert assign_nba_tier(90.0) == "Premium"

    def test_strong_tier(self):
        from nba_confidence_engine import assign_nba_tier

        assert assign_nba_tier(75.0) == "Strong"

    def test_marginal_tier(self):
        from nba_confidence_engine import assign_nba_tier

        assert assign_nba_tier(60.0) == "Marginal"

    def test_pass_tier(self):
        from nba_confidence_engine import assign_nba_tier

        assert assign_nba_tier(40.0) == "Pass"

    def test_boundary_premium(self):
        from nba_confidence_engine import assign_nba_tier

        assert assign_nba_tier(90.0) == "Premium"

    def test_boundary_strong(self):
        from nba_confidence_engine import assign_nba_tier

        assert assign_nba_tier(75.0) == "Strong"

    def test_boundary_marginal(self):
        from nba_confidence_engine import assign_nba_tier

        assert assign_nba_tier(60.0) == "Marginal"


# ---------------------------------------------------------------------------
# score_nba_plays — batch processing
# ---------------------------------------------------------------------------


class TestScorePlays:
    """Batch scoring of a DataFrame of value plays."""

    def test_adds_confidence_score_column(self):
        from nba_confidence_engine import score_nba_plays

        df = _make_df()
        result = score_nba_plays(df)
        assert "confidence_score" in result.columns

    def test_adds_tier_column(self):
        from nba_confidence_engine import score_nba_plays

        df = _make_df()
        result = score_nba_plays(df)
        assert "confidence_tier" in result.columns

    def test_tier_values_are_valid(self):
        from nba_confidence_engine import score_nba_plays

        valid_tiers = {"Premium", "Strong", "Marginal", "Pass"}
        df = _make_df(
            [
                _make_row(edge_percentage=0.25, mu=30.0, sigma=3.0, fga_share=0.25),
                _make_row(edge_percentage=0.02, mu=20.0, sigma=8.0, fga_share=0.05),
            ]
        )
        result = score_nba_plays(df)
        assert set(result["confidence_tier"].unique()).issubset(valid_tiers)

    def test_empty_df_returns_empty(self):
        from nba_confidence_engine import score_nba_plays

        df = pd.DataFrame()
        result = score_nba_plays(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# filter_by_nba_confidence
# ---------------------------------------------------------------------------


class TestFilterByConfidence:
    """Filter DataFrame by minimum tier level."""

    def test_filter_removes_low_tier(self):
        from nba_confidence_engine import score_nba_plays, filter_by_nba_confidence

        df = _make_df(
            [
                _make_row(edge_percentage=0.25, mu=30.0, sigma=3.0, fga_share=0.25, volatility_score=10.0),
                _make_row(edge_percentage=0.01, mu=15.0, sigma=10.0, fga_share=0.02, volatility_score=90.0),
            ]
        )
        scored = score_nba_plays(df)
        filtered = filter_by_nba_confidence(scored, min_tier=2)
        # Only strong/premium should remain, pass/marginal removed
        assert all(t in ("Premium", "Strong") for t in filtered["confidence_tier"])

    def test_filter_keeps_all_when_min_tier_4(self):
        from nba_confidence_engine import score_nba_plays, filter_by_nba_confidence

        df = _make_df([_make_row()])
        scored = score_nba_plays(df)
        filtered = filter_by_nba_confidence(scored, min_tier=4)
        assert len(filtered) == len(scored)

    def test_empty_df_returns_empty(self):
        from nba_confidence_engine import filter_by_nba_confidence

        df = pd.DataFrame(columns=["confidence_tier", "confidence_score"])
        result = filter_by_nba_confidence(df, min_tier=1)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# No config dependency
# ---------------------------------------------------------------------------


class TestNoConfigImport:
    """NBA confidence engine should NOT import from config module."""

    def test_does_not_import_config(self):
        import importlib
        import sys

        # Remove nba_confidence_engine from cache to get a fresh import
        mod_name = "nba_confidence_engine"
        if mod_name in sys.modules:
            del sys.modules[mod_name]

        # Track imports
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__
        imported_modules = []

        def tracking_import(name, *args, **kwargs):
            imported_modules.append(name)
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=tracking_import):
            try:
                importlib.import_module(mod_name)
            except ImportError:
                pass  # Module may not exist yet (TDD)

        config_imports = [m for m in imported_modules if m == "config"]
        assert len(config_imports) == 0, "nba_confidence_engine should not import config"
