"""Tests for NCAAB bracket modifier functions (8 signals)."""

from __future__ import annotations

import pytest

from utils.ncaab_modifiers import (
    barttorvik_factor,
    blend_with_seed_prior,
    blend_with_vegas,
    coaching_factor,
    enhanced_composite_rating,
    experience_factor,
    momentum_factor,
    seed_matchup_prior,
    tempo_factor,
    vegas_implied_prob,
)


class TestBartTorvik:
    def test_agreement_neutral(self):
        assert barttorvik_factor(5, 5) == 1.0

    def test_bt_higher_gives_boost(self):
        assert barttorvik_factor(kenpom_rank=10, barttorvik_rank=1) > 1.0

    def test_bt_lower_gives_penalty(self):
        assert barttorvik_factor(kenpom_rank=1, barttorvik_rank=10) < 1.0

    def test_capped_boost(self):
        assert barttorvik_factor(100, 1) <= 1.05

    def test_capped_penalty(self):
        assert barttorvik_factor(1, 100) >= 0.95


class TestCoachingFactor:
    def test_neutral(self):
        assert coaching_factor(0.500) == 1.0

    def test_elite_boost(self):
        assert coaching_factor(0.700) > 1.0

    def test_weak_penalty(self):
        assert coaching_factor(0.300) < 1.0

    def test_capped(self):
        assert 0.95 <= coaching_factor(0.0) <= 1.05
        assert 0.95 <= coaching_factor(1.0) <= 1.05


class TestExperienceFactor:
    def test_neutral_at_50pct(self):
        assert experience_factor(0.50) == 1.0

    def test_high_returning_boost(self):
        assert experience_factor(0.80) > 1.0

    def test_freshman_heavy_penalty(self):
        assert experience_factor(0.20) < 1.0

    def test_capped(self):
        assert experience_factor(1.0) <= 1.04
        assert experience_factor(0.0) >= 0.96


class TestMomentumFactor:
    def test_conf_champion_boost(self):
        assert momentum_factor(8, "champion", 5) > 1.0

    def test_first_round_exit_penalty(self):
        assert momentum_factor(4, "first_round", 0) < 1.0

    def test_capped(self):
        assert momentum_factor(10, "champion", 20) <= 1.05
        assert momentum_factor(0, "did_not_qualify", 0) >= 0.97


class TestSeedMatchupPrior:
    def test_1_vs_16(self):
        assert seed_matchup_prior(1, 16) > 0.98

    def test_8_vs_9_coinflip(self):
        p = seed_matchup_prior(8, 9)
        assert 0.45 <= p <= 0.55

    def test_symmetry(self):
        p_ab = seed_matchup_prior(5, 12)
        p_ba = seed_matchup_prior(12, 5)
        assert abs(p_ab + p_ba - 1.0) < 1e-10

    def test_unknown_matchup_returns_50(self):
        assert seed_matchup_prior(1, 1) == 0.50


class TestBlendWithSeedPrior:
    def test_blending_formula(self):
        hist = seed_matchup_prior(5, 12)
        blended = blend_with_seed_prior(0.50, 5, 12)
        expected = 0.70 * 0.50 + 0.30 * hist
        assert abs(blended - expected) < 1e-6

    def test_equal_seeds_no_change(self):
        blended = blend_with_seed_prior(0.60, 8, 8)
        expected = 0.70 * 0.60 + 0.30 * 0.50
        assert abs(blended - expected) < 1e-6


class TestVegasImpliedProb:
    def test_favorite(self):
        p = vegas_implied_prob(-200)
        assert abs(p - 0.6667) < 0.01

    def test_underdog(self):
        p = vegas_implied_prob(200)
        assert abs(p - 0.3333) < 0.01


class TestBlendWithVegas:
    def test_small_disagreement_unchanged(self):
        assert blend_with_vegas(0.60, 0.65) == 0.60

    def test_large_disagreement_blends(self):
        blended = blend_with_vegas(0.80, 0.55)
        assert 0.55 < blended < 0.80

    def test_none_returns_model(self):
        assert blend_with_vegas(0.65, None) == 0.65


class TestTempoFactor:
    def test_slow_underdog_boost(self):
        f = tempo_factor(adj_t_a=65.0, adj_t_b=75.0, is_a_underdog=True)
        assert f > 1.0

    def test_same_tempo_neutral(self):
        assert tempo_factor(70.0, 70.0, True) == 1.0

    def test_small_diff_neutral(self):
        assert tempo_factor(70.0, 68.0, True) == 1.0

    def test_capped(self):
        assert tempo_factor(55.0, 80.0, True) <= 1.03


class TestEnhancedCompositeRating:
    def test_neutral_factors_unchanged(self):
        assert abs(enhanced_composite_rating(0.75, 1.0, 1.0, 1.0, 1.0) - 0.75) < 1e-6

    def test_positive_boosts(self):
        assert enhanced_composite_rating(0.60, 1.03, 1.02, 1.02, 1.04) > 0.60

    def test_clamped(self):
        assert enhanced_composite_rating(0.98, 1.05, 1.05, 1.04, 1.05) <= 0.99
        assert enhanced_composite_rating(0.02, 0.95, 0.95, 0.96, 0.97) >= 0.01
