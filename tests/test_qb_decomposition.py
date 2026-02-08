"""Tests for QB passing yards volume/efficiency decomposition.

Covers:
- Volume (pass_attempts_predicted) computation via EWMA + game-script
- Efficiency (yards_per_attempt_predicted) via EWMA + opponent adjustment
- Composite: volume x efficiency = total passing yards
- Playoff inflation factor application
- Context sensitivity scoring
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from data_pipeline import (
    apply_playoff_volume_factor,
    compute_context_sensitivity,
    compute_pass_attempts_predicted,
    compute_yards_per_attempt_predicted,
    decompose_qb_passing,
    DataPipeline,
    QB_BASELINE_ATTEMPTS,
    QB_BASELINE_YPA,
)


# -----------------------------------------------------------------------
# Volume (pass_attempts_predicted)
# -----------------------------------------------------------------------

class TestPassAttemptsPredicted:

    def test_returns_baseline_when_no_history(self):
        result = compute_pass_attempts_predicted(pd.Series(dtype=float))
        assert result == pytest.approx(QB_BASELINE_ATTEMPTS, rel=0.01)

    def test_ewma_weights_recent_games_more(self):
        # Recent game has 40 attempts, older games have 25
        attempts = pd.Series([40, 25, 25, 25, 25])
        result = compute_pass_attempts_predicted(attempts)
        assert result > 30.0, f"EWMA should weight recent 40-attempt game higher, got {result}"

    def test_positive_game_script_increases_volume(self):
        attempts = pd.Series([35, 34, 33])
        neutral = compute_pass_attempts_predicted(attempts, game_script=0.0)
        trailing = compute_pass_attempts_predicted(attempts, game_script=1.0)
        assert trailing > neutral, "Trailing game-script should increase volume"

    def test_negative_game_script_decreases_volume(self):
        attempts = pd.Series([35, 34, 33])
        neutral = compute_pass_attempts_predicted(attempts, game_script=0.0)
        leading = compute_pass_attempts_predicted(attempts, game_script=-1.0)
        assert leading < neutral, "Leading game-script should decrease volume"

    def test_game_script_factor_clamped(self):
        attempts = pd.Series([35])
        extreme_trailing = compute_pass_attempts_predicted(attempts, game_script=10.0)
        # Factor should be clamped to 1.25
        assert extreme_trailing <= 35 * 1.26, "Game-script factor should be clamped at 1.25"

    def test_minimum_floor(self):
        result = compute_pass_attempts_predicted(pd.Series([1, 1, 1]), game_script=-5.0)
        assert result >= 10.0, f"Volume should have floor of 10, got {result}"


# -----------------------------------------------------------------------
# Efficiency (yards_per_attempt_predicted)
# -----------------------------------------------------------------------

class TestYardsPerAttemptPredicted:

    def test_returns_baseline_when_no_history(self):
        result = compute_yards_per_attempt_predicted(pd.Series(dtype=float))
        assert result == pytest.approx(QB_BASELINE_YPA, rel=0.01)

    def test_ewma_weights_recent_games(self):
        ypa = pd.Series([10.0, 6.0, 6.0, 6.0])
        result = compute_yards_per_attempt_predicted(ypa)
        assert result > 7.0, f"EWMA should weight recent high YPA, got {result}"

    def test_weak_defense_increases_efficiency(self):
        ypa = pd.Series([7.0, 7.0, 7.0])
        neutral = compute_yards_per_attempt_predicted(ypa, opp_pass_defense_rank=16.0)
        weak_d = compute_yards_per_attempt_predicted(ypa, opp_pass_defense_rank=30.0)
        assert weak_d > neutral, "Weak pass defense (high rank) should increase efficiency"

    def test_strong_defense_decreases_efficiency(self):
        ypa = pd.Series([7.0, 7.0, 7.0])
        neutral = compute_yards_per_attempt_predicted(ypa, opp_pass_defense_rank=16.0)
        strong_d = compute_yards_per_attempt_predicted(ypa, opp_pass_defense_rank=2.0)
        assert strong_d < neutral, "Strong pass defense (low rank) should decrease efficiency"

    def test_opponent_factor_clamped(self):
        ypa = pd.Series([7.0])
        extreme = compute_yards_per_attempt_predicted(ypa, opp_pass_defense_rank=100.0)
        assert extreme <= 7.0 * 1.16, "Opponent factor should be clamped at 1.15"

    def test_minimum_floor(self):
        result = compute_yards_per_attempt_predicted(pd.Series([0.5, 0.3]))
        assert result >= 3.0, f"Efficiency should have floor of 3.0, got {result}"


# -----------------------------------------------------------------------
# Composite decomposition
# -----------------------------------------------------------------------

class TestDecomposeQBPassing:

    def test_composite_equals_volume_times_efficiency(self):
        result = decompose_qb_passing(
            hist_pass_yards=pd.Series([280, 260, 300]),
            hist_pass_attempts=pd.Series([35, 32, 38]),
            game_script=0.0,
            opp_pass_defense_rank=16.0,
        )
        expected = result["pass_attempts_predicted"] * result["yards_per_attempt_predicted"]
        assert result["passing_yards_decomposed"] == pytest.approx(expected, rel=0.01)

    def test_all_keys_present(self):
        result = decompose_qb_passing(
            hist_pass_yards=pd.Series([250]),
            hist_pass_attempts=pd.Series([30]),
        )
        assert "pass_attempts_predicted" in result
        assert "yards_per_attempt_predicted" in result
        assert "passing_yards_decomposed" in result
        assert "context_sensitivity" in result

    def test_empty_history_uses_baselines(self):
        result = decompose_qb_passing(
            hist_pass_yards=pd.Series(dtype=float),
            hist_pass_attempts=pd.Series(dtype=float),
        )
        expected_baseline = QB_BASELINE_ATTEMPTS * QB_BASELINE_YPA
        assert result["passing_yards_decomposed"] == pytest.approx(expected_baseline, rel=0.02)

    def test_handles_zero_attempts_in_history(self):
        result = decompose_qb_passing(
            hist_pass_yards=pd.Series([0, 200, 250]),
            hist_pass_attempts=pd.Series([0, 28, 32]),
        )
        # Should not crash from division by zero
        assert result["passing_yards_decomposed"] > 0

    def test_high_volume_qb(self):
        result = decompose_qb_passing(
            hist_pass_yards=pd.Series([320, 310, 290, 305]),
            hist_pass_attempts=pd.Series([42, 40, 38, 41]),
        )
        assert result["pass_attempts_predicted"] > 38.0
        assert result["passing_yards_decomposed"] > 250.0


# -----------------------------------------------------------------------
# Playoff inflation
# -----------------------------------------------------------------------

class TestPlayoffInflation:

    def test_regular_season_no_inflation(self):
        result = apply_playoff_volume_factor(35.0, week=10, season_type="regular")
        assert result == 35.0

    def test_wildcard_inflation(self):
        result = apply_playoff_volume_factor(35.0, week=19, season_type="post")
        assert result > 35.0
        assert result == pytest.approx(35.0 * 1.03, rel=0.01)

    def test_divisional_inflation(self):
        result = apply_playoff_volume_factor(35.0, week=20, season_type="post")
        assert result == pytest.approx(35.0 * 1.05, rel=0.01)

    def test_conference_inflation(self):
        result = apply_playoff_volume_factor(35.0, week=21, season_type="post")
        assert result == pytest.approx(35.0 * 1.07, rel=0.01)

    def test_superbowl_inflation(self):
        result = apply_playoff_volume_factor(35.0, week=22, season_type="post")
        assert result == pytest.approx(35.0 * 1.10, rel=0.01)

    def test_week_19_triggers_inflation_regardless_of_season_type(self):
        # Week > 18 should always inflate even if season_type is not "post"
        result = apply_playoff_volume_factor(35.0, week=19, season_type="regular")
        assert result > 35.0

    def test_decompose_applies_playoff_inflation(self):
        regular = decompose_qb_passing(
            hist_pass_yards=pd.Series([280, 260]),
            hist_pass_attempts=pd.Series([35, 32]),
            week=10,
            season_type="regular",
        )
        playoff = decompose_qb_passing(
            hist_pass_yards=pd.Series([280, 260]),
            hist_pass_attempts=pd.Series([35, 32]),
            week=21,
            season_type="post",
        )
        assert playoff["pass_attempts_predicted"] > regular["pass_attempts_predicted"]
        assert playoff["passing_yards_decomposed"] > regular["passing_yards_decomposed"]


# -----------------------------------------------------------------------
# Context sensitivity
# -----------------------------------------------------------------------

class TestContextSensitivity:

    def test_score_between_zero_and_one(self):
        for n in [0, 2, 5, 10]:
            hist = pd.Series(np.random.uniform(25, 45, n))
            score = compute_context_sensitivity(hist)
            assert 0.0 <= score <= 1.0, f"Score out of range for n={n}: {score}"

    def test_small_sample_high_sensitivity(self):
        small = compute_context_sensitivity(pd.Series([35, 30]))
        large = compute_context_sensitivity(pd.Series([35, 30, 33, 34, 32, 36, 31, 35, 37, 34]))
        assert small > large, "Small sample should have higher context sensitivity"

    def test_high_game_script_increases_sensitivity(self):
        hist = pd.Series([35, 34, 33, 35, 34, 33, 35, 34])
        neutral = compute_context_sensitivity(hist, game_script=0.0)
        trailing = compute_context_sensitivity(hist, game_script=2.0)
        assert trailing > neutral, "High game-script magnitude should increase sensitivity"

    def test_playoff_increases_sensitivity(self):
        hist = pd.Series([35, 34, 33, 35, 34, 33, 35, 34])
        regular = compute_context_sensitivity(hist, season_type="regular", week=10)
        playoff = compute_context_sensitivity(hist, season_type="post", week=20)
        assert playoff > regular, "Playoff context should increase sensitivity"

    def test_empty_history_maximum_sensitivity(self):
        score = compute_context_sensitivity(pd.Series(dtype=float))
        assert score >= 0.2, f"Empty history should yield high sensitivity, got {score}"


# -----------------------------------------------------------------------
# DataPipeline integration
# -----------------------------------------------------------------------

class TestDataPipelineQBIntegration:

    def test_compute_market_mu_routes_to_decomposition(self, monkeypatch):
        hist = pd.DataFrame({
            "passing_yards": [280, 260, 300],
            "passing_attempts": [35, 32, 38],
        })
        monkeypatch.setattr("data_pipeline.read_dataframe", lambda *_, **__: hist)

        dp = DataPipeline.__new__(DataPipeline)
        row = SimpleNamespace(
            player_id="KC_patrick_mahomes",
            passing_yards=280,
            passing_attempts=35,
            game_script=0.5,
            opp_pass_defense_rank=20.0,
        )
        mu = dp._compute_market_mu(row, "passing_yards", season=2025, week=10)
        assert mu > 200.0, f"QB mu should be substantial, got {mu}"
        assert mu < 400.0, f"QB mu should be reasonable, got {mu}"

    def test_compute_qb_passing_mu_returns_decomposition(self, monkeypatch):
        hist = pd.DataFrame({
            "passing_yards": [280, 260],
            "passing_attempts": [35, 32],
        })
        monkeypatch.setattr("data_pipeline.read_dataframe", lambda *_, **__: hist)

        dp = DataPipeline.__new__(DataPipeline)
        row = SimpleNamespace(
            player_id="BUF_josh_allen",
            game_script=0.0,
            opp_pass_defense_rank=16.0,
        )
        result = dp._compute_qb_passing_mu(row, season=2025, week=10)
        assert "passing_yards_decomposed" in result
        assert "context_sensitivity" in result
        assert result["passing_yards_decomposed"] > 0
