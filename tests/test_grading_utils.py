"""Tests for shared grading utilities.

Covers:
- grade_bet: over/under win/loss/push, NaN handling, exact line push
- calculate_profit_units: positive/negative odds, win/loss/push, edge cases
- get_confidence_tier: threshold boundaries and exact boundary values
"""

from __future__ import annotations

import math

import pytest

from utils.grading import calculate_profit_units, get_confidence_tier, grade_bet


# ======================================================================
# grade_bet
# ======================================================================


class TestGradeBet:
    # --- Over side ---

    def test_over_win_when_actual_exceeds_line(self):
        assert grade_bet(125.5, 115.5, "over") == "win"

    def test_over_win_by_small_margin(self):
        assert grade_bet(20.1, 20.0, "over") == "win"

    def test_over_loss_when_actual_below_line(self):
        assert grade_bet(100.0, 115.5, "over") == "loss"

    def test_over_loss_by_small_margin(self):
        assert grade_bet(19.9, 20.0, "over") == "loss"

    def test_over_push_when_actual_equals_line(self):
        assert grade_bet(20.0, 20.0, "over") == "push"

    # --- Under side ---

    def test_under_win_when_actual_below_line(self):
        assert grade_bet(100.0, 115.5, "under") == "win"

    def test_under_win_by_small_margin(self):
        assert grade_bet(19.9, 20.0, "under") == "win"

    def test_under_loss_when_actual_exceeds_line(self):
        assert grade_bet(125.5, 115.5, "under") == "loss"

    def test_under_loss_by_small_margin(self):
        assert grade_bet(20.1, 20.0, "under") == "loss"

    def test_under_push_when_actual_equals_line(self):
        assert grade_bet(20.0, 20.0, "under") == "push"

    # --- Exact line push (both sides) ---

    def test_exact_line_is_push_regardless_of_side_over(self):
        assert grade_bet(47.5, 47.5, "over") == "push"

    def test_exact_line_is_push_regardless_of_side_under(self):
        assert grade_bet(47.5, 47.5, "under") == "push"

    def test_push_at_integer_line(self):
        assert grade_bet(300.0, 300.0, "over") == "push"

    def test_push_at_zero_line(self):
        assert grade_bet(0.0, 0.0, "over") == "push"

    # --- NaN actual ---

    def test_nan_actual_returns_push(self):
        assert grade_bet(float("nan"), 20.0, "over") == "push"

    def test_nan_actual_returns_push_under(self):
        assert grade_bet(float("nan"), 20.0, "under") == "push"

    def test_math_nan_actual_returns_push(self):
        assert grade_bet(math.nan, 115.5, "over") == "push"

    # --- Case insensitivity ---

    def test_side_upper_over_win(self):
        assert grade_bet(25.0, 20.0, "OVER") == "win"

    def test_side_upper_under_win(self):
        assert grade_bet(15.0, 20.0, "UNDER") == "win"

    def test_side_mixed_case(self):
        assert grade_bet(25.0, 20.0, "Over") == "win"

    # --- Negative actual (e.g., stat can be 0 but model edge case) ---

    def test_actual_zero_under_positive_line(self):
        assert grade_bet(0.0, 5.5, "under") == "win"

    def test_actual_zero_over_positive_line(self):
        assert grade_bet(0.0, 5.5, "over") == "loss"


# ======================================================================
# calculate_profit_units
# ======================================================================


class TestCalculateProfitUnits:
    # --- Push ---

    def test_push_returns_zero(self):
        assert calculate_profit_units("push", -110) == 0.0

    def test_push_with_positive_odds_returns_zero(self):
        assert calculate_profit_units("push", 150) == 0.0

    def test_push_with_zero_odds_returns_zero(self):
        assert calculate_profit_units("push", 0) == 0.0

    # --- Loss ---

    def test_loss_returns_negative_one(self):
        assert calculate_profit_units("loss", -110) == -1.0

    def test_loss_with_positive_odds_returns_negative_one(self):
        assert calculate_profit_units("loss", 200) == -1.0

    def test_loss_with_standard_juice_returns_negative_one(self):
        assert calculate_profit_units("loss", -115) == -1.0

    # --- Win with negative odds (favorites) ---

    def test_win_at_minus_110_returns_correct_units(self):
        result = calculate_profit_units("win", -110)
        assert abs(result - (100.0 / 110.0)) < 1e-9

    def test_win_at_minus_100_returns_one_unit(self):
        result = calculate_profit_units("win", -100)
        assert abs(result - 1.0) < 1e-9

    def test_win_at_minus_200_returns_half_unit(self):
        result = calculate_profit_units("win", -200)
        assert abs(result - 0.5) < 1e-9

    def test_win_at_minus_150_correct(self):
        result = calculate_profit_units("win", -150)
        assert abs(result - (100.0 / 150.0)) < 1e-9

    def test_win_negative_odds_formula(self):
        """Win with negative odds: profit = 100 / abs(price)."""
        for price in [-110, -120, -130, -150, -200]:
            expected = 100.0 / abs(price)
            assert abs(calculate_profit_units("win", price) - expected) < 1e-9

    # --- Win with positive odds (underdogs) ---

    def test_win_at_plus_150_returns_one_and_half_units(self):
        result = calculate_profit_units("win", 150)
        assert abs(result - 1.5) < 1e-9

    def test_win_at_plus_100_returns_one_unit(self):
        result = calculate_profit_units("win", 100)
        assert abs(result - 1.0) < 1e-9

    def test_win_at_plus_200_returns_two_units(self):
        result = calculate_profit_units("win", 200)
        assert abs(result - 2.0) < 1e-9

    def test_win_at_plus_300_returns_three_units(self):
        result = calculate_profit_units("win", 300)
        assert abs(result - 3.0) < 1e-9

    def test_win_positive_odds_formula(self):
        """Win with positive odds: profit = price / 100."""
        for price in [100, 110, 125, 150, 200]:
            expected = price / 100.0
            assert abs(calculate_profit_units("win", price) - expected) < 1e-9

    # --- Unknown result ---

    def test_unknown_result_returns_zero(self):
        assert calculate_profit_units("unknown", -110) == 0.0

    def test_empty_string_result_returns_zero(self):
        assert calculate_profit_units("", -110) == 0.0

    # --- Boundary: negative odds at -100 vs positive at +100 ---

    def test_minus_100_and_plus_100_both_yield_one_unit(self):
        neg = calculate_profit_units("win", -100)
        pos = calculate_profit_units("win", 100)
        assert abs(neg - 1.0) < 1e-9
        assert abs(pos - 1.0) < 1e-9


# ======================================================================
# get_confidence_tier
# ======================================================================


class TestGetConfidenceTier:
    # --- HIGH tier (>= 15.0) ---

    def test_exactly_15_is_high(self):
        assert get_confidence_tier(15.0) == "HIGH"

    def test_above_15_is_high(self):
        assert get_confidence_tier(20.0) == "HIGH"

    def test_large_edge_is_high(self):
        assert get_confidence_tier(50.0) == "HIGH"

    def test_just_above_15_is_high(self):
        assert get_confidence_tier(15.001) == "HIGH"

    # --- MEDIUM tier (>= 8.0 and < 15.0) ---

    def test_exactly_8_is_medium(self):
        assert get_confidence_tier(8.0) == "MEDIUM"

    def test_mid_range_8_to_15_is_medium(self):
        assert get_confidence_tier(11.5) == "MEDIUM"

    def test_just_below_15_is_medium(self):
        assert get_confidence_tier(14.999) == "MEDIUM"

    def test_just_above_8_is_medium(self):
        assert get_confidence_tier(8.001) == "MEDIUM"

    # --- LOW tier (>= 3.0 and < 8.0) ---

    def test_exactly_3_is_low(self):
        assert get_confidence_tier(3.0) == "LOW"

    def test_mid_range_3_to_8_is_low(self):
        assert get_confidence_tier(5.5) == "LOW"

    def test_just_below_8_is_low(self):
        assert get_confidence_tier(7.999) == "LOW"

    def test_just_above_3_is_low(self):
        assert get_confidence_tier(3.001) == "LOW"

    # --- MINIMAL tier (< 3.0) ---

    def test_just_below_3_is_minimal(self):
        assert get_confidence_tier(2.999) == "MINIMAL"

    def test_zero_is_minimal(self):
        assert get_confidence_tier(0.0) == "MINIMAL"

    def test_negative_edge_is_minimal(self):
        assert get_confidence_tier(-5.0) == "MINIMAL"

    def test_small_positive_is_minimal(self):
        assert get_confidence_tier(1.0) == "MINIMAL"

    # --- Boundary precision ---

    def test_15_boundary_is_inclusive_high_not_medium(self):
        assert get_confidence_tier(15.0) == "HIGH"
        assert get_confidence_tier(14.9) == "MEDIUM"

    def test_8_boundary_is_inclusive_medium_not_low(self):
        assert get_confidence_tier(8.0) == "MEDIUM"
        assert get_confidence_tier(7.9) == "LOW"

    def test_3_boundary_is_inclusive_low_not_minimal(self):
        assert get_confidence_tier(3.0) == "LOW"
        assert get_confidence_tier(2.9) == "MINIMAL"
