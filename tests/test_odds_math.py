"""Tests for American-odds conversions (utils/odds_math.py)."""

from __future__ import annotations

import pytest

from utils.odds_math import implied_probability, implied_probability_no_vig


@pytest.mark.parametrize(
    ("odds", "expected"),
    [(-110, 110 / 210), (-200, 200 / 300), (100, 0.5), (150, 0.4)],
)
def test_implied_probability_converts_american_odds(odds: int, expected: float) -> None:
    assert implied_probability(odds) == pytest.approx(expected)


def test_no_vig_splits_a_symmetric_book_evenly() -> None:
    p_over, p_under = implied_probability_no_vig(-110, -110)
    assert p_over == pytest.approx(0.5)
    assert p_under == pytest.approx(0.5)


def test_no_vig_sums_to_one_and_keeps_the_favorite_ahead() -> None:
    p_over, p_under = implied_probability_no_vig(-130, 110)
    assert p_over + p_under == pytest.approx(1.0)
    assert p_over > p_under


def test_no_vig_removes_margin_from_the_raw_price() -> None:
    p_over, _ = implied_probability_no_vig(-110, -110)
    assert p_over < implied_probability(-110)
