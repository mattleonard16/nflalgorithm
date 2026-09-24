"""Tests for the shared American-odds math (utils/odds_math.py).

Moved out of tests/test_no_vig_probability.py, which only runs next to the
private engine, so CI now checks the de-vig that CLV grading depends on.
"""

from __future__ import annotations

import pytest

import utils.odds_math
from utils.odds_math import implied_probability, implied_probability_no_vig


def test_implied_probability_no_vig_symmetric_book():
    """-110/-110 → 50/50 after vig removal."""
    p_over, p_under = implied_probability_no_vig(-110, -110)
    assert abs(p_over - 0.5) < 1e-9
    assert abs(p_under - 0.5) < 1e-9
    assert abs(p_over + p_under - 1.0) < 1e-9


def test_implied_probability_no_vig_asymmetric_book():
    """Asymmetric quotes still sum to 1.0 after normalization."""
    p_over, p_under = implied_probability_no_vig(-130, +110)
    assert abs(p_over + p_under - 1.0) < 1e-9
    # Over favored, so its no-vig prob > under
    assert p_over > p_under


def test_implied_probability_no_vig_strictly_below_raw():
    """Removing vig must reduce the over-side implied prob relative to raw."""
    raw_over = implied_probability(-110)  # 0.5238
    p_over, _ = implied_probability_no_vig(-110, -110)
    assert p_over < raw_over


def test_implied_probability_no_vig_raises_on_zero_total(monkeypatch):
    """Guards against degenerate input — both raw probs sum to 0."""
    monkeypatch.setattr(utils.odds_math, "implied_probability", lambda _: 0.0)
    with pytest.raises(ValueError):
        implied_probability_no_vig(-110, -110)
