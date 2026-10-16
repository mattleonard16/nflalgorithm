"""Property tests for the betting math (CLAUDE.md punch item 24).

``hypothesis`` is not installed in this environment, so these follow the
repo's existing property-style convention (see ``tests/test_risk_manager.py``):
each invariant is checked over a seeded random sample of the input space plus
its boundary values. A failure names the sampled input.

Covered: the no-vig pair (``utils.odds_math``), decimal odds against implied
probability, profit units on a win (``utils.grading``), the tracked Kelly
sizing (``nba_value_engine.kelly_fraction``; the NFL engine's copy is private),
and the CLV sign invariants (``utils.clv``).
"""

from __future__ import annotations

import numpy as np
import pytest

from nba_value_engine import kelly_fraction
from utils.clv import compute_clv
from utils.grading import calculate_profit_units
from utils.odds_math import (
    american_to_decimal,
    implied_probability,
    implied_probability_no_vig,
)

_RNG = np.random.default_rng(20261015)
_BOUNDARY_ODDS = [-10000, -1000, -200, -110, -101, -100, 100, 101, 110, 200, 1000, 10000]


def _sample_odds(n: int) -> list[int]:
    """Valid American odds: at or below -100, or at or above +100."""
    magnitude = _RNG.integers(100, 5001, size=n)
    sign = _RNG.choice([-1, 1], size=n)
    return [int(v) for v in magnitude * sign] + _BOUNDARY_ODDS


ODDS = _sample_odds(100)
ODDS_PAIRS = list(zip(_sample_odds(100), reversed(_sample_odds(100))))


# ---------------------------------------------------------------------------
# odds_math
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("over,under", ODDS_PAIRS)
def test_no_vig_pair_sums_to_one_and_preserves_order(over, under):
    p_over, p_under = implied_probability_no_vig(over, under)

    assert 0.0 < p_over < 1.0 and 0.0 < p_under < 1.0
    assert p_over + p_under == pytest.approx(1.0, abs=1e-12)
    raw_over, raw_under = implied_probability(over), implied_probability(under)
    # De-vigging rescales both sides by one factor, so it never reorders them.
    assert np.sign(p_over - p_under) == np.sign(round(raw_over - raw_under, 12))


@pytest.mark.parametrize("odds", ODDS)
def test_decimal_odds_are_the_reciprocal_of_implied_probability(odds):
    assert american_to_decimal(odds) == pytest.approx(1.0 / implied_probability(odds))


@pytest.mark.parametrize("odds", ODDS)
def test_a_win_pays_decimal_odds_minus_the_stake(odds):
    assert calculate_profit_units("win", odds) == pytest.approx(american_to_decimal(odds) - 1.0)
    assert calculate_profit_units("loss", odds) == -1.0
    assert calculate_profit_units("push", odds) == 0.0


@pytest.mark.parametrize("odds", [0, 1, 50, 99, -1, -50, -99, float("nan")])
def test_values_between_minus_and_plus_100_are_not_odds(odds):
    with pytest.raises(ValueError):
        implied_probability(odds)
    with pytest.raises(ValueError):
        american_to_decimal(odds)


# ---------------------------------------------------------------------------
# Kelly sizing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("odds", ODDS[::5])
def test_kelly_is_zero_without_an_edge_and_monotone_in_win_probability(odds):
    breakeven = implied_probability(odds)
    probs = np.linspace(0.0, 1.0, 101)
    stakes = [kelly_fraction(float(p), odds) for p in probs]

    for p, stake in zip(probs, stakes):
        assert 0.0 <= stake <= 0.10
        if p <= breakeven:
            assert stake == 0.0, f"stake {stake} at p={p} with no edge over {breakeven}"
    assert all(b >= a for a, b in zip(stakes, stakes[1:])), "Kelly fell as p rose"


# ---------------------------------------------------------------------------
# CLV sign invariants
# ---------------------------------------------------------------------------


def _clv_cases(n: int) -> list[tuple]:
    cases = []
    for _ in range(n):
        side = str(_RNG.choice(["over", "under"]))
        entry_line = float(_RNG.integers(10, 120)) + 0.5
        move = float(_RNG.integers(1, 10))
        mu = entry_line + float(_RNG.normal(0.0, 10.0))
        sigma = float(_RNG.uniform(10.0, 30.0))
        two_sided = bool(_RNG.integers(0, 2))
        cases.append((side, entry_line, move, max(mu, 15.0), sigma, two_sided))
    return cases


@pytest.mark.parametrize("side,entry_line,move,mu,sigma,two_sided", _clv_cases(60))
def test_a_line_move_toward_our_side_is_positive_clv(side, entry_line, move, mu, sigma, two_sided):
    """Over: the close rose. Under: the close fell. Either branch, CLV > 0;
    the mirror move is negative."""
    toward = entry_line + move if side == "over" else entry_line - move
    away = entry_line - move if side == "over" else entry_line + move
    entry = {
        "line": entry_line,
        "side": side,
        "price": -110,
        "under_price": -110 if two_sided else None,
        "mu": mu,
        "sigma": sigma,
        "market": "receiving_yards",
    }

    def close(line: float) -> dict:
        return {
            "close_line": line,
            "close_price": -110,
            "close_under_price": -110 if two_sided else None,
            "closed_at": "2025-11-27T18:00:00+00:00",
            "snapshot_count": 3,
        }

    won = compute_clv(entry, close(toward))
    lost = compute_clv(entry, close(away))

    assert won["clv_points"] == pytest.approx(move)
    assert lost["clv_points"] == pytest.approx(-move)
    assert won["clv_bp"] > 0
    assert lost["clv_bp"] < 0


@pytest.mark.parametrize("over,under", ODDS_PAIRS[::10])
@pytest.mark.parametrize("side", ["over", "under"])
def test_a_price_move_at_the_same_line_is_signed_by_our_side(over, under, side):
    """Same line, the close shortened one side: positive for that side's bettor."""
    entry = {"line": 50.5, "side": side, "price": over, "under_price": under}
    entry_p_over, _ = implied_probability_no_vig(over, under)
    close = {
        "close_line": 50.5,
        "close_price": -150,
        "close_under_price": 130,
        "closed_at": "2025-11-27T18:00:00+00:00",
        "snapshot_count": 3,
    }
    close_p_over, _ = implied_probability_no_vig(-150, 130)

    result = compute_clv(entry, close)

    moved_toward_over = close_p_over - entry_p_over
    signed = moved_toward_over if side == "over" else -moved_toward_over
    assert result["clv_bp"] == pytest.approx(signed * 10_000, abs=1e-2)
