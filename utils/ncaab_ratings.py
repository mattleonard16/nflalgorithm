"""NCAAB rating math for March Madness bracket prediction.

Pure-Python, no scipy. Provides:
- sigmoid: logistic function for normalizing unbounded metrics to [0,1]
- pyth_win_pct: Pythagorean win% from KenPom AdjOE/AdjDE
- compute_trapezoid_score: 0-4 Trapezoid of Excellence score
- composite_rating: blended tournament rating from KenPom metrics
- log5: Bill James head-to-head win probability
- confidence_tier: classify prediction confidence
"""

from __future__ import annotations

import math


def sigmoid(x: float) -> float:
    """Logistic sigmoid, safely handling large |x|."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def pyth_win_pct(adj_oe: float, adj_de: float, exponent: float = 11.5) -> float:
    """KenPom Pythagorean win% using exponent 11.5 for college basketball."""
    o = adj_oe ** exponent
    d = adj_de ** exponent
    if o + d == 0:
        return 0.5
    return o / (o + d)


def compute_trapezoid_score(
    adj_oe_rank: int,
    adj_de_rank: int,
    sos_rank: int,
    adj_em_rank: int,
) -> int:
    """Trapezoid of Excellence: 0-4 points for elite KenPom dimensions.

    Teams scoring 4/4 historically dominate March Madness.
    Thresholds based on national D1 rankings (~364 teams).
    """
    score = 0
    if adj_oe_rank <= 40:
        score += 1
    if adj_de_rank <= 40:
        score += 1
    if sos_rank <= 40:
        score += 1
    if adj_em_rank <= 15:
        score += 1
    return score


# Historical seed win rates from ~40 years of NCAA tournament data.
# Used as a small prior nudge (+-5% range) -- not enough to override metrics.
SEED_WIN_RATES: dict[int, float] = {
    1: 0.85, 2: 0.72, 3: 0.65, 4: 0.57,
    5: 0.50, 6: 0.50, 7: 0.45, 8: 0.40,
    9: 0.38, 10: 0.38, 11: 0.38, 12: 0.35,
    13: 0.21, 14: 0.15, 15: 0.07, 16: 0.02,
}


def composite_rating(
    adj_em: float,
    adj_oe: float,
    adj_de: float,
    pyth_win: float,
    sos_adj_em: float,
    luck: float,
    seed: int,
    trapezoid_score: int,
) -> float:
    """Compute tournament composite rating in [0.01, 0.99].

    Weights tuned for March Madness prediction:
    - 40% AdjEM (efficiency margin -- best single predictor)
    - 20% AdjDE (defense travels to neutral sites)
    - 15% Pyth win% (overall quality)
    - 15% SOS (battle-tested schedule)
    - 10% Anti-luck (positive luck = regression risk)
    """
    em_score = sigmoid(adj_em / 15.0)
    de_score = sigmoid((105.0 - adj_de) / 8.0)
    pyth_score = pyth_win
    sos_score = sigmoid(sos_adj_em / 10.0)
    luck_score = sigmoid(-luck * 5.0)

    raw = (
        0.40 * em_score
        + 0.20 * de_score
        + 0.15 * pyth_score
        + 0.15 * sos_score
        + 0.10 * luck_score
    )

    trapezoid_bonus = trapezoid_score * 0.0125

    seed_rate = SEED_WIN_RATES.get(seed, 0.30)
    seed_mult = 0.95 + (seed_rate * 0.10)

    adjusted = (raw + trapezoid_bonus) * seed_mult
    return max(0.01, min(0.99, adjusted))


def log5(r_a: float, r_b: float) -> float:
    """Bill James log5: P(A beats B) given composite ratings."""
    r_a = max(0.001, min(0.999, r_a))
    r_b = max(0.001, min(0.999, r_b))
    num = r_a * (1.0 - r_b)
    den = num + r_b * (1.0 - r_a)
    return num / den


def confidence_tier(p_winner: float, is_upset: bool) -> str:
    """Classify prediction confidence."""
    if is_upset and p_winner >= 0.40:
        return "upset_pick"
    if p_winner >= 0.80:
        return "lock"
    if p_winner >= 0.65:
        return "likely"
    return "toss-up"
