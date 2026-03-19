"""NCAAB bracket modifier functions for 8-signal Smart Modifiers system.

Pure-Python, no scipy. Each modifier is a capped multiplicative factor
that adjusts composite_rating or log5 probability. All functions are
pure (no side effects, no DB calls).

Signals:
1. BartTorvik T-Rank agreement (pre-game)
2. Historical seed matchup prior (game-level)
3. Vegas line calibration (game-level)
4. Coaching tournament experience (pre-game)
5. Roster experience / returning minutes (pre-game)
6. Hot streak / momentum (pre-game)
7. Enhanced luck regression (handled in ncaab_ratings.py weight change)
8. Tempo matchup factor (game-level)
"""

from __future__ import annotations


# Signal 2: 40 years of seed-vs-seed NCAA tournament upset rates.
# Key: (favorite_seed, underdog_seed) where favorite < underdog.
# Value: P(underdog wins) i.e. the historical upset rate.
SEED_MATCHUP_UPSET_RATES: dict[tuple[int, int], float] = {
    # First Round
    (1, 16): 0.013,
    (2, 15): 0.060,
    (3, 14): 0.153,
    (4, 13): 0.213,
    (5, 12): 0.357,
    (6, 11): 0.374,
    (7, 10): 0.390,
    (8, 9): 0.480,
    # Second Round (typical matchups)
    (1, 8): 0.200,
    (1, 9): 0.170,
    (2, 7): 0.260,
    (2, 10): 0.240,
    (3, 6): 0.310,
    (3, 11): 0.279,
    (4, 5): 0.460,
    (4, 12): 0.330,
    # Sweet 16 (typical matchups)
    (1, 4): 0.280,
    (1, 5): 0.250,
    (2, 3): 0.420,
    (2, 6): 0.350,
    # Elite 8
    (1, 2): 0.400,
    (1, 3): 0.350,
}


def barttorvik_factor(kenpom_rank: int, barttorvik_rank: int) -> float:
    """Signal 1: Cross-system rank agreement modifier.

    When BartTorvik ranks a team higher than KenPom (lower rank number),
    it indicates the team may be undervalued. Cap: +/-5%.
    """
    raw_delta = (kenpom_rank - barttorvik_rank) * 0.002
    clamped = max(-0.05, min(0.05, raw_delta))
    return 1.0 + clamped


def coaching_factor(tournament_win_rate: float) -> float:
    """Signal 4: Head coach tournament experience modifier.

    Coaches with strong tournament records (> .500) get a boost.
    Uses 0.500 as neutral baseline. Cap: +/-5%.
    """
    raw_delta = (tournament_win_rate - 0.500) * 0.10
    clamped = max(-0.05, min(0.05, raw_delta))
    return 1.0 + clamped


def experience_factor(returning_minutes_pct: float) -> float:
    """Signal 5: Roster continuity modifier.

    Teams with high returning minutes are battle-tested for March.
    Baseline: 0.50 (half the minutes returning). Cap: +/-4%.
    """
    raw_delta = (returning_minutes_pct - 0.50) * 0.08
    clamped = max(-0.04, min(0.04, raw_delta))
    return 1.0 + clamped


def momentum_factor(
    last_10_wins: int,
    conf_tourney_result: str,
    winning_streak: int,
) -> float:
    """Signal 6: Hot streak and conference tournament momentum.

    Components: recent record, conf tourney result, active streak.
    Cap: [-3%, +5%].
    """
    form_delta = (last_10_wins - 5) * 0.004

    conf_bonus_map = {
        "champion": 0.030,
        "runner_up": 0.015,
        "semifinal": 0.005,
        "quarterfinal": 0.000,
        "first_round": -0.005,
        "did_not_qualify": -0.015,
    }
    conf_delta = conf_bonus_map.get(conf_tourney_result, 0.0)

    streak_delta = min(0.020, winning_streak * 0.005)

    raw_total = form_delta + conf_delta + streak_delta
    clamped = max(-0.03, min(0.05, raw_total))
    return 1.0 + clamped


def enhanced_composite_rating(
    base_rating: float,
    bt_factor: float,
    coach_factor: float,
    exp_factor: float,
    mom_factor: float,
) -> float:
    """Apply pre-game multiplicative modifiers to base composite rating.

    Returns new rating clamped to [0.01, 0.99].
    """
    enhanced = base_rating * bt_factor * coach_factor * exp_factor * mom_factor
    return max(0.01, min(0.99, enhanced))


def seed_matchup_prior(seed_a: int, seed_b: int) -> float:
    """Signal 2: Historical P(A wins) from 40 years of seed matchup data.

    Returns 0.50 for seed combinations not in the lookup table.
    """
    if seed_a == seed_b:
        return 0.50
    fav_seed = min(seed_a, seed_b)
    dog_seed = max(seed_a, seed_b)
    upset_rate = SEED_MATCHUP_UPSET_RATES.get((fav_seed, dog_seed))
    if upset_rate is None:
        return 0.50
    if seed_a > seed_b:
        return upset_rate  # A is the underdog
    return 1.0 - upset_rate  # A is the favorite


def blend_with_seed_prior(
    model_p: float,
    seed_a: int,
    seed_b: int,
    prior_weight: float = 0.30,
) -> float:
    """Signal 2 applied: blend model probability with historical seed prior.

    70% model, 30% historical prior by default.
    """
    historical_p = seed_matchup_prior(seed_a, seed_b)
    return (1.0 - prior_weight) * model_p + prior_weight * historical_p


def vegas_implied_prob(moneyline: int) -> float:
    """Convert American moneyline to implied win probability (no-vig)."""
    if moneyline > 0:
        return 100.0 / (moneyline + 100.0)
    return abs(moneyline) / (abs(moneyline) + 100.0)


def blend_with_vegas(
    model_p: float,
    vegas_p: float | None,
    model_weight: float = 0.60,
) -> float:
    """Signal 3: Blend with Vegas when disagreement exceeds 10%.

    Returns model_p unchanged when vegas_p is None or within 10%.
    """
    if vegas_p is None:
        return model_p
    if abs(model_p - vegas_p) <= 0.10:
        return model_p
    return model_weight * model_p + (1.0 - model_weight) * vegas_p


def tempo_factor(adj_t_a: float, adj_t_b: float, is_a_underdog: bool) -> float:
    """Signal 8: Tempo matchup modifier for underdogs who control pace.

    Slow underdogs gain an edge by controlling tempo in March.
    Threshold: 4+ possessions/game differential. Cap: [0.98, 1.03].
    """
    tempo_diff = adj_t_a - adj_t_b  # positive = A is faster

    if is_a_underdog and tempo_diff < -4:
        # A is slower underdog controlling pace
        raw = min(0.03, abs(tempo_diff + 4) * 0.005)
        return 1.0 + raw
    if not is_a_underdog and tempo_diff > 4:
        # B is slower underdog — boost applies from B's perspective
        raw = min(0.03, abs(tempo_diff - 4) * 0.005)
        return 1.0 + raw
    return 1.0
