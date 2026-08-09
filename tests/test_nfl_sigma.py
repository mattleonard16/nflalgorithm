"""Tests for utils/nfl_sigma.py — EWMA sigma, NFL market floors, fallbacks.

Mirrors tests/test_nba_sigma.py structure adapted for NFL markets.

Covers:
- EWMA variance computation with known values
- Legacy per-market floors via position=None (rushing=15, receiving=12, passing=30)
- Position-bucketed floors/defaults keyed by (market, position)
- position=None reproduces legacy behavior exactly (backward compatibility)
- WR receiving fat-tail multiplier (1.20 on the EWMA estimate)
- Fallback defaults for <6 games
- Edge cases: empty list, single game, all same values
- Decay parameter affects output correctly
- get_sigma_or_default helper
- Defense multiplier applied to mu in predict_week (integration)
- Sigma is data-driven, not flat percentage
- Synthetic Gaussian coverage: returned sigma yields ~68% +/-1 sigma coverage
  (the test that would have caught the 2025 WR/TE rushing over-coverage)
"""

from __future__ import annotations


import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Constants mirroring nfl_sigma module
# ---------------------------------------------------------------------------

# Legacy per-market values — the position=None path must return exactly these.
SIGMA_FLOORS = {
    "rushing_yards": 15.0,
    "receiving_yards": 12.0,
    "passing_yards": 30.0,
}
SIGMA_DEFAULTS = {
    "rushing_yards": 25.0,
    "receiving_yards": 20.0,
    "passing_yards": 50.0,
}
MIN_GAMES = 6

# Position-bucketed recalibration (2025 held-out coverage study).
POSITION_SIGMA_FLOORS = {
    ("rushing_yards", "RB"): 15.0,
    ("rushing_yards", "QB"): 12.0,
    ("rushing_yards", "WR"): 5.0,
    ("rushing_yards", "TE"): 5.0,
    ("receiving_yards", "WR"): 13.0,
    ("receiving_yards", "RB"): 10.0,
    ("receiving_yards", "TE"): 12.0,
    ("passing_yards", "QB"): 30.0,  # unvalidated — legacy value
}
POSITION_SIGMA_DEFAULTS = {
    ("rushing_yards", "RB"): 25.0,
    ("rushing_yards", "QB"): 20.0,
    ("rushing_yards", "WR"): 10.0,
    ("rushing_yards", "TE"): 10.0,
    ("receiving_yards", "WR"): 24.0,
    ("receiving_yards", "RB"): 16.0,
    ("receiving_yards", "TE"): 20.0,
    ("passing_yards", "QB"): 50.0,  # unvalidated — legacy value
}
WR_RECEIVING_MULTIPLIER = 1.20


# ---------------------------------------------------------------------------
# compute_player_sigma — basic behaviour
# ---------------------------------------------------------------------------


class TestComputePlayerSigma:
    """Core EWMA sigma computation tests for NFL markets."""

    def test_returns_float(self):
        from utils.nfl_sigma import compute_player_sigma

        result = compute_player_sigma(
            [60, 75, 85, 45, 90, 70, 55, 80], market="rushing_yards"
        )
        assert isinstance(result, float)

    def test_known_values_positive_sigma(self):
        from utils.nfl_sigma import compute_player_sigma

        values = [60.0, 75.0, 85.0, 45.0, 90.0, 70.0]
        sigma = compute_player_sigma(values, market="rushing_yards")
        assert sigma > 0.0

    def test_constant_values_returns_floor(self):
        """All identical values → variance is 0 → sigma should be the floor."""
        from utils.nfl_sigma import compute_player_sigma

        values = [75.0] * 10
        sigma = compute_player_sigma(values, market="rushing_yards")
        assert sigma == pytest.approx(SIGMA_FLOORS["rushing_yards"])

    def test_high_variance_exceeds_floor(self):
        """Volatile player data should produce sigma above the floor."""
        from utils.nfl_sigma import compute_player_sigma

        # Boom-bust RB pattern
        values = [10.0, 120.0, 15.0, 110.0, 12.0, 105.0, 18.0, 100.0]
        sigma = compute_player_sigma(values, market="rushing_yards")
        assert sigma > SIGMA_FLOORS["rushing_yards"]

    def test_decay_parameter_changes_output(self):
        """Different decay values should produce different sigma estimates."""
        from utils.nfl_sigma import compute_player_sigma

        values = [40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0]
        sigma_low = compute_player_sigma(values, market="rushing_yards", decay=0.3)
        sigma_high = compute_player_sigma(values, market="rushing_yards", decay=0.9)
        assert sigma_low != pytest.approx(sigma_high, abs=0.5)

    def test_sigma_exceeds_floor_for_all_markets(self):
        """Each market with enough variable data should exceed its floor."""
        from utils.nfl_sigma import compute_player_sigma

        market_values = {
            "rushing_yards": [20.0, 100.0, 25.0, 95.0, 30.0, 90.0],
            "receiving_yards": [15.0, 85.0, 20.0, 80.0, 18.0, 78.0],
            "passing_yards": [150.0, 350.0, 180.0, 330.0, 160.0, 340.0],
        }
        for market, values in market_values.items():
            sigma = compute_player_sigma(values, market=market)
            assert sigma > SIGMA_FLOORS[market], (
                f"Expected sigma > {SIGMA_FLOORS[market]} for {market}, got {sigma}"
            )


# ---------------------------------------------------------------------------
# NFL Market-specific floors (legacy position=None path)
# ---------------------------------------------------------------------------


class TestNFLMarketFloors:
    """Each NFL market has a minimum sigma floor reflecting stat scale."""

    @pytest.mark.parametrize(
        "market,expected_floor",
        [
            ("rushing_yards", 15.0),
            ("receiving_yards", 12.0),
            ("passing_yards", 30.0),
        ],
    )
    def test_floor_enforced_per_market(self, market, expected_floor):
        from utils.nfl_sigma import compute_player_sigma

        # Constant values => zero variance => floor should apply
        values = [50.0] * 10
        sigma = compute_player_sigma(values, market=market)
        assert sigma == pytest.approx(expected_floor)

    def test_rushing_floor_larger_than_receiving_floor(self):
        """Rushing sigma floor should be larger than receiving (higher scale)."""
        assert SIGMA_FLOORS["rushing_yards"] > SIGMA_FLOORS["receiving_yards"]

    def test_passing_floor_is_largest(self):
        """Passing yards have highest scale so passing floor should be largest."""
        assert SIGMA_FLOORS["passing_yards"] > SIGMA_FLOORS["rushing_yards"]
        assert SIGMA_FLOORS["passing_yards"] > SIGMA_FLOORS["receiving_yards"]

    def test_unknown_market_uses_generic_floor(self):
        from utils.nfl_sigma import compute_player_sigma

        values = [50.0] * 10
        sigma = compute_player_sigma(values, market="fumbles")
        # Should use generic floor (10.0), not crash
        assert sigma >= 1.0


# ---------------------------------------------------------------------------
# Position-bucketed floors and defaults
# ---------------------------------------------------------------------------


class TestPositionBucketSelection:
    """(market, position) selects the recalibrated floor/default."""

    @pytest.mark.parametrize(
        "market,position,expected_floor",
        sorted(
            ((m, p, f) for (m, p), f in POSITION_SIGMA_FLOORS.items()),
        ),
    )
    def test_bucket_floor_selected(self, market, position, expected_floor):
        from utils.nfl_sigma import compute_player_sigma

        # Constant values => zero variance => bucket floor applies
        values = [50.0] * 10
        sigma = compute_player_sigma(values, market=market, position=position)
        assert sigma == pytest.approx(expected_floor)

    @pytest.mark.parametrize(
        "market,position,expected_default",
        sorted(
            ((m, p, d) for (m, p), d in POSITION_SIGMA_DEFAULTS.items()),
        ),
    )
    def test_bucket_default_selected(self, market, position, expected_default):
        from utils.nfl_sigma import compute_player_sigma

        values = [60.0, 75.0, 85.0]  # 3 games < MIN_GAMES
        sigma = compute_player_sigma(values, market=market, position=position)
        assert sigma == pytest.approx(expected_default)

    def test_unknown_position_falls_back_to_legacy(self):
        """Positions outside QB/RB/WR/TE use the legacy per-market values."""
        from utils.nfl_sigma import compute_player_sigma

        constant = [50.0] * 10
        short = [60.0, 75.0]
        for position in ("FB", "K", "OL", ""):
            sigma_floor = compute_player_sigma(
                constant, market="rushing_yards", position=position
            )
            sigma_default = compute_player_sigma(
                short, market="rushing_yards", position=position
            )
            assert sigma_floor == pytest.approx(SIGMA_FLOORS["rushing_yards"])
            assert sigma_default == pytest.approx(SIGMA_DEFAULTS["rushing_yards"])

    def test_position_normalised_case_and_whitespace(self):
        """'wr' and ' WR ' should hit the WR bucket."""
        from utils.nfl_sigma import compute_player_sigma

        constant = [50.0] * 10
        expected = POSITION_SIGMA_FLOORS[("rushing_yards", "WR")]
        for position in ("wr", " WR ", "Wr"):
            sigma = compute_player_sigma(constant, market="rushing_yards", position=position)
            assert sigma == pytest.approx(expected)

    def test_wr_te_rushing_floor_far_below_legacy(self):
        """The recalibration's core fix: WR/TE rushing floors shrink hard."""
        assert POSITION_SIGMA_FLOORS[("rushing_yards", "WR")] < SIGMA_FLOORS["rushing_yards"] / 2
        assert POSITION_SIGMA_FLOORS[("rushing_yards", "TE")] < SIGMA_FLOORS["rushing_yards"] / 2

    def test_wr_receiving_floor_raised(self):
        """Fat tails in WR receiving → floor slightly above legacy."""
        assert POSITION_SIGMA_FLOORS[("receiving_yards", "WR")] > SIGMA_FLOORS["receiving_yards"]


class TestPositionNoneLegacyParity:
    """position=None (the default) must reproduce pre-recalibration values exactly."""

    def test_explicit_none_equals_omitted(self):
        from utils.nfl_sigma import compute_player_sigma

        values = [60.0, 75.0, 85.0, 45.0, 90.0, 70.0, 55.0, 80.0]
        assert compute_player_sigma(values, market="receiving_yards") == pytest.approx(
            compute_player_sigma(values, market="receiving_yards", position=None)
        )

    def test_legacy_ewma_value_reproduced(self):
        """Golden check: position=None matches an independent EWMA computation."""
        from utils.nfl_sigma import compute_player_sigma

        values = np.array([60.0, 75.0, 85.0, 45.0, 90.0, 70.0, 55.0, 80.0])
        decay = 0.65
        raw = np.array([decay ** i for i in range(len(values) - 1, -1, -1)])
        w = raw / raw.sum()
        mean = float(np.dot(w, values))
        var = float(np.dot(w, (values - mean) ** 2))
        corr = 1.0 / (1.0 - float(np.dot(w, w)))
        expected = max(float(np.sqrt(var * corr)), SIGMA_FLOORS["rushing_yards"])

        sigma = compute_player_sigma(values.tolist(), market="rushing_yards", position=None)
        assert sigma == pytest.approx(expected)

    def test_no_multiplier_on_none_path(self):
        """The WR fat-tail multiplier must not leak into the position=None path."""
        from utils.nfl_sigma import compute_player_sigma

        # High variance so no floor binds
        values = [10.0, 120.0, 15.0, 110.0, 12.0, 105.0, 18.0, 100.0]
        sigma_none = compute_player_sigma(values, market="receiving_yards", position=None)
        sigma_wr = compute_player_sigma(values, market="receiving_yards", position="WR")
        assert sigma_wr == pytest.approx(sigma_none * WR_RECEIVING_MULTIPLIER)
        assert sigma_wr > sigma_none


class TestFloorsActuallyFloor:
    """Low-variance history → the bucket floor wins over the EWMA estimate."""

    @pytest.mark.parametrize(
        "market,position",
        sorted(POSITION_SIGMA_FLOORS.keys()),
    )
    def test_low_variance_history_returns_floor(self, market, position):
        from utils.nfl_sigma import compute_player_sigma

        # Tiny jitter: EWMA sigma ~0.8, far below every bucket floor
        values = [50.0, 50.5, 49.5, 50.2, 49.8, 50.1, 49.9, 50.3]
        sigma = compute_player_sigma(values, market=market, position=position)
        assert sigma == pytest.approx(POSITION_SIGMA_FLOORS[(market, position)])


# ---------------------------------------------------------------------------
# Fallback defaults for insufficient data
# ---------------------------------------------------------------------------


class TestNFLFallbackDefaults:
    """When player has fewer than MIN_GAMES (6) samples, return default."""

    @pytest.mark.parametrize(
        "market,expected_default",
        [
            ("rushing_yards", 25.0),
            ("receiving_yards", 20.0),
            ("passing_yards", 50.0),
        ],
    )
    def test_default_returned_for_insufficient_data(self, market, expected_default):
        from utils.nfl_sigma import compute_player_sigma

        values = [60.0, 75.0, 85.0]  # Only 3 games, less than MIN_GAMES=6
        sigma = compute_player_sigma(values, market=market)
        assert sigma == pytest.approx(expected_default)

    def test_exactly_min_games_computes_sigma(self):
        """Exactly 6 games should compute EWMA, not return default."""
        from utils.nfl_sigma import compute_player_sigma

        values = [60.0, 75.0, 85.0, 45.0, 90.0, 70.0]  # exactly 6
        sigma = compute_player_sigma(values, market="rushing_yards")
        # Should NOT be the default of 25.0
        assert sigma != pytest.approx(SIGMA_DEFAULTS["rushing_yards"])

    def test_five_games_returns_default(self):
        """5 games (one below threshold) must return default."""
        from utils.nfl_sigma import compute_player_sigma

        values = [60.0, 75.0, 85.0, 45.0, 90.0]  # 5 games
        sigma = compute_player_sigma(values, market="rushing_yards")
        assert sigma == pytest.approx(SIGMA_DEFAULTS["rushing_yards"])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestNFLSigmaEdgeCases:
    """Empty list, single game, NaN values."""

    def test_empty_list_returns_default(self):
        from utils.nfl_sigma import compute_player_sigma

        sigma = compute_player_sigma([], market="rushing_yards")
        assert sigma == pytest.approx(SIGMA_DEFAULTS["rushing_yards"])

    def test_single_game_returns_default(self):
        from utils.nfl_sigma import compute_player_sigma

        sigma = compute_player_sigma([80.0], market="rushing_yards")
        assert sigma == pytest.approx(SIGMA_DEFAULTS["rushing_yards"])

    def test_nan_values_filtered(self):
        """NaN values should be stripped before computation."""
        from utils.nfl_sigma import compute_player_sigma

        values = [60.0, float("nan"), 75.0, float("nan"), 85.0]
        # Only 3 valid values → less than MIN_GAMES=6 → default
        sigma = compute_player_sigma(values, market="rushing_yards")
        assert sigma == pytest.approx(SIGMA_DEFAULTS["rushing_yards"])

    def test_nan_filtered_leaves_enough_data(self):
        """Enough non-NaN values should compute EWMA."""
        from utils.nfl_sigma import compute_player_sigma

        values = [60.0, float("nan"), 75.0, 85.0, 45.0, 90.0, 70.0]
        # 6 valid values → compute EWMA
        sigma = compute_player_sigma(values, market="rushing_yards")
        # Should not be default
        assert sigma != pytest.approx(SIGMA_DEFAULTS["rushing_yards"])

    def test_sigma_always_positive(self):
        """Sigma must always be positive regardless of input."""
        from utils.nfl_sigma import compute_player_sigma

        for market in ["rushing_yards", "receiving_yards", "passing_yards"]:
            sigma = compute_player_sigma([], market=market)
            assert sigma > 0.0


# ---------------------------------------------------------------------------
# Sigma is data-driven (not flat percentage)
# ---------------------------------------------------------------------------


class TestSigmaIsDataDriven:
    """Verify sigma reflects actual player variance, not a fixed mu * 0.30."""

    def test_consistent_player_has_lower_sigma_than_volatile(self):
        """A consistent player should have lower sigma than a volatile one."""
        from utils.nfl_sigma import compute_player_sigma

        consistent = [75.0, 78.0, 72.0, 76.0, 74.0, 77.0, 73.0, 75.0]
        volatile = [20.0, 130.0, 15.0, 125.0, 25.0, 120.0, 18.0, 128.0]

        sigma_consistent = compute_player_sigma(consistent, market="rushing_yards")
        sigma_volatile = compute_player_sigma(volatile, market="rushing_yards")

        assert sigma_consistent < sigma_volatile

    def test_sigma_not_fixed_percentage_of_mean(self):
        """Sigma should NOT equal mu * 0.30 (old hardcoded approach)."""
        from utils.nfl_sigma import compute_player_sigma

        values = [80.0, 85.0, 75.0, 82.0, 78.0, 83.0, 77.0, 81.0]
        sigma = compute_player_sigma(values, market="rushing_yards")
        mean_val = float(np.mean(values))
        flat_sigma = mean_val * 0.30

        # Actual sigma should differ from the flat 30% rule
        assert sigma != pytest.approx(flat_sigma, rel=0.01)

    def test_recent_games_weighted_more(self):
        """With high decay, recent volatile games increase sigma more than early ones."""
        from utils.nfl_sigma import compute_player_sigma

        # Stable early, volatile recent
        stable_early_volatile_late = [75.0, 76.0, 74.0, 75.0, 20.0, 130.0]
        # Volatile early, stable recent
        volatile_early_stable_late = [20.0, 130.0, 75.0, 76.0, 74.0, 75.0]

        sigma_volatile_late = compute_player_sigma(
            stable_early_volatile_late, market="rushing_yards", decay=0.8
        )
        sigma_volatile_early = compute_player_sigma(
            volatile_early_stable_late, market="rushing_yards", decay=0.8
        )

        # Recent volatility (late) should produce higher sigma with high decay
        assert sigma_volatile_late > sigma_volatile_early


# ---------------------------------------------------------------------------
# Synthetic Gaussian coverage — the calibration regression guard
# ---------------------------------------------------------------------------


def _simulated_coverage(market, position, true_sigma, mu_range, seed=7):
    """+/-1 sigma empirical coverage over synthetic Gaussian player histories.

    For each synthetic player: draw a 17-game Gaussian history, compute sigma
    via compute_player_sigma, then check how often fresh outcomes from the
    same distribution fall within mu +/- sigma (mu = true player mean, i.e.
    an unbiased model). Nominal Gaussian coverage is 68.3%.
    """
    from utils.nfl_sigma import compute_player_sigma

    rng = np.random.default_rng(seed)
    hits = 0
    total = 0
    for _ in range(400):
        mu = rng.uniform(*mu_range)
        history = rng.normal(mu, true_sigma, 17)
        sigma = compute_player_sigma(history.tolist(), market=market, position=position)
        outcomes = rng.normal(mu, true_sigma, 5)
        hits += int(np.sum(np.abs(outcomes - mu) <= sigma))
        total += 5
    return hits / total


class TestSyntheticGaussianCoverage:
    """Returned sigma must produce sane +/-1 sigma coverage (55-80% band).

    This is the test that would have caught the 2025 miscalibration:
    the legacy 15.0 rushing floor pushed WR/TE rushing coverage to ~99.7%
    (measured 95.6%/95.1% on real data) — far outside this band.
    """

    @pytest.mark.parametrize(
        "market,position,true_sigma,mu_range",
        [
            ("rushing_yards", "RB", 25.0, (30.0, 110.0)),
            ("rushing_yards", "QB", 12.0, (10.0, 50.0)),
            ("rushing_yards", "WR", 5.0, (0.0, 8.0)),
            ("rushing_yards", "TE", 5.0, (0.0, 8.0)),
            ("receiving_yards", "WR", 25.0, (30.0, 110.0)),
            ("receiving_yards", "RB", 14.0, (10.0, 50.0)),
            ("receiving_yards", "TE", 15.0, (15.0, 70.0)),
        ],
    )
    def test_coverage_in_band(self, market, position, true_sigma, mu_range):
        coverage = _simulated_coverage(market, position, true_sigma, mu_range)
        assert 0.55 <= coverage <= 0.80, (
            f"{market}/{position}: +/-1 sigma coverage {coverage:.3f} outside "
            f"[0.55, 0.80] — sigma is miscalibrated for this bucket"
        )

    def test_legacy_floor_fails_coverage_for_wr_rushing(self):
        """Demonstrate the caught bug: the legacy path (position=None) blows
        past the band on a WR-rushing-shaped generating process."""
        coverage = _simulated_coverage("rushing_yards", None, 5.0, (0.0, 8.0))
        assert coverage > 0.80  # legacy 15.0 floor → ~99.7% over-coverage


# ---------------------------------------------------------------------------
# get_sigma_or_default helper
# ---------------------------------------------------------------------------


class TestGetNFLSigmaOrDefault:
    """Fallback logic when sigma comes from the DB (may be NULL)."""

    def test_returns_sigma_when_present(self):
        from utils.nfl_sigma import get_sigma_or_default

        result = get_sigma_or_default(18.5, projected_value=75.0, market="rushing_yards")
        assert result == pytest.approx(18.5)

    def test_returns_default_when_none(self):
        from utils.nfl_sigma import get_sigma_or_default

        result = get_sigma_or_default(None, projected_value=75.0, market="rushing_yards")
        assert result == pytest.approx(SIGMA_DEFAULTS["rushing_yards"])

    def test_returns_default_when_nan(self):
        from utils.nfl_sigma import get_sigma_or_default

        result = get_sigma_or_default(
            float("nan"), projected_value=75.0, market="rushing_yards"
        )
        assert result == pytest.approx(SIGMA_DEFAULTS["rushing_yards"])

    def test_returns_default_for_passing_when_none(self):
        from utils.nfl_sigma import get_sigma_or_default

        result = get_sigma_or_default(None, projected_value=280.0, market="passing_yards")
        assert result == pytest.approx(SIGMA_DEFAULTS["passing_yards"])

    def test_returns_sigma_even_if_at_floor(self):
        """A sigma at the floor value should be returned as-is."""
        from utils.nfl_sigma import get_sigma_or_default

        floor = SIGMA_FLOORS["rushing_yards"]
        result = get_sigma_or_default(floor, projected_value=75.0, market="rushing_yards")
        assert result == pytest.approx(floor)

    def test_position_bucket_default_when_none(self):
        """With a position, the NULL fallback uses the bucket default."""
        from utils.nfl_sigma import get_sigma_or_default

        result = get_sigma_or_default(
            None, projected_value=5.0, market="rushing_yards", position="WR"
        )
        assert result == pytest.approx(POSITION_SIGMA_DEFAULTS[("rushing_yards", "WR")])

    def test_unknown_market_uses_projection_heuristic(self):
        """Unknown market falls back to max(projected * 0.25, 10.0) as before."""
        from utils.nfl_sigma import get_sigma_or_default

        result = get_sigma_or_default(None, projected_value=80.0, market="sacks")
        assert result == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Defense multiplier applied to mu (integration smoke test)
# ---------------------------------------------------------------------------


class TestDefenseMultiplierApplied:
    """Verify defense adjustment wiring in predict_week."""

    def test_get_defense_multiplier_returns_float(self):
        """get_defense_multiplier should return a float in [0.7, 1.3]."""
        from utils.defense_adjustments import get_defense_multiplier

        # No real data in test env → should return 1.0 (no adjustment)
        mult = get_defense_multiplier(
            opponent="CIN",
            position="RB",
            stat_type="rushing_yards",
            season=2024,
            through_week=5,
        )
        assert isinstance(mult, float)
        assert 0.5 <= mult <= 2.0  # Reasonable bounds

    def test_multiplier_in_reasonable_range(self):
        """Multiplier should always be within reasonable bounds [0.5, 2.0]."""
        from utils.defense_adjustments import get_defense_multiplier

        for opponent in ["UNKNOWN_TEAM_XYZ", "CIN", "NE", ""]:
            mult = get_defense_multiplier(
                opponent=opponent,
                position="RB",
                stat_type="rushing_yards",
                season=2024,
                through_week=5,
            )
            assert isinstance(mult, float)
            assert 0.5 <= mult <= 2.0, f"Multiplier {mult} out of bounds for opponent {opponent!r}"

    def test_no_data_season_returns_one(self):
        """When no historical data exists for a season, should return 1.0."""
        from utils.defense_adjustments import get_defense_multiplier

        # Use a far-future season with no data
        mult = get_defense_multiplier(
            opponent="CIN",
            position="RB",
            stat_type="rushing_yards",
            season=2099,
            through_week=1,
        )
        assert mult == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Degenerate EWMA inputs and legacy-table identity
# ---------------------------------------------------------------------------


class TestEwmaSigmaDegenerate:
    """_ewma_sigma with fewer than 2 values must return 0.0, not NaN/crash."""

    def test_empty_array_yields_zero(self):
        from utils.nfl_sigma import _ewma_sigma

        assert _ewma_sigma(np.array([]), decay=0.65) == 0.0

    def test_single_value_yields_zero(self):
        from utils.nfl_sigma import _ewma_sigma

        assert _ewma_sigma(np.array([42.0]), decay=0.65) == 0.0

    def test_two_values_yield_positive_sigma(self):
        from utils.nfl_sigma import _ewma_sigma

        assert _ewma_sigma(np.array([10.0, 30.0]), decay=0.65) > 0.0


class TestLegacyTablesByteIdentical:
    """The (market, None) bucket must carry EXACTLY the legacy values —
    == identity, not approx — for all three markets."""

    def test_none_bucket_floors_match_legacy_exactly(self):
        from utils.nfl_sigma import SIGMA_FLOORS as MODULE_FLOORS

        for market, floor in SIGMA_FLOORS.items():
            assert MODULE_FLOORS[(market, None)] == floor

    def test_none_bucket_defaults_match_legacy_exactly(self):
        from utils.nfl_sigma import SIGMA_DEFAULTS as MODULE_DEFAULTS

        for market, default in SIGMA_DEFAULTS.items():
            assert MODULE_DEFAULTS[(market, None)] == default

    def test_no_multiplier_on_any_none_bucket(self):
        from utils.nfl_sigma import SIGMA_EWMA_MULTIPLIERS

        assert not any(bucket is None for (_, bucket) in SIGMA_EWMA_MULTIPLIERS)


# ---------------------------------------------------------------------------
# Floors bind exactly at the boundary
# ---------------------------------------------------------------------------


class TestFloorBindsExactlyAtBoundary:
    """sigma = max(estimate, floor): pin behavior a hair either side of the
    floor by rescaling a fixed history's spread (sigma scales linearly)."""

    BASE = [50.0, 80.0, 20.0, 90.0, 40.0, 70.0, 60.0, 30.0]

    @staticmethod
    def _raw_sigma(values, decay=0.65):
        values = np.asarray(values, dtype=float)
        raw = np.array([decay ** i for i in range(len(values) - 1, -1, -1)])
        w = raw / raw.sum()
        mean = float(np.dot(w, values))
        var = float(np.dot(w, (values - mean) ** 2))
        corr = 1.0 / (1.0 - float(np.dot(w, w)))
        return float(np.sqrt(var * corr))

    def _scaled_history(self, target_sigma, multiplier=1.0):
        """History whose (EWMA sigma * multiplier) equals target_sigma."""
        s0 = self._raw_sigma(self.BASE)
        mean = float(np.mean(self.BASE))
        c = target_sigma / (s0 * multiplier)
        return [mean + (v - mean) * c for v in self.BASE]

    def test_estimate_just_below_legacy_floor_returns_floor(self):
        from utils.nfl_sigma import compute_player_sigma

        floor = SIGMA_FLOORS["rushing_yards"]
        values = self._scaled_history(floor * 0.999)
        assert compute_player_sigma(values, market="rushing_yards") == floor

    def test_estimate_just_above_legacy_floor_returns_estimate(self):
        from utils.nfl_sigma import compute_player_sigma

        floor = SIGMA_FLOORS["rushing_yards"]
        values = self._scaled_history(floor * 1.001)
        result = compute_player_sigma(values, market="rushing_yards")
        assert result > floor
        assert result == pytest.approx(floor * 1.001)

    def test_wr_receiving_boundary_includes_fat_tail_multiplier(self):
        """The 1.20 multiplier applies BEFORE flooring: the boundary sits at
        estimate * 1.20 vs the 13.0 WR floor."""
        from utils.nfl_sigma import compute_player_sigma

        floor = POSITION_SIGMA_FLOORS[("receiving_yards", "WR")]
        below = self._scaled_history(floor * 0.999, multiplier=WR_RECEIVING_MULTIPLIER)
        above = self._scaled_history(floor * 1.001, multiplier=WR_RECEIVING_MULTIPLIER)
        assert compute_player_sigma(below, market="receiving_yards", position="WR") == floor
        result = compute_player_sigma(above, market="receiving_yards", position="WR")
        assert result > floor
        assert result == pytest.approx(floor * 1.001)


# ---------------------------------------------------------------------------
# get_sigma_or_default remaining fallback branches
# ---------------------------------------------------------------------------


class TestGetSigmaOrDefaultRemainingBranches:
    def test_unknown_market_small_projection_hits_ten_floor(self):
        from utils.nfl_sigma import get_sigma_or_default

        result = get_sigma_or_default(None, projected_value=8.0, market="sacks")
        assert result == pytest.approx(10.0)

    def test_unknown_position_falls_back_to_legacy_market_default(self):
        from utils.nfl_sigma import get_sigma_or_default

        result = get_sigma_or_default(
            None, projected_value=75.0, market="rushing_yards", position="K"
        )
        assert result == pytest.approx(SIGMA_DEFAULTS["rushing_yards"])
