"""Tests for nba_value_engine.py.

Covers:
- Math helper functions (implied_probability, american_to_decimal, prob_over,
  kelly_fraction, expected_roi)
- Sigma estimation logic
- rank_nba_value integration tests
- materialize_nba_value persistence and idempotency
- Edge cases (empty tables, no matching player_ids, high min_edge threshold)
"""

from __future__ import annotations

import math

import pytest

from schema_migrations import MigrationManager
from utils.db import executemany, read_dataframe


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_nba_value.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    import config as cfg

    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")

    MigrationManager(db_path).run()
    return db_path


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

GAME_DATE = "2026-02-17"
SEASON = 2025
PLAYER_ID = 1628369
PLAYER_NAME = "Jayson Tatum"
TEAM = "BOS"


def _seed_projections(
    projected_value: float = 28.5,
    market: str = "pts",
    player_id: int = PLAYER_ID,
    player_name: str = PLAYER_NAME,
) -> None:
    executemany(
        "INSERT INTO nba_projections "
        "(player_id, player_name, team, season, game_date, game_id, market, projected_value, confidence) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (
                player_id,
                player_name,
                TEAM,
                SEASON,
                GAME_DATE,
                "0022500001",
                market,
                projected_value,
                0.85,
            )
        ],
    )


def _seed_odds(
    line: float = 25.5,
    over_price: int = -115,
    under_price: int = -105,
    market: str = "pts",
    player_id: int | None = PLAYER_ID,
    player_name: str = PLAYER_NAME,
    event_id: str = "evt001",
) -> None:
    executemany(
        "INSERT INTO nba_odds "
        "(event_id, season, game_date, player_id, player_name, team, market, sportsbook, line, over_price, under_price, as_of) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                event_id,
                SEASON,
                GAME_DATE,
                player_id,
                player_name,
                TEAM,
                market,
                "FanDuel",
                line,
                over_price,
                under_price,
                "2026-02-17T10:00:00",
            )
        ],
    )


# ---------------------------------------------------------------------------
# 1. Math helper tests
# ---------------------------------------------------------------------------


class TestImpliedProbability:
    def test_negative_american_odds(self):
        from nba_value_engine import implied_probability

        result = implied_probability(-110)
        assert abs(result - 0.5238) < 0.001

    def test_positive_american_odds(self):
        from nba_value_engine import implied_probability

        result = implied_probability(150)
        assert abs(result - 0.4) < 0.001

    def test_pick_em_even_money(self):
        from nba_value_engine import implied_probability

        result = implied_probability(100)
        assert abs(result - 0.5) < 0.001


class TestAmericanToDecimal:
    def test_negative_two_hundred(self):
        from nba_value_engine import american_to_decimal

        assert american_to_decimal(-200) == pytest.approx(1.5)

    def test_positive_two_hundred(self):
        from nba_value_engine import american_to_decimal

        assert american_to_decimal(200) == pytest.approx(3.0)

    def test_negative_one_ten(self):
        from nba_value_engine import american_to_decimal

        result = american_to_decimal(-110)
        assert abs(result - 1.909) < 0.001

    def test_positive_one_ten(self):
        from nba_value_engine import american_to_decimal

        result = american_to_decimal(110)
        assert abs(result - 2.1) < 0.001


class TestProbOver:
    def test_projection_above_line_gives_gt_half(self):
        from nba_value_engine import prob_over

        result = prob_over(mu=25.0, sigma=5.0, line=20.0)
        assert result > 0.5

    def test_projection_below_line_gives_lt_half(self):
        from nba_value_engine import prob_over

        result = prob_over(mu=25.0, sigma=5.0, line=30.0)
        assert result < 0.5

    def test_projection_at_line_gives_approx_half(self):
        from nba_value_engine import prob_over

        result = prob_over(mu=25.0, sigma=5.0, line=25.0)
        assert abs(result - 0.5) < 0.01

    def test_zero_sigma_projection_above_line(self):
        from nba_value_engine import prob_over

        result = prob_over(mu=30.0, sigma=0.0, line=25.0)
        assert result == 1.0

    def test_zero_sigma_projection_at_or_below_line(self):
        from nba_value_engine import prob_over

        result = prob_over(mu=25.0, sigma=0.0, line=25.0)
        assert result == 0.0


class TestKellyFraction:
    def test_positive_edge_returns_gt_zero(self):
        from nba_value_engine import kelly_fraction

        result = kelly_fraction(0.6, -110)
        assert result > 0.0

    def test_negative_edge_returns_zero(self):
        from nba_value_engine import kelly_fraction

        result = kelly_fraction(0.4, -110)
        assert result == 0.0

    def test_kelly_capped_at_ten_percent(self):
        from nba_value_engine import kelly_fraction

        # Extremely favorable odds + high win_prob would exceed 10% uncapped
        result = kelly_fraction(0.99, 1000)
        assert result <= 0.10

    def test_breakeven_probability_returns_zero(self):
        from nba_value_engine import kelly_fraction

        # For -110, breakeven implied_prob = 110/210 ≈ 0.5238
        # Passing exactly that prob should yield 0 (no edge)
        implied = 110 / 210
        result = kelly_fraction(implied, -110)
        assert result == 0.0

    def test_fraction_parameter_scales_kelly(self):
        from nba_value_engine import kelly_fraction

        full = kelly_fraction(0.6, -110, fraction=1.0)
        quarter = kelly_fraction(0.6, -110, fraction=0.25)
        # full fraction result should be larger (assuming neither hits the cap)
        assert full >= quarter


class TestExpectedRoi:
    def test_profitable_bet_returns_positive(self):
        from nba_value_engine import expected_roi

        result = expected_roi(0.6, -110)
        assert result > 0.0

    def test_losing_bet_returns_negative(self):
        from nba_value_engine import expected_roi

        result = expected_roi(0.4, -110)
        assert result < 0.0

    def test_positive_american_odds(self):
        from nba_value_engine import expected_roi

        # win_prob 0.6 with +200 payout: 0.6*2.0 - 0.4 = 0.8
        result = expected_roi(0.6, 200)
        assert abs(result - 0.8) < 0.001


# ---------------------------------------------------------------------------
# 2. Sigma estimation tests
# ---------------------------------------------------------------------------


class TestSigmaEstimation:
    """Verify sigma = max(projected_value * 0.20, 3.0) inside rank_nba_value."""

    def test_sigma_floor_applied(self, db):
        """When projection is small, sigma must be at least 3.0."""
        # projected_value=10.0 → 10*0.20=2.0 < 3.0 → floor kicks in
        _seed_projections(projected_value=10.0)
        _seed_odds(line=5.0, over_price=-200)

        from nba_value_engine import rank_nba_value

        results = rank_nba_value(GAME_DATE, SEASON, min_edge=0.0)
        assert len(results) >= 1
        assert results[0]["sigma"] >= 3.0

    def test_sigma_scales_with_projection(self, db):
        """When sigma is NULL, falls back to SIGMA_DEFAULTS[market]."""
        # With no sigma in projections, get_sigma_or_default returns SIGMA_DEFAULTS["pts"] = 5.5
        _seed_projections(projected_value=30.0)
        _seed_odds(line=20.0, over_price=-200)

        from nba_value_engine import rank_nba_value

        results = rank_nba_value(GAME_DATE, SEASON, min_edge=0.0)
        assert len(results) >= 1
        assert results[0]["sigma"] == pytest.approx(5.5, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. rank_nba_value integration tests
# ---------------------------------------------------------------------------


class TestRankNbaValue:
    def test_returns_non_empty_when_edge_exists(self, db):
        """Projection 28.5 vs line 25.5 should produce a positive edge."""
        _seed_projections(projected_value=28.5)
        _seed_odds(line=25.5, over_price=-115)

        from nba_value_engine import rank_nba_value

        results = rank_nba_value(GAME_DATE, SEASON, min_edge=0.05)
        assert len(results) > 0

    def test_required_keys_present(self, db):
        """Every returned row must contain all required keys."""
        _seed_projections(projected_value=28.5)
        _seed_odds(line=25.5, over_price=-115)

        from nba_value_engine import rank_nba_value

        results = rank_nba_value(GAME_DATE, SEASON, min_edge=0.0)
        assert len(results) > 0

        required_keys = {
            "player_id",
            "player_name",
            "market",
            "line",
            "over_price",
            "mu",
            "sigma",
            "p_win",
            "edge_percentage",
            "expected_roi",
            "kelly_fraction",
        }
        for row in results:
            missing = required_keys - set(row.keys())
            assert not missing, f"Row is missing keys: {missing}"

    def test_edge_percentage_positive_for_all_returned_rows(self, db):
        """All returned rows must have edge_percentage > 0."""
        _seed_projections(projected_value=28.5)
        _seed_odds(line=25.5, over_price=-115)

        from nba_value_engine import rank_nba_value

        results = rank_nba_value(GAME_DATE, SEASON, min_edge=0.0)
        for row in results:
            assert row["edge_percentage"] > 0.0

    def test_results_sorted_by_edge_descending(self, db):
        """Results must be sorted by edge_percentage in descending order."""
        # Seed two odds rows with different lines to create different edges
        _seed_projections(projected_value=30.0, player_id=PLAYER_ID)
        # Second player with smaller edge
        _seed_projections(
            projected_value=20.0,
            player_id=1628384,
            player_name="Jaylen Brown",
        )
        _seed_odds(line=22.0, over_price=-115, event_id="evt001", player_id=PLAYER_ID)
        _seed_odds(
            line=19.0,
            over_price=-115,
            event_id="evt002",
            player_id=1628384,
            player_name="Jaylen Brown",
        )

        from nba_value_engine import rank_nba_value

        results = rank_nba_value(GAME_DATE, SEASON, min_edge=0.0)
        edges = [r["edge_percentage"] for r in results]
        assert edges == sorted(edges, reverse=True)

    def test_mu_matches_projected_value(self, db):
        """The mu field should equal the projected_value from nba_projections."""
        _seed_projections(projected_value=28.5)
        _seed_odds(line=25.5, over_price=-115)

        from nba_value_engine import rank_nba_value

        results = rank_nba_value(GAME_DATE, SEASON, min_edge=0.0)
        assert len(results) >= 1
        assert results[0]["mu"] == pytest.approx(28.5)

    def test_correct_player_name_in_result(self, db):
        _seed_projections(projected_value=28.5)
        _seed_odds(line=25.5, over_price=-115)

        from nba_value_engine import rank_nba_value

        results = rank_nba_value(GAME_DATE, SEASON, min_edge=0.0)
        assert len(results) >= 1
        assert results[0]["player_name"] == PLAYER_NAME

    def test_season_field_matches_argument(self, db):
        _seed_projections(projected_value=28.5)
        _seed_odds(line=25.5, over_price=-115)

        from nba_value_engine import rank_nba_value

        results = rank_nba_value(GAME_DATE, SEASON, min_edge=0.0)
        assert len(results) >= 1
        assert results[0]["season"] == SEASON


# ---------------------------------------------------------------------------
# 4. materialize_nba_value tests
# ---------------------------------------------------------------------------


class TestMaterializeNbaValue:
    def test_writes_rows_to_materialized_table(self, db):
        """materialize_nba_value must persist rows to nba_materialized_value_view."""
        _seed_projections(projected_value=28.5)
        _seed_odds(line=25.5, over_price=-115)

        from nba_value_engine import materialize_nba_value

        count = materialize_nba_value(GAME_DATE, SEASON, min_edge=0.05)
        assert count > 0

        rows = read_dataframe(
            "SELECT * FROM nba_materialized_value_view WHERE game_date = ?",
            (GAME_DATE,),
        )
        assert len(rows) == count

    def test_returns_correct_count(self, db):
        _seed_projections(projected_value=28.5)
        _seed_odds(line=25.5, over_price=-115)

        from nba_value_engine import materialize_nba_value

        count = materialize_nba_value(GAME_DATE, SEASON, min_edge=0.0)
        assert count >= 1

    def test_idempotent_no_duplicate_rows(self, db):
        """Calling materialize_nba_value twice must not duplicate rows."""
        _seed_projections(projected_value=28.5)
        _seed_odds(line=25.5, over_price=-115)

        from nba_value_engine import materialize_nba_value

        count_first = materialize_nba_value(GAME_DATE, SEASON, min_edge=0.0)
        count_second = materialize_nba_value(GAME_DATE, SEASON, min_edge=0.0)

        # Both calls should report the same count
        assert count_first == count_second

        rows = read_dataframe(
            "SELECT * FROM nba_materialized_value_view WHERE game_date = ?",
            (GAME_DATE,),
        )
        # The table should not have more rows than a single run produced
        assert len(rows) == count_first

    def test_edge_percentage_stored_correctly(self, db):
        _seed_projections(projected_value=28.5)
        _seed_odds(line=25.5, over_price=-115)

        from nba_value_engine import materialize_nba_value

        materialize_nba_value(GAME_DATE, SEASON, min_edge=0.0)

        rows = read_dataframe(
            "SELECT edge_percentage FROM nba_materialized_value_view WHERE game_date = ?",
            (GAME_DATE,),
        )
        assert len(rows) >= 1
        assert (rows["edge_percentage"] > 0).all()


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_projections_returns_empty_list(self, db):
        """No projections seeded → rank_nba_value must return []."""
        _seed_odds(line=25.5, over_price=-115)

        from nba_value_engine import rank_nba_value

        results = rank_nba_value(GAME_DATE, SEASON)
        assert results == []

    def test_empty_odds_returns_empty_list(self, db):
        """No odds seeded → rank_nba_value must return []."""
        _seed_projections(projected_value=28.5)

        from nba_value_engine import rank_nba_value

        results = rank_nba_value(GAME_DATE, SEASON)
        assert results == []

    def test_no_matching_player_ids_returns_empty(self, db):
        """Non-overlapping player_ids → no join rows → empty result."""
        _seed_projections(projected_value=28.5, player_id=PLAYER_ID)
        # Different player_id in odds — no id match, no name match (different name)
        _seed_odds(
            line=25.5,
            over_price=-115,
            player_id=9999999,
            player_name="Unknown Player",
        )

        from nba_value_engine import rank_nba_value

        results = rank_nba_value(GAME_DATE, SEASON)
        assert results == []

    def test_min_edge_too_high_returns_empty(self, db):
        """When min_edge threshold exceeds computed edge → returns []."""
        _seed_projections(projected_value=28.5)
        _seed_odds(line=25.5, over_price=-115)

        from nba_value_engine import rank_nba_value

        # min_edge=1.0 (100%) is impossible — no bet can have that edge
        results = rank_nba_value(GAME_DATE, SEASON, min_edge=1.0)
        assert results == []

    def test_materialize_returns_zero_when_no_edge(self, db):
        """materialize_nba_value returns 0 and writes nothing when no rows pass."""
        _seed_projections(projected_value=28.5)
        _seed_odds(line=25.5, over_price=-115)

        from nba_value_engine import materialize_nba_value

        count = materialize_nba_value(GAME_DATE, SEASON, min_edge=1.0)
        assert count == 0

        rows = read_dataframe(
            "SELECT * FROM nba_materialized_value_view WHERE game_date = ?",
            (GAME_DATE,),
        )
        assert len(rows) == 0

    def test_different_date_not_returned(self, db):
        """Data seeded for a different date must not appear in results."""
        # Seed data for tomorrow
        other_date = "2026-02-18"
        executemany(
            "INSERT INTO nba_projections "
            "(player_id, player_name, team, season, game_date, game_id, market, projected_value, confidence) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            [(PLAYER_ID, PLAYER_NAME, TEAM, SEASON, other_date, "0022500002", "pts", 28.5, 0.85)],
        )
        executemany(
            "INSERT INTO nba_odds "
            "(event_id, season, game_date, player_id, player_name, team, market, sportsbook, line, over_price, under_price, as_of) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [("evt999", SEASON, other_date, PLAYER_ID, PLAYER_NAME, TEAM, "pts", "FanDuel", 25.5, -115, -105, "2026-02-18T10:00:00")],
        )

        from nba_value_engine import rank_nba_value

        # Query for today's date — nothing seeded for today
        results = rank_nba_value(GAME_DATE, SEASON, min_edge=0.0)
        assert results == []

    def test_name_fallback_join_when_odds_has_no_player_id(self, db):
        """Odds row with player_id=NULL should match via normalised name."""
        _seed_projections(projected_value=28.5)
        # Seed odds with NULL player_id but matching name
        _seed_odds(line=25.5, over_price=-115, player_id=None)

        from nba_value_engine import rank_nba_value

        results = rank_nba_value(GAME_DATE, SEASON, min_edge=0.0)
        # Name-based fallback should find the match
        assert len(results) >= 1
        assert results[0]["player_name"] == PLAYER_NAME


# ---------------------------------------------------------------------------
# 6. prob_under tests (Phase 4)
# ---------------------------------------------------------------------------


class TestProbUnder:
    def test_prob_under_complement_of_prob_over(self):
        """prob_over + prob_under should equal ~1.0 for any valid inputs."""
        from nba_value_engine import prob_over, prob_under

        mu, sigma, line = 25.0, 5.0, 22.5
        total = prob_over(mu, sigma, line) + prob_under(mu, sigma, line)
        assert abs(total - 1.0) < 1e-6

    def test_prob_under_high_when_line_above_mu(self):
        """When line is well above mu, prob_under should be high."""
        from nba_value_engine import prob_under

        p = prob_under(mu=18.0, sigma=4.0, line=25.0)
        assert p > 0.9

    def test_prob_under_low_when_line_below_mu(self):
        """When line is well below mu, prob_under should be low."""
        from nba_value_engine import prob_under

        p = prob_under(mu=30.0, sigma=4.0, line=20.0)
        assert p < 0.01

    def test_prob_under_zero_sigma_line_at_mu(self):
        """Zero sigma: line <= mu → prob_under = 0.0."""
        from nba_value_engine import prob_under

        assert prob_under(25.0, 0.0, 20.0) == 0.0

    def test_prob_under_zero_sigma_line_above_mu(self):
        """Zero sigma: line > mu → prob_under = 1.0."""
        from nba_value_engine import prob_under

        assert prob_under(25.0, 0.0, 30.0) == 1.0

    def test_prob_under_midpoint_approx_half(self):
        """At line == mu, prob_under should be approximately 0.5."""
        from nba_value_engine import prob_under

        p = prob_under(mu=25.0, sigma=5.0, line=25.0)
        assert abs(p - 0.5) < 0.01


# ---------------------------------------------------------------------------
# 7. Under-bet evaluation in rank_nba_value (Phase 4)
# ---------------------------------------------------------------------------


class TestUnderBets:
    """Test under-bet evaluation in rank_nba_value."""

    def test_under_bet_returned_when_line_above_mu(self, db):
        """When projection is well below line, an under bet should appear."""
        executemany(
            "INSERT INTO nba_projections "
            "(player_id, player_name, team, season, game_date, game_id, market, projected_value, confidence, sigma) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(1, "Test Player", "BOS", 2025, "2026-02-20", "G1", "pts", 18.0, 0.8, 4.0)],
        )
        executemany(
            "INSERT INTO nba_odds "
            "(event_id, player_id, player_name, market, sportsbook, line, over_price, under_price, game_date, season, as_of) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [("E1", 1, "Test Player", "pts", "DraftKings", 25.5, -110, -110, "2026-02-20", 2025, "2026-02-20T10:00:00")],
        )

        from nba_value_engine import rank_nba_value

        results = rank_nba_value("2026-02-20", season=2025, min_edge=0.05)
        assert len(results) > 0
        assert results[0]["side"] == "under"

    def test_over_bet_returned_when_projection_above_line(self, db):
        """When projection is well above line, an over bet should appear."""
        executemany(
            "INSERT INTO nba_projections "
            "(player_id, player_name, team, season, game_date, game_id, market, projected_value, confidence, sigma) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(2, "Over Player", "LAL", 2025, "2026-02-20", "G2", "pts", 32.0, 0.85, 4.0)],
        )
        executemany(
            "INSERT INTO nba_odds "
            "(event_id, player_id, player_name, market, sportsbook, line, over_price, under_price, game_date, season, as_of) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [("E2", 2, "Over Player", "pts", "DraftKings", 24.5, -110, -110, "2026-02-20", 2025, "2026-02-20T10:00:00")],
        )

        from nba_value_engine import rank_nba_value

        results = rank_nba_value("2026-02-20", season=2025, min_edge=0.05)
        assert len(results) > 0
        assert results[0]["side"] == "over"

    def test_result_has_side_field(self, db):
        """All results should have a 'side' field that is 'over' or 'under'."""
        executemany(
            "INSERT INTO nba_projections "
            "(player_id, player_name, team, season, game_date, game_id, market, projected_value, confidence, sigma) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(3, "Side Player", "BOS", 2025, "2026-02-20", "G3", "pts", 30.0, 0.8, 4.0)],
        )
        executemany(
            "INSERT INTO nba_odds "
            "(event_id, player_id, player_name, market, sportsbook, line, over_price, under_price, game_date, season, as_of) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [("E3", 3, "Side Player", "pts", "DraftKings", 22.5, -110, -110, "2026-02-20", 2025, "2026-02-20T10:00:00")],
        )

        from nba_value_engine import rank_nba_value

        results = rank_nba_value("2026-02-20", season=2025, min_edge=0.05)
        for r in results:
            assert "side" in r
            assert r["side"] in ("over", "under")

    def test_materialize_persists_side_column(self, db):
        """After materialize, side column should be stored in DB."""
        executemany(
            "INSERT INTO nba_projections "
            "(player_id, player_name, team, season, game_date, game_id, market, projected_value, confidence, sigma) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(4, "Mat Player", "MIA", 2025, "2026-02-20", "G4", "pts", 30.0, 0.8, 4.0)],
        )
        executemany(
            "INSERT INTO nba_odds "
            "(event_id, player_id, player_name, market, sportsbook, line, over_price, under_price, game_date, season, as_of) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [("E4", 4, "Mat Player", "pts", "FanDuel", 24.0, -110, -110, "2026-02-20", 2025, "2026-02-20T10:00:00")],
        )

        from nba_value_engine import materialize_nba_value

        count = materialize_nba_value("2026-02-20", season=2025, min_edge=0.05)
        assert count > 0

        rows = read_dataframe(
            "SELECT side FROM nba_materialized_value_view WHERE game_date = ?",
            ("2026-02-20",),
        )
        assert len(rows) == count
        assert rows["side"].notna().all()
        assert rows["side"].isin(["over", "under"]).all()


# ---------------------------------------------------------------------------
# 8. Confidence input wiring (Phase 3B)
# ---------------------------------------------------------------------------


class TestConfidenceInputWiring:
    """Test that real volatility_score and usage_rate flow into confidence scoring."""

    def test_confidence_uses_real_usage_rate(self, db):
        """When projections have usage_rate, it flows into confidence scoring."""
        executemany(
            "INSERT INTO nba_projections "
            "(player_id, player_name, team, season, game_date, game_id, market, "
            "projected_value, confidence, sigma, usage_rate, volatility_score) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    100,
                    "Usage Player",
                    "BOS",
                    2025,
                    "2026-02-20",
                    "G100",
                    "pts",
                    28.0,
                    0.85,
                    5.0,
                    0.28,
                    65.0,
                )
            ],
        )
        executemany(
            "INSERT INTO nba_odds "
            "(event_id, season, game_date, player_id, player_name, team, market, "
            "sportsbook, line, over_price, under_price, as_of) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    "evt_usage2",
                    2025,
                    "2026-02-20",
                    100,
                    "Usage Player",
                    "BOS",
                    "pts",
                    "FanDuel",
                    22.0,
                    -110,
                    -110,
                    "2026-02-20T10:00:00",
                )
            ],
        )

        from nba_value_engine import rank_nba_value

        results = rank_nba_value("2026-02-20", season=2025, min_edge=0.0)
        assert len(results) >= 1
        assert results[0]["confidence_score"] is not None
        assert isinstance(results[0]["confidence_score"], (int, float))
