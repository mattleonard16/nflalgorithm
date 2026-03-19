"""Tests for NCAAB bracket prediction math and simulation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.ncaab_ratings import (
    composite_rating,
    compute_trapezoid_score,
    confidence_tier,
    log5,
    pyth_win_pct,
    sigmoid,
)

from schema_migrations import MigrationManager
from utils.db import execute, executemany, read_dataframe


class TestLog5:
    def test_symmetry(self):
        """P(A beats B) + P(B beats A) = 1.0 for any inputs."""
        r_a, r_b = 0.75, 0.60
        assert abs(log5(r_a, r_b) + log5(r_b, r_a) - 1.0) < 1e-10

    def test_dominant_team(self):
        """When r_a >> r_b, P(A) approaches 1.0."""
        assert log5(0.95, 0.10) > 0.98

    def test_equal_teams(self):
        """Equal ratings yield 50/50."""
        assert abs(log5(0.60, 0.60) - 0.5) < 1e-10

    def test_clips_extreme_inputs(self):
        """Edge values are safely clipped."""
        assert 0.0 < log5(0.001, 0.999) < 1.0
        assert 0.0 < log5(0.999, 0.001) < 1.0


class TestCompositeRating:
    def test_output_range(self):
        """Composite rating always in [0.01, 0.99]."""
        r = composite_rating(
            adj_em=38.0, adj_oe=128.0, adj_de=89.0,
            pyth_win=0.99, sos_adj_em=14.0, luck=0.05,
            seed=1, trapezoid_score=4,
        )
        assert 0.01 <= r <= 0.99

        r2 = composite_rating(
            adj_em=-2.0, adj_oe=105.0, adj_de=110.0,
            pyth_win=0.40, sos_adj_em=-5.0, luck=-0.05,
            seed=16, trapezoid_score=0,
        )
        assert 0.01 <= r2 <= 0.99

    def test_elite_beats_weak(self):
        """A 1-seed with elite metrics rates higher than a 16-seed."""
        elite = composite_rating(
            adj_em=35.0, adj_oe=126.0, adj_de=91.0,
            pyth_win=0.97, sos_adj_em=12.0, luck=0.02,
            seed=1, trapezoid_score=4,
        )
        weak = composite_rating(
            adj_em=-2.0, adj_oe=106.0, adj_de=110.0,
            pyth_win=0.45, sos_adj_em=-10.0, luck=0.0,
            seed=16, trapezoid_score=0,
        )
        assert elite > weak


class TestTrapezoid:
    def test_perfect_score(self):
        assert compute_trapezoid_score(adj_oe_rank=5, adj_de_rank=2, sos_rank=10, adj_em_rank=1) == 4

    def test_zero_score(self):
        assert compute_trapezoid_score(adj_oe_rank=100, adj_de_rank=200, sos_rank=150, adj_em_rank=50) == 0

    def test_partial_score(self):
        assert compute_trapezoid_score(adj_oe_rank=10, adj_de_rank=100, sos_rank=20, adj_em_rank=5) == 3


class TestConfidenceTier:
    def test_lock(self):
        assert confidence_tier(0.85, is_upset=False) == "lock"

    def test_likely(self):
        assert confidence_tier(0.70, is_upset=False) == "likely"

    def test_toss_up(self):
        assert confidence_tier(0.55, is_upset=False) == "toss-up"

    def test_upset_pick(self):
        assert confidence_tier(0.52, is_upset=True) == "upset_pick"

    def test_upset_but_low_confidence(self):
        """Upsets below 40% are just toss-ups."""
        assert confidence_tier(0.35, is_upset=True) == "toss-up"


class TestPythWin:
    def test_duke_2026(self):
        """Duke's KenPom numbers: AdjOE=128.0, AdjDE=89.1 -> very high pyth."""
        pw = pyth_win_pct(128.0, 89.1)
        assert pw > 0.95

    def test_average_team(self):
        """AdjOE ~ AdjDE -> ~50%."""
        pw = pyth_win_pct(105.0, 105.0)
        assert abs(pw - 0.5) < 0.01


# ============================================================
# Integration Tests
# ============================================================


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_ncaab.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    import config as cfg
    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")

    MigrationManager(db_path).run()
    return db_path


def _seed_test_ratings(season: int = 2026) -> None:
    """Insert minimal test team ratings for 4 teams."""
    teams = [
        ("Duke", 1, 38.88, 128.0, 4, 89.1, 2, 0.049, 14.29, 15),
        ("Siena", 16, -2.14, 107.1, 211, 109.3, 175, 0.005, -9.50, 347),
        ("UConn", 2, 27.84, 122.0, 30, 94.2, 11, 0.055, 12.01, 37),
        ("Furman", 15, -1.97, 107.5, 202, 109.4, 181, 0.010, -6.27, 290),
    ]
    rows = []
    for name, seed, em, oe, oe_r, de, de_r, luck, sos, sos_r in teams:
        pw = pyth_win_pct(oe, de)
        trap = 4 if em > 25 else (2 if em > 10 else 0)
        cr = composite_rating(em, oe, de, pw, sos, luck, seed, trap)
        rows.append((
            name, season, seed, "East", "TEST", 25, 5,
            em, oe, oe_r, de, de_r, 66.0, luck,
            sos, 115.0, 104.0, 5.0, pw, trap, cr, oe_r, "2026-03-18",
        ))
    executemany(
        """INSERT OR REPLACE INTO ncaab_team_ratings
        (team_name,season,seed,region,conf,wins,losses,
         adj_em,adj_oe,adj_oe_rank,adj_de,adj_de_rank,adj_t,luck,
         sos_adj_em,sos_oe,sos_de,ncsos_adj_em,pyth_win,trapezoid_score,
         composite_rating,kenpom_rank,scraped_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )


def _seed_mini_bracket(season: int = 2026) -> None:
    """Insert a 3-game mini bracket: 2 R1 games feeding 1 R2 game."""
    games = [
        ("E_R1_1", "East", 1, 1, "Duke", 1, "Siena", 16, None, None, "E_R2_1", season),
        ("E_R1_8", "East", 1, 8, "UConn", 2, "Furman", 15, None, None, "E_R2_1", season),
        ("E_R2_1", "East", 2, 1, None, None, None, None, "E_R1_1", "E_R1_8", None, season),
    ]
    executemany(
        """INSERT OR REPLACE INTO ncaab_bracket
        (game_id,region,round,slot,team_a,seed_a,team_b,seed_b,
         prev_game_a,prev_game_b,feeds_game_id,season)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        games,
    )


class TestSimulation:
    def test_simulate_mini_bracket(self, db):
        """Simulate a 3-game mini bracket and verify propagation."""
        _seed_test_ratings()
        _seed_mini_bracket()

        from models.ncaab.bracket_predictor import load_bracket, load_ratings, simulate_bracket
        ratings = load_ratings(2026)
        bracket = load_bracket(2026)
        preds = simulate_bracket(ratings, bracket)

        assert len(preds) == 3
        r1_preds = [p for p in preds if p["round"] == 1]
        r2_preds = [p for p in preds if p["round"] == 2]
        assert len(r1_preds) == 2
        assert len(r2_preds) == 1

        # 1-seed Duke should beat 16-seed Siena
        duke_game = [p for p in r1_preds if p["team_a"] == "Duke"][0]
        assert duke_game["predicted_winner"] == "Duke"
        assert duke_game["is_upset"] == 0
        assert duke_game["p_a_wins"] > 0.80

    def test_upset_detection(self, db):
        """Lower seed winning yields is_upset=1."""
        _seed_test_ratings()
        _seed_mini_bracket()

        from models.ncaab.bracket_predictor import load_bracket, load_ratings, simulate_bracket
        ratings = load_ratings(2026)
        bracket = load_bracket(2026)
        preds = simulate_bracket(ratings, bracket)

        for p in preds:
            if p["winner_seed"] > p["loser_seed"]:
                assert p["is_upset"] == 1
            else:
                assert p["is_upset"] == 0

    def test_predictions_idempotent(self, db):
        """Running simulation twice gives same results."""
        _seed_test_ratings()
        _seed_mini_bracket()

        from models.ncaab.bracket_predictor import load_bracket, load_ratings, simulate_bracket
        ratings = load_ratings(2026)
        bracket = load_bracket(2026)

        preds1 = simulate_bracket(ratings, bracket)
        preds2 = simulate_bracket(ratings, bracket)

        for p1, p2 in zip(preds1, preds2):
            assert p1["predicted_winner"] == p2["predicted_winner"]
            assert abs(p1["p_a_wins"] - p2["p_a_wins"]) < 1e-10


class TestModifierIntegration:
    """Integration tests: modifiers change bracket outcomes."""

    def test_enhanced_rating_used_when_present(self, db):
        """When enhanced_rating is populated, predictor uses it over composite_rating."""
        _seed_test_ratings()
        # Set an enhanced_rating that differs from composite — Siena artificially boosted
        execute(
            "UPDATE ncaab_team_ratings SET enhanced_rating = 0.35 WHERE team_name = 'Duke' AND season = 2026"
        )
        execute(
            "UPDATE ncaab_team_ratings SET enhanced_rating = 0.80 WHERE team_name = 'Siena' AND season = 2026"
        )
        _seed_mini_bracket()

        from models.ncaab.bracket_predictor import load_bracket, load_ratings, simulate_bracket
        ratings = load_ratings(2026)
        bracket = load_bracket(2026)
        preds = simulate_bracket(ratings, bracket)

        # Siena should now win with the artificially boosted enhanced_rating
        duke_game = [p for p in preds if p["game_id"] == "E_R1_1"][0]
        assert duke_game["predicted_winner"] == "Siena"
        assert duke_game["is_upset"] == 1

    def test_fallback_to_composite_when_no_enhanced(self, db):
        """When enhanced_rating is NULL, falls back to composite_rating."""
        _seed_test_ratings()
        _seed_mini_bracket()

        from models.ncaab.bracket_predictor import load_bracket, load_ratings, simulate_bracket
        ratings = load_ratings(2026)
        bracket = load_bracket(2026)
        preds = simulate_bracket(ratings, bracket)

        # Should work same as before — Duke beats Siena
        duke_game = [p for p in preds if p["game_id"] == "E_R1_1"][0]
        assert duke_game["predicted_winner"] == "Duke"

    def test_predictions_still_deterministic(self, db):
        """Enhanced predictions are still deterministic."""
        _seed_test_ratings()
        execute(
            "UPDATE ncaab_team_ratings SET enhanced_rating = composite_rating * 1.02, "
            "bt_factor = 1.02, coaching_factor = 1.0, experience_factor = 1.0, momentum_factor = 1.0 "
            "WHERE season = 2026"
        )
        _seed_mini_bracket()

        from models.ncaab.bracket_predictor import load_bracket, load_ratings, simulate_bracket
        ratings = load_ratings(2026)
        bracket = load_bracket(2026)
        p1 = simulate_bracket(ratings, bracket)
        p2 = simulate_bracket(ratings, bracket)

        for a, b in zip(p1, p2):
            assert a["predicted_winner"] == b["predicted_winner"]

    def test_transparency_fields_present(self, db):
        """Predictions include modifier transparency fields."""
        _seed_test_ratings()
        _seed_mini_bracket()

        from models.ncaab.bracket_predictor import load_bracket, load_ratings, simulate_bracket
        ratings = load_ratings(2026)
        bracket = load_bracket(2026)
        preds = simulate_bracket(ratings, bracket)

        for p in preds:
            assert "p_raw_log5" in p
            assert "seed_historical_p" in p
            assert "tempo_factor" in p
            assert "final_p_a" in p
            assert "enhanced_rating_a" in p
            assert "enhanced_rating_b" in p
            assert "modifier_json_a" in p
            assert "modifier_json_b" in p


class TestLuckRegressionWeight:
    """Signal 7: verify increased anti-luck weight impact."""

    def test_lucky_team_penalized(self):
        """Teams with high luck scores get lower composite ratings."""
        lucky = composite_rating(20.0, 118.0, 95.0, 0.85, 8.0, 0.15, 4, 2)
        unlucky = composite_rating(20.0, 118.0, 95.0, 0.85, 8.0, -0.05, 4, 2)
        assert unlucky > lucky
