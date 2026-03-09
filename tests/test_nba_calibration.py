"""Tests for NBA probability calibration (utils/nba_calibration.py).

Covers:
1. fit() returns sample counts per market
2. Calibrated probabilities always in [0, 1]
3. Identity when insufficient data (< min_samples)
4. Well-calibrated input passes through ~unchanged
5. Save/load roundtrip produces same output
6. reliability_data() returns correct binned structure
7. rank_nba_value(calibrated=True) populates p_win_raw key in results
8. Missing calibration file -> silently ignored (same as calibrated=False)
"""

from __future__ import annotations

import math
import os
import random

import pytest

from schema_migrations import MigrationManager
from utils.db import executemany, read_dataframe


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_calibration.db")
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


def _seed_bet_outcomes(
    db_path: str,
    market: str,
    n_win: int,
    n_loss: int,
    edge: float = 0.10,
) -> None:
    """Insert synthetic nba_bet_outcomes rows for calibration training."""
    rows = []
    for i in range(n_win):
        rows.append((
            f"bet_{market}_win_{i}",
            SEASON,
            GAME_DATE,
            PLAYER_ID,
            PLAYER_NAME,
            market,
            "FanDuel",
            "over",
            25.5,
            -110,
            None,
            "win",
            0.9,
            "A",
            edge,
            f"2026-02-17T10:{i:02d}:00",
        ))
    for i in range(n_loss):
        rows.append((
            f"bet_{market}_loss_{i}",
            SEASON,
            GAME_DATE,
            PLAYER_ID,
            PLAYER_NAME,
            market,
            "FanDuel",
            "over",
            25.5,
            -110,
            None,
            "loss",
            -1.0,
            "A",
            edge,
            f"2026-02-17T11:{i:02d}:00",
        ))

    executemany(
        "INSERT OR IGNORE INTO nba_bet_outcomes "
        "(bet_id, season, game_date, player_id, player_name, market, sportsbook, "
        "side, line, price, actual_result, result, profit_units, confidence_tier, "
        "edge_at_placement, recorded_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )


def _seed_projections_and_odds(game_date: str = GAME_DATE) -> None:
    executemany(
        "INSERT INTO nba_projections "
        "(player_id, player_name, team, season, game_date, game_id, market, projected_value, confidence) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        [(PLAYER_ID, PLAYER_NAME, TEAM, SEASON, game_date, "0022500001", "pts", 28.5, 0.85)],
    )
    executemany(
        "INSERT INTO nba_odds "
        "(event_id, season, game_date, player_id, player_name, team, market, sportsbook, "
        "line, over_price, under_price, as_of) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [("evt001", SEASON, game_date, PLAYER_ID, PLAYER_NAME, TEAM, "pts", "FanDuel",
          25.5, -115, -105, f"{game_date}T10:00:00")],
    )


# ---------------------------------------------------------------------------
# Test 1: fit() returns sample counts per market
# ---------------------------------------------------------------------------


class TestFitReturnsSampleCounts:
    def test_fit_returns_dict_with_market_keys(self, db):
        """fit() should return a dict keyed by market with sample counts."""
        # Seed 60 outcomes for 'pts' (>= default min_samples=50)
        _seed_bet_outcomes(db, "pts", n_win=36, n_loss=24)

        from utils.nba_calibration import NBACalibrator

        cal = NBACalibrator()
        counts = cal.fit(min_samples=50)

        assert isinstance(counts, dict)
        assert "pts" in counts
        assert counts["pts"] == 60

    def test_fit_skips_markets_below_min_samples(self, db):
        """Markets with < min_samples outcomes should not appear in result dict."""
        _seed_bet_outcomes(db, "pts", n_win=36, n_loss=24)
        _seed_bet_outcomes(db, "reb", n_win=15, n_loss=10)  # only 25 total

        from utils.nba_calibration import NBACalibrator

        cal = NBACalibrator()
        counts = cal.fit(min_samples=50)

        assert "pts" in counts
        assert "reb" not in counts

    def test_fit_empty_table_returns_empty_dict(self, db):
        """No outcomes at all should return empty dict."""
        from utils.nba_calibration import NBACalibrator

        cal = NBACalibrator()
        counts = cal.fit()

        assert counts == {}


# ---------------------------------------------------------------------------
# Test 2: Calibrated probabilities always in [0, 1]
# ---------------------------------------------------------------------------


class TestCalibratedProbabilityRange:
    def test_extreme_low_input_clipped_to_zero(self, db):
        """calibrate() with p_raw < 0 must return a value in [0, 1]."""
        _seed_bet_outcomes(db, "pts", n_win=36, n_loss=24)

        from utils.nba_calibration import NBACalibrator

        cal = NBACalibrator()
        cal.fit(min_samples=50)

        result = cal.calibrate(-0.5, "pts")
        assert 0.0 <= result <= 1.0

    def test_extreme_high_input_clipped_to_one(self, db):
        """calibrate() with p_raw > 1 must return a value in [0, 1]."""
        _seed_bet_outcomes(db, "pts", n_win=36, n_loss=24)

        from utils.nba_calibration import NBACalibrator

        cal = NBACalibrator()
        cal.fit(min_samples=50)

        result = cal.calibrate(1.5, "pts")
        assert 0.0 <= result <= 1.0

    def test_normal_range_stays_in_range(self, db):
        """calibrate() with p_raw in [0, 1] always returns value in [0, 1]."""
        _seed_bet_outcomes(db, "pts", n_win=36, n_loss=24)

        from utils.nba_calibration import NBACalibrator

        cal = NBACalibrator()
        cal.fit(min_samples=50)

        for p in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            result = cal.calibrate(p, "pts")
            assert 0.0 <= result <= 1.0, f"calibrate({p}) = {result} out of [0, 1]"


# ---------------------------------------------------------------------------
# Test 3: Identity when insufficient data
# ---------------------------------------------------------------------------


class TestIdentityFallback:
    def test_calibrate_returns_p_raw_when_no_calibrator(self, db):
        """calibrate() should return p_raw unchanged when market has no calibrator."""
        from utils.nba_calibration import NBACalibrator

        cal = NBACalibrator()
        # No fit() called, so no calibrators

        p_raw = 0.65
        result = cal.calibrate(p_raw, "pts")
        assert result == p_raw

    def test_calibrate_returns_p_raw_when_below_min_samples(self, db):
        """When market has < min_samples, calibrate() returns p_raw unchanged."""
        _seed_bet_outcomes(db, "pts", n_win=10, n_loss=5)  # only 15 total

        from utils.nba_calibration import NBACalibrator

        cal = NBACalibrator()
        cal.fit(min_samples=50)  # pts won't have calibrator

        p_raw = 0.70
        result = cal.calibrate(p_raw, "pts")
        assert result == p_raw

    def test_calibrate_identity_for_unknown_market(self, db):
        """calibrate() for an unknown market returns p_raw unchanged."""
        _seed_bet_outcomes(db, "pts", n_win=36, n_loss=24)

        from utils.nba_calibration import NBACalibrator

        cal = NBACalibrator()
        cal.fit(min_samples=50)

        p_raw = 0.55
        result = cal.calibrate(p_raw, "unknown_market")
        assert result == p_raw


# ---------------------------------------------------------------------------
# Test 4: Well-calibrated input passes through ~unchanged
# ---------------------------------------------------------------------------


class TestWellCalibratedInput:
    def test_calibrated_prob_stays_near_true_rate(self, db):
        """When predicted ≈ observed freq, calibration should not distort much.

        Create synthetic data where edge_at_placement ~ 0.10 (p_raw proxy ~ 0.60)
        and 60% of outcomes are wins -> calibrated 0.60 should stay near 0.60.
        """
        # Build 100 rows: 60% win rate at edge ~0.10
        n_win = 60
        n_loss = 40
        _seed_bet_outcomes(db, "pts", n_win=n_win, n_loss=n_loss, edge=0.10)

        from utils.nba_calibration import NBACalibrator

        cal = NBACalibrator()
        cal.fit(min_samples=50)

        # p_raw proxy for edge=0.10: 0.10 + 0.5 = 0.60
        p_raw = 0.60
        result = cal.calibrate(p_raw, "pts")
        # Allow 15% deviation — isotonic regression can shift due to binning
        assert abs(result - p_raw) < 0.15, (
            f"Expected calibrated prob near {p_raw}, got {result}"
        )


# ---------------------------------------------------------------------------
# Test 5: Save/load roundtrip produces same output
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip:
    def test_save_load_same_calibration_output(self, db, tmp_path):
        """After save+load, calibrate() should return the same result."""
        _seed_bet_outcomes(db, "pts", n_win=36, n_loss=24)
        _seed_bet_outcomes(db, "reb", n_win=40, n_loss=20)

        from utils.nba_calibration import NBACalibrator

        cal1 = NBACalibrator()
        cal1.fit(min_samples=50)

        save_path = str(tmp_path / "calibration.joblib")
        cal1.save(save_path)

        cal2 = NBACalibrator()
        cal2.load(save_path)

        # Both should return the same result for the same input
        for market in ["pts", "reb"]:
            for p in [0.4, 0.55, 0.70]:
                r1 = cal1.calibrate(p, market)
                r2 = cal2.calibrate(p, market)
                assert abs(r1 - r2) < 1e-9, (
                    f"Mismatch for {market} at p={p}: {r1} vs {r2}"
                )

    def test_load_preserves_sample_counts(self, db, tmp_path):
        """Sample counts should be preserved through save/load."""
        _seed_bet_outcomes(db, "pts", n_win=36, n_loss=24)

        from utils.nba_calibration import NBACalibrator

        cal1 = NBACalibrator()
        counts_before = cal1.fit(min_samples=50)

        save_path = str(tmp_path / "calibration2.joblib")
        cal1.save(save_path)

        cal2 = NBACalibrator()
        cal2.load(save_path)

        assert cal2._sample_counts == counts_before


# ---------------------------------------------------------------------------
# Test 6: reliability_data() returns correct binned structure
# ---------------------------------------------------------------------------


class TestReliabilityData:
    def test_reliability_data_returns_expected_columns(self, db):
        """reliability_data() must return DataFrame with required columns."""
        _seed_bet_outcomes(db, "pts", n_win=36, n_loss=24)

        from utils.nba_calibration import NBACalibrator

        cal = NBACalibrator()
        cal.fit(min_samples=50)

        rel_df = cal.reliability_data("pts", n_bins=10)
        required_cols = {"bin_center", "predicted_prob", "observed_freq", "count"}
        assert required_cols.issubset(set(rel_df.columns))

    def test_reliability_data_correct_number_of_bins(self, db):
        """reliability_data() should return exactly n_bins rows."""
        _seed_bet_outcomes(db, "pts", n_win=36, n_loss=24)

        from utils.nba_calibration import NBACalibrator

        cal = NBACalibrator()
        cal.fit(min_samples=50)

        for n_bins in [5, 10, 20]:
            rel_df = cal.reliability_data("pts", n_bins=n_bins)
            assert len(rel_df) == n_bins, f"Expected {n_bins} bins, got {len(rel_df)}"

    def test_reliability_data_bin_centers_in_range(self, db):
        """All bin_center values must be in [0, 1]."""
        _seed_bet_outcomes(db, "pts", n_win=36, n_loss=24)

        from utils.nba_calibration import NBACalibrator

        cal = NBACalibrator()
        cal.fit(min_samples=50)

        rel_df = cal.reliability_data("pts", n_bins=10)
        assert (rel_df["bin_center"] >= 0.0).all()
        assert (rel_df["bin_center"] <= 1.0).all()

    def test_reliability_data_empty_for_missing_market(self, db):
        """reliability_data() for a market with no outcomes returns empty DataFrame."""
        from utils.nba_calibration import NBACalibrator

        cal = NBACalibrator()
        rel_df = cal.reliability_data("fg3m", n_bins=10)
        assert len(rel_df) == 0 or (rel_df["count"] == 0).all()


# ---------------------------------------------------------------------------
# Test 7: rank_nba_value(calibrated=True) populates p_win_raw key
# ---------------------------------------------------------------------------


class TestRankNbaValueCalibratedFlag:
    def test_calibrated_true_adds_p_win_raw_key(self, db):
        """rank_nba_value(calibrated=True) must include 'p_win_raw' in results."""
        _seed_projections_and_odds()
        _seed_bet_outcomes(db, "pts", n_win=36, n_loss=24)

        from utils.nba_calibration import NBACalibrator

        # Pre-train and save calibration model
        cal = NBACalibrator()
        cal.fit(min_samples=50)

        import os
        from pathlib import Path
        model_dir = Path(db).parent / "models" / "nba"
        model_dir.mkdir(parents=True, exist_ok=True)
        cal_path = str(model_dir / "calibration.joblib")
        cal.save(cal_path)

        # Monkeypatch the default calibration path in nba_value_engine
        import nba_value_engine

        original_path = getattr(nba_value_engine, "_DEFAULT_CALIBRATION_PATH", None)

        try:
            nba_value_engine._DEFAULT_CALIBRATION_PATH = cal_path
            results = nba_value_engine.rank_nba_value(
                GAME_DATE, SEASON, min_edge=0.0, calibrated=True
            )
        finally:
            if original_path is not None:
                nba_value_engine._DEFAULT_CALIBRATION_PATH = original_path

        assert len(results) > 0
        for r in results:
            assert "p_win_raw" in r, "p_win_raw key missing from calibrated result"
            assert "calibrated" in r, "calibrated key missing from result"
            assert r["calibrated"] is True

    def test_calibrated_false_p_win_raw_is_none(self, db):
        """rank_nba_value(calibrated=False) should have p_win_raw=None and calibrated=False."""
        _seed_projections_and_odds()

        from nba_value_engine import rank_nba_value

        results = rank_nba_value(GAME_DATE, SEASON, min_edge=0.0, calibrated=False)
        if results:
            assert results[0].get("p_win_raw") is None
            assert results[0].get("calibrated") is False


# ---------------------------------------------------------------------------
# Test 8: Missing calibration file -> silently ignored
# ---------------------------------------------------------------------------


class TestMissingCalibrationFile:
    def test_missing_file_falls_back_to_uncalibrated(self, db):
        """When calibration file does not exist, rank_nba_value behaves as calibrated=False."""
        _seed_projections_and_odds()

        import nba_value_engine

        # Point to a non-existent path
        original_path = getattr(nba_value_engine, "_DEFAULT_CALIBRATION_PATH", None)
        nba_value_engine._DEFAULT_CALIBRATION_PATH = "/nonexistent/path/calibration.joblib"

        try:
            results_calibrated = nba_value_engine.rank_nba_value(
                GAME_DATE, SEASON, min_edge=0.0, calibrated=True
            )
            results_uncalibrated = nba_value_engine.rank_nba_value(
                GAME_DATE, SEASON, min_edge=0.0, calibrated=False
            )
        finally:
            if original_path is not None:
                nba_value_engine._DEFAULT_CALIBRATION_PATH = original_path

        # Both should return results (not crash)
        assert isinstance(results_calibrated, list)
        assert isinstance(results_uncalibrated, list)

        # p_win values should match since calibration file was missing
        if results_calibrated and results_uncalibrated:
            p_cal = results_calibrated[0]["p_win"]
            p_uncal = results_uncalibrated[0]["p_win"]
            assert abs(p_cal - p_uncal) < 1e-9, (
                f"Missing calibration file changed p_win: {p_cal} vs {p_uncal}"
            )
