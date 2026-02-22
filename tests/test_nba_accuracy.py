"""Tests for NBA line accuracy computation."""
import pytest

from schema_migrations import MigrationManager
from utils.db import executemany, read_dataframe

from scripts.record_nba_outcomes import compute_nba_line_accuracy, summarize_nba_accuracy


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Create a temporary SQLite database with all NBA tables."""
    db_path = tmp_path / "test_nba_accuracy.db"
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))
    try:
        from config import config
        monkeypatch.setattr(config.database, "backend", "sqlite")
        monkeypatch.setattr(config.database, "path", str(db_path))
    except (ImportError, AttributeError):
        pass
    MigrationManager(db_path).run()
    return db_path


class TestComputeLineAccuracy:
    def test_computes_both_errors(self, db):
        """Model and line errors should be computed correctly."""
        executemany(
            """INSERT INTO nba_bet_outcomes
            (bet_id, season, game_date, player_id, player_name, market, sportsbook, side, line, price, actual_result, result, profit_units, confidence_tier, edge_at_placement, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [("b1", 2025, "2026-02-20", 1, "Player A", "pts", "DK", "over", 22.5, -110, 28.0, "win", 0.91, "high", 0.12, "2026-02-20T23:00:00")],
        )
        executemany(
            """INSERT INTO nba_projections
            (player_id, player_name, team, season, game_date, game_id, market, projected_value, confidence, sigma)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(1, "Player A", "BOS", 2025, "2026-02-20", "G1", "pts", 26.0, 0.8, 4.0)],
        )

        count = compute_nba_line_accuracy("2026-02-20")
        assert count == 1

        df = read_dataframe("SELECT * FROM nba_line_accuracy_history WHERE game_date = '2026-02-20'")
        assert len(df) == 1
        row = df.iloc[0]

        # model predicted 26.0, actual was 28.0 -> model error = 2.0
        assert abs(row["model_abs_error"] - 2.0) < 0.01
        # line was 22.5, actual was 28.0 -> line error = 5.5
        assert abs(row["line_abs_error"] - 5.5) < 0.01
        # model error (2.0) < line error (5.5) -> model wins
        assert row["model_beats_line"] == 1
        # actual (28) > line (22.5) -> over hit
        assert row["is_over_hit"] == 1

    def test_model_beats_line_flag_when_line_closer(self, db):
        """model_beats_line should be 0 when line is more accurate."""
        executemany(
            """INSERT INTO nba_bet_outcomes
            (bet_id, season, game_date, player_id, player_name, market, sportsbook, side, line, price, actual_result, result, profit_units, confidence_tier, edge_at_placement, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [("b2", 2025, "2026-02-20", 2, "Player B", "reb", "FD", "over", 8.5, -110, 9.0, "win", 0.91, "med", 0.10, "2026-02-20T23:00:00")],
        )
        executemany(
            """INSERT INTO nba_projections
            (player_id, player_name, team, season, game_date, game_id, market, projected_value, confidence, sigma)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(2, "Player B", "LAL", 2025, "2026-02-20", "G2", "reb", 12.0, 0.7, 3.0)],
        )

        compute_nba_line_accuracy("2026-02-20")
        df = read_dataframe("SELECT * FROM nba_line_accuracy_history WHERE player_id = 2")
        assert len(df) == 1
        # model predicted 12.0, actual 9.0 -> error = 3.0
        # line was 8.5, actual 9.0 -> error = 0.5
        # line is closer -> model_beats_line = 0
        assert df.iloc[0]["model_beats_line"] == 0

    def test_idempotent(self, db):
        """Running twice should not create duplicate rows (INSERT OR REPLACE)."""
        executemany(
            """INSERT INTO nba_bet_outcomes
            (bet_id, season, game_date, player_id, player_name, market, sportsbook, side, line, price, actual_result, result, profit_units, confidence_tier, edge_at_placement, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [("b3", 2025, "2026-02-21", 3, "Player C", "ast", "BM", "over", 6.5, -115, 8.0, "win", 0.87, "high", 0.15, "2026-02-21T23:00:00")],
        )
        executemany(
            """INSERT INTO nba_projections
            (player_id, player_name, team, season, game_date, game_id, market, projected_value, confidence, sigma)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(3, "Player C", "MIA", 2025, "2026-02-21", "G3", "ast", 7.5, 0.85, 2.0)],
        )

        count1 = compute_nba_line_accuracy("2026-02-21")
        count2 = compute_nba_line_accuracy("2026-02-21")
        assert count1 == 1
        assert count2 == 1

        df = read_dataframe("SELECT * FROM nba_line_accuracy_history WHERE game_date = '2026-02-21'")
        assert len(df) == 1  # No duplicates

    def test_no_outcomes_returns_zero(self, db):
        """Should return 0 when no graded outcomes exist."""
        count = compute_nba_line_accuracy("2026-01-01")
        assert count == 0

    def test_no_projections_returns_zero(self, db):
        """Should return 0 when outcomes exist but no matching projections."""
        executemany(
            """INSERT INTO nba_bet_outcomes
            (bet_id, season, game_date, player_id, player_name, market, sportsbook, side, line, price, actual_result, result, profit_units, confidence_tier, edge_at_placement, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [("b4", 2025, "2026-02-22", 4, "Player D", "pts", "DK", "over", 20.5, -110, 22.0, "win", 0.91, "high", 0.10, "2026-02-22T23:00:00")],
        )
        # No matching projection inserted
        count = compute_nba_line_accuracy("2026-02-22")
        assert count == 0


class TestSummarizeAccuracy:
    def test_summary_correct_averages(self, db):
        """Summary should compute correct aggregate MAE values."""
        executemany(
            """INSERT INTO nba_line_accuracy_history
            (season, game_date, player_id, market, sportsbook, line, actual, mu, sigma, model_abs_error, line_abs_error, model_beats_line, is_over_hit, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (2025, "2026-02-20", 1, "pts", "DK", 22.5, 28.0, 26.0, 4.0, 2.0, 5.5, 1, 1, "now"),
                (2025, "2026-02-20", 2, "reb", "FD", 8.5, 9.0, 12.0, 3.0, 3.0, 0.5, 0, 1, "now"),
                (2025, "2026-02-20", 3, "ast", "BM", 6.5, 8.0, 7.5, 2.0, 0.5, 1.5, 1, 1, "now"),
            ],
        )

        summary = summarize_nba_accuracy(2025)
        assert summary["n_bets"] == 3
        # avg model MAE = (2.0 + 3.0 + 0.5) / 3 = 1.833
        assert abs(summary["avg_model_mae"] - 1.8333) < 0.01
        # avg line MAE = (5.5 + 0.5 + 1.5) / 3 = 2.5
        assert abs(summary["avg_line_mae"] - 2.5) < 0.01
        # model beats line = 2/3 = 66.67%
        assert abs(summary["model_beats_line_pct"] - 66.67) < 0.1

    def test_summary_empty(self, db):
        """Empty data should return None values."""
        summary = summarize_nba_accuracy(2025)
        assert summary["n_bets"] == 0
        assert summary["avg_model_mae"] is None

    def test_summary_date_filter(self, db):
        """Date filters should restrict results correctly."""
        executemany(
            """INSERT INTO nba_line_accuracy_history
            (season, game_date, player_id, market, sportsbook, line, actual, mu, sigma, model_abs_error, line_abs_error, model_beats_line, is_over_hit, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (2025, "2026-02-18", 1, "pts", "DK", 22.5, 28.0, 26.0, 4.0, 2.0, 5.5, 1, 1, "now"),
                (2025, "2026-02-20", 2, "reb", "FD", 8.5, 9.0, 12.0, 3.0, 3.0, 0.5, 0, 1, "now"),
            ],
        )

        # Only the Feb 20 record
        summary = summarize_nba_accuracy(2025, start_date="2026-02-20")
        assert summary["n_bets"] == 1
        assert abs(summary["avg_model_mae"] - 3.0) < 0.01
