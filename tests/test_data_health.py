"""Tests for data health invariant checks (P1: Data Correctness)."""

import pytest

from schema_migrations import MigrationManager
from utils.db import execute
from api.data_health import (
    check_missing_player_info,
    check_duplicate_player_dim,
    check_null_lines,
    check_projection_coverage,
    run_all_checks,
)


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    import config as cfg
    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")

    MigrationManager(db_path).run()
    return db_path


def _seed_player_dim(db, player_id="P001", name="Test Player", pos="WR", team="KC"):
    execute(
        """
        INSERT INTO player_dim (player_id, player_name, position, team, last_season, last_week, updated_at)
        VALUES (?, ?, ?, ?, 2025, 22, datetime('now'))
        """,
        params=(player_id, name, pos, team),
    )


def _seed_bet(db, player_id="P001", season=2025, week=22, market="receiving_yards"):
    execute(
        """
        INSERT INTO materialized_value_view
            (season, week, player_id, event_id, team, market, sportsbook,
             line, price, mu, sigma, p_win, edge_percentage, expected_roi,
             kelly_fraction, stake, generated_at)
        VALUES (?, ?, ?, 'evt1', 'KC', ?, 'draftkings',
                75.5, -110, 85.0, 8.0, 0.65, 0.15, 0.12,
                0.02, 20.0, datetime('now'))
        """,
        params=(season, week, player_id, market),
    )


def _seed_projection(db, player_id="P001", season=2025, week=22, market="receiving_yards"):
    execute(
        """
        INSERT INTO weekly_projections
            (season, week, player_id, team, opponent, market, mu, sigma,
             model_version, featureset_hash, generated_at)
        VALUES (?, ?, ?, 'KC', 'DEN', ?, 85.0, 8.0,
                'v2', 'abc123', datetime('now'))
        """,
        params=(season, week, player_id, market),
    )


class TestMissingPlayerInfo:
    def test_all_present(self, db):
        _seed_player_dim(db, "P001")
        _seed_bet(db, "P001")
        result = check_missing_player_info(2025, 22)
        assert result["status"] == "pass"
        assert result["missing_name"] == 0
        assert result["missing_position"] == 0

    def test_missing_name(self, db):
        _seed_bet(db, "P999")  # No player_dim entry
        result = check_missing_player_info(2025, 22)
        assert result["missing_name"] == 1
        assert result["missing_name_rate"] == 1.0

    def test_empty_week(self, db):
        result = check_missing_player_info(9999, 99)
        assert result["total_bets"] == 0
        assert result["status"] == "pass"


class TestDuplicatePlayerDim:
    def test_no_duplicates(self, db):
        _seed_player_dim(db, "P001")
        _seed_player_dim(db, "P002", name="Another Player")
        result = check_duplicate_player_dim()
        assert result["status"] == "pass"
        assert result["duplicates"] == 0

    def test_empty_table(self, db):
        result = check_duplicate_player_dim()
        assert result["total_rows"] == 0
        assert result["status"] == "pass"


class TestNullLines:
    def test_no_nulls(self, db):
        _seed_bet(db, "P001")
        result = check_null_lines(2025, 22)
        assert result["status"] == "pass"
        assert result["null_line"] == 0
        assert result["null_mu"] == 0

    def test_empty_week(self, db):
        result = check_null_lines(9999, 99)
        assert result["total_bets"] == 0
        assert result["status"] == "pass"


class TestProjectionCoverage:
    def test_full_coverage(self, db):
        _seed_bet(db, "P001")
        _seed_projection(db, "P001")
        result = check_projection_coverage(2025, 22)
        assert result["status"] == "pass"
        assert result["coverage_rate"] == 1.0

    def test_no_projections(self, db):
        _seed_bet(db, "P001")
        result = check_projection_coverage(2025, 22)
        assert result["with_projection"] == 0
        assert result["status"] == "warn"


class TestRunAllChecks:
    def test_all_pass(self, db):
        _seed_player_dim(db, "P001")
        _seed_bet(db, "P001")
        _seed_projection(db, "P001")
        report = run_all_checks(2025, 22)
        assert report["overall"] == "pass"
        assert len(report["checks"]) == 4

    def test_warns_on_issues(self, db):
        _seed_bet(db, "P999")  # No player_dim, no projection
        report = run_all_checks(2025, 22)
        assert report["overall"] == "warn"

    def test_empty_data(self, db):
        report = run_all_checks(9999, 99)
        assert report["overall"] == "pass"
