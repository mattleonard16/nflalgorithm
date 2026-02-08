"""Tests for export and agent review endpoints (P1/P2)."""

import json

import pytest

from schema_migrations import MigrationManager
from utils.db import execute


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


@pytest.fixture()
def client(db):
    from fastapi.testclient import TestClient
    from api.server import app
    return TestClient(app)


def _seed_bets(db, season=2025, week=22):
    """Seed player_dim and materialized_value_view for export tests."""
    execute(
        """
        INSERT INTO player_dim (player_id, player_name, position, team, last_season, last_week, updated_at)
        VALUES ('P001', 'Patrick Mahomes', 'QB', 'KC', ?, ?, datetime('now'))
        """,
        params=(season, week),
    )
    execute(
        """
        INSERT INTO materialized_value_view
            (season, week, player_id, event_id, team, market, sportsbook,
             line, price, mu, sigma, p_win, edge_percentage, expected_roi,
             kelly_fraction, stake, generated_at)
        VALUES (?, ?, 'P001', 'evt1', 'KC', 'passing_yards', 'draftkings',
                280.5, -110, 310.0, 30.0, 0.65, 0.12, 0.10, 0.02, 20.0, datetime('now'))
        """,
        params=(season, week),
    )


def _seed_pipeline_run(db, run_id="test-run-001", season=2025, week=22, status="completed"):
    execute(
        """
        INSERT INTO pipeline_runs (run_id, season, week, status, stages_requested, stages_completed, started_at)
        VALUES (?, ?, ?, ?, 8, 8, datetime('now'))
        """,
        params=(run_id, season, week, status),
    )


class TestExportCSV:
    def test_csv_returns_200(self, client, db):
        _seed_bets(db)
        resp = client.get("/api/export/csv?season=2025&week=22")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]

    def test_csv_contains_headers(self, client, db):
        _seed_bets(db)
        resp = client.get("/api/export/csv?season=2025&week=22")
        content = resp.text
        assert "player_id" in content
        assert "player_name" in content
        assert "edge_percentage" in content

    def test_csv_contains_data(self, client, db):
        _seed_bets(db)
        resp = client.get("/api/export/csv?season=2025&week=22")
        content = resp.text
        assert "Patrick Mahomes" in content
        assert "passing_yards" in content

    def test_csv_empty_returns_no_data(self, client, db):
        resp = client.get("/api/export/csv?season=2024&week=1")
        assert resp.status_code == 200

    def test_csv_content_disposition(self, client, db):
        _seed_bets(db)
        resp = client.get("/api/export/csv?season=2025&week=22")
        assert "attachment" in resp.headers.get("content-disposition", "")
        assert ".csv" in resp.headers.get("content-disposition", "")


class TestExportBundle:
    def test_bundle_returns_json(self, client, db):
        _seed_bets(db)
        resp = client.get("/api/export/bundle?season=2025&week=22")
        assert resp.status_code == 200
        data = json.loads(resp.text)
        assert "bets" in data
        assert "total_bets" in data
        assert "exported_at" in data

    def test_bundle_includes_bets(self, client, db):
        _seed_bets(db)
        resp = client.get("/api/export/bundle?season=2025&week=22")
        data = json.loads(resp.text)
        assert data["total_bets"] == 1
        assert data["bets"][0]["player_name"] == "Patrick Mahomes"

    def test_bundle_includes_run_metadata(self, client, db):
        _seed_bets(db)
        _seed_pipeline_run(db)
        resp = client.get("/api/export/bundle?season=2025&week=22")
        data = json.loads(resp.text)
        assert data["pipeline_run"] is not None
        assert data["pipeline_run"]["run_id"] == "test-run-001"

    def test_bundle_empty_week(self, client, db):
        resp = client.get("/api/export/bundle?season=2024&week=1")
        assert resp.status_code == 200
        data = json.loads(resp.text)
        assert data["total_bets"] == 0
        assert data["bets"] == []


class TestAgentReview:
    def test_review_returns_404_for_missing_run(self, client, db):
        resp = client.post("/api/run/nonexistent/review?season=2025&week=22")
        assert resp.status_code == 404

    def test_review_starts_for_valid_run(self, client, db):
        _seed_pipeline_run(db)
        resp = client.post("/api/run/test-run-001/review?season=2025&week=22")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == "test-run-001"
        assert data["review_status"] == "started"

    def test_review_status_returns_not_reviewed(self, client, db):
        _seed_pipeline_run(db)
        resp = client.get("/api/run/test-run-001/review-status?season=2025&week=22")
        assert resp.status_code == 200
        data = resp.json()
        assert data["reviewed"] is False
        assert data["decision_count"] == 0

    def test_review_status_404_for_missing(self, client, db):
        resp = client.get("/api/run/nonexistent/review-status?season=2025&week=22")
        assert resp.status_code == 404


class TestDataHealthInPipelineRun:
    def test_pipeline_run_includes_data_health_field(self, client, db):
        _seed_pipeline_run(db)
        resp = client.get("/api/run/test-run-001")
        assert resp.status_code == 200
        data = resp.json()
        # data_health may be null if no checks were run
        assert "data_health" in data

    def test_latest_run_includes_data_health(self, client, db):
        _seed_pipeline_run(db)
        resp = client.get("/api/run/latest?season=2025&week=22")
        assert resp.status_code == 200
        data = resp.json()
        assert "data_health" in data
