"""Tests for the pipeline run API endpoints (Feature 1)."""

import json
import time

import pytest

from schema_migrations import MigrationManager
from utils.db import execute, fetchone


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Provision a temporary SQLite database with schema."""
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
    """FastAPI TestClient."""
    from fastapi.testclient import TestClient
    from api.server import app
    return TestClient(app)


class TestPipelineRunAPI:
    def test_post_returns_run_id(self, client):
        resp = client.post("/api/run?season=2025&week=22&skip_ingest=true&skip_odds=true")
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert data["status"] == "running"
        assert data["season"] == 2025
        assert data["week"] == 22
        assert data["started_at"] is not None

    def test_get_404_for_missing(self, client):
        resp = client.get("/api/run/nonexistent-id")
        assert resp.status_code == 404

    def test_get_returns_run(self, client):
        # Create a run first
        post_resp = client.post("/api/run?season=2025&week=22&skip_ingest=true&skip_odds=true")
        run_id = post_resp.json()["run_id"]

        # Poll until done (with timeout)
        for _ in range(10):
            resp = client.get(f"/api/run/{run_id}")
            assert resp.status_code == 200
            data = resp.json()
            if data["status"] != "running":
                break
            time.sleep(0.5)

        assert data["run_id"] == run_id
        assert data["status"] in ("completed", "failed")

    def test_latest_returns_none_when_empty(self, client):
        resp = client.get("/api/run/latest?season=9999&week=99")
        assert resp.status_code == 200
        assert resp.json() is None

    def test_latest_returns_most_recent(self, client):
        client.post("/api/run?season=2025&week=22&skip_ingest=true&skip_odds=true")
        time.sleep(0.3)
        resp2 = client.post("/api/run?season=2025&week=22&skip_ingest=true&skip_odds=true")
        second_id = resp2.json()["run_id"]

        time.sleep(0.3)
        latest = client.get("/api/run/latest?season=2025&week=22")
        assert latest.status_code == 200
        data = latest.json()
        assert data is not None
        assert data["run_id"] == second_id
