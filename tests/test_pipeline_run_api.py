"""Tests for the pipeline run API endpoints (Feature 1)."""

import json
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from starlette.requests import Request

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

    from api.application import app
    from api.pipeline_router import require_pipeline_operator, require_pipeline_reader

    app.dependency_overrides[require_pipeline_operator] = lambda: "test-operator"
    app.dependency_overrides[require_pipeline_reader] = lambda: "test-reader"
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestPipelineRunAPI:
    def test_post_returns_run_id(self, client):
        resp = client.post("/api/run?season=2025&week=22&skip_ingest=true&skip_odds=true")
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert data["status"] == "queued"
        assert data["season"] == 2025
        assert data["week"] == 22
        assert data["started_at"] is not None

    def test_get_404_for_missing(self, client):
        resp = client.get("/api/run/nonexistent-id")
        assert resp.status_code == 404

    def test_get_returns_queued_run_without_api_execution(self, client):
        post_resp = client.post("/api/run?season=2025&week=22&skip_ingest=true&skip_odds=true")
        run_id = post_resp.json()["run_id"]

        resp = client.get(f"/api/run/{run_id}")
        assert resp.status_code == 200
        data = resp.json()

        assert data["run_id"] == run_id
        assert data["status"] == "queued"
        assert data["job_id"]
        assert data["worker_id"] is None

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

    def test_cancel_queued_run(self, client):
        created = client.post("/api/run?season=2025&week=22").json()

        response = client.post(f"/api/run/{created['run_id']}/cancel")

        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

    def test_mutation_requires_operator_auth(self, db):
        from fastapi.testclient import TestClient

        from api.application import app

        app.dependency_overrides.clear()
        response = TestClient(app).post("/api/run?season=2025&week=22")
        assert response.status_code == 401

    def test_architecture_read_requires_auth(self, db):
        from fastapi.testclient import TestClient

        from api.application import app

        app.dependency_overrides.clear()
        response = TestClient(app).get("/api/system/architecture")
        assert response.status_code == 401


def test_free_account_is_not_pipeline_operator(monkeypatch) -> None:
    from api.auth import UserResponse
    from api.pipeline_router import require_pipeline_operator

    monkeypatch.setattr(
        "api.pipeline_router.validate_session",
        lambda token: UserResponse(
            id="free-user",
            email="free@example.com",
            name=None,
            subscription_tier="free",
            bankroll=1000,
            created_at="2026-01-01T00:00:00+00:00",
        ),
    )
    request = Request({"type": "http", "headers": [(b"authorization", b"Bearer valid-session")]})

    with pytest.raises(HTTPException) as exc_info:
        require_pipeline_operator(request)

    assert exc_info.value.status_code == 403


def test_private_server_uses_real_session_auth_for_pipeline_routes(db) -> None:
    from fastapi.testclient import TestClient

    from api.application import app

    now = datetime.now(timezone.utc)
    expires = (now + timedelta(hours=1)).isoformat()
    for user_id, email, tier, session_id in (
        ("reader-user", "reader@example.com", "free", "reader-session"),
        ("operator-user", "operator@example.com", "operator", "operator-session"),
    ):
        execute(
            """
            INSERT INTO users
                (id, email, password_hash, name, subscription_tier, bankroll, created_at, updated_at)
            VALUES (?, ?, 'unused', ?, ?, 1000, ?, ?)
            """,
            (user_id, email, tier.title(), tier, now.isoformat(), now.isoformat()),
        )
        execute(
            "INSERT INTO user_sessions (session_id, user_id, expires_at, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, user_id, expires, now.isoformat()),
        )

    app.dependency_overrides.clear()
    with TestClient(app) as private_client:
        reader_headers = {"Authorization": "Bearer reader-session"}
        operator_headers = {"Authorization": "Bearer operator-session"}

        assert (
            private_client.get("/api/system/pipeline-metrics", headers=reader_headers).status_code
            == 200
        )
        assert (
            private_client.post("/api/run?season=2026&week=1", headers=reader_headers).status_code
            == 403
        )
        assert (
            private_client.post("/api/run?season=2026&week=1", headers=operator_headers).status_code
            == 200
        )
