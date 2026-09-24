"""Who may read and control pipeline runs (api/pipeline_router.py).

Requests go through the served app (`api.application`), so the checks cover
the real route table: the tracked pipeline routes replace the legacy
unauthenticated ones in `api.server`. Sessions are real rows in a throwaway
SQLite database; the control token and operator tiers come from the
environment the way a deployment sets them.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from config import config
from schema_migrations import MigrationManager
from utils.db import execute

CONTROL_TOKEN = "ctl-secret-for-tests"

OPERATOR_ROUTES = [
    "/api/run/some-run/cancel",
    "/api/run/some-run/retry",
    "/api/run/some-run/review?season=2026&week=1",
]

# POST /api/run and GET /api/system/architecture are covered anonymously in
# tests/test_pipeline_run_api.py.
READER_ROUTES = [
    "/api/run/latest?season=2026&week=1",
    "/api/run/some-run",
    "/api/run/some-run/review-status?season=2026&week=1",
    "/api/system/pipeline-metrics",
]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "pipeline-auth.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)
    monkeypatch.setattr(config.database, "backend", "sqlite")
    monkeypatch.setattr(config.database, "path", db_path)
    monkeypatch.delenv("PIPELINE_CONTROL_TOKEN", raising=False)
    monkeypatch.delenv("PIPELINE_OPERATOR_TIERS", raising=False)
    MigrationManager(db_path).run()

    from api.application import app

    app.dependency_overrides.clear()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _session(tier: str, *, expires_in: timedelta = timedelta(hours=1)) -> dict[str, str]:
    """Insert a user on `tier` with one session; return its auth header."""
    now = datetime.now(timezone.utc)
    user_id = f"user-{tier.lower()}"
    session_id = f"session-{tier.lower()}"
    execute(
        "INSERT INTO users (id, email, password_hash, subscription_tier, created_at, updated_at) "
        "VALUES (?, ?, 'unused', ?, ?, ?)",
        (user_id, f"{user_id}@example.com", tier, now.isoformat(), now.isoformat()),
    )
    execute(
        "INSERT INTO user_sessions (session_id, user_id, expires_at, created_at) "
        "VALUES (?, ?, ?, ?)",
        (session_id, user_id, (now + expires_in).isoformat(), now.isoformat()),
    )
    return {"Authorization": f"Bearer {session_id}"}


def _trigger_run(client: TestClient, headers: dict[str, str]) -> int:
    return int(client.post("/api/run?season=2026&week=1", headers=headers).status_code)


# ---------------------------------------------------------------------------
# Control token
# ---------------------------------------------------------------------------


def test_the_control_token_can_trigger_a_run(client, monkeypatch):
    monkeypatch.setenv("PIPELINE_CONTROL_TOKEN", CONTROL_TOKEN)

    assert _trigger_run(client, {"Authorization": f"Bearer {CONTROL_TOKEN}"}) == 200


def test_a_wrong_bearer_is_rejected_when_a_control_token_is_set(client, monkeypatch):
    monkeypatch.setenv("PIPELINE_CONTROL_TOKEN", CONTROL_TOKEN)

    assert _trigger_run(client, {"Authorization": "Bearer ctl-guess"}) == 401


@pytest.mark.parametrize("configured", [None, ""])
def test_an_empty_bearer_never_matches_an_unconfigured_control_token(
    client, monkeypatch, configured
):
    if configured is not None:
        monkeypatch.setenv("PIPELINE_CONTROL_TOKEN", configured)

    assert _trigger_run(client, {"Authorization": "Bearer "}) == 401


@pytest.mark.parametrize("header", [CONTROL_TOKEN, f"Basic {CONTROL_TOKEN}"])
def test_the_control_token_counts_only_as_a_bearer_credential(client, monkeypatch, header):
    monkeypatch.setenv("PIPELINE_CONTROL_TOKEN", CONTROL_TOKEN)

    assert _trigger_run(client, {"Authorization": header}) == 401


# ---------------------------------------------------------------------------
# Operator tiers
# ---------------------------------------------------------------------------


def test_the_operator_tier_env_replaces_the_default_tiers(client, monkeypatch):
    monkeypatch.setenv("PIPELINE_OPERATOR_TIERS", "pro")
    pro = _session("pro")
    operator = _session("operator")

    assert _trigger_run(client, pro) == 200
    assert _trigger_run(client, operator) == 403


def test_operator_tier_matching_ignores_case(client):
    assert _trigger_run(client, _session("Admin")) == 200


@pytest.mark.parametrize("route", OPERATOR_ROUTES)
def test_a_reader_session_cannot_call_an_operator_route(client, route):
    assert client.post(route, headers=_session("free")).status_code == 403


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", READER_ROUTES)
def test_an_anonymous_request_cannot_read_pipeline_state(client, route):
    assert client.get(route).status_code == 401


def test_an_expired_session_cannot_read_pipeline_state(client):
    expired = _session("operator", expires_in=-timedelta(minutes=1))

    assert client.get("/api/system/pipeline-metrics", headers=expired).status_code == 401
