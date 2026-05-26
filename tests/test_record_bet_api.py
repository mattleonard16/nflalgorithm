"""Tests for T0 #3: POST /api/user/bets BetCreate + schema-correct INSERT."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.auth import UserResponse
from config import config
from schema_migrations import MigrationManager


@pytest.fixture
def temp_db_with_user(monkeypatch):
    tmp = Path(tempfile.mkstemp(suffix=".db")[1])
    orig_path = config.database.path
    orig_backend = config.database.backend
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp))
    config.database.backend = "sqlite"
    config.database.path = str(tmp)
    MigrationManager(tmp).run()

    # Seed a user so FK passes
    with sqlite3.connect(tmp) as conn:
        conn.execute(
            """
            INSERT INTO users (id, email, password_hash, name, created_at, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
            """,
            ("u-test", "test@example.com", "x", "Test"),
        )
        conn.commit()

    yield str(tmp)

    config.database.path = orig_path
    config.database.backend = orig_backend
    Path(tmp).unlink(missing_ok=True)


@pytest.fixture
def client(temp_db_with_user):
    from api.server import app, get_current_user

    async def _override_user():
        return UserResponse(
            id="u-test", email="test@example.com", name="Test",
            subscription_tier="free", bankroll=100.0,
            created_at="2024-01-01T00:00:00Z",
        )

    app.dependency_overrides[get_current_user] = _override_user
    # The endpoint reads .user_id off UserResponse (legacy alias). Until T0 #2
    # renames the field, monkeypatch the attribute for this test session.
    UserResponse.user_id = property(lambda self: self.id)

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


def test_record_bet_inserts_with_correct_columns(client, temp_db_with_user):
    resp = client.post(
        "/api/user/bets",
        json={
            "season": 2024,
            "week": 5,
            "player_id": "p1",
            "player_name": "Test Player",
            "market": "rushing_yards",
            "sportsbook": "BookA",
            "side": "over",
            "line": 75.5,
            "price": -110,
            "stake_units": 1.0,
            "model_edge": 0.08,
            "confidence_tier": "Strong",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "bet_id" in body

    with sqlite3.connect(temp_db_with_user) as conn:
        row = conn.execute(
            "SELECT user_id, side, stake_units, model_edge, market, line FROM user_bets"
        ).fetchone()
    assert row == ("u-test", "over", 1.0, 0.08, "rushing_yards", 75.5)


def test_record_bet_rejects_invalid_side(client):
    resp = client.post(
        "/api/user/bets",
        json={
            "season": 2024, "week": 5, "player_id": "p1",
            "market": "rushing_yards", "sportsbook": "BookA",
            "side": "middle", "line": 75.5, "price": -110, "stake_units": 1.0,
        },
    )
    assert resp.status_code == 422


def test_record_bet_rejects_missing_required_fields(client):
    resp = client.post(
        "/api/user/bets",
        json={"season": 2024, "week": 5, "player_id": "p1"},
    )
    assert resp.status_code == 422


def test_record_bet_accepts_under_side(client, temp_db_with_user):
    resp = client.post(
        "/api/user/bets",
        json={
            "season": 2024, "week": 5, "player_id": "p1",
            "market": "rushing_yards", "sportsbook": "BookA",
            "side": "under", "line": 75.5, "price": -110, "stake_units": 1.0,
        },
    )
    assert resp.status_code == 200
    with sqlite3.connect(temp_db_with_user) as conn:
        sides = [r[0] for r in conn.execute("SELECT side FROM user_bets").fetchall()]
    assert "under" in sides


def test_record_bet_rejects_unauthenticated():
    """Without auth dep override, endpoint must return 401."""
    from api.server import app
    with TestClient(app) as c:
        resp = c.post("/api/user/bets", json={
            "season": 2024, "week": 5, "player_id": "p1",
            "market": "rushing_yards", "sportsbook": "BookA",
            "side": "over", "line": 75.5, "price": -110, "stake_units": 1.0,
        })
    assert resp.status_code == 401
