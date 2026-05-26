"""Tests for T0 #2: bcrypt + endpoint signatures + legacy SHA256 verify."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import bcrypt
import pytest
from fastapi.testclient import TestClient

from config import config
from schema_migrations import MigrationManager


@pytest.fixture
def temp_db(monkeypatch):
    tmp = Path(tempfile.mkstemp(suffix=".db")[1])
    orig_path = config.database.path
    orig_backend = config.database.backend
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp))
    config.database.backend = "sqlite"
    config.database.path = str(tmp)
    MigrationManager(tmp).run()
    yield str(tmp)
    config.database.backend = orig_backend
    config.database.path = orig_path
    Path(tmp).unlink(missing_ok=True)


def test_hash_password_uses_bcrypt(temp_db):
    from api.auth import hash_password
    h = hash_password("hunter2_safe")
    assert h.startswith(("$2a$", "$2b$", "$2y$")), "expected bcrypt prefix"


def test_verify_password_bcrypt_roundtrip(temp_db):
    from api.auth import hash_password, verify_password
    h = hash_password("hunter2_safe")
    assert verify_password("hunter2_safe", h) is True
    assert verify_password("wrong-password", h) is False


def test_verify_password_legacy_sha256_still_works(temp_db):
    import hashlib
    import secrets

    from api.auth import verify_password

    pw = "legacy_password"
    salt = secrets.token_hex(16)
    legacy = f"{salt}${hashlib.sha256((pw + salt).encode()).hexdigest()}"
    assert verify_password(pw, legacy) is True
    assert verify_password("wrong", legacy) is False


def test_verify_password_rejects_empty_hash(temp_db):
    from api.auth import verify_password
    assert verify_password("anything", "") is False


def test_userresponse_user_id_alias(temp_db):
    from api.auth import UserResponse
    u = UserResponse(
        id="u-123", email="x@y.com", name=None,
        subscription_tier="free", bankroll=100.0, created_at="2024-01-01",
    )
    assert u.user_id == "u-123"


def test_register_endpoint_uses_pydantic_signature(temp_db):
    from api.server import app
    with TestClient(app) as c:
        resp = c.post(
            "/api/auth/register",
            json={"email": "new@example.com", "password": "strongpass1", "name": "New User"},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["user"]["email"] == "new@example.com"
    assert "session_id" in body

    # Hash actually bcrypt
    with sqlite3.connect(temp_db) as conn:
        hash_value = conn.execute(
            "SELECT password_hash FROM users WHERE email = ?", ("new@example.com",)
        ).fetchone()[0]
    assert hash_value.startswith(("$2a$", "$2b$", "$2y$"))


def test_login_endpoint_uses_pydantic_signature(temp_db):
    from api.server import app
    with TestClient(app) as c:
        c.post(
            "/api/auth/register",
            json={"email": "user1@example.com", "password": "strongpass1"},
        )
        resp = c.post(
            "/api/auth/login",
            json={"email": "user1@example.com", "password": "strongpass1"},
        )
    assert resp.status_code == 200, resp.text
    assert "session_id" in resp.json()


def test_login_rejects_bad_password(temp_db):
    from api.server import app
    with TestClient(app) as c:
        c.post(
            "/api/auth/register",
            json={"email": "user2@example.com", "password": "strongpass1"},
        )
        resp = c.post(
            "/api/auth/login",
            json={"email": "user2@example.com", "password": "wrong-pass"},
        )
    assert resp.status_code == 401


def test_legacy_sha256_user_can_login_and_gets_rehashed(temp_db):
    """Existing pre-T0 #2 users still authenticate; their hash rehashes to bcrypt."""
    import hashlib
    import secrets

    from api.server import app

    pw = "legacy_password"
    salt = secrets.token_hex(16)
    legacy = f"{salt}${hashlib.sha256((pw + salt).encode()).hexdigest()}"

    with sqlite3.connect(temp_db) as conn:
        conn.execute(
            """
            INSERT INTO users (id, email, password_hash, name, subscription_tier,
                               bankroll, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'free', 100.0, datetime('now'), datetime('now'))
            """,
            ("u-legacy", "legacy@example.com", legacy, "Legacy User"),
        )
        conn.commit()

    with TestClient(app) as c:
        resp = c.post(
            "/api/auth/login",
            json={"email": "legacy@example.com", "password": pw},
        )
    assert resp.status_code == 200, resp.text

    with sqlite3.connect(temp_db) as conn:
        new_hash = conn.execute(
            "SELECT password_hash FROM users WHERE email = ?", ("legacy@example.com",)
        ).fetchone()[0]
    assert new_hash.startswith(("$2a$", "$2b$", "$2y$")), "legacy hash should be rehashed"
