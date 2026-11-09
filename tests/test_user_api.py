"""Per-user data isolation and account endpoints in api/server.py.

Two users with real sessions in a throwaway SQLite database, driven through
the served app. tests/test_record_bet_api.py covers the bet payload with the
current user stubbed; this file checks that one user's session never reads or
writes another user's rows.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from config import config
from schema_migrations import MigrationManager
from utils.db import execute, fetchall, fetchone

ALICE = {"Authorization": "Bearer session-alice"}
BOB = {"Authorization": "Bearer session-bob"}

BET = {
    "season": 2026,
    "week": 3,
    "player_id": "BUF_receiver_a",
    "market": "receiving_yards",
    "sportsbook": "DraftKings",
    "side": "over",
    "line": 64.5,
    "price": -110,
    "stake_units": 1.0,
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "user-api.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)
    monkeypatch.setattr(config.database, "backend", "sqlite")
    monkeypatch.setattr(config.database, "path", db_path)
    MigrationManager(db_path).run()
    for name in ("alice", "bob"):
        _user(name)

    from api.application import app

    app.dependency_overrides.clear()
    # Status codes are the contract here, so a 500 must come back as a
    # response rather than re-raise inside the test.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _user(name: str) -> None:
    now = datetime.now(timezone.utc)
    execute(
        "INSERT INTO users (id, email, password_hash, created_at, updated_at) "
        "VALUES (?, ?, 'unused', ?, ?)",
        (f"user-{name}", f"{name}@example.com", now.isoformat(), now.isoformat()),
    )
    _open_session(name, f"session-{name}")


def _open_session(name: str, session_id: str) -> None:
    now = datetime.now(timezone.utc)
    execute(
        "INSERT INTO user_sessions (session_id, user_id, expires_at, created_at) "
        "VALUES (?, ?, ?, ?)",
        (session_id, f"user-{name}", (now + timedelta(hours=1)).isoformat(), now.isoformat()),
    )


def _graded_bet(name: str, outcome: str, profit_units: float) -> None:
    execute(
        """
        INSERT INTO user_bets (
            id, user_id, season, week, player_id, market, sportsbook, side,
            line, price, stake_units, outcome, profit_units, placed_at
        ) VALUES (?, ?, 2026, 3, 'BUF_receiver_a', 'receiving_yards', 'DraftKings',
                  'over', 64.5, -110, 1.0, ?, ?, ?)
        """,
        (
            f"bet-{name}-{outcome}",
            f"user-{name}",
            outcome,
            profit_units,
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def test_a_user_cannot_list_another_users_bets(client):
    assert client.post("/api/user/bets", json=BET, headers=ALICE).status_code == 200

    assert client.get("/api/user/bets", headers=BOB).json()["total"] == 0
    assert client.get("/api/user/bets", headers=ALICE).json()["total"] == 1


def test_betting_stats_count_only_the_callers_bets(client):
    _graded_bet("alice", "win", 0.91)
    _graded_bet("alice", "loss", -1.0)
    _graded_bet("bob", "win", 0.91)

    stats = client.get("/api/user/stats", headers=BOB).json()

    assert (stats["total_bets"], stats["wins"], stats["losses"]) == (1, 1, 0)


def test_a_recorded_bet_belongs_to_the_session_user_not_the_request_body(client):
    response = client.post("/api/user/bets", json={**BET, "user_id": "user-bob"}, headers=ALICE)

    assert response.status_code == 200
    assert fetchall("SELECT user_id FROM user_bets") == [("user-alice",)]


def test_logging_out_keeps_the_users_other_sessions_signed_in(client):
    _open_session("alice", "session-alice-laptop")
    laptop = {"Authorization": "Bearer session-alice-laptop"}

    assert client.post("/api/auth/logout", headers=ALICE).status_code == 200

    assert client.get("/api/auth/me", headers=ALICE).status_code == 401
    assert client.get("/api/auth/me", headers=laptop).status_code == 200


def test_registering_a_taken_email_is_a_client_error(client):
    response = client.post(
        "/api/auth/register",
        json={"email": "alice@example.com", "password": "strongpass1"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Email already registered"


@pytest.mark.parametrize("bankroll", ["-500", "inf", "nan"])
def test_an_invalid_bankroll_is_rejected_and_the_account_still_loads(client, bankroll):
    response = client.put(f"/api/user/bankroll?bankroll={bankroll}", headers=ALICE)

    assert response.status_code == 422
    assert client.get("/api/auth/me", headers=ALICE).status_code == 200
    assert fetchone("SELECT bankroll FROM users WHERE id = 'user-alice'")[0] == 1000.0


@pytest.mark.parametrize("bankroll", [0.0, 2500.5])
def test_a_zero_or_positive_bankroll_is_stored(client, bankroll):
    response = client.put(f"/api/user/bankroll?bankroll={bankroll}", headers=ALICE)

    assert response.status_code == 200
    assert fetchone("SELECT bankroll FROM users WHERE id = 'user-alice'")[0] == bankroll
