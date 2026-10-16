"""Regression tests for api/server.py input bounds, JSON safety and fail-closed review."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Imported at collection: api.application adds middleware at import time, which
# Starlette refuses once another test has started the shared api.server app.
from api.application import app
from api.cache import value_bets_cache
from config import config
from schema_migrations import MigrationManager
from utils.db import execute, fetchall, fetchone

AUTH = {"Authorization": "Bearer session-hardening"}

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
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "api-hardening.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)
    monkeypatch.setattr(config.database, "backend", "sqlite")
    monkeypatch.setattr(config.database, "path", db_path)
    monkeypatch.setattr(config.api, "demo_mode", True)
    MigrationManager(db_path).run()
    value_bets_cache.invalidate_all()

    now = datetime.now(timezone.utc)
    execute(
        "INSERT INTO users (id, email, password_hash, created_at, updated_at) "
        "VALUES ('user-h', 'h@example.com', 'unused', ?, ?)",
        (now.isoformat(), now.isoformat()),
    )
    execute(
        "INSERT INTO user_sessions (session_id, user_id, expires_at, created_at) "
        "VALUES ('session-hardening', 'user-h', ?, ?)",
        ((now + timedelta(hours=1)).isoformat(), now.isoformat()),
    )
    yield db_path
    value_bets_cache.invalidate_all()


@pytest.fixture()
def client(db):
    from api.pipeline_router import require_pipeline_operator, require_pipeline_reader

    app.dependency_overrides[require_pipeline_reader] = lambda: "test-reader"
    app.dependency_overrides[require_pipeline_operator] = lambda: "test-operator"
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _seed_run(run_id: str = "run-1", season: int = 2026, week: int = 3) -> None:
    execute(
        "INSERT INTO pipeline_runs (run_id, season, week, status, stages_requested, "
        "stages_completed, started_at) VALUES (?, ?, ?, 'completed', 6, 6, ?)",
        (run_id, season, week, datetime.now(timezone.utc).isoformat()),
    )


def _seed_value_row(player_id: str, run_id: str | None, season: int = 2026, week: int = 3) -> None:
    execute(
        """
        INSERT INTO materialized_value_view
            (season, week, player_id, event_id, team, market, sportsbook,
             line, price, mu, sigma, p_win, edge_percentage, expected_roi,
             kelly_fraction, stake, generated_at, published_run_id)
        VALUES (?, ?, ?, 'evt1', 'BUF', 'receiving_yards', 'draftkings',
                64.5, -110, 70.0, 10.0, 0.6, 0.1, 0.1, 0.02, 10.0, ?, ?)
        """,
        (season, week, player_id, datetime.now(timezone.utc).isoformat(), run_id),
    )


# ── NaN in JSON ────────────────────────────────────────────────────────


def test_user_bets_with_partially_graded_rows_serialize(client):
    assert client.post("/api/user/bets", json=BET, headers=AUTH).status_code == 200
    execute(
        """
        INSERT INTO user_bets (
            id, user_id, season, week, player_id, market, sportsbook, side,
            line, price, stake_units, stake_dollars, actual_result, outcome,
            profit_units, placed_at, graded_at
        ) VALUES ('graded', 'user-h', 2026, 3, 'BUF_receiver_a', 'receiving_yards',
                  'DraftKings', 'over', 64.5, -110, 1.0, 25.0, 71.0, 'win', 0.91, ?, ?)
        """,
        (datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat()),
    )

    response = client.get("/api/user/bets", headers=AUTH)

    assert response.status_code == 200, response.text
    bets = {bet["bet_id"]: bet for bet in response.json()["bets"]}
    graded = bets.pop("graded")
    (ungraded,) = bets.values()
    assert graded["actual_result"] == 71.0
    assert graded["profit_units"] == 0.91
    assert graded["stake_dollars"] == 25.0
    assert ungraded["actual_result"] is None
    assert ungraded["profit_units"] is None
    assert ungraded["stake_dollars"] is None
    assert isinstance(graded["price"], int)


def test_records_helper_maps_nan_to_none():
    from api.server import _records

    frame = pd.DataFrame({"a": [1.0, float("nan")], "b": [1, 2], "c": ["x", None]})

    assert _records(frame) == [{"a": 1.0, "b": 1, "c": "x"}, {"a": None, "b": 2, "c": None}]
    assert _records(pd.DataFrame()) == []


# ── Request validation ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "override",
    [
        {"stake_units": -3},
        {"stake_units": 0},
        {"stake_units": 1_000_000},
        {"price": 0},
        {"price": -99},
        {"price": 99},
        {"week": 0},
        {"week": 23},
        {"season": 1900},
        {"line": -1},
    ],
)
def test_record_bet_rejects_out_of_range_values(client, override):
    response = client.post("/api/user/bets", json={**BET, **override}, headers=AUTH)

    assert response.status_code == 422


def test_record_bet_rejects_nan_stake(client):
    body = json.dumps({**BET, "stake_units": float("nan")})
    response = client.post(
        "/api/user/bets",
        content=body,
        headers={**AUTH, "Content-Type": "application/json"},
    )

    assert response.status_code == 422


@pytest.mark.parametrize("price", [-100, 100, -250, 150])
def test_record_bet_accepts_american_odds(client, price):
    response = client.post("/api/user/bets", json={**BET, "price": price}, headers=AUTH)

    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    "override",
    [
        {"default_min_edge": -0.1},
        {"default_kelly_fraction": 5},
        {"default_max_stake": 2},
        {"preferred_markets": "x" * 1001},
    ],
)
def test_preferences_reject_out_of_range_values(client, override):
    response = client.put("/api/user/preferences", json=override, headers=AUTH)

    assert response.status_code == 422


def test_edge_distribution_bins_are_bounded(client):
    assert client.get("/api/analytics/edge-distribution?season=2026&week=3&bins=1000000").status_code == 422
    assert client.get("/api/analytics/edge-distribution?season=2026&week=3&bins=0").status_code == 422
    assert client.get("/api/analytics/edge-distribution?season=2026&week=3&bins=100").status_code == 200


# ── Exception text never reaches the client ───────────────────────────


def test_explainability_error_hides_exception_text(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("/srv/secret/path.db is locked")

    monkeypatch.setattr("api.explainability.build_why_payload", boom)

    response = client.get("/api/explain/p1/receiving_yards?season=2026&week=3")

    assert response.status_code == 500
    assert "secret" not in response.text


def test_value_bets_error_hides_exception_text(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("no such table: secret_internal_table")

    monkeypatch.setattr("api.server.read_dataframe", boom)

    response = client.get("/api/value-bets?season=2026&week=3")

    assert response.status_code == 500
    assert "secret" not in response.text


def test_why_batch_payload_hides_exception_text(monkeypatch):
    import api.explainability as explainability

    def boom(*args, **kwargs):
        raise RuntimeError("secret detail")

    monkeypatch.setattr(explainability, "build_why_payload", boom)

    payloads = explainability.build_why_payloads_batch(2026, 3, [{"player_id": "p", "market": "m"}])

    assert "secret" not in json.dumps(payloads)


# ── Value bets: best-line grouping and run pinning ────────────────────


def test_best_line_only_keeps_rows_with_null_side(client, monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "player_id": "legacy",
                "player_name": "Legacy",
                "position": "WR",
                "market": "receiving_yards",
                "sportsbook": "draftkings",
                "line": 64.5,
                "price": -110,
                "side": None,
                "mu": 70.0,
                "sigma": 10.0,
                "p_win": 0.6,
                "edge_percentage": 0.1,
                "expected_roi": 0.1,
                "kelly_fraction": 0.02,
                "stake": 10.0,
            }
        ]
    )
    monkeypatch.setattr("api.server.read_dataframe", lambda *args, **kwargs: frame.copy())

    response = client.get("/api/value-bets?season=2026&week=3&best_line_only=true")

    assert response.status_code == 200, response.text
    assert [bet["player_id"] for bet in response.json()["bets"]] == ["legacy"]


def test_value_bets_pinned_run_id_returns_only_that_run(client):
    _seed_value_row("from-run-a", "run-a")
    _seed_value_row("from-run-b", "run-b")

    pinned_a = client.get("/api/value-bets?season=2026&week=3&run_id=run-a").json()
    pinned_b = client.get("/api/value-bets?season=2026&week=3&run_id=run-b").json()

    assert [bet["player_id"] for bet in pinned_a["bets"]] == ["from-run-a"]
    assert [bet["player_id"] for bet in pinned_b["bets"]] == ["from-run-b"]


# ── Agent review fails closed ──────────────────────────────────────────


def _report(run_id: str = "run-1") -> dict:
    row = fetchone("SELECT report_json FROM pipeline_runs WHERE run_id = ?", (run_id,))
    return json.loads(row[0]) if row and row[0] else {}


def test_failing_agent_review_writes_no_approved_rows(client, monkeypatch):
    import agents.coordinator
    import api.server as server

    _seed_run()
    _seed_value_row("p1", "run-1")

    def boom(*args, **kwargs):
        raise RuntimeError("agent crashed")

    monkeypatch.setattr(agents.coordinator, "run_all_agents", boom)

    server._run_agent_review_background("run-1", 2026, 3)

    assert fetchall("SELECT decision FROM agent_decisions") == []
    report = _report()
    assert "agent_review_at" not in report
    assert report["agent_review_error"] == "Agent review failed"

    status = client.get("/api/run/run-1/review-status?season=2026&week=3").json()
    assert status["reviewed"] is False
    assert status["review_error"] == "Agent review failed"
    assert status["decision_count"] == 0


def test_agent_review_with_no_decisions_is_a_failure(monkeypatch, db):
    import agents.coordinator
    import api.server as server

    _seed_run()
    monkeypatch.setattr(agents.coordinator, "run_all_agents", lambda season, week: [])

    server._run_agent_review_background("run-1", 2026, 3)

    report = _report()
    assert "agent_review_at" not in report
    assert report["agent_review_error"] == "Agent review produced no decisions"
    assert fetchall("SELECT 1 FROM agent_decisions") == []


def test_successful_agent_review_stamps_review_time(monkeypatch, db):
    import agents.coordinator
    import api.server as server

    _seed_run()
    calls = []

    def fake_run_all_agents(season, week):
        calls.append((season, week))
        return [{"player_id": "p1", "market": "receiving_yards", "decision": "REJECTED"}]

    monkeypatch.setattr(agents.coordinator, "run_all_agents", fake_run_all_agents)
    execute(
        "UPDATE pipeline_runs SET report_json = ? WHERE run_id = 'run-1'",
        (json.dumps({"agent_review_error": "earlier failure", "kept": 1}),),
    )

    server._run_agent_review_background("run-1", 2026, 3)

    assert calls == [(2026, 3)]
    report = _report()
    assert report["agent_review_at"]
    assert report["agent_review_error"] is None
    assert report["kept"] == 1
