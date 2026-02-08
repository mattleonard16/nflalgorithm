"""Contract tests for API shape enforcement (Feature 2).

Ensures the API never breaks its contract with the frontend by
validating response schemas, field presence, and enum values.
"""

import pytest

from schema_migrations import MigrationManager
from utils.db import execute


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
    from fastapi.testclient import TestClient
    from api.server import app
    return TestClient(app)


def _seed_value_bet(db, player_id="P001", season=2025, week=22, market="receiving_yards",
                    sportsbook="draftkings", edge=0.15, confidence_tier="Premium"):
    """Seed a materialized_value_view row and player_dim row."""
    execute(
        """
        INSERT INTO player_dim (player_id, player_name, position, team, last_season, last_week, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        params=(player_id, "Test Player", "WR", "KC", season, week),
    )
    execute(
        """
        INSERT INTO materialized_value_view
            (season, week, player_id, event_id, team, market, sportsbook,
             line, price, mu, sigma, p_win, edge_percentage, expected_roi,
             kelly_fraction, stake, generated_at, confidence_score, confidence_tier)
        VALUES (?, ?, ?, 'evt1', 'KC', ?, ?,
                75.5, -110, 85.0, 8.0, 0.65, ?, 0.12,
                0.02, 20.0, datetime('now'), 0.82, ?)
        """,
        params=(season, week, player_id, market, sportsbook, edge, confidence_tier),
    )


class TestOpenAPIContract:
    def test_openapi_contains_expected_paths(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        openapi = resp.json()
        paths = openapi.get("paths", {})

        expected_paths = [
            "/api/value-bets",
            "/api/meta",
            "/api/performance",
            "/api/run/{run_id}",
            "/api/explain/{player_id}/{market}",
            "/api/analytics/correlation",
            "/api/analytics/risk-summary",
            "/api/export/csv",
            "/api/export/bundle",
            "/api/run/{run_id}/review",
            "/api/run/{run_id}/review-status",
        ]
        for path in expected_paths:
            assert path in paths, f"Missing path: {path}"


class TestValueBetsContract:
    def test_response_has_required_fields(self, client, db):
        _seed_value_bet(db)
        resp = client.get("/api/value-bets?season=2025&week=22")
        assert resp.status_code == 200
        data = resp.json()

        assert "bets" in data
        assert "total" in data
        assert "filters" in data
        assert isinstance(data["bets"], list)
        assert len(data["bets"]) > 0

        bet = data["bets"][0]
        required_fields = [
            "player_id", "player_name", "position", "market",
            "sportsbook", "line", "price", "mu", "sigma",
            "p_win", "edge_percentage", "expected_roi",
            "kelly_fraction", "stake",
        ]
        for field in required_fields:
            assert field in bet, f"Missing field: {field}"

    def test_player_name_and_position_present(self, client, db):
        _seed_value_bet(db)
        resp = client.get("/api/value-bets?season=2025&week=22")
        bet = resp.json()["bets"][0]
        assert bet["player_name"] == "Test Player"
        assert bet["position"] == "WR"

    def test_confidence_tier_enum_values(self, client, db):
        valid_tiers = {"Premium", "Strong", "Marginal", "Pass"}

        for tier in valid_tiers:
            _seed_value_bet(db, player_id=f"P_{tier}", confidence_tier=tier)

        resp = client.get("/api/value-bets?season=2025&week=22&min_edge=0")
        data = resp.json()

        tiers_found = {b["confidence_tier"] for b in data["bets"] if b["confidence_tier"]}
        assert tiers_found.issubset(valid_tiers)

    def test_empty_response_shape(self, client, db):
        resp = client.get("/api/value-bets?season=9999&week=99")
        assert resp.status_code == 200
        data = resp.json()
        assert data["bets"] == []
        assert data["total"] == 0
        assert "filters" in data

    def test_include_why_param(self, client, db):
        _seed_value_bet(db)
        resp = client.get("/api/value-bets?season=2025&week=22&include_why=true")
        assert resp.status_code == 200
        data = resp.json()
        # why should be present (may be null or dict)
        bet = data["bets"][0]
        assert "why" in bet


class TestPipelineRunContract:
    def test_post_returns_run_fields(self, client):
        resp = client.post("/api/run?season=2025&week=22&skip_ingest=true&skip_odds=true")
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert "status" in data
        assert "started_at" in data
        assert "season" in data
        assert "week" in data
