"""Tests for risk & correlation API endpoints (Feature 5)."""

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
    """Seed materialized_value_view with multiple bets for correlation detection."""
    # Player dim
    for pid, name, pos, team in [
        ("P001", "Patrick Mahomes", "QB", "KC"),
        ("P002", "Travis Kelce", "TE", "KC"),
        ("P003", "Derrick Henry", "RB", "BAL"),
    ]:
        execute(
            """
            INSERT INTO player_dim (player_id, player_name, position, team, last_season, last_week, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            params=(pid, name, pos, team, season, week),
        )

    # KC QB passing yards
    execute(
        """
        INSERT INTO materialized_value_view
            (season, week, player_id, event_id, team, market, sportsbook,
             line, price, mu, sigma, p_win, edge_percentage, expected_roi,
             kelly_fraction, stake, generated_at)
        VALUES (?, ?, 'P001', 'evt_kc_buf', 'KC', 'passing_yards', 'draftkings',
                280.5, -110, 310.0, 30.0, 0.65, 0.12, 0.10, 0.02, 20.0, datetime('now'))
        """,
        params=(season, week),
    )
    # KC TE receiving yards (same team, same game = correlated)
    execute(
        """
        INSERT INTO materialized_value_view
            (season, week, player_id, event_id, team, market, sportsbook,
             line, price, mu, sigma, p_win, edge_percentage, expected_roi,
             kelly_fraction, stake, generated_at)
        VALUES (?, ?, 'P002', 'evt_kc_buf', 'KC', 'receiving_yards', 'draftkings',
                55.5, -110, 65.0, 10.0, 0.60, 0.10, 0.08, 0.015, 15.0, datetime('now'))
        """,
        params=(season, week),
    )
    # BAL RB (different game)
    execute(
        """
        INSERT INTO materialized_value_view
            (season, week, player_id, event_id, team, market, sportsbook,
             line, price, mu, sigma, p_win, edge_percentage, expected_roi,
             kelly_fraction, stake, generated_at)
        VALUES (?, ?, 'P003', 'evt_bal_cin', 'BAL', 'rushing_yards', 'fanduel',
                95.5, -105, 110.0, 15.0, 0.62, 0.15, 0.11, 0.025, 25.0, datetime('now'))
        """,
        params=(season, week),
    )


class TestCorrelationAPI:
    def test_empty_returns_empty(self, client, db):
        resp = client.get("/api/analytics/correlation?season=9999&week=99")
        assert resp.status_code == 200
        data = resp.json()
        assert data["correlation_groups"] == []
        assert data["team_stacks"] == []

    def test_detects_correlations(self, client, db):
        _seed_bets(db)
        resp = client.get("/api/analytics/correlation?season=2025&week=22")
        assert resp.status_code == 200
        data = resp.json()

        # Should detect KC team stack (same_team or pos_corr)
        assert len(data["team_stacks"]) >= 1
        kc_stack = next((s for s in data["team_stacks"] if s["team"] == "KC"), None)
        assert kc_stack is not None
        assert kc_stack["count"] == 2

    def test_correlation_groups_have_players(self, client, db):
        _seed_bets(db)
        resp = client.get("/api/analytics/correlation?season=2025&week=22")
        data = resp.json()

        if data["correlation_groups"]:
            group = data["correlation_groups"][0]
            assert "group" in group
            assert "type" in group
            assert "players" in group
            assert "combined_stake" in group
            assert len(group["players"]) >= 2


class TestRiskSummaryAPI:
    def test_empty_returns_zeros(self, client, db):
        resp = client.get("/api/analytics/risk-summary?season=9999&week=99")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_stake"] == 0.0
        assert "guardrails" in data
        assert "warnings" in data

    def test_calculates_exposure(self, client, db):
        _seed_bets(db)
        resp = client.get("/api/analytics/risk-summary?season=2025&week=22")
        assert resp.status_code == 200
        data = resp.json()

        assert data["total_stake"] > 0
        assert len(data["team_exposure"]) >= 2  # KC and BAL

        # Find KC exposure
        kc_exp = next((e for e in data["team_exposure"] if e["team"] == "KC"), None)
        assert kc_exp is not None
        assert kc_exp["stake"] == 35.0  # 20 + 15

    def test_guardrails_present(self, client, db):
        _seed_bets(db)
        resp = client.get("/api/analytics/risk-summary?season=2025&week=22")
        data = resp.json()

        g = data["guardrails"]
        assert "max_team_exposure" in g
        assert "max_game_exposure" in g
        assert "max_player_exposure" in g
