"""Tests for explainability payload (Feature 4)."""

import pytest

from schema_migrations import MigrationManager
from utils.db import execute
from api.explainability import build_why_payload, build_why_payloads_batch


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


def _seed_projection(db, player_id="P001", season=2025, week=22, market="receiving_yards"):
    execute(
        """
        INSERT INTO weekly_projections
            (season, week, player_id, team, opponent, market, mu, sigma,
             context_sensitivity, pass_attempts_predicted, yards_per_attempt_predicted,
             model_version, featureset_hash, generated_at,
             volatility_score, target_share)
        VALUES (?, ?, ?, 'KC', 'DEN', ?, 85.0, 8.0, 0.3, 35.0, 7.5,
                'v2', 'abc123', datetime('now'), 0.65, 0.22)
        """,
        params=(season, week, player_id, market),
    )


def _seed_value_view(db, player_id="P001", season=2025, week=22, market="receiving_yards"):
    execute(
        """
        INSERT INTO materialized_value_view
            (season, week, player_id, event_id, market, sportsbook,
             line, price, mu, sigma, p_win, edge_percentage, expected_roi,
             kelly_fraction, stake, generated_at, confidence_score, confidence_tier)
        VALUES (?, ?, ?, 'evt1', ?, 'draftkings',
                75.5, -110, 85.0, 8.0, 0.65, 0.15, 0.12,
                0.02, 20.0, datetime('now'), 0.82, 'Premium')
        """,
        params=(season, week, player_id, market),
    )


def _seed_risk(db, player_id="P001", season=2025, week=22, market="receiving_yards"):
    execute(
        """
        INSERT INTO risk_assessments
            (season, week, player_id, market, sportsbook, event_id,
             correlation_group, exposure_warning, risk_adjusted_kelly, assessed_at)
        VALUES (?, ?, ?, ?, 'draftkings', 'evt1',
                'same_team_1', 'team_exposure(KC=15%)', 0.018, datetime('now'))
        """,
        params=(season, week, player_id, market),
    )


def _seed_agent_decision(db, player_id="P001", season=2025, week=22, market="receiving_yards"):
    execute(
        """
        INSERT INTO agent_decisions
            (season, week, player_id, market, decision, merged_confidence,
             votes, rationale, decided_at)
        VALUES (?, ?, ?, ?, 'APPROVED', 0.78,
                '{"model": "APPROVE", "risk": "APPROVE", "market": "APPROVE", "diagnostics": "REJECT"}',
                'Strong edge with positive model consensus', datetime('now'))
        """,
        params=(season, week, player_id, market),
    )


class TestBuildWhyPayload:
    def test_returns_all_sections(self, db):
        _seed_projection(db)
        _seed_value_view(db)
        _seed_risk(db)
        _seed_agent_decision(db)

        why = build_why_payload(2025, 22, "P001", "receiving_yards")

        assert "model" in why
        assert "volume" in why
        assert "volatility" in why
        assert "confidence" in why
        assert "risk" in why
        assert "agents" in why

    def test_model_section(self, db):
        _seed_projection(db)
        why = build_why_payload(2025, 22, "P001", "receiving_yards")
        assert why["model"]["mu"] == 85.0
        assert why["model"]["sigma"] == 8.0
        assert why["model"]["context_sensitivity"] == 0.3

    def test_risk_section(self, db):
        _seed_risk(db)
        why = build_why_payload(2025, 22, "P001", "receiving_yards")
        assert why["risk"]["correlation_group"] == "same_team_1"
        assert why["risk"]["exposure_warning"] is not None
        assert why["risk"]["risk_adjusted_kelly"] == 0.018

    def test_agents_section(self, db):
        _seed_agent_decision(db)
        why = build_why_payload(2025, 22, "P001", "receiving_yards")
        assert why["agents"]["decision"] == "APPROVED"
        assert why["agents"]["merged_confidence"] == 0.78
        assert isinstance(why["agents"]["votes"], dict)

    def test_handles_missing_data(self, db):
        why = build_why_payload(2025, 22, "MISSING", "missing_market")
        assert why["model"]["mu"] is None
        assert why["risk"]["correlation_group"] is None
        assert why["agents"]["decision"] is None

    def test_volume_section(self, db):
        _seed_projection(db)
        why = build_why_payload(2025, 22, "P001", "receiving_yards")
        assert why["volume"]["target_share"] == 0.22

    def test_volatility_section(self, db):
        _seed_projection(db)
        why = build_why_payload(2025, 22, "P001", "receiving_yards")
        assert why["volatility"]["score"] == 0.65


class TestBuildWhyBatch:
    def test_batch_returns_keyed_results(self, db):
        _seed_projection(db)
        _seed_value_view(db)

        bets = [{"player_id": "P001", "market": "receiving_yards"}]
        result = build_why_payloads_batch(2025, 22, bets)

        assert "P001:receiving_yards" in result
        assert "model" in result["P001:receiving_yards"]

    def test_batch_handles_missing(self, db):
        bets = [{"player_id": "MISSING", "market": "fake"}]
        result = build_why_payloads_batch(2025, 22, bets)

        assert "MISSING:fake" in result
        assert result["MISSING:fake"]["model"]["mu"] is None
