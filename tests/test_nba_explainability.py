"""Tests for the NBA explainability module (api/nba_explainability.py).

Validates each section of the why payload: model, recency, variance,
confidence, risk, agents.  Also tests batch keying logic.
"""

from __future__ import annotations

import json

import pytest

from schema_migrations import MigrationManager
from utils.db import execute


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_nba_explain.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    import config as cfg

    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")

    MigrationManager(db_path).run()
    return db_path


GAME_DATE = "2026-02-17"
PLAYER_ID = 1234
MARKET = "pts"


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _seed_projection(db: str) -> None:
    execute(
        "INSERT INTO nba_projections "
        "(player_id, player_name, team, season, game_date, game_id, "
        "market, projected_value, confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params=(PLAYER_ID, "Test Player", "LAL", 2025, GAME_DATE,
                "gid1", MARKET, 28.0, 0.85),
    )


def _seed_game_logs(db: str) -> None:
    """Seed 12 game logs for pts recency analysis."""
    for i in range(12):
        pts = 30 if i < 5 else 22  # Last 5 games: hot, older: cold
        execute(
            "INSERT INTO nba_player_game_logs "
            "(player_id, player_name, team_abbreviation, season, game_id, "
            "game_date, matchup, wl, min, pts, reb, ast, fg3m, fgm, fga, "
            "ftm, fta, stl, blk, tov, plus_minus) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params=(PLAYER_ID, "Test Player", "LAL", 2025, f"game{i}",
                    f"2026-02-{16 - i:02d}",
                    "LAL vs BOS", "W", 32.0, pts, 5, 3, 2,
                    8, 15, 4, 5, 1, 0, 2, 5.0),
        )


def _seed_value_view(db: str) -> None:
    execute(
        "INSERT INTO nba_materialized_value_view "
        "(season, game_date, player_id, player_name, team, event_id, market, "
        "sportsbook, line, over_price, under_price, mu, sigma, p_win, "
        "edge_percentage, expected_roi, kelly_fraction, confidence, generated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params=(2025, GAME_DATE, PLAYER_ID, "Test Player", "LAL", "evt1", MARKET,
                "draftkings", 25.5, -110, 110, 28.0, 3.0, 0.65, 12.0,
                0.10, 0.05, 0.85, "2026-02-17T00:00:00"),
    )


def _seed_risk_assessment(db: str) -> None:
    execute(
        "INSERT INTO nba_risk_assessments "
        "(game_date, player_id, market, sportsbook, event_id, "
        "correlation_group, exposure_warning, risk_adjusted_kelly, "
        "mean_drawdown, max_drawdown, p95_drawdown, assessed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params=(GAME_DATE, PLAYER_ID, MARKET, "draftkings", "evt1",
                "LAL_pts_fg3m", None, 0.04, 0.05, 0.12, 0.09,
                "2026-02-17T00:00:00"),
    )


def _seed_agent_decision(db: str) -> None:
    execute(
        "INSERT INTO nba_agent_decisions "
        "(game_date, player_id, market, decision, merged_confidence, "
        "votes, rationale, coordinator_override, agent_reports, decided_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params=(GAME_DATE, PLAYER_ID, MARKET, "APPROVED", 0.82,
                json.dumps({"APPROVE": 3, "REJECT": 1}),
                "3 of 4 agents approved", 0, "[]",
                "2026-02-17T00:00:00"),
    )


# ---------------------------------------------------------------------------
# Model section
# ---------------------------------------------------------------------------


class TestModelSection:
    def test_empty_returns_nulls(self, db):
        from api.nba_explainability import build_why_payload

        payload = build_why_payload(GAME_DATE, PLAYER_ID, MARKET)
        assert payload["model"]["projected_value"] is None
        assert payload["model"]["sigma"] is None
        assert payload["model"]["confidence"] is None

    def test_with_projection_returns_values(self, db):
        from api.nba_explainability import build_why_payload

        _seed_projection(db)
        payload = build_why_payload(GAME_DATE, PLAYER_ID, MARKET)
        assert payload["model"]["projected_value"] == 28.0
        assert payload["model"]["confidence"] == 0.85
        # sigma = max(28.0 * 0.20, 3.0) = 5.6
        assert payload["model"]["sigma"] == pytest.approx(5.6)


# ---------------------------------------------------------------------------
# Recency section
# ---------------------------------------------------------------------------


class TestRecencySection:
    def test_empty_returns_nulls(self, db):
        from api.nba_explainability import build_why_payload

        payload = build_why_payload(GAME_DATE, PLAYER_ID, MARKET)
        assert payload["recency"]["last5_avg"] is None
        assert payload["recency"]["last10_avg"] is None
        assert payload["recency"]["trend"] is None

    def test_with_game_logs_returns_averages(self, db):
        from api.nba_explainability import build_why_payload

        _seed_game_logs(db)
        payload = build_why_payload(GAME_DATE, PLAYER_ID, MARKET)
        # Last 5 games: pts=30 each => avg=30.0
        assert payload["recency"]["last5_avg"] == pytest.approx(30.0)
        # Last 10 games: 5*30 + 5*22 = 260/10 = 26.0
        assert payload["recency"]["last10_avg"] == pytest.approx(26.0)

    def test_trend_up_when_last5_exceeds_last10(self, db):
        from api.nba_explainability import build_why_payload

        _seed_game_logs(db)
        payload = build_why_payload(GAME_DATE, PLAYER_ID, MARKET)
        # last5=30, last10=26. 30 > 26*1.05=27.3 => trend="up"
        assert payload["recency"]["trend"] == "up"

    def test_invalid_market_returns_nulls(self, db):
        from api.nba_explainability import build_why_payload

        _seed_game_logs(db)
        payload = build_why_payload(GAME_DATE, PLAYER_ID, "unknown_market")
        assert payload["recency"]["last5_avg"] is None


# ---------------------------------------------------------------------------
# Variance section
# ---------------------------------------------------------------------------


class TestVarianceSection:
    def test_empty_returns_nulls(self, db):
        from api.nba_explainability import build_why_payload

        payload = build_why_payload(GAME_DATE, PLAYER_ID, MARKET)
        assert payload["variance"]["sigma"] is None
        assert payload["variance"]["cv"] is None

    def test_with_projection_returns_sigma_and_cv(self, db):
        from api.nba_explainability import build_why_payload

        _seed_projection(db)
        payload = build_why_payload(GAME_DATE, PLAYER_ID, MARKET)
        # sigma = max(28.0 * 0.20, 3.0) = 5.6
        assert payload["variance"]["sigma"] == pytest.approx(5.6)
        # cv = 5.6 / 28.0 = 0.2
        assert payload["variance"]["cv"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Confidence section
# ---------------------------------------------------------------------------


class TestConfidenceSection:
    def test_empty_returns_nulls(self, db):
        from api.nba_explainability import build_why_payload

        payload = build_why_payload(GAME_DATE, PLAYER_ID, MARKET)
        assert payload["confidence"]["p_win"] is None
        assert payload["confidence"]["edge_percentage"] is None
        assert payload["confidence"]["expected_roi"] is None
        assert payload["confidence"]["kelly_fraction"] is None

    def test_with_value_view_returns_values(self, db):
        from api.nba_explainability import build_why_payload

        _seed_value_view(db)
        payload = build_why_payload(GAME_DATE, PLAYER_ID, MARKET)
        assert payload["confidence"]["p_win"] == pytest.approx(0.65)
        assert payload["confidence"]["edge_percentage"] == pytest.approx(12.0)
        assert payload["confidence"]["expected_roi"] == pytest.approx(0.10)
        assert payload["confidence"]["kelly_fraction"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Risk section
# ---------------------------------------------------------------------------


class TestRiskSection:
    def test_empty_returns_nulls(self, db):
        from api.nba_explainability import build_why_payload

        payload = build_why_payload(GAME_DATE, PLAYER_ID, MARKET)
        assert payload["risk"]["correlation_group"] is None
        assert payload["risk"]["exposure_warning"] is None
        assert payload["risk"]["risk_adjusted_kelly"] is None

    def test_with_risk_data_returns_values(self, db):
        from api.nba_explainability import build_why_payload

        _seed_risk_assessment(db)
        payload = build_why_payload(GAME_DATE, PLAYER_ID, MARKET)
        assert payload["risk"]["correlation_group"] == "LAL_pts_fg3m"
        assert payload["risk"]["exposure_warning"] is None
        assert payload["risk"]["risk_adjusted_kelly"] == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# Agents section
# ---------------------------------------------------------------------------


class TestAgentsSection:
    def test_empty_returns_nulls(self, db):
        from api.nba_explainability import build_why_payload

        payload = build_why_payload(GAME_DATE, PLAYER_ID, MARKET)
        assert payload["agents"]["decision"] is None
        assert payload["agents"]["merged_confidence"] is None
        assert payload["agents"]["votes"] is None
        assert payload["agents"]["top_rationale"] is None

    def test_with_agent_data_returns_values(self, db):
        from api.nba_explainability import build_why_payload

        _seed_agent_decision(db)
        payload = build_why_payload(GAME_DATE, PLAYER_ID, MARKET)
        assert payload["agents"]["decision"] == "APPROVED"
        assert payload["agents"]["merged_confidence"] == pytest.approx(0.82)
        assert payload["agents"]["votes"] == {"APPROVE": 3, "REJECT": 1}
        assert payload["agents"]["top_rationale"] == "3 of 4 agents approved"


# ---------------------------------------------------------------------------
# Batch keying
# ---------------------------------------------------------------------------


class TestBatchPayload:
    def test_batch_returns_correct_keys(self, db):
        from api.nba_explainability import build_why_payloads_batch

        _seed_projection(db)
        bets = [
            {"player_id": PLAYER_ID, "market": MARKET},
            {"player_id": 9999, "market": "reb"},
        ]
        result = build_why_payloads_batch(GAME_DATE, bets)
        assert f"{PLAYER_ID}:{MARKET}" in result
        assert "9999:reb" in result

    def test_batch_payload_has_all_sections(self, db):
        from api.nba_explainability import build_why_payloads_batch

        _seed_projection(db)
        _seed_value_view(db)
        _seed_risk_assessment(db)
        _seed_agent_decision(db)

        bets = [{"player_id": PLAYER_ID, "market": MARKET}]
        result = build_why_payloads_batch(GAME_DATE, bets)
        payload = result[f"{PLAYER_ID}:{MARKET}"]

        assert "model" in payload
        assert "recency" in payload
        assert "variance" in payload
        assert "confidence" in payload
        assert "risk" in payload
        assert "agents" in payload

    def test_batch_handles_missing_player_gracefully(self, db):
        from api.nba_explainability import build_why_payloads_batch

        bets = [{"player_id": None, "market": "pts"}]
        result = build_why_payloads_batch(GAME_DATE, bets)
        assert "None:pts" in result
        # Should still return structure, not error
        payload = result["None:pts"]
        assert "model" in payload

    def test_batch_empty_list(self, db):
        from api.nba_explainability import build_why_payloads_batch

        result = build_why_payloads_batch(GAME_DATE, [])
        assert result == {}


# ---------------------------------------------------------------------------
# Full payload (all sections populated)
# ---------------------------------------------------------------------------


class TestFullPayload:
    def test_full_payload_has_six_sections(self, db):
        from api.nba_explainability import build_why_payload

        _seed_projection(db)
        _seed_game_logs(db)
        _seed_value_view(db)
        _seed_risk_assessment(db)
        _seed_agent_decision(db)

        payload = build_why_payload(GAME_DATE, PLAYER_ID, MARKET)
        expected_sections = {"model", "recency", "variance", "confidence", "risk", "agents"}
        assert expected_sections == set(payload.keys())

    def test_full_payload_model_and_variance_sigma_match(self, db):
        from api.nba_explainability import build_why_payload

        _seed_projection(db)
        payload = build_why_payload(GAME_DATE, PLAYER_ID, MARKET)
        # Both model and variance sections should compute same sigma
        assert payload["model"]["sigma"] == pytest.approx(payload["variance"]["sigma"])
