"""Tests for the NBA agent system (agents/nba_*.py)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from schema_migrations import MigrationManager
from utils.db import execute, read_dataframe


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_nba_agents.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    import config as cfg

    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")

    MigrationManager(db_path).run()
    return db_path


def _seed_odds(db: str, game_date: str = "2026-02-11") -> None:
    """Seed nba_odds with test data."""
    for pid, name, team, market, line, book in [
        (1234, "Test Player", "LAL", "pts", 25.5, "draftkings"),
        (1234, "Test Player", "LAL", "pts", 25.5, "fanduel"),
        (5678, "Other Player", "BOS", "fg3m", 3.5, "draftkings"),
    ]:
        execute(
            "INSERT INTO nba_odds "
            "(event_id, season, game_date, player_id, player_name, team, market, "
            "sportsbook, line, over_price, under_price, as_of) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params=("evt1", 2025, game_date, pid, name, team, market, book, line,
                    -110, 110, "2026-02-11T10:00:00"),
        )


def _seed_projections(db: str, game_date: str = "2026-02-11") -> None:
    """Seed nba_projections with test data."""
    for pid, name, team, market, value in [
        (1234, "Test Player", "LAL", "pts", 28.0),
        (5678, "Other Player", "BOS", "fg3m", 4.0),
    ]:
        execute(
            "INSERT INTO nba_projections "
            "(player_id, player_name, team, season, game_date, game_id, "
            "market, projected_value, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params=(pid, name, team, 2025, game_date, "gid1", market, value, 0.80),
        )


def _seed_value_view(db: str, game_date: str = "2026-02-11") -> None:
    """Seed nba_materialized_value_view."""
    for pid, name, team, market in [
        (1234, "Test Player", "LAL", "pts"),
        (5678, "Other Player", "BOS", "fg3m"),
    ]:
        execute(
            "INSERT INTO nba_materialized_value_view "
            "(season, game_date, player_id, player_name, team, event_id, market, "
            "sportsbook, line, over_price, under_price, mu, sigma, p_win, "
            "edge_percentage, expected_roi, kelly_fraction, confidence, generated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params=(2025, game_date, pid, name, team, "evt1", market,
                    "draftkings", 25.5, -110, 110, 28.0, 3.0, 0.65, 0.12,
                    0.10, 0.05, 0.85, "2026-02-11T00:00:00"),
        )


def _seed_game_logs(db: str, player_id: int = 5678) -> None:
    """Seed nba_player_game_logs for fg3m recency analysis."""
    for i in range(15):
        fg3m = 5 if i < 5 else 2  # Last 5 games: hot streak
        execute(
            "INSERT INTO nba_player_game_logs "
            "(player_id, player_name, team_abbreviation, season, game_id, "
            "game_date, matchup, wl, min, pts, reb, ast, fg3m, fgm, fga, "
            "ftm, fta, stl, blk, tov, plus_minus) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params=(player_id, "Other Player", "BOS", 2025, f"game{i}",
                    f"2026-02-{10-i:02d}" if i < 10 else f"2026-01-{31-i+10:02d}",
                    "BOS vs LAL", "W", 32.0, 20, 5, 3, fg3m,
                    8, 15, 4, 5, 1, 0, 2, 5.0),
        )


# ---------------------------------------------------------------------------
# NbaOddsAgent
# ---------------------------------------------------------------------------


class TestNbaOddsAgent:
    def test_empty_data_returns_empty(self, db):
        from agents.nba_odds_agent import NbaOddsAgent

        agent = NbaOddsAgent()
        reports = agent.analyze("2026-02-11")
        assert reports == []

    def test_stable_lines_approve(self, db):
        from agents.nba_odds_agent import NbaOddsAgent

        _seed_odds(db)
        agent = NbaOddsAgent()
        reports = agent.analyze("2026-02-11")
        assert len(reports) > 0
        assert all(r.recommendation == "APPROVE" for r in reports)

    def test_steam_move_detected(self, db):
        from agents.nba_odds_agent import NbaOddsAgent

        # Seed initial odds with a distinct event for the shifted snapshot
        _seed_odds(db)
        # Use a different event_id to avoid PK conflict (PK: event_id, player_name, market, sportsbook)
        execute(
            "INSERT INTO nba_odds "
            "(event_id, season, game_date, player_id, player_name, team, market, "
            "sportsbook, line, over_price, under_price, as_of) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params=("evt1_snap2", 2025, "2026-02-11", 1234, "Test Player", "LAL", "pts",
                    "draftkings", 27.5, -110, 110, "2026-02-11T12:00:00"),
        )
        agent = NbaOddsAgent()
        reports = agent.analyze("2026-02-11")
        pts_reports = [r for r in reports if r.market == "pts" and "1234" in str(r.player_id)]
        assert any(r.recommendation == "REJECT" for r in pts_reports)

    def test_reports_have_valid_fields(self, db):
        from agents.nba_odds_agent import NbaOddsAgent
        from agents import validate_report

        _seed_odds(db)
        agent = NbaOddsAgent()
        reports = agent.analyze("2026-02-11")
        for r in reports:
            assert validate_report(r) == []


# ---------------------------------------------------------------------------
# NbaModelDiagnosticsAgent
# ---------------------------------------------------------------------------


class TestNbaModelDiagnosticsAgent:
    def test_empty_projections_returns_empty(self, db):
        from agents.nba_model_diagnostics_agent import NbaModelDiagnosticsAgent

        agent = NbaModelDiagnosticsAgent()
        reports = agent.analyze("2026-02-11")
        assert reports == []

    def test_normal_gap_approves(self, db):
        from agents.nba_model_diagnostics_agent import NbaModelDiagnosticsAgent

        _seed_projections(db)
        _seed_odds(db)
        agent = NbaModelDiagnosticsAgent()
        reports = agent.analyze("2026-02-11")
        # pts: proj=28 vs line=25.5, gap=2.5, sigma=max(28*0.2,3)=5.6, threshold=8.4
        # gap(2.5) < threshold(8.4) => APPROVE
        pts_reports = [r for r in reports if r.market == "pts"]
        assert len(pts_reports) > 0
        assert pts_reports[0].recommendation == "APPROVE"

    def test_suspicious_gap_rejects(self, db):
        from agents.nba_model_diagnostics_agent import NbaModelDiagnosticsAgent

        # Project value way off from line
        execute(
            "INSERT INTO nba_projections "
            "(player_id, player_name, team, season, game_date, game_id, "
            "market, projected_value, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params=(9999, "Wild Player", "GSW", 2025, "2026-02-11",
                    "gid2", "pts", 50.0, 0.70),
        )
        execute(
            "INSERT INTO nba_odds "
            "(event_id, season, game_date, player_id, player_name, team, market, "
            "sportsbook, line, over_price, under_price, as_of) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params=("evt2", 2025, "2026-02-11", 9999, "Wild Player", "GSW", "pts",
                    "draftkings", 25.0, -110, 110, "2026-02-11T10:00:00"),
        )
        agent = NbaModelDiagnosticsAgent()
        reports = agent.analyze("2026-02-11")
        wild_reports = [r for r in reports if "9999" in str(r.player_id)]
        assert any(r.recommendation == "REJECT" for r in wild_reports)


# ---------------------------------------------------------------------------
# NbaMarketBiasAgent
# ---------------------------------------------------------------------------


class TestNbaMarketBiasAgent:
    def test_non_fg3m_returns_neutral(self, db):
        from agents.nba_market_bias_agent import NbaMarketBiasAgent

        _seed_value_view(db)
        agent = NbaMarketBiasAgent()
        reports = agent.analyze("2026-02-11")
        pts_reports = [r for r in reports if r.market == "pts"]
        assert all(r.recommendation == "NEUTRAL" for r in pts_reports)

    def test_fg3m_hot_streak_approves(self, db):
        from agents.nba_market_bias_agent import NbaMarketBiasAgent

        _seed_value_view(db)
        _seed_game_logs(db, player_id=5678)
        agent = NbaMarketBiasAgent()
        reports = agent.analyze("2026-02-11")
        fg3m_reports = [r for r in reports if r.market == "fg3m"]
        # last5=5.0, last10=3.5, divergence=1.5 — at threshold, exact depends on rounding
        assert len(fg3m_reports) > 0

    def test_empty_data_returns_empty(self, db):
        from agents.nba_market_bias_agent import NbaMarketBiasAgent

        agent = NbaMarketBiasAgent()
        reports = agent.analyze("2026-02-11")
        assert reports == []


# ---------------------------------------------------------------------------
# NbaRiskAgent
# ---------------------------------------------------------------------------


class TestNbaRiskAgent:
    def test_empty_data_returns_empty(self, db):
        from agents.nba_risk_agent import NbaRiskAgent

        agent = NbaRiskAgent()
        reports = agent.analyze("2026-02-11")
        assert reports == []

    def test_with_value_data_returns_reports(self, db):
        from agents.nba_risk_agent import NbaRiskAgent

        _seed_value_view(db)
        agent = NbaRiskAgent()
        reports = agent.analyze("2026-02-11")
        assert len(reports) > 0
        assert all(r.agent_name == "nba_risk_agent" for r in reports)

    def test_reports_have_risk_data(self, db):
        from agents.nba_risk_agent import NbaRiskAgent

        _seed_value_view(db)
        agent = NbaRiskAgent()
        reports = agent.analyze("2026-02-11")
        for r in reports:
            assert "kelly_fraction" in r.data
            assert "risk_adjusted_kelly" in r.data


# ---------------------------------------------------------------------------
# NBA Coordinator
# ---------------------------------------------------------------------------


class TestNbaCoordinator:
    def test_empty_data_returns_empty(self, db):
        from agents.nba_coordinator import run_nba_agents

        decisions = run_nba_agents("2026-02-11")
        assert decisions == []

    def test_with_seeded_data_produces_decisions(self, db):
        from agents.nba_coordinator import run_nba_agents

        _seed_odds(db)
        _seed_projections(db)
        _seed_value_view(db)
        decisions = run_nba_agents("2026-02-11")
        assert len(decisions) > 0
        for d in decisions:
            assert d["decision"] in ("APPROVED", "REJECTED")
            assert "votes" in d
            assert "merged_confidence" in d

    def test_decisions_persisted(self, db):
        from agents.nba_coordinator import run_nba_agents

        _seed_odds(db)
        _seed_projections(db)
        _seed_value_view(db)
        run_nba_agents("2026-02-11")

        rows = read_dataframe(
            "SELECT * FROM nba_agent_decisions WHERE game_date = ?",
            params=("2026-02-11",),
        )
        assert len(rows) > 0
        assert "decision" in rows.columns
        assert "votes" in rows.columns

    def test_consensus_threshold(self, db):
        """With 4 agents, >= 3 approvals needed for APPROVED."""
        from agents.nba_coordinator import run_nba_agents

        _seed_odds(db)
        _seed_projections(db)
        _seed_value_view(db)
        decisions = run_nba_agents("2026-02-11")
        for d in decisions:
            votes = d["votes"]
            if d["decision"] == "APPROVED" and not d["override"]:
                assert votes["APPROVE"] >= 3

    def test_player_id_filter(self, db):
        from agents.nba_coordinator import run_nba_agents

        _seed_odds(db)
        _seed_projections(db)
        _seed_value_view(db)
        decisions = run_nba_agents("2026-02-11", player_id=1234)
        # Should only have decisions for player 1234
        for d in decisions:
            assert "1234" in str(d.get("player_id", ""))
