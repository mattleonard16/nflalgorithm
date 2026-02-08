"""Tests for the agent orchestration system.

Covers:
- AgentReport dataclass and validation
- Individual agent report structure
- Consensus logic and conflict resolution
- Coordinator output format
- Edge cases (empty data, missing agents, all-reject)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agents import AgentReport, VALID_RECOMMENDATIONS, validate_report
from agents.base_agent import BaseAgent
from agents.coordinator import (
    CONSENSUS_THRESHOLD,
    _group_reports,
    _resolve_consensus,
    run_all_agents,
)
from agents.odds_agent import (
    OddsAgent,
    _best_prices,
    _detect_steam_moves,
)
from agents.model_diagnostics_agent import (
    ModelDiagnosticsAgent,
    _flag_suspicious,
)


# ======================================================================
# AgentReport dataclass
# ======================================================================


class TestAgentReport:
    def test_creation(self):
        report = AgentReport(
            agent_name="test_agent",
            recommendation="APPROVE",
            confidence=0.85,
            rationale="Looks good",
        )
        assert report.agent_name == "test_agent"
        assert report.recommendation == "APPROVE"
        assert report.confidence == 0.85
        assert report.rationale == "Looks good"
        assert report.data == {}
        assert report.player_id is None
        assert report.market is None

    def test_with_all_fields(self):
        report = AgentReport(
            agent_name="odds_agent",
            recommendation="REJECT",
            confidence=0.90,
            rationale="Steam move detected",
            data={"steam_move": True},
            player_id="P001",
            market="rushing_yards",
        )
        assert report.player_id == "P001"
        assert report.market == "rushing_yards"
        assert report.data["steam_move"] is True

    def test_immutability(self):
        report = AgentReport(
            agent_name="test",
            recommendation="APPROVE",
            confidence=0.5,
            rationale="ok",
        )
        with pytest.raises(AttributeError):
            report.confidence = 0.9

    def test_generated_at_auto_set(self):
        report = AgentReport(
            agent_name="test",
            recommendation="APPROVE",
            confidence=0.5,
            rationale="ok",
        )
        assert report.generated_at is not None
        assert len(report.generated_at) > 10


# ======================================================================
# Report validation
# ======================================================================


class TestValidateReport:
    def test_valid_report(self):
        report = AgentReport(
            agent_name="test",
            recommendation="APPROVE",
            confidence=0.5,
            rationale="ok",
        )
        assert validate_report(report) == []

    def test_invalid_recommendation(self):
        report = AgentReport(
            agent_name="test",
            recommendation="MAYBE",
            confidence=0.5,
            rationale="ok",
        )
        errors = validate_report(report)
        assert len(errors) == 1
        assert "MAYBE" in errors[0]

    def test_confidence_too_high(self):
        report = AgentReport(
            agent_name="test",
            recommendation="APPROVE",
            confidence=1.5,
            rationale="ok",
        )
        errors = validate_report(report)
        assert len(errors) == 1
        assert "1.5" in errors[0]

    def test_confidence_too_low(self):
        report = AgentReport(
            agent_name="test",
            recommendation="APPROVE",
            confidence=-0.1,
            rationale="ok",
        )
        errors = validate_report(report)
        assert len(errors) == 1

    def test_empty_agent_name(self):
        report = AgentReport(
            agent_name="",
            recommendation="APPROVE",
            confidence=0.5,
            rationale="ok",
        )
        errors = validate_report(report)
        assert len(errors) == 1
        assert "agent_name" in errors[0]

    def test_multiple_errors(self):
        report = AgentReport(
            agent_name="",
            recommendation="INVALID",
            confidence=2.0,
            rationale="ok",
        )
        errors = validate_report(report)
        assert len(errors) == 3

    def test_boundary_confidence_values(self):
        for conf in [0.0, 1.0]:
            report = AgentReport(
                agent_name="test",
                recommendation="APPROVE",
                confidence=conf,
                rationale="ok",
            )
            assert validate_report(report) == []

    def test_all_valid_recommendations(self):
        for rec in VALID_RECOMMENDATIONS:
            report = AgentReport(
                agent_name="test",
                recommendation=rec,
                confidence=0.5,
                rationale="ok",
            )
            assert validate_report(report) == []


# ======================================================================
# Consensus logic
# ======================================================================


def _make_report(agent: str, rec: str, conf: float = 0.7) -> AgentReport:
    return AgentReport(
        agent_name=agent,
        recommendation=rec,
        confidence=conf,
        rationale=f"{agent} says {rec}",
        player_id="P001",
        market="rushing_yards",
    )


class TestResolveConsensus:
    def test_unanimous_approve(self):
        reports = [
            _make_report("odds", "APPROVE"),
            _make_report("model", "APPROVE"),
            _make_report("bias", "APPROVE"),
            _make_report("risk", "APPROVE"),
        ]
        result = _resolve_consensus(reports)
        assert result["decision"] == "APPROVED"
        assert result["votes"]["APPROVE"] == 4
        assert result["override"] is False

    def test_unanimous_reject(self):
        reports = [
            _make_report("odds", "REJECT"),
            _make_report("model", "REJECT"),
            _make_report("bias", "REJECT"),
            _make_report("risk", "REJECT"),
        ]
        result = _resolve_consensus(reports)
        assert result["decision"] == "REJECTED"
        assert result["votes"]["REJECT"] == 4
        assert result["override"] is False

    def test_three_approve_one_reject(self):
        reports = [
            _make_report("odds", "APPROVE"),
            _make_report("model", "APPROVE"),
            _make_report("bias", "APPROVE"),
            _make_report("risk", "REJECT"),
        ]
        result = _resolve_consensus(reports)
        assert result["decision"] == "APPROVED"
        assert result["override"] is False

    def test_three_reject_one_approve(self):
        reports = [
            _make_report("odds", "REJECT"),
            _make_report("model", "REJECT"),
            _make_report("bias", "REJECT"),
            _make_report("risk", "APPROVE"),
        ]
        result = _resolve_consensus(reports)
        assert result["decision"] == "REJECTED"
        assert result["override"] is False

    def test_split_two_two_high_confidence_override(self):
        """2 approve + 2 reject with high confidence -> override approve."""
        reports = [
            _make_report("odds", "APPROVE", 0.90),
            _make_report("model", "APPROVE", 0.85),
            _make_report("bias", "REJECT", 0.50),
            _make_report("risk", "REJECT", 0.45),
        ]
        result = _resolve_consensus(reports)
        assert result["override"] is True
        # merged confidence = avg of all > 0.6
        assert result["merged_confidence"] > 0.6

    def test_split_two_two_low_confidence_reject(self):
        """2 approve + 2 reject with low confidence -> override reject."""
        reports = [
            _make_report("odds", "APPROVE", 0.30),
            _make_report("model", "APPROVE", 0.25),
            _make_report("bias", "REJECT", 0.50),
            _make_report("risk", "REJECT", 0.55),
        ]
        result = _resolve_consensus(reports)
        assert result["decision"] == "REJECTED"
        assert result["override"] is True

    def test_neutral_votes_counted(self):
        reports = [
            _make_report("odds", "APPROVE"),
            _make_report("model", "APPROVE"),
            _make_report("bias", "NEUTRAL"),
            _make_report("risk", "APPROVE"),
        ]
        result = _resolve_consensus(reports)
        assert result["decision"] == "APPROVED"
        assert result["votes"]["NEUTRAL"] == 1

    def test_merged_confidence_calculation(self):
        reports = [
            _make_report("a", "APPROVE", 0.80),
            _make_report("b", "APPROVE", 0.60),
        ]
        result = _resolve_consensus(reports)
        expected = (0.80 + 0.60) / 2
        assert abs(result["merged_confidence"] - expected) < 0.01

    def test_rationale_merged(self):
        reports = [
            _make_report("odds", "APPROVE"),
            _make_report("risk", "REJECT"),
        ]
        result = _resolve_consensus(reports)
        assert "[odds]" in result["rationale"]
        assert "[risk]" in result["rationale"]

    def test_agent_reports_included(self):
        reports = [_make_report("odds", "APPROVE")]
        result = _resolve_consensus(reports)
        assert len(result["agent_reports"]) == 1
        assert result["agent_reports"][0]["agent"] == "odds"


# ======================================================================
# Report grouping
# ======================================================================


class TestGroupReports:
    def test_groups_by_player_market(self):
        reports = [
            _make_report("odds", "APPROVE"),
            AgentReport(
                agent_name="odds",
                recommendation="REJECT",
                confidence=0.5,
                rationale="other player",
                player_id="P002",
                market="receiving_yards",
            ),
        ]
        grouped = _group_reports(reports)
        assert len(grouped) == 2
        assert ("P001", "rushing_yards") in grouped
        assert ("P002", "receiving_yards") in grouped

    def test_merges_same_key(self):
        reports = [
            _make_report("odds", "APPROVE"),
            _make_report("risk", "REJECT"),
        ]
        grouped = _group_reports(reports)
        assert len(grouped) == 1
        assert len(grouped[("P001", "rushing_yards")]) == 2

    def test_empty_reports(self):
        grouped = _group_reports([])
        assert len(grouped) == 0


# ======================================================================
# OddsAgent helpers
# ======================================================================


class TestSteamMoveDetection:
    def test_no_steam_below_threshold(self):
        df = pd.DataFrame([
            {"player_id": "P1", "market": "rushing_yards", "sportsbook": "DK",
             "line": 80.5, "as_of": "2025-01-01T10:00:00"},
            {"player_id": "P1", "market": "rushing_yards", "sportsbook": "DK",
             "line": 81.0, "as_of": "2025-01-01T12:00:00"},
        ])
        result = _detect_steam_moves(df)
        assert result.empty

    def test_steam_above_threshold(self):
        df = pd.DataFrame([
            {"player_id": "P1", "market": "rushing_yards", "sportsbook": "DK",
             "line": 80.5, "as_of": "2025-01-01T10:00:00"},
            {"player_id": "P1", "market": "rushing_yards", "sportsbook": "DK",
             "line": 83.0, "as_of": "2025-01-01T12:00:00"},
        ])
        result = _detect_steam_moves(df)
        assert len(result) == 1
        assert result.iloc[0]["movement"] == 2.5

    def test_empty_dataframe(self):
        result = _detect_steam_moves(pd.DataFrame())
        assert result.empty

    def test_single_snapshot_no_steam(self):
        df = pd.DataFrame([
            {"player_id": "P1", "market": "rushing_yards", "sportsbook": "DK",
             "line": 80.5, "as_of": "2025-01-01T10:00:00"},
        ])
        result = _detect_steam_moves(df)
        assert result.empty


class TestBestPrices:
    def test_picks_lowest_line(self):
        df = pd.DataFrame([
            {"player_id": "P1", "market": "rushing_yards", "sportsbook": "DK",
             "line": 82.5, "as_of": "2025-01-01T12:00:00"},
            {"player_id": "P1", "market": "rushing_yards", "sportsbook": "FD",
             "line": 80.5, "as_of": "2025-01-01T12:00:00"},
        ])
        result = _best_prices(df)
        assert len(result) == 1
        assert result.iloc[0]["line"] == 80.5
        assert result.iloc[0]["sportsbook"] == "FD"

    def test_empty_dataframe(self):
        result = _best_prices(pd.DataFrame())
        assert result.empty


# ======================================================================
# ModelDiagnosticsAgent helpers
# ======================================================================


class TestFlagSuspicious:
    def test_suspicious_gap(self):
        projections = pd.DataFrame([
            {"player_id": "P1", "market": "rushing_yards",
             "mu": 100.0, "sigma": 10.0},
        ])
        odds = pd.DataFrame([
            {"player_id": "P1", "market": "rushing_yards", "sportsbook": "DK",
             "line": 75.0, "as_of": "2025-01-01T12:00:00"},
        ])
        result = _flag_suspicious(projections, odds)
        assert len(result) == 1
        assert result.iloc[0]["player_id"] == "P1"

    def test_not_suspicious_within_range(self):
        projections = pd.DataFrame([
            {"player_id": "P1", "market": "rushing_yards",
             "mu": 80.0, "sigma": 10.0},
        ])
        odds = pd.DataFrame([
            {"player_id": "P1", "market": "rushing_yards", "sportsbook": "DK",
             "line": 78.0, "as_of": "2025-01-01T12:00:00"},
        ])
        result = _flag_suspicious(projections, odds)
        assert result.empty

    def test_empty_projections(self):
        result = _flag_suspicious(pd.DataFrame(), pd.DataFrame())
        assert result.empty

    def test_no_matching_odds(self):
        projections = pd.DataFrame([
            {"player_id": "P1", "market": "rushing_yards",
             "mu": 80.0, "sigma": 10.0},
        ])
        odds = pd.DataFrame([
            {"player_id": "P2", "market": "rushing_yards", "sportsbook": "DK",
             "line": 60.0, "as_of": "2025-01-01T12:00:00"},
        ])
        result = _flag_suspicious(projections, odds)
        assert result.empty


# ======================================================================
# Coordinator output format
# ======================================================================


class TestCoordinatorOutput:
    @patch("agents.coordinator.OddsAgent")
    @patch("agents.coordinator.ModelDiagnosticsAgent")
    @patch("agents.coordinator.MarketBiasAgent")
    @patch("agents.coordinator.RiskAgent")
    @patch("agents.coordinator._persist_decisions")
    def test_output_structure(
        self, mock_persist, mock_risk, mock_bias, mock_model, mock_odds
    ):
        mock_persist.return_value = 1

        for mock_cls in [mock_odds, mock_model, mock_bias, mock_risk]:
            instance = MagicMock()
            instance.analyze.return_value = [
                AgentReport(
                    agent_name="mock",
                    recommendation="APPROVE",
                    confidence=0.75,
                    rationale="test",
                    player_id="P001",
                    market="rushing_yards",
                )
            ]
            mock_cls.return_value = instance

        decisions = run_all_agents(2025, 13)

        assert len(decisions) >= 1
        d = decisions[0]
        assert "decision" in d
        assert "merged_confidence" in d
        assert "votes" in d
        assert "rationale" in d
        assert "override" in d
        assert "agent_reports" in d
        assert "player_id" in d
        assert "market" in d
        assert d["decision"] in ("APPROVED", "REJECTED")

    @patch("agents.coordinator.OddsAgent")
    @patch("agents.coordinator.ModelDiagnosticsAgent")
    @patch("agents.coordinator.MarketBiasAgent")
    @patch("agents.coordinator.RiskAgent")
    @patch("agents.coordinator._persist_decisions")
    def test_no_reports_returns_empty(
        self, mock_persist, mock_risk, mock_bias, mock_model, mock_odds
    ):
        for mock_cls in [mock_odds, mock_model, mock_bias, mock_risk]:
            instance = MagicMock()
            instance.analyze.return_value = []
            instance.name = "mock"
            mock_cls.return_value = instance

        decisions = run_all_agents(2025, 13)
        assert decisions == []

    @patch("agents.coordinator.OddsAgent")
    @patch("agents.coordinator.ModelDiagnosticsAgent")
    @patch("agents.coordinator.MarketBiasAgent")
    @patch("agents.coordinator.RiskAgent")
    @patch("agents.coordinator._persist_decisions")
    def test_agent_failure_handled(
        self, mock_persist, mock_risk, mock_bias, mock_model, mock_odds
    ):
        """A failing agent should not crash the coordinator."""
        mock_persist.return_value = 1

        # One agent raises, others return reports
        failing = MagicMock()
        failing.analyze.side_effect = RuntimeError("boom")
        failing.name = "failing_agent"
        mock_odds.return_value = failing

        for mock_cls in [mock_model, mock_bias, mock_risk]:
            instance = MagicMock()
            instance.analyze.return_value = [
                AgentReport(
                    agent_name="mock",
                    recommendation="APPROVE",
                    confidence=0.75,
                    rationale="test",
                    player_id="P001",
                    market="rushing_yards",
                )
            ]
            mock_cls.return_value = instance

        decisions = run_all_agents(2025, 13)
        # Should still produce decisions from the 3 working agents
        assert len(decisions) >= 1


# ======================================================================
# Consensus threshold constant
# ======================================================================


class TestConsensusThreshold:
    def test_threshold_is_three_of_four(self):
        """Consensus requires >= 3 of 4 agents to agree."""
        assert CONSENSUS_THRESHOLD == 3

    def test_exactly_threshold_approves(self):
        reports = [
            _make_report("a", "APPROVE"),
            _make_report("b", "APPROVE"),
            _make_report("c", "APPROVE"),
            _make_report("d", "NEUTRAL"),
        ]
        result = _resolve_consensus(reports)
        assert result["decision"] == "APPROVED"

    def test_below_threshold_needs_override(self):
        reports = [
            _make_report("a", "APPROVE", 0.90),
            _make_report("b", "APPROVE", 0.80),
            _make_report("c", "NEUTRAL", 0.60),
            _make_report("d", "REJECT", 0.40),
        ]
        result = _resolve_consensus(reports)
        # 2 approve, 1 neutral, 1 reject -> no consensus
        # but approve > reject and conf > 0.6, so override approve
        assert result["override"] is True
