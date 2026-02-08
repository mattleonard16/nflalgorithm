"""Tests for the dry-run validation pipeline.

Covers:
- Edge decay measurement and summary
- Agent vs raw comparison logic
- Grading picks against actuals
- Validation report structure
- Empty data edge cases
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from scripts.dry_run_validation import (
    LINE_MOVEMENT_OFFSETS,
    MARKET_TO_STAT,
    _apply_agent_filter,
    _compute_week_metrics,
    _empty_metrics,
    _grade_picks,
    _summarize_picks,
    compare_agent_vs_raw,
    edge_decay_summary,
    generate_validation_report,
    measure_edge_decay,
    replay_week,
)


# ======================================================================
# Edge decay
# ======================================================================


class TestMeasureEdgeDecay:
    @patch("scripts.dry_run_validation.rank_weekly_value")
    def test_returns_dataframe_with_expected_columns(self, mock_rank):
        mock_rank.return_value = pd.DataFrame([{
            "player_id": "P1",
            "market": "rushing_yards",
            "edge_percentage": 0.10,
            "mu": 80.0,
            "sigma": 12.0,
            "line": 75.0,
            "price": -110,
        }])
        result = measure_edge_decay(2025, 19, offsets=[-1, 1])
        assert not result.empty
        assert "player_id" in result.columns
        assert "offset" in result.columns
        assert "shifted_edge" in result.columns
        assert "edge_change" in result.columns
        assert len(result) == 2

    @patch("scripts.dry_run_validation.rank_weekly_value")
    def test_empty_when_no_picks(self, mock_rank):
        mock_rank.return_value = pd.DataFrame()
        result = measure_edge_decay(2025, 19)
        assert result.empty

    @patch("scripts.dry_run_validation.rank_weekly_value")
    def test_positive_offset_reduces_edge(self, mock_rank):
        mock_rank.return_value = pd.DataFrame([{
            "player_id": "P1",
            "market": "rushing_yards",
            "edge_percentage": 0.10,
            "mu": 80.0,
            "sigma": 12.0,
            "line": 75.0,
            "price": -110,
        }])
        result = measure_edge_decay(2025, 19, offsets=[3])
        assert len(result) == 1
        assert result.iloc[0]["edge_change"] < 0


class TestEdgeDecaySummary:
    def test_summary_groups_by_market_and_offset(self):
        df = pd.DataFrame([
            {"market": "rushing_yards", "offset": -1, "edge_change": -0.02},
            {"market": "rushing_yards", "offset": -1, "edge_change": -0.03},
            {"market": "rushing_yards", "offset": 1, "edge_change": 0.01},
            {"market": "receiving_yards", "offset": -1, "edge_change": -0.01},
        ])
        result = edge_decay_summary(df)
        assert not result.empty
        assert "mean_edge_change" in result.columns
        assert len(result) == 3

    def test_empty_input(self):
        result = edge_decay_summary(pd.DataFrame())
        assert result.empty


# ======================================================================
# Agent filtering
# ======================================================================


class TestApplyAgentFilter:
    def test_keeps_approved_picks(self):
        picks = pd.DataFrame([
            {"player_id": "P1", "market": "rushing_yards", "edge": 0.1},
            {"player_id": "P2", "market": "receiving_yards", "edge": 0.2},
        ])
        decisions = [
            {"player_id": "P1", "market": "rushing_yards", "decision": "APPROVED"},
            {"player_id": "P2", "market": "receiving_yards", "decision": "REJECTED"},
        ]
        result = _apply_agent_filter(picks, decisions)
        assert len(result) == 1
        assert result.iloc[0]["player_id"] == "P1"

    def test_empty_decisions_returns_copy(self):
        picks = pd.DataFrame([{"player_id": "P1", "market": "rushing_yards"}])
        result = _apply_agent_filter(picks, [])
        assert len(result) == 1

    def test_all_rejected_returns_empty(self):
        picks = pd.DataFrame([
            {"player_id": "P1", "market": "rushing_yards"},
        ])
        decisions = [
            {"player_id": "P1", "market": "rushing_yards", "decision": "REJECTED"},
        ]
        result = _apply_agent_filter(picks, decisions)
        assert result.empty

    def test_empty_picks(self):
        result = _apply_agent_filter(pd.DataFrame(), [{"decision": "APPROVED"}])
        assert result.empty


# ======================================================================
# Grading
# ======================================================================


class TestGradePicks:
    def test_win_over(self):
        picks = pd.DataFrame([{
            "player_id": "P1", "market": "rushing_yards", "line": 70.0,
        }])
        actuals = pd.DataFrame([{
            "player_id": "P1", "rushing_yards": 85.0,
        }])
        graded = _grade_picks(picks, actuals)
        assert graded.iloc[0]["result"] == "win"
        assert graded.iloc[0]["actual"] == 85.0

    def test_loss_under(self):
        picks = pd.DataFrame([{
            "player_id": "P1", "market": "rushing_yards", "line": 80.0,
        }])
        actuals = pd.DataFrame([{
            "player_id": "P1", "rushing_yards": 60.0,
        }])
        graded = _grade_picks(picks, actuals)
        assert graded.iloc[0]["result"] == "loss"

    def test_push_on_exact(self):
        picks = pd.DataFrame([{
            "player_id": "P1", "market": "rushing_yards", "line": 75.0,
        }])
        actuals = pd.DataFrame([{
            "player_id": "P1", "rushing_yards": 75.0,
        }])
        graded = _grade_picks(picks, actuals)
        assert graded.iloc[0]["result"] == "push"

    def test_unknown_when_no_actuals(self):
        picks = pd.DataFrame([{
            "player_id": "P1", "market": "rushing_yards", "line": 75.0,
        }])
        graded = _grade_picks(picks, pd.DataFrame())
        assert graded.iloc[0]["result"] == "unknown"

    def test_unknown_for_missing_player(self):
        picks = pd.DataFrame([{
            "player_id": "P1", "market": "rushing_yards", "line": 75.0,
        }])
        actuals = pd.DataFrame([{
            "player_id": "P2", "rushing_yards": 85.0,
        }])
        graded = _grade_picks(picks, actuals)
        assert graded.iloc[0]["result"] == "unknown"


# ======================================================================
# Metrics and summaries
# ======================================================================


class TestComputeWeekMetrics:
    def test_with_data(self):
        raw = pd.DataFrame([
            {"player_id": "P1", "market": "rushing_yards", "line": 70.0,
             "edge_percentage": 0.12},
        ])
        filtered = raw.copy()
        actuals = pd.DataFrame([
            {"player_id": "P1", "rushing_yards": 85.0},
        ])
        metrics = _compute_week_metrics(raw, filtered, actuals)
        assert metrics["raw"]["wins"] == 1
        assert metrics["filtered"]["wins"] == 1

    def test_empty_raw(self):
        metrics = _compute_week_metrics(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        assert metrics["raw"]["total_picks"] == 0
        assert metrics["filtered"]["total_picks"] == 0


class TestSummarizePicks:
    def test_with_data(self):
        df = pd.DataFrame([
            {"edge_percentage": 0.10, "kelly_fraction": 0.05, "stake": 50.0},
            {"edge_percentage": 0.15, "kelly_fraction": 0.08, "stake": 80.0},
        ])
        result = _summarize_picks(df, "test")
        assert result["total_picks"] == 2
        assert result["avg_edge"] > 0
        assert result["total_stake"] == 130.0

    def test_empty(self):
        result = _summarize_picks(pd.DataFrame(), "test")
        assert result["total_picks"] == 0


class TestEmptyMetrics:
    def test_structure(self):
        m = _empty_metrics()
        assert "raw" in m
        assert "filtered" in m
        assert m["raw"]["total_picks"] == 0
        assert m["filtered"]["hit_rate"] == 0.0


# ======================================================================
# A/B comparison
# ======================================================================


class TestCompareAgentVsRaw:
    def test_comparison_structure(self):
        results = [
            {
                "raw_picks": pd.DataFrame([{
                    "edge_percentage": 0.10, "kelly_fraction": 0.05, "stake": 50.0,
                }]),
                "agent_filtered": pd.DataFrame([{
                    "edge_percentage": 0.15, "kelly_fraction": 0.08, "stake": 80.0,
                }]),
            }
        ]
        comparison = compare_agent_vs_raw(results)
        assert "raw" in comparison
        assert "agent_filtered" in comparison
        assert "improvement" in comparison
        assert comparison["weeks_analyzed"] == 1

    def test_empty_results(self):
        comparison = compare_agent_vs_raw([])
        assert comparison["raw"]["total_picks"] == 0
        assert comparison["weeks_analyzed"] == 0


# ======================================================================
# Validation report
# ======================================================================


class TestGenerateValidationReport:
    def test_report_structure(self):
        results = [
            {
                "week": 19,
                "raw_picks": pd.DataFrame([{"edge_percentage": 0.10}]),
                "agent_filtered": pd.DataFrame([{"edge_percentage": 0.15}]),
                "agent_decisions": [
                    {"decision": "APPROVED", "player_id": "P1", "market": "rushing_yards"},
                ],
                "metrics": _empty_metrics(),
            }
        ]
        report = generate_validation_report(2025, results)
        assert report["season"] == 2025
        assert "generated_at" in report
        assert "weeks" in report
        assert "ab_comparison" in report
        assert len(report["weeks"]) == 1

    def test_empty_results(self):
        report = generate_validation_report(2025, [])
        assert report["season"] == 2025
        assert len(report["weeks"]) == 0


# ======================================================================
# Replay integration
# ======================================================================


class TestReplayWeek:
    @patch("scripts.dry_run_validation.run_all_agents")
    @patch("scripts.dry_run_validation.rank_weekly_value")
    @patch("scripts.dry_run_validation._load_actuals")
    def test_returns_expected_keys(self, mock_actuals, mock_rank, mock_agents):
        mock_rank.return_value = pd.DataFrame()
        mock_agents.return_value = []
        mock_actuals.return_value = pd.DataFrame()

        result = replay_week(2025, 19)
        assert "season" in result
        assert "week" in result
        assert "raw_picks" in result
        assert "agent_decisions" in result
        assert "metrics" in result

    @patch("scripts.dry_run_validation.run_all_agents")
    @patch("scripts.dry_run_validation.rank_weekly_value")
    @patch("scripts.dry_run_validation._load_actuals")
    def test_empty_picks_handled(self, mock_actuals, mock_rank, mock_agents):
        mock_rank.return_value = pd.DataFrame()
        mock_agents.return_value = []
        mock_actuals.return_value = pd.DataFrame()

        result = replay_week(2025, 19)
        assert result["raw_picks"].empty
        assert result["agent_filtered"].empty
