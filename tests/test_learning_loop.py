"""Tests for the post-game learning loop.

Covers:
- Outcome attribution logic
- Agent performance tracking correctness checks
- Confidence threshold recommendations
- Learning report structure
- Data loading edge cases
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest

from learning_loop import (
    MARKET_TO_STAT,
    _current_thresholds,
    _was_recommendation_correct,
    attribute_outcome,
    batch_attribute_outcomes,
    generate_learning_report,
    recommend_threshold_updates,
    update_agent_performance,
)
from utils.db import execute, fetchone


# ======================================================================
# Outcome attribution
# ======================================================================


class TestAttributeOutcome:
    @patch("learning_loop._load_best_line")
    @patch("learning_loop._load_actual_stat")
    @patch("learning_loop._load_projection")
    def test_accurate_model(self, mock_proj, mock_actual, mock_line):
        mock_proj.return_value = pd.Series({"mu": 80.0, "sigma": 15.0})
        mock_actual.return_value = 82.0
        mock_line.return_value = 75.0

        result = attribute_outcome(2025, 19, "P1", "rushing_yards")
        assert result["attribution"] == "model_accurate"
        assert result["model_error"] < 1.0
        assert result["variance_component"] is True
        assert result["actual"] == 82.0

    @patch("learning_loop._load_best_line")
    @patch("learning_loop._load_actual_stat")
    @patch("learning_loop._load_projection")
    def test_model_miss(self, mock_proj, mock_actual, mock_line):
        mock_proj.return_value = pd.Series({"mu": 80.0, "sigma": 10.0})
        mock_actual.return_value = 60.0
        mock_line.return_value = 75.0

        result = attribute_outcome(2025, 19, "P1", "rushing_yards")
        assert result["attribution"] == "model_miss"
        assert result["model_error"] > 1.0
        assert result["model_error"] <= 2.0

    @patch("learning_loop._load_best_line")
    @patch("learning_loop._load_actual_stat")
    @patch("learning_loop._load_projection")
    def test_high_variance(self, mock_proj, mock_actual, mock_line):
        mock_proj.return_value = pd.Series({"mu": 80.0, "sigma": 10.0})
        mock_actual.return_value = 110.0
        mock_line.return_value = 75.0

        result = attribute_outcome(2025, 19, "P1", "rushing_yards")
        assert result["attribution"] == "high_variance"
        assert result["model_error"] > 2.0

    @patch("learning_loop._load_actual_stat")
    @patch("learning_loop._load_projection")
    def test_no_projection(self, mock_proj, mock_actual):
        mock_proj.return_value = None
        mock_actual.return_value = 80.0

        result = attribute_outcome(2025, 19, "P1", "rushing_yards")
        assert result["attribution"] == "no_projection"

    @patch("learning_loop._load_actual_stat")
    @patch("learning_loop._load_projection")
    def test_no_actual(self, mock_proj, mock_actual):
        mock_proj.return_value = pd.Series({"mu": 80.0, "sigma": 12.0})
        mock_actual.return_value = None

        result = attribute_outcome(2025, 19, "P1", "rushing_yards")
        assert result["attribution"] == "no_actual"

    @patch("learning_loop._load_best_line")
    @patch("learning_loop._load_actual_stat")
    @patch("learning_loop._load_projection")
    def test_line_value_positive_when_beat(self, mock_proj, mock_actual, mock_line):
        mock_proj.return_value = pd.Series({"mu": 80.0, "sigma": 15.0})
        mock_actual.return_value = 90.0
        mock_line.return_value = 75.0

        result = attribute_outcome(2025, 19, "P1", "rushing_yards")
        assert result["line_value"] == 15.0

    @patch("learning_loop._load_best_line")
    @patch("learning_loop._load_actual_stat")
    @patch("learning_loop._load_projection")
    def test_zero_sigma_handled(self, mock_proj, mock_actual, mock_line):
        mock_proj.return_value = pd.Series({"mu": 80.0, "sigma": 0.0})
        mock_actual.return_value = 80.5
        mock_line.return_value = 75.0

        result = attribute_outcome(2025, 19, "P1", "rushing_yards")
        assert result["model_error"] >= 0


class TestBatchAttributeOutcomes:
    @patch("learning_loop._load_all_projections")
    @patch("learning_loop.attribute_outcome")
    def test_batch_calls_attribute(self, mock_attr, mock_projs):
        mock_projs.return_value = pd.DataFrame([
            {"player_id": "P1", "market": "rushing_yards", "mu": 80, "sigma": 12},
            {"player_id": "P2", "market": "receiving_yards", "mu": 60, "sigma": 10},
        ])
        mock_attr.return_value = {"attribution": "model_accurate"}

        results = batch_attribute_outcomes(2025, 19)
        assert len(results) == 2
        assert mock_attr.call_count == 2

    @patch("learning_loop._load_all_projections")
    def test_empty_projections(self, mock_projs):
        mock_projs.return_value = pd.DataFrame()
        results = batch_attribute_outcomes(2025, 19)
        assert results == []


# ======================================================================
# Agent performance correctness
# ======================================================================


class TestWasRecommendationCorrect:
    def test_approve_approved_win(self):
        assert _was_recommendation_correct("APPROVE", "APPROVED", "win") is True

    def test_approve_approved_loss(self):
        assert _was_recommendation_correct("APPROVE", "APPROVED", "loss") is False

    def test_reject_rejected_loss(self):
        assert _was_recommendation_correct("REJECT", "REJECTED", "loss") is True

    def test_reject_approved_win(self):
        assert _was_recommendation_correct("REJECT", "APPROVED", "win") is False

    def test_neutral_not_counted(self):
        assert _was_recommendation_correct("NEUTRAL", "APPROVED", "win") is False

    def test_push_result(self):
        assert _was_recommendation_correct("APPROVE", "APPROVED", "push") is False

    def test_reject_any_loss(self):
        assert _was_recommendation_correct("REJECT", "APPROVED", "loss") is True


class TestUpdateAgentPerformance:
    @patch("learning_loop._load_bet_outcomes")
    @patch("learning_loop._load_agent_decisions")
    def test_no_decisions_returns_zero(self, mock_decisions, mock_outcomes):
        mock_decisions.return_value = pd.DataFrame()
        mock_outcomes.return_value = pd.DataFrame()
        assert update_agent_performance(2025, 19) == 0

    @patch("learning_loop.executemany")
    @patch("learning_loop._load_bet_outcomes")
    @patch("learning_loop._load_agent_decisions")
    def test_inserts_records(self, mock_decisions, mock_outcomes, mock_exec):
        mock_decisions.return_value = pd.DataFrame([{
            "player_id": "P1",
            "market": "rushing_yards",
            "decision": "APPROVED",
            "merged_confidence": 0.8,
            "votes": '{"APPROVE": 3, "REJECT": 1}',
            "agent_reports": json.dumps([
                {"agent": "odds_agent", "recommendation": "APPROVE", "confidence": 0.85},
                {"agent": "risk_agent", "recommendation": "REJECT", "confidence": 0.6},
            ]),
        }])
        mock_outcomes.return_value = pd.DataFrame([{
            "player_id": "P1",
            "market": "rushing_yards",
            "result": "win",
            "profit_units": 0.91,
            "confidence_tier": "HIGH",
            "edge_at_placement": 0.12,
        }])

        result = update_agent_performance(2025, 19)
        assert result == 2
        mock_exec.assert_called_once()


class TestAgentPerformanceUpsert:
    """Relearning a week replaces its records, on SQLite and MySQL."""

    @pytest.fixture()
    def graded_week(self, matrix_database):
        for table in ("agent_performance", "agent_decisions", "bet_outcomes"):
            execute(f"DELETE FROM {table}")
        execute(
            "INSERT INTO agent_decisions (season, week, player_id, market, decision, "
            "merged_confidence, votes, rationale, coordinator_override, agent_reports, "
            "decided_at) VALUES (2026, 1, 'P1', 'rushing_yards', 'APPROVED', 0.8, '{}', "
            "'test', 0, ?, '2026-09-10T00:00:00Z')",
            (json.dumps([{"agent": "odds_agent", "recommendation": "APPROVE"}]),),
        )
        execute(
            "INSERT INTO bet_outcomes (bet_id, season, week, player_id, market, sportsbook, "
            "side, line, price, result, profit_units, recorded_at) VALUES ('b1', 2026, 1, "
            "'P1', 'rushing_yards', 'DraftKings', 'over', 64.5, -110, 'win', 0.91, "
            "'2026-09-15T00:00:00Z')"
        )

    def test_relearning_a_week_replaces_its_records(self, graded_week):
        update_agent_performance(2026, 1)

        assert update_agent_performance(2026, 1) == 1
        assert fetchone("SELECT COUNT(*) FROM agent_performance")[0] == 1


# ======================================================================
# Threshold recommendations
# ======================================================================


class TestRecommendThresholdUpdates:
    @patch("learning_loop.read_dataframe")
    def test_recommends_raise_for_low_hit_rate(self, mock_read):
        mock_read.return_value = pd.DataFrame([
            {"confidence_tier": "LOW", "result": "loss", "profit_units": -1.0, "edge_at_placement": 0.06},
            {"confidence_tier": "LOW", "result": "loss", "profit_units": -1.0, "edge_at_placement": 0.05},
            {"confidence_tier": "LOW", "result": "loss", "profit_units": -1.0, "edge_at_placement": 0.07},
            {"confidence_tier": "LOW", "result": "loss", "profit_units": -1.0, "edge_at_placement": 0.04},
            {"confidence_tier": "LOW", "result": "win", "profit_units": 0.91, "edge_at_placement": 0.08},
        ])

        result = recommend_threshold_updates()
        recs = result["recommendations"]
        assert len(recs) >= 1
        assert recs[0]["action"] == "raise_threshold"

    @patch("learning_loop.read_dataframe")
    def test_recommends_maintain_for_good_performance(self, mock_read):
        mock_read.return_value = pd.DataFrame([
            {"confidence_tier": "HIGH", "result": "win", "profit_units": 0.91, "edge_at_placement": 0.15},
            {"confidence_tier": "HIGH", "result": "win", "profit_units": 0.91, "edge_at_placement": 0.14},
            {"confidence_tier": "HIGH", "result": "win", "profit_units": 0.91, "edge_at_placement": 0.16},
            {"confidence_tier": "HIGH", "result": "win", "profit_units": 0.91, "edge_at_placement": 0.13},
            {"confidence_tier": "HIGH", "result": "loss", "profit_units": -1.0, "edge_at_placement": 0.12},
        ])

        result = recommend_threshold_updates()
        recs = result["recommendations"]
        assert len(recs) >= 1
        assert recs[0]["action"] == "maintain_or_lower"

    @patch("learning_loop.read_dataframe")
    def test_empty_outcomes(self, mock_read):
        mock_read.return_value = pd.DataFrame()
        result = recommend_threshold_updates()
        assert result["tier_performance"] == []
        assert result["recommendations"] == []


class TestCurrentThresholds:
    def test_returns_expected_keys(self):
        thresholds = _current_thresholds()
        assert "min_edge_threshold" in thresholds
        assert "min_confidence" in thresholds
        assert "confidence_min_tier" in thresholds


# ======================================================================
# Learning report
# ======================================================================


class TestGenerateLearningReport:
    @patch("learning_loop._attribution_summary")
    @patch("learning_loop.recommend_threshold_updates")
    @patch("learning_loop._model_accuracy_trends")
    @patch("learning_loop.get_agent_accuracy_summary")
    @patch("learning_loop._tier_performance")
    def test_report_structure(
        self, mock_tier, mock_agent, mock_trends, mock_thresh, mock_attr
    ):
        mock_tier.return_value = [
            {"tier": "HIGH", "total": 10, "wins": 7, "losses": 3,
             "hit_rate": 0.70, "total_profit": 3.37, "avg_edge": 0.12},
        ]
        mock_agent.return_value = pd.DataFrame([
            {"agent_name": "odds_agent", "total": 20, "correct": 14,
             "accuracy": 0.70, "avg_confidence": 0.8},
        ])
        mock_trends.return_value = [
            {"week": 19, "market": "rushing_yards", "mae": 12.5, "sample_size": 30},
        ]
        mock_thresh.return_value = {
            "tier_performance": [],
            "recommendations": [],
            "current_thresholds": _current_thresholds(),
        }
        mock_attr.return_value = {"total": 0, "breakdown": {}}

        report = generate_learning_report(2025, week_range=(19, 21))

        assert report["season"] == 2025
        assert report["week_range"] == [19, 21]
        assert "generated_at" in report
        assert "tier_performance" in report
        assert "agent_performance" in report
        assert "model_accuracy_trends" in report
        assert "threshold_recommendations" in report
        assert "attribution_summary" in report


# ======================================================================
# Market mapping
# ======================================================================


class TestMarketToStat:
    def test_all_expected_markets(self):
        expected = {
            "rushing_yards",
            "receiving_yards",
            "passing_yards",
            "receptions",
            "targets",
            "anytime_touchdown",
        }
        assert set(MARKET_TO_STAT.keys()) == expected

    def test_values_match_db_columns(self):
        for market, col in MARKET_TO_STAT.items():
            assert isinstance(col, str)
            assert len(col) > 0


# ======================================================================
# Projection ids vs stats ids
# ======================================================================


def _insert(table: str, **values) -> None:
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", tuple(values.values()))


@pytest.fixture
def bridged_db(monkeypatch, tmp_path):
    """A migrated database where projections and stats spell one player two ways.

    weekly_projections mints ``ARI_james_conner`` from the roster while
    player_stats_enhanced keeps ``ARI_j_conner``; gsis_id is the shared key.
    """
    from config import config
    from schema_migrations import MigrationManager

    db_path = tmp_path / "learning.db"
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))
    monkeypatch.setattr(config.database, "backend", "sqlite")
    monkeypatch.setattr(config.database, "path", str(db_path))
    MigrationManager(db_path).run()
    _insert(
        "nfl_roster_players",
        season=2026,
        gsis_id="00-0033553",
        player_id="ARI_james_conner",
        player_name="James Conner",
        team="ARI",
        position="RB",
        roster_week=1,
        updated_at="2026-09-01",
    )
    stat_columns = (
        "age games_played snap_count snap_percentage rushing_attempts passing_yards "
        "passing_attempts receiving_yards receptions targets passing_tds receiving_tds "
        "red_zone_touches target_share air_yards yac_yards game_script"
    ).split()
    _insert(
        "player_stats_enhanced",
        player_id="ARI_j_conner",
        gsis_id="00-0033553",
        season=2026,
        week=1,
        name="J.Conner",
        team="ARI",
        position="RB",
        rushing_yards=80,
        rushing_tds=1,
        created_at="2026-09-08",
        updated_at="2026-09-08",
        **{column: 0 for column in stat_columns},
    )
    _insert(
        "weekly_projections",
        season=2026,
        week=1,
        player_id="ARI_james_conner",
        team="ARI",
        opponent="SEA",
        market="rushing_yards",
        mu=70,
        sigma=20,
        model_version="v",
        featureset_hash="h",
        generated_at="2026-09-06",
    )
    return db_path


def test_accuracy_trends_match_projections_to_stats_through_gsis_id(bridged_db) -> None:
    from learning_loop import _model_accuracy_trends

    trends = _model_accuracy_trends(2026, 1, 1)

    assert {"week": 1, "market": "rushing_yards", "mae": 10.0, "sample_size": 1} in trends


def test_actual_stat_loads_for_a_projection_style_id(bridged_db) -> None:
    from learning_loop import _load_actual_stat

    assert _load_actual_stat(2026, 1, "ARI_james_conner", "rushing_yards") == 80.0


def test_actual_stat_lookup_failure_is_logged_not_swallowed(bridged_db, caplog) -> None:
    from learning_loop import _load_actual_stat

    execute("DROP TABLE player_stats_enhanced")

    with caplog.at_level("WARNING", logger="learning_loop"):
        assert _load_actual_stat(2026, 1, "ARI_james_conner", "rushing_yards") is None

    assert "ARI_james_conner" in caplog.text
