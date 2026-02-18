"""Tests for the NBA risk manager (nba_risk_manager.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from schema_migrations import MigrationManager
from utils.db import execute, read_dataframe


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_nba_risk.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    import config as cfg

    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")

    MigrationManager(db_path).run()
    return db_path


def _make_value_df(**overrides) -> pd.DataFrame:
    """Create a minimal NBA value view DataFrame."""
    defaults = {
        "game_date": "2026-02-11",
        "player_id": 1234,
        "player_name": "Test Player",
        "team": "LAL",
        "event_id": "evt1",
        "market": "pts",
        "sportsbook": "draftkings",
        "line": 25.5,
        "over_price": -110,
        "kelly_fraction": 0.05,
        "p_win": 0.65,
        "stake": 50.0,
        "edge_percentage": 0.12,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# Correlation detection
# ---------------------------------------------------------------------------


class TestNbaCorrelationDetection:
    def test_empty_df_returns_empty_with_column(self):
        from nba_risk_manager import detect_correlations

        df = pd.DataFrame()
        result = detect_correlations(df)
        assert result.empty
        assert "correlation_group" in result.columns

    def test_single_row_no_correlation(self):
        from nba_risk_manager import detect_correlations

        df = _make_value_df()
        result = detect_correlations(df)
        assert result["correlation_group"].isna().all()

    def test_pts_fg3m_positive_correlation(self):
        from nba_risk_manager import detect_correlations

        df = pd.concat([
            _make_value_df(market="pts", player_id=1),
            _make_value_df(market="fg3m", player_id=2),
        ], ignore_index=True)
        result = detect_correlations(df)
        assert result["correlation_group"].notna().any()
        groups = result["correlation_group"].dropna().unique()
        assert any("pos_corr" in g for g in groups)

    def test_pts_ast_positive_correlation(self):
        from nba_risk_manager import detect_correlations

        df = pd.concat([
            _make_value_df(market="pts", player_id=1),
            _make_value_df(market="ast", player_id=2),
        ], ignore_index=True)
        result = detect_correlations(df)
        assert result["correlation_group"].notna().any()

    def test_same_team_same_market_tagged(self):
        from nba_risk_manager import detect_correlations

        df = pd.concat([
            _make_value_df(market="pts", player_id=1, team="LAL"),
            _make_value_df(market="pts", player_id=2, team="LAL"),
        ], ignore_index=True)
        result = detect_correlations(df)
        assert result["correlation_group"].notna().any()
        groups = result["correlation_group"].dropna().unique()
        assert any("same_team" in g for g in groups)

    def test_different_event_no_correlation(self):
        from nba_risk_manager import detect_correlations

        df = pd.concat([
            _make_value_df(market="pts", player_id=1, team="LAL", event_id="e1"),
            _make_value_df(market="fg3m", player_id=2, team="BOS", event_id="e2"),
        ], ignore_index=True)
        result = detect_correlations(df)
        assert result["correlation_group"].isna().all()

    def test_reb_ast_not_correlated(self):
        from nba_risk_manager import detect_correlations

        df = pd.concat([
            _make_value_df(market="reb", player_id=1, team="LAL", event_id="e1"),
            _make_value_df(market="ast", player_id=2, team="BOS", event_id="e2"),
        ], ignore_index=True)
        result = detect_correlations(df)
        assert result["correlation_group"].isna().all()


# ---------------------------------------------------------------------------
# Team stacks
# ---------------------------------------------------------------------------


class TestNbaTeamStacks:
    def test_empty_df_returns_empty_dict(self):
        from nba_risk_manager import detect_team_stacks

        assert detect_team_stacks(pd.DataFrame()) == {}

    def test_single_team_two_bets(self):
        from nba_risk_manager import detect_team_stacks

        df = pd.concat([
            _make_value_df(team="LAL", player_id=1),
            _make_value_df(team="LAL", player_id=2),
        ], ignore_index=True)
        stacks = detect_team_stacks(df)
        assert "LAL" in stacks
        assert len(stacks["LAL"]) == 2

    def test_single_bet_per_team_no_stacks(self):
        from nba_risk_manager import detect_team_stacks

        df = pd.concat([
            _make_value_df(team="LAL", player_id=1),
            _make_value_df(team="BOS", player_id=2),
        ], ignore_index=True)
        stacks = detect_team_stacks(df)
        assert stacks == {}


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------


class TestNbaExposure:
    def test_empty_df_returns_empty_with_column(self):
        from nba_risk_manager import compute_exposure

        df = pd.DataFrame()
        result = compute_exposure(df, 1000.0)
        assert result.empty
        assert "exposure_warning" in result.columns

    def test_below_threshold_no_warning(self):
        from nba_risk_manager import compute_exposure

        df = _make_value_df(stake=10.0, team="LAL")
        result = compute_exposure(df, 10000.0)
        assert result["exposure_warning"].isna().all()

    def test_high_team_exposure_flagged(self, monkeypatch):
        import config as cfg

        monkeypatch.setattr(cfg.config.risk, "max_team_exposure", 0.05)
        from nba_risk_manager import compute_exposure

        df = _make_value_df(stake=600.0, team="LAL")
        result = compute_exposure(df, 1000.0)
        assert result["exposure_warning"].notna().any()
        assert "team_exposure" in result.iloc[0]["exposure_warning"]


# ---------------------------------------------------------------------------
# Full assess_risk pipeline
# ---------------------------------------------------------------------------


class TestNbaAssessRisk:
    def test_empty_df_returns_all_columns(self):
        from nba_risk_manager import assess_risk

        df = pd.DataFrame()
        result = assess_risk(df)
        expected_cols = {
            "correlation_group", "exposure_warning",
            "risk_adjusted_kelly", "mean_drawdown",
            "max_drawdown", "p95_drawdown",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_single_row_adds_columns(self):
        from nba_risk_manager import assess_risk

        df = _make_value_df()
        result = assess_risk(df, bankroll=1000.0)
        assert "risk_adjusted_kelly" in result.columns
        assert "correlation_group" in result.columns
        assert "exposure_warning" in result.columns
        assert len(result) == 1

    def test_risk_adjusted_kelly_is_numeric(self):
        from nba_risk_manager import assess_risk

        df = _make_value_df()
        result = assess_risk(df, bankroll=1000.0)
        val = result.iloc[0]["risk_adjusted_kelly"]
        assert isinstance(val, float)
        assert val > 0

    def test_immutability(self):
        from nba_risk_manager import assess_risk

        df = _make_value_df()
        original_cols = set(df.columns)
        assess_risk(df, bankroll=1000.0)
        assert set(df.columns) == original_cols


# ---------------------------------------------------------------------------
# Persistence integration
# ---------------------------------------------------------------------------


class TestNbaRiskPersistence:
    def test_persist_writes_to_table(self, db):
        from nba_risk_manager import _persist_risk_assessments, assess_risk

        df = _make_value_df()
        assessed = assess_risk(df, bankroll=1000.0)
        count = _persist_risk_assessments(assessed, "2026-02-11")
        assert count == 1

        rows = read_dataframe(
            "SELECT * FROM nba_risk_assessments WHERE game_date = ?",
            params=("2026-02-11",),
        )
        assert len(rows) == 1
        assert rows.iloc[0]["market"] == "pts"

    def test_persist_empty_df_returns_zero(self, db):
        from nba_risk_manager import _persist_risk_assessments

        count = _persist_risk_assessments(pd.DataFrame(), "2026-02-11")
        assert count == 0
