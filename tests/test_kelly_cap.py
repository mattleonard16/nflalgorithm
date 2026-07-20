"""Tests for T1 #13: Kelly cap in NFL ranking path.

NFR-6 gate: behavior must be off by default and toggle via
`config.features.kelly_cap_enabled` (env NFL_FEATURE_KELLY_CAP).
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from config import config
from schema_migrations import MigrationManager
from value_betting_engine import kelly_fraction, rank_weekly_value


@pytest.fixture
def temp_db_high_edge(monkeypatch):
    """DB with a single high-edge bet so kelly_fraction comes out > max_kelly."""
    tmp = Path(tempfile.mkstemp(suffix=".db")[1])
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp))
    monkeypatch.setattr(config.database, "backend", "sqlite")
    monkeypatch.setattr(config.database, "path", str(tmp))

    MigrationManager(tmp).run()

    with sqlite3.connect(tmp) as conn:
        conn.execute(
            """
            INSERT INTO weekly_projections
            (season, week, player_id, team, opponent, market, mu, sigma,
             model_version, featureset_hash, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                2024,
                1,
                "p1",
                "MIA",
                "NE",
                "rushing_yards",
                110.0,
                8.0,
                "v1",
                "hash1",
                "2024-09-01T00:00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO weekly_odds
            (event_id, season, week, player_id, market, sportsbook, line, price, as_of)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("evt1", 2024, 1, "p1", "rushing_yards", "BookA", 70.5, -110, "2024-09-01T00:00:00"),
        )
        conn.commit()

    yield str(tmp)

    try:
        Path(tmp).unlink()
    except FileNotFoundError:
        pass


def test_kelly_cap_flag_off_passes_full_kelly(temp_db_high_edge, monkeypatch):
    """Flag off → engine emits full Kelly (Tier 0 baseline)."""
    monkeypatch.setattr(config.features, "kelly_cap_enabled", False)
    df = rank_weekly_value(2024, 1, min_edge=-1.0)
    assert not df.empty
    row = df.iloc[0]
    full = kelly_fraction(float(row["p_win"]), int(row["price"]))
    assert row["kelly_fraction"] == pytest.approx(full, abs=1e-9)


def test_kelly_cap_flag_on_scales_by_fraction_and_clips_to_max(temp_db_high_edge, monkeypatch):
    """Flag on → full Kelly × kelly_fraction, clipped at max_kelly."""
    monkeypatch.setattr(config.features, "kelly_cap_enabled", True)
    monkeypatch.setattr(config.betting, "kelly_fraction", 0.25)
    monkeypatch.setattr(config.betting, "max_kelly", 0.10)

    df = rank_weekly_value(2024, 1, min_edge=-1.0)
    assert not df.empty
    row = df.iloc[0]
    full = kelly_fraction(float(row["p_win"]), int(row["price"]))
    expected = min(full * 0.25, 0.10)
    expected = max(expected, 0.0)
    assert row["kelly_fraction"] == pytest.approx(expected, abs=1e-9)
    assert row["kelly_fraction"] <= 0.10


def test_kelly_cap_flag_on_stake_reflects_capped_fraction(temp_db_high_edge, monkeypatch):
    """stake column must reflect the capped kelly_fraction, not the raw."""
    monkeypatch.setattr(config.features, "kelly_cap_enabled", True)
    df = rank_weekly_value(2024, 1, min_edge=-1.0)
    row = df.iloc[0]
    assert row["stake"] == pytest.approx(row["kelly_fraction"] * config.betting.bankroll, abs=1e-9)


def test_kelly_cap_default_off(monkeypatch):
    """NFR-6 gate: flag must default to False when env is unset."""
    import importlib.util

    monkeypatch.delenv("NFL_FEATURE_KELLY_CAP", raising=False)

    cfg_path = Path(__file__).parent.parent / "config" / "runtime.py"
    spec = importlib.util.spec_from_file_location("_config_under_test", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.config.features.kelly_cap_enabled is False
