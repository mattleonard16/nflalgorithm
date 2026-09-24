"""Tests for T1 #8: no-vig probability for NFL value engine.

NFR-6 gate: behavior must be off by default and toggle via
`config.features.no_vig_enabled` (env NFL_FEATURE_NO_VIG).
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from config import config
from schema_migrations import MigrationManager
from value_betting_engine import _implied_probability, rank_weekly_value

# The no-vig math itself is tested in tests/test_odds_math.py, which runs without
# the private engine.

# ---------------------------------------------------------------------------
# Integration: rank_weekly_value branches on the flag
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db_two_sided():
    tmp = Path(tempfile.mkstemp(suffix=".db")[1])
    orig_path = config.database.path
    orig_backend = config.database.backend
    env_backend = os.environ.get("DB_BACKEND")
    env_sqlite_path = os.environ.get("SQLITE_DB_PATH")
    os.environ["DB_BACKEND"] = "sqlite"
    os.environ["SQLITE_DB_PATH"] = str(tmp)
    config.database.backend = "sqlite"
    config.database.path = str(tmp)

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
                95.0,
                10.0,
                "v1",
                "hash1",
                "2024-09-01T00:00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO weekly_odds
            (event_id, season, week, player_id, market, sportsbook, line, price, under_price, as_of)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "evt1",
                2024,
                1,
                "p1",
                "rushing_yards",
                "BookA",
                70.5,
                -110,
                -110,
                "2024-09-01T00:00:00",
            ),
        )
        conn.commit()

    yield str(tmp)

    config.database.path = orig_path
    config.database.backend = orig_backend
    if env_backend is not None:
        os.environ["DB_BACKEND"] = env_backend
    else:
        os.environ.pop("DB_BACKEND", None)
    if env_sqlite_path is not None:
        os.environ["SQLITE_DB_PATH"] = env_sqlite_path
    else:
        os.environ.pop("SQLITE_DB_PATH", None)
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass


def test_no_vig_flag_off_uses_raw_implied_prob(temp_db_two_sided, monkeypatch):
    """With flag off, engine reproduces Tier-0 single-sided edge."""
    monkeypatch.setattr(config.features, "no_vig_enabled", False)
    df = rank_weekly_value(2024, 1, min_edge=-1.0)
    assert not df.empty
    # Raw -110 implied = 100/210 ≈ 0.5238
    # Our row would not have an under_price effect at all
    assert df.iloc[0]["edge_percentage"] == pytest.approx(
        df.iloc[0]["p_win"] - _implied_probability(-110), abs=1e-9
    )


def test_no_vig_flag_on_lifts_edge_when_under_present(temp_db_two_sided, monkeypatch):
    """Flag on + symmetric -110/-110 quote → implied prob drops to exactly 0.5,
    which lifts edge_percentage vs the flag-off baseline."""
    monkeypatch.setattr(config.features, "no_vig_enabled", False)
    df_raw = rank_weekly_value(2024, 1, min_edge=-1.0).copy()

    monkeypatch.setattr(config.features, "no_vig_enabled", True)
    df_novig = rank_weekly_value(2024, 1, min_edge=-1.0).copy()

    edge_raw = df_raw.iloc[0]["edge_percentage"]
    edge_novig = df_novig.iloc[0]["edge_percentage"]
    # No-vig fair prob = 0.5 < 0.5238 raw → edge increases by ~2.38 pp
    assert edge_novig > edge_raw
    assert edge_novig - edge_raw == pytest.approx(_implied_probability(-110) - 0.5, abs=1e-9)


@pytest.fixture
def temp_db_no_under():
    """Same shape as temp_db_two_sided but `under_price` stays NULL."""
    tmp = Path(tempfile.mkstemp(suffix=".db")[1])
    orig_path = config.database.path
    orig_backend = config.database.backend
    env_backend = os.environ.get("DB_BACKEND")
    env_sqlite_path = os.environ.get("SQLITE_DB_PATH")
    os.environ["DB_BACKEND"] = "sqlite"
    os.environ["SQLITE_DB_PATH"] = str(tmp)
    config.database.backend = "sqlite"
    config.database.path = str(tmp)

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
                95.0,
                10.0,
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

    config.database.path = orig_path
    config.database.backend = orig_backend
    if env_backend is not None:
        os.environ["DB_BACKEND"] = env_backend
    else:
        os.environ.pop("DB_BACKEND", None)
    if env_sqlite_path is not None:
        os.environ["SQLITE_DB_PATH"] = env_sqlite_path
    else:
        os.environ.pop("SQLITE_DB_PATH", None)
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass


def test_no_vig_flag_on_falls_back_when_under_missing(temp_db_no_under, monkeypatch):
    """Flag on but under_price IS NULL → engine falls back to single-sided."""
    monkeypatch.setattr(config.features, "no_vig_enabled", True)
    df = rank_weekly_value(2024, 1, min_edge=-1.0)
    assert not df.empty
    assert df.iloc[0]["edge_percentage"] == pytest.approx(
        df.iloc[0]["p_win"] - _implied_probability(-110), abs=1e-9
    )


def test_no_vig_default_off(monkeypatch):
    """NFR-6 gate: flag must default to False when env is unset.

    Reloads the underlying config module so we assert against the live
    `config.features.no_vig_enabled` value, not a replayed parse.
    """
    import importlib.util
    from pathlib import Path

    monkeypatch.delenv("NFL_FEATURE_NO_VIG", raising=False)

    cfg_path = Path(__file__).parent.parent / "config" / "runtime.py"
    spec = importlib.util.spec_from_file_location("_config_under_test", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.config.features.no_vig_enabled is False
