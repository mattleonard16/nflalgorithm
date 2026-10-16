"""Learning-loop actuals must reach stats rows through gsis_id.

Projections carry the roster-form player_id ("DAL_ceedee_lamb"); stats carry
the abbreviated form ("DAL_c_lamb"). Joining on player_id alone matched
almost nothing, so MAE trends and attributions were computed on an empty set.
"""

from __future__ import annotations

import pytest

from learning_loop import _load_actual_stat, _model_accuracy_trends
from schema_migrations import MigrationManager
from utils.db import execute


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "learning.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)
    import config as cfg

    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")
    MigrationManager(db_path).run()
    return db_path


def _seed_stats(player_id: str, gsis_id: str, receiving_yards: float, week: int = 3) -> None:
    execute(
        """
        INSERT INTO player_stats_enhanced
            (player_id, gsis_id, season, week, name, team, position, age, games_played,
             snap_count, snap_percentage, rushing_yards, rushing_attempts,
             passing_yards, passing_attempts, receiving_yards, receptions,
             targets, red_zone_touches, target_share, air_yards, yac_yards, game_script)
        VALUES (?, ?, 2026, ?, 'Name', 'DAL', 'WR', 27, 1, 50, 80, 0, 0, 0, 0, ?, 6, 9,
                0, 0.2, 0, 0, 0)
        """,
        params=(player_id, gsis_id, week, receiving_yards),
    )


def _seed_roster(player_id: str, gsis_id: str) -> None:
    execute(
        """
        INSERT INTO nfl_roster_players
            (season, gsis_id, player_id, player_name, team, position, roster_status,
             roster_week, updated_at)
        VALUES (2026, ?, ?, 'Name', 'DAL', 'WR', 'ACT', 3, 'now')
        """,
        params=(gsis_id, player_id),
    )


def _seed_projection(player_id: str, mu: float, week: int = 3) -> None:
    execute(
        """
        INSERT INTO weekly_projections
            (season, week, player_id, team, opponent, market, mu, sigma,
             model_version, featureset_hash, generated_at)
        VALUES (2026, ?, ?, 'DAL', 'NYG', 'receiving_yards', ?, 20, 'v', 'h', 'now')
        """,
        params=(week, player_id, mu),
    )


def test_actual_stat_bridges_roster_id_to_stats_id(db) -> None:
    _seed_stats("DAL_c_lamb", "00-0036358", 101.0)
    _seed_roster("DAL_ceedee_lamb", "00-0036358")

    assert _load_actual_stat(2026, 3, "DAL_ceedee_lamb", "receiving_yards") == 101.0


def test_actual_stat_falls_back_to_player_id_without_roster_row(db) -> None:
    _seed_stats("DAL_c_lamb", "00-0036358", 101.0)

    assert _load_actual_stat(2026, 3, "DAL_c_lamb", "receiving_yards") == 101.0
    assert _load_actual_stat(2026, 3, "DAL_ceedee_lamb", "receiving_yards") is None


def test_actual_stat_does_not_cross_to_another_player(db) -> None:
    _seed_stats("DAL_c_lamb", "00-0036358", 101.0)
    _seed_stats("DAL_g_pickens", "00-0037247", 40.0)
    _seed_roster("DAL_ceedee_lamb", "00-0036358")
    _seed_roster("DAL_george_pickens", "00-0037247")

    assert _load_actual_stat(2026, 3, "DAL_george_pickens", "receiving_yards") == 40.0


def test_accuracy_trends_match_projections_through_gsis(db) -> None:
    _seed_stats("DAL_c_lamb", "00-0036358", 101.0)
    _seed_stats("DAL_g_pickens", "00-0037247", 40.0)
    _seed_roster("DAL_ceedee_lamb", "00-0036358")
    _seed_roster("DAL_george_pickens", "00-0037247")
    _seed_projection("DAL_ceedee_lamb", 91.0)
    _seed_projection("DAL_george_pickens", 60.0)

    trends = [row for row in _model_accuracy_trends(2026, 3, 3) if row["market"] == "receiving_yards"]

    assert trends == [{"week": 3, "market": "receiving_yards", "mae": 15.0, "sample_size": 2}]
