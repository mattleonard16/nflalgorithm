"""Algorithm slate endpoint: projections without a published betting card.

Slate membership is public; projected values (mu/sigma) require a session.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from schema_migrations import MigrationManager
from utils.db import execute


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "projections-api.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    import config as cfg

    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")
    MigrationManager(db_path).run()
    return db_path


@pytest.fixture()
def client(db):
    from fastapi.testclient import TestClient

    from api.application import app

    with TestClient(app) as test_client:
        yield test_client


def _seed_session(session_id: str = "slate-session") -> dict[str, str]:
    """Insert a signed-in user and return the Authorization header for it."""
    now = datetime.now(timezone.utc)
    execute(
        """
        INSERT INTO users (id, email, password_hash, name, subscription_tier, bankroll,
                           created_at, updated_at)
        VALUES ('user-slate', 'slate@example.com', 'x', 'Slate Viewer', 'free', 1000.0, ?, ?)
        """,
        params=(now.isoformat(), now.isoformat()),
    )
    execute(
        """
        INSERT INTO user_sessions (session_id, user_id, expires_at, created_at)
        VALUES (?, 'user-slate', ?, ?)
        """,
        params=(session_id, (now + timedelta(days=1)).isoformat(), now.isoformat()),
    )
    return {"Authorization": f"Bearer {session_id}"}


def _seed_snapshot(
    *,
    player_id: str,
    position: str,
    depth_rank: int,
    is_starter: int,
    expected_rushing_attempts: float = 0.0,
    expected_targets: float = 0.0,
    expected_passing_attempts: float = 0.0,
) -> None:
    execute(
        """
        INSERT INTO nfl_player_context_snapshots
            (season, week, gsis_id, player_id, team, position, roster_status,
             depth_position, depth_rank, is_starter, injury_status, practice_status,
             primary_injury, expected_snap_count, expected_snap_percentage,
             expected_rushing_attempts, expected_targets, expected_passing_attempts,
             expected_target_share, expected_air_yards, expected_yac_yards,
             expected_red_zone_touches, expected_game_script, is_rookie, is_new_team,
             uncertainty_multiplier, prior_source, source_updated_at, captured_at)
        VALUES
            (2026, 1, ?, ?, 'KC', ?, 'ACT', ?, ?, ?, NULL, NULL, NULL, 40, 62,
             ?, ?, ?, 0.18, 60, 20, 1, 0, 0, 0, 1.0, 'depth_chart',
             '2026-09-01T00:00:00Z', '2026-09-01T00:00:00Z')
        """,
        params=(
            player_id,
            player_id,
            position,
            position,
            depth_rank,
            is_starter,
            expected_rushing_attempts,
            expected_targets,
            expected_passing_attempts,
        ),
    )


def _seed_projection(
    *,
    player_id: str = "KC_xavier_worthy",
    name: str = "Xavier Worthy",
    position: str = "WR",
    market: str = "receiving_yards",
    mu: float = 41.8,
    depth_rank: int | None = 1,
    is_starter: int = 1,
) -> None:
    execute(
        """
        INSERT INTO player_dim
            (player_id, player_name, position, team, last_season, last_week, updated_at)
        VALUES (?, ?, ?, 'KC', 2025, 18, datetime('now'))
        """,
        params=(player_id, name, position),
    )
    execute(
        """
        INSERT INTO weekly_projections
            (season, week, player_id, team, opponent, market, mu, sigma,
             model_version, featureset_hash, generated_at)
        VALUES (2026, 1, ?, 'KC', 'LAC', ?, ?, 12.4, 'causal_asof_v1', 'abc123', datetime('now'))
        """,
        params=(player_id, market, mu),
    )
    if depth_rank is not None:
        _seed_snapshot(
            player_id=player_id,
            position=position,
            depth_rank=depth_rank,
            is_starter=is_starter,
            expected_rushing_attempts=12.0 if market == "rushing_yards" else 0.0,
            expected_targets=7.0 if market == "receiving_yards" else 0.0,
            expected_passing_attempts=34.0 if market == "passing_yards" else 0.0,
        )


def _seed_game(*, home_team: str = "KC", away_team: str = "LAC") -> None:
    execute(
        """
        INSERT INTO games
            (game_id, season, week, home_team, away_team, kickoff_utc, game_date)
        VALUES (?, 2026, 1, ?, ?, '2026-09-10T17:00:00Z', '2026-09-10')
        """,
        params=(f"2026_01_{away_team}_{home_team}", home_team, away_team),
    )


def test_projection_weeks_are_empty_without_rows(client) -> None:
    assert client.get("/api/projections/weeks").json() == {"available_weeks": []}


def test_projection_weeks_list_model_output_not_published_cards(client, db) -> None:
    _seed_projection()
    payload = client.get("/api/projections/weeks").json()
    assert payload["available_weeks"] == [{"season": 2026, "week": 1}]


def test_projections_return_algorithm_picks(client, db) -> None:
    _seed_projection()
    _seed_game()
    payload = client.get("/api/projections?season=2026&week=1", headers=_seed_session()).json()
    assert payload["total_count"] == 1
    assert payload["values_visible"] is True
    pick = payload["picks"][0]
    assert pick["player_name"] == "Xavier Worthy"
    assert pick["position"] == "WR"
    assert pick["market"] == "receiving_yards"
    assert pick["market_label"] == "Rec"
    assert pick["mu"] == 41.8
    assert pick["team"] == "KC"
    assert pick["home_team"] == "KC"
    assert pick["away_team"] == "LAC"
    assert pick["is_home"] is True
    assert pick["depth_rank"] == 1
    assert pick["is_starter"] is True
    assert pick["injury_status"] is None
    assert pick["roster_status"] == "ACT"


def test_projections_carry_the_road_side_of_the_matchup(client, db) -> None:
    _seed_projection()
    _seed_game(home_team="LAC", away_team="KC")
    pick = client.get("/api/projections?season=2026&week=1").json()["picks"][0]
    assert pick["home_team"] == "LAC"
    assert pick["away_team"] == "KC"
    assert pick["is_home"] is False


def test_projections_leave_matchup_sides_null_without_a_scheduled_game(client, db) -> None:
    _seed_projection()
    pick = client.get("/api/projections?season=2026&week=1").json()["picks"][0]
    assert pick["home_team"] is None
    assert pick["away_team"] is None
    assert pick["is_home"] is None


def test_projections_filter_by_market_and_position(client, db) -> None:
    _seed_projection()
    _seed_projection(
        player_id="BAL_l_jackson",
        name="Lamar Jackson",
        position="QB",
        market="passing_yards",
        mu=220.0,
    )
    receiving = client.get("/api/projections?season=2026&week=1&market=receiving_yards").json()
    assert {row["player_id"] for row in receiving["picks"]} == {"KC_xavier_worthy"}
    qbs = client.get("/api/projections?season=2026&week=1&position=QB").json()
    assert {row["player_id"] for row in qbs["picks"]} == {"BAL_l_jackson"}


def _seed_starter_and_backup_qbs() -> None:
    _seed_projection(
        player_id="BAL_l_jackson",
        name="Lamar Jackson",
        position="QB",
        market="passing_yards",
        mu=220.0,
        depth_rank=1,
        is_starter=1,
    )
    _seed_projection(
        player_id="CLE_t_siemian",
        name="Trevor Siemian",
        position="QB",
        market="passing_yards",
        mu=180.0,
        depth_rank=3,
        is_starter=0,
    )


def test_likely_filter_drops_backup_quarterbacks(client, db) -> None:
    _seed_starter_and_backup_qbs()
    payload = client.get("/api/projections?season=2026&week=1").json()
    assert {row["player_id"] for row in payload["picks"]} == {"BAL_l_jackson"}
    assert payload["total_count"] == 1


def test_likely_filter_keeps_the_starting_quarterback(client, db) -> None:
    _seed_starter_and_backup_qbs()
    payload = client.get("/api/projections?season=2026&week=1&likely=true").json()
    starter = payload["picks"][0]
    assert starter["player_id"] == "BAL_l_jackson"
    assert starter["depth_rank"] == 1
    assert starter["is_starter"] is True


def test_likely_false_returns_the_full_projection_file(client, db) -> None:
    _seed_starter_and_backup_qbs()
    payload = client.get("/api/projections?season=2026&week=1&likely=false").json()
    assert {row["player_id"] for row in payload["picks"]} == {
        "BAL_l_jackson",
        "CLE_t_siemian",
    }
    backup = next(row for row in payload["picks"] if row["player_id"] == "CLE_t_siemian")
    assert backup["depth_rank"] == 3
    assert backup["is_starter"] is False


def test_signed_out_slate_hides_projected_values(client, db) -> None:
    _seed_projection()
    payload = client.get("/api/projections?season=2026&week=1").json()
    assert payload["values_visible"] is False
    pick = payload["picks"][0]
    assert pick["mu"] is None
    assert pick["sigma"] is None
    assert pick["player_name"] == "Xavier Worthy"


def test_invalid_session_is_treated_as_signed_out(client, db) -> None:
    _seed_projection()
    payload = client.get(
        "/api/projections?season=2026&week=1",
        headers={"Authorization": "Bearer not-a-session"},
    ).json()
    assert payload["values_visible"] is False
    assert payload["picks"][0]["mu"] is None


def test_signed_in_slate_carries_projected_values(client, db) -> None:
    _seed_projection()
    headers = _seed_session()
    payload = client.get("/api/projections?season=2026&week=1", headers=headers).json()
    assert payload["values_visible"] is True
    pick = payload["picks"][0]
    assert pick["mu"] == 41.8
    assert pick["sigma"] == 12.4


def test_duplicate_snapshot_rows_do_not_duplicate_picks(client, db) -> None:
    """Snapshot PK is (season, week, gsis_id); a second capture under the same
    player_id must collapse to the latest row, not fan the pick out."""
    _seed_projection()
    execute(
        """
        INSERT INTO nfl_player_context_snapshots
            (season, week, gsis_id, player_id, team, position, roster_status,
             depth_position, depth_rank, is_starter, injury_status, practice_status,
             primary_injury, expected_snap_count, expected_snap_percentage,
             expected_rushing_attempts, expected_targets, expected_passing_attempts,
             expected_target_share, expected_air_yards, expected_yac_yards,
             expected_red_zone_touches, expected_game_script, is_rookie, is_new_team,
             uncertainty_multiplier, prior_source, source_updated_at, captured_at)
        VALUES
            (2026, 1, 'GSIS-DUP', 'KC_xavier_worthy', 'KC', 'WR', 'ACT', 'WR', 2, 0,
             'Questionable', NULL, NULL, 40, 62, 0, 7.0, 0, 0.18, 60, 20, 1, 0, 0, 0,
             1.0, 'depth_chart', '2026-09-02T00:00:00Z', '2026-09-02T00:00:00Z')
        """
    )
    payload = client.get("/api/projections?season=2026&week=1").json()
    assert payload["total_count"] == 1
    pick = payload["picks"][0]
    assert pick["depth_rank"] == 2
    assert pick["injury_status"] == "Questionable"
