"""Public API reads expose only published, gradeable NFL value rows."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from api.cache import value_bets_cache
from schema_migrations import MigrationManager
from utils.db import execute
from utils.odds_quality import SYNTHETIC_SPORTSBOOK


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "api-visibility.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    import config as cfg

    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")
    monkeypatch.setattr(cfg.config.api, "demo_mode", False)
    MigrationManager(db_path).run()
    value_bets_cache.invalidate_all()
    return db_path


@pytest.fixture()
def client(db):
    from fastapi.testclient import TestClient

    from api.application import app
    from api.pipeline_router import require_pipeline_operator, require_pipeline_reader

    app.dependency_overrides[require_pipeline_operator] = lambda: "test-operator"
    app.dependency_overrides[require_pipeline_reader] = lambda: "test-reader"
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    value_bets_cache.invalidate_all()


def _seed_run(run_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    execute(
        """
        INSERT INTO pipeline_runs (
            run_id, season, week, status, stages_requested, stages_completed,
            started_at, finished_at, updated_at
        ) VALUES (?, 2026, 1, 'completed', 6, 6, ?, ?, ?)
        """,
        (run_id, now, now, now),
    )
    execute(
        """
        INSERT INTO pipeline_odds_validations (
            run_id, attempt, valid, reason_code, reason, metrics_json, validated_at
        ) VALUES (?, 1, 1, 'validated', 'valid', '{}', ?)
        """,
        (run_id, now),
    )


def _seed_game(game_id: str, home: str, away: str) -> None:
    execute(
        """
        INSERT INTO games (
            game_id, season, week, home_team, away_team, kickoff_utc, game_date
        ) VALUES (?, 2026, 1, ?, ?, '2026-09-10T00:00:00Z', '2026-09-10')
        """,
        (game_id, home, away),
    )


def _seed_bet(
    player_id: str,
    *,
    event_id: str,
    sportsbook: str,
    published_run_id: str | None,
    stake: float,
) -> None:
    execute(
        """
        INSERT INTO player_dim (
            player_id, player_name, position, team, last_season, last_week, updated_at
        ) VALUES (?, ?, 'WR', 'KC', 2026, 1, datetime('now'))
        """,
        (player_id, f"Player {player_id}"),
    )
    execute(
        """
        INSERT INTO materialized_value_view (
            season, week, player_id, event_id, team, market, sportsbook,
            line, price, side, mu, sigma, p_win, edge_percentage, expected_roi,
            kelly_fraction, stake, generated_at, confidence_score, confidence_tier,
            published_run_id
        ) VALUES (
            2026, 1, ?, ?, 'KC', 'receiving_yards', ?,
            65.5, -110, 'over', 72.0, 10.0, 0.62, 0.10, 0.15,
            0.02, ?, datetime('now'), 0.82, 'Premium', ?
        )
        """,
        (player_id, event_id, sportsbook, stake, published_run_id),
    )


def _seed_mixed_card() -> None:
    _seed_run("published-run")
    _seed_run("empty-run")
    _seed_game("2026_01_KC_BUF", "BUF", "KC")
    _seed_game("2026_01_BAL_CIN", "CIN", "BAL")
    _seed_game("2026_01_DAL_PHI", "PHI", "DAL")

    _seed_bet(
        "visible",
        event_id="2026_01_KC_BUF",
        sportsbook="DraftKings",
        published_run_id="published-run",
        stake=20.0,
    )
    _seed_bet(
        "synthetic",
        event_id="2026_01_BAL_CIN",
        sportsbook=SYNTHETIC_SPORTSBOOK,
        published_run_id="published-run",
        stake=99.0,
    )
    _seed_bet(
        "unpublished",
        event_id="2026_01_DAL_PHI",
        sportsbook="FanDuel",
        published_run_id=None,
        stake=88.0,
    )
    _seed_bet(
        "unjoinable",
        event_id="provider-event",
        sportsbook="BetMGM",
        published_run_id="published-run",
        stake=77.0,
    )


def test_public_value_surfaces_share_production_visibility(client, db) -> None:
    _seed_mixed_card()

    meta = client.get("/api/meta").json()
    assert meta["available_weeks"] == [{"season": 2026, "week": 1}]
    assert meta["sportsbooks"] == ["DraftKings"]

    value_bets = client.get("/api/value-bets?season=2026&week=1").json()
    assert [bet["player_id"] for bet in value_bets["bets"]] == ["visible"]

    by_market = client.get("/api/analytics/by-market?season=2026&week=1").json()
    assert by_market["by_market"][0]["bet_count"] == 1

    by_position = client.get("/api/analytics/by-position?season=2026&week=1").json()
    assert by_position["by_position"][0]["bet_count"] == 1

    edge_distribution = client.get("/api/analytics/edge-distribution?season=2026&week=1").json()
    assert sum(edge_distribution["counts"]) == 1

    risk = client.get("/api/analytics/risk-summary?season=2026&week=1").json()
    assert risk["total_stake"] == 20.0

    csv_export = client.get("/api/export/csv?season=2026&week=1").text
    assert "Player visible" in csv_export
    assert "Player synthetic" not in csv_export
    assert "Player unpublished" not in csv_export

    bundle = client.get("/api/export/bundle?season=2026&week=1").json()
    assert bundle["total_bets"] == 1
    assert bundle["bets"][0]["player_id"] == "visible"
    assert bundle["pipeline_run"]["run_id"] == "published-run"

    why = client.get("/api/explain/unpublished/receiving_yards?season=2026&week=1").json()
    assert why["why"]["confidence"]["total"] is None


def test_review_rejects_a_run_without_public_bets(client, db) -> None:
    _seed_mixed_card()

    response = client.post("/api/run/empty-run/review?season=2026&week=1")

    assert response.status_code == 409
    assert response.json()["detail"] == "Run has no reviewable published bets"


def test_review_routes_require_server_side_authentication(client, db) -> None:
    from api.application import app
    from api.pipeline_router import require_pipeline_operator, require_pipeline_reader

    operator_override = app.dependency_overrides.pop(require_pipeline_operator)
    reader_override = app.dependency_overrides.pop(require_pipeline_reader)
    try:
        assert client.post("/api/run/any/review?season=2026&week=1").status_code == 401
        assert client.get("/api/run/any/review-status?season=2026&week=1").status_code == 401
    finally:
        app.dependency_overrides[require_pipeline_operator] = operator_override
        app.dependency_overrides[require_pipeline_reader] = reader_override


def test_demo_mode_restores_fixture_rows(client, db, monkeypatch) -> None:
    _seed_mixed_card()
    import config as cfg

    monkeypatch.setattr(cfg.config.api, "demo_mode", True)
    value_bets_cache.invalidate_all()

    response = client.get("/api/value-bets?season=2026&week=1")

    assert {bet["player_id"] for bet in response.json()["bets"]} == {
        "synthetic",
        "unjoinable",
        "unpublished",
        "visible",
    }


def test_metadata_database_failure_is_not_reported_as_an_empty_card(client, monkeypatch) -> None:
    def fail_query(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr("api.server.read_dataframe", fail_query)

    response = client.get("/api/meta")

    assert response.status_code == 500
    assert response.json()["detail"] == "Metadata unavailable"
