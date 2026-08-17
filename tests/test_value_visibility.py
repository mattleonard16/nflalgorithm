"""Production visibility rules for materialized NFL value rows."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from api.value_visibility import value_visibility_scope
from schema_migrations import MigrationManager
from utils.db import execute, fetchall


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "visibility.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    import config as cfg

    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")
    MigrationManager(db_path).run()
    return db_path


def _seed_run(run_id: str, *, status: str, valid: bool) -> None:
    now = datetime.now(timezone.utc).isoformat()
    execute(
        """
        INSERT INTO pipeline_runs (
            run_id, season, week, status, stages_requested, stages_completed,
            started_at, finished_at, updated_at
        ) VALUES (?, 2026, 1, ?, 6, 6, ?, ?, ?)
        """,
        (run_id, status, now, now, now),
    )
    execute(
        """
        INSERT INTO pipeline_odds_validations (
            run_id, attempt, valid, reason_code, reason, metrics_json, validated_at
        ) VALUES (?, 1, ?, ?, ?, '{}', ?)
        """,
        (
            run_id,
            int(valid),
            "validated" if valid else "stale_snapshot",
            "valid" if valid else "stale",
            now,
        ),
    )


def _seed_game(game_id: str) -> None:
    execute(
        """
        INSERT INTO games (
            game_id, season, week, home_team, away_team, kickoff_utc, game_date
        ) VALUES (?, 2026, 1, 'BUF', 'KC', '2026-09-10T00:00:00Z', '2026-09-10')
        """,
        (game_id,),
    )


def _seed_value_row(
    player_id: str,
    *,
    event_id: str,
    sportsbook: str = "DraftKings",
    published_run_id: str | None,
) -> None:
    execute(
        """
        INSERT INTO materialized_value_view (
            season, week, player_id, event_id, team, market, sportsbook,
            line, price, side, mu, sigma, p_win, edge_percentage, expected_roi,
            kelly_fraction, stake, generated_at, published_run_id
        ) VALUES (
            2026, 1, ?, ?, 'KC', 'passing_yards', ?,
            275.5, -110, 'over', 290.0, 25.0, 0.62, 0.10, 0.15,
            0.02, 20.0, datetime('now'), ?
        )
        """,
        (player_id, event_id, sportsbook, published_run_id),
    )


def _seed_visibility_matrix() -> None:
    _seed_run("valid-run", status="completed", valid=True)
    _seed_run("failed-run", status="failed", valid=True)
    _seed_run("invalid-run", status="completed", valid=False)

    for game_id in (
        "2026_01_KC_BUF",
        "2026_01_BAL_CIN",
        "2026_01_DAL_PHI",
        "2026_01_GB_CHI",
        "2026_01_MIA_NE",
    ):
        _seed_game(game_id)

    _seed_value_row("visible", event_id="2026_01_KC_BUF", published_run_id="valid-run")
    _seed_value_row("unpublished", event_id="2026_01_BAL_CIN", published_run_id=None)
    _seed_value_row("failed", event_id="2026_01_DAL_PHI", published_run_id="failed-run")
    _seed_value_row("invalid", event_id="2026_01_GB_CHI", published_run_id="invalid-run")
    _seed_value_row("unjoinable", event_id="provider-event", published_run_id="valid-run")
    _seed_value_row(
        "synthetic",
        event_id="2026_01_MIA_NE",
        sportsbook=" simbook ",
        published_run_id="valid-run",
    )


def _visible_player_ids(*, demo_mode: bool) -> list[str]:
    predicate, params = value_visibility_scope(demo_mode=demo_mode)
    return [
        str(row[0])
        for row in fetchall(
            f"SELECT v.player_id FROM materialized_value_view v "
            f"WHERE {predicate} ORDER BY v.player_id",
            params,
        )
    ]


def test_production_scope_excludes_every_disqualified_row(db) -> None:
    _seed_visibility_matrix()

    assert _visible_player_ids(demo_mode=False) == ["visible"]


def test_demo_scope_preserves_fixture_rows(db) -> None:
    _seed_visibility_matrix()

    assert _visible_player_ids(demo_mode=True) == [
        "failed",
        "invalid",
        "synthetic",
        "unjoinable",
        "unpublished",
        "visible",
    ]


def test_latest_odds_validation_attempt_controls_visibility(db) -> None:
    _seed_run("retried-run", status="completed", valid=True)
    _seed_game("2026_01_KC_BUF")
    _seed_value_row(
        "retried",
        event_id="2026_01_KC_BUF",
        published_run_id="retried-run",
    )
    execute("""
        INSERT INTO pipeline_odds_validations (
            run_id, attempt, valid, reason_code, reason, metrics_json, validated_at
        ) VALUES ('retried-run', 2, 0, 'stale_snapshot', 'stale', '{}', datetime('now'))
        """)

    assert _visible_player_ids(demo_mode=False) == []
