"""Focused tests for actionable startup diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

from scripts.preflight import (
    check_database,
    check_odds_key,
    check_private_api,
    check_private_modules,
    check_runtime_config,
    evaluate_nfl_week_readiness,
)


def test_missing_sqlite_database_points_to_migration(tmp_path) -> None:
    config = SimpleNamespace(
        database=SimpleNamespace(backend="sqlite", path=str(tmp_path / "missing.db"))
    )

    diagnostics = check_database(config, check_schema=True)

    assert diagnostics[0].status == "fail"
    assert "does not exist" in diagnostics[0].message
    assert diagnostics[0].action == "Run `make migrate` to create and migrate it."


def test_mysql_without_url_has_actionable_error(monkeypatch) -> None:
    import config as config_module

    monkeypatch.setattr(config_module.config.database, "backend", "mysql")
    monkeypatch.setattr(config_module.config.database, "db_url", "")
    monkeypatch.delenv("DB_URL", raising=False)

    _, diagnostics = check_runtime_config()

    failure = next(item for item in diagnostics if item.name == "database_config")
    assert failure.status == "fail"
    assert "requires DB_URL" in failure.message
    assert "mysql+pymysql://" in (failure.action or "")


def test_missing_odds_key_warns_locally_and_fails_when_required() -> None:
    config = SimpleNamespace(api=SimpleNamespace(odds_api_key=""))

    local = check_odds_key(config, required=False)
    production = check_odds_key(config, required=True)

    assert local.status == "warn"
    assert production.status == "fail"
    assert "fail closed" in production.message


def test_missing_private_api_blocks_startup(tmp_path) -> None:
    diagnostic = check_private_api(tmp_path)

    assert diagnostic.status == "fail"
    assert "api/server.py" in diagnostic.message
    assert "before starting API services" in (diagnostic.action or "")


def test_missing_private_modules_explain_read_only_mode(tmp_path) -> None:
    diagnostic = check_private_modules(tmp_path, required=False)

    assert diagnostic.status == "warn"
    assert "Private NFL execution modules are unavailable" in diagnostic.message
    assert "API read-only features" in (diagnostic.action or "")


def test_pregame_readiness_requires_schedule_roster_and_history() -> None:
    diagnostics = evaluate_nfl_week_readiness(
        2026,
        1,
        phase="pre-run",
        counts={
            "games": 16,
            "games_with_kickoff": 16,
            "roster_players": 1800,
            "history_rows": 10000,
        },
    )

    assert all(item.status == "pass" for item in diagnostics)

    missing = evaluate_nfl_week_readiness(
        2026,
        1,
        phase="pre-run",
        counts={"games": 0, "games_with_kickoff": 0, "roster_players": 0, "history_rows": 0},
    )
    assert {item.name for item in missing if item.status == "fail"} == {
        "nfl_schedule",
        "nfl_roster",
        "nfl_history",
    }


def test_postrun_readiness_requires_persisted_worker_evidence() -> None:
    diagnostics = evaluate_nfl_week_readiness(
        2026,
        1,
        phase="post-run",
        counts={
            "games": 16,
            "games_with_kickoff": 16,
            "roster_players": 1800,
            "history_rows": 10000,
            "context_rows": 500,
            "projection_rows": 300,
            "odds_rows": 1200,
            "decision_rows": 40,
            "completed_runs": 1,
            "valid_odds_runs": 1,
            "artifact_rows": 1,
            "card_rows": 0,
        },
    )

    assert not any(item.status == "fail" for item in diagnostics)
    card = next(item for item in diagnostics if item.name == "nfl_card")
    assert card.status == "warn"
    assert "valid zero-play card" in card.message

    no_odds = dict(
        games=16,
        games_with_kickoff=16,
        roster_players=1800,
        history_rows=10000,
        context_rows=500,
        projection_rows=300,
        odds_rows=0,
        decision_rows=0,
        completed_runs=0,
        valid_odds_runs=0,
        artifact_rows=0,
        card_rows=0,
    )
    failed = evaluate_nfl_week_readiness(2026, 1, phase="post-run", counts=no_odds)

    assert "nfl_live_odds" in {item.name for item in failed if item.status == "fail"}
    assert "nfl_worker_run" in {item.name for item in failed if item.status == "fail"}
