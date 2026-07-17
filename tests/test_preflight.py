"""Focused tests for actionable startup diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

from scripts.preflight import (
    check_database,
    check_odds_key,
    check_private_api,
    check_private_modules,
    check_runtime_config,
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
