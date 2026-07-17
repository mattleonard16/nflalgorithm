"""API liveness and readiness contract tests."""

from __future__ import annotations

import sqlite3

import pytest

from schema_migrations import MigrationManager


@pytest.fixture()
def client(tmp_path, monkeypatch):
    database = tmp_path / "readiness.db"
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", str(database))

    import config as config_module

    monkeypatch.setattr(config_module.config.database, "backend", "sqlite")
    monkeypatch.setattr(config_module.config.database, "path", str(database))
    MigrationManager(database).run()

    from fastapi.testclient import TestClient

    from api.application import app

    with TestClient(app) as test_client:
        yield test_client, database, config_module.config


def test_liveness_does_not_require_database(client) -> None:
    test_client, database, _config = client
    database.unlink()

    response = test_client.get("/livez")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "api"}


def test_readiness_passes_with_database_and_migrations(client) -> None:
    test_client, _database, _config = client

    response = test_client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["checks"] == {"database": "ok", "migrations": "ok"}


def test_readiness_fails_when_migration_table_is_missing(client) -> None:
    test_client, database, _config = client
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE pipeline_runs")

    response = test_client.get("/readyz")

    assert response.status_code == 503
    assert "migrations are incomplete" in response.json()["detail"]
    assert "pipeline_runs" in response.json()["detail"]


def test_readiness_fails_when_database_is_unavailable(client, tmp_path) -> None:
    test_client, _database, config = client
    config.database.path = str(tmp_path / "missing-parent" / "app.db")

    response = test_client.get("/readyz")

    assert response.status_code == 503
    assert "Verify DB_BACKEND" in response.json()["detail"]
