"""Tests for NBA endpoint caching."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.cache import nba_cache, make_cache_key
from api.nba_router import router


def _make_client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _setup_db(monkeypatch, tmp_path):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    import config as cfg

    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")
    monkeypatch.setattr(cfg.config.database, "path", db_path)
    return db_path


class TestNbaCacheInstance:
    def test_nba_cache_exists(self):
        """nba_cache should be a separate instance from value_bets_cache."""
        from api.cache import value_bets_cache

        assert nba_cache is not value_bets_cache

    def test_nba_cache_default_ttl(self):
        """NBA cache should have 600s (10-min) default TTL."""
        assert nba_cache._default_ttl == 600

    def test_nba_cache_max_size(self):
        """NBA cache should support 200 entries."""
        assert nba_cache._max_size == 200


class TestNbaMetaCache:
    def setup_method(self):
        nba_cache.invalidate_all()

    def test_meta_cache_hit(self, monkeypatch, tmp_path):
        """Second call to /meta should use cache, not DB."""
        _setup_db(monkeypatch, tmp_path)

        from utils.db import execute

        execute(
            "CREATE TABLE IF NOT EXISTS nba_player_game_logs "
            "(season INT, game_date TEXT, player_id INT, game_id TEXT)"
        )
        execute(
            "INSERT INTO nba_player_game_logs VALUES (2025, '2026-03-10', 1, 'G1')"
        )
        execute(
            "INSERT INTO nba_player_game_logs VALUES (2025, '2026-03-10', 2, 'G1')"
        )

        client = _make_client()

        r1 = client.get("/api/nba/meta")
        assert r1.status_code == 200
        assert r1.json()["total_players"] == 2

        # Delete data — cache should still serve old result
        execute("DELETE FROM nba_player_game_logs")

        r2 = client.get("/api/nba/meta")
        assert r2.status_code == 200
        assert r2.json()["total_players"] == 2  # cached!
