"""HTTP cache provenance required by fail-closed live-odds validation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import requests

from scripts.simple_cache import SimpleCachedClient


def test_cache_provenance_includes_source_timestamp_and_age() -> None:
    response = requests.Response()
    created_at = datetime.now(timezone.utc) - timedelta(seconds=90)

    SimpleCachedClient._annotate_provenance(
        response,
        "HIT",
        created_at=created_at,
    )

    assert response.headers["X-Cache"] == "HIT"
    assert datetime.fromisoformat(response.headers["X-Cache-Created-At"]) == created_at
    assert float(response.headers["X-Cache-Age-Seconds"]) >= 90


def test_odds_cache_expiry_uses_odds_specific_ttl(monkeypatch) -> None:
    client = SimpleCachedClient.__new__(SimpleCachedClient)
    monkeypatch.setattr("scripts.simple_cache.config.pipeline.odds_max_age_seconds", 300)
    response = requests.Response()
    response.headers["X-Cache-Created-At"] = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).isoformat()

    assert client._is_cache_expired(response, "odds") is True


def test_odds_cache_ttl_uses_validation_max_age_seconds(monkeypatch) -> None:
    client = SimpleCachedClient.__new__(SimpleCachedClient)
    monkeypatch.setattr("scripts.simple_cache.config.pipeline.odds_max_age_seconds", 120)

    assert client._get_ttl_for_api("odds") == timedelta(seconds=120)


def test_odds_cache_inside_validation_window_is_fresh(monkeypatch) -> None:
    client = SimpleCachedClient.__new__(SimpleCachedClient)
    monkeypatch.setattr("scripts.simple_cache.config.pipeline.odds_max_age_seconds", 300)
    response = requests.Response()
    response.headers["X-Cache-Created-At"] = (
        datetime.now(timezone.utc) - timedelta(seconds=299)
    ).isoformat()

    assert client._is_cache_expired(response, "odds") is False


def test_cache_without_source_timestamp_is_never_treated_as_fresh() -> None:
    client = SimpleCachedClient.__new__(SimpleCachedClient)
    response = requests.Response()

    SimpleCachedClient._annotate_provenance(response, "HIT")

    assert response.headers["X-Cache"] == "HIT"
    assert "X-Cache-Created-At" not in response.headers
    assert "X-Cache-Age-Seconds" not in response.headers
    assert client._is_cache_expired(response, "odds") is True


def test_future_dated_cache_timestamp_is_never_treated_as_fresh() -> None:
    client = SimpleCachedClient.__new__(SimpleCachedClient)
    response = requests.Response()
    created_at = datetime.now(timezone.utc) + timedelta(minutes=5)

    SimpleCachedClient._annotate_provenance(response, "HIT", created_at=created_at)

    assert float(response.headers["X-Cache-Age-Seconds"]) < 0
    assert client._is_cache_expired(response, "odds") is True


def test_get_passes_odds_ttl_and_force_refresh_to_requests_cache(monkeypatch) -> None:
    client = SimpleCachedClient.__new__(SimpleCachedClient)
    client.rate_limiter = Mock()
    client.rate_limiter.consume.return_value = True
    response = requests.Response()
    response.status_code = 200
    response.headers = {}
    client.session = Mock()
    client.session.get.return_value = response
    monkeypatch.setattr(client, "_get_from_cache", Mock(return_value=None))
    monkeypatch.setattr("scripts.simple_cache.config.pipeline.odds_max_age_seconds", 180)
    monkeypatch.setattr("scripts.simple_cache.config.api.cache_offline_mode", False)
    monkeypatch.setattr("scripts.simple_cache.config.api.force_cache_refresh", False)

    client.get("https://example.test/odds", api_type="odds", force_refresh=True)

    client.session.get.assert_called_once_with(
        "https://example.test/odds",
        params=None,
        timeout=30,
        expire_after=timedelta(seconds=180),
        force_refresh=True,
    )
