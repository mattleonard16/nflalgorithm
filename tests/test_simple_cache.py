"""HTTP cache provenance required by fail-closed live-odds validation."""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
import requests
from requests.adapters import HTTPAdapter
from urllib3 import HTTPResponse as U3HTTPResponse

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


class _CannedAdapter(HTTPAdapter):
    """Serves one canned JSON body without touching the network."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def send(self, request, **kwargs):  # noqa: ANN001, ANN003
        self.calls += 1
        raw = U3HTTPResponse(
            body=io.BytesIO(b'{"ok": true}'),
            status=200,
            headers={"Content-Type": "application/json"},
            preload_content=False,
        )
        raw._request_url = request.url
        return self.build_response(request, raw)


def _offline_capable_client(tmp_path, monkeypatch):
    """Real client with an isolated cache dir and a canned transport."""
    monkeypatch.setattr("scripts.simple_cache.config.cache_dir", tmp_path)
    monkeypatch.setattr("scripts.simple_cache.config.api.cache_offline_mode", False)
    monkeypatch.setattr("scripts.simple_cache.config.api.force_cache_refresh", False)
    client = SimpleCachedClient()
    adapter = _CannedAdapter()
    client.session.mount("https://cache.test", adapter)
    return client, adapter


def test_get_from_cache_round_trips_a_cached_response(tmp_path, monkeypatch) -> None:
    client, _ = _offline_capable_client(tmp_path, monkeypatch)
    first = client.get("https://cache.test/odds", params={"week": 1})
    assert first.headers["X-Cache"] == "MISS"

    cached = client._get_from_cache("https://cache.test/odds", {"week": 1})

    assert cached is not None, "cache lookup must find the entry the session just stored"
    assert cached.content == b'{"ok": true}'
    assert "X-Cache-Created-At" in cached.headers


def test_get_from_cache_miss_returns_none_without_error(tmp_path, monkeypatch) -> None:
    client, _ = _offline_capable_client(tmp_path, monkeypatch)

    assert client._get_from_cache("https://cache.test/never-fetched") is None


def test_fresh_cache_hit_short_circuits_network(tmp_path, monkeypatch) -> None:
    client, adapter = _offline_capable_client(tmp_path, monkeypatch)
    client.get("https://cache.test/odds", params={"week": 1})

    second = client.get("https://cache.test/odds", params={"week": 1})

    assert adapter.calls == 1
    assert second.headers["X-Cache"] == "HIT"


def test_offline_mode_serves_cache_and_fails_closed_when_empty(tmp_path, monkeypatch) -> None:
    client, _ = _offline_capable_client(tmp_path, monkeypatch)
    client.get("https://cache.test/odds", params={"week": 1})
    monkeypatch.setattr("scripts.simple_cache.config.api.cache_offline_mode", True)

    offline = client.get("https://cache.test/odds", params={"week": 1})
    assert offline.headers["X-Cache"] == "HIT-OFFLINE"

    with pytest.raises(requests.ConnectionError):
        client.get("https://cache.test/never-fetched")


def test_get_cache_stats_counts_entries(tmp_path, monkeypatch) -> None:
    client, _ = _offline_capable_client(tmp_path, monkeypatch)
    assert client.get_cache_stats()["cached_urls"] == 0

    client.get("https://cache.test/odds", params={"week": 1})

    assert client.get_cache_stats()["cached_urls"] == 1
