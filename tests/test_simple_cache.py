"""HTTP cache provenance required by fail-closed live-odds validation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
    response = requests.Response()
    response.headers["X-Cache-Created-At"] = (
        datetime.now(timezone.utc) - timedelta(minutes=31)
    ).isoformat()

    assert client._is_cache_expired(response, "odds") is True


def test_cache_without_source_timestamp_is_never_treated_as_fresh() -> None:
    client = SimpleCachedClient.__new__(SimpleCachedClient)
    response = requests.Response()

    SimpleCachedClient._annotate_provenance(response, "HIT")

    assert response.headers["X-Cache"] == "HIT"
    assert "X-Cache-Created-At" not in response.headers
    assert "X-Cache-Age-Seconds" not in response.headers
    assert client._is_cache_expired(response, "odds") is True
