"""Tests for in-memory endpoint cache (P2: Performance)."""

import time

from api.cache import EndpointCache, make_cache_key


class TestEndpointCache:
    def test_set_and_get(self):
        cache = EndpointCache(default_ttl=60)
        cache.set("key1", {"data": "value"})
        assert cache.get("key1") == {"data": "value"}

    def test_miss_returns_none(self):
        cache = EndpointCache(default_ttl=60)
        assert cache.get("missing") is None

    def test_ttl_expiration(self):
        cache = EndpointCache(default_ttl=1)
        cache.set("key1", "value", ttl=0)
        time.sleep(0.05)
        assert cache.get("key1") is None

    def test_invalidate_by_prefix(self):
        cache = EndpointCache(default_ttl=60)
        cache.set("bets:s2025:w22:why=false", "data1")
        cache.set("bets:s2025:w22:why=true", "data2")
        cache.set("other:key", "data3")

        removed = cache.invalidate("bets:")
        assert removed == 2
        assert cache.get("bets:s2025:w22:why=false") is None
        assert cache.get("other:key") == "data3"

    def test_invalidate_all(self):
        cache = EndpointCache(default_ttl=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.invalidate_all()
        assert cache.size() == 0

    def test_size(self):
        cache = EndpointCache(default_ttl=60)
        assert cache.size() == 0
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.size() == 2

    def test_max_size_eviction(self):
        cache = EndpointCache(default_ttl=60, max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        assert cache.size() == 3
        cache.set("d", 4)  # Should evict oldest
        assert cache.size() == 3
        assert cache.get("d") == 4


class TestMakeCacheKey:
    def test_basic_key(self):
        key = make_cache_key("value-bets", season=2025, week=22)
        assert key == "value-bets:season=2025:week=22"

    def test_ignores_none(self):
        key = make_cache_key("value-bets", season=2025, sportsbook=None)
        assert "sportsbook" not in key

    def test_sorted_params(self):
        key1 = make_cache_key("ep", b=2, a=1)
        key2 = make_cache_key("ep", a=1, b=2)
        assert key1 == key2

    def test_include_why(self):
        key_no = make_cache_key("value-bets", season=2025, include_why=False)
        key_yes = make_cache_key("value-bets", season=2025, include_why=True)
        assert key_no != key_yes
