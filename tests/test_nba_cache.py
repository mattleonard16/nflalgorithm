"""Tests for NBA endpoint caching."""

from api.cache import nba_cache, make_cache_key


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
