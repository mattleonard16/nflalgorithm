"""Compatibility wrapper for the advanced cache manager.

This module re-exports the cache interfaces from ``scripts.cache_manager`` so
they can be imported as ``cache_manager`` (as expected by tests and CLI
utilities) while the implementation continues to live under ``scripts/``.
"""

from scripts.cache_manager import DatabaseCache, CachedAPIClient, cached_client  # noqa: F401
