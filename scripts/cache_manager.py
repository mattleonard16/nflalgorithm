"""
Advanced API Caching System for NFL Algorithm
============================================

Implements comprehensive caching with:
- HTTP caching via requests-cache
- Database persistence layer
- Stale-while-revalidate pattern
- Rate limiting with token bucket
- Cache metrics and monitoring
"""

import hashlib
import json
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple, Union, List
import threading
import requests
import requests_cache
from dataclasses import asdict
import logging

from config import config

logger = logging.getLogger(__name__)


class TokenBucket:
    """Rate limiting using token bucket algorithm"""
    
    def __init__(self, capacity: int, refill_rate: int):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per minute
        self.tokens = capacity
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful."""
        now = time.time()
        
        # Add tokens based on time passed
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * (self.refill_rate / 60.0)  # convert to per second
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def wait_time(self, tokens: int = 1) -> float:
        """Returns seconds to wait before tokens are available"""
        if self.tokens >= tokens:
            return 0.0
        needed = tokens - self.tokens
        return needed / (self.refill_rate / 60.0)


class DatabaseCache:
    """Database-based caching with stale-while-revalidate support"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.database.path
        self._init_cache_tables()
    
    def _init_cache_tables(self):
        """Initialize cache tables"""
        conn = sqlite3.connect(self.db_path)
        try:
            # API Cache table - stores API responses with TTL
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE NOT NULL,
                    endpoint TEXT NOT NULL,
                    params_hash TEXT,
                    data TEXT NOT NULL,
                    content_type TEXT DEFAULT 'application/json',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    hit_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    stale_served_count INTEGER DEFAULT 0
                )
            """)
            
            # Raw odds storage for analysis
            conn.execute("""
                CREATE TABLE IF NOT EXISTS odds_raw (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sportsbook TEXT NOT NULL,
                    sport TEXT DEFAULT 'americanfootball_nfl',
                    market TEXT NOT NULL,
                    raw_response TEXT NOT NULL,
                    response_size INTEGER,
                    fetch_duration_ms INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    api_request_id TEXT
                )
            """)
            
            # Cache performance metrics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_type TEXT NOT NULL, -- 'hit', 'miss', 'stale_served', 'refresh'
                    cache_key TEXT NOT NULL,
                    endpoint TEXT,
                    response_time_ms INTEGER,
                    cache_age_seconds INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    additional_data TEXT -- JSON for extra metrics
                )
            """)
            
            # Indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_cache_key ON api_cache(cache_key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_cache_expires ON api_cache(expires_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_metrics_type ON cache_metrics(metric_type)")
            
            conn.commit()
        finally:
            conn.close()
    
    def _generate_cache_key(self, endpoint: str, params: Dict[str, Any] = None) -> str:
        """Generate deterministic cache key"""
        key_data = {
            'endpoint': endpoint,
            'params': params or {}
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, endpoint: str, params: Dict[str, Any] = None, 
            allow_stale: bool = True) -> Tuple[Optional[Any], bool]:
        """
        Get cached data
        Returns: (data, is_stale)
        """
        cache_key = self._generate_cache_key(endpoint, params)
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("""
                SELECT data, expires_at, created_at, hit_count
                FROM api_cache
                WHERE cache_key = ?
            """, (cache_key,))
            
            row = cursor.fetchone()
            if not row:
                self._record_metric('miss', cache_key, endpoint)
                return None, False
            
            data_str, expires_at_str, created_at_str, hit_count = row
            expires_at = datetime.fromisoformat(expires_at_str)
            created_at = datetime.fromisoformat(created_at_str)
            now = datetime.now()
            
            # Update hit count and last accessed
            conn.execute("""
                UPDATE api_cache 
                SET hit_count = hit_count + 1, last_accessed = CURRENT_TIMESTAMP
                WHERE cache_key = ?
            """, (cache_key,))
            conn.commit()
            
            # Check if expired
            is_stale = now > expires_at
            
            if is_stale and not allow_stale:
                self._record_metric('miss', cache_key, endpoint, 
                                  cache_age_seconds=int((now - created_at).total_seconds()))
                return None, True
            
            # Parse data
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                data = data_str  # Return as string if not JSON
            
            if is_stale:
                # Record stale serve
                conn.execute("""
                    UPDATE api_cache 
                    SET stale_served_count = stale_served_count + 1
                    WHERE cache_key = ?
                """, (cache_key,))
                conn.commit()
                
                self._record_metric('stale_served', cache_key, endpoint,
                                  cache_age_seconds=int((now - expires_at).total_seconds()))
            else:
                self._record_metric('hit', cache_key, endpoint,
                                  cache_age_seconds=int((now - created_at).total_seconds()))
            
            return data, is_stale
            
        finally:
            conn.close()
    
    def set(self, endpoint: str, data: Any, ttl_seconds: int, 
            params: Dict[str, Any] = None):
        """Store data in cache"""
        cache_key = self._generate_cache_key(endpoint, params)
        
        # Serialize data
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data)
            content_type = 'application/json'
        else:
            data_str = str(data)
            content_type = 'text/plain'
        
        expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
        params_hash = hashlib.md5(json.dumps(params or {}, sort_keys=True).encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO api_cache 
                (cache_key, endpoint, params_hash, data, content_type, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    endpoint = excluded.endpoint,
                    params_hash = excluded.params_hash,
                    data = excluded.data,
                    content_type = excluded.content_type,
                    expires_at = excluded.expires_at,
                    last_accessed = CURRENT_TIMESTAMP
                """,
                (cache_key, endpoint, params_hash, data_str, content_type, expires_at.isoformat()),
            )
            conn.commit()

            self._record_metric('refresh', cache_key, endpoint)

        finally:
            conn.close()
    
    def _record_metric(self, metric_type: str, cache_key: str, endpoint: str,
                      response_time_ms: int = None, cache_age_seconds: int = None,
                      additional_data: Dict = None):
        """Record cache performance metric"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO cache_metrics 
                (metric_type, cache_key, endpoint, response_time_ms, 
                 cache_age_seconds, additional_data)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                metric_type, cache_key, endpoint, response_time_ms,
                cache_age_seconds, json.dumps(additional_data) if additional_data else None
            ))
            conn.commit()
        finally:
            conn.close()
    
    def cleanup_expired(self, force: bool = False):
        """Remove expired cache entries"""
        if not force:
            # Keep expired entries for stale-while-revalidate
            cutoff = datetime.now() - timedelta(seconds=config.cache.stale_while_revalidate_window)
        else:
            cutoff = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                DELETE FROM api_cache 
                WHERE expires_at < ?
                """,
                (cutoff.isoformat(),),
            )
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired cache entries")
                
        finally:
            conn.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        conn = sqlite3.connect(self.db_path)
        try:
            # Cache entry counts
            cursor = conn.execute("SELECT COUNT(*) FROM api_cache")
            total_entries = cursor.fetchone()[0]
            
            cursor = conn.execute("""
                SELECT COUNT(*) FROM api_cache 
                WHERE expires_at > CURRENT_TIMESTAMP
            """)
            valid_entries = cursor.fetchone()[0]
            
            # Hit rate over last hour
            one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
            cursor = conn.execute("""
                SELECT metric_type, COUNT(*) 
                FROM cache_metrics 
                WHERE created_at > ?
                GROUP BY metric_type
            """, (one_hour_ago,))
            
            recent_metrics = dict(cursor.fetchall())
            
            hits = recent_metrics.get('hit', 0)
            misses = recent_metrics.get('miss', 0)
            stale_served = recent_metrics.get('stale_served', 0)
            
            total_requests = hits + misses + stale_served
            hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'total_entries': total_entries,
                'valid_entries': valid_entries,
                'expired_entries': total_entries - valid_entries,
                'hit_rate_percent': round(hit_rate, 2),
                'total_requests_last_hour': total_requests,
                'cache_hits': hits,
                'cache_misses': misses,
                'stale_served': stale_served
            }
        finally:
            conn.close()


class CachedAPIClient:
    """HTTP client with comprehensive caching"""
    
    def __init__(self):
        # Initialize requests-cache for HTTP caching
        cache_dir = config.cache_dir / config.cache.http_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests_cache.CachedSession(
            cache_name=str(cache_dir / 'http_cache'),
            backend=config.cache.http_cache_backend,
            expire_after=config.cache.http_cache_expire_after,
            allowable_codes=(200, 304),
            allowable_methods=['GET', 'HEAD'],
            stale_if_error=True
        )
        
        # Database cache layer
        self.db_cache = DatabaseCache()
        
        # Rate limiting
        self.rate_limiter = TokenBucket(
            capacity=config.cache.rate_limit_burst_capacity,
            refill_rate=config.cache.rate_limit_tokens_per_minute
        )
        
        # API-specific settings
        self.api_settings = {
            'odds': {
                'ttl_season': config.cache.odds_cache_ttl_season,
                'ttl_offseason': config.cache.odds_cache_ttl_offseason,
                'endpoints': ['/odds', '/markets', '/events']
            },
            'weather': {
                'ttl': config.cache.weather_cache_ttl,
                'ttl_dome': config.cache.weather_cache_ttl_dome,
                'endpoints': ['/weather', '/forecast']
            },
            'player': {
                'ttl': config.cache.player_cache_ttl,
                'endpoints': ['/players', '/stats', '/projections']
            }
        }
    
    def get(self, url: str, params: Dict[str, Any] = None, 
            force_refresh: bool = False, allow_stale: bool = True,
            api_type: str = 'generic') -> requests.Response:
        """
        Make cached HTTP request with comprehensive fallback
        """
        if config.api.cache_offline_mode:
            # Try cache only
            cached_data, is_stale = self.db_cache.get(url, params, allow_stale=True)
            if cached_data:
                # Create mock response
                response = requests.Response()
                response._content = json.dumps(cached_data).encode()
                response.status_code = 200
                return response
            else:
                raise requests.ConnectionError("Offline mode: No cached data available")
        
        # Rate limiting
        if not self.rate_limiter.consume():
            wait_time = self.rate_limiter.wait_time()
            if wait_time > 0:
                logger.warning(f"Rate limit hit, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
        
        start_time = time.time()
        
        try:
            # Check database cache first (faster than HTTP cache)
            if not force_refresh and not config.api.force_cache_refresh:
                cached_data, is_stale = self.db_cache.get(url, params, allow_stale)
                
                if cached_data and not is_stale:
                    # Fresh cache hit
                    response = requests.Response()
                    response._content = json.dumps(cached_data).encode() if isinstance(cached_data, (dict, list)) else str(cached_data).encode()
                    response.status_code = 200
                    response.headers['X-Cache'] = 'HIT-DB'
                    return response
                
                if cached_data and is_stale and allow_stale:
                    # Stale data available - refresh in background, serve stale now
                    logger.info(f"Serving stale data for {url}, scheduling background refresh")
                    response = requests.Response()
                    response._content = json.dumps(cached_data).encode() if isinstance(cached_data, (dict, list)) else str(cached_data).encode()
                    response.status_code = 200
                    response.headers['X-Cache'] = 'STALE-WHILE-REVALIDATE'

                    # Fire-and-forget background refresh so caller isn't blocked
                    threading.Thread(
                        target=self._refresh_background,
                        args=(url, params, api_type),
                        daemon=True,
                    ).start()
                    return response
            
            # Make HTTP request with requests-cache
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Determine TTL based on API type and season
            ttl = self._get_ttl_for_api(api_type, url)
            
            # Store in database cache
            try:
                data = response.json()
            except (ValueError, json.JSONDecodeError):
                data = response.text
            
            self.db_cache.set(url, data, ttl, params)
            
            # Store raw response for analysis if it's odds data
            if 'odds' in url.lower() or api_type == 'odds':
                self._store_raw_odds(url, response, params)
            
            response_time = int((time.time() - start_time) * 1000)
            self.db_cache._record_metric('refresh', 
                                       self.db_cache._generate_cache_key(url, params),
                                       url, response_time_ms=response_time)
            
            response.headers['X-Cache'] = 'MISS'
            return response
            
        except requests.RequestException as e:
            # API failed - try to serve stale data
            logger.warning(f"API request failed for {url}: {e}")
            
            cached_data, is_stale = self.db_cache.get(url, params, allow_stale=True)
            if cached_data:
                logger.info(f"Serving stale data due to API failure: {url}")
                response = requests.Response()
                response._content = json.dumps(cached_data).encode() if isinstance(cached_data, (dict, list)) else str(cached_data).encode()
                response.status_code = 200
                response.headers['X-Cache'] = 'STALE-ON-ERROR'
                return response
            
            # No cached data available
            raise
    
    def _get_ttl_for_api(self, api_type: str, url: str) -> int:
        """Get appropriate TTL based on API type and simple season/venue heuristics.

        Odds:
            - Use shorter TTL in-season, longer TTL in the off-season.
        Weather:
            - Use longer TTL for dome stadium forecasts when identifiable from the URL.
        """
        if api_type == 'odds':
            if self._is_in_season(datetime.utcnow()):
                return self.api_settings['odds']['ttl_season']
            return self.api_settings['odds']['ttl_offseason']
        elif api_type == 'weather':
            # Very lightweight heuristic: if caller encodes dome in the URL path or query,
            # prefer the dome TTL; otherwise fall back to the default weather TTL.
            url_lower = url.lower()
            if 'dome=1' in url_lower or '/dome/' in url_lower:
                return self.api_settings['weather']['ttl_dome']
            return self.api_settings['weather']['ttl']
        elif api_type == 'player':
            return self.api_settings['player']['ttl']
        else:
            return config.cache.http_cache_expire_after

    def _is_in_season(self, when: datetime) -> bool:
        """Determine whether we're roughly in the NFL regular/post-season.

        This avoids another config surface by using calendar months only:
        treat September through February as "in-season" for cache tightening.
        """
        month = when.month
        return month >= 9 or month <= 2

    def _refresh_background(self, url: str, params: Dict[str, Any], api_type: str) -> None:
        """Background refresh helper used for stale-while-revalidate.

        Errors are logged but never raised to the original caller.
        """
        try:
            logger.info(f"Background cache refresh started for {url}")
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            ttl = self._get_ttl_for_api(api_type, url)
            try:
                data = resp.json()
            except (ValueError, json.JSONDecodeError):
                data = resp.text
            self.db_cache.set(url, data, ttl, params)
            logger.info(f"Background cache refresh completed for {url}")
        except Exception as exc:
            logger.warning(f"Background cache refresh failed for {url}: {exc}")
    
    def _store_raw_odds(self, url: str, response: requests.Response, 
                       params: Dict[str, Any] = None):
        """Store raw odds response for analysis"""
        conn = sqlite3.connect(self.db_cache.db_path)
        try:
            # Extract sportsbook and market from URL/params
            sportsbook = params.get('bookmaker', 'unknown') if params else 'unknown'
            market = params.get('market', 'unknown') if params else 'unknown'
            
            conn.execute("""
                INSERT INTO odds_raw 
                (sportsbook, market, raw_response, response_size, api_request_id)
                VALUES (?, ?, ?, ?, ?)
            """, (
                sportsbook, market, response.text,
                len(response.content), f"req_{int(time.time())}"
            ))
            conn.commit()
        finally:
            conn.close()
    
    def warm_cache(self, endpoints: List[str] = None):
        """Pre-populate cache with popular endpoints. Accepts relative or full URLs."""
        if not config.cache.cache_warm_enabled:
            return
        
        endpoints = endpoints or []
        logger.info(f"Warming cache for {len(endpoints)} endpoints")
        
        for endpoint in endpoints:
            try:
                # Normalize endpoint to full URL
                url = endpoint
                if not endpoint.lower().startswith('http'):
                    base = "https://api.the-odds-api.com/v4"
                    if endpoint.startswith('/'):
                        url = base + endpoint
                    else:
                        url = base + '/' + endpoint

                params = {}
                if config.api.odds_api_key and 'apiKey=' not in url:
                    params['apiKey'] = config.api.odds_api_key

                self.get(url, params=params or None, api_type='odds')
                time.sleep(0.1)  # Avoid overwhelming APIs
            except Exception as e:
                logger.warning(f"Cache warm failed for {endpoint}: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        db_stats = self.db_cache.get_stats()
        
        # HTTP cache stats
        http_cache_info = getattr(self.session.cache, 'responses', {})
        
        return {
            'database_cache': db_stats,
            'http_cache': {
                'cached_urls': len(http_cache_info),
            },
            'rate_limiter': {
                'tokens_available': self.rate_limiter.tokens,
                'capacity': self.rate_limiter.capacity
            }
        }


# Global cached client instance
cached_client = CachedAPIClient()