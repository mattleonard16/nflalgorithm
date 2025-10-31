"""
Simplified HTTP Caching for NFL Algorithm
=======================================

Implements essential caching with requests-cache only.
Removes database persistence layer for simplicity and maintainability.
"""

import logging
import time
from typing import Dict, Any, Optional
import requests
import requests_cache
from datetime import datetime, timedelta

from config import config

logger = logging.getLogger(__name__)

class SimpleRateLimiter:
    """Simple token bucket rate limiter."""
    
    def __init__(self, capacity: int = 60, refill_rate: int = 60):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per minute
        self.tokens = capacity
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful."""
        now = time.time()
        
        # Add tokens based on time passed
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * (self.refill_rate / 60.0)
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


class SimpleCachedClient:
    """Simplified HTTP client with essential caching only."""
    
    def __init__(self):
        # Initialize requests-cache for HTTP caching only
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
        
        # Simple rate limiting
        self.rate_limiter = SimpleRateLimiter(
            capacity=config.cache.rate_limit_burst_capacity,
            refill_rate=config.cache.rate_limit_tokens_per_minute
        )
        
        logger.info("Simplified cache client initialized")
    
    def get(self, url: str, params: Dict[str, Any] = None, 
            force_refresh: bool = False, api_type: str = 'generic') -> requests.Response:
        """
        Make cached HTTP request with simplified approach.
        """
        if config.api.cache_offline_mode:
            # Try cache only
            cached_response = self._get_from_cache(url, params)
            if cached_response:
                cached_response.headers['X-Cache'] = 'HIT-OFFLINE'
                return cached_response
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
            # Check if we have a fresh cached response
            if not force_refresh and not config.api.force_cache_refresh:
                cached_response = self._get_from_cache(url, params)
                if cached_response and not self._is_cache_expired(cached_response):
                    cached_response.headers['X-Cache'] = 'HIT'
                    return cached_response
            
            # Make actual API request
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Log performance
            response_time = (time.time() - start_time) * 1000
            logger.info(f"API call completed in {response_time:.0f}ms: {url}")
            
            response.headers['X-Cache'] = 'MISS'
            return response
            
        except requests.RequestException as e:
            # Try to serve stale cache if available
            logger.warning(f"API request failed for {url}: {e}")
            cached_response = self._get_from_cache(url, params)
            if cached_response:
                logger.info(f"Serving stale cache for {url}")
                cached_response.headers['X-Cache'] = 'STALE-ON-ERROR'
                return cached_response
            raise
    
    def _get_from_cache(self, url: str, params: Dict[str, Any] = None) -> Optional[requests.Response]:
        """Get response from cache if available."""
        try:
            cache_key = requests_cache.utils.create_url(url, params)
            cached_response = self.session.cache.responses.get(cache_key)
            if cached_response:
                response = requests.Response()
                response._content = cached_response.content
                response.status_code = cached_response.status_code
                response.headers = dict(cached_response.headers)
                response.url = cached_response.url
                return response
        except Exception:
            pass
        return None
    
    def _is_cache_expired(self, response: requests.Response) -> bool:
        """Check if cached response is expired."""
        try:
            if 'x-cache-created' in response.headers:
                created_time = datetime.fromisoformat(response.headers['x-cache-created'])
                ttl = self._get_ttl_for_api(api_type='generic')
                expired = (datetime.now() - created_time) > ttl
                return expired
        except Exception:
            pass
        return False
    
    def _get_ttl_for_api(self, api_type: str) -> timedelta:
        """Get appropriate TTL based on API type."""
        if api_type == 'odds':
            return timedelta(minutes=config.cache.odds_cache_ttl_season)
        elif api_type == 'weather':
            return timedelta(minutes=config.cache.weather_cache_ttl)
        elif api_type == 'player':
            return timedelta(minutes=config.cache.player_cache_ttl)
        else:
            return timedelta(seconds=config.cache.http_cache_expire_after)
    
    def warm_cache(self, endpoints: list = None):
        """Pre-populate cache with popular endpoints."""
        if not config.cache.cache_warm_enabled or not endpoints:
            return
        
        logger.info(f"Warming cache for {len(endpoints)} endpoints")
        
        for endpoint in endpoints:
            try:
                if not endpoint.startswith('http'):
                    endpoint = f"https://api.the-odds-api.com/v4/{endpoint}"
                
                self.get(endpoint, api_type='odds')
                time.sleep(0.1)  # Avoid overwhelming APIs
            except Exception as e:
                logger.warning(f"Cache warm failed for {endpoint}: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get basic cache statistics."""
        cache_info = getattr(self.session.cache.responses, 'keys', [])
        return {
            'cached_urls': len(cache_info),
            'cache_backend': config.cache.http_cache_backend,
            'rate_limiter_tokens': self.rate_limiter.tokens,
            'cache_dir': str(config.cache_dir / config.cache.http_cache_dir)
        }


# Global simplified cached client instance
simple_cached_client = SimpleCachedClient()
