#!/usr/bin/env python3
"""
NFL Algorithm Cache Management CLI
=================================

Provides command-line interface for cache management with options:
- --no-cache: Bypass caching for testing
- --refresh: Force cache refresh  
- --offline: Cache-only mode
- --cache-warm: Pre-load common markets
"""

import argparse
import sys
from typing import List
import logging
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from cache_manager import cached_client, DatabaseCache
from config import config
import prop_line_scraper

console = Console()
logger = logging.getLogger(__name__)


class CacheCLI:
    """Command-line interface for cache management"""
    
    def __init__(self):
        self.db_cache = DatabaseCache()
        self.client = cached_client
    
    def stats(self):
        """Display comprehensive cache statistics"""
        console.print("📊 [bold blue]NFL Algorithm Cache Statistics[/]")
        console.print("=" * 50)
        
        stats = self.client.get_cache_stats()
        
        # Database cache stats
        db_stats = stats['database_cache']
        
        table = Table(title="Database Cache Performance")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green", justify="right")
        
        table.add_row("Total Entries", f"{db_stats['total_entries']:,}")
        table.add_row("Valid Entries", f"{db_stats['valid_entries']:,}")
        table.add_row("Expired Entries", f"{db_stats['expired_entries']:,}")
        table.add_row("Hit Rate (Last Hour)", f"{db_stats['hit_rate_percent']:.1f}%")
        table.add_row("Total Requests", f"{db_stats['total_requests_last_hour']:,}")
        table.add_row("Cache Hits", f"{db_stats['cache_hits']:,}")
        table.add_row("Cache Misses", f"{db_stats['cache_misses']:,}")
        table.add_row("Stale Served", f"{db_stats['stale_served']:,}")
        
        console.print(table)
        
        # Rate limiter stats
        rl_stats = stats['rate_limiter']
        console.print(f"\n🔄 [bold]Rate Limiter:[/]")
        console.print(f"   • Available tokens: [green]{rl_stats['tokens_available']:.1f}[/] / {rl_stats['capacity']}")
        
        # Cache efficiency calculation
        total_requests = db_stats['total_requests_last_hour']
        if total_requests > 0:
            efficiency = ((db_stats['cache_hits'] + db_stats['stale_served']) / total_requests) * 100
            console.print(f"   • Cache efficiency: [green]{efficiency:.1f}%[/]")
            
            # Estimated cost savings (assuming $0.01 per API call)
            cost_savings = (db_stats['cache_hits'] + db_stats['stale_served']) * 0.01
            console.print(f"   • Estimated cost savings: [green]${cost_savings:.2f}[/] (last hour)")
    
    def warm_cache(self, endpoints: List[str] = None):
        """Warm up cache with popular endpoints"""
        if not endpoints:
            endpoints = [
                "/sports/americanfootball_nfl/odds?markets=player_pass_tds&regions=us&oddsFormat=american&dateFormat=iso",
                "/sports/americanfootball_nfl/odds?markets=player_rush_yds&regions=us&oddsFormat=american&dateFormat=iso",
                "/sports/americanfootball_nfl/odds?markets=player_receptions&regions=us&oddsFormat=american&dateFormat=iso",
                "/sports/americanfootball_nfl/odds?markets=spreads&regions=us&oddsFormat=american&dateFormat=iso",
                "/sports/americanfootball_nfl/odds?markets=totals&regions=us&oddsFormat=american&dateFormat=iso"
            ]
        
        console.print(f"🔥 [bold]Warming cache for {len(endpoints)} popular endpoints...[/]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Warming cache...", total=len(endpoints))
            
            for endpoint in endpoints:
                try:
                    progress.update(task, description=f"Caching: {endpoint}")
                    
                    # Use the prop scraper to warm up realistic endpoints
                    if "odds" in endpoint:
                        base = "https://api.the-odds-api.com/v4"
                        url = base + endpoint
                        params = {}
                        if config.api.odds_api_key:
                            params["apiKey"] = config.api.odds_api_key
                        self.client.get(url, params=params, api_type='odds')
                    
                    progress.advance(task)
                    
                except Exception as e:
                    console.print(f"   [yellow]Warning: Failed to warm {endpoint}: {e}[/]")
                    continue
        
        console.print("✅ [green]Cache warming complete![/]")
    
    def cleanup(self, force: bool = False):
        """Clean up expired cache entries"""
        console.print("🧹 [bold]Cleaning up cache...[/]")
        
        # Get stats before cleanup
        before_stats = self.db_cache.get_stats()
        
        # Perform cleanup
        self.db_cache.cleanup_expired(force=force)
        
        # Get stats after cleanup
        after_stats = self.db_cache.get_stats()
        
        cleaned = before_stats['total_entries'] - after_stats['total_entries']
        
        if cleaned > 0:
            console.print(f"✅ [green]Cleaned {cleaned:,} expired entries[/]")
        else:
            console.print("ℹ️ [blue]No expired entries to clean[/]")
    
    def test_cache(self):
        """Test cache functionality with sample requests"""
        console.print("🧪 [bold]Testing cache functionality...[/]")
        
        test_urls = [
            "https://httpbin.org/json",  # Simple JSON endpoint
            "https://httpbin.org/delay/1",  # Endpoint with delay
        ]
        
        for url in test_urls:
            console.print(f"\n   Testing: {url}")
            
            try:
                # First request (should be cache miss)
                start_time = time.time()
                response1 = self.client.get(url, api_type='generic')
                time1 = (time.time() - start_time) * 1000
                
                console.print(f"   • First request: {time1:.0f}ms [{'HIT' if 'X-Cache' in response1.headers and 'HIT' in response1.headers['X-Cache'] else 'MISS'}]")
                
                # Second request (should be cache hit)
                start_time = time.time()
                response2 = self.client.get(url, api_type='generic')
                time2 = (time.time() - start_time) * 1000
                
                console.print(f"   • Second request: {time2:.0f}ms [{'HIT' if 'X-Cache' in response2.headers and 'HIT' in response2.headers['X-Cache'] else 'MISS'}]")
                
                if time2 < time1 / 2:  # At least 50% faster
                    console.print(f"   ✅ [green]Cache working! {((time1 - time2) / time1 * 100):.0f}% faster[/]")
                else:
                    console.print(f"   ⚠️ [yellow]Cache may not be working optimally[/]")
                    
            except Exception as e:
                console.print(f"   ❌ [red]Test failed: {e}[/]")
    
    def offline_mode_test(self):
        """Test offline mode functionality"""
        console.print("📡 [bold]Testing offline mode...[/]")
        
        # Set offline mode
        original_offline = config.api.cache_offline_mode
        config.api.cache_offline_mode = True
        
        try:
            # Try to make a request that should come from cache
            response = self.client.get("https://httpbin.org/json", api_type='generic')
            console.print("✅ [green]Offline mode working - serving from cache[/]")
            
        except Exception as e:
            console.print(f"⚠️ [yellow]Offline mode test: {e}[/]")
            console.print("   (This is expected if no cached data exists)")
            
        finally:
            # Restore original setting
            config.api.cache_offline_mode = original_offline
    
    def reset_cache(self):
        """Reset all cache data"""
        console.print("⚠️ [bold red]Resetting all cache data...[/]")
        
        # Confirm with user
        if not console.input("Are you sure? This will delete all cached data [y/N]: ").lower().startswith('y'):
            console.print("Cancelled.")
            return
        
        try:
            # Clear database cache
            conn = self.db_cache._connect()
            conn.execute("DELETE FROM api_cache")
            conn.execute("DELETE FROM odds_raw") 
            conn.execute("DELETE FROM cache_metrics")
            conn.commit()
            conn.close()
            
            # Clear HTTP cache
            self.client.session.cache.clear()
            
            console.print("✅ [green]All cache data reset[/]")
            
        except Exception as e:
            console.print(f"❌ [red]Reset failed: {e}[/]")


def main():
    parser = argparse.ArgumentParser(
        description="NFL Algorithm Cache Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cache_cli.py stats              # Show cache statistics
  python cache_cli.py warm               # Warm up cache
  python cache_cli.py cleanup            # Clean expired entries
  python cache_cli.py test               # Test cache functionality
  python cache_cli.py --no-cache stats   # Bypass cache for command
  python cache_cli.py --offline stats    # Use cache-only mode
  python cache_cli.py --refresh warm     # Force refresh while warming
        """
    )
    
    # Global options
    parser.add_argument('--no-cache', action='store_true',
                       help='Bypass caching for this command')
    parser.add_argument('--refresh', action='store_true', 
                       help='Force cache refresh')
    parser.add_argument('--offline', action='store_true',
                       help='Cache-only mode (no API calls)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    # Commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show cache statistics')
    
    # Warm command
    warm_parser = subparsers.add_parser('warm', help='Warm up cache')
    warm_parser.add_argument('--endpoints', nargs='*',
                            help='Specific endpoints to warm')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean expired cache entries')
    cleanup_parser.add_argument('--force', action='store_true',
                               help='Force cleanup of all expired entries')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test cache functionality')
    
    # Reset command  
    reset_parser = subparsers.add_parser('reset', help='Reset all cache data')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    # Apply global options
    if args.no_cache:
        config.api.enable_caching = False
        console.print("🚫 [yellow]Caching disabled for this session[/]")
    
    if args.refresh:
        config.api.force_cache_refresh = True
        console.print("🔄 [yellow]Force refresh enabled[/]")
    
    if args.offline:
        config.api.cache_offline_mode = True  
        console.print("📡 [yellow]Offline mode enabled[/]")
    
    # Execute command
    cli = CacheCLI()
    
    if args.command == 'stats':
        cli.stats()
    elif args.command == 'warm':
        cli.warm_cache(args.endpoints)
    elif args.command == 'cleanup':
        cli.cleanup(args.force)
    elif args.command == 'test':
        cli.test_cache()
        cli.offline_mode_test()
    elif args.command == 'reset':
        cli.reset_cache()
    else:
        parser.print_help()


if __name__ == "__main__":
    import time
    main()