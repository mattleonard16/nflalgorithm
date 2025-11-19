#!/usr/bin/env python3
"""Verify Supabase migration - check tables and data counts."""

from utils.db import read_dataframe
from config import config

def verify_migration():
    """Verify the Supabase migration was successful."""
    print("="*60)
    print("Supabase Migration Verification")
    print("="*60)
    print(f"Backend: {config.database.backend}")
    print()
    
    # Get all tables
    tables_df = read_dataframe("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """)
    
    nfl_tables = [
        'player_stats', 'player_stats_enhanced', 'players',
        'value_bets', 'enhanced_value_bets',
        'prop_lines', 'weekly_prop_lines',
        'odds_data', 'weekly_odds',
        'games', 'weather_data', 'injury_data',
        'api_cache', 'cache_metrics', 'freshness', 'feed_freshness'
    ]
    
    print(f"Total tables in database: {len(tables_df)}")
    print()
    print("NFL Algorithm Tables & Row Counts:")
    print("-" * 60)
    
    total_rows = 0
    found_tables = 0
    
    for table in nfl_tables:
        if table in tables_df['table_name'].values:
            try:
                count_df = read_dataframe(f"SELECT COUNT(*) as count FROM {table}")
                count = count_df.iloc[0]['count']
                print(f"  ✅ {table:30s} {count:>10,} rows")
                total_rows += count
                found_tables += 1
            except Exception as e:
                print(f"  ❌ {table:30s} ERROR: {e}")
        else:
            print(f"  ⚠️  {table:30s} NOT FOUND")
    
    print("-" * 60)
    print(f"Found {found_tables}/{len(nfl_tables)} NFL tables")
    print(f"Total rows across NFL tables: {total_rows:,}")
    print()
    
    # Check for test tables
    test_tables = tables_df[tables_df['table_name'].str.startswith('test_')]
    if len(test_tables) > 0:
        print("Test tables (can be removed):")
        for table in test_tables['table_name']:
            print(f"  - {table}")
        print()
    
    # Verify key functionality
    print("Functionality Tests:")
    print("-" * 60)
    
    try:
        # Test read
        df = read_dataframe("SELECT COUNT(*) as count FROM player_stats")
        print("  ✅ read_dataframe: Working")
    except Exception as e:
        print(f"  ❌ read_dataframe: {e}")
    
    try:
        from utils.db import get_table_columns
        cols = get_table_columns('player_stats')
        print(f"  ✅ get_table_columns: Working ({len(cols)} columns)")
    except Exception as e:
        print(f"  ❌ get_table_columns: {e}")
    
    try:
        from utils.db import get_connection
        with get_connection() as conn:
            print("  ✅ get_connection: Working")
    except Exception as e:
        print(f"  ❌ get_connection: {e}")
    
    print()
    print("="*60)
    if found_tables >= len(nfl_tables) * 0.8:  # 80% threshold
        print("✅ Migration appears successful!")
    else:
        print("⚠️  Some tables may be missing - review above")
    print("="*60)

if __name__ == "__main__":
    verify_migration()

