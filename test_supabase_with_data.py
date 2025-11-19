#!/usr/bin/env python3
"""Test Supabase connection and create test data."""

import os
import sys
from datetime import datetime
from utils.db import get_connection, read_dataframe, execute, executemany

def setup_test_environment():
    """Set environment variables for Supabase connection."""
    os.environ['DB_BACKEND'] = 'supabase'
    os.environ['SUPABASE_DB_URL'] = 'postgresql://postgres:nflalgo20032@db.rweztompmzjxypismzzf.supabase.co:5432/postgres'
    print("✅ Environment variables set for Supabase connection")

def test_connection():
    """Test basic connection to Supabase."""
    print("\n" + "="*60)
    print("Testing Supabase Connection")
    print("="*60)
    
    try:
        with get_connection() as conn:
            print("✅ Connection established successfully!")
            
            # Test database version
            result = read_dataframe("SELECT version() as version")
            version = result['version'].iloc[0]
            print(f"✅ PostgreSQL version: {version.split(',')[0]}")
            
            # Test current database
            result = read_dataframe("SELECT current_database() as db_name, current_user as user_name")
            print(f"✅ Connected to database: {result['db_name'].iloc[0]}")
            print(f"✅ Connected as user: {result['user_name'].iloc[0]}")
            
            return True
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_table():
    """Create a test table for NFL algorithm data."""
    print("\n" + "="*60)
    print("Creating Test Table")
    print("="*60)
    
    try:
        # Drop table if exists (for clean testing)
        execute("""
            DROP TABLE IF EXISTS test_nfl_data
        """)
        print("✅ Cleaned up any existing test table")
        
        # Create test table with NFL-relevant schema
        execute("""
            CREATE TABLE test_nfl_data (
                id SERIAL PRIMARY KEY,
                player_id VARCHAR(50) NOT NULL,
                player_name VARCHAR(100) NOT NULL,
                team VARCHAR(10),
                position VARCHAR(10),
                season INTEGER,
                week INTEGER,
                stat_value NUMERIC(10, 2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(player_id, season, week)
            )
        """)
        print("✅ Created test_nfl_data table")
        
        # Create index
        execute("""
            CREATE INDEX idx_test_nfl_player_season_week 
            ON test_nfl_data(player_id, season, week)
        """)
        print("✅ Created index on test_nfl_data")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create test table: {e}")
        import traceback
        traceback.print_exc()
        return False

def insert_test_data():
    """Insert sample NFL test data."""
    print("\n" + "="*60)
    print("Inserting Test Data")
    print("="*60)
    
    test_players = [
        ("QB001", "Patrick Mahomes", "KC", "QB", 2024, 1, 312.5),
        ("RB001", "Christian McCaffrey", "SF", "RB", 2024, 1, 125.0),
        ("WR001", "Tyreek Hill", "MIA", "WR", 2024, 1, 98.5),
        ("QB001", "Patrick Mahomes", "KC", "QB", 2024, 2, 298.0),
        ("RB001", "Christian McCaffrey", "SF", "RB", 2024, 2, 118.0),
        ("WR001", "Tyreek Hill", "MIA", "WR", 2024, 2, 105.5),
    ]
    
    try:
        # Use executemany for batch insert
        executemany("""
            INSERT INTO test_nfl_data 
            (player_id, player_name, team, position, season, week, stat_value)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (player_id, season, week) 
            DO UPDATE SET 
                player_name = EXCLUDED.player_name,
                team = EXCLUDED.team,
                position = EXCLUDED.position,
                stat_value = EXCLUDED.stat_value
        """, test_players)
        
        print(f"✅ Inserted {len(test_players)} test records")
        return True
        
    except Exception as e:
        print(f"❌ Failed to insert test data: {e}")
        import traceback
        traceback.print_exc()
        return False

def query_test_data():
    """Query and display test data."""
    print("\n" + "="*60)
    print("Querying Test Data")
    print("="*60)
    
    try:
        # Get all test data
        df = read_dataframe("""
            SELECT 
                player_id,
                player_name,
                team,
                position,
                season,
                week,
                stat_value,
                created_at
            FROM test_nfl_data
            ORDER BY player_id, season, week
        """)
        
        print(f"\n✅ Retrieved {len(df)} records:")
        print("\n" + df.to_string(index=False))
        
        # Aggregate query
        df_agg = read_dataframe("""
            SELECT 
                player_name,
                team,
                position,
                COUNT(*) as games,
                AVG(stat_value) as avg_stat,
                SUM(stat_value) as total_stat
            FROM test_nfl_data
            GROUP BY player_name, team, position
            ORDER BY avg_stat DESC
        """)
        
        print("\n✅ Aggregated statistics:")
        print("\n" + df_agg.to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to query test data: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_transactions():
    """Test transaction support."""
    print("\n" + "="*60)
    print("Testing Transactions")
    print("="*60)
    
    try:
        with get_connection() as conn:
            # Check if it's PostgreSQL (has cursor method) or SQLite
            import sqlite3
            if isinstance(conn, sqlite3.Connection):
                # SQLite
                conn.execute("""
                    INSERT INTO test_nfl_data 
                    (player_id, player_name, team, position, season, week, stat_value)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, ("TEST001", "Test Player", "TEST", "QB", 2024, 99, 100.0))
                conn.rollback()
                print("✅ Transaction test completed (SQLite)")
            else:
                # PostgreSQL - don't use context manager for cursor to keep it open
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        INSERT INTO test_nfl_data 
                        (player_id, player_name, team, position, season, week, stat_value)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, ("TEST001", "Test Player", "TEST", "QB", 2024, 99, 100.0))
                    
                    # Verify it's in the transaction
                    cursor.execute("SELECT COUNT(*) FROM test_nfl_data WHERE player_id = %s", ("TEST001",))
                    count = cursor.fetchone()[0]
                    print(f"✅ Transaction insert successful (count: {count})")
                    
                    # Rollback to test rollback
                    conn.rollback()
                    print("✅ Rollback successful")
                    
                    # Verify it's gone (need new cursor after rollback)
                    cursor2 = conn.cursor()
                    cursor2.execute("SELECT COUNT(*) FROM test_nfl_data WHERE player_id = %s", ("TEST001",))
                    count = cursor2.fetchone()[0]
                    cursor2.close()
                    print(f"✅ Verified rollback (count: {count})")
                finally:
                    cursor.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Transaction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup():
    """Clean up test data (optional)."""
    print("\n" + "="*60)
    print("Cleanup (Optional)")
    print("="*60)
    
    try:
        response = input("Do you want to drop the test table? (y/N): ").strip().lower()
        if response == 'y':
            try:
                execute("DROP TABLE IF EXISTS test_nfl_data")
                print("✅ Test table dropped")
            except Exception as e:
                print(f"⚠️  Failed to drop table: {e}")
        else:
            print("ℹ️  Test table preserved for inspection")
    except EOFError:
        # Non-interactive environment, skip cleanup prompt
        print("ℹ️  Test table preserved (non-interactive mode)")

def main():
    """Run all tests."""
    print("="*60)
    print("Supabase Connection & Data Test")
    print("="*60)
    
    # Setup environment
    setup_test_environment()
    
    # Run tests
    tests = [
        ("Connection Test", test_connection),
        ("Create Test Table", create_test_table),
        ("Insert Test Data", insert_test_data),
        ("Query Test Data", query_test_data),
        ("Transaction Test", test_transactions),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} raised exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    # Optional cleanup
    if all_passed:
        cleanup()
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

