#!/usr/bin/env python3
"""Test script to verify Supabase connection is working."""

import os
import sys
from utils.db import get_connection, read_dataframe

def test_connection():
    """Test basic connection to Supabase."""
    print("Testing Supabase connection...")
    print(f"DB_BACKEND: {os.getenv('DB_BACKEND', 'not set')}")
    print(f"SUPABASE_DB_URL: {'set' if os.getenv('SUPABASE_DB_URL') else 'not set'}")
    print()
    
    try:
        with get_connection() as conn:
            print("✅ Connection established successfully!")
            
            # Test a simple query
            result = read_dataframe("SELECT version() as version")
            print(f"✅ Database version: {result['version'].iloc[0]}")
            
            # Test current database
            result = read_dataframe("SELECT current_database() as db_name")
            print(f"✅ Connected to database: {result['db_name'].iloc[0]}")
            
            # List available schemas
            result = read_dataframe("""
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
                ORDER BY schema_name
            """)
            if not result.empty:
                print(f"✅ Available schemas: {', '.join(result['schema_name'].tolist())}")
            else:
                print("ℹ️  No custom schemas found (this is normal for a fresh database)")
            
            print("\n✅ All connection tests passed!")
            return True
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure DB_BACKEND=supabase is set")
        print("2. Make sure SUPABASE_DB_URL is set with your actual password")
        print("3. Check that psycopg2-binary is installed: pip install psycopg2-binary")
        print("4. Verify your Supabase password in the dashboard")
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)

