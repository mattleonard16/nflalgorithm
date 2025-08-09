#!/usr/bin/env python3
"""
Quick NFL Data Population - Get System Working Fast
==================================================
Simple script to populate database with real NFL data from nfl_data_py
"""

import sqlite3
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

def main():
    console.print("🚀 [bold blue]Quick NFL Data Population[/]")
    
    # Install and import nfl_data_py
    try:
        import subprocess
        result = subprocess.run(["uv", "pip", "install", "nfl_data_py"], capture_output=True)
        import nfl_data_py as nfl
        console.print("✅ nfl_data_py ready")
    except Exception as e:
        console.print(f"❌ Failed to install nfl_data_py: {e}")
        return
    
    # Connect to database
    conn = sqlite3.connect("nfl_data.db")
    
    # Create simple tables
    conn.execute("""
    CREATE TABLE IF NOT EXISTS player_stats (
        player_id TEXT,
        player_name TEXT,
        position TEXT,
        team TEXT,
        season INTEGER,
        week INTEGER,
        fantasy_points REAL,
        rushing_yards INTEGER DEFAULT 0,
        receiving_yards INTEGER DEFAULT 0,
        passing_yards INTEGER DEFAULT 0,
        rushing_tds INTEGER DEFAULT 0,
        receiving_tds INTEGER DEFAULT 0,
        passing_tds INTEGER DEFAULT 0,
        PRIMARY KEY (player_id, season, week)
    )
    """)
    
    conn.execute("""
    CREATE TABLE IF NOT EXISTS enhanced_features (
        player_id TEXT,
        player_name TEXT,
        position TEXT,
        season INTEGER,
        fantasy_points_per_game REAL,
        value_score REAL DEFAULT 0.5,
        consistency_score REAL DEFAULT 0.5,
        PRIMARY KEY (player_id, season)
    )
    """)
    
    console.print("📊 Fetching NFL data for 2024...")
    
    # Fetch 2024 data
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn()) as progress:
        task = progress.add_task("Loading 2024 season data...", total=None)
        
        # Get weekly data for 2024
        df = nfl.import_weekly_data([2024])
        progress.update(task, description=f"Processing {len(df)} records...")
        
        # Clean and prepare data
        df = df.fillna(0)
        df = df[df['season_type'] == 'REG']  # Regular season only
        
        # Insert player stats
        inserted = 0
        for _, row in df.iterrows():
            try:
                conn.execute("""
                INSERT OR REPLACE INTO player_stats 
                (player_id, player_name, position, team, season, week, fantasy_points,
                 rushing_yards, receiving_yards, passing_yards, rushing_tds, receiving_tds, passing_tds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row.get('player_id', 'unknown'),
                    row.get('player_name', 'Unknown'),
                    row.get('position', 'UNK'),
                    row.get('recent_team', 'UNK'),
                    row.get('season', 2024),
                    row.get('week', 1),
                    row.get('fantasy_points_ppr', 0),
                    int(row.get('rushing_yards', 0) or 0),
                    int(row.get('receiving_yards', 0) or 0),
                    int(row.get('passing_yards', 0) or 0),
                    int(row.get('rushing_tds', 0) or 0),
                    int(row.get('receiving_tds', 0) or 0),
                    int(row.get('passing_tds', 0) or 0)
                ))
                inserted += 1
            except Exception as e:
                continue
                
        progress.update(task, description=f"Inserted {inserted} records")
    
    # Create enhanced features
    console.print("⚙️ Creating enhanced features...")
    
    # Calculate player season aggregates
    season_stats = conn.execute("""
    SELECT player_id, player_name, position,
           AVG(fantasy_points) as avg_fp,
           COUNT(*) as games,
           SUM(fantasy_points) as total_fp
    FROM player_stats 
    WHERE season = 2024
    GROUP BY player_id, player_name, position
    HAVING games >= 4
    """).fetchall()
    
    # Insert enhanced features
    for stats in season_stats:
        player_id, name, pos, avg_fp, games, total_fp = stats
        value_score = min(1.0, max(0.0, avg_fp / 15.0))  # Simple value score
        consistency = max(0.1, 1.0 - (total_fp / games / max(avg_fp, 1)))  # Simple consistency
        
        conn.execute("""
        INSERT OR REPLACE INTO enhanced_features 
        (player_id, player_name, position, season, fantasy_points_per_game, value_score, consistency_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (player_id, name, pos, 2024, round(avg_fp, 2), round(value_score, 3), round(consistency, 3)))
    
    conn.commit()
    
    # Summary
    player_count = conn.execute("SELECT COUNT(*) FROM player_stats").fetchone()[0]
    enhanced_count = conn.execute("SELECT COUNT(*) FROM enhanced_features").fetchone()[0]
    
    console.print(f"\n✅ [bold green]Database populated successfully![/]")
    console.print(f"   • Player-week records: {player_count:,}")
    console.print(f"   • Enhanced features: {enhanced_count:,}")
    
    # Show top performers
    console.print(f"\n🏆 [bold]Top Fantasy Performers (2024):[/]")
    top_players = conn.execute("""
    SELECT player_name, position, 
           ROUND(fantasy_points_per_game, 1) as fpg,
           ROUND(value_score, 3) as value
    FROM enhanced_features 
    WHERE season = 2024 
    ORDER BY fantasy_points_per_game DESC 
    LIMIT 10
    """).fetchall()
    
    for name, pos, fpg, value in top_players:
        console.print(f"   • {name} ({pos}): {fpg} FPG, {value} value")
    
    conn.close()
    console.print(f"\n🚀 Ready for model training and dashboard!")

if __name__ == "__main__":
    main()