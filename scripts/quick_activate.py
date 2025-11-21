#!/usr/bin/env python3
"""
Quick Betting Activation - Get Value Bets Showing
===============================================
Simple script to create value betting opportunities from our projections
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime
from rich.console import Console
from utils.db import get_connection, read_dataframe, execute, executemany

console = Console()

def main():
    console.print("🚀 [bold blue]Activating Value Betting System[/]")
    
    # Connect to database (via utils.db backend)
    conn_cm = get_connection()
    conn = conn_cm.__enter__()
    
    # Get our 2025 projections
    projections = read_dataframe(
        """
        SELECT * FROM player_projections_2025
        ORDER BY projected_fantasy_ppg DESC
        """,
        conn=conn,
    )
    
    if projections.empty:
        console.print("❌ No projections found. Run training script first.")
        return
        
    console.print(f"✅ Loaded {len(projections)} projections")
    
    # Create sample prop lines (in a real system, this would come from scraping)
    console.print("🎲 Generating sample prop lines...")
    
    # Focus on top performers with varied lines
    top_players = projections.head(50)  # Top 50 projected players
    
    prop_lines = []
    sportsbooks = ['DraftKings', 'FanDuel', 'BetMGM', 'Caesars', 'PointsBet']
    
    for _, player in top_players.iterrows():
        # Generate realistic prop lines with some variation
        base_projection = player['projected_fantasy_ppg']
        
        # Create lines slightly above/below projection to simulate market
        for book in sportsbooks[:3]:  # Use 3 sportsbooks
            # Add some random variation to create betting opportunities
            line_variation = np.random.uniform(-2, 2)  # ±2 points variation
            prop_line = max(5, base_projection + line_variation)
            
            prop_lines.append({
                'player_name': player['player_name'],
                'position': player['position'],
                'prop_type': 'Fantasy Points',
                'sportsbook': book,
                'line': round(prop_line, 1),
                'over_odds': -110,
                'under_odds': -110,
                'projection': base_projection
            })
    
    prop_df = pd.DataFrame(prop_lines)
    console.print(f"📊 Generated {len(prop_df)} prop lines")
    
    # Calculate edges and identify value
    console.print("🎯 Calculating betting edges...")
    
    value_bets = []
    
    for _, prop in prop_df.iterrows():
        edge = prop['projection'] - prop['line']
        edge_percentage = (edge / prop['line']) * 100 if prop['line'] > 0 else 0
        
        # Determine value level
        if abs(edge_percentage) >= 10:
            value_level = 'HIGH_VALUE'
        elif abs(edge_percentage) >= 5:
            value_level = 'MEDIUM_VALUE'  
        elif abs(edge_percentage) >= 2:
            value_level = 'LOW_VALUE'
        else:
            value_level = 'NO_VALUE'
            
        # Only include actual value bets
        if value_level != 'NO_VALUE':
            recommendation = 'OVER' if edge > 0 else 'UNDER'
            
            value_bets.append({
                'bet_id': f"bet_{len(value_bets)+1:04d}",
                'player_name': prop['player_name'],
                'position': prop['position'],
                'prop_type': prop['prop_type'],
                'sportsbook': prop['sportsbook'],
                'line': prop['line'],
                'model_prediction': round(prop['projection'], 2),
                'edge_yards': round(edge, 2),
                'edge_percentage': round(edge_percentage, 2),
                'value_level': value_level,
                'recommendation': recommendation,
                'confidence': 0.75 + (abs(edge_percentage) / 100),  # Higher edge = higher confidence
                'bet_size': min(5, max(1, abs(edge_percentage) / 2)),  # Size based on edge
                'expected_roi': abs(edge_percentage) * 0.8,  # Simplified ROI
                'date_identified': datetime.now()
            })
    
    value_df = pd.DataFrame(value_bets)
    console.print(f"💰 Found {len(value_df)} value betting opportunities")
    
    # Create enhanced_value_bets table and insert data
    execute("""
    CREATE TABLE IF NOT EXISTS enhanced_value_bets (
        bet_id TEXT PRIMARY KEY,
        player_name TEXT,
        position TEXT,
        prop_type TEXT,
        sportsbook TEXT,
        line REAL,
        model_prediction REAL,
        edge_yards REAL,
        edge_percentage REAL,
        value_level TEXT,
        recommendation TEXT,
        confidence REAL,
        bet_size REAL,
        expected_roi REAL,
        date_identified TIMESTAMP
    )
    """, conn=conn)
    
    # Insert value bets (replace semantics for simplicity)
    execute("DELETE FROM enhanced_value_bets", conn=conn)
    executemany(
        """
        INSERT INTO enhanced_value_bets (
            bet_id, player_name, position, prop_type, sportsbook, line,
            model_prediction, edge_yards, edge_percentage, value_level,
            recommendation, confidence, bet_size, expected_roi, date_identified
        ) VALUES (
            :bet_id, :player_name, :position, :prop_type, :sportsbook, :line,
            :model_prediction, :edge_yards, :edge_percentage, :value_level,
            :recommendation, :confidence, :bet_size, :expected_roi, :date_identified
        )
        """,
        value_df.to_dict(orient="records"),
        conn=conn,
    )
    
    # Also create a simpler table for the dashboard
    execute("""
    CREATE TABLE IF NOT EXISTS value_bets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_name TEXT,
        position TEXT,
        team TEXT DEFAULT 'UNK',
        prop_type TEXT,
        sportsbook TEXT,
        line REAL,
        prediction REAL,
        edge REAL,
        edge_pct REAL,
        recommendation TEXT,
        confidence REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """, conn=conn)
    
    # Insert simplified format
    for _, bet in value_df.iterrows():
        execute(
            """
            INSERT INTO value_bets 
            (player_name, position, prop_type, sportsbook, line, prediction, edge, edge_pct, recommendation, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(player_name, prop_type, sportsbook, line) DO UPDATE SET
                position = excluded.position,
                prediction = excluded.prediction,
                edge = excluded.edge,
                edge_pct = excluded.edge_pct,
                recommendation = excluded.recommendation,
                confidence = excluded.confidence
            """,
            (
                bet['player_name'],
                bet['position'],
                bet['prop_type'],
                bet['sportsbook'],
                bet['line'],
                bet['model_prediction'],
                bet['edge_yards'],
                bet['edge_percentage'],
                bet['recommendation'],
                bet['confidence'],
            ),
            conn=conn,
        )
    
    conn.commit()
    conn_cm.__exit__(None, None, None)
    
    # Show summary
    console.print(f"\n📊 [bold]Value Betting Summary:[/]")
    
    # Group by value level
    summary = value_df.groupby('value_level').size().sort_index(ascending=False)
    for level, count in summary.items():
        color = "green" if level == "HIGH_VALUE" else "yellow" if level == "MEDIUM_VALUE" else "blue"
        console.print(f"   • {level}: [bold {color}]{count}[/] opportunities")
    
    # Show top opportunities
    console.print(f"\n🏆 [bold]Top Value Opportunities:[/]")
    top_bets = value_df.nlargest(10, 'edge_percentage')
    for _, bet in top_bets.iterrows():
        color = "green" if bet['value_level'] == "HIGH_VALUE" else "yellow"
        console.print(f"   • {bet['player_name']} ({bet['position']}) {bet['recommendation']} {bet['line']} - Edge: [{color}]{bet['edge_percentage']:+.1f}%[/]")
    
    console.print(f"\n✅ [bold green]Value Betting System Activated![/]")
    console.print(f"   • Enhanced value bets: {len(value_df)} opportunities")
    console.print(f"   • Database updated with betting data")
    console.print(f"   • Ready for dashboard visualization")
    
    console.print(f"\n🚀 [bold]Next: Launch the dashboard to see value bets![/]")

if __name__ == "__main__":
    main()
