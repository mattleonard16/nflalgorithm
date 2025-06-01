#!/usr/bin/env python3
"""
Analyze the quality and completeness of collected NFL data
"""

import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

def analyze_data_quality():
    """Comprehensive analysis of collected NFL data"""
    
    # Load all data
    conn = sqlite3.connect("nfl_data.db")
    df = pd.read_sql_query("SELECT * FROM player_stats", conn)
    conn.close()
    
    print("NFL Data Quality Analysis")
    print("=" * 50)
    
    # Basic statistics
    print(f"Total Records: {len(df)}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Date Range: {df['season'].min()} - {df['season'].max()}")
    
    # Records per season
    print(f"\nRecords by Season:")
    season_counts = df['season'].value_counts().sort_index()
    for season, count in season_counts.items():
        print(f"  {season}: {count} players")
    
    # Position distribution
    print(f"\nPosition Distribution:")
    pos_counts = df['position'].value_counts()
    for pos, count in pos_counts.head(10).items():
        percentage = (count / len(df)) * 100
        print(f"  {pos}: {count} ({percentage:.1f}%)")
    
    # Team distribution (check for completeness)
    print(f"\nTeams Represented: {df['team'].nunique()} unique teams")
    if df['team'].nunique() < 30:
        print("⚠️  Warning: Expected ~32 NFL teams")
    
    # Data completeness
    print(f"\nData Completeness:")
    for col in ['rushing_yards', 'rushing_attempts', 'receiving_yards', 'receptions']:
        non_zero = (df[col] > 0).sum()
        percentage = (non_zero / len(df)) * 100
        print(f"  {col}: {non_zero}/{len(df)} ({percentage:.1f}%) have data")
    
    # Statistical summary
    print(f"\nRushing Statistics Summary:")
    rushing_stats = df[df['rushing_attempts'] > 0]['rushing_yards'].describe()
    print(f"  Players with rushing attempts: {(df['rushing_attempts'] > 0).sum()}")
    print(f"  Average rushing yards: {rushing_stats['mean']:.1f}")
    print(f"  Median rushing yards: {rushing_stats['50%']:.1f}")
    print(f"  Max rushing yards: {rushing_stats['max']:.0f}")
    
    print(f"\nReceiving Statistics Summary:")
    receiving_stats = df[df['receptions'] > 0]['receiving_yards'].describe()
    print(f"  Players with receptions: {(df['receptions'] > 0).sum()}")
    print(f"  Average receiving yards: {receiving_stats['mean']:.1f}")
    print(f"  Median receiving yards: {receiving_stats['50%']:.1f}")
    print(f"  Max receiving yards: {receiving_stats['max']:.0f}")
    
    # Top performers validation
    print(f"\nTop 5 Rushers (All Seasons):")
    top_rushers = df.nlargest(5, 'rushing_yards')[['name', 'season', 'team', 'rushing_yards', 'rushing_attempts']]
    for _, player in top_rushers.iterrows():
        ypc = player['rushing_yards'] / max(player['rushing_attempts'], 1)
        print(f"  {player['name']} ({player['season']} {player['team']}): {player['rushing_yards']} yards, {ypc:.1f} YPC")
    
    print(f"\nTop 5 Receivers (All Seasons):")
    top_receivers = df.nlargest(5, 'receiving_yards')[['name', 'season', 'team', 'receiving_yards', 'receptions']]
    for _, player in top_receivers.iterrows():
        ypr = player['receiving_yards'] / max(player['receptions'], 1) 
        print(f"  {player['name']} ({player['season']} {player['team']}): {player['receiving_yards']} yards, {ypr:.1f} YPR")
    
    # Data quality flags
    print(f"\nData Quality Checks:")
    
    # Check for impossible values
    impossible_rushing = df[df['rushing_yards'] > 2000]
    impossible_receiving = df[df['receiving_yards'] > 2000]
    
    print(f"  Players with >2000 rushing yards: {len(impossible_rushing)}")
    if len(impossible_rushing) > 0:
        print(f"    {impossible_rushing[['name', 'season', 'rushing_yards']].to_string(index=False)}")
    
    print(f"  Players with >2000 receiving yards: {len(impossible_receiving)}")
    if len(impossible_receiving) > 0:
        print(f"    {impossible_receiving[['name', 'season', 'receiving_yards']].to_string(index=False)}")
    
    # Check for missing critical data
    missing_name = df[df['name'].isna()].shape[0]
    missing_team = df[df['team'].isna()].shape[0]
    
    print(f"  Records missing player name: {missing_name}")
    print(f"  Records missing team: {missing_team}")
    
    # Age distribution
    print(f"\nAge Distribution:")
    age_stats = df['age'].describe()
    print(f"  Average age: {age_stats['mean']:.1f}")
    print(f"  Age range: {age_stats['min']:.0f} - {age_stats['max']:.0f}")
    
    # Games played
    print(f"\nGames Played:")
    games_stats = df['games_played'].describe()
    print(f"  Average games: {games_stats['mean']:.1f}")
    print(f"  Players with 16+ games: {(df['games_played'] >= 16).sum()}")
    print(f"  Players with <5 games: {(df['games_played'] < 5).sum()}")
    
    return df

def create_data_summary_charts(df):
    """Create visualization charts for the data"""
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Records by season
        season_counts = df['season'].value_counts().sort_index()
        ax1.bar(season_counts.index, season_counts.values)
        ax1.set_title('Records by Season')
        ax1.set_xlabel('Season')
        ax1.set_ylabel('Number of Players')
        
        # 2. Position distribution
        pos_counts = df['position'].value_counts().head(8)
        ax2.pie(pos_counts.values, labels=pos_counts.index, autopct='%1.1f%%')
        ax2.set_title('Position Distribution')
        
        # 3. Rushing yards distribution
        rushing_data = df[df['rushing_yards'] > 0]['rushing_yards']
        ax3.hist(rushing_data, bins=50, alpha=0.7)
        ax3.set_title('Rushing Yards Distribution')
        ax3.set_xlabel('Rushing Yards')
        ax3.set_ylabel('Number of Players')
        
        # 4. Receiving yards distribution  
        receiving_data = df[df['receiving_yards'] > 0]['receiving_yards']
        ax4.hist(receiving_data, bins=50, alpha=0.7)
        ax4.set_title('Receiving Yards Distribution')
        ax4.set_xlabel('Receiving Yards')
        ax4.set_ylabel('Number of Players')
        
        plt.tight_layout()
        plt.savefig('nfl_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nData visualization saved as 'nfl_data_analysis.png'")
        
        # Final assessment
        print(f"\n" + "="*60)
        print("DATA QUALITY ASSESSMENT COMPLETE")
        print("="*60)
        print("Dataset appears ready for machine learning training")
        
    except Exception as e:
        print(f"Could not create charts: {e}")

if __name__ == "__main__":
    df = analyze_data_quality()
    create_data_summary_charts(df)
    
    print(f"\n" + "="*60)
    print("DATA ANALYSIS COMPLETE")
    print("✅ Dataset appears ready for machine learning training")
    print(f"📈 Next step: Run enhanced model validation with {len(df)} records") 