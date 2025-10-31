#!/usr/bin/env python3
"""
Complete Week 9 Training and Prediction Script
Implements WEEK9_TRAINING_PLAN.md - Collects data, trains models, predicts Week 9
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline import update_week
from models.position_specific.weekly import train_weekly_models, predict_week
from materialized_value_view import materialize_week
from scripts.prop_line_scraper import NFLPropScraper
import pandas as pd

def check_existing_data():
    """Check what weeks already exist in database."""
    import sqlite3
    conn = sqlite3.connect('nfl_data.db')
    cursor = conn.cursor()
    
    # Check 2025 weeks
    cursor.execute("SELECT DISTINCT week FROM player_stats_enhanced WHERE season=2025 ORDER BY week")
    weeks_2025 = [row[0] for row in cursor.fetchall()]
    
    # Check 2024 weeks
    cursor.execute("SELECT DISTINCT week FROM player_stats_enhanced WHERE season=2024 ORDER BY week")
    weeks_2024 = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return weeks_2025, weeks_2024

def main():
    print("=" * 60)
    print("WEEK 9 TRAINING PLAN - COMPLETE IMPLEMENTATION")
    print("=" * 60)
    
    # Check existing data
    print("\n📊 Checking existing data...")
    weeks_2025, weeks_2024 = check_existing_data()
    print(f"  Existing 2025 weeks: {sorted(weeks_2025)}")
    print(f"  Existing 2024 weeks: {sorted(weeks_2024)}")
    
    # PHASE 1: Collect 2025 Weeks 1-8 (Current Season)
    print("\n" + "=" * 60)
    print("PHASE 1: Collecting 2025 weeks 1-8 (current season)...")
    print("=" * 60)
    
    missing_2025 = [w for w in range(1, 9) if w not in weeks_2025]
    if missing_2025:
        print(f"  Missing weeks: {missing_2025}")
        for week in missing_2025:
            print(f"  📥 Updating 2025 Week {week}...")
            try:
                update_week(2025, week)
                print(f"  ✅ Week {week} complete")
            except Exception as e:
                print(f"  ⚠️  Week {week} failed: {e}")
    else:
        print("  ✅ All 2025 weeks 1-8 already collected")
    
    # PHASE 2: Collect 2024 Full Season (Historical Data)
    print("\n" + "=" * 60)
    print("PHASE 2: Collecting 2024 full season (historical data)...")
    print("=" * 60)
    
    missing_2024 = [w for w in range(1, 19) if w not in weeks_2024]
    if missing_2024:
        print(f"  Missing weeks: {missing_2024}")
        for week in missing_2024:
            print(f"  📥 Updating 2024 Week {week}...")
            try:
                update_week(2024, week)
                print(f"  ✅ Week {week} complete")
            except Exception as e:
                print(f"  ⚠️  Week {week} failed: {e}")
    else:
        print("  ✅ All 2024 weeks 1-18 already collected")
    
    # PHASE 3: Train Models
    print("\n" + "=" * 60)
    print("PHASE 3: Training models...")
    print("=" * 60)
    
    current_weeks = [(2025, w) for w in range(1, 9)]  # 2025 weeks 1-8
    previous_year_weeks = [(2024, w) for w in range(1, 19)]  # 2024 full season
    training_weeks = current_weeks + previous_year_weeks
    
    print(f"  Training on {len(training_weeks)} weeks:")
    print(f"    - 2025 weeks 1-8: {len(current_weeks)} weeks")
    print(f"    - 2024 full season: {len(previous_year_weeks)} weeks")
    
    try:
        trained_models = train_weekly_models(training_weeks)
        print(f"  ✅ Models trained: {list(trained_models.keys())}")
        for market, path in trained_models.items():
            print(f"    - {market}: {path}")
    except Exception as e:
        print(f"  ❌ Model training failed: {e}")
        raise
    
    # PHASE 4: Generate Week 9 Predictions (2025)
    print("\n" + "=" * 60)
    print("PHASE 4: Generating Week 9 predictions for 2025...")
    print("=" * 60)
    
    print("  📥 Ensuring Week 9 data exists...")
    try:
        update_week(2025, 9)
        print("  ✅ Week 9 data ready")
    except Exception as e:
        print(f"  ⚠️  Week 9 data update failed: {e}")
    
    print("  🔮 Generating predictions...")
    try:
        projections = predict_week(2025, 9)
        print(f"  ✅ Generated {len(projections)} projections")
        if not projections.empty:
            markets = projections['market'].unique().tolist()
            print(f"    Markets: {markets}")
    except Exception as e:
        print(f"  ❌ Prediction failed: {e}")
        raise
    
    # PHASE 5: Fetch Week 9 Odds (Optional but Recommended)
    print("\n" + "=" * 60)
    print("PHASE 5: Fetching Week 9 odds for 2025...")
    print("=" * 60)
    
    try:
        scraper = NFLPropScraper()
        scraper.get_upcoming_week_props(9, 2025)
        print("  ✅ Week 9 odds fetched")
    except Exception as e:
        print(f"  ⚠️  Odds fetch failed (may not be available): {e}")
    
    # PHASE 6: Calculate Value Bets
    print("\n" + "=" * 60)
    print("PHASE 6: Calculating Week 9 value bets for 2025...")
    print("=" * 60)
    
    try:
        value_bets = materialize_week(2025, 9, min_edge=0.05)
        bet_opportunities = value_bets[value_bets['recommendation'] == 'BET']
        
        print(f"\n  ✅ COMPLETE!")
        print(f"    Total opportunities analyzed: {len(value_bets)}")
        print(f"    Value bets found: {len(bet_opportunities)}")
        
        if not bet_opportunities.empty:
            print(f"    Average edge: {bet_opportunities['edge_percentage'].mean()*100:.2f}%")
            print(f"    Average ROI: {bet_opportunities['expected_roi'].mean()*100:.2f}%")
            
            # Show top 10
            top = bet_opportunities.nlargest(10, 'edge_percentage')
            print(f"\n  📊 Top 10 Value Opportunities:")
            for idx, (_, row) in enumerate(top.iterrows(), 1):
                print(f"    {idx}. {row['player_name']} ({row['position']}): {row['market']}")
                print(f"       Line: {row['line']:.1f} | Model: {row['mu']:.1f} | "
                      f"Edge: {row['edge_percentage']*100:.1f}% | ROI: {row['expected_roi']*100:.1f}%")
        else:
            print("    ⚠️  No value bets found with 5%+ edge")
    except Exception as e:
        print(f"  ❌ Value calculation failed: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("🎉 ALL PHASES COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run: make dashboard")
    print("  2. Select Season=2025, Week=9")
    print("  3. Click 'Refresh Data'")
    print("  4. View value betting opportunities")
    print("=" * 60)

if __name__ == "__main__":
    main()

