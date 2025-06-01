#!/usr/bin/env python3
"""
Collect multiple seasons of NFL data for better model training
"""

from data_collection import NFLDataCollector
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_multiple_seasons(start_year=2020, end_year=2023):
    """Collect data for multiple NFL seasons"""
    collector = NFLDataCollector()
    
    seasons_to_collect = list(range(start_year, end_year + 1))
    logger.info(f"Collecting data for seasons: {seasons_to_collect}")
    
    total_collected = 0
    
    for year in seasons_to_collect:
        logger.info(f"Processing {year} season...")
        
        try:
            collector.collect_season_data(year)
            
            # Check how much data we collected for this season
            df = collector.get_stored_stats(year)
            season_count = len(df)
            total_collected += season_count
            
            logger.info(f"Successfully collected {season_count} players for {year}")
            
            # Be respectful - wait between seasons
            if year < end_year:
                logger.info("Waiting 5 seconds before next season...")
                time.sleep(5)
                
        except Exception as e:
            logger.error(f"Error collecting {year} season: {e}")
            continue
    
    logger.info(f"Collection complete! Total players: {total_collected}")
    
    # Show summary
    all_data = collector.get_stored_stats()
    if not all_data.empty:
        print(f"\nHistorical Data Summary:")
        print("=" * 40)
        
        season_summary = all_data.groupby('season').agg({
            'player_id': 'count',
            'rushing_yards': 'mean',
            'receiving_yards': 'mean'
        }).round(1)
        
        season_summary.columns = ['Players', 'Avg Rush Yds', 'Avg Rec Yds']
        print(season_summary)
        
        print(f"\nTotal dataset: {len(all_data)} player-season records")
        print(f"Seasons covered: {sorted(all_data['season'].unique())}")
        
        # Top rushers across all seasons
        top_rushers = all_data.nlargest(10, 'rushing_yards')[['name', 'season', 'team', 'rushing_yards']]
        print(f"\nTop 10 Rushing Seasons:")
        print(top_rushers.to_string(index=False))
        
    return all_data

if __name__ == "__main__":
    print("NFL Historical Data Collection")
    print("=" * 50)
    
    # Collect 2020-2023 seasons
    historical_data = collect_multiple_seasons(2020, 2023)
    
    print(f"\nData collection complete!")
    print(f"Ready for enhanced model training with {len(historical_data)} records") 