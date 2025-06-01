#!/usr/bin/env python3
"""
Collect recent seasons of NFL data (2022-2023) for improved model training
"""

from data_collection import NFLDataCollector
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_recent_seasons():
    """Collect 2022-2023 seasons for better model training"""
    collector = NFLDataCollector()
    
    seasons = [2022, 2023]
    logger.info(f"Collecting data for seasons: {seasons}")
    
    for year in seasons:
        logger.info(f"Processing {year} season...")
        
        try:
            collector.collect_season_data(year)
            
            # Check collected data
            df = collector.get_stored_stats(year)
            logger.info(f"Collected {len(df)} players for {year}")
            
            # Wait between requests
            if year == 2022:
                logger.info("Waiting 5 seconds before next season...")
                time.sleep(5)
                
        except Exception as e:
            logger.error(f"Error collecting {year}: {e}")
    
    # Show final summary
    all_data = collector.get_stored_stats()
    if not all_data.empty:
        print(f"\nFinal Dataset Summary:")
        print("=" * 30)
        
        by_season = all_data.groupby('season').size()
        print("Players by season:")
        for season, count in by_season.items():
            print(f"  {season}: {count} players")
        
        print(f"\nTotal: {len(all_data)} player-season records")
    
    return all_data

if __name__ == "__main__":
    print("NFL Recent Data Collection (2022-2023)")
    print("=" * 45)
    
    historical_data = collect_recent_seasons()
    
    print(f"\nData ready for enhanced model training!") 