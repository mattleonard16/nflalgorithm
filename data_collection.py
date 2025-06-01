import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
from datetime import datetime
import time
import logging
from typing import List, Dict, Optional
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NFLDataCollector:
    def __init__(self):
        self.db_path = "nfl_data.db"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        self.setup_database()
    
    def setup_database(self):
        """Create database schema for NFL data"""
        conn = sqlite3.connect(self.db_path)
        
        # Player stats table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS player_stats (
                player_id TEXT,
                season INTEGER,
                name TEXT,
                team TEXT,
                position TEXT,
                age INTEGER,
                games_played INTEGER,
                rushing_yards INTEGER,
                rushing_attempts INTEGER,
                receiving_yards INTEGER,
                receptions INTEGER,
                targets INTEGER,
                snap_count INTEGER,
                injury_games_missed INTEGER,
                PRIMARY KEY (player_id, season)
            )
        ''')
        
        # Team context table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS team_context (
                team TEXT,
                season INTEGER,
                offensive_rank INTEGER,
                qb_rating REAL,
                oline_rank INTEGER,
                pace_rank INTEGER,
                red_zone_efficiency REAL,
                PRIMARY KEY (team, season)
            )
        ''')
        
        # Betting lines table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS betting_lines (
                player_id TEXT,
                season INTEGER,
                sportsbook TEXT,
                rushing_over_under REAL,
                receiving_over_under REAL,
                date_scraped TEXT,
                PRIMARY KEY (player_id, season, sportsbook)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database schema created successfully")
    
    def scrape_rushing_stats(self, year: int = 2024) -> List[Dict]:
        """Scrape rushing stats from Pro-Football-Reference"""
        logger.info(f"Scraping rushing stats for {year}...")
        
        url = f"https://www.pro-football-reference.com/years/{year}/rushing.htm"
        
        try:
            # Add random delay to be respectful
            time.sleep(random.uniform(1, 3))
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the rushing stats table
            table = soup.find('table', {'id': 'rushing'})
            if not table:
                logger.error("Could not find rushing stats table")
                return []
            
            players = []
            
            # Skip header rows and parse data
            rows = table.find_all('tr')
            logger.info(f"Found {len(rows)} total rows in rushing table")
            
            for idx, row in enumerate(rows):
                try:
                    # Skip header rows and rows without data
                    if row.find('th', {'scope': 'col'}):
                        continue
                        
                    # Extract player data
                    player_cell = row.find('td', {'data-stat': 'name_display'})
                    if not player_cell:
                        continue
                    
                    # Get player link for ID
                    player_link = player_cell.find('a')
                    if not player_link:
                        continue
                    
                    player_id = player_link.get('href', '').split('/')[-1].replace('.htm', '')
                    name = player_cell.text.strip()
                    
                    # Extract stats
                    def get_stat(stat_name):
                        cell = row.find('td', {'data-stat': stat_name})
                        if cell and cell.text.strip():
                            try:
                                return int(cell.text.strip())
                            except ValueError:
                                return 0
                        return 0
                    
                    player_data = {
                        'player_id': player_id,
                        'season': year,
                        'name': name,
                        'team': row.find('td', {'data-stat': 'team_name_abbr'}).text.strip() if row.find('td', {'data-stat': 'team_name_abbr'}) else 'UNK',
                        'position': row.find('td', {'data-stat': 'pos'}).text.strip() if row.find('td', {'data-stat': 'pos'}) else 'UNK',
                        'age': get_stat('age'),
                        'games_played': get_stat('games'),
                        'rushing_yards': get_stat('rush_yds'),
                        'rushing_attempts': get_stat('rush_att')
                    }
                    
                    players.append(player_data)
                    
                except Exception as e:
                    logger.warning(f"Error parsing row: {e}")
                    continue
            
            logger.info(f"Successfully scraped {len(players)} rushing records")
            return players
            
        except requests.RequestException as e:
            logger.error(f"Error fetching rushing stats: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in rushing stats scraping: {e}")
            return []
    
    def scrape_receiving_stats(self, year: int = 2024) -> List[Dict]:
        """Scrape receiving stats from Pro-Football-Reference"""
        logger.info(f"Scraping receiving stats for {year}...")
        
        url = f"https://www.pro-football-reference.com/years/{year}/receiving.htm"
        
        try:
            # Add random delay
            time.sleep(random.uniform(1, 3))
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the receiving stats table
            table = soup.find('table', {'id': 'receiving'})
            if not table:
                logger.error("Could not find receiving stats table")
                return []
            
            players = []
            
            rows = table.find_all('tr')
            
            for row in rows:
                try:
                    # Skip header rows and rows without data
                    if row.find('th', {'scope': 'col'}):
                        continue
                        
                    # Extract player data
                    player_cell = row.find('td', {'data-stat': 'name_display'})
                    if not player_cell:
                        continue
                    
                    player_link = player_cell.find('a')
                    if not player_link:
                        continue
                    
                    player_id = player_link.get('href', '').split('/')[-1].replace('.htm', '')
                    
                    def get_stat(stat_name):
                        cell = row.find('td', {'data-stat': stat_name})
                        if cell and cell.text.strip():
                            try:
                                return int(cell.text.strip())
                            except ValueError:
                                return 0
                        return 0
                    
                    player_data = {
                        'player_id': player_id,
                        'receiving_yards': get_stat('rec_yds'),
                        'receptions': get_stat('rec'),
                        'targets': get_stat('tgt')
                    }
                    
                    players.append(player_data)
                    
                except Exception as e:
                    logger.warning(f"Error parsing receiving row: {e}")
                    continue
            
            logger.info(f"Successfully scraped {len(players)} receiving records")
            return players
            
        except requests.RequestException as e:
            logger.error(f"Error fetching receiving stats: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in receiving stats scraping: {e}")
            return []
    
    def merge_and_save_stats(self, rushing_stats: List[Dict], receiving_stats: List[Dict], year: int):
        """Merge rushing and receiving stats and save to database"""
        logger.info("Merging and saving player stats...")
        
        # Convert to DataFrames for easier merging
        rushing_df = pd.DataFrame(rushing_stats)
        receiving_df = pd.DataFrame(receiving_stats)
        
        # Merge on player_id
        if not rushing_df.empty and not receiving_df.empty:
            # Remove duplicates from rushing_df first (some players may appear multiple times)
            rushing_df = rushing_df.drop_duplicates(subset=['player_id'], keep='first')
            receiving_df = receiving_df.drop_duplicates(subset=['player_id'], keep='first')
            
            merged_df = rushing_df.merge(
                receiving_df[['player_id', 'receiving_yards', 'receptions', 'targets']], 
                on='player_id', 
                how='left'
            )
            
            # Fill NaN values with 0 for players without receiving stats
            merged_df = merged_df.fillna(0)
            
            # Convert float columns to int
            int_columns = ['age', 'games_played', 'rushing_yards', 'rushing_attempts', 
                          'receiving_yards', 'receptions', 'targets']
            for col in int_columns:
                if col in merged_df.columns:
                    merged_df[col] = merged_df[col].astype(int)
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            try:
                # Clear existing data for this season to avoid duplicates
                conn.execute("DELETE FROM player_stats WHERE season = ?", (year,))
                
                # Insert new data
                merged_df.to_sql('player_stats', conn, if_exists='append', index=False)
                conn.commit()
                logger.info(f"Saved {len(merged_df)} player records to database")
            except Exception as e:
                logger.error(f"Error saving to database: {e}")
                conn.rollback()
            finally:
                conn.close()
    
    def collect_season_data(self, year: int = 2024):
        """Collect all data for a given season"""
        logger.info(f"Starting data collection for {year} season...")
        
        # Scrape rushing stats
        rushing_stats = self.scrape_rushing_stats(year)
        
        # Scrape receiving stats
        receiving_stats = self.scrape_receiving_stats(year)
        
        # Merge and save
        if rushing_stats:
            self.merge_and_save_stats(rushing_stats, receiving_stats, year)
        else:
            logger.error("No data collected to save")
    
    def collect_betting_lines(self):
        """Collect current betting lines from sportsbooks"""
        logger.info("Collecting betting lines...")
        
        # This is still a template - real implementation would need:
        # 1. API keys for odds providers (e.g., The Odds API)
        # 2. Or web scraping of individual sportsbooks
        # 3. Proper data transformation
        
        logger.info("Betting lines collection not yet implemented")
        logger.info("Consider using The Odds API or similar service")
        
        return []
    
    def get_stored_stats(self, season: Optional[int] = None) -> pd.DataFrame:
        """Retrieve stored stats from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM player_stats"
        params = []
        
        if season:
            query += " WHERE season = ?"
            params.append(season)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df

# Initialize data collector
if __name__ == "__main__":
    collector = NFLDataCollector()
    
    print("NFL Data Collection Pipeline")
    print("=" * 50)
    
    # Collect data for 2023 season (2024 data might not be complete)
    collector.collect_season_data(2023)
    
    # Display sample of collected data
    df = collector.get_stored_stats(2023)
    if not df.empty:
        print(f"\nCollected {len(df)} player records")
        print("\nSample data:")
        print(df.head())
    else:
        print("\nNo data collected. Check logs for errors.")
    
    print("\nNext Steps:")
    print("1. Run for multiple seasons to build historical data")
    print("2. Implement betting lines collection")
    print("3. Add team context data scraping")
    print("4. Set up automated daily updates")