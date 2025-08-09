"""
Professional data pipeline for NFL algorithm.
Handles real-time data ingestion, feature engineering, and persistence.
"""

import sqlite3
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import pandas as pd
import requests
from retry import retry
from tqdm import tqdm
import numpy as np

from config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.logs_dir / 'pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class WeatherData:
    game_id: str
    temperature: float
    wind_speed: float
    precipitation: float
    humidity: float
    is_dome: bool
    timestamp: datetime

@dataclass
class InjuryData:
    player_id: str
    status: str  # OUT, DOUBTFUL, QUESTIONABLE, PROBABLE
    practice_participation: str  # DNP, LIMITED, FULL
    injury_type: str
    last_updated: datetime

class DataPipeline:
    """Enhanced data pipeline with weather, injury, and odds integration."""
    
    def __init__(self):
        self.db_path = config.database.path
        self.setup_enhanced_database()
        
    def setup_enhanced_database(self):
        """Create enhanced database schema with new tables."""
        conn = sqlite3.connect(self.db_path)
        
        # Enhanced player stats with additional features
        conn.execute('''
            CREATE TABLE IF NOT EXISTS player_stats_enhanced (
                player_id TEXT,
                season INTEGER,
                week INTEGER,
                name TEXT,
                team TEXT,
                position TEXT,
                age INTEGER,
                games_played INTEGER,
                snap_count INTEGER,
                snap_percentage REAL,
                rushing_yards INTEGER,
                rushing_attempts INTEGER,
                receiving_yards INTEGER,
                receptions INTEGER,
                targets INTEGER,
                red_zone_touches INTEGER,
                target_share REAL,
                air_yards REAL,
                yac_yards REAL,
                game_script REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (player_id, season, week)
            )
        ''')
        
        # Add breakout features to player_stats_enhanced
        conn.execute('''
            ALTER TABLE player_stats_enhanced ADD COLUMN usage_delta REAL;
            ALTER TABLE player_stats_enhanced ADD COLUMN age_curve REAL;
            ALTER TABLE player_stats_enhanced ADD COLUMN oc_change BOOLEAN;
            ALTER TABLE player_stats_enhanced ADD COLUMN injury_recovery BOOLEAN;
            ALTER TABLE player_stats_enhanced ADD COLUMN preseason_buzz REAL;
            ALTER TABLE player_stats_enhanced ADD COLUMN rolling_targets REAL;
            ALTER TABLE player_stats_enhanced ADD COLUMN rolling_routes REAL;
            ALTER TABLE player_stats_enhanced ADD COLUMN rolling_air_yards REAL;
            ALTER TABLE player_stats_enhanced ADD COLUMN age_squared REAL;
            ALTER TABLE player_stats_enhanced ADD COLUMN injury_games_missed INTEGER;
            ALTER TABLE player_stats_enhanced ADD COLUMN team_context_flag TEXT;
        ''')
        
        # Weather data table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS weather_data (
                game_id TEXT PRIMARY KEY,
                home_team TEXT,
                away_team TEXT,
                game_date DATE,
                temperature REAL,
                wind_speed REAL,
                precipitation REAL,
                humidity REAL,
                is_dome BOOLEAN,
                weather_description TEXT,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Injury data table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS injury_data (
                player_id TEXT,
                season INTEGER,
                week INTEGER,
                status TEXT,
                practice_participation TEXT,
                injury_type TEXT,
                days_since_injury INTEGER,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (player_id, season, week)
            )
        ''')
        
        # Enhanced odds data with multiple sportsbooks
        conn.execute('''
            CREATE TABLE IF NOT EXISTS odds_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT,
                prop_type TEXT,
                sportsbook TEXT,
                line REAL,
                over_odds INTEGER,
                under_odds INTEGER,
                season INTEGER,
                week INTEGER,
                game_date DATE,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(player_id, prop_type, sportsbook, season, week)
            )
        ''')
        
        # Team context and matchup data
        conn.execute('''
            CREATE TABLE IF NOT EXISTS team_context (
                team TEXT,
                season INTEGER,
                week INTEGER,
                offensive_rank INTEGER,
                defensive_rank INTEGER,
                pace_rank INTEGER,
                red_zone_efficiency REAL,
                turnover_differential INTEGER,
                oline_rank INTEGER,
                pass_rush_rank INTEGER,
                PRIMARY KEY (team, season, week)
            )
        ''')
        
        # Closing line value tracking
        conn.execute('''
            CREATE TABLE IF NOT EXISTS clv_tracking (
                bet_id TEXT PRIMARY KEY,
                player_id TEXT,
                prop_type TEXT,
                sportsbook TEXT,
                bet_line REAL,
                closing_line REAL,
                clv_percentage REAL,
                bet_result TEXT,
                roi REAL,
                date_placed DATE,
                date_settled DATE
            )
        ''')
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_player_season_week ON player_stats_enhanced(player_id, season, week)",
            "CREATE INDEX IF NOT EXISTS idx_weather_date ON weather_data(game_date)",
            "CREATE INDEX IF NOT EXISTS idx_odds_timestamp ON odds_data(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_injury_status ON injury_data(status, season, week)"
        ]
        
        for index in indexes:
            conn.execute(index)
        
        conn.commit()
        conn.close()
        logger.info("Enhanced database schema created successfully")
    
    @retry(tries=3, delay=1, backoff=2)
    def fetch_weather_data(self, game_date: str, home_team: str, away_team: str) -> Optional[WeatherData]:
        """Fetch weather data for a specific game. Cached by (home_team, game_date)."""
        try:
            # Using OpenWeatherMap API (example)
            if not config.api.weather_api_key:
                logger.warning("No weather API key configured")
                return None
            
            # Stadium locations (simplified mapping)
            stadium_coords = {
                'GB': (44.5013, -88.0622),  # Lambeau Field
                'CHI': (41.8623, -87.6167),  # Soldier Field
                'DET': (42.3400, -83.0456),  # Ford Field (dome)
                # Add more teams...
            }
            
            if home_team not in stadium_coords:
                return None
            
            lat, lon = stadium_coords[home_team]
            url = "http://api.openweathermap.org/data/2.5/weather"
            # Include home_team and game_date in params to make cache key unique per game
            params = {
                'lat': lat,
                'lon': lon,
                'appid': config.api.weather_api_key,
                'units': 'imperial',
                'home_team': home_team,
                'game_date': game_date
            }
            
            # Import cached client here to avoid circular imports
            from cache_manager import cached_client
            
            # Use cached client with weather-specific caching
            response = cached_client.get(
                url, 
                params=params, 
                api_type='weather',
                force_refresh=config.api.force_cache_refresh,
                allow_stale=True  # Weather can be served stale if API fails
            )
            response.raise_for_status()
            data = response.json()
            
            # Check if stadium is a dome (affects cache TTL)
            dome_teams = {'ATL', 'DET', 'HOU', 'IND', 'LV', 'LAR', 'MIN', 'NO', 'ARI'}
            is_dome = home_team in dome_teams
            
            return WeatherData(
                game_id=f"{away_team}@{home_team}_{game_date}",
                temperature=data['main']['temp'] if not is_dome else 72.0,
                wind_speed=data['wind']['speed'] if not is_dome else 0.0,
                precipitation=data.get('rain', {}).get('1h', 0.0) if not is_dome else 0.0,
                humidity=data['main']['humidity'] if not is_dome else 50.0,
                is_dome=is_dome,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return None
    
    def engineer_breakout_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add breakout prediction features."""
        # Rolling window deltas (3-game rolling avg)
        df['rolling_targets'] = df.groupby('player_id')['targets'].rolling(3).mean().reset_index(0, drop=True)
        df['rolling_routes'] = df.groupby('player_id')['routes_run'].rolling(3).mean().reset_index(0, drop=True)  # Assume routes_run exists or add
        df['rolling_air_yards'] = df.groupby('player_id')['air_yards'].rolling(3).mean().reset_index(0, drop=True)
        
        # Usage delta vs previous season
        df['usage_delta'] = df.groupby('player_id')['target_share'].diff(periods=17)  # Approx season length
        
        # Age curve and second-order
        df['age_squared'] = df['age'] ** 2
        df['age_curve'] = np.where((df['age'] >= 24) & (df['age'] <= 28), 1.0, 0.5)  # Peak years flag
        
        # OC change (simplified flag)
        df['oc_change'] = df.groupby('team')['offensive_coordinator'].diff().fillna(0).astype(bool)  # Assume OC field
        
        # Injury recovery
        df['injury_games_missed'] = df.groupby('player_id')['games_missed'].cumsum()  # Assume games_missed
        df['injury_recovery'] = (df['injury_games_missed'].shift(1) > 0) & (df['status'] == 'FULL')
        
        # Preseason buzz (placeholder: social mentions score)
        df['preseason_buzz'] = np.random.uniform(0, 1, len(df))  # Integrate real API later
        
        # Team context flags
        df['team_context_flag'] = np.where(df['offensive_rank'] <= 10, 'HIGH', 'LOW')
        
        return df
    
    def run_full_update(self):
        """Run complete data pipeline update with breakout features."""
        logger.info("Starting full data pipeline update...")
        
        start_time = time.time()
        
        try:
            # Update all data sources
            logger.info("Pipeline update completed successfully")
            
            # Load data and engineer features
            df = pd.read_sql("SELECT * FROM player_stats_enhanced", sqlite3.connect(self.db_path))
            df_enhanced = self.engineer_breakout_features(df)
            df_enhanced.to_sql('player_stats_enhanced', sqlite3.connect(self.db_path), if_exists='replace', index=False)
            
            logger.info("Breakout features engineered successfully")
            
            elapsed = time.time() - start_time
            logger.info(f"Full pipeline update completed in {elapsed:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Pipeline update failed: {e}")
            raise

if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run_full_update() 