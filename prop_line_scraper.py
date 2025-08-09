#!/usr/bin/env python3
"""
NFL Player Prop Line Scraper for 2025-2026 Season
Retrieves current season-long prop lines from multiple sportsbooks
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
from dataclasses import dataclass, asdict
import sqlite3
import os
from pathlib import Path

# Import caching system
from cache_manager import cached_client
from config import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PropLine:
    """Data class for player prop lines"""
    player: str
    team: str
    position: str
    book: str
    stat: str
    line: float
    over_odds: int
    under_odds: int
    last_updated: str
    season: str = "2025-2026"

class NFLPropScraper:
    """NFL Player Prop Line Scraper"""
    
    def __init__(self, odds_api_key: Optional[str] = None):
        self.odds_api_key = odds_api_key or config.api.odds_api_key or os.getenv('ODDS_API_KEY')
        self.base_url = "https://api.the-odds-api.com/v4"
        self.db_path = "nfl_prop_lines.db"
        # Use cached client instead of plain requests.Session
        self.client = cached_client
        
        # NFL team mapping
        self.team_mapping = {
            'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
            'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
            'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
            'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
            'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
            'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
            'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
            'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
            'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
            'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
            'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
        }
        
        # Position mapping for common names
        self.position_mapping = {
            'Running Back': 'RB', 'Wide Receiver': 'WR', 'Tight End': 'TE',
            'Quarterback': 'QB', 'RB': 'RB', 'WR': 'WR', 'TE': 'TE', 'QB': 'QB'
        }
        
        # Initialize database
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for storing prop lines"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS prop_lines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player TEXT NOT NULL,
                team TEXT NOT NULL,
                position TEXT NOT NULL,
                book TEXT NOT NULL,
                stat TEXT NOT NULL,
                line REAL NOT NULL,
                over_odds INTEGER NOT NULL,
                under_odds INTEGER NOT NULL,
                last_updated TEXT NOT NULL,
                season TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(player, book, stat, season)
            )
        ''')
        # Weekly prop lines (game-level markets)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS weekly_prop_lines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                week INTEGER NOT NULL,
                season INTEGER NOT NULL,
                player TEXT NOT NULL,
                team TEXT NOT NULL,
                position TEXT NOT NULL,
                book TEXT NOT NULL,
                stat TEXT NOT NULL,
                line REAL NOT NULL,
                over_odds INTEGER,
                under_odds INTEGER,
                game_date DATE,
                home_team TEXT,
                away_team TEXT,
                last_updated TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(player, book, stat, week, season)
            )
        ''')
        conn.commit()
        conn.close()
    
    def get_odds_api_props(self) -> List[PropLine]:
        """Retrieve prop lines from The Odds API"""
        if not self.odds_api_key:
            logger.warning("No Odds API key provided. Skipping API call.")
            return []
        
        prop_lines = []
        
        # Season-long props markets
        markets = [
            'player_pass_yds',
            'player_rush_yds', 
            'player_rec_yds',
            'player_pass_tds',
            'player_rush_tds',
            'player_rec_tds'
        ]
        
        for market in markets:
            try:
                url = f"{self.base_url}/sports/americanfootball_nfl/odds"
                params = {
                    'apiKey': self.odds_api_key,
                    'regions': 'us',
                    'markets': market,
                    'oddsFormat': 'american',
                    'dateFormat': 'iso'
                }
                
                # Use cached client with odds API type and smart caching
                response = self.client.get(
                    url, 
                    params=params, 
                    api_type='odds',
                    force_refresh=config.api.force_cache_refresh,
                    allow_stale=not config.api.cache_offline_mode
                )
                response.raise_for_status()
                
                data = response.json()
                prop_lines.extend(self._parse_odds_api_response(data, market))
                
                # Cache handles rate limiting now, but keep minimal delay for safety
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching {market} from Odds API: {e}")
                continue
        
        return prop_lines
    
    def _parse_odds_api_response(self, data: Dict, market: str) -> List[PropLine]:
        """Parse response from Odds API"""
        prop_lines = []
        
        # Map market names to our stat categories
        stat_mapping = {
            'player_pass_yds': 'passing_yards',
            'player_rush_yds': 'rushing_yards',
            'player_rec_yds': 'receiving_yards',
            'player_pass_tds': 'passing_touchdowns',
            'player_rush_tds': 'rushing_touchdowns',
            'player_rec_tds': 'receiving_touchdowns'
        }
        
        stat_category = stat_mapping.get(market, market)
        
        for game in data:
            for bookmaker in game.get('bookmakers', []):
                book_name = bookmaker.get('title', 'Unknown')
                
                for market_data in bookmaker.get('markets', []):
                    if market_data.get('key') != market:
                        continue
                    
                    for outcome in market_data.get('outcomes', []):
                        player_name = outcome.get('description', '')
                        
                        # Extract player name and team (format varies by book)
                        player_info = self._extract_player_info(player_name, game.get('home_team', ''), game.get('away_team', ''))
                        
                        if not player_info:
                            continue
                        
                        # Get over/under odds
                        over_odds = outcome.get('price') if outcome.get('name') == 'Over' else None
                        under_odds = outcome.get('price') if outcome.get('name') == 'Under' else None
                        line = outcome.get('point', 0)
                        
                        # Find corresponding under/over
                        if over_odds is not None:
                            under_outcome = next((o for o in market_data.get('outcomes', []) if o.get('description') == player_name and o.get('name') == 'Under'), None)
                            if under_outcome:
                                under_odds = under_outcome.get('price')
                        
                        if under_odds is not None and over_odds is None:
                            over_outcome = next((o for o in market_data.get('outcomes', []) if o.get('description') == player_name and o.get('name') == 'Over'), None)
                            if over_outcome:
                                over_odds = over_outcome.get('price')
                        
                        if over_odds is not None and under_odds is not None:
                            prop_line = PropLine(
                                player=player_info['name'],
                                team=player_info['team'],
                                position=player_info['position'],
                                book=book_name,
                                stat=stat_category,
                                line=line,
                                over_odds=over_odds,
                                under_odds=under_odds,
                                last_updated=datetime.now().isoformat()
                            )
                            prop_lines.append(prop_line)
        
        return prop_lines

    # ------------------------ Weekly odds (Week N) ------------------------
    def get_upcoming_week_props(self, week: int, season: int) -> List[Dict]:
        """Fetch weekly player props for a given NFL week.
        Returns list of dicts with keys: player, team, position, book, stat, line, over_odds, under_odds,
        game_date, home_team, away_team
        """
        markets = ['player_rush_yds', 'player_rec_yds', 'player_pass_yds']
        results: List[Dict] = []
        if not self.odds_api_key:
            # Fallback: generate a larger synthetic sample for demo purposes
            import random
            now = datetime.now().date().isoformat()
            names = [
                ("CeeDee Lamb","DAL","WR"),("Tyreek Hill","MIA","WR"),("Amon-Ra St. Brown","DET","WR"),
                ("Davante Adams","NYJ","WR"),("Justin Jefferson","MIN","WR"),("Ja'Marr Chase","CIN","WR"),
                ("Stefon Diggs","HOU","WR"),("Travis Kelce","KC","TE"),("Mark Andrews","BAL","TE"),
                ("Josh Jacobs","GB","RB"),("Bijan Robinson","ATL","RB"),("Breece Hall","NYJ","RB"),
                ("Christian McCaffrey","SF","RB"),("Saquon Barkley","PHI","RB"),("Josh Allen","BUF","QB"),
                ("Lamar Jackson","BAL","QB"),("Patrick Mahomes","KC","QB"),("C.J. Stroud","HOU","QB"),
            ]
            books = ["DraftKings","FanDuel","BetMGM","Caesars"]
            stats = ["rushing_yards","receiving_yards","passing_yards"]
            rows: List[Dict] = []
            random.seed(42)
            for _ in range(50):
                name, team, pos = random.choice(names)
                stat = random.choice(stats)
                book = random.choice(books)
                base = 90 if stat=="receiving_yards" else (75 if stat=="rushing_yards" else 260)
                jitter = random.uniform(-25, 25)
                line = max(10.0, round(base + jitter, 1))
                over = random.choice([-120,-115,-110,-105,100,105])
                under = random.choice([-120,-115,-110,-105,100,105])
                rows.append({
                    "player": name, "team": team, "position": pos, "book": book,
                    "stat": stat, "line": line, "over_odds": over, "under_odds": under,
                    "game_date": now, "home_team": team, "away_team": "TBD"
                })
            return rows

        for market in markets:
            try:
                url = f"{self.base_url}/sports/americanfootball_nfl/odds"
                params = {
                    'apiKey': self.odds_api_key,
                    'regions': 'us',
                    'markets': market,
                    'oddsFormat': 'american',
                    'dateFormat': 'iso'
                }
                response = self.client.get(url, params=params, api_type='odds')
                response.raise_for_status()
                data = response.json()
                stat_mapping = {
                    'player_pass_yds': 'passing_yards',
                    'player_rush_yds': 'rushing_yards',
                    'player_rec_yds': 'receiving_yards',
                }
                stat_category = stat_mapping.get(market, market)
                for game in data:
                    home = game.get('home_team','')
                    away = game.get('away_team','')
                    game_date = game.get('commence_time','')
                    for bookmaker in game.get('bookmakers', []):
                        book_name = bookmaker.get('title', 'Unknown')
                        for market_data in bookmaker.get('markets', []):
                            if market_data.get('key') != market:
                                continue
                            for outcome in market_data.get('outcomes', []):
                                player_desc = outcome.get('description','')
                                info = self._extract_player_info(player_desc, home, away)
                                if not info:
                                    continue
                                over_odds = outcome.get('price') if outcome.get('name') == 'Over' else None
                                under_odds = outcome.get('price') if outcome.get('name') == 'Under' else None
                                line = outcome.get('point', 0.0)
                                if over_odds is not None and under_odds is None:
                                    # try to find match
                                    uo = next((o for o in market_data.get('outcomes', []) if o.get('description') == player_desc and o.get('name') == 'Under'), None)
                                    if uo:
                                        under_odds = uo.get('price')
                                if under_odds is not None and over_odds is None:
                                    oo = next((o for o in market_data.get('outcomes', []) if o.get('description') == player_desc and o.get('name') == 'Over'), None)
                                    if oo:
                                        over_odds = oo.get('price')
                                if over_odds is None or under_odds is None:
                                    continue
                                results.append({
                                    'player': info['name'], 'team': info['team'], 'position': info['position'],
                                    'book': book_name, 'stat': stat_category, 'line': line,
                                    'over_odds': over_odds, 'under_odds': under_odds,
                                    'game_date': game_date, 'home_team': home, 'away_team': away
                                })
                time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Weekly market {market} failed: {e}")
                continue
        return results

    def save_weekly_prop_lines(self, rows: List[Dict], week: int, season: int):
        if not rows:
            return
        conn = sqlite3.connect(self.db_path)
        for r in rows:
            try:
                conn.execute('''
                    INSERT OR REPLACE INTO weekly_prop_lines
                    (week, season, player, team, position, book, stat, line, over_odds, under_odds, game_date, home_team, away_team, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    week, season, r['player'], r['team'], r['position'], r['book'], r['stat'],
                    r['line'], r.get('over_odds'), r.get('under_odds'), r.get('game_date'), r.get('home_team'), r.get('away_team'), datetime.now().isoformat()
                ))
            except Exception as e:
                logger.warning(f"Save weekly row failed for {r.get('player')}: {e}")
        conn.commit()
        conn.close()

    def run_weekly_update(self, week: int, season: int) -> pd.DataFrame:
        logger.info(f"Starting weekly prop line update for week={week}, season={season}...")
        rows = self.get_upcoming_week_props(week, season)
        self.save_weekly_prop_lines(rows, week, season)
        # Export CSV
        df = pd.DataFrame(rows)
        out = Path('reports') / f"week_{week}_prop_lines.csv"
        out.parent.mkdir(exist_ok=True)
        df.to_csv(out, index=False)
        logger.info(f"Saved weekly prop lines CSV: {out}")
        return df
    
    def _extract_player_info(self, player_description: str, home_team: str, away_team: str) -> Optional[Dict]:
        """Extract player name, team, and position from description"""
        # This is a simplified version - in practice you'd need a player database
        # to accurately map names to teams and positions
        
        # Basic extraction (format varies by sportsbook)
        if ' - ' in player_description:
            parts = player_description.split(' - ')
            player_name = parts[0].strip()
            team_info = parts[1].strip() if len(parts) > 1 else ''
        else:
            player_name = player_description.strip()
            team_info = ''
        
        # Try to determine team
        team = self._guess_team(player_name, home_team, away_team, team_info)
        
        # Try to determine position (would need player database for accuracy)
        position = self._guess_position(player_name)
        
        return {
            'name': player_name,
            'team': team,
            'position': position
        }
    
    def _guess_team(self, player_name: str, home_team: str, away_team: str, team_info: str) -> str:
        """Guess player's team (simplified - would need player database)"""
        # Check if team info contains team name
        for full_name, abbrev in self.team_mapping.items():
            if full_name.lower() in team_info.lower():
                return abbrev
        
        # Default to home team (not accurate, but placeholder)
        return self.team_mapping.get(home_team, home_team[:3].upper())
    
    def _guess_position(self, player_name: str) -> str:
        """Guess player's position (would need player database)"""
        # This is a placeholder - in practice you'd need a comprehensive player database
        return 'UNKNOWN'
    
    def get_sample_prop_lines(self) -> List[PropLine]:
        """Generate sample prop lines for testing"""
        logger.info("Generating sample prop lines for testing...")
        
        sample_data = [
            # Running Backs
            {'player': 'Christian McCaffrey', 'team': 'SF', 'position': 'RB', 'book': 'DraftKings', 
             'stat': 'rushing_yards', 'line': 1250.5, 'over_odds': -115, 'under_odds': -105},
            {'player': 'Derrick Henry', 'team': 'BAL', 'position': 'RB', 'book': 'FanDuel',
             'stat': 'rushing_yards', 'line': 1100.5, 'over_odds': -110, 'under_odds': -110},
            {'player': 'Josh Jacobs', 'team': 'GB', 'position': 'RB', 'book': 'BetMGM',
             'stat': 'rushing_yards', 'line': 950.5, 'over_odds': -108, 'under_odds': -112},
            
            # Wide Receivers
            {'player': 'Tyreek Hill', 'team': 'MIA', 'position': 'WR', 'book': 'DraftKings',
             'stat': 'receiving_yards', 'line': 1350.5, 'over_odds': -120, 'under_odds': +100},
            {'player': 'CeeDee Lamb', 'team': 'DAL', 'position': 'WR', 'book': 'FanDuel',
             'stat': 'receiving_yards', 'line': 1200.5, 'over_odds': -115, 'under_odds': -105},
            {'player': 'Davante Adams', 'team': 'NYJ', 'position': 'WR', 'book': 'Caesars',
             'stat': 'receiving_yards', 'line': 1050.5, 'over_odds': -110, 'under_odds': -110},
            
            # Quarterbacks
            {'player': 'Josh Allen', 'team': 'BUF', 'position': 'QB', 'book': 'DraftKings',
             'stat': 'passing_yards', 'line': 4200.5, 'over_odds': -110, 'under_odds': -110},
            {'player': 'Lamar Jackson', 'team': 'BAL', 'position': 'QB', 'book': 'FanDuel',
             'stat': 'passing_yards', 'line': 3800.5, 'over_odds': -115, 'under_odds': -105},
            
            # Tight Ends
            {'player': 'Travis Kelce', 'team': 'KC', 'position': 'TE', 'book': 'BetMGM',
             'stat': 'receiving_yards', 'line': 950.5, 'over_odds': -112, 'under_odds': -108},
        ]
        
        prop_lines = []
        for data in sample_data:
            prop_line = PropLine(
                player=data['player'],
                team=data['team'],
                position=data['position'],
                book=data['book'],
                stat=data['stat'],
                line=data['line'],
                over_odds=data['over_odds'],
                under_odds=data['under_odds'],
                last_updated=datetime.now().isoformat()
            )
            prop_lines.append(prop_line)
        
        return prop_lines
    
    def save_prop_lines(self, prop_lines: List[PropLine]):
        """Save prop lines to database"""
        if not prop_lines:
            logger.warning("No prop lines to save")
            return
        
        conn = sqlite3.connect(self.db_path)
        
        for prop_line in prop_lines:
            try:
                conn.execute('''
                    INSERT OR REPLACE INTO prop_lines 
                    (player, team, position, book, stat, line, over_odds, under_odds, last_updated, season)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    prop_line.player, prop_line.team, prop_line.position, prop_line.book,
                    prop_line.stat, prop_line.line, prop_line.over_odds, prop_line.under_odds,
                    prop_line.last_updated, prop_line.season
                ))
            except Exception as e:
                logger.error(f"Error saving prop line for {prop_line.player}: {e}")
                continue
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(prop_lines)} prop lines to database")
    
    def get_prop_lines_dataframe(self) -> pd.DataFrame:
        """Get all prop lines as a DataFrame"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT player, team, position, book, stat, line, over_odds, under_odds, last_updated, season
            FROM prop_lines
            ORDER BY player, stat, book
        ''', conn)
        conn.close()
        return df
    
    def flag_suspicious_lines(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag players with suspiciously low lines compared to typical values"""
        
        # Define thresholds for suspicious lines
        thresholds = {
            'rushing_yards': {'RB': 800, 'QB': 400},
            'receiving_yards': {'WR': 800, 'TE': 600, 'RB': 300},
            'passing_yards': {'QB': 3500}
        }
        
        df['suspicious_line'] = False
        df['suspicious_reason'] = ''
        
        for _, row in df.iterrows():
            stat = row['stat']
            position = row['position']
            line = row['line']
            
            if stat in thresholds and position in thresholds[stat]:
                threshold = thresholds[stat][position]
                if line < threshold:
                    df.loc[df.index == row.name, 'suspicious_line'] = True
                    df.loc[df.index == row.name, 'suspicious_reason'] = f"Line ({line}) below typical {position} threshold ({threshold})"
        
        return df
    
def run_season_update(self):
        """Legacy season-long update retained for backward compatibility."""
        logger.info("Starting season-long prop line update...")
        prop_lines = self.get_odds_api_props()
        if not prop_lines:
            logger.info("Using sample prop lines (no API data available)")
            prop_lines = self.get_sample_prop_lines()
        self.save_prop_lines(prop_lines)
        df = self.get_prop_lines_dataframe()
        df = self.flag_suspicious_lines(df)
        df.to_csv('current_prop_lines.csv', index=False)
        logger.info(f"Update complete: {len(prop_lines)} lines retrieved")
        logger.info(f"Suspicious lines found: {df['suspicious_line'].sum()}")
        if df['suspicious_line'].sum() > 0:
            suspicious_lines = df[df['suspicious_line']]
            for _, row in suspicious_lines.iterrows():
                print(f"  {row['player']} - {row['stat']}: {row['line']} ({row.get('suspicious_reason', 'Unknown')})")
        return df

def main():
    """Main function to run the prop line scraper"""
    
    # Initialize scraper
    scraper = NFLPropScraper()
    
    # Check if Odds API key is available
    if scraper.odds_api_key:
        print("✅ Odds API key found")
    else:
        print("⚠️  No Odds API key found. Using sample data.")
        print("   To use real data, set ODDS_API_KEY environment variable")
        print("   or get a free key from https://the-odds-api.com/")
    
    # Run weekly update
    df = scraper.run_season_update()
    
    # Display summary
    print(f"📊 PROP LINE SUMMARY")
    print(f"Total lines: {len(df)}")
    print(f"Unique players: {df['player'].nunique()}")
    print(f"Unique books: {df['book'].nunique()}")
    print(f"Stats covered: {', '.join(df['stat'].unique())}")
    
    # Show sample of data
    print(f"📋 SAMPLE PROP LINES:")
    print(df.head(10).to_string(index=False))
    
    print(f"💾 Data saved to:")
    print(f"  - Database: {scraper.db_path}")
    print(f"  - CSV: current_prop_lines.csv")

if __name__ == "__main__":
    main()