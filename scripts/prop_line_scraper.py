#!/usr/bin/env python3
"""
NFL Player Prop Line Scraper for 2025-2026 Season
Retrieves current season-long prop lines from multiple sportsbooks
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import requests

# Import simplified caching system and validation
from scripts.simple_cache import simple_cached_client
from scripts.data_validation import data_validator, api_error_handler
from config import config
from utils.player_id_utils import make_player_id
from utils.db import get_connection, execute

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
        # Use the configured database backend via utils.db
        # Use cached client instead of plain requests.Session
        self.client = simple_cached_client
        
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
        with get_connection() as conn:
            execute('''
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
        ''', conn=conn)
            # Weekly prop lines (game-level markets)
            execute('''
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
        ''', conn=conn)

    @staticmethod
    def _save_snapshot(path: Path, snapshot: Dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2))
        logger.info("Saved odds snapshot to %s", path)

    @staticmethod
    def _load_snapshot(path: Optional[Path]) -> Optional[Dict]:
        if not path:
            return None
        path = Path(path)
        if not path.exists():
            logger.warning("Snapshot path %s not found for fallback", path)
            return None
        try:
            data = json.loads(path.read_text())
            logger.info("Loaded odds snapshot from %s", path)
            return data
        except Exception as exc:
            logger.error("Failed to load snapshot %s: %s", path, exc)
            return None
    
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
                
                # Use cached client with odds API type
                response = self.client.get(url, params=params, api_type='odds')
                response.raise_for_status()
                
                data = response.json()
                
                # Validate API response
                is_valid, errors = data_validator.validate_apis_response(data, 'odds')
                if not is_valid:
                    logger.warning(f"Invalid odds API response: {errors}")
                    # Try to recover by parsing partial data
                    prop_lines.extend(self._parse_odds_api_response(data, market, validate=False))
                else:
                    prop_lines.extend(self._parse_odds_api_response(data, market, validate=True))
                
                # Minimal delay between requests
                time.sleep(0.1)
                
            except Exception as e:
                # Handle API failure with circuit breaker
                error_result = api_error_handler.handle_api_failure('odds', market, e)
                
                if error_result['circuit_open']:
                    logger.error(f"Circuit breaker open for odds API market {market}")
                    continue
                
                if error_result['should_retry'] and error_result['wait_time'] > 0:
                    logger.info(f"Retrying odds API for {market} in {error_result['wait_time']}s")
                    time.sleep(error_result['wait_time'])
                    try:
                        response = self.client.get(url, params=params, api_type='odds')
                        response.raise_for_status()
                        data = response.json()
                        prop_lines.extend(self._parse_odds_api_response(data, market))
                    except Exception as retry_error:
                        logger.error(f"Retry failed for {market}: {retry_error}")
                
                logger.error(f"Error fetching {market} from Odds API: {e}")
                continue
        
        return prop_lines
    
    def _parse_odds_api_response(self, data: Dict, market: str, validate: bool = True) -> List[PropLine]:
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
    def get_upcoming_week_props(
        self,
        week: int,
        season: int,
        save_json_path: Optional[Path] = None,
        load_json_path: Optional[Path] = None,
    ) -> List[Dict]:
        """Fetch weekly player props for a given NFL week.
        Returns list of dicts with keys: player, team, position, book, stat, line, over_odds, under_odds,
        game_date, home_team, away_team
        """
        markets = ['player_rush_yds', 'player_rec_yds', 'player_pass_yds']
        results: List[Dict] = []
        fallback_snapshot = self._load_snapshot(load_json_path)
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
                    "source_player_id": None,
                    "player_id": make_player_id(name, team),
                    "player": name, "team": team, "position": pos, "book": book,
                    "stat": stat, "line": line, "over_odds": over, "under_odds": under,
                    "game_date": now, "home_team": team, "away_team": "TBD"
                })
            return rows

        snapshot = {
            "season": season,
            "week": week,
            "generated_at": datetime.utcnow().isoformat(),
            "events": [],
            "event_markets": {},
        }

        # First, get all upcoming events
        try:
            events_url = f"{self.base_url}/sports/americanfootball_nfl/events"
            events_params = {'apiKey': self.odds_api_key}
            events_response = self.client.get(events_url, params=events_params, api_type='odds')
            events_response.raise_for_status()
            events = events_response.json()
            logger.info(f"Found {len(events)} upcoming NFL games")
        except Exception as e:
            if fallback_snapshot:
                logger.warning(
                    "Failed to fetch NFL events (%s); using fallback snapshot",
                    e,
                )
                events = fallback_snapshot.get("events", [])
            else:
                logger.error(f"Failed to fetch NFL events: {e}")
                return []

        snapshot["events"] = events
        
        stat_mapping = {
            'player_pass_yds': 'passing_yards',
            'player_rush_yds': 'rushing_yards',
            'player_rec_yds': 'receiving_yards',
        }
        
        # Fetch props for each event and market
        for event in events[:10]:  # Limit to first 10 games to save API credits
            event_id = event.get('id')
            home = event.get('home_team', '')
            away = event.get('away_team', '')
            game_date = event.get('commence_time', '')
            
            for market in markets:
                url = f"{self.base_url}/sports/americanfootball_nfl/events/{event_id}/odds"
                params = {
                    'apiKey': self.odds_api_key,
                    'regions': 'us',
                    'markets': market,
                    'oddsFormat': 'american',
                    'dateFormat': 'iso'
                }
                market_payload = None
                for attempt in range(1, 4):
                    try:
                        response = self.client.get(url, params=params, api_type='odds')
                        response.raise_for_status()
                        market_payload = response.json()
                        break
                    except Exception as e:
                        if attempt >= 3:
                            logger.warning(
                                "Event %s market %s failed after %d attempts: %s",
                                event_id,
                                market,
                                attempt,
                                e,
                            )
                        else:
                            backoff = 0.4 * attempt
                            logger.warning(
                                "Event %s market %s attempt %d failed: %s. Retrying in %.1fs",
                                event_id,
                                market,
                                attempt,
                                e,
                                backoff,
                            )
                            time.sleep(backoff)
                if market_payload is None:
                    fallback_data = (
                        fallback_snapshot.get("event_markets", {}).get(event_id, {}).get(market)
                        if fallback_snapshot
                        else None
                    )
                    if fallback_data is None:
                        continue
                    logger.warning(
                        "Using fallback odds snapshot for event %s market %s",
                        event_id,
                        market,
                    )
                    market_payload = fallback_data
                    snapshot.setdefault("event_markets", {}).setdefault(event_id, {})[market] = market_payload
                else:
                    snapshot.setdefault("event_markets", {}).setdefault(event_id, {})[market] = market_payload

                stat_category = stat_mapping.get(market, market)

                # Data structure: single event with bookmakers
                for bookmaker in market_payload.get('bookmakers', []):
                    book_name = bookmaker.get('title', 'Unknown')
                    for market_data in bookmaker.get('markets', []):
                        if market_data.get('key') != market:
                            continue
                        for outcome in market_data.get('outcomes', []):
                            player_desc = outcome.get('description', '')
                            info = self._extract_player_info(player_desc, home, away)
                            if not info:
                                logger.debug(
                                    "Skipping outcome with unparsed player: raw=%s event=%s market=%s",
                                    player_desc,
                                    event_id,
                                    market,
                                )
                                continue
                            logger.debug(
                                "Parsed weekly odds player: raw=%s normalized=%s team=%s",
                                player_desc,
                                info['name'],
                                info['team'],
                            )
                            over_odds = outcome.get('price') if outcome.get('name') == 'Over' else None
                            under_odds = outcome.get('price') if outcome.get('name') == 'Under' else None
                            line = outcome.get('point', 0.0)
                            if over_odds is not None and under_odds is None:
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
                                'source_player_id': outcome.get('id'),
                                'player_id': make_player_id(info['name'], info['team']),
                                'player': info['name'], 'team': info['team'], 'position': info['position'],
                                'book': book_name, 'stat': stat_category, 'line': line,
                                'over_odds': over_odds, 'under_odds': under_odds,
                                'game_date': game_date, 'home_team': home, 'away_team': away
                            })

                time.sleep(0.2)
        if save_json_path:
            self._save_snapshot(Path(save_json_path), snapshot)
        return results

    def save_weekly_prop_lines(self, rows: List[Dict], week: int, season: int):
        if not rows:
            return
        with get_connection() as conn:
            for r in rows:
                try:
                    execute(
                        '''
                        INSERT INTO weekly_prop_lines
                        (week, season, player, team, position, book, stat, line, over_odds, under_odds, game_date, home_team, away_team, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(player, book, stat, week, season)
                        DO UPDATE SET
                            team = excluded.team,
                            position = excluded.position,
                            line = excluded.line,
                            over_odds = excluded.over_odds,
                            under_odds = excluded.under_odds,
                            game_date = excluded.game_date,
                            home_team = excluded.home_team,
                            away_team = excluded.away_team,
                            last_updated = excluded.last_updated
                        ''',
                        (
                            week,
                            season,
                            r['player'],
                            r['team'],
                            r['position'],
                            r['book'],
                            r['stat'],
                            r['line'],
                            r.get('over_odds'),
                            r.get('under_odds'),
                            r.get('game_date'),
                            r.get('home_team'),
                            r.get('away_team'),
                            datetime.now().isoformat(),
                        ),
                        conn=conn,
                    )
                except Exception as e:
                    logger.warning(f"Save weekly row failed for {r.get('player')}: {e}")
            try:
                conn.commit()
            except Exception:
                pass

    def run_weekly_update(
        self,
        week: int,
        season: int,
        save_json_path: Optional[Path] = None,
        load_json_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        logger.info(f"Starting weekly prop line update for week={week}, season={season}...")
        rows = self.get_upcoming_week_props(
            week,
            season,
            save_json_path=save_json_path,
            load_json_path=load_json_path,
        )
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
        
        with get_connection() as conn:
            for prop_line in prop_lines:
                try:
                    execute(
                        '''
                        INSERT INTO prop_lines 
                        (player, team, position, book, stat, line, over_odds, under_odds, last_updated, season)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(player, book, stat, season)
                        DO UPDATE SET
                            team = excluded.team,
                            position = excluded.position,
                            line = excluded.line,
                            over_odds = excluded.over_odds,
                            under_odds = excluded.under_odds,
                            last_updated = excluded.last_updated
                        ''',
                        (
                            prop_line.player,
                            prop_line.team,
                            prop_line.position,
                            prop_line.book,
                            prop_line.stat,
                            prop_line.line,
                            prop_line.over_odds,
                            prop_line.under_odds,
                            prop_line.last_updated,
                            prop_line.season,
                        ),
                        conn=conn,
                    )
                except Exception as e:
                    logger.error(f"Error saving prop line for {prop_line.player}: {e}")
                    continue
        
        logger.info(f"Saved {len(prop_lines)} prop lines to database")
    
    def get_prop_lines_dataframe(self) -> pd.DataFrame:
        """Get all prop lines as a DataFrame"""
        return read_dataframe(
            '''
            SELECT player, team, position, book, stat, line, over_odds, under_odds, last_updated, season
            FROM prop_lines
            ORDER BY player, stat, book
            '''
        )
    
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

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NFL weekly prop line scraper")
    parser.add_argument("--season", type=int, default=2025, help="Season to fetch (default: 2025)")
    parser.add_argument("--week", type=int, default=10, help="Week to fetch (default: 10)")
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="File path to save the raw odds snapshot",
    )
    parser.add_argument(
        "--load-json",
        type=Path,
        default=None,
        help="Fallback snapshot file to load if live calls fail",
    )
    return parser.parse_args()


def main():
    """CLI entry point for the prop line scraper."""
    args = _parse_args()
    scraper = NFLPropScraper()

    if scraper.odds_api_key:
        logger.info("Odds API key detected.")
    else:
        logger.warning(
            "No Odds API key found. Live odds calls will fall back to cached snapshots or synthetic data."
        )

    df = scraper.run_weekly_update(
        args.week,
        args.season,
        save_json_path=args.save_json,
        load_json_path=args.load_json,
    )

    logger.info("Weekly prop scrape complete: %d rows", len(df))
    if not df.empty:
        logger.info(
            "Players=%d, Books=%d, Markets=%s",
            df["player"].nunique(),
            df["book"].nunique(),
            ", ".join(df["stat"].unique()),
        )

if __name__ == "__main__":
    main()
