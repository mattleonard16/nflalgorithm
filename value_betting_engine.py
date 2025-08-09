"""
Enhanced value betting engine with multi-sportsbook comparison and CLV tracking.
"""

import sqlite3
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

import pandas as pd
import numpy as np
from scipy import stats

from config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.logs_dir / 'value_betting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ValueBet:
    """Enhanced value bet with CLV tracking."""
    bet_id: str
    player_id: str
    player_name: str
    position: str
    team: str
    prop_type: str
    sportsbook: str
    line: float
    best_line: float
    over_odds: int
    under_odds: int
    model_prediction: float
    prediction_lower: float
    prediction_upper: float
    model_confidence: float
    edge_yards: float
    edge_percentage: float
    kelly_fraction: float
    fractional_kelly: float
    expected_roi: float
    risk_level: str
    recommendation: str
    bet_size_units: float
    correlation_risk: str
    market_efficiency: float
    timestamp: datetime
    
@dataclass
class CLVResult:
    """Closing Line Value tracking result."""
    bet_id: str
    opening_line: float
    closing_line: float
    clv_percentage: float
    market_move_direction: str
    sharp_money_indicator: bool

class ValueBettingEngine:
    """Enhanced value betting engine with professional features."""
    
    def __init__(self):
        self.db_path = config.database.path
        self.min_edge = config.betting.min_edge_threshold
        self.min_confidence = config.betting.min_confidence
        self.max_kelly = config.betting.max_kelly_fraction
        self.min_roi = config.betting.min_expected_roi
        
    def calculate_fractional_kelly(self, win_prob: float, odds: int, fraction: float = 0.5) -> float:
        """Calculate fractional Kelly criterion for conservative betting."""
        # Convert American odds to decimal
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1
        
        # Kelly formula: f = (bp - q) / b
        b = decimal_odds - 1
        p = win_prob
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Apply fractional Kelly
        fractional_kelly = kelly_fraction * fraction
        
        # Cap at maximum
        return min(fractional_kelly, self.max_kelly)
    
    def assess_correlation_risk(self, bets: List[ValueBet]) -> Dict[str, str]:
        """Assess correlation risk between potential bets."""
        correlation_risks = {}
        
        # Group bets by team/game
        team_bets = {}
        game_bets = {}
        
        for bet in bets:
            # Team correlation
            if bet.team not in team_bets:
                team_bets[bet.team] = []
            team_bets[bet.team].append(bet)
            
            # Game correlation (simplified - would need game mapping)
            game_key = f"{bet.team}_{datetime.now().strftime('%Y-%m-%d')}"
            if game_key not in game_bets:
                game_bets[game_key] = []
            game_bets[game_key].append(bet)
        
        for bet in bets:
            risk_level = "LOW"
            
            # Check team correlation
            team_bet_count = len(team_bets[bet.team])
            if team_bet_count > 2:
                risk_level = "HIGH"
            elif team_bet_count > 1:
                risk_level = "MEDIUM"
            
            # Check prop correlation (same player, different props)
            same_player_bets = [b for b in bets if b.player_id == bet.player_id and b.bet_id != bet.bet_id]
            if same_player_bets:
                risk_level = "HIGH"
            
            correlation_risks[bet.bet_id] = risk_level
        
        return correlation_risks
    
    def calculate_market_efficiency(self, player_odds: List[Dict]) -> float:
        """Calculate market efficiency based on line variance across books."""
        if len(player_odds) < 2:
            return 0.5  # Default efficiency
        
        lines = [odd['line'] for odd in player_odds]
        line_variance = np.var(lines)
        
        # Normalize variance to efficiency score (0-1)
        # Higher variance = lower efficiency
        efficiency = 1 / (1 + line_variance)
        
        return min(efficiency, 1.0)
    
    def calculate_breakout_percentile(self, player_stats: pd.Series) -> float:
        """Calculate breakout percentile based on features."""
        # Simplified: weighted sum of breakout signals
        score = (player_stats['usage_delta'] * 0.3 +
                 player_stats['preseason_buzz'] * 0.2 +
                 (1 if player_stats['oc_change'] else 0) * 0.15 +
                 (1 if player_stats['injury_recovery'] else 0) * 0.15 +
                 (1 - abs(player_stats['age'] - 26) / 10) * 0.2)  # Peak age 26
        
        # Normalize to percentile (0-1)
        percentile = np.clip(score / 1.0, 0, 1)  # Assuming max score ~1
        
        return percentile
    
    def find_value_opportunities(self, predictions_df: pd.DataFrame) -> List[ValueBet]:
        """Find value opportunities with breakout re-ranking."""
        value_bets = super().find_value_opportunities(predictions_df)
        
        # Load player stats for breakout calc
        conn = sqlite3.connect(self.db_path)
        stats_df = pd.read_sql("SELECT * FROM player_stats_enhanced", conn)
        conn.close()
        
        # Calculate breakout percentile for each
        for bet in value_bets:
            player_stats = stats_df[stats_df['player_id'] == bet.player_id].iloc[-1]
            breakout_perc = self.calculate_breakout_percentile(player_stats)
            
            # Re-rank by breakout × edge
            bet.edge_percentage *= breakout_perc
        
        # Sort by new adjusted edge
        value_bets.sort(key=lambda b: b.edge_percentage, reverse=True)
        
        logger.info(f"Found {len(value_bets)} value opportunities after breakout re-ranking")
        return value_bets
    
    def _analyze_single_opportunity(self, prediction: pd.Series, line_data: pd.Series) -> Optional[ValueBet]:
        """Analyze a single betting opportunity."""
        
        model_pred = prediction.get('prediction', 0)
        pred_lower = prediction.get('prediction_lower', model_pred - 5)
        pred_upper = prediction.get('prediction_upper', model_pred + 5)
        confidence = prediction.get('confidence', 0.75)
        
        line = line_data['best_line']
        direction = line_data['direction']
        odds = line_data['best_odds']
        sportsbook = line_data['best_sportsbook']
        all_books = line_data['all_books']
        
        # Calculate edge
        if direction == 'over':
            edge_yards = model_pred - line
            win_prob = confidence if edge_yards > 0 else 1 - confidence
        else:
            edge_yards = line - model_pred
            win_prob = confidence if edge_yards > 0 else 1 - confidence
        
        edge_percentage = abs(edge_yards) / line if line > 0 else 0
        
        # Filter based on minimum thresholds
        if (abs(edge_yards) < 5.0 or  # Minimum 5 yard edge
            edge_percentage < self.min_edge or
            win_prob < self.min_confidence):
            return None
        
        # Calculate Kelly and ROI
        kelly_fraction = self.calculate_fractional_kelly(win_prob, odds, 1.0)
        fractional_kelly = self.calculate_fractional_kelly(win_prob, odds, 0.5)
        
        expected_roi = self._calculate_expected_roi(win_prob, odds)
        
        if expected_roi < self.min_roi:
            return None
        
        # Assess risk level
        risk_level = self._assess_risk_level(edge_percentage, confidence, kelly_fraction)
        
        # Calculate market efficiency
        market_efficiency = self.calculate_market_efficiency(all_books)
        
        # Create value bet
        bet_id = str(uuid.uuid4())
        
        return ValueBet(
            bet_id=bet_id,
            player_id=prediction.get('player_id', ''),
            player_name=prediction.get('player_name', ''),
            position=prediction.get('position', ''),
            team=prediction.get('team', ''),
            prop_type=line_data['prop_type'],
            sportsbook=sportsbook,
            line=line,
            best_line=line,  # Same for now
            over_odds=odds if direction == 'over' else 0,
            under_odds=odds if direction == 'under' else 0,
            model_prediction=model_pred,
            prediction_lower=pred_lower,
            prediction_upper=pred_upper,
            model_confidence=confidence,
            edge_yards=abs(edge_yards),
            edge_percentage=edge_percentage,
            kelly_fraction=kelly_fraction,
            fractional_kelly=fractional_kelly,
            expected_roi=expected_roi,
            risk_level=risk_level,
            recommendation=direction.upper(),
            bet_size_units=fractional_kelly * 100,
            correlation_risk="PENDING",  # Will be set later
            market_efficiency=market_efficiency,
            timestamp=datetime.now()
        )
    
    def _calculate_expected_roi(self, win_prob: float, odds: int) -> float:
        """Calculate expected return on investment."""
        if odds > 0:
            win_payout = odds / 100
        else:
            win_payout = 100 / abs(odds)
        
        expected_roi = (win_prob * win_payout) - (1 - win_prob)
        return expected_roi
    
    def _assess_risk_level(self, edge_percentage: float, confidence: float, kelly_fraction: float) -> str:
        """Assess risk level of the bet."""
        if edge_percentage >= 0.15 and confidence >= 0.85 and kelly_fraction >= 0.15:
            return "HIGH_VALUE"
        elif edge_percentage >= 0.10 and confidence >= 0.80 and kelly_fraction >= 0.10:
            return "MEDIUM_VALUE"
        elif edge_percentage >= 0.05 and confidence >= 0.75 and kelly_fraction >= 0.05:
            return "LOW_VALUE"
        else:
            return "NO_VALUE"
    
    def track_closing_line_value(self, bet_id: str, closing_odds: Dict) -> CLVResult:
        """Track closing line value for placed bets."""
        conn = sqlite3.connect(self.db_path)
        
        # Get original bet details
        cursor = conn.execute(
            "SELECT bet_line, sportsbook, prop_type FROM clv_tracking WHERE bet_id = ?",
            (bet_id,)
        )
        bet_data = cursor.fetchone()
        
        if not bet_data:
            logger.warning(f"Bet {bet_id} not found in CLV tracking")
            return None
        
        opening_line = bet_data[0]
        closing_line = closing_odds.get('line', opening_line)
        
        # Calculate CLV
        clv_percentage = (closing_line - opening_line) / opening_line * 100
        
        # Determine market move direction
        if abs(clv_percentage) < 1:
            move_direction = "STABLE"
        elif clv_percentage > 0:
            move_direction = "FAVORABLE"
        else:
            move_direction = "UNFAVORABLE"
        
        # Sharp money indicator (simplified)
        sharp_money = abs(clv_percentage) > 3  # 3%+ move indicates sharp action
        
        # Update CLV tracking
        conn.execute('''
            UPDATE clv_tracking 
            SET closing_line = ?, clv_percentage = ?
            WHERE bet_id = ?
        ''', (closing_line, clv_percentage, bet_id))
        
        conn.commit()
        conn.close()
        
        return CLVResult(
            bet_id=bet_id,
            opening_line=opening_line,
            closing_line=closing_line,
            clv_percentage=clv_percentage,
            market_move_direction=move_direction,
            sharp_money_indicator=sharp_money
        )
    
    def save_value_bets(self, value_bets: List[ValueBet]) -> None:
        """Save value bets to database with CLV tracking."""
        if not value_bets:
            return
        
        conn = sqlite3.connect(self.db_path)
        
        for bet in value_bets:
            # Save to enhanced value bets table
            conn.execute('''
                INSERT OR REPLACE INTO enhanced_value_bets 
                (bet_id, player_name, position, team, prop_type, sportsbook, line, 
                 model_prediction, model_confidence, edge_yards, edge_percentage,
                 kelly_fraction, expected_roi, risk_level, recommendation, bet_size_units,
                 correlation_risk, market_efficiency, date_identified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                bet.bet_id, bet.player_name, bet.position, bet.team, bet.prop_type,
                bet.sportsbook, bet.line, bet.model_prediction, bet.model_confidence,
                bet.edge_yards, bet.edge_percentage, bet.kelly_fraction, bet.expected_roi,
                bet.risk_level, bet.recommendation, bet.bet_size_units,
                bet.correlation_risk, bet.market_efficiency, bet.timestamp
            ))
            
            # Initialize CLV tracking
            conn.execute('''
                INSERT OR REPLACE INTO clv_tracking
                (bet_id, player_id, prop_type, sportsbook, bet_line, date_placed)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                bet.bet_id, bet.player_id, bet.prop_type, bet.sportsbook,
                bet.line, bet.timestamp.date()
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(value_bets)} value bets to database")
    
    def generate_clv_report(self) -> pd.DataFrame:
        """Generate closing line value report."""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
        SELECT 
            bet_id,
            player_id,
            prop_type,
            sportsbook,
            bet_line,
            closing_line,
            clv_percentage,
            bet_result,
            roi,
            date_placed,
            date_settled
        FROM clv_tracking
        WHERE clv_percentage IS NOT NULL
        ORDER BY date_placed DESC
        '''
        
        clv_df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not clv_df.empty:
            # Add summary statistics
            avg_clv = clv_df['clv_percentage'].mean()
            positive_clv_rate = (clv_df['clv_percentage'] > 0).mean()
            
            logger.info(f"CLV Report: Avg CLV = {avg_clv:.2f}%, Positive CLV Rate = {positive_clv_rate:.1%}")
            
            # Save to CSV
            clv_df.to_csv(config.logs_dir / 'clv_report.csv', index=False)
        
        return clv_df 