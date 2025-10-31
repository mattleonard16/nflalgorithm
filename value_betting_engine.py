"""
Enhanced value betting engine with multi-sportsbook comparison and CLV tracking.
"""

import json
import math
import sqlite3
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import config
from prop_integration import join_odds_projections

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.logs_dir / 'value_betting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds."""
    if odds >= 0:
        return (odds / 100) + 1
    return (100 / abs(odds)) + 1


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


def prob_over(mu: float, sigma: float, line: float) -> float:
    """Probability the true result clears the posted line under normal assumption."""
    if sigma <= 0:
        return float(mu > line)
    z_score = (mu - line) / sigma
    return float(0.5 * (1 + math.erf(z_score / math.sqrt(2))))


def kelly_fraction(p: float, price: int) -> float:
    """Full Kelly fraction without bankroll caps."""
    decimal_odds = _american_to_decimal(price)
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    p = min(max(p, 0.0), 1.0)
    q = 1 - p
    fraction = (b * p - q) / b
    return max(0.0, fraction)


def rank_weekly_value(
    season: int,
    week: int,
    min_edge: float,
    place: bool = False,
    bankroll: float = 1000.0
) -> pd.DataFrame:
    """Rank weekly value opportunities and optionally persist bets."""

    engine = ValueBettingEngine()
    joined = join_odds_projections(season, week)
    if joined.empty:
        logger.warning(
            "rank_weekly_value: no data to rank for season=%s week=%s",
            season,
            week
        )
        return pd.DataFrame()

    df = joined.copy()

    # Ensure season and week are present and not null
    if 'season' not in df.columns or 'week' not in df.columns:
        logger.error(
            "rank_weekly_value: joined data missing season/week columns for season=%s week=%s",
            season,
            week
        )
        return pd.DataFrame()
    
    # Fill any missing season/week values
    missing_season = df['season'].isna().sum()
    missing_week = df['week'].isna().sum()
    if missing_season > 0 or missing_week > 0:
        logger.warning(
            "rank_weekly_value: filling %d missing season and %d missing week values for season=%s week=%s",
            missing_season,
            missing_week,
            season,
            week
        )
    df['season'] = df['season'].fillna(season).astype(int)
    df['week'] = df['week'].fillna(week).astype(int)

    df['p_win'] = df.apply(lambda row: prob_over(row['mu'], row['sigma'], row['line']), axis=1)
    df['decimal_odds'] = df['price'].apply(_american_to_decimal)
    df['implied_prob'] = 1 / df['decimal_odds']
    df['edge_percentage'] = df['p_win'] - df['implied_prob']
    df['expected_roi'] = df['decimal_odds'] * df['p_win'] - 1
    df['kelly_fraction'] = df.apply(
        lambda row: min(
            engine.calculate_fractional_kelly(row['p_win'], int(row['price'])),
            engine.max_bankroll_fraction
        ),
        axis=1
    )
    df['stake'] = df['kelly_fraction'] * bankroll
    df['recommendation'] = np.where(df['edge_percentage'] >= min_edge, 'BET', 'PASS')
    df['generated_at'] = datetime.utcnow().isoformat()

    if place:
        bets = df[df['recommendation'] == 'BET']
        if not bets.empty:
            _persist_weekly_bets(engine, bets, bankroll)
            logger.info(
                "Placed %s weekly bets for season %s week %s",
                len(bets),
                season,
                week
            )

    df = df.drop(columns=['decimal_odds', 'implied_prob'])

    columns = [
        'season', 'week', 'player_id', 'player_name', 'team', 'opponent', 'position', 'market',
        'sportsbook', 'line', 'price', 'mu', 'sigma', 'p_win', 'edge_percentage', 'expected_roi',
        'kelly_fraction', 'stake', 'recommendation', 'model_version', 'as_of', 'event_id'
    ]

    return df.reindex(columns=columns + ['generated_at']).sort_values(
        ['edge_percentage', 'expected_roi'], ascending=False
    ).reset_index(drop=True)


def _persist_weekly_bets(engine: 'ValueBettingEngine', bets: pd.DataFrame, bankroll: float) -> None:
    payload = bets.copy()
    payload['bet_id'] = [str(uuid.uuid4()) for _ in range(len(payload))]
    payload['placed_at'] = datetime.utcnow().isoformat()

    sql = (
        """
        INSERT INTO bets_weekly (
            bet_id, season, week, event_id, player_id, market, sportsbook, side, line, price,
            p_win, kelly_fraction, stake, bankroll_before, placed_at, model_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(bet_id) DO NOTHING
        """
    )

    tuples = []
    for row in payload.itertuples():
        tuples.append((
            row.bet_id,
            int(row.season),
            int(row.week),
            row.event_id,
            row.player_id,
            row.market,
            row.sportsbook,
            'OVER',
            float(row.line),
            int(row.price),
            float(row.p_win),
            float(row.kelly_fraction),
            float(row.stake),
            float(bankroll),
            row.placed_at,
            row.model_version
        ))

    with sqlite3.connect(engine.db_path) as conn:
        conn.executemany(sql, tuples)
        conn.commit()

class ValueBettingEngine:
    """Enhanced value betting engine with professional features."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.database.path
        self.min_edge = config.betting.min_edge_threshold
        self.min_confidence = config.betting.min_confidence
        self.max_kelly = config.betting.max_kelly_fraction
        self.min_roi = config.betting.min_expected_roi
        self.max_bankroll_fraction = config.betting.max_bankroll_fraction
        self.daily_loss_stop = config.betting.daily_loss_stop
        self.per_market_unit_cap = config.betting.per_market_unit_cap
        
    def calculate_fractional_kelly(self, win_prob: float, odds: int, fraction: float | None = None) -> float:
        """Calculate fractional Kelly criterion for conservative betting."""
        fraction = fraction if fraction is not None else config.betting.kelly_fraction_default
        decimal_odds = _american_to_decimal(odds)
        b = decimal_odds - 1
        p = max(0.0, min(1.0, win_prob))
        q = 1 - p
        kelly_fraction = (b * p - q) / b
        if kelly_fraction <= 0:
            return 0.0
        fractional_kelly = kelly_fraction * fraction
        return min(fractional_kelly, self.max_kelly)

    def _fair_price_probability(self, projection: float, line: float, sigma: float) -> float:
        if sigma <= 0:
            return float(projection >= line)
        z_score = (projection - line) / sigma
        return float(0.5 * (1 + math.erf(z_score / math.sqrt(2))))
    
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
        """Find value bets using EV, calibrated probability, and bankroll rules."""
        if predictions_df is None or predictions_df.empty:
            return []

        required_columns = ['player_id', 'player_name', 'position', 'team']
        for column in required_columns:
            if column not in predictions_df.columns:
                raise ValueError(f"Predictions missing required column: {column}")

        bets: List[ValueBet] = []
        stats_map = self._load_latest_player_stats()
        for _, row in predictions_df.iterrows():
            projection = float(row.get('projection', 0.0))
            line = float(row.get('line', projection))
            odds = int(row.get('odds', -110))
            sigma = float(row.get('sigma', 5))
            win_probability = self._fair_price_probability(projection, line, sigma)
            kelly_fraction = self.calculate_fractional_kelly(win_probability, odds, fraction=config.betting.kelly_fraction_default)
            fractional_kelly = min(kelly_fraction, self.max_bankroll_fraction)
            expected_roi = self._calculate_expected_roi(win_probability, odds)

            if expected_roi < self.min_roi or fractional_kelly <= 0:
                continue

            value_bet = ValueBet(
                bet_id=str(uuid.uuid4()),
                player_id=row['player_id'],
                player_name=row['player_name'],
                position=row['position'],
                team=row['team'],
                prop_type=row.get('market', 'unknown_market'),
                sportsbook=row.get('sportsbook', 'Unknown'),
                line=line,
                best_line=line,
                over_odds=odds,
                under_odds=0,
                model_prediction=projection,
                prediction_lower=float(row.get('prediction_lower', projection - sigma)),
                prediction_upper=float(row.get('prediction_upper', projection + sigma)),
                model_confidence=float(win_probability),
                edge_yards=float(projection - line),
                edge_percentage=float(max(abs(projection - line) / max(line, 1e-6), 0)),
                kelly_fraction=float(kelly_fraction),
                fractional_kelly=float(fractional_kelly),
                expected_roi=float(expected_roi),
                risk_level='PENDING',
                recommendation='OVER' if row['projection'] > row['line'] else 'UNDER',
                bet_size_units=float(min(fractional_kelly * 100, self.per_market_unit_cap)),
                correlation_risk='PENDING',
                market_efficiency=0.0,
                timestamp=datetime.now()
            )

            if stats_map:
                player_row = stats_map.get(row['player_id'])
                if player_row is not None:
                    breakout_percentile = self.calculate_breakout_percentile(player_row)
                    value_bet.edge_percentage *= breakout_percentile

            bets.append(value_bet)

        correlation_map = self.assess_correlation_risk(bets)
        for bet in bets:
            bet.correlation_risk = correlation_map.get(bet.bet_id, 'LOW')
            bet.risk_level = self._assess_risk_level(bet.edge_percentage, bet.model_confidence, bet.kelly_fraction)
        bets.sort(key=lambda bet: bet.edge_percentage, reverse=True)
        return bets

    def _load_latest_player_stats(self) -> Dict[str, pd.Series]:
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT *
                FROM player_stats_enhanced
                ORDER BY player_id, season DESC, week DESC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            if df.empty:
                return {}
            latest = df.sort_values(['player_id', 'season', 'week']).groupby('player_id').tail(1)
            return {row['player_id']: row for _, row in latest.iterrows()}
        except Exception:
            return {}
    
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
