"""
📊 Record Bet Outcomes - Tracks actual results and calculates P&L

This script grades bets from the materialized view against actual player stats
and updates the bet_outcomes and weekly_performance tables.

Usage:
    python scripts/record_outcomes.py --season 2025 --week 12
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from utils.db import execute, executemany, read_dataframe

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


MARKET_TO_STAT = {
    'rushing_yards': 'rushing_yards',
    'receiving_yards': 'receiving_yards',
}


def _american_to_decimal(odds: int) -> float:
    if odds >= 0:
        return (odds / 100) + 1
    return (100 / abs(odds)) + 1


def calculate_profit(price: int, stake: float, won: bool) -> float:
    """Calculate profit in units (1 unit = base stake)."""
    if not won:
        return -stake
    decimal = _american_to_decimal(price)
    return stake * (decimal - 1)


def get_confidence_tier(edge_pct: float) -> str:
    """Assign confidence tier based on edge percentage."""
    if edge_pct >= 0.15:
        return "💎 Premium"
    elif edge_pct >= 0.10:
        return "🔥 Strong"
    elif edge_pct >= 0.05:
        return "⭐ Standard"
    else:
        return "📉 Low"


def grade_bets(season: int, week: int) -> pd.DataFrame:
    """Grade all bets for a given week by comparing predictions to actuals."""
    
    logger.info(f"Grading bets for Season {season} Week {week}...")
    
    # Only select best line (highest edge) per player/market combination
    bets_query = """
        WITH ranked_bets AS (
            SELECT 
                mv.season, mv.week, mv.player_id, mv.market, mv.sportsbook,
                mv.line, mv.price, mv.mu, mv.edge_percentage, mv.stake,
                ps.name as player_name,
                ROW_NUMBER() OVER (PARTITION BY mv.player_id, mv.market ORDER BY mv.edge_percentage DESC) as rn
            FROM materialized_value_view mv
            LEFT JOIN player_stats_enhanced ps 
                ON ps.player_id = mv.player_id 
                AND ps.season = mv.season 
                AND ps.week = mv.week
            WHERE mv.season = ? AND mv.week = ? AND mv.edge_percentage >= 0.05
        )
        SELECT season, week, player_id, market, sportsbook, line, price, mu, edge_percentage, stake, player_name
        FROM ranked_bets
        WHERE rn = 1
    """
    bets = read_dataframe(bets_query, params=(season, week))
    
    if bets.empty:
        logger.warning("No bets found to grade")
        return pd.DataFrame()
    
    actuals_query = """
        SELECT player_id, rushing_yards, receiving_yards
        FROM player_stats_enhanced
        WHERE season = ? AND week = ?
    """
    actuals = read_dataframe(actuals_query, params=(season, week))
    
    if actuals.empty:
        logger.warning("No actual stats found for this week")
        return pd.DataFrame()
    
    actuals_dict = actuals.set_index('player_id').to_dict('index')
    
    results = []
    for _, bet in bets.iterrows():
        player_stats = actuals_dict.get(bet['player_id'], {})
        stat_col = MARKET_TO_STAT.get(bet['market'])
        
        if not stat_col or stat_col not in player_stats:
            continue
        
        actual = player_stats[stat_col]
        line = bet['line']
        
        if actual > line:
            result = 'WIN'
            won = True
        elif actual < line:
            result = 'LOSS'
            won = False
        else:
            result = 'PUSH'
            won = None
        
        stake = bet.get('stake', 1.0) or 1.0
        profit = calculate_profit(bet['price'], stake, won) if won is not None else 0.0
        
        results.append({
            'bet_id': f"{bet['player_id']}_{bet['market']}_{bet['sportsbook']}_{season}_{week}",
            'season': season,
            'week': week,
            'player_id': bet['player_id'],
            'player_name': bet.get('player_name') or bet['player_id'],
            'market': bet['market'],
            'sportsbook': bet['sportsbook'],
            'side': 'OVER',
            'line': line,
            'price': bet['price'],
            'actual_result': actual,
            'result': result,
            'profit_units': profit,
            'confidence_tier': get_confidence_tier(bet['edge_percentage']),
            'edge_at_placement': bet['edge_percentage'],
            'recorded_at': datetime.utcnow().isoformat(),
        })
    
    return pd.DataFrame(results)


def save_outcomes(outcomes: pd.DataFrame) -> int:
    """Save graded outcomes to bet_outcomes table."""
    if outcomes.empty:
        return 0
    
    sql = """
        INSERT OR REPLACE INTO bet_outcomes 
        (bet_id, season, week, player_id, player_name, market, sportsbook, 
         side, line, price, actual_result, result, profit_units, 
         confidence_tier, edge_at_placement, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    tuples = [
        (
            row['bet_id'], row['season'], row['week'], row['player_id'],
            row['player_name'], row['market'], row['sportsbook'], row['side'],
            row['line'], row['price'], row['actual_result'], row['result'],
            row['profit_units'], row['confidence_tier'], row['edge_at_placement'],
            row['recorded_at']
        )
        for _, row in outcomes.iterrows()
    ]
    
    executemany(sql, tuples)
    return len(tuples)


def update_weekly_performance(season: int, week: int) -> dict:
    """Aggregate weekly performance from bet outcomes."""
    
    query = """
        SELECT result, profit_units, edge_at_placement, player_name, market
        FROM bet_outcomes
        WHERE season = ? AND week = ?
    """
    outcomes = read_dataframe(query, params=(season, week))
    
    if outcomes.empty:
        return {}
    
    wins = (outcomes['result'] == 'WIN').sum()
    losses = (outcomes['result'] == 'LOSS').sum()
    pushes = (outcomes['result'] == 'PUSH').sum()
    total = len(outcomes)
    profit = outcomes['profit_units'].sum()
    
    total_staked = total
    roi = (profit / total_staked * 100) if total_staked > 0 else 0
    avg_edge = outcomes['edge_at_placement'].mean() * 100
    
    best_idx = outcomes['profit_units'].idxmax() if not outcomes.empty else None
    worst_idx = outcomes['profit_units'].idxmin() if not outcomes.empty else None
    
    best_bet = f"{outcomes.loc[best_idx, 'player_name']} {outcomes.loc[best_idx, 'market']}" if best_idx is not None else None
    worst_bet = f"{outcomes.loc[worst_idx, 'player_name']} {outcomes.loc[worst_idx, 'market']}" if worst_idx is not None else None
    
    sql = """
        INSERT OR REPLACE INTO weekly_performance
        (season, week, total_bets, wins, losses, pushes, profit_units, 
         roi_pct, avg_edge, clv_avg, best_bet, worst_bet, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    execute(sql, (
        season, week, int(total), int(wins), int(losses), int(pushes), float(profit), float(roi), float(avg_edge),
        0.0, best_bet, worst_bet, datetime.utcnow().isoformat()
    ))
    
    return {
        'total_bets': total,
        'record': f"{wins}-{losses}-{pushes}",
        'profit_units': profit,
        'roi_pct': roi,
        'avg_edge': avg_edge,
    }


def main():
    parser = argparse.ArgumentParser(description='Record bet outcomes for a week')
    parser.add_argument('--season', type=int, required=True)
    parser.add_argument('--week', type=int, required=True)
    args = parser.parse_args()
    
    outcomes = grade_bets(args.season, args.week)
    
    if outcomes.empty:
        logger.info("No outcomes to record")
        return
    
    saved = save_outcomes(outcomes)
    logger.info(f"Saved {saved} bet outcomes")
    
    perf = update_weekly_performance(args.season, args.week)
    
    print("\n" + "="*50)
    print(f"📊 WEEK {args.week} PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Total Bets:  {perf.get('total_bets', 0)}")
    print(f"Record:      {perf.get('record', '0-0-0')}")
    print(f"Profit:      {perf.get('profit_units', 0):+.2f} units")
    print(f"ROI:         {perf.get('roi_pct', 0):+.1f}%")
    print(f"Avg Edge:    {perf.get('avg_edge', 0):.1f}%")
    print("="*50)


if __name__ == '__main__':
    main()

