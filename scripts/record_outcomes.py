"""
Record Bet Outcomes - Tracks actual results and calculates P&L

This script grades bets from the materialized view against actual player stats
and updates the bet_outcomes and weekly_performance tables.

Usage:
    python scripts/record_outcomes.py --season 2025 --week 12
"""

import argparse
import uuid
from datetime import datetime, timezone
from typing import List, Dict

import pandas as pd

from config import config
from utils.db import read_dataframe, execute, executemany, get_connection


# Market to stat column mapping
MARKET_TO_STAT = {
    'rushing_yards': 'rushing_yards',
    'receiving_yards': 'receiving_yards',
    'passing_yards': 'passing_yards',
    'receptions': 'receptions',
    'targets': 'targets',
}


def get_confidence_tier(edge_percentage: float) -> str:
    """
    Determine confidence tier based on edge percentage.

    Args:
        edge_percentage: Edge percentage at time of placement

    Returns:
        Confidence tier: HIGH, MEDIUM, LOW, or MINIMAL
    """
    if edge_percentage >= 15.0:
        return 'HIGH'
    elif edge_percentage >= 8.0:
        return 'MEDIUM'
    elif edge_percentage >= 3.0:
        return 'LOW'
    else:
        return 'MINIMAL'


def calculate_profit_units(result: str, price: int) -> float:
    """
    Calculate profit in units for a bet.

    Args:
        result: Bet result ('win', 'loss', or 'push')
        price: American odds (e.g., -110, +150)

    Returns:
        Profit in units (1 unit = stake)
    """
    if result == 'push':
        return 0.0
    elif result == 'loss':
        return -1.0
    elif result == 'win':
        if price < 0:
            # Negative odds: profit = 100 / abs(odds)
            return 100.0 / abs(price)
        else:
            # Positive odds: profit = odds / 100
            return price / 100.0
    else:
        return 0.0


def grade_bet(actual: float, line: float, side: str) -> str:
    """
    Grade a single bet based on actual result vs line.

    Args:
        actual: Actual stat value
        line: Bet line
        side: Bet side ('over' or 'under')

    Returns:
        Result: 'win', 'loss', or 'push'
    """
    if pd.isna(actual):
        # No actual data - treat as push
        return 'push'

    if actual == line:
        return 'push'

    if side.lower() == 'over':
        return 'win' if actual > line else 'loss'
    else:  # under
        return 'win' if actual < line else 'loss'


def grade_bets(season: int, week: int) -> List[Dict]:
    """
    Compare predictions to actuals for a given week.

    Args:
        season: NFL season year
        week: NFL week number

    Returns:
        List of outcome dictionaries with bet results
    """
    print(f"Grading bets for {season} Week {week}...")

    # Load predictions/bets from materialized view
    predictions_query = """
        SELECT
            season, week, player_id, event_id, team, team_odds,
            market, sportsbook, line, price, mu, sigma, p_win,
            edge_percentage, expected_roi, kelly_fraction, stake,
            generated_at
        FROM materialized_value_view
        WHERE season = ? AND week = ?
    """
    predictions = read_dataframe(predictions_query, params=(season, week))

    if predictions.empty:
        print(f"No predictions found for {season} Week {week}")
        return []

    print(f"Found {len(predictions)} predictions")

    # Load actual stats
    actuals_query = """
        SELECT
            player_id, season, week, name, team, position,
            rushing_yards, receiving_yards, passing_yards,
            receptions, targets
        FROM player_stats_enhanced
        WHERE season = ? AND week = ?
    """
    actuals = read_dataframe(actuals_query, params=(season, week))

    if actuals.empty:
        print(f"WARNING: No actual stats found for {season} Week {week}")
        print("All bets will be marked as pushes")
    else:
        print(f"Found actual stats for {len(actuals)} players")

    # Grade each bet
    outcomes = []

    for _, pred in predictions.iterrows():
        player_id = pred['player_id']
        market = pred['market']
        line = pred['line']
        price = pred['price']
        edge_pct = pred['edge_percentage']

        # Default to "over" if side not in materialized view
        # (this script assumes all bets are "over" unless schema includes side column)
        side = 'over'

        # Get stat column for this market
        stat_column = MARKET_TO_STAT.get(market)

        if not stat_column:
            print(f"WARNING: Unknown market '{market}' for player {player_id}")
            continue

        # Find actual result
        player_actuals = actuals[actuals['player_id'] == player_id]

        if player_actuals.empty:
            # No actual data - treat as push
            actual_result = None
            result = 'push'
        else:
            player_row = player_actuals.iloc[0]
            actual_result = player_row.get(stat_column)

            if pd.isna(actual_result):
                result = 'push'
            else:
                result = grade_bet(actual_result, line, side)

        # Calculate profit
        profit_units = calculate_profit_units(result, price)

        # Determine confidence tier
        confidence_tier = get_confidence_tier(edge_pct)

        # Create outcome record
        outcome = {
            'bet_id': str(uuid.uuid4()),
            'season': season,
            'week': week,
            'player_id': player_id,
            'player_name': actuals[actuals['player_id'] == player_id]['name'].iloc[0]
                          if not actuals[actuals['player_id'] == player_id].empty
                          else 'Unknown',
            'market': market,
            'sportsbook': pred['sportsbook'],
            'side': side,
            'line': line,
            'price': price,
            'actual_result': actual_result if not pd.isna(actual_result) else None,
            'result': result,
            'profit_units': profit_units,
            'confidence_tier': confidence_tier,
            'edge_at_placement': edge_pct,
            'recorded_at': datetime.now(timezone.utc).isoformat(),
        }

        outcomes.append(outcome)

    print(f"Graded {len(outcomes)} bets")

    # Print summary
    if outcomes:
        wins = sum(1 for o in outcomes if o['result'] == 'win')
        losses = sum(1 for o in outcomes if o['result'] == 'loss')
        pushes = sum(1 for o in outcomes if o['result'] == 'push')
        total_profit = sum(o['profit_units'] for o in outcomes)

        print(f"Results: {wins} wins, {losses} losses, {pushes} pushes")
        print(f"Total profit: {total_profit:.2f} units")

    return outcomes


def save_outcomes(outcomes: List[Dict]) -> None:
    """
    Persist outcomes to bet_outcomes table and update weekly_performance.

    Args:
        outcomes: List of outcome dictionaries from grade_bets()
    """
    if not outcomes:
        print("No outcomes to save")
        return

    print(f"Saving {len(outcomes)} outcomes to database...")

    # Insert into bet_outcomes
    insert_sql = """
        INSERT OR REPLACE INTO bet_outcomes (
            bet_id, season, week, player_id, player_name, market,
            sportsbook, side, line, price, actual_result, result,
            profit_units, confidence_tier, edge_at_placement, recorded_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """

    outcome_tuples = [
        (
            o['bet_id'], o['season'], o['week'], o['player_id'],
            o['player_name'], o['market'], o['sportsbook'], o['side'],
            o['line'], o['price'], o['actual_result'], o['result'],
            o['profit_units'], o['confidence_tier'], o['edge_at_placement'],
            o['recorded_at']
        )
        for o in outcomes
    ]

    executemany(insert_sql, outcome_tuples)
    print(f"Inserted {len(outcome_tuples)} outcomes into bet_outcomes")

    # Aggregate weekly performance
    df = pd.DataFrame(outcomes)
    season = df['season'].iloc[0]
    week = df['week'].iloc[0]

    total_bets = len(outcomes)
    wins = len(df[df['result'] == 'win'])
    losses = len(df[df['result'] == 'loss'])
    pushes = len(df[df['result'] == 'push'])
    profit_units = df['profit_units'].sum()

    # ROI calculation: profit / units risked (excluding pushes)
    units_risked = wins + losses  # Each bet risks 1 unit
    roi_pct = (profit_units / units_risked * 100) if units_risked > 0 else 0.0

    avg_edge = df['edge_at_placement'].mean()

    # Best/worst bets (by profit)
    best_bet_row = df.loc[df['profit_units'].idxmax()] if not df.empty else None
    worst_bet_row = df.loc[df['profit_units'].idxmin()] if not df.empty else None

    best_bet = f"{best_bet_row['player_name']} {best_bet_row['market']} {best_bet_row['side']} {best_bet_row['line']}" if best_bet_row is not None else None
    worst_bet = f"{worst_bet_row['player_name']} {worst_bet_row['market']} {worst_bet_row['side']} {worst_bet_row['line']}" if worst_bet_row is not None else None

    # CLV (Closing Line Value) - placeholder for now
    clv_avg = None

    # Update weekly_performance
    perf_sql = """
        INSERT OR REPLACE INTO weekly_performance (
            season, week, total_bets, wins, losses, pushes,
            profit_units, roi_pct, avg_edge, clv_avg,
            best_bet, worst_bet, updated_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """

    execute(
        perf_sql,
        (
            season, week, total_bets, wins, losses, pushes,
            profit_units, roi_pct, avg_edge, clv_avg,
            best_bet, worst_bet,
            datetime.now(timezone.utc).isoformat()
        )
    )

    print(f"Updated weekly_performance for {season} Week {week}")
    print(f"  Total bets: {total_bets}")
    print(f"  Record: {wins}-{losses}-{pushes}")
    print(f"  Profit: {profit_units:.2f} units")
    print(f"  ROI: {roi_pct:.2f}%")
    print(f"  Avg edge: {avg_edge:.2f}%")


def main():
    """CLI entry point for recording bet outcomes."""
    parser = argparse.ArgumentParser(
        description='Record bet outcomes by comparing predictions to actuals'
    )
    parser.add_argument(
        '--season',
        type=int,
        required=True,
        help='NFL season year (e.g., 2025)'
    )
    parser.add_argument(
        '--week',
        type=int,
        required=True,
        help='NFL week number (e.g., 13)'
    )

    args = parser.parse_args()

    try:
        # Grade bets
        outcomes = grade_bets(args.season, args.week)

        # Save outcomes
        if outcomes:
            save_outcomes(outcomes)
            print(f"\nSuccessfully recorded outcomes for {args.season} Week {args.week}")
        else:
            print(f"\nNo outcomes to record for {args.season} Week {args.week}")

    except Exception as e:
        print(f"Error recording outcomes: {e}")
        raise


if __name__ == '__main__':
    main()

