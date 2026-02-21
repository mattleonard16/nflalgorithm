"""Record NBA Bet Outcomes - Tracks actual results and calculates P&L.

Grades bets from the NBA materialized value view against actual player game logs
and updates the nba_bet_outcomes and nba_daily_performance tables.

Usage:
    python scripts/record_nba_outcomes.py --game-date 2026-02-17
"""

import argparse
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.db import execute, executemany, read_dataframe
from utils.grading import calculate_profit_units, get_confidence_tier, grade_bet

logger = logging.getLogger(__name__)

# NBA market to game log column mapping (direct 1:1)
NBA_MARKET_TO_STAT = {
    "pts": "pts",
    "reb": "reb",
    "ast": "ast",
    "fg3m": "fg3m",
}

# All NBA value bets target the over side (over_price is used for Kelly/value calc)
NBA_DEFAULT_SIDE = "over"


def grade_nba_bets(game_date: str) -> List[Dict]:
    """Compare NBA predictions to actuals for a given game date.

    Args:
        game_date: ISO date string (e.g., '2026-02-17')

    Returns:
        List of outcome dictionaries with bet results
    """
    logger.info("Grading NBA bets for %s...", game_date)

    predictions = read_dataframe(
        """
        SELECT
            season, game_date, player_id, player_name, event_id, team,
            market, sportsbook, line, over_price, mu, sigma, p_win,
            edge_percentage, expected_roi, kelly_fraction, confidence,
            generated_at
        FROM nba_materialized_value_view
        WHERE game_date = ?
        """,
        params=(game_date,),
    )

    if predictions.empty:
        logger.info("No predictions found for %s", game_date)
        return []

    logger.info("Found %d predictions", len(predictions))

    actuals = read_dataframe(
        """
        SELECT
            player_id, player_name, game_date, pts, reb, ast, fg3m
        FROM nba_player_game_logs
        WHERE game_date = ?
        """,
        params=(game_date,),
    )

    if actuals.empty:
        logger.warning("No actual stats found for %s - all bets will be marked as pushes", game_date)
    else:
        logger.info("Found actual stats for %d players", len(actuals))

    outcomes: List[Dict] = []

    for _, pred in predictions.iterrows():
        player_id = pred["player_id"]
        market = pred["market"]
        line = pred["line"]
        price = int(pred["over_price"])
        edge_pct = pred["edge_percentage"]
        season = pred["season"]

        # Side is always "over" - the value engine uses over_price for Kelly criterion
        side = NBA_DEFAULT_SIDE

        stat_column = NBA_MARKET_TO_STAT.get(market)
        if not stat_column:
            logger.warning("Unknown market '%s' for player %s", market, player_id)
            continue

        player_actuals = actuals[actuals["player_id"] == player_id]

        if player_actuals.empty:
            actual_result = None
            result = "push"
        else:
            player_row = player_actuals.iloc[0]
            actual_result = player_row.get(stat_column)

            if pd.isna(actual_result):
                result = "push"
            else:
                actual_result = float(actual_result)
                result = grade_bet(actual_result, line, side)

        profit_units = calculate_profit_units(result, price)
        confidence_tier = get_confidence_tier(edge_pct)

        player_name = pred["player_name"]
        if player_name is None or (isinstance(player_name, float) and pd.isna(player_name)):
            if not player_actuals.empty:
                player_name = player_actuals.iloc[0]["player_name"]
            else:
                player_name = "Unknown"

        outcome = {
            "bet_id": str(uuid.uuid4()),
            "season": int(season),
            "game_date": game_date,
            "player_id": int(player_id) if player_id is not None else None,
            "player_name": str(player_name),
            "market": market,
            "sportsbook": pred["sportsbook"],
            "side": side,
            "line": float(line),
            "price": price,
            "actual_result": actual_result,
            "result": result,
            "profit_units": profit_units,
            "confidence_tier": confidence_tier,
            "edge_at_placement": float(edge_pct),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }

        outcomes.append(outcome)

    logger.info("Graded %d bets", len(outcomes))

    if outcomes:
        wins = sum(1 for o in outcomes if o["result"] == "win")
        losses = sum(1 for o in outcomes if o["result"] == "loss")
        pushes = sum(1 for o in outcomes if o["result"] == "push")
        total_profit = sum(o["profit_units"] for o in outcomes)

        logger.info("Results: %d wins, %d losses, %d pushes", wins, losses, pushes)
        logger.info("Total profit: %.2f units", total_profit)

    return outcomes


def save_nba_outcomes(outcomes: List[Dict]) -> None:
    """Persist outcomes to nba_bet_outcomes and update nba_daily_performance.

    Args:
        outcomes: List of outcome dictionaries from grade_nba_bets()
    """
    if not outcomes:
        logger.info("No outcomes to save")
        return

    logger.info("Saving %d outcomes to database...", len(outcomes))

    insert_sql = """
        INSERT OR REPLACE INTO nba_bet_outcomes (
            bet_id, season, game_date, player_id, player_name, market,
            sportsbook, side, line, price, actual_result, result,
            profit_units, confidence_tier, edge_at_placement, recorded_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """

    outcome_tuples = [
        (
            o["bet_id"], o["season"], o["game_date"], o["player_id"],
            o["player_name"], o["market"], o["sportsbook"], o["side"],
            o["line"], o["price"], o["actual_result"], o["result"],
            o["profit_units"], o["confidence_tier"], o["edge_at_placement"],
            o["recorded_at"],
        )
        for o in outcomes
    ]

    executemany(insert_sql, outcome_tuples)
    logger.info("Inserted %d outcomes into nba_bet_outcomes", len(outcome_tuples))

    df = pd.DataFrame(outcomes)
    season = int(df["season"].iloc[0])
    game_date = df["game_date"].iloc[0]

    total_bets = len(outcomes)
    wins = len(df[df["result"] == "win"])
    losses = len(df[df["result"] == "loss"])
    pushes = len(df[df["result"] == "push"])
    profit_units = df["profit_units"].sum()

    units_risked = wins + losses
    roi_pct = (profit_units / units_risked * 100) if units_risked > 0 else 0.0

    avg_edge = df["edge_at_placement"].mean()

    best_bet_row = df.loc[df["profit_units"].idxmax()] if not df.empty else None
    worst_bet_row = df.loc[df["profit_units"].idxmin()] if not df.empty else None

    best_bet = (
        f"{best_bet_row['player_name']} {best_bet_row['market']} {best_bet_row['side']} {best_bet_row['line']}"
        if best_bet_row is not None
        else None
    )
    worst_bet = (
        f"{worst_bet_row['player_name']} {worst_bet_row['market']} {worst_bet_row['side']} {worst_bet_row['line']}"
        if worst_bet_row is not None
        else None
    )

    perf_sql = """
        INSERT OR REPLACE INTO nba_daily_performance (
            season, game_date, total_bets, wins, losses, pushes,
            profit_units, roi_pct, avg_edge,
            best_bet, worst_bet, updated_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """

    execute(
        perf_sql,
        (
            season, game_date, total_bets, wins, losses, pushes,
            profit_units, roi_pct, avg_edge,
            best_bet, worst_bet,
            datetime.now(timezone.utc).isoformat(),
        ),
    )

    logger.info("Updated nba_daily_performance for %s", game_date)
    logger.info("  Total bets: %d", total_bets)
    logger.info("  Record: %d-%d-%d", wins, losses, pushes)
    logger.info("  Profit: %.2f units", profit_units)
    logger.info("  ROI: %.2f%%", roi_pct)
    logger.info("  Avg edge: %.2f%%", avg_edge)

    # Compute CLV after saving outcomes
    compute_and_save_clv(game_date)


def compute_and_save_clv(game_date: str) -> int:
    """Compute closing-line value for all bets on a given game date.

    Compares opening and closing lines from nba_odds snapshots
    and writes CLV metrics to the nba_clv table.

    Args:
        game_date: ISO date string (e.g., '2026-02-17')

    Returns:
        Number of CLV records inserted.
    """
    logger.info("Computing CLV for %s...", game_date)

    bet_outcomes = read_dataframe(
        """
        SELECT bet_id, player_id, market, sportsbook, game_date
        FROM nba_bet_outcomes
        WHERE game_date = ?
        """,
        params=(game_date,),
    )

    if bet_outcomes.empty:
        logger.info("No bet outcomes found for CLV computation on %s", game_date)
        return 0

    # Get open lines (earliest snapshot per player/market/sportsbook)
    open_lines = read_dataframe(
        """
        SELECT o.player_id, o.market, o.sportsbook, o.line AS open_line
        FROM nba_odds o
        INNER JOIN (
            SELECT player_id, market, sportsbook, MIN(as_of) AS min_as_of
            FROM nba_odds
            WHERE game_date = ? AND player_id IS NOT NULL
            GROUP BY player_id, market, sportsbook
        ) earliest
            ON o.player_id = earliest.player_id
            AND o.market = earliest.market
            AND o.sportsbook = earliest.sportsbook
            AND o.as_of = earliest.min_as_of
        WHERE o.game_date = ?
        """,
        params=(game_date, game_date),
    )

    # Get close lines (latest snapshot per player/market/sportsbook)
    close_lines = read_dataframe(
        """
        SELECT o.player_id, o.market, o.sportsbook, o.line AS close_line
        FROM nba_odds o
        INNER JOIN (
            SELECT player_id, market, sportsbook, MAX(as_of) AS max_as_of
            FROM nba_odds
            WHERE game_date = ? AND player_id IS NOT NULL
            GROUP BY player_id, market, sportsbook
        ) latest
            ON o.player_id = latest.player_id
            AND o.market = latest.market
            AND o.sportsbook = latest.sportsbook
            AND o.as_of = latest.max_as_of
        WHERE o.game_date = ?
        """,
        params=(game_date, game_date),
    )

    if open_lines.empty or close_lines.empty:
        logger.info("No odds snapshots found for CLV on %s", game_date)
        return 0

    # Merge open and close lines
    merge_keys = ["player_id", "market", "sportsbook"]
    lines_df = pd.merge(open_lines, close_lines, on=merge_keys, how="inner")

    # Join with bet outcomes
    merged = pd.merge(
        bet_outcomes,
        lines_df,
        on=merge_keys,
        how="inner",
    )

    if merged.empty:
        logger.info("No matching odds found for CLV on %s", game_date)
        return 0

    records = []
    for _, row in merged.iterrows():
        open_line = float(row["open_line"])
        close_line = float(row["close_line"])
        clv_points = open_line - close_line
        clv_pct = (clv_points / open_line * 100) if open_line != 0 else 0.0

        records.append((
            row["bet_id"],
            int(row["player_id"]) if row["player_id"] is not None else None,
            row["market"],
            row["sportsbook"],
            game_date,
            open_line,
            close_line,
            round(clv_points, 4),
            round(clv_pct, 4),
        ))

    insert_sql = """
        INSERT OR REPLACE INTO nba_clv (
            bet_id, player_id, market, sportsbook, game_date,
            open_line, close_line, clv_points, clv_pct
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    executemany(insert_sql, records)
    logger.info("Inserted %d CLV records for %s", len(records), game_date)
    return len(records)


def main():
    """CLI entry point for recording NBA bet outcomes."""
    parser = argparse.ArgumentParser(
        description="Record NBA bet outcomes by comparing predictions to actuals"
    )
    parser.add_argument(
        "--game-date",
        type=str,
        required=True,
        help="Game date in YYYY-MM-DD format (e.g., 2026-02-17)",
    )

    args = parser.parse_args()

    try:
        datetime.strptime(args.game_date, "%Y-%m-%d")
    except ValueError:
        logger.error("Invalid date format '%s'. Use YYYY-MM-DD.", args.game_date)
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    try:
        outcomes = grade_nba_bets(args.game_date)

        if outcomes:
            save_nba_outcomes(outcomes)
            logger.info("Successfully recorded outcomes for %s", args.game_date)
        else:
            logger.info("No outcomes to record for %s", args.game_date)

    except Exception as e:
        logger.error("Error recording outcomes: %s", e)
        raise


if __name__ == "__main__":
    main()
