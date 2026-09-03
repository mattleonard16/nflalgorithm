"""
Record Bet Outcomes - Tracks actual results and calculates P&L

This script grades bets from the materialized view against actual player stats
and updates the bet_outcomes and weekly_performance tables.

Usage:
    python scripts/record_outcomes.py --season 2025 --week 12
"""

import argparse
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from config import config
from utils.clv import STATUS_OK, compute_clv, resolve_closing_lines
from utils.db import execute, executemany, get_backend, get_connection, read_dataframe
from utils.grading import calculate_profit_units, get_confidence_tier, grade_bet
from utils.live_odds import kickoffs_from_games
from utils.nfl_markets import MARKET_TO_STAT


def make_bet_id(
    season: int,
    week: int,
    player_id: str,
    market: str,
    sportsbook: str,
    side: str,
    line: float,
) -> str:
    """Derive a stable bet_id from the bet's natural key.

    A random UUID per grading run defeats the upsert, so re-running a week
    duplicates every row in bet_outcomes. Hashing the natural key makes
    re-grading idempotent and lets clv_weekly.bet_id join back reliably.
    """
    key = f"{season}|{week}|{player_id}|{market}|{sportsbook}|{side}|{float(line):.4f}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


# Market to stat column mapping
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
            market, sportsbook, line, price, side, mu, sigma, p_win,
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
            receptions, targets, rushing_tds, receiving_tds
        FROM player_stats_enhanced
        WHERE season = ? AND week = ?
    """
    actuals = read_dataframe(actuals_query, params=(season, week))

    if actuals.empty:
        print(f"WARNING: No actual stats found for {season} Week {week}")
        print("All bets will be marked as pushes")
    else:
        print(f"Found actual stats for {len(actuals)} players")
        if "anytime_td" not in actuals.columns:
            rush_td = pd.to_numeric(actuals.get("rushing_tds", 0), errors="coerce").fillna(0.0)
            rec_td = pd.to_numeric(actuals.get("receiving_tds", 0), errors="coerce").fillna(0.0)
            actuals["anytime_td"] = (rush_td + rec_td > 0).astype(int)

    # Grade each bet
    outcomes = []

    for _, pred in predictions.iterrows():
        player_id = pred["player_id"]
        market = pred["market"]
        line = pred["line"]
        price = pred["price"]
        edge_pct = pred["edge_percentage"]

        # T0 #4: read side from view (over or under). Default to 'over' for
        # legacy rows materialized before the side column existed.
        side_val = pred.get("side") if "side" in pred.index else None
        side = side_val if isinstance(side_val, str) and side_val else "over"

        # Get stat column for this market
        stat_column = MARKET_TO_STAT.get(market)

        if not stat_column:
            print(f"WARNING: Unknown market '{market}' for player {player_id}")
            continue

        # Find actual result
        player_actuals = actuals[actuals["player_id"] == player_id]

        if player_actuals.empty:
            # No actual data - treat as push
            actual_result = None
            result = "push"
        else:
            player_row = player_actuals.iloc[0]
            actual_result = player_row.get(stat_column)

            if pd.isna(actual_result):
                result = "push"
            else:
                result = grade_bet(actual_result, line, side)

        # Calculate profit
        profit_units = calculate_profit_units(result, price)

        # Determine confidence tier
        confidence_tier = get_confidence_tier(edge_pct)

        # Create outcome record
        outcome = {
            "bet_id": make_bet_id(season, week, player_id, market, pred["sportsbook"], side, line),
            "season": season,
            "week": week,
            "event_id": pred.get("event_id"),
            "player_id": player_id,
            "player_name": (
                actuals[actuals["player_id"] == player_id]["name"].iloc[0]
                if not actuals[actuals["player_id"] == player_id].empty
                else "Unknown"
            ),
            "market": market,
            "sportsbook": pred["sportsbook"],
            "side": side,
            "line": line,
            "price": price,
            "actual_result": actual_result if not pd.isna(actual_result) else None,
            "result": result,
            "profit_units": profit_units,
            "confidence_tier": confidence_tier,
            "edge_at_placement": edge_pct,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            # Carried for CLV only — not persisted to bet_outcomes.
            "mu": pred.get("mu"),
            "sigma": pred.get("sigma"),
        }

        outcomes.append(outcome)

    print(f"Graded {len(outcomes)} bets")

    # Print summary
    if outcomes:
        wins = sum(1 for o in outcomes if o["result"] == "win")
        losses = sum(1 for o in outcomes if o["result"] == "loss")
        pushes = sum(1 for o in outcomes if o["result"] == "push")
        total_profit = sum(o["profit_units"] for o in outcomes)

        print(f"Results: {wins} wins, {losses} losses, {pushes} pushes")
        print(f"Total profit: {total_profit:.2f} units")

    return outcomes


def compute_and_save_clv(season: int, week: int, outcomes: List[Dict]) -> Optional[float]:
    """Compute closing-line value per bet and persist it to clv_weekly.

    Closing is the last snapshot at or before kickoff when the week's games
    rows have ``kickoff_utc``. Keys with no schedule row keep ``MAX(as_of)``.

    Args:
        season: NFL season year
        week: NFL week number
        outcomes: Graded outcomes from grade_bets()

    Returns:
        Mean clv_bp across bets with a resolvable close, or None when no bet
        had enough snapshot depth. Never 0 as a stand-in for "unknown".
    """
    if not outcomes:
        return None

    odds = read_dataframe(
        """
        SELECT event_id, player_id, market, sportsbook, line, price, under_price, as_of
        FROM weekly_odds
        WHERE season = ? AND week = ?
        """,
        params=(season, week),
    )

    if odds.empty:
        print(f"No weekly_odds snapshots for {season} Week {week} — CLV unavailable")
        return None

    games = read_dataframe(
        """
        SELECT game_id, kickoff_utc
        FROM games
        WHERE season = ? AND week = ?
        """,
        params=(season, week),
    )
    closing = resolve_closing_lines(odds, kickoffs_from_games(games))
    close_by_key = {
        (row["event_id"], row["player_id"], row["market"], row["sportsbook"]): row
        for row in closing.to_dict("records")
    }

    records: List[tuple] = []
    clv_values: List[float] = []
    insufficient = 0
    closed_at = datetime.now(timezone.utc).isoformat()

    for outcome in outcomes:
        key = (
            outcome.get("event_id"),
            outcome["player_id"],
            outcome["market"],
            outcome["sportsbook"],
        )
        result = compute_clv(outcome, close_by_key.get(key))

        if result["status"] != STATUS_OK:
            insufficient += 1
            continue

        if result["clv_bp"] is not None:
            clv_values.append(result["clv_bp"])

        records.append(
            (
                outcome["bet_id"],
                result["close_line"],
                result["close_price"],
                result["clv_bp"],
                result["closed_at"] or closed_at,
            )
        )

    if insufficient:
        print(f"CLV skipped for {insufficient} bets (insufficient odds snapshots)")

    if not records:
        print("No bets had enough snapshot depth to compute CLV")
        return None

    # `INSERT OR REPLACE` is SQLite-only syntax. Same upsert semantics on the
    # same primary key, incompatible spelling.
    if get_backend() == "mysql":
        clv_sql = """
            INSERT INTO clv_weekly (
                bet_id, close_line, close_price, clv_bp, closed_at
            ) VALUES (?, ?, ?, ?, ?)
            ON DUPLICATE KEY UPDATE
                close_line=VALUES(close_line),
                close_price=VALUES(close_price),
                clv_bp=VALUES(clv_bp),
                closed_at=VALUES(closed_at)
            """
    else:
        clv_sql = """
            INSERT INTO clv_weekly (
                bet_id, close_line, close_price, clv_bp, closed_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(bet_id) DO UPDATE SET
                close_line=excluded.close_line,
                close_price=excluded.close_price,
                clv_bp=excluded.clv_bp,
                closed_at=excluded.closed_at
            """
    executemany(clv_sql, records)
    print(f"Wrote {len(records)} CLV records to clv_weekly")

    if not clv_values:
        return None

    return sum(clv_values) / len(clv_values)


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

    # Insert into bet_outcomes. Re-grading a week must overwrite in place on
    # both backends; `INSERT OR REPLACE` would fail outright on MySQL.
    bet_outcomes_columns = """
        bet_id, season, week, player_id, player_name, market,
        sportsbook, side, line, price, actual_result, result,
        profit_units, confidence_tier, edge_at_placement, recorded_at
    """
    if get_backend() == "mysql":
        insert_sql = f"""
            INSERT INTO bet_outcomes ({bet_outcomes_columns})
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON DUPLICATE KEY UPDATE
                season=VALUES(season),
                week=VALUES(week),
                player_id=VALUES(player_id),
                player_name=VALUES(player_name),
                market=VALUES(market),
                sportsbook=VALUES(sportsbook),
                side=VALUES(side),
                line=VALUES(line),
                price=VALUES(price),
                actual_result=VALUES(actual_result),
                result=VALUES(result),
                profit_units=VALUES(profit_units),
                confidence_tier=VALUES(confidence_tier),
                edge_at_placement=VALUES(edge_at_placement),
                recorded_at=VALUES(recorded_at)
            """
    else:
        insert_sql = f"""
            INSERT INTO bet_outcomes ({bet_outcomes_columns})
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(bet_id) DO UPDATE SET
                season=excluded.season,
                week=excluded.week,
                player_id=excluded.player_id,
                player_name=excluded.player_name,
                market=excluded.market,
                sportsbook=excluded.sportsbook,
                side=excluded.side,
                line=excluded.line,
                price=excluded.price,
                actual_result=excluded.actual_result,
                result=excluded.result,
                profit_units=excluded.profit_units,
                confidence_tier=excluded.confidence_tier,
                edge_at_placement=excluded.edge_at_placement,
                recorded_at=excluded.recorded_at
            """

    outcome_tuples = [
        (
            o["bet_id"],
            o["season"],
            o["week"],
            o["player_id"],
            o["player_name"],
            o["market"],
            o["sportsbook"],
            o["side"],
            o["line"],
            o["price"],
            o["actual_result"],
            o["result"],
            o["profit_units"],
            o["confidence_tier"],
            o["edge_at_placement"],
            o["recorded_at"],
        )
        for o in outcomes
    ]

    executemany(insert_sql, outcome_tuples)
    print(f"Inserted {len(outcome_tuples)} outcomes into bet_outcomes")

    # Aggregate weekly performance
    df = pd.DataFrame(outcomes)
    # Cast out of numpy scalars: sqlite3 stores an unconverted np.int64 as a
    # BLOB, which silently breaks every later season/week lookup.
    season = int(df["season"].iloc[0])
    week = int(df["week"].iloc[0])

    total_bets = len(outcomes)
    wins = len(df[df["result"] == "win"])
    losses = len(df[df["result"] == "loss"])
    pushes = len(df[df["result"] == "push"])
    profit_units = float(df["profit_units"].sum())

    # ROI calculation: profit / units risked (excluding pushes)
    units_risked = wins + losses  # Each bet risks 1 unit
    roi_pct = (profit_units / units_risked * 100) if units_risked > 0 else 0.0

    avg_edge = float(df["edge_at_placement"].mean())

    # Best/worst bets (by profit)
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

    # CLV (Closing Line Value) against the closing snapshot, in basis points.
    clv_avg = compute_and_save_clv(season, week, outcomes)

    # weekly_performance.clv_avg is NOT NULL DEFAULT 0: binding an explicit
    # None works on SQLite but fails on MySQL, so omit the column entirely when
    # CLV is unknown and let the column default stand.
    clv_columns = "clv_avg," if clv_avg is not None else ""
    clv_placeholder = "?," if clv_avg is not None else ""
    # Updated columns exclude the (season, week) PK. clv_avg is only refreshed
    # when it was computed, so a re-grade that loses CLV keeps the prior value
    # rather than resetting it to the column default.
    perf_updates = [
        "total_bets",
        "wins",
        "losses",
        "pushes",
        "profit_units",
        "roi_pct",
        "avg_edge",
        "best_bet",
        "worst_bet",
        "updated_at",
    ]
    if clv_avg is not None:
        perf_updates.append("clv_avg")
    if get_backend() == "mysql":
        perf_assignments = ", ".join(f"{col}=VALUES({col})" for col in perf_updates)
        conflict_clause = f"ON DUPLICATE KEY UPDATE {perf_assignments}"
    else:
        perf_assignments = ", ".join(f"{col}=excluded.{col}" for col in perf_updates)
        conflict_clause = f"ON CONFLICT(season, week) DO UPDATE SET {perf_assignments}"
    perf_sql = f"""
        INSERT INTO weekly_performance (
            season, week, total_bets, wins, losses, pushes,
            profit_units, roi_pct, avg_edge, {clv_columns}
            best_bet, worst_bet, updated_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, {clv_placeholder} ?, ?, ?
        )
        {conflict_clause}
    """

    perf_params: List[object] = [
        season,
        week,
        total_bets,
        wins,
        losses,
        pushes,
        profit_units,
        roi_pct,
        avg_edge,
    ]
    if clv_avg is not None:
        perf_params.append(clv_avg)
    perf_params.extend([best_bet, worst_bet, datetime.now(timezone.utc).isoformat()])

    execute(perf_sql, tuple(perf_params))

    print(f"Updated weekly_performance for {season} Week {week}")
    print(f"  Total bets: {total_bets}")
    print(f"  Record: {wins}-{losses}-{pushes}")
    print(f"  Profit: {profit_units:.2f} units")
    print(f"  ROI: {roi_pct:.2f}%")
    print(f"  Avg edge: {avg_edge:.2f}%")
    print(f"  Avg CLV: {f'{clv_avg:.1f} bp' if clv_avg is not None else 'unavailable'}")


def main():
    """CLI entry point for recording bet outcomes."""
    parser = argparse.ArgumentParser(
        description="Record bet outcomes by comparing predictions to actuals"
    )
    parser.add_argument("--season", type=int, required=True, help="NFL season year (e.g., 2025)")
    parser.add_argument("--week", type=int, required=True, help="NFL week number (e.g., 13)")

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


if __name__ == "__main__":
    main()
