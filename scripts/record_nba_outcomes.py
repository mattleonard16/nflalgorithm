"""Record NBA Bet Outcomes - Tracks actual results and calculates P&L.

Grades bets from the NBA materialized value view against actual player game logs
and updates the nba_bet_outcomes and nba_daily_performance tables.

Usage:
    python scripts/record_nba_outcomes.py --game-date 2026-02-17
"""

import argparse
import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from sports.markets import get_sport
from utils.db import execute, executemany, get_backend, read_dataframe
from utils.grading import calculate_profit_units, get_confidence_tier, grade_bet

logger = logging.getLogger(__name__)

# Compatibility alias for callers that still import the NBA-specific mapping.
NBA_MARKET_TO_STAT = {market: spec.stat_column for market, spec in get_sport("nba").markets.items()}

# All NBA value bets target the over side (over_price is used for Kelly/value calc)
NBA_DEFAULT_SIDE = "over"


def make_nba_bet_id(
    season: int,
    game_date: str,
    player_id: object,
    market: str,
    sportsbook: str,
    side: str,
    line: float,
) -> str:
    """Derive a stable bet_id from the bet's natural key.

    NBA counterpart of ``make_bet_id`` in ``scripts/record_outcomes.py``. A
    random UUID per grading run defeats the ``bet_id`` upsert, so every re-grade
    appended a full duplicate copy of the day to ``nba_bet_outcomes`` (and from
    there to ``nba_clv`` and everything that reads them). Hashing the natural
    key makes re-grading idempotent.
    """
    key = f"{season}|{game_date}|{player_id}|{market}|{sportsbook}|{side}|{float(line):.4f}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _upsert_sql(table: str, columns: Sequence[str], key_columns: Sequence[str]) -> str:
    """Build an insert-or-overwrite statement for the active backend.

    ``INSERT OR REPLACE`` is SQLite-only syntax, so SQLite keeps it unchanged
    and MySQL gets the same overwrite-on-primary-key semantics through
    ``ON DUPLICATE KEY UPDATE``.
    """
    column_list = ", ".join(columns)
    placeholders = ", ".join("?" for _ in columns)
    if get_backend() == "mysql":
        updates = ", ".join(f"{c}=VALUES({c})" for c in columns if c not in key_columns)
        return (
            f"INSERT INTO {table} ({column_list}) VALUES ({placeholders}) "
            f"ON DUPLICATE KEY UPDATE {updates}"
        )
    return f"INSERT OR REPLACE INTO {table} ({column_list}) VALUES ({placeholders})"


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
            market, sportsbook, line, over_price, under_price, mu, sigma, p_win,
            edge_percentage, expected_roi, kelly_fraction, confidence, side,
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
        # Missing box scores mean "not graded yet", not a push: a push is a
        # settled zero-profit result and would be persisted as one. Leave the
        # day ungraded so a re-run after ingest grades it properly.
        logger.warning(
            "No actual stats found for %s - leaving all %d bets ungraded",
            game_date,
            len(predictions),
        )
        return []

    logger.info("Found actual stats for %d players", len(actuals))

    outcomes: List[Dict] = []
    ungraded = 0

    for _, pred in predictions.iterrows():
        player_id = pred["player_id"]
        market = pred["market"]
        line = pred["line"]
        edge_pct = pred["edge_percentage"]
        season = pred["season"]

        # Use the side stored in the materialized view (over or under)
        side = pred.get("side", "over") or "over"
        if side == "under":
            under_price_raw = pred.get("under_price")
            price = int(under_price_raw) if under_price_raw is not None else int(pred["over_price"])
        else:
            price = int(pred["over_price"])

        stat_column = NBA_MARKET_TO_STAT.get(market)
        if not stat_column:
            logger.warning("Unknown market '%s' for player %s", market, player_id)
            continue

        player_actuals = actuals[actuals["player_id"] == player_id]

        # No box-score row (or a NULL stat) is an unknown outcome, not a push.
        # Skip it so it stays ungraded instead of being stored as settled.
        if player_actuals.empty:
            ungraded += 1
            continue
        actual_result = player_actuals.iloc[0].get(stat_column)
        if actual_result is None or pd.isna(actual_result):
            ungraded += 1
            continue
        actual_result = float(actual_result)
        result = grade_bet(actual_result, line, side)

        profit_units = calculate_profit_units(result, price)
        # `nba_materialized_value_view.edge_percentage` holds a fraction (p_win
        # minus implied probability) while `get_confidence_tier` takes a percent.
        confidence_tier = get_confidence_tier(edge_pct * 100.0)

        player_name = pred["player_name"]
        if player_name is None or (isinstance(player_name, float) and pd.isna(player_name)):
            if not player_actuals.empty:
                player_name = player_actuals.iloc[0]["player_name"]
            else:
                player_name = "Unknown"

        outcome = {
            # Hash the stored (int) player_id so a float-typed column (NULLs
            # elsewhere in the frame) cannot change the key between runs.
            "bet_id": make_nba_bet_id(
                int(season),
                game_date,
                int(player_id) if player_id is not None else None,
                market,
                pred["sportsbook"],
                side,
                line,
            ),
            "season": int(season),
            "game_date": game_date,
            "player_id": int(player_id) if player_id is not None else None,
            "player_name": str(player_name),
            "market": market,
            "sportsbook": pred["sportsbook"],
            "side": side,
            "line": float(line),
            "placed_at_line": float(line),
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
    if ungraded:
        logger.warning("Left %d bets ungraded (no actual stat for the player)", ungraded)

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

    insert_sql = _upsert_sql(
        "nba_bet_outcomes",
        (
            "bet_id", "season", "game_date", "player_id", "player_name", "market",
            "sportsbook", "side", "line", "placed_at_line", "price", "actual_result", "result",
            "profit_units", "confidence_tier", "edge_at_placement", "recorded_at",
        ),
        key_columns=("bet_id",),
    )

    outcome_tuples = [
        (
            o["bet_id"],
            o["season"],
            o["game_date"],
            o["player_id"],
            o["player_name"],
            o["market"],
            o["sportsbook"],
            o["side"],
            o["line"],
            o.get("placed_at_line"),
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

    perf_sql = _upsert_sql(
        "nba_daily_performance",
        (
            "season", "game_date", "total_bets", "wins", "losses", "pushes",
            "profit_units", "roi_pct", "avg_edge",
            "best_bet", "worst_bet", "updated_at",
        ),
        key_columns=("season", "game_date"),
    )

    execute(
        perf_sql,
        (
            season,
            game_date,
            total_bets,
            wins,
            losses,
            pushes,
            profit_units,
            roi_pct,
            avg_edge,
            best_bet,
            worst_bet,
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
        SELECT bet_id, player_id, market, sportsbook, game_date, side, placed_at_line
        FROM nba_bet_outcomes
        WHERE game_date = ?
        """,
        params=(game_date,),
    )

    if bet_outcomes.empty:
        logger.info("No bet outcomes found for CLV computation on %s", game_date)
        return 0

    snapshots = read_dataframe(
        """
        SELECT player_id, market, sportsbook, line, as_of
        FROM nba_odds
        WHERE game_date = ? AND player_id IS NOT NULL
        """,
        params=(game_date,),
    )

    if snapshots.empty:
        logger.info("No odds snapshots found for CLV on %s", game_date)
        return 0

    # Open = earliest snapshot, close = latest snapshot per player/market/book.
    # `as_of` is TEXT, and SQL MIN/MAX on it orders lexically, which breaks as
    # soon as two writers format the timestamp differently (naive vs "+00:00",
    # "T" vs " "). Parse it so the ordering is chronological.
    merge_keys = ["player_id", "market", "sportsbook"]
    snapshots["_as_of_ts"] = pd.to_datetime(
        snapshots["as_of"], errors="coerce", utc=True, format="mixed"
    )
    snapshots = snapshots.dropna(subset=["_as_of_ts"]).sort_values(
        merge_keys + ["_as_of_ts"], kind="mergesort"
    )
    grouped = snapshots.groupby(merge_keys, sort=False)
    open_lines = grouped.head(1)[merge_keys + ["line"]].rename(columns={"line": "open_line"})
    close_lines = grouped.tail(1)[merge_keys + ["line"]].rename(columns={"line": "close_line"})

    if open_lines.empty or close_lines.empty:
        logger.info("No odds snapshots found for CLV on %s", game_date)
        return 0

    # Merge open and close lines
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
        # Use placed_at_line (actual line taken) when available; fall back to open line
        placed = row.get("placed_at_line")
        open_line = (
            float(placed) if placed is not None and not pd.isna(placed) else float(row["open_line"])
        )
        close_line = float(row["close_line"])
        # Positive CLV = the bet beat the close. An over bettor wins that when
        # the market moves the line up after entry (they hold the lower line);
        # an under bettor wins it when the line moves down.
        if (row.get("side") or NBA_DEFAULT_SIDE) == "under":
            clv_points = open_line - close_line
        else:
            clv_points = close_line - open_line
        clv_pct = (clv_points / open_line * 100) if open_line != 0 else 0.0

        records.append(
            (
                row["bet_id"],
                int(row["player_id"]) if row["player_id"] is not None else None,
                row["market"],
                row["sportsbook"],
                game_date,
                open_line,
                close_line,
                round(clv_points, 4),
                round(clv_pct, 4),
            )
        )

    insert_sql = _upsert_sql(
        "nba_clv",
        (
            "bet_id", "player_id", "market", "sportsbook", "game_date",
            "open_line", "close_line", "clv_points", "clv_pct",
        ),
        key_columns=("bet_id",),
    )
    executemany(insert_sql, records)
    logger.info("Inserted %d CLV records for %s", len(records), game_date)
    return len(records)


def compute_nba_line_accuracy(game_date: str) -> int:
    """Compute model vs line accuracy for graded bets.

    Joins nba_bet_outcomes with nba_projections and nba_odds to compare
    model prediction error against sportsbook line error.

    Returns number of records written.
    """
    logger.info("Computing line accuracy for %s...", game_date)

    # Get graded outcomes with actuals
    outcomes = read_dataframe(
        """
        SELECT season, game_date, player_id, market, sportsbook,
               line, actual_result, side
        FROM nba_bet_outcomes
        WHERE game_date = ? AND actual_result IS NOT NULL
        """,
        params=(game_date,),
    )

    if outcomes.empty:
        logger.info("No graded outcomes with actuals for %s", game_date)
        return 0

    # Get model projections
    projections = read_dataframe(
        "SELECT player_id, market, projected_value AS mu, sigma "
        "FROM nba_projections WHERE game_date = ?",
        params=(game_date,),
    )

    if projections.empty:
        logger.info("No projections found for %s", game_date)
        return 0

    # Merge outcomes with projections
    merged = pd.merge(
        outcomes,
        projections,
        on=["player_id", "market"],
        how="inner",
    )

    if merged.empty:
        logger.info("No matches between outcomes and projections for %s", game_date)
        return 0

    now = datetime.now(timezone.utc).isoformat()
    records = []

    for _, row in merged.iterrows():
        actual = float(row["actual_result"])
        mu = float(row["mu"]) if row["mu"] is not None else None
        sigma = float(row["sigma"]) if row.get("sigma") is not None else None
        line = float(row["line"])

        model_abs_error = abs(mu - actual) if mu is not None else None
        line_abs_error = abs(line - actual)

        model_beats_line = None
        if model_abs_error is not None:
            model_beats_line = 1 if model_abs_error < line_abs_error else 0

        is_over_hit = 1 if actual > line else 0

        records.append(
            (
                int(row["season"]),
                row["game_date"],
                int(row["player_id"]),
                row["market"],
                row["sportsbook"],
                line,
                actual,
                mu,
                sigma,
                model_abs_error,
                line_abs_error,
                model_beats_line,
                is_over_hit,
                now,
            )
        )

    insert_sql = _upsert_sql(
        "nba_line_accuracy_history",
        (
            "season", "game_date", "player_id", "market", "sportsbook",
            "line", "actual", "mu", "sigma",
            "model_abs_error", "line_abs_error", "model_beats_line",
            "is_over_hit", "computed_at",
        ),
        key_columns=("season", "game_date", "player_id", "market", "sportsbook"),
    )
    executemany(insert_sql, records)
    logger.info("Wrote %d line accuracy records for %s", len(records), game_date)
    return len(records)


def summarize_nba_accuracy(
    season: int,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Return aggregate accuracy metrics.

    Returns dict with: avg_model_mae, avg_line_mae, model_beats_line_pct, n_bets
    """
    where_clauses = ["season = ?"]
    params: list = [season]

    if start_date:
        where_clauses.append("game_date >= ?")
        params.append(start_date)
    if end_date:
        where_clauses.append("game_date <= ?")
        params.append(end_date)

    where = " AND ".join(where_clauses)

    df = read_dataframe(
        f"""
        SELECT model_abs_error, line_abs_error, model_beats_line
        FROM nba_line_accuracy_history
        WHERE {where}
        """,
        params=tuple(params),
    )

    if df.empty:
        return {
            "avg_model_mae": None,
            "avg_line_mae": None,
            "model_beats_line_pct": None,
            "n_bets": 0,
        }

    return {
        "avg_model_mae": round(float(df["model_abs_error"].mean()), 4),
        "avg_line_mae": round(float(df["line_abs_error"].mean()), 4),
        "model_beats_line_pct": round(float(df["model_beats_line"].mean()) * 100, 2),
        "n_bets": len(df),
    }


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

        compute_nba_line_accuracy(args.game_date)
        summary = summarize_nba_accuracy(
            season=outcomes[0]["season"] if outcomes else 2025,
            start_date=args.game_date,
            end_date=args.game_date,
        )
        if summary["n_bets"] > 0:
            logger.info(
                "Accuracy: model MAE=%.2f, line MAE=%.2f, model beats line %.1f%% (%d bets)",
                summary["avg_model_mae"],
                summary["avg_line_mae"],
                summary["model_beats_line_pct"],
                summary["n_bets"],
            )

    except Exception as e:
        logger.error("Error recording outcomes: %s", e)
        raise


if __name__ == "__main__":
    main()
