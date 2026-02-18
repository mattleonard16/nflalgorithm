"""NBA Value Betting Engine.

Joins NBA projections with NBA odds to compute value bets using Kelly criterion.
Modeled after the NFL value_betting_engine.py but adapted for NBA markets.

Usage:
    python nba_value_engine.py --date 2026-02-17
    python nba_value_engine.py --date 2026-02-17 --min-edge 0.08
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from utils.db import executemany, read_dataframe

# ---------------------------------------------------------------------------
# Normal CDF approximation (avoids scipy dependency)
# Abramowitz & Stegun approximation — max absolute error < 7.5e-8
# ---------------------------------------------------------------------------
_A1 = 0.254829592
_A2 = -0.284496736
_A3 = 1.421413741
_A4 = -1.453152027
_A5 = 1.061405429
_P = 0.3275911


def _standard_norm_cdf(z: float) -> float:
    """CDF of the standard normal distribution N(0,1) at *z*."""
    sign = 1.0 if z >= 0 else -1.0
    x = abs(z) / math.sqrt(2)
    t = 1.0 / (1.0 + _P * x)
    poly = t * (_A1 + t * (_A2 + t * (_A3 + t * (_A4 + t * _A5))))
    result = 1.0 - poly * math.exp(-(x * x))
    return 0.5 * (1.0 + sign * result)


# ---------------------------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------------------------


def implied_probability(odds: int) -> float:
    """Convert American odds to implied win probability (no vig removed)."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal (European) odds."""
    if odds < 0:
        return 1 + 100 / abs(odds)
    return 1 + odds / 100


def prob_over(mu: float, sigma: float, line: float) -> float:
    """Probability that a player exceeds *line* given a N(mu, sigma) projection."""
    if sigma <= 0:
        return 0.0 if line >= mu else 1.0
    z = (line - mu) / sigma
    return 1.0 - _standard_norm_cdf(z)


def kelly_fraction(win_prob: float, odds: int, fraction: float = 0.25) -> float:
    """Fractional Kelly criterion stake as a fraction of bankroll, capped at 10%."""
    if odds == 0:
        return 0.0
    decimal_odds = american_to_decimal(odds)
    if decimal_odds <= 1:
        return 0.0
    full_kelly = (decimal_odds * win_prob - 1) / (decimal_odds - 1)
    return max(0.0, min(full_kelly * fraction, 0.10))


def expected_roi(win_prob: float, odds: int) -> float:
    """Expected return on a $1 wager (positive = profitable)."""
    if odds == 0:
        return 0.0
    payout = 100 / abs(odds) if odds < 0 else odds / 100
    return win_prob * payout - (1 - win_prob)


# ---------------------------------------------------------------------------
# Player name normalisation for fallback matching
# ---------------------------------------------------------------------------


def _normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    name = name.lower()
    name = re.sub(r"[^a-z\s]", "", name)
    return re.sub(r"\s+", " ", name).strip()


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------


def rank_nba_value(
    game_date: str,
    season: int,
    min_edge: float = 0.05,
) -> list[dict]:
    """Return ranked NBA value bets for *game_date*.

    Algorithm:
    1. Load projections and odds from the database.
    2. Primary join on player_id + market for odds rows that have a player_id.
    3. Fallback name join for odds rows where player_id is NULL.
    4. Estimate sigma = max(projected_value * 0.20, 3.0).
    5. Compute p_win, edge, ROI, and Kelly for each matched row.
    6. Filter to rows with edge_percentage >= min_edge.
    7. Return sorted by edge_percentage descending.
    """
    projections = read_dataframe(
        "SELECT player_id, player_name, team, market, projected_value, confidence "
        "FROM nba_projections WHERE game_date = ?",
        (game_date,),
    )
    odds = read_dataframe(
        "SELECT event_id, player_id, player_name, market, sportsbook, "
        "line, over_price, under_price "
        "FROM nba_odds WHERE game_date = ?",
        (game_date,),
    )

    if projections.empty or odds.empty:
        return []

    # ------------------------------------------------------------------ #
    # Primary join: player_id (not null) + market                         #
    # ------------------------------------------------------------------ #
    odds_with_id = odds[odds["player_id"].notna()].copy()
    if not odds_with_id.empty:
        # Coerce types to match projections so the merge key is compatible
        odds_with_id["player_id"] = odds_with_id["player_id"].astype(
            projections["player_id"].dtype
        )

    matched = pd.merge(
        projections,
        odds_with_id,
        on=["player_id", "market"],
        how="inner",
        suffixes=("_proj", "_odds"),
    )

    # ------------------------------------------------------------------ #
    # Fallback join: normalised name for odds rows without player_id       #
    # ------------------------------------------------------------------ #
    odds_no_id = odds[odds["player_id"].isna()].copy()
    if not odds_no_id.empty:
        projections_copy = projections.copy()
        projections_copy["_norm_name"] = projections_copy["player_name"].map(_normalize_name)
        odds_no_id["_norm_name"] = odds_no_id["player_name"].map(_normalize_name)

        name_matched = pd.merge(
            projections_copy,
            odds_no_id.drop(columns=["player_id"]),
            on=["_norm_name", "market"],
            how="inner",
            suffixes=("_proj", "_odds"),
        )
        name_matched = name_matched.drop(columns=["_norm_name"], errors="ignore")

        # Align to matched columns before concatenating
        for col in matched.columns:
            if col not in name_matched.columns:
                name_matched[col] = None
        name_matched = name_matched[[c for c in matched.columns if c in name_matched.columns]]

        matched = pd.concat([matched, name_matched], ignore_index=True)

    if matched.empty:
        return []

    # Resolve a single player_name column
    if "player_name_proj" in matched.columns:
        odds_name_col = "player_name_odds" if "player_name_odds" in matched.columns else "player_name"
        matched["player_name"] = matched["player_name_proj"].fillna(matched.get(odds_name_col, ""))

    # ------------------------------------------------------------------ #
    # Per-row value calculations                                           #
    # ------------------------------------------------------------------ #
    results: list[dict] = []

    for _, row in matched.iterrows():
        try:
            projected_value = float(row["projected_value"])
            line = float(row["line"])
            over_price_raw = row.get("over_price")
            if over_price_raw is None or (
                isinstance(over_price_raw, float) and math.isnan(over_price_raw)
            ):
                continue
            over_price = int(over_price_raw)
        except (TypeError, ValueError):
            continue

        sigma = max(projected_value * 0.20, 3.0)
        p_win = prob_over(mu=projected_value, sigma=sigma, line=line)
        implied_prob = implied_probability(over_price)
        edge_pct = p_win - implied_prob

        if edge_pct < min_edge:
            continue

        roi = expected_roi(p_win, over_price)
        kelly = kelly_fraction(p_win, over_price)

        under_price_raw = row.get("under_price")
        under_price: int | None = None
        if under_price_raw is not None and not (
            isinstance(under_price_raw, float) and math.isnan(under_price_raw)
        ):
            under_price = int(under_price_raw)

        results.append(
            {
                "season": season,
                "game_date": game_date,
                "player_id": row.get("player_id"),
                "player_name": str(row.get("player_name", "")),
                "team": row.get("team"),
                "event_id": str(row["event_id"]),
                "market": str(row["market"]),
                "sportsbook": str(row["sportsbook"]),
                "line": line,
                "over_price": over_price,
                "under_price": under_price,
                "mu": projected_value,
                "sigma": sigma,
                "p_win": round(p_win, 6),
                "edge_percentage": round(edge_pct, 6),
                "expected_roi": round(roi, 6),
                "kelly_fraction": round(kelly, 6),
                "confidence": row.get("confidence"),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    results.sort(key=lambda r: r["edge_percentage"], reverse=True)
    return results


def materialize_nba_value(
    game_date: str,
    season: int,
    min_edge: float = 0.05,
) -> int:
    """Persist ranked NBA value bets to *nba_materialized_value_view*.

    Returns the number of rows written.
    """
    rows = rank_nba_value(game_date=game_date, season=season, min_edge=min_edge)

    if not rows:
        return 0

    sql = """
        INSERT OR REPLACE INTO nba_materialized_value_view (
            season, game_date, player_id, player_name, team,
            event_id, market, sportsbook, line, over_price, under_price,
            mu, sigma, p_win, edge_percentage, expected_roi, kelly_fraction,
            confidence, generated_at
        ) VALUES (
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?
        )
    """

    params = [
        (
            r["season"],
            r["game_date"],
            r["player_id"],
            r["player_name"],
            r["team"],
            r["event_id"],
            r["market"],
            r["sportsbook"],
            r["line"],
            r["over_price"],
            r["under_price"],
            r["mu"],
            r["sigma"],
            r["p_win"],
            r["edge_percentage"],
            r["expected_roi"],
            r["kelly_fraction"],
            r["confidence"],
            r["generated_at"],
        )
        for r in rows
    ]

    executemany(sql, params)
    return len(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute NBA value bets and materialise results to DB."
    )
    parser.add_argument(
        "--date",
        required=True,
        metavar="YYYY-MM-DD",
        help="Game date to process (e.g. 2026-02-17)",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2025,
        help="NBA season year (default: 2025)",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.05,
        dest="min_edge",
        help="Minimum edge threshold (default: 0.05 = 5%%)",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        count = materialize_nba_value(
            game_date=args.date,
            season=args.season,
            min_edge=args.min_edge,
        )
        print(
            f"Materialised {count} NBA value bet(s) for {args.date} "
            f"(min_edge={args.min_edge:.0%})"
        )
    except Exception as exc:
        print(f"Error computing NBA value bets: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
