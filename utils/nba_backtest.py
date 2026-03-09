"""NBA Walk-Forward Historical Backtest.

Grades pre-computed value bets from nba_materialized_value_view against
actual results in nba_player_game_logs.  No lookahead bias: each day's bets
are graded only against that day's actuals.

Usage:
    from utils.nba_backtest import BacktestConfig, run_backtest
    config = BacktestConfig(start_date="2025-11-01", end_date="2026-02-28")
    result = run_backtest(config)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd

from utils.db import read_dataframe


# ---------------------------------------------------------------------------
# Market → stat column mapping
# ---------------------------------------------------------------------------

_MARKET_STAT_COL: dict[str, str] = {
    "pts": "pts",
    "reb": "reb",
    "ast": "ast",
    "fg3m": "fg3m",
}


# ---------------------------------------------------------------------------
# Config / Result dataclasses (frozen = immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str
    end_date: str
    initial_bankroll: float = 1.0
    min_edge: float = 0.08
    markets: list[str] = field(default_factory=lambda: ["pts", "reb", "ast", "fg3m"])
    tiers: list[str] | None = None
    use_calibrated: bool = False
    use_monte_carlo: bool = False


@dataclass(frozen=True)
class BacktestResult:
    total_bets: int
    wins: int
    losses: int
    pushes: int
    win_rate: float
    roi_pct: float
    total_profit_units: float
    max_drawdown: float
    sharpe_ratio: float
    clv_avg: float
    daily_pnl: pd.DataFrame
    per_market: pd.DataFrame
    per_tier: pd.DataFrame


# ---------------------------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------------------------


def max_drawdown_from_bankroll(bankroll_series: list[float]) -> float:
    """Compute maximum drawdown from a bankroll time series.

    Returns the maximum (peak - trough) where trough follows peak chronologically.
    Returns 0.0 for monotonically increasing series or series with < 2 elements.
    """
    if len(bankroll_series) < 2:
        return 0.0

    peak = bankroll_series[0]
    max_dd = 0.0
    for value in bankroll_series[1:]:
        if value > peak:
            peak = value
        drawdown = peak - value
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd


def _american_to_decimal(odds: int) -> float:
    if odds < 0:
        return 1.0 + 100.0 / abs(odds)
    return 1.0 + odds / 100.0


def _grade_bet(side: str, line: float, actual: float) -> str:
    """Grade a bet as 'win', 'loss', or 'push'."""
    if side == "over":
        if actual > line:
            return "win"
        if actual == line:
            return "push"
        return "loss"
    else:  # under
        if actual < line:
            return "win"
        if actual == line:
            return "push"
        return "loss"


def _profit_for_bet(grade: str, kelly: float, price: int) -> float:
    """Return profit units for a single bet outcome."""
    if grade == "push":
        return 0.0
    if grade == "win":
        decimal = _american_to_decimal(price)
        return kelly * (decimal - 1.0)
    # loss
    return -kelly


def _date_range(start: str, end: str) -> list[str]:
    """Return list of ISO date strings from start to end inclusive."""
    start_dt = date.fromisoformat(start)
    end_dt = date.fromisoformat(end)
    if end_dt < start_dt:
        return []
    dates: list[str] = []
    current = start_dt
    while current <= end_dt:
        dates.append(current.isoformat())
        current += timedelta(days=1)
    return dates


def _load_value_bets_for_date(game_date: str, config: BacktestConfig) -> pd.DataFrame:
    """Load pre-computed value bets from the materialized view."""
    placeholders = ",".join("?" for _ in config.markets)
    sql = f"""
        SELECT player_id, player_name, market, sportsbook, line, over_price,
               kelly_fraction, confidence_tier, edge_percentage, side
        FROM nba_materialized_value_view
        WHERE game_date = ?
        AND market IN ({placeholders})
    """
    params = (game_date, *config.markets)
    df = read_dataframe(sql, params)

    if df.empty:
        return df

    if config.tiers is not None and len(config.tiers) > 0:
        df = df[df["confidence_tier"].isin(config.tiers)]

    return df


def _load_actuals_for_date(game_date: str) -> dict[int, dict[str, Any]]:
    """Return a dict keyed by player_id with actual stat values for the date."""
    df = read_dataframe(
        "SELECT player_id, pts, reb, ast, fg3m FROM nba_player_game_logs WHERE game_date = ?",
        (game_date,),
    )
    if df.empty:
        return {}
    result: dict[int, dict[str, Any]] = {}
    for row in df.to_dict("records"):
        pid = int(row["player_id"])
        result[pid] = {
            "pts": row.get("pts", 0) or 0,
            "reb": row.get("reb", 0) or 0,
            "ast": row.get("ast", 0) or 0,
            "fg3m": row.get("fg3m", 0) or 0,
        }
    return result


def _sharpe(daily_returns: list[float]) -> float:
    """Annualised Sharpe ratio from daily returns. Returns 0 for < 2 observations."""
    if len(daily_returns) < 2:
        return 0.0
    n = len(daily_returns)
    mean = sum(daily_returns) / n
    variance = sum((r - mean) ** 2 for r in daily_returns) / (n - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    if std == 0.0:
        return 0.0
    return (mean / std) * math.sqrt(252)


def compute_clv_analysis(game_date: str, bets: list[dict]) -> float:
    """Compute average Closing Line Value for a set of bets.

    Returns average CLV in percentage points.  Returns 0.0 if no CLV data
    is available in nba_clv for any of the supplied bets.
    """
    if not bets:
        return 0.0

    player_ids = list({b["player_id"] for b in bets if b.get("player_id") is not None})
    if not player_ids:
        return 0.0

    placeholders = ",".join("?" for _ in player_ids)
    df = read_dataframe(
        f"SELECT clv_pct FROM nba_clv WHERE game_date = ? AND player_id IN ({placeholders})",
        (game_date, *player_ids),
    )
    if df.empty:
        return 0.0

    valid = df["clv_pct"].dropna()
    if valid.empty:
        return 0.0
    return float(valid.mean())


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------


def run_backtest(config: BacktestConfig) -> BacktestResult:
    """Walk-forward backtest over [start_date, end_date].

    For each date:
    1. Load pre-computed value bets from nba_materialized_value_view.
    2. Load actual results from nba_player_game_logs.
    3. Grade each bet and compute profit.
    4. Track daily P&L and running bankroll.
    """
    dates = _date_range(config.start_date, config.end_date)

    total_bets = 0
    wins = 0
    losses = 0
    pushes = 0
    total_profit = 0.0
    bankroll = config.initial_bankroll

    daily_rows: list[dict] = []
    bet_records: list[dict] = []

    for game_date in dates:
        bets_df = _load_value_bets_for_date(game_date, config)
        actuals = _load_actuals_for_date(game_date)

        daily_profit = 0.0
        day_bets = 0
        day_wins = 0
        day_losses = 0
        day_pushes = 0

        if not bets_df.empty:
            for row in bets_df.to_dict("records"):
                player_id_raw = row.get("player_id")
                if player_id_raw is None:
                    continue
                player_id = int(player_id_raw)
                market = str(row["market"])
                stat_col = _MARKET_STAT_COL.get(market)
                if stat_col is None:
                    continue

                player_actuals = actuals.get(player_id)
                if player_actuals is None:
                    continue

                actual = float(player_actuals.get(stat_col, 0) or 0)
                line = float(row["line"])
                side = str(row.get("side", "over"))
                kelly = float(row.get("kelly_fraction", 0.0) or 0.0)
                price = int(row.get("over_price", -110) or -110)
                tier = str(row.get("confidence_tier", "C") or "C")

                grade = _grade_bet(side, line, actual)
                profit = _profit_for_bet(grade, kelly, price)

                if grade == "win":
                    wins += 1
                    day_wins += 1
                elif grade == "loss":
                    losses += 1
                    day_losses += 1
                else:
                    pushes += 1
                    day_pushes += 1

                total_bets += 1
                day_bets += 1
                daily_profit += profit
                total_profit += profit

                bet_records.append(
                    {
                        "date": game_date,
                        "market": market,
                        "tier": tier,
                        "grade": grade,
                        "profit": profit,
                        "kelly": kelly,
                    }
                )

        bankroll = config.initial_bankroll + total_profit
        daily_rows.append(
            {
                "date": game_date,
                "pnl": daily_profit,
                "cumulative": total_profit,
                "bankroll": bankroll,
                "bets": day_bets,
                "wins": day_wins,
                "losses": day_losses,
                "pushes": day_pushes,
            }
        )

    daily_pnl = pd.DataFrame(daily_rows)
    if daily_pnl.empty:
        daily_pnl = pd.DataFrame(
            columns=["date", "pnl", "cumulative", "bankroll", "bets", "wins", "losses", "pushes"]
        )

    # Drawdown from bankroll series
    bankroll_series = (
        daily_pnl["bankroll"].tolist() if not daily_pnl.empty else [config.initial_bankroll]
    )
    max_dd = max_drawdown_from_bankroll(bankroll_series)

    # Sharpe from daily P&L
    daily_returns = daily_pnl["pnl"].tolist() if not daily_pnl.empty else []
    sharpe = _sharpe(daily_returns)

    # ROI
    roi_pct = (total_profit / total_bets * 100) if total_bets > 0 else 0.0

    # Win rate
    win_rate = wins / total_bets if total_bets > 0 else 0.0

    # CLV (best-effort across all dates with data)
    clv_avg = 0.0
    if bet_records:
        date_bets: dict[str, list[dict]] = {}
        for b in bet_records:
            date_bets.setdefault(b["date"], []).append(b)
        clv_values: list[float] = []
        for d, day_bets_list in date_bets.items():
            clv = compute_clv_analysis(d, day_bets_list)
            if clv != 0.0:
                clv_values.append(clv)
        clv_avg = sum(clv_values) / len(clv_values) if clv_values else 0.0

    # Per-market breakdown
    per_market = _build_per_market(bet_records)

    # Per-tier breakdown
    per_tier = _build_per_tier(bet_records)

    return BacktestResult(
        total_bets=total_bets,
        wins=wins,
        losses=losses,
        pushes=pushes,
        win_rate=win_rate,
        roi_pct=roi_pct,
        total_profit_units=total_profit,
        max_drawdown=max_dd,
        sharpe_ratio=sharpe,
        clv_avg=clv_avg,
        daily_pnl=daily_pnl,
        per_market=per_market,
        per_tier=per_tier,
    )


def _build_per_market(bet_records: list[dict]) -> pd.DataFrame:
    if not bet_records:
        return pd.DataFrame(columns=["market", "bets", "wins", "roi_pct"])

    df = pd.DataFrame(bet_records)
    grouped = (
        df.groupby("market")
        .agg(
            bets=("grade", "count"),
            wins=("grade", lambda x: (x == "win").sum()),
            total_profit=("profit", "sum"),
        )
        .reset_index()
    )
    grouped["roi_pct"] = grouped.apply(
        lambda r: (r["total_profit"] / r["bets"] * 100) if r["bets"] > 0 else 0.0, axis=1
    )
    return grouped[["market", "bets", "wins", "roi_pct"]]


def _build_per_tier(bet_records: list[dict]) -> pd.DataFrame:
    if not bet_records:
        return pd.DataFrame(columns=["tier", "bets", "wins", "roi_pct"])

    df = pd.DataFrame(bet_records)
    if "tier" not in df.columns:
        return pd.DataFrame(columns=["tier", "bets", "wins", "roi_pct"])

    grouped = (
        df.groupby("tier")
        .agg(
            bets=("grade", "count"),
            wins=("grade", lambda x: (x == "win").sum()),
            total_profit=("profit", "sum"),
        )
        .reset_index()
    )
    grouped["roi_pct"] = grouped.apply(
        lambda r: (r["total_profit"] / r["bets"] * 100) if r["bets"] > 0 else 0.0, axis=1
    )
    return grouped[["tier", "bets", "wins", "roi_pct"]]
