"""Run NBA walk-forward backtest over a date range.

Usage:
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python scripts/run_nba_backtest.py \
        --start-date 2025-11-01 --end-date 2026-02-28

Options:
    --start-date    Start date (YYYY-MM-DD, required)
    --end-date      End date (YYYY-MM-DD, required)
    --min-edge      Minimum edge threshold (default: 0.08)
    --markets       Comma-separated list of markets (default: pts,reb,ast,fg3m)
    --tiers         Comma-separated list of confidence tiers to include (default: all)
    --calibrated    Use calibrated probabilities
    --monte-carlo   Use Monte Carlo probabilities
    --output        Optional path to write JSON results
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.nba_backtest import BacktestConfig, run_backtest
from utils.db import executemany


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run NBA walk-forward backtest over a date range.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-date", required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--min-edge", type=float, default=0.08, dest="min_edge")
    parser.add_argument(
        "--markets",
        default="pts,reb,ast,fg3m",
        help="Comma-separated list of markets",
    )
    parser.add_argument(
        "--tiers",
        default=None,
        help="Comma-separated confidence tiers to include (e.g. A,B)",
    )
    parser.add_argument(
        "--calibrated",
        action="store_true",
        default=False,
        help="Use calibrated probabilities",
    )
    parser.add_argument(
        "--monte-carlo",
        action="store_true",
        default=False,
        dest="monte_carlo",
        help="Use Monte Carlo probabilities",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Optional JSON output path",
    )
    return parser


def _save_to_db(run_id: str, config: BacktestConfig, result) -> None:
    """Persist backtest run summary to nba_backtest_runs table."""
    config_dict = {
        "start_date": config.start_date,
        "end_date": config.end_date,
        "min_edge": config.min_edge,
        "markets": list(config.markets),
        "tiers": list(config.tiers) if config.tiers else None,
        "use_calibrated": config.use_calibrated,
        "use_monte_carlo": config.use_monte_carlo,
        "initial_bankroll": config.initial_bankroll,
    }

    results_dict = {
        "total_bets": result.total_bets,
        "wins": result.wins,
        "losses": result.losses,
        "pushes": result.pushes,
        "win_rate": result.win_rate,
        "roi_pct": result.roi_pct,
        "total_profit_units": result.total_profit_units,
        "max_drawdown": result.max_drawdown,
        "sharpe_ratio": result.sharpe_ratio,
        "clv_avg": result.clv_avg,
    }

    executemany(
        """
        INSERT OR IGNORE INTO nba_backtest_runs
        (run_id, start_date, end_date, config_json, results_json,
         total_bets, roi_pct, sharpe_ratio, max_drawdown, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                run_id,
                config.start_date,
                config.end_date,
                json.dumps(config_dict),
                json.dumps(results_dict),
                result.total_bets,
                result.roi_pct,
                result.sharpe_ratio,
                result.max_drawdown,
                datetime.now(timezone.utc).isoformat(),
            )
        ],
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    tiers = (
        [t.strip() for t in args.tiers.split(",") if t.strip()]
        if args.tiers
        else None
    )

    config = BacktestConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        min_edge=args.min_edge,
        markets=markets,
        tiers=tiers,
        use_calibrated=args.calibrated,
        use_monte_carlo=args.monte_carlo,
    )

    print(f"Running NBA backtest: {config.start_date} → {config.end_date}")
    print(f"  Markets : {', '.join(config.markets)}")
    print(f"  Min edge: {config.min_edge:.1%}")
    if config.tiers:
        print(f"  Tiers   : {', '.join(config.tiers)}")

    try:
        result = run_backtest(config)
    except Exception as exc:
        print(f"Backtest failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n=== Backtest Results ===")
    print(f"Total bets  : {result.total_bets}")
    print(f"Wins        : {result.wins}")
    print(f"Losses      : {result.losses}")
    print(f"Pushes      : {result.pushes}")
    if result.total_bets > 0:
        print(f"Win rate    : {result.win_rate:.1%}")
    print(f"ROI         : {result.roi_pct:.2f}%")
    print(f"Profit (u)  : {result.total_profit_units:.4f}")
    print(f"Max drawdown: {result.max_drawdown:.4f}")
    print(f"Sharpe ratio: {result.sharpe_ratio:.3f}")
    print(f"CLV avg     : {result.clv_avg:.4f}")

    if not result.per_market.empty:
        print("\nPer-market breakdown:")
        print(result.per_market.to_string(index=False))

    # ------------------------------------------------------------------
    # Save to DB
    # ------------------------------------------------------------------
    run_id = str(uuid.uuid4())
    try:
        _save_to_db(run_id, config, result)
        print(f"\nSaved to nba_backtest_runs (run_id={run_id})")
    except Exception as exc:
        print(f"Warning: could not save to DB: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Optional JSON output
    # ------------------------------------------------------------------
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "config": {
                "start_date": config.start_date,
                "end_date": config.end_date,
                "min_edge": config.min_edge,
                "markets": list(config.markets),
                "tiers": list(config.tiers) if config.tiers else None,
            },
            "results": {
                "total_bets": result.total_bets,
                "wins": result.wins,
                "losses": result.losses,
                "pushes": result.pushes,
                "win_rate": result.win_rate,
                "roi_pct": result.roi_pct,
                "total_profit_units": result.total_profit_units,
                "max_drawdown": result.max_drawdown,
                "sharpe_ratio": result.sharpe_ratio,
                "clv_avg": result.clv_avg,
            },
        }
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
