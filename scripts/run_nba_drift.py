"""Run NBA drift detection checks.

Computes PSI-based prediction drift for all markets and saves alerts
to the nba_drift_alerts table.

Usage:
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python scripts/run_nba_drift.py
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python scripts/run_nba_drift.py --date 2026-03-04
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.nba_drift_detector import run_drift_checks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NBA prediction drift detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--date",
        default=str(date.today()),
        help="Date for drift check context (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--markets",
        default=None,
        help="Comma-separated markets to check (default: all)",
    )
    args = parser.parse_args()

    markets = (
        [m.strip() for m in args.markets.split(",") if m.strip()]
        if args.markets
        else None
    )

    print(f"Running NBA drift detection for date={args.date}")
    if markets:
        print(f"  Markets: {', '.join(markets)}")

    checks = run_drift_checks(args.date, markets=markets)

    print(f"\n{'='*60}")
    print(f"  Drift Detection Results")
    print(f"{'='*60}")

    for check in checks:
        market = check["market"]
        psi = check.get("psi", 0.0)
        level = check["alert_level"]
        n_recent = check.get("n_recent", 0)
        n_ref = check.get("n_reference", 0)

        level_icon = {"stable": "OK", "monitor": "!!", "alert": "XX"}
        icon = level_icon.get(level, "??")

        print(f"  [{icon}] {market:<6s}  PSI={psi:.4f}  level={level:<8s}  "
              f"recent={n_recent}  reference={n_ref}")

        if "explanation" in check:
            print(f"         {check['explanation']}")

    alert_count = sum(1 for c in checks if c["alert_level"] == "alert")
    monitor_count = sum(1 for c in checks if c["alert_level"] == "monitor")
    stable_count = sum(1 for c in checks if c["alert_level"] == "stable")

    print(f"\nSummary: {len(checks)} checks — "
          f"{stable_count} stable, {monitor_count} monitor, {alert_count} alert")
    print(f"Results saved to nba_drift_alerts (date={args.date})")


if __name__ == "__main__":
    main()
