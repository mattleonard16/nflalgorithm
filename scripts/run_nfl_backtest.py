"""Run the NFL walk-forward backtest with the production weekly model.

For every requested week W the production model is retrained from scratch on
history strictly before W (the model's own loaders enforce the cutoff), then
asked to predict W; utils.nfl_backtest grades the predictions against actuals.

Two production-safety guarantees, enforced here:
- model artifacts are written to a temporary directory, never to the
  production ``models/weekly`` bundles;
- ``weekly_projections`` is never written — the persistence hook is replaced
  with a no-op, so stored pregame evidence for past weeks stays untouched.

The weekly model is proprietary and gitignored; this script fails with a clear
message where that module is absent (e.g. CI), and the harness it drives is
covered by tests/test_nfl_backtest.py with a stub model instead.

Usage:
    uv run python -m scripts.run_nfl_backtest run --season 2025 --weeks 5 6 7
    uv run python -m scripts.run_nfl_backtest compare baseline.json candidate.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import config
from utils.db import read_dataframe
from utils.nfl_backtest import (
    WalkForwardConfig,
    compare_walk_forward,
    run_walk_forward,
)
from utils.nfl_markets import MARKET_TO_STAT

DEFAULT_WEEKS = tuple(range(1, 19))


def _import_weekly_model():
    try:
        from models.position_specific import weekly
    except ImportError as exc:
        raise SystemExit(
            "The proprietary weekly model (models/position_specific/weekly.py) is "
            f"not available here: {exc}. The backtest runner needs it; the harness "
            "itself is tested in CI via tests/test_nfl_backtest.py."
        ) from exc
    return weekly


def _patch_for_backtest(weekly, model_dir: Path) -> None:
    """Redirect artifacts to model_dir and disable projection persistence."""
    for attribute in ("MODEL_DIR", "_write_predictions", "train_weekly_models", "predict_week"):
        if not hasattr(weekly, attribute):
            raise SystemExit(
                f"weekly model no longer exposes {attribute}; "
                "update scripts/run_nfl_backtest.py to match"
            )
    weekly.MODEL_DIR = model_dir

    def _no_write(*args, **kwargs) -> None:
        return None

    weekly._write_predictions = _no_write


def _training_tuples(season: int, week: int, history_seasons: int) -> list[tuple[int, int]]:
    """(season, week) pairs strictly before the target week, bounded by depth."""
    frame = read_dataframe(
        "SELECT DISTINCT season, week FROM player_stats_enhanced "
        "WHERE (season < ? OR (season = ? AND week < ?)) AND season >= ? "
        "ORDER BY season, week",
        (season, season, week, season - history_seasons),
    )
    return [(int(row.season), int(row.week)) for row in frame.itertuples(index=False)]


def _load_actuals(season: int, weeks: tuple[int, ...]) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in weeks)
    stat_columns = ", ".join(sorted(set(MARKET_TO_STAT.values())))
    return read_dataframe(
        f"SELECT season, week, player_id, position, {stat_columns} "
        f"FROM player_stats_enhanced WHERE season = ? AND week IN ({placeholders})",
        (season, *weeks),
    )


def _make_predict_fn(weekly, history_seasons: int):
    def predict_fn(season: int, week: int) -> pd.DataFrame:
        tuples = _training_tuples(season, week, history_seasons)
        if not tuples:
            print(f"week {week}: no training history before cutoff; skipping")
            return pd.DataFrame()
        print(f"week {week}: training on {len(tuples)} season-week pairs "
              f"({tuples[0]} .. {tuples[-1]})")
        weekly.train_weekly_models(tuples)
        return weekly.predict_week(season, week, roster_backed=False)

    return predict_fn


def _print_summary(report: dict) -> None:
    print(f"\n=== walk-forward {report['label']} season {report['season']} ===")
    print(f"weeks evaluated: {report['weeks_evaluated']}")
    overall = report["overall"]
    line = (
        f"overall: n={overall['count']} mae={overall['mae']:.2f} "
        f"bias={overall['mean_bias']:+.2f}"
    )
    if "coverage_1sigma" in overall:
        line += f" cover1s={overall['coverage_1sigma']:.1%} z_std={overall['z_std']:.2f}"
    print(line)
    for market, group in sorted(report["by_market"].items()):
        line = f"  {market}: n={group['count']} mae={group['mae']:.2f} bias={group['mean_bias']:+.2f}"
        if "coverage_1sigma" in group:
            line += f" cover1s={group['coverage_1sigma']:.1%}"
        if group["small_sample"]:
            line += " [small sample]"
        print(line)
    for problem in report["problems"]:
        print(f"  problem: {problem}")


def _run(args: argparse.Namespace) -> dict:
    weeks = tuple(sorted(set(args.weeks)))
    if any(week < 1 for week in weeks):
        raise SystemExit("weeks must be positive")

    actuals = _load_actuals(args.season, weeks)
    if actuals.empty:
        raise SystemExit(
            f"No actuals in player_stats_enhanced for season {args.season} weeks {list(weeks)}"
        )

    weekly = _import_weekly_model()
    with tempfile.TemporaryDirectory(prefix="nfl_backtest_models_") as tmp:
        _patch_for_backtest(weekly, Path(tmp))
        result = run_walk_forward(
            _make_predict_fn(weekly, args.history_seasons),
            actuals,
            WalkForwardConfig(
                season=args.season,
                weeks=weeks,
                label=args.label,
                min_week_rows=args.min_week_rows,
            ),
        )
    if args.rows_output is not None:
        args.rows_output.parent.mkdir(parents=True, exist_ok=True)
        result.evaluated.to_csv(args.rows_output, index=False)
        print(f"scored rows written to {args.rows_output}")
    report = dict(result.report)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["history_seasons"] = args.history_seasons
    return report


def _write_report(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nreport written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="retrain-per-week walk-forward backtest")
    run.add_argument("--season", type=int, required=True)
    run.add_argument("--weeks", type=int, nargs="+", default=list(DEFAULT_WEEKS))
    run.add_argument("--label", default="baseline")
    run.add_argument(
        "--history-seasons",
        type=int,
        default=2,
        help="how many seasons before --season may contribute training data",
    )
    run.add_argument("--min-week-rows", type=int, default=20)
    run.add_argument("--output", type=Path, default=None)
    run.add_argument(
        "--rows-output",
        type=Path,
        default=None,
        help="also write the per-row scored frame as CSV (for calibration analysis)",
    )

    compare = subparsers.add_parser("compare", help="compare two backtest reports")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    if args.command == "run":
        report = _run(args)
        _print_summary(report)
        output = args.output or (
            config.reports_dir / f"nfl_backtest_{args.season}_{args.label}.json"
        )
        _write_report(report, output)
        return

    comparison = compare_walk_forward(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")),
    )
    print(json.dumps(comparison, indent=2, default=str))
    if args.output is not None:
        _write_report(comparison, args.output)
    if not comparison["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
