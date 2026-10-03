"""Run the NFL walk-forward backtest with the production weekly model.

For every requested week W the production model is retrained from scratch on
history strictly before W (the model's own loaders enforce the cutoff), then
asked to predict W; utils.nfl_backtest grades the predictions against actuals.

Two production-safety guarantees, enforced here:
- model artifacts are written to a temporary directory, never to the
  production ``models/weekly`` bundles;
- ``weekly_projections`` is never written — the persistence hook is replaced
  with a no-op, so stored pregame evidence for past weeks stays untouched.

The ``replay`` command predicts through the roster path instead, the one
production uses. For each week it rewrites that season's roster and the week's
context snapshot as they stood at the week's first kickoff, so it refuses to
run against anything but a scratch copy of the database.

The weekly model is proprietary and gitignored; this script fails with a clear
message where that module is absent (e.g. CI), and the harness it drives is
covered by tests/test_nfl_backtest.py with a stub model instead.

Usage:
    uv run python -m scripts.run_nfl_backtest run --season 2025 --weeks 5 6 7
    uv run python -m scripts.run_nfl_backtest run --season 2025 --context-factors on \
        --label ctx --output ctx.json
    uv run python -m scripts.run_nfl_backtest replay --season 2025 --database scratch.db
    uv run python -m scripts.run_nfl_backtest compare baseline.json candidate.json

Feature flags the private model reads from ``config.features`` can be pinned
per run (``--context-factors on|off``); the report records what was in effect
under ``features`` so two runs can be told apart after the fact.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from config import config
from utils.db import execute, get_backend, read_dataframe
from utils.nfl_backtest import (
    WalkForwardConfig,
    compare_walk_forward,
    feature_overrides,
    run_walk_forward,
)
from utils.nfl_markets import DATABASE_STAT_COLUMNS
from utils.nfl_replay import (
    first_kickoff_cutoffs,
    players_without_stats,
    rekey_to_stat_ids,
    require_scratch_database,
    roster_for_week,
)

DEFAULT_WEEKS = tuple(range(1, 19))


def _import_weekly_model():
    # The private weekly.py when installed, the public baseline otherwise.
    from models.position_specific import weekly_implementation

    return weekly_implementation()


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
    stat_columns = ", ".join(DATABASE_STAT_COLUMNS)
    return read_dataframe(
        f"SELECT season, week, player_id, position, {stat_columns} "
        f"FROM player_stats_enhanced WHERE season = ? AND week IN ({placeholders})",
        (season, *weeks),
    )


def _train_before(weekly, season: int, week: int, history_seasons: int) -> bool:
    tuples = _training_tuples(season, week, history_seasons)
    if not tuples:
        print(f"week {week}: no training history before cutoff; skipping", flush=True)
        return False
    print(
        f"week {week}: training on {len(tuples)} season-week pairs "
        f"({tuples[0]} .. {tuples[-1]})",
        flush=True,
    )
    weekly.train_weekly_models(tuples)
    return True


def _make_predict_fn(weekly, history_seasons: int):
    def predict_fn(season: int, week: int) -> pd.DataFrame:
        if not _train_before(weekly, season, week, history_seasons):
            return pd.DataFrame()
        return weekly.predict_week(season, week, roster_backed=False)

    return predict_fn


class ReplayInputs(NamedTuple):
    rosters: pd.DataFrame
    depth_charts: pd.DataFrame
    injuries: pd.DataFrame


def _fetch_replay_inputs(season: int) -> ReplayInputs:
    from scripts.ingest_real_nfl_data import (
        fetch_depth_charts,
        fetch_injuries,
        fetch_weekly_rosters,
    )

    return ReplayInputs(
        fetch_weekly_rosters([season]), fetch_depth_charts([season]), fetch_injuries([season])
    )


def _stage_replay_week(season: int, week: int, inputs: ReplayInputs) -> None:
    """Write the week's roster and first-kickoff snapshot into the scratch database."""
    from scripts.ingest_real_nfl_data import (
        build_player_context_snapshots,
        load_snapshot_history,
        upsert_player_context_snapshots,
        upsert_roster_players,
    )

    games = read_dataframe(
        "SELECT season, week, home_team, away_team, spread_line, kickoff_utc "
        "FROM games WHERE season = ? AND week = ?",
        (season, week),
    )
    cutoffs = first_kickoff_cutoffs(games, season=season, week=week)
    roster = roster_for_week(inputs.rosters, season=season, week=week)
    execute("DELETE FROM nfl_roster_players WHERE season = ?", (season,))
    upsert_roster_players(roster)
    snapshots = build_player_context_snapshots(
        roster,
        inputs.depth_charts,
        inputs.injuries,
        load_snapshot_history(),
        target_week=week,
        target_cutoffs=cutoffs,
        captured_at=cutoffs[season],
        schedule=games,
    )
    execute(
        "DELETE FROM nfl_player_context_snapshots WHERE season = ? AND week = ?", (season, week)
    )
    upsert_player_context_snapshots(snapshots)


def _make_replay_predict_fn(
    weekly,
    history_seasons: int,
    inputs: ReplayInputs,
    actuals: pd.DataFrame,
    without_stats: dict[int, int],
):
    def predict_fn(season: int, week: int) -> pd.DataFrame:
        _stage_replay_week(season, week, inputs)
        if not _train_before(weekly, season, week, history_seasons):
            return pd.DataFrame()
        id_map = read_dataframe(
            "SELECT r.player_id AS roster_id, s.player_id AS stat_id "
            "FROM nfl_roster_players r JOIN player_stats_enhanced s "
            "ON s.gsis_id = r.gsis_id AND s.season = r.season AND s.week = ? "
            "WHERE r.season = ?",
            (week, season),
        )
        predictions = rekey_to_stat_ids(
            weekly.predict_week(season, week, roster_backed=True), id_map
        )
        without_stats[week] = players_without_stats(predictions, actuals, season=season, week=week)
        return predictions

    return predict_fn


def _use_scratch_database(path: Path) -> Path:
    if get_backend() != "sqlite":
        raise SystemExit("The replay runs on a SQLite scratch copy only")
    try:
        scratch = require_scratch_database(path, Path(config.database.path))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    os.environ["SQLITE_DB_PATH"] = str(scratch)
    config.database.path = str(scratch)
    return scratch


def _print_summary(report: dict) -> None:
    print(
        f"\n=== {report.get('mode', 'walk-forward')} {report['label']} season {report['season']} ==="
    )
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
        line = (
            f"  {market}: n={group['count']} mae={group['mae']:.2f} bias={group['mean_bias']:+.2f}"
        )
        if "coverage_1sigma" in group:
            line += f" cover1s={group['coverage_1sigma']:.1%}"
        if group["small_sample"]:
            line += " [small sample]"
        print(line)
    for problem in report["problems"]:
        print(f"  problem: {problem}")
    if "players_without_stats" in report:
        print(f"players projected with no stat line: {report['players_without_stats']['total']}")


# Feature flags a run may pin. Each maps a CLI name to the config.features
# attribute the private model reads, so "on"/"off" here is exactly what the
# weekly cron's NFL_FEATURE_* environment would have set.
FEATURE_FLAGS = {
    "context_factors": "context_factors_enabled",
}


def _feature_overrides(args: argparse.Namespace) -> dict[str, bool]:
    overrides: dict[str, bool] = {}
    for flag, attribute in FEATURE_FLAGS.items():
        choice = getattr(args, flag, None)
        if choice in ("on", "off"):
            overrides[attribute] = choice == "on"
    return overrides


def _run(args: argparse.Namespace) -> dict:
    weeks = tuple(sorted(set(args.weeks)))
    if any(week < 1 for week in weeks):
        raise SystemExit("weeks must be positive")
    replay = args.command == "replay"
    if replay:
        print(f"replaying against {_use_scratch_database(args.database)}", flush=True)

    actuals = _load_actuals(args.season, weeks)
    if actuals.empty:
        raise SystemExit(
            f"No actuals in player_stats_enhanced for season {args.season} weeks {list(weeks)}"
        )

    weekly = _import_weekly_model()
    overrides = _feature_overrides(args)
    with (
        tempfile.TemporaryDirectory(prefix="nfl_backtest_models_") as tmp,
        feature_overrides(config.features, **overrides) as features,
    ):
        _patch_for_backtest(weekly, Path(tmp))
        print(f"features in effect: {features}", flush=True)
        without_stats: dict[int, int] = {}
        predict_fn = (
            _make_replay_predict_fn(
                weekly,
                args.history_seasons,
                _fetch_replay_inputs(args.season),
                actuals,
                without_stats,
            )
            if replay
            else _make_predict_fn(weekly, args.history_seasons)
        )
        result = run_walk_forward(
            predict_fn,
            actuals,
            WalkForwardConfig(
                season=args.season,
                weeks=weeks,
                label=args.label,
                min_week_rows=args.min_week_rows,
                features=features,
            ),
        )
    if args.rows_output is not None:
        args.rows_output.parent.mkdir(parents=True, exist_ok=True)
        result.evaluated.to_csv(args.rows_output, index=False)
        print(f"scored rows written to {args.rows_output}")
    report = dict(result.report)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["history_seasons"] = args.history_seasons
    if replay:
        report["mode"] = "replay"
        report["players_without_stats"] = {
            "total": sum(without_stats.values()),
            "by_week": without_stats,
        }
    return report


def _write_report(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nreport written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--season", type=int, required=True)
    common.add_argument("--weeks", type=int, nargs="+", default=list(DEFAULT_WEEKS))
    common.add_argument("--label", default="baseline")
    common.add_argument(
        "--history-seasons",
        type=int,
        default=2,
        help="how many seasons before --season may contribute training data",
    )
    common.add_argument("--min-week-rows", type=int, default=20)
    common.add_argument(
        "--context-factors",
        choices=("on", "off", "inherit"),
        default="inherit",
        help=(
            "pin config.features.context_factors_enabled for this run; 'inherit' keeps "
            "whatever NFL_FEATURE_CONTEXT_FACTORS resolved to (default off). Run once each "
            "way with the same --season/--weeks, then `compare` the two reports."
        ),
    )
    common.add_argument("--output", type=Path, default=None)
    common.add_argument(
        "--rows-output",
        type=Path,
        default=None,
        help="also write the per-row scored frame as CSV (for calibration analysis)",
    )
    subparsers.add_parser("run", parents=[common], help="retrain-per-week walk-forward backtest")
    replay = subparsers.add_parser(
        "replay", parents=[common], help="retrain per week and predict through the roster path"
    )
    replay.add_argument(
        "--database",
        type=Path,
        required=True,
        help="scratch copy of nfl_data.db; the replay rewrites its rosters and snapshots",
    )

    compare = subparsers.add_parser("compare", help="compare two backtest reports")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    if args.command in ("run", "replay"):
        report = _run(args)
        _print_summary(report)
        kind = "replay" if args.command == "replay" else "backtest"
        output = args.output or (config.reports_dir / f"nfl_{kind}_{args.season}_{args.label}.json")
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
