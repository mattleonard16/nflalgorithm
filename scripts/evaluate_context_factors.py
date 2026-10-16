"""Measure utils.context_factors on core players against a transparent baseline.

Punch-list item 31 asks whether the context multipliers improve projections.
The production model is private, so this answers the part a public checkout
can: against a baseline built only from each player's own prior games, does
multiplying by each component lower the error on the players who actually
carry prop lines?

Core player: a starter by trailing usage (last five played games) -- QBs on
passing attempts, RBs on carries, WRs and TEs on targets. Baseline mu: an
exponentially weighted mean of the player's prior games, keyed by ``gsis_id``
so a trade does not reset his history. Each component is scored on its own
and as the composite the projection path applies, with a bootstrap interval
over season-weeks (games in one week share a slate, so weeks are the unit).

Usage:
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python -m \\
        scripts.evaluate_context_factors --seasons 2024,2025 [--halflife 4]

A baseline with more recency is the stricter test for the usage trend: it is
the case the real model is in. Run a few half-lives before concluding.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from utils.context_factors import (
    DEFAULT_PARAMS,
    MARKET_OPPORTUNITY_COLUMN,
    ContextParams,
    compute_context_factors,
)

COMPONENTS = (
    "composite_factor",
    "game_script_factor",
    "matchup_history_factor",
    "usage_trend_factor",
)

# Minimum trailing opportunity that marks a starter, per (market, position).
CORE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "passing_yards": {"QB": 25.0},
    "rushing_yards": {"RB": 12.0},
    "receiving_yards": {"WR": 6.0, "TE": 4.0},
}

STAT_COLUMNS = (
    "player_id",
    "gsis_id",
    "season",
    "week",
    "team",
    "position",
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "passing_attempts",
    "rushing_attempts",
    "targets",
)


@dataclass(frozen=True)
class EvalConfig:
    halflife: float = 4.0
    min_prior_games: int = 3
    history_games: int = 16
    trend_window: int = 5
    first_week: int = 3
    last_week: int = 18
    bootstrap_draws: int = 1000
    seed: int = 0


def _baseline(
    prior: pd.DataFrame, market: str, opportunity: str, config: EvalConfig
) -> pd.DataFrame:
    """Per-player EWMA mu over prior games plus trailing opportunity."""
    records = []
    for gsis_id, games in prior.groupby("gsis_id", sort=False):
        games = games.tail(config.history_games)
        if len(games) < config.min_prior_games:
            continue
        values = pd.to_numeric(games[market], errors="coerce").fillna(0.0).to_numpy()
        weights = 0.5 ** (np.arange(len(values))[::-1] / config.halflife)
        mu = float(np.dot(weights, values) / weights.sum())
        trailing = float(
            pd.to_numeric(games[opportunity], errors="coerce")
            .fillna(0.0)
            .tail(config.trend_window)
            .mean()
        )
        records.append((gsis_id, mu, trailing))
    return pd.DataFrame(records, columns=["gsis_id", "mu", "trailing_opportunity"])


def build_rows(
    stats: pd.DataFrame,
    games: pd.DataFrame,
    seasons: Iterable[int],
    *,
    config: EvalConfig = EvalConfig(),
    params: ContextParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    """One row per core player-game with baseline mu, actual and every factor."""
    stats = stats[stats["gsis_id"].notna()].copy()
    stats["order"] = stats["season"].astype(int) * 100 + stats["week"].astype(int)
    stats = stats.sort_values(["gsis_id", "order"], kind="stable")

    frames: List[pd.DataFrame] = []
    for season in seasons:
        for week in range(config.first_week, config.last_week + 1):
            target = stats[(stats["season"] == season) & (stats["week"] == week)]
            if target.empty:
                continue
            prior = stats[stats["order"] < season * 100 + week]
            for market, thresholds in CORE_THRESHOLDS.items():
                opportunity = MARKET_OPPORTUNITY_COLUMN[market]
                candidates = target[target["position"].isin(thresholds)]
                if candidates.empty:
                    continue
                base = _baseline(
                    prior[prior["gsis_id"].isin(candidates["gsis_id"])],
                    market,
                    opportunity,
                    config,
                )
                if base.empty:
                    continue
                core = candidates.merge(base, on="gsis_id")
                floor = core["position"].map(thresholds)
                core = core[core["trailing_opportunity"] >= floor]
                core = core.drop_duplicates("player_id")
                if core.empty:
                    continue
                factors = compute_context_factors(
                    season=season,
                    week=week,
                    market=market,
                    players=core[["player_id", "team", "position"]],
                    games=games,
                    history=stats,
                    params=params,
                )
                core = core.merge(
                    factors[["player_id", *COMPONENTS]], on="player_id", how="left"
                )
                core["market"] = market
                core["actual"] = pd.to_numeric(core[market], errors="coerce")
                frames.append(core)

    if not frames:
        return pd.DataFrame(columns=["season", "week", "market", "mu", "actual", *COMPONENTS])
    rows = pd.concat(frames, ignore_index=True)
    rows = rows.dropna(subset=["actual"])
    for column in COMPONENTS:
        rows[column] = rows[column].astype(float).fillna(1.0)
    return rows


def _bootstrap_delta(
    rows: pd.DataFrame, component: str, rng: np.random.Generator, draws: int
) -> Dict[str, float]:
    adjusted = (rows["mu"] * rows[component] - rows["actual"]).abs()
    baseline = (rows["mu"] - rows["actual"]).abs()
    delta = adjusted - baseline
    slate = rows["season"].astype(int) * 100 + rows["week"].astype(int)
    by_week = delta.groupby(slate).agg(["sum", "count"])
    keys = by_week.index.to_numpy()
    samples = []
    for _ in range(draws):
        pick = by_week.loc[rng.choice(keys, len(keys))]
        samples.append(pick["sum"].sum() / pick["count"].sum())
    lo, hi = np.percentile(samples, [2.5, 97.5]) if samples else (np.nan, np.nan)
    return {
        "n": int(len(rows)),
        "baseline_mae": float(baseline.mean()),
        "mae": float(adjusted.mean()),
        "delta_mae": float(delta.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "bias": float((rows["mu"] * rows[component] - rows["actual"]).mean()),
    }


def summarize(rows: pd.DataFrame, *, config: EvalConfig = EvalConfig()) -> Dict[str, Dict]:
    """``{component: {market|ALL: metrics}}``; a negative delta_mae helps."""
    rng = np.random.default_rng(config.seed)
    report: Dict[str, Dict] = {}
    groups = [(market, frame) for market, frame in rows.groupby("market")]
    groups.append(("ALL", rows))
    for component in COMPONENTS:
        report[component] = {
            market: _bootstrap_delta(frame, component, rng, config.bootstrap_draws)
            for market, frame in groups
            if not frame.empty
        }
    return report


def _format(report: Dict[str, Dict]) -> str:
    lines = []
    for component, markets in report.items():
        lines.append(component)
        for market, m in markets.items():
            lines.append(
                f"  {market:16s} n={m['n']:5d}  MAE {m['baseline_mae']:7.3f} -> {m['mae']:7.3f}"
                f"  delta {m['delta_mae']:+.3f}  95% CI [{m['ci_low']:+.3f}, {m['ci_high']:+.3f}]"
            )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    from utils.db import read_dataframe

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seasons", default="2024,2025", help="comma-separated seasons")
    parser.add_argument("--halflife", type=float, default=EvalConfig.halflife)
    parser.add_argument("--json", dest="json_path", help="also write the report as JSON")
    parser.add_argument("--rows-output", help="write the per-row frame as CSV")
    args = parser.parse_args(argv)

    seasons = [int(value) for value in args.seasons.split(",") if value.strip()]
    config = EvalConfig(halflife=args.halflife)
    first = min(seasons) - DEFAULT_PARAMS.history_seasons
    stats = read_dataframe(
        f"SELECT {', '.join(STAT_COLUMNS)} FROM player_stats_enhanced WHERE season >= ?",
        (first,),
    )
    games = read_dataframe(
        "SELECT season, week, home_team, away_team, spread_line, total_line "
        "FROM games WHERE season >= ?",
        (first,),
    )
    rows = build_rows(stats, games, seasons, config=config)
    if rows.empty:
        print("No core-player rows: ingest history for these seasons first.")
        return 1
    report = summarize(rows, config=config)
    print(f"Core-player context factor evaluation, seasons {seasons}, halflife {config.halflife}")
    print(_format(report))
    if args.json_path:
        with open(args.json_path, "w") as handle:
            json.dump({"seasons": seasons, "halflife": config.halflife, "report": report}, handle, indent=2)
    if args.rows_output:
        rows.to_csv(args.rows_output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
