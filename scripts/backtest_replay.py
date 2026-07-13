"""Replay stored NFL pregame snapshots against actual outcomes.

This evaluator deliberately does not call the live value engine or place bets.
It reconstructs what was knowable before kickoff from stored projections,
roster-role snapshots, and timestamped odds, then grades those decisions using
actual player stats.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

from config import config
from utils.db import read_dataframe
from utils.grading import calculate_profit_units, grade_bet
from utils.nfl_markets import melt_actuals
from value_betting_engine import market_implied_probabilities, prob_over

MAX_PROJECTION_AGE = pd.Timedelta(days=7)
MAX_CONTEXT_AGE = pd.Timedelta(days=7)
MAX_ENTRY_ODDS_AGE = pd.Timedelta(hours=48)
PLAYER_WEEK_KEYS = ("season", "week", "player_id")
PROJECTION_KEYS = (*PLAYER_WEEK_KEYS, "market")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay NFL predictions against outcomes")
    parser.add_argument("--season", type=int, required=True, help="Season to replay")
    parser.add_argument("--weeks", nargs="+", type=int, required=True, help="Weeks to replay")
    parser.add_argument("--min-edge", type=float, default=0.05, help="Minimum model edge")
    return parser.parse_args()


def _timestamps(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _team_kickoffs(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(columns=["season", "week", "team", "kickoff_utc"])
    home = games[["season", "week", "home_team", "kickoff_utc"]].rename(
        columns={"home_team": "team"}
    )
    away = games[["season", "week", "away_team", "kickoff_utc"]].rename(
        columns={"away_team": "team"}
    )
    return pd.concat([home, away], ignore_index=True).drop_duplicates(
        ["season", "week", "team"], keep="last"
    )


def _role(row: pd.Series) -> str:
    starter = row.get("is_starter")
    rookie = row.get("is_rookie")
    if pd.notna(starter) and int(starter) == 1:
        return "starter"
    if pd.notna(rookie) and int(rookie) == 1:
        return "rookie_depth"
    rank = row.get("depth_rank")
    if pd.notna(rank):
        return f"depth_{int(rank)}"
    position = row.get("position")
    return f"{str(position).lower()}_unranked" if pd.notna(position) else "unknown"


def _freshness_failures(row: pd.Series) -> list[str]:
    failures: list[str] = []
    kickoff = row["kickoff_utc"]
    generated = row["generated_at"]
    captured = row["captured_at"]
    source_updated = row["source_updated_at"]
    entry_as_of = row["entry_as_of"]

    if pd.isna(kickoff):
        return ["missing_kickoff"]
    if pd.isna(generated):
        failures.append("missing_projection_timestamp")
    elif generated >= kickoff:
        failures.append("projection_after_kickoff")
    elif kickoff - generated > MAX_PROJECTION_AGE:
        failures.append("stale_projection")
    if pd.isna(captured):
        failures.append("missing_context_snapshot")
    elif captured >= kickoff:
        failures.append("context_after_kickoff")
    elif kickoff - captured > MAX_CONTEXT_AGE:
        failures.append("stale_context")
    if pd.isna(source_updated):
        failures.append("missing_context_source_timestamp")
    elif source_updated >= kickoff:
        failures.append("context_source_after_kickoff")
    elif kickoff - source_updated > MAX_CONTEXT_AGE:
        failures.append("stale_context_source")
    if pd.isna(entry_as_of):
        failures.append("missing_pregame_odds")
    elif entry_as_of >= kickoff:
        failures.append("odds_after_kickoff")
    elif kickoff - entry_as_of > MAX_ENTRY_ODDS_AGE:
        failures.append("stale_odds")
    elif pd.notna(generated) and entry_as_of < generated:
        failures.append("odds_before_projection")
    return failures


def build_replay_dataset(
    projections: pd.DataFrame,
    odds: pd.DataFrame,
    actuals: pd.DataFrame,
    context: pd.DataFrame,
    games: pd.DataFrame,
    *,
    min_edge: float = 0.05,
) -> pd.DataFrame:
    """Build one replay row per projection and sportsbook using pregame data only."""
    if projections.empty:
        return pd.DataFrame()

    keys = list(PLAYER_WEEK_KEYS)
    base = projections.copy()
    base["generated_at"] = _timestamps(base["generated_at"])
    context_cols = keys + [
        "position",
        "is_starter",
        "depth_rank",
        "is_rookie",
        "is_new_team",
        "source_updated_at",
        "captured_at",
    ]
    if set(keys).issubset(context.columns):
        available_context = [column for column in context_cols if column in context.columns]
        base = base.merge(context[available_context], on=keys, how="left")
    else:
        for column in context_cols[len(keys) :]:
            base[column] = pd.NA
    if "captured_at" not in base:
        base["captured_at"] = pd.NaT
    base["captured_at"] = _timestamps(base["captured_at"])
    if "source_updated_at" not in base:
        base["source_updated_at"] = pd.NaT
    base["source_updated_at"] = _timestamps(base["source_updated_at"])
    base = base.merge(_team_kickoffs(games), on=["season", "week", "team"], how="left")
    base["kickoff_utc"] = _timestamps(base["kickoff_utc"])
    base = base.merge(melt_actuals(actuals), on=keys + ["market"], how="left")

    odds_frame = odds.copy()
    if "under_price" not in odds_frame:
        odds_frame["under_price"] = pd.NA
    odds_frame["as_of"] = _timestamps(odds_frame["as_of"])
    replay_rows: list[dict[str, object]] = []
    odds_keys = list(PROJECTION_KEYS)
    odds_groups = {
        key if isinstance(key, tuple) else (key,): group
        for key, group in odds_frame.groupby(odds_keys, sort=False)
    }

    for _, projection in base.iterrows():
        matching = odds_groups.get(
            tuple(projection[key] for key in odds_keys), odds_frame.iloc[0:0]
        )
        books: list[object] = matching["sportsbook"].dropna().unique().tolist() or [None]

        for book in books:
            book_odds = matching[matching["sportsbook"] == book].sort_values("as_of")
            kickoff = projection["kickoff_utc"]
            pregame = (
                book_odds[book_odds["as_of"] < kickoff]
                if pd.notna(kickoff)
                else book_odds.iloc[0:0]
            )
            generated = projection["generated_at"]
            available = pregame[pregame["as_of"] >= generated] if pd.notna(generated) else pregame
            entry = (
                available.iloc[0]
                if not available.empty
                else (pregame.iloc[0] if not pregame.empty else pd.Series(dtype=object))
            )
            close = pregame.iloc[-1] if not pregame.empty else pd.Series(dtype=object)

            row = projection.to_dict()
            row.update(
                sportsbook=book,
                entry_line=entry.get("line", np.nan),
                entry_price=entry.get("price", np.nan),
                entry_under_price=entry.get("under_price", np.nan),
                entry_as_of=entry.get("as_of", pd.NaT),
                close_line=close.get("line", np.nan),
                close_price=close.get("price", np.nan),
                close_as_of=close.get("as_of", pd.NaT),
            )
            replay_rows.append(row)

    replay = pd.DataFrame(replay_rows)
    replay["role"] = replay.apply(_role, axis=1)
    failure_lists = replay.apply(_freshness_failures, axis=1)
    replay["freshness_failures"] = failure_lists.apply(lambda values: ";".join(values))
    replay["freshness_pass"] = failure_lists.apply(lambda values: not values)

    def score(row: pd.Series) -> pd.Series:
        if pd.isna(row["entry_line"]) or pd.isna(row["entry_price"]):
            return pd.Series(
                {"side": None, "p_win": np.nan, "edge": np.nan, "recommendation": False}
            )
        p_over = prob_over(
            float(row["mu"]), max(float(row["sigma"]), 0.01), float(row["entry_line"])
        )
        implied_over, implied_under = market_implied_probabilities(
            int(row["entry_price"]), row["entry_under_price"]
        )
        over_edge = p_over - implied_over
        under_edge = (1.0 - p_over) - implied_under
        side = "over" if over_edge >= under_edge else "under"
        p_win = p_over if side == "over" else 1.0 - p_over
        edge = over_edge if side == "over" else under_edge
        return pd.Series(
            {"side": side, "p_win": p_win, "edge": edge, "recommendation": edge >= min_edge}
        )

    replay = pd.concat([replay, replay.apply(score, axis=1)], axis=1)
    replay["price"] = replay.apply(
        lambda row: (
            row["entry_under_price"]
            if row["side"] == "under" and pd.notna(row["entry_under_price"])
            else row["entry_price"]
        ),
        axis=1,
    )
    replay["result"] = replay.apply(
        lambda row: (
            grade_bet(float(row["actual"]), float(row["entry_line"]), row["side"])
            if pd.notna(row["actual"]) and row["side"] in {"over", "under"}
            else "ungraded"
        ),
        axis=1,
    )
    replay["profit_units"] = replay.apply(
        lambda row: (
            calculate_profit_units(row["result"], int(row["price"]))
            if row["result"] != "ungraded" and pd.notna(row["price"])
            else np.nan
        ),
        axis=1,
    )
    replay["outcome"] = replay["result"].map({"win": 1.0, "loss": 0.0})
    replay["abs_error"] = (replay["mu"] - replay["actual"]).abs()
    replay["brier"] = (replay["p_win"] - replay["outcome"]) ** 2
    replay["clv_line"] = np.where(
        replay["side"] == "over",
        replay["close_line"] - replay["entry_line"],
        replay["entry_line"] - replay["close_line"],
    )
    return replay


def _calibration(rows: pd.DataFrame) -> list[dict[str, object]]:
    graded = rows.dropna(subset=["p_win", "outcome"]).copy()
    if graded.empty:
        return []
    graded["probability_bin"] = pd.cut(
        graded["p_win"], bins=np.linspace(0, 1, 11), include_lowest=True
    )
    result: list[dict[str, object]] = []
    for interval, group in graded.groupby("probability_bin", observed=True):
        result.append(
            {
                "bin": str(interval),
                "bets": int(len(group)),
                "mean_predicted_probability": float(group["p_win"].mean()),
                "observed_win_rate": float(group["outcome"].mean()),
            }
        )
    return result


def _group_metrics(rows: pd.DataFrame, column: str) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for value, group in rows.groupby(column, dropna=False):
        projection_rows = group.dropna(subset=["actual"]).drop_duplicates(list(PROJECTION_KEYS))
        evaluated = group[group["result"].isin(["win", "loss"])]
        bets = group[group["recommendation"] & group["result"].isin(["win", "loss", "push"])]
        output[str(value)] = {
            "projections": int(len(projection_rows)),
            "bets": int(len(bets)),
            "mae": float(projection_rows["abs_error"].mean()) if not projection_rows.empty else 0.0,
            "brier": float(evaluated["brier"].mean()) if not evaluated.empty else 0.0,
            "roi": float(bets["profit_units"].sum() / len(bets)) if not bets.empty else 0.0,
            "avg_clv_line": float(bets["clv_line"].mean()) if not bets.empty else 0.0,
        }
    return output


def _count_freshness_failures(values: pd.Series) -> dict[str, int]:
    failures = Counter(
        failure for item in values.fillna("") for failure in str(item).split(";") if failure
    )
    return dict(sorted(failures.items()))


def _freshness_by(rows: pd.DataFrame, column: str) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for value, group in rows.groupby(column, dropna=False):
        output[str(value)] = {
            "rows": int(len(group)),
            "fresh_rows": int(group["freshness_pass"].sum()),
            "fresh_rate": float(group["freshness_pass"].mean()),
            "failures": _count_freshness_failures(group["freshness_failures"]),
        }
    return output


def compute_metrics(df: pd.DataFrame) -> dict[str, object]:
    """Compute realized accuracy, calibration, ROI, CLV, and freshness metrics."""
    empty = {
        "rows_seen": int(len(df)),
        "fresh_rows": 0,
        "bets_placed": 0,
        "wins": 0,
        "losses": 0,
        "pushes": 0,
        "profit_units": 0.0,
        "roi": 0.0,
        "mae": 0.0,
        "brier": 0.0,
        "avg_clv_line": 0.0,
        "max_drawdown_units": 0.0,
        "freshness_failures": {},
        "freshness_by_market": {},
        "freshness_by_role": {},
        "by_market": {},
        "by_role": {},
        "calibration": [],
    }
    if df.empty:
        return empty

    fresh = df[df["freshness_pass"]].copy()
    evaluated = fresh[fresh["result"].isin(["win", "loss"])]
    projection_rows = fresh.dropna(subset=["actual"]).drop_duplicates(list(PROJECTION_KEYS))
    bets = fresh[fresh["recommendation"] & fresh["result"].isin(["win", "loss", "push"])].copy()
    empty["fresh_rows"] = int(len(fresh))
    empty["freshness_failures"] = _count_freshness_failures(df["freshness_failures"])
    empty["freshness_by_market"] = _freshness_by(df, "market")
    empty["freshness_by_role"] = _freshness_by(df, "role")
    empty["mae"] = float(projection_rows["abs_error"].mean()) if not projection_rows.empty else 0.0
    empty["brier"] = float(evaluated["brier"].mean()) if not evaluated.empty else 0.0
    empty["by_market"] = _group_metrics(fresh, "market")
    empty["by_role"] = _group_metrics(fresh, "role")
    empty["calibration"] = _calibration(evaluated)
    if bets.empty:
        return empty

    cumulative = bets["profit_units"].fillna(0).cumsum()
    drawdown = cumulative - cumulative.cummax().clip(lower=0)
    empty.update(
        bets_placed=int(len(bets)),
        wins=int((bets["result"] == "win").sum()),
        losses=int((bets["result"] == "loss").sum()),
        pushes=int((bets["result"] == "push").sum()),
        profit_units=float(bets["profit_units"].sum()),
        roi=float(bets["profit_units"].sum() / len(bets)),
        avg_clv_line=float(bets["clv_line"].mean()),
        max_drawdown_units=float(drawdown.min()),
    )
    return empty


def _load_replay_inputs(season: int, weeks: list[int]) -> tuple[pd.DataFrame, ...]:
    placeholders = ",".join("?" for _ in weeks)
    params = (season, *weeks)
    where = f"season = ? AND week IN ({placeholders})"
    projections = read_dataframe(
        f"SELECT season, week, player_id, team, market, mu, sigma, generated_at "
        f"FROM weekly_projections WHERE {where}",
        params=params,
    )
    odds = read_dataframe(
        f"SELECT season, week, player_id, market, sportsbook, line, price, under_price, as_of "
        f"FROM weekly_odds WHERE {where}",
        params=params,
    )
    actuals = read_dataframe(
        f"SELECT season, week, player_id, passing_yards, rushing_yards, receiving_yards, "
        f"receptions, targets FROM player_stats_enhanced WHERE {where}",
        params=params,
    )
    context = read_dataframe(
        f"SELECT season, week, player_id, team, position, is_starter, depth_rank, is_rookie, "
        f"is_new_team, source_updated_at, captured_at "
        f"FROM nfl_player_context_snapshots WHERE {where}",
        params=params,
    )
    games = read_dataframe(
        f"SELECT season, week, home_team, away_team, kickoff_utc FROM games WHERE {where}",
        params=params,
    )
    return projections, odds, actuals, context, games


def replay_weeks(
    season: int,
    weeks: Iterable[int],
    min_edge: float = 0.05,
) -> pd.DataFrame:
    requested = sorted(set(int(week) for week in weeks))
    if not requested:
        raise ValueError("At least one week is required")
    return build_replay_dataset(
        *_load_replay_inputs(season, requested),
        min_edge=min_edge,
    )


def save_metrics(season: int, weeks: list[int], metrics: dict[str, object]) -> None:
    logs_dir = config.logs_dir / "metrics"
    logs_dir.mkdir(parents=True, exist_ok=True)
    week_label = f"{min(weeks)}-{max(weeks)}" if len(weeks) > 1 else str(weeks[0])
    path = logs_dir / f"season-{season}-weeks-{week_label}.json"
    payload = {
        "season": season,
        "weeks": weeks,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **metrics,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    weeks = sorted(set(args.weeks))
    replay = replay_weeks(args.season, weeks, min_edge=args.min_edge)
    metrics = compute_metrics(replay)
    save_metrics(args.season, weeks, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
