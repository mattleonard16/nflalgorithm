"""Data freshness check: is each pipeline table caught up with the NFL calendar?

In 2026 every feed stopped after week 1 and nothing noticed, because each stage
fails closed on its own inputs and none compares its output to the schedule.
This module does that comparison, one verdict per table, against the `games`
kickoff times. It is read-only and takes the clock as an argument.

It is separate from ``api.data_health``: those checks validate the contents of
one finished run; these check that runs and feeds are happening at all.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import pandas as pd

from utils.current_week import SeasonOverError, as_utc, next_week_from_schedule
from utils.db import DBConnection, read_dataframe

# A week's box scores land nightly; two days covers a late Monday-night ingest.
STATS_GRACE_HOURS = 48
# Lines are scraped on a schedule; older than a day means the scraper stopped.
ODDS_MAX_AGE_HOURS = 24
# Books post a week's player props early in the week of the game, not before.
ODDS_POSTED_LEAD_DAYS = 5
# `make week-auto` runs weekly; one missed run plus slack.
RUN_MAX_AGE_DAYS = 8
# After the last game of the season, nothing more is expected.
SEASON_END_GRACE_DAYS = 1

PASS, FAIL, SKIP = "pass", "fail", "skip"


class FreshnessResult(NamedTuple):
    table: str
    status: str
    reason: str


def _ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")


def _weeks(sql: str, params: tuple, conn: DBConnection | None) -> set[int]:
    frame = read_dataframe(sql, params=params, conn=conn)
    return {int(w) for w in frame["week"].dropna()}


def _fmt_weeks(weeks: Any) -> str:
    return ", ".join(str(w) for w in sorted(weeks))


def check_data_freshness(
    now: Any = None, conn: DBConnection | None = None
) -> list[FreshnessResult]:
    """Return one PASS/FAIL/SKIP result per pipeline table.

    Outside the regular season every table is SKIP, never FAIL.
    """
    clock = as_utc(now)
    schedule = read_dataframe(
        "SELECT game_id, season, week, kickoff_utc FROM games WHERE kickoff_utc IS NOT NULL",
        conn=conn,
    )
    schedule["_ts"] = _ts(schedule["kickoff_utc"])
    schedule = schedule.loc[schedule["_ts"].notna()]

    try:
        upcoming = next_week_from_schedule(schedule, now=clock)
    except SeasonOverError as exc:
        return _all(SKIP, f"offseason: {exc}")
    except ValueError as exc:
        # An empty or unreadable schedule is the failure this check exists to catch.
        return _all(FAIL, f"cannot resolve the NFL week: {exc}")
    season, week = upcoming.season, upcoming.week

    this_season = schedule.loc[schedule["season"] == season]
    if clock > this_season["_ts"].max() + pd.Timedelta(days=SEASON_END_GRACE_DAYS) or (
        # next_week_from_schedule resolves to the first game of a future
        # season in the offseason; nothing is due before it kicks off.
        clock < this_season["_ts"].min()
    ):
        return _all(SKIP, f"offseason: season {season} is not in progress")

    week_kickoffs = this_season.groupby("week")["_ts"].agg(["min", "max"])
    cutoff = clock - pd.Timedelta(hours=STATS_GRACE_HOURS)
    completed = {int(w) for w, row in week_kickoffs.iterrows() if row["max"] < cutoff and w < week}

    scope = f"season {season} week {week}"
    results = [
        _check_stats(season, completed, conn),
        _check_projections(season, week, scope, conn),
    ]
    odds_result, odds_exist = _check_odds(
        season, week, scope, week_kickoffs.loc[week, "min"], clock, conn
    )
    results.append(odds_result)
    results.append(_check_card(season, week, scope, odds_exist, conn))
    results.append(_check_grading(season, completed, conn))
    results.append(_check_runs(clock, conn))
    return results


def _all(status: str, reason: str) -> list[FreshnessResult]:
    tables = (
        "stats", "projections", "odds", "card", "grading", "runs",
    )
    return [FreshnessResult(t, status, reason) for t in tables]


def _check_stats(season: int, completed: set[int], conn) -> FreshnessResult:
    if not completed:
        return FreshnessResult("stats", SKIP, "no week has been complete for 48h yet")
    have = _weeks(
        "SELECT DISTINCT week FROM player_stats_enhanced WHERE season = ?", (season,), conn
    )
    missing = completed - have
    if missing:
        return FreshnessResult(
            "stats",
            FAIL,
            f"player_stats_enhanced has no rows for completed week(s) {_fmt_weeks(missing)} "
            f"of {season} (latest ingested: {max(have) if have else 'none'})",
        )
    return FreshnessResult(
        "stats", PASS, f"player_stats_enhanced covers completed weeks {_fmt_weeks(completed)}"
    )


def _check_projections(season: int, week: int, scope: str, conn) -> FreshnessResult:
    have = _weeks(
        "SELECT DISTINCT week FROM weekly_projections WHERE season = ?", (season,), conn
    )
    if week in have:
        return FreshnessResult("projections", PASS, f"weekly_projections has rows for {scope}")
    return FreshnessResult(
        "projections",
        FAIL,
        f"weekly_projections has no rows for {scope} "
        f"(latest projected week: {max(have) if have else 'none'})",
    )


def _check_odds(
    season: int, week: int, scope: str, first_kickoff: pd.Timestamp, clock: pd.Timestamp, conn
) -> tuple[FreshnessResult, bool]:
    frame = read_dataframe(
        "SELECT as_of FROM weekly_odds WHERE season = ? AND week = ?", (season, week), conn=conn
    )
    stamps = _ts(frame["as_of"]).dropna()
    exist = not frame.empty
    posted_from = first_kickoff - pd.Timedelta(days=ODDS_POSTED_LEAD_DAYS)
    if clock < posted_from:
        return FreshnessResult(
            "odds", SKIP, f"{scope} props are not expected before {posted_from.isoformat()}"
        ), exist
    if stamps.empty:
        return FreshnessResult("odds", FAIL, f"weekly_odds has no snapshot for {scope}"), exist
    age = clock - stamps.max()
    if age > pd.Timedelta(hours=ODDS_MAX_AGE_HOURS):
        return FreshnessResult(
            "odds",
            FAIL,
            f"newest weekly_odds snapshot for {scope} is {age.days}d {age.seconds // 3600}h old "
            f"({stamps.max().isoformat()}); limit {ODDS_MAX_AGE_HOURS}h",
        ), exist
    return FreshnessResult("odds", PASS, f"weekly_odds snapshot for {scope} is under {ODDS_MAX_AGE_HOURS}h old"), exist


def _check_card(season: int, week: int, scope: str, odds_exist: bool, conn) -> FreshnessResult:
    if not odds_exist:
        return FreshnessResult("card", SKIP, f"no odds for {scope}, so no card is due")
    have = _weeks(
        "SELECT DISTINCT week FROM materialized_value_view WHERE season = ?", (season,), conn
    )
    if week in have:
        return FreshnessResult("card", PASS, f"materialized_value_view has rows for {scope}")
    return FreshnessResult(
        "card",
        FAIL,
        f"odds exist for {scope} but materialized_value_view has no card "
        f"(latest card week: {max(have) if have else 'none'})",
    )


def _check_grading(season: int, completed: set[int], conn) -> FreshnessResult:
    carded = _weeks(
        "SELECT DISTINCT week FROM materialized_value_view WHERE season = ?", (season,), conn
    ) & completed
    if not carded:
        return FreshnessResult("grading", SKIP, "no completed week had a card")
    graded = _weeks("SELECT DISTINCT week FROM bet_outcomes WHERE season = ?", (season,), conn)
    missing = carded - graded
    if missing:
        return FreshnessResult(
            "grading",
            FAIL,
            f"completed week(s) {_fmt_weeks(missing)} had a card but bet_outcomes has no rows",
        )
    return FreshnessResult("grading", PASS, f"bet_outcomes covers carded weeks {_fmt_weeks(carded)}")


def _check_runs(clock: pd.Timestamp, conn) -> FreshnessResult:
    frame = read_dataframe(
        "SELECT finished_at FROM pipeline_runs WHERE status = 'completed'", conn=conn
    )
    stamps = _ts(frame["finished_at"]).dropna()
    if stamps.empty:
        return FreshnessResult("runs", FAIL, "pipeline_runs has no completed run")
    age = clock - stamps.max()
    if age > pd.Timedelta(days=RUN_MAX_AGE_DAYS):
        return FreshnessResult(
            "runs",
            FAIL,
            f"latest completed pipeline run finished {age.days}d ago "
            f"({stamps.max().isoformat()}); limit {RUN_MAX_AGE_DAYS}d",
        )
    return FreshnessResult("runs", PASS, f"latest completed pipeline run is {age.days}d old")
