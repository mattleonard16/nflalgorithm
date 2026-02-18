"""NBA risk manager: correlation detection and exposure management.

Detects correlated NBA props (pts+fg3m, pts+ast stacks, same-team exposure),
enforces per-team/game/player bankroll caps, and runs Monte Carlo simulation
to produce risk-adjusted Kelly fractions.

Adapted from ``risk_manager.py`` for NBA's date-based structure.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import config
from utils.db import execute, executemany, get_connection, read_dataframe
from utils.risk_utils import append_warning, monte_carlo_drawdown, risk_adjusted_kelly

logger = logging.getLogger(__name__)

# ── NBA prop-type correlation definitions ────────────────────────────

NBA_POSITIVE_CORRELATIONS: List[Tuple[str, str]] = [
    ("pts", "fg3m"),
    ("pts", "ast"),
]

# No negative correlations for NBA (insufficient graded data to validate).
NBA_NEGATIVE_CORRELATIONS: List[Tuple[str, str]] = []


def _same_game(row_a: pd.Series, row_b: pd.Series) -> bool:
    """True when two value rows share the same game event."""
    if "event_id" in row_a.index and "event_id" in row_b.index:
        eid_a = row_a.get("event_id")
        eid_b = row_b.get("event_id")
        if pd.notna(eid_a) and pd.notna(eid_b) and eid_a == eid_b:
            return True
    team_a = row_a.get("team")
    team_b = row_b.get("team")
    return pd.notna(team_a) and pd.notna(team_b) and team_a == team_b


# ── Correlation detection ────────────────────────────────────────────

def detect_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Tag every row with a ``correlation_group`` label.

    Returns a *new* DataFrame with one extra column:
    ``correlation_group`` (str | None).
    """
    if df.empty:
        return df.assign(correlation_group=pd.Series(dtype="object"))

    groups: Dict[int, str] = {}
    group_counter = 0

    idx_list = list(df.index)
    for i, j in combinations(range(len(idx_list)), 2):
        row_a = df.iloc[i]
        row_b = df.iloc[j]

        if not _same_game(row_a, row_b):
            continue

        pair = _classify_pair(row_a, row_b)
        if pair is None:
            continue

        label = pair
        if i in groups:
            label = groups[i]
        elif j in groups:
            label = groups[j]
        else:
            group_counter += 1
            label = f"{pair}_{group_counter}"

        groups[i] = label
        groups[j] = label

    group_series = pd.Series(
        [groups.get(k) for k in range(len(idx_list))],
        index=df.index,
        dtype="object",
    )
    return df.assign(correlation_group=group_series)


def _classify_pair(a: pd.Series, b: pd.Series) -> Optional[str]:
    """Return a label if (a, b) form a known correlated pair."""
    mkt_a = str(a.get("market", "")).lower()
    mkt_b = str(b.get("market", "")).lower()
    for pos_a, pos_b in NBA_POSITIVE_CORRELATIONS:
        if (mkt_a == pos_a and mkt_b == pos_b) or (
            mkt_a == pos_b and mkt_b == pos_a
        ):
            return "pos_corr"
    for neg_a, neg_b in NBA_NEGATIVE_CORRELATIONS:
        if (mkt_a == neg_a and mkt_b == neg_b) or (
            mkt_a == neg_b and mkt_b == neg_a
        ):
            return "neg_corr"

    # Same-team stacking (multiple props on same team, same game)
    team_a = a.get("team")
    team_b = b.get("team")
    if pd.notna(team_a) and pd.notna(team_b) and team_a == team_b:
        return "same_team"

    return None


# ── Same-team stacking detection ────────────────────────────────────

def detect_team_stacks(df: pd.DataFrame) -> Dict[str, List[int]]:
    """Return {team: [row_indices]} for teams with >1 bet."""
    if df.empty:
        return {}
    team_groups: Dict[str, List[int]] = {}
    for i, row in df.iterrows():
        team = row.get("team")
        if pd.notna(team):
            team_groups.setdefault(str(team), []).append(i)
    return {t: idxs for t, idxs in team_groups.items() if len(idxs) > 1}


# ── Exposure caps ───────────────────────────────────────────────────

def compute_exposure(df: pd.DataFrame, bankroll: float) -> pd.DataFrame:
    """Compute exposure fractions and flag violations.

    Returns a *new* DataFrame with ``exposure_warning`` column added.
    """
    if df.empty:
        return df.assign(exposure_warning=pd.Series(dtype="object"))

    if bankroll <= 0:
        return df.assign(exposure_warning=pd.Series(dtype="object"))

    warnings: List[Optional[str]] = [None] * len(df)
    risk = config.risk

    team_stake = df.groupby("team")["stake"].sum()
    for team, total in team_stake.items():
        if total / bankroll > risk.max_team_exposure:
            for i, row in df.iterrows():
                if row.get("team") == team:
                    idx = df.index.get_loc(i)
                    warnings[idx] = append_warning(
                        warnings[idx],
                        f"team_exposure({team}={total / bankroll:.0%})",
                    )

    game_col = "event_id" if "event_id" in df.columns else "team"
    game_stake = df.groupby(game_col)["stake"].sum()
    for game, total in game_stake.items():
        if total / bankroll > risk.max_game_exposure:
            for i, row in df.iterrows():
                if row.get(game_col) == game:
                    idx = df.index.get_loc(i)
                    warnings[idx] = append_warning(
                        warnings[idx],
                        f"game_exposure({total / bankroll:.0%})",
                    )

    player_stake = df.groupby("player_id")["stake"].sum()
    for pid, total in player_stake.items():
        if total / bankroll > risk.max_player_exposure:
            for i, row in df.iterrows():
                if row.get("player_id") == pid:
                    idx = df.index.get_loc(i)
                    warnings[idx] = append_warning(
                        warnings[idx],
                        f"player_exposure({pid}={total / bankroll:.0%})",
                    )

    return df.assign(
        exposure_warning=pd.Series(warnings, index=df.index, dtype="object")
    )


# ── Integration entry point ─────────────────────────────────────────

def assess_risk(df: pd.DataFrame, bankroll: Optional[float] = None) -> pd.DataFrame:
    """Full NBA risk pipeline: correlations + exposure + Monte Carlo.

    Expects a DataFrame with columns: kelly_fraction, p_win, over_price,
    player_id, market, team, event_id, stake.

    Returns a new DataFrame with added columns:
        - correlation_group (str | None)
        - exposure_warning  (str | None)
        - risk_adjusted_kelly (float)
        - mean_drawdown (float)
        - max_drawdown (float)
        - p95_drawdown (float)
    """
    if df.empty:
        return df.assign(
            correlation_group=pd.Series(dtype="object"),
            exposure_warning=pd.Series(dtype="object"),
            risk_adjusted_kelly=pd.Series(dtype="float64"),
            mean_drawdown=pd.Series(dtype="float64"),
            max_drawdown=pd.Series(dtype="float64"),
            p95_drawdown=pd.Series(dtype="float64"),
        )

    bk = bankroll if bankroll is not None else config.betting.bankroll

    result = detect_correlations(df)
    result = compute_exposure(result, bk)

    kelly_arr = result["kelly_fraction"].to_numpy(dtype=float)
    win_arr = result["p_win"].to_numpy(dtype=float)
    odds_arr = result["over_price"].to_numpy(dtype=float)

    dd_stats = monte_carlo_drawdown(kelly_arr, win_arr, odds_arr)

    adjusted = np.array([
        risk_adjusted_kelly(k, dd_stats) for k in kelly_arr
    ])
    result = result.assign(
        risk_adjusted_kelly=adjusted,
        mean_drawdown=dd_stats["mean_drawdown"],
        max_drawdown=dd_stats["max_drawdown"],
        p95_drawdown=dd_stats["p95_drawdown"],
    )

    return result


# ── Persistence ─────────────────────────────────────────────────────

def _persist_risk_assessments(df: pd.DataFrame, game_date: str) -> int:
    """Write assessed rows to nba_risk_assessments. Returns row count."""
    if df.empty:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    params = []
    for _, row in df.iterrows():
        params.append((
            game_date,
            row.get("player_id"),
            str(row.get("market", "")),
            str(row.get("sportsbook", "")),
            str(row.get("event_id", "")),
            row.get("correlation_group"),
            row.get("exposure_warning"),
            float(row.get("risk_adjusted_kelly", 0)),
            float(row.get("mean_drawdown", 0)),
            float(row.get("max_drawdown", 0)),
            float(row.get("p95_drawdown", 0)),
            now,
        ))

    sql = """
        INSERT OR REPLACE INTO nba_risk_assessments
        (game_date, player_id, market, sportsbook, event_id,
         correlation_group, exposure_warning, risk_adjusted_kelly,
         mean_drawdown, max_drawdown, p95_drawdown, assessed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    executemany(sql, params)
    return len(params)


# ── CLI entry point ─────────────────────────────────────────────────

def run_risk_check(game_date: str) -> pd.DataFrame:
    """Load NBA materialized value view and run risk assessment."""
    query = "SELECT * FROM nba_materialized_value_view WHERE game_date = ?"
    try:
        df = read_dataframe(query, params=(game_date,))
    except Exception as exc:
        logger.warning("No data for game_date=%s: %s", game_date, exc)
        return pd.DataFrame()

    if df.empty:
        logger.info("No NBA value bets for game_date=%s", game_date)
        return df

    # Derive stake from kelly_fraction * bankroll
    bankroll = config.betting.bankroll
    df = df.assign(stake=df["kelly_fraction"] * bankroll)

    assessed = assess_risk(df, bankroll)
    persisted = _persist_risk_assessments(assessed, game_date)
    _print_risk_report(assessed, persisted)
    return assessed


def _print_risk_report(df: pd.DataFrame, persisted: int = 0) -> None:
    """Log human-readable risk summary."""
    n_total = len(df)
    n_corr = df["correlation_group"].notna().sum()
    n_warn = df["exposure_warning"].notna().sum()

    logger.info("NBA Risk Assessment: %d bets analyzed", n_total)
    logger.info("  Correlated groups: %d", n_corr)
    logger.info("  Exposure warnings: %d", n_warn)
    logger.info("  Persisted to nba_risk_assessments: %d", persisted)

    stacks = detect_team_stacks(df)
    if stacks:
        logger.info("  Team stacks:")
        for team, idxs in stacks.items():
            logger.info("    %s: %d bets", team, len(idxs))

    if "risk_adjusted_kelly" in df.columns:
        orig = df["kelly_fraction"].sum()
        adj = df["risk_adjusted_kelly"].sum()
        logger.info("  Total Kelly: %.4f -> risk-adjusted: %.4f", orig, adj)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NBA risk check")
    parser.add_argument(
        "--date",
        required=True,
        metavar="YYYY-MM-DD",
        help="Game date to assess",
    )
    args = parser.parse_args()
    run_risk_check(args.date)


if __name__ == "__main__":
    main()
