"""Risk manager: correlation detection and exposure management.

Detects correlated props (QB+WR stacks, same-team exposure),
enforces per-team/game/player bankroll caps, and runs Monte Carlo
simulation to produce risk-adjusted Kelly fractions.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import config
from utils.db import read_dataframe
from utils.risk_utils import (
    append_warning as _append_warning,
    monte_carlo_drawdown,
    risk_adjusted_kelly,
)


# ── Prop-type correlation definitions ──────────────────────────────────

POSITIVE_CORRELATIONS: List[Tuple[str, str]] = [
    ("passing_yards", "receiving_yards"),
    ("passing_yards", "receptions"),
    ("passing_tds", "receiving_tds"),
]

NEGATIVE_CORRELATIONS: List[Tuple[str, str]] = [
    ("rushing_yards", "passing_yards"),
]


def _same_game(row_a: pd.Series, row_b: pd.Series) -> bool:
    """True when two value rows share the same game event."""
    if "event_id" in row_a.index and "event_id" in row_b.index:
        eid_a = row_a.get("event_id")
        eid_b = row_b.get("event_id")
        if pd.notna(eid_a) and pd.notna(eid_b) and eid_a == eid_b:
            return True
    return row_a.get("team") == row_b.get("team")


# ── Correlation detection ──────────────────────────────────────────────

def detect_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Tag every row with a ``correlation_group`` label.

    Returns a *new* DataFrame (original is not mutated) with one extra
    column: ``correlation_group`` (str | None).
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
    for pos_a, pos_b in POSITIVE_CORRELATIONS:
        if (mkt_a == pos_a and mkt_b == pos_b) or (
            mkt_a == pos_b and mkt_b == pos_a
        ):
            return "pos_corr"
    for neg_a, neg_b in NEGATIVE_CORRELATIONS:
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


# ── Same-team stacking detection ──────────────────────────────────────

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


# ── Exposure caps ─────────────────────────────────────────────────────

def compute_exposure(df: pd.DataFrame, bankroll: float) -> pd.DataFrame:
    """Compute exposure fractions and flag violations.

    Returns a *new* DataFrame with ``exposure_warning`` column added.
    """
    if df.empty:
        return df.assign(exposure_warning=pd.Series(dtype="object"))

    warnings: List[Optional[str]] = [None] * len(df)
    risk = config.risk

    team_stake = df.groupby("team")["stake"].sum()
    for team, total in team_stake.items():
        if total / bankroll > risk.max_team_exposure:
            for i, row in df.iterrows():
                if row.get("team") == team:
                    idx = df.index.get_loc(i)
                    warnings[idx] = _append_warning(
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
                    warnings[idx] = _append_warning(
                        warnings[idx],
                        f"game_exposure({total / bankroll:.0%})",
                    )

    player_stake = df.groupby("player_id")["stake"].sum()
    for pid, total in player_stake.items():
        if total / bankroll > risk.max_player_exposure:
            for i, row in df.iterrows():
                if row.get("player_id") == pid:
                    idx = df.index.get_loc(i)
                    warnings[idx] = _append_warning(
                        warnings[idx],
                        f"player_exposure({pid}={total / bankroll:.0%})",
                    )

    return df.assign(
        exposure_warning=pd.Series(warnings, index=df.index, dtype="object")
    )


# ── Historical correlation matrix ─────────────────────────────────────

def build_correlation_matrix(
    season: int,
    week: Optional[int] = None,
) -> pd.DataFrame:
    """Compute correlation matrix of stat columns from player_stats_enhanced.

    Uses historical data up to (but not including) *week* so we only look
    at information that was available before prediction time.
    """
    stat_cols = [
        "rushing_yards",
        "receiving_yards",
        "receptions",
        "targets",
        "air_yards",
    ]
    col_str = ", ".join(stat_cols)

    if week is not None:
        query = (
            f"SELECT {col_str} FROM player_stats_enhanced "
            f"WHERE season = ? AND week < ?"
        )
        params: tuple = (season, week)
    else:
        query = (
            f"SELECT {col_str} FROM player_stats_enhanced WHERE season = ?"
        )
        params = (season,)

    try:
        df = read_dataframe(query, params=params)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    return df[stat_cols].corr()


# ── Integration entry point ───────────────────────────────────────────

def assess_risk(df: pd.DataFrame, bankroll: Optional[float] = None) -> pd.DataFrame:
    """Full risk pipeline: correlations + exposure + Monte Carlo.

    Called by ``value_betting_engine`` before final ranking.
    Returns a new DataFrame with added columns:
        - correlation_group (str | None)
        - exposure_warning  (str | None)
        - risk_adjusted_kelly (float)
    """
    if df.empty:
        return df.assign(
            correlation_group=pd.Series(dtype="object"),
            exposure_warning=pd.Series(dtype="object"),
            risk_adjusted_kelly=pd.Series(dtype="float64"),
        )

    bk = bankroll if bankroll is not None else config.betting.bankroll

    result = detect_correlations(df)
    result = compute_exposure(result, bk)

    kelly_arr = result["kelly_fraction"].to_numpy(dtype=float)
    win_arr = result["p_win"].to_numpy(dtype=float)
    odds_arr = result["price"].to_numpy(dtype=float)

    dd_stats = monte_carlo_drawdown(kelly_arr, win_arr, odds_arr)

    adjusted = np.array([
        risk_adjusted_kelly(k, dd_stats) for k in kelly_arr
    ])
    result = result.assign(risk_adjusted_kelly=adjusted)

    return result


# ── CLI entry point ───────────────────────────────────────────────────

def run_risk_check(season: int, week: int) -> pd.DataFrame:
    """Load materialized value view and print risk report."""
    query = """
    SELECT * FROM materialized_value_view
    WHERE season = ? AND week = ?
    """
    try:
        df = read_dataframe(query, params=(season, week))
    except Exception:
        print(f"No data for season={season} week={week}")
        return pd.DataFrame()

    if df.empty:
        print(f"No value bets for season={season} week={week}")
        return df

    assessed = assess_risk(df)
    _print_risk_report(assessed)
    return assessed


def _print_risk_report(df: pd.DataFrame) -> None:
    """Print human-readable risk summary to stdout."""
    n_total = len(df)
    n_corr = df["correlation_group"].notna().sum()
    n_warn = df["exposure_warning"].notna().sum()

    print(f"Risk Assessment: {n_total} bets analyzed")
    print(f"  Correlated groups: {n_corr}")
    print(f"  Exposure warnings: {n_warn}")

    stacks = detect_team_stacks(df)
    if stacks:
        print("  Team stacks:")
        for team, idxs in stacks.items():
            print(f"    {team}: {len(idxs)} bets")

    if "risk_adjusted_kelly" in df.columns:
        orig = df["kelly_fraction"].sum()
        adj = df["risk_adjusted_kelly"].sum()
        print(f"  Total Kelly: {orig:.4f} -> risk-adjusted: {adj:.4f}")


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run risk check")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_risk_check(args.season, args.week)


if __name__ == "__main__":
    main()
