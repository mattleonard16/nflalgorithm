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
    # A missing team identifies nothing: None == None must not stack two
    # unrelated rows into one game.
    team_a = row_a.get("team")
    team_b = row_b.get("team")
    return pd.notna(team_a) and pd.notna(team_b) and team_a == team_b


# ── Correlation detection ──────────────────────────────────────────────

# Pair types in label priority: a group's label leads with its strongest type
# (the API reports ``label.split("_")[0]`` as the group type).
PAIR_TYPE_PRIORITY: Tuple[str, ...] = ("pos_corr", "neg_corr", "same_team")


def detect_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Tag every row with a ``correlation_group`` label.

    Pairs are classified by :func:`_classify_pair` and grouped with
    union-find, so any chain of correlated pairs is one group however the
    rows are ordered. The label lists every pair type present in the group,
    strongest first, plus a sequence number: ``pos_corr_1``,
    ``pos_corr+same_team_2``.

    Returns a *new* DataFrame (original is not mutated) with one extra
    column: ``correlation_group`` (str | None).
    """
    if df.empty:
        return df.assign(correlation_group=pd.Series(dtype="object"))

    n_rows = len(df)
    parent = list(range(n_rows))

    def find(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    edge_types: List[Tuple[int, int, str]] = []
    for i, j in combinations(range(n_rows), 2):
        row_a = df.iloc[i]
        row_b = df.iloc[j]

        if not _same_game(row_a, row_b):
            continue

        pair = _classify_pair(row_a, row_b)
        if pair is None:
            continue

        edge_types.append((i, j, pair))
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[max(root_i, root_j)] = min(root_i, root_j)

    types_by_root: Dict[int, set] = {}
    for i, _j, pair in edge_types:
        types_by_root.setdefault(find(i), set()).add(pair)

    # Number groups in order of their first row, so labels are stable.
    labels: Dict[int, str] = {}
    for root in sorted(types_by_root):
        kinds = [t for t in PAIR_TYPE_PRIORITY if t in types_by_root[root]]
        labels[root] = f"{'+'.join(kinds)}_{len(labels) + 1}"

    group_series = pd.Series(
        [labels.get(find(k)) for k in range(n_rows)],
        index=df.index,
        dtype="object",
    )
    return df.assign(correlation_group=group_series)


def _side(row: pd.Series) -> str:
    """The bet side; rows without one are overs (the pre-``side`` default)."""
    side = row.get("side")
    if side is None or (not isinstance(side, str) and pd.isna(side)):
        return "over"
    side = str(side).strip().lower()
    return side or "over"


def _market_pair(mkt_a: str, mkt_b: str, pairs: List[Tuple[str, str]]) -> bool:
    return any(
        (mkt_a == x and mkt_b == y) or (mkt_a == y and mkt_b == x) for x, y in pairs
    )


def _classify_pair(a: pd.Series, b: pd.Series) -> Optional[str]:
    """Return a label if (a, b) concentrate risk on one outcome, else None.

    - Same player_id: not a correlation. Multi-book quotes of one prop are
      the same bet; per-player concentration is capped by
      :func:`compute_exposure`'s player limit instead.
    - Every type requires the same known team. A QB's passing yards and an
      OPPOSING receiver's yards are not a stack.
    - Side matters. Positively correlated markets concentrate when both bets
      take the same side (QB over + his WR over); negatively correlated ones
      when they take opposite sides (QB passing over + his RB rushing under).
      The other combinations offset each other and are not flagged.
    - Any other two props on one team in one game are ``same_team``.
    """
    pid_a = a.get("player_id")
    pid_b = b.get("player_id")
    if pd.notna(pid_a) and pd.notna(pid_b) and pid_a == pid_b:
        return None

    team_a = a.get("team")
    team_b = b.get("team")
    if not (pd.notna(team_a) and pd.notna(team_b) and team_a == team_b):
        return None

    mkt_a = str(a.get("market", "")).lower()
    mkt_b = str(b.get("market", "")).lower()
    same_side = _side(a) == _side(b)
    if _market_pair(mkt_a, mkt_b, POSITIVE_CORRELATIONS):
        return "pos_corr" if same_side else None
    if _market_pair(mkt_a, mkt_b, NEGATIVE_CORRELATIONS):
        return "neg_corr" if not same_side else None

    # Same-team stacking (multiple props on same team, same game)
    return "same_team"


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


# ── Portfolio-level stake normalization ──────────────────────────────────────

def normalize_portfolio_stakes(
    df: pd.DataFrame,
    bankroll: float,
    max_total_fraction: float = 1.0,
) -> pd.DataFrame:
    """Scale stakes so total stake <= bankroll * max_total_fraction.

    Per-bet Kelly capping bounds each bet individually, but a large card can
    still sum past the bankroll. When it does, every ``stake`` (and the
    stake-derived ``kelly_fraction``, when present) is multiplied by the same
    factor so the total lands exactly on the limit — relative sizing across
    the card is preserved. Cards at or under the limit are returned
    unchanged, as are empty or zero-stake frames. Stakes must be
    non-negative and non-NaN; anything else raises ValueError, because a
    proportional scale over mixed-sign or NaN stakes cannot guarantee the
    cap holds per bet.

    This is a global constraint on the whole card: it must run after all
    rows for the week have been ranked, never per-row. Returns a *new*
    DataFrame; the caller's frame is not mutated.
    """
    if bankroll <= 0:
        raise ValueError(f"bankroll must be positive, got {bankroll}")
    if max_total_fraction <= 0:
        raise ValueError(
            f"max_total_fraction must be positive, got {max_total_fraction}"
        )
    if df.empty:
        return df.copy()

    stakes = df["stake"].astype(float)
    invalid = stakes.isna() | (stakes < 0)
    if invalid.any():
        raise ValueError(
            "stakes must be non-negative and non-NaN to enforce the "
            f"portfolio cap; offending values: {stakes[invalid].tolist()}"
        )

    total = float(stakes.sum())
    limit = bankroll * max_total_fraction
    if total <= 0 or total <= limit:
        return df.copy()

    scale = limit / total
    scaled = {"stake": stakes * scale}
    if "kelly_fraction" in df.columns:
        scaled["kelly_fraction"] = df["kelly_fraction"].astype(float) * scale
    return df.assign(**scaled)


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
