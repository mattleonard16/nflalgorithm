"""
🛡️ Defense vs Position Adjustments
===================================
Calculates how players perform RELATIVE to their own average against specific defenses.
Used to boost/reduce projections based on matchup strength.

Example:
- If RBs typically get 30% MORE than their average vs CIN → 1.30x multiplier (weak D)
- If QBs typically get 10% LESS than their average vs DEN → 0.90x multiplier (strong D)

Estimation pipeline (per (position, stat_type) group, pure math in
``compute_multipliers_from_game_stats`` so CI can test it without a database):

1. Raw estimate: per-defense symmetric 10% trimmed mean of each player's
   relative performance (game stat / that player's season baseline). Trimming
   both tails equally is robust to the right-skewed ratio distribution without
   the low bias of the old one-sided ``rel < 10`` filter, and it stays stable
   on zero-inflated stats where a plain median jumps between 0 and large
   values.
2. Normalize: rescale the group so the raw mean across defenses is exactly 1.0.
   Purely redistributive — a multiplier only ever says "better/worse than the
   average defense", so the group must average to ~1.0 by construction.
3. Shrinkage: pull toward 1.0 by sample size, ``1 + (m - 1) * n / (n + k)``
   with ``k = SHRINKAGE_K`` player-games. A defense needs ~k observed
   player-games (roughly 8-10 weeks of one position group) before its signal
   gets half weight; thin early-season samples stay near 1.0 instead of
   slamming into the clamps.
4. Clamp to ``CLAMP_BOUNDS`` as a rare guardrail, not a routine truncation.

WR/TE rushing yards are supported matchups but deliberately NEUTRAL (always
1.0): those stats are zero-inflated gadget plays with no per-defense signal —
on real 2025 data even the robust estimator clamped 10+ of 32 teams.
"""

import logging
from functools import lru_cache
from typing import Dict, FrozenSet, List, Tuple

import pandas as pd

try:
    import nflreadpy as nfl
except ImportError:
    nfl = None

from utils.db import read_dataframe

logger = logging.getLogger(__name__)

# Player-games of evidence at which a defense's raw signal gets half weight.
SHRINKAGE_K = 25.0

# Fraction trimmed from EACH tail of the relative-performance distribution.
TRIM_FRACTION = 0.1

# Guardrail bounds applied after normalization + shrinkage.
CLAMP_BOUNDS = (0.7, 1.3)

# A player needs this many games for a usable baseline...
MIN_PLAYER_GAMES = 3
# ...and at least this season-average in the stat: a WR averaging 0 rushing
# yards carries no matchup signal, and his all-zero relative-performance rows
# would only drag the group estimate toward 0.
MIN_BASELINE_YARDS = 1.0

# (position, stat_type) pairs with enough per-defense signal to estimate —
# mirrors MARKET_CONFIGS positions in models/position_specific/weekly.py.
COMPUTED_MATCHUPS: FrozenSet[Tuple[str, str]] = frozenset(
    {
        ("RB", "rushing_yards"),
        ("QB", "rushing_yards"),
        ("WR", "receiving_yards"),
        ("TE", "receiving_yards"),
        ("RB", "receiving_yards"),
        ("QB", "passing_yards"),
    }
)

# Valid pipeline matchups whose stat is too zero-inflated to grade a defense
# on; they always resolve to a neutral 1.0 (see module docstring).
NEUTRAL_MATCHUPS: FrozenSet[Tuple[str, str]] = frozenset(
    {
        ("WR", "rushing_yards"),
        ("TE", "rushing_yards"),
    }
)

# Every (position, stat_type) pair the projection pipeline may ask about.
SUPPORTED_MATCHUPS: FrozenSet[Tuple[str, str]] = COMPUTED_MATCHUPS | NEUTRAL_MATCHUPS

_STAT_COLUMNS = tuple(sorted({stat for _, stat in COMPUTED_MATCHUPS}))


@lru_cache(maxsize=1)
def _load_schedule(season: int) -> pd.DataFrame:
    """Load NFL schedule for opponent mapping."""
    if nfl is None:
        return pd.DataFrame()
    try:
        schedule = nfl.load_schedules([season]).to_pandas()
        return schedule[['week', 'home_team', 'away_team']].copy()
    except Exception as e:
        logger.warning(f"Could not load schedule: {e}")
        return pd.DataFrame()


def _trimmed_mean(values: List[float], trim_fraction: float) -> float:
    """Symmetric trimmed mean: drop floor(trim_fraction * n) from each tail."""
    ordered = sorted(values)
    k = int(len(ordered) * trim_fraction)
    if k:
        ordered = ordered[k:-k]
    return sum(ordered) / len(ordered)


def _raw_group_multipliers(
    stats: pd.DataFrame,
    position: str,
    stat_type: str,
    *,
    min_games: int,
    min_player_games: int,
    min_baseline: float,
    trim_fraction: float,
) -> Dict[str, Tuple[float, int]]:
    """Per-defense (trimmed-mean relative performance, n player-games) for one group."""
    pos = stats[stats["position"] == position]
    if pos.empty:
        return {}

    pos = pos.assign(_value=pd.to_numeric(pos[stat_type], errors="coerce"))
    pos = pos.dropna(subset=["_value", "opponent"])

    baselines = pos.groupby("player_id")["_value"].agg(["mean", "count"])
    eligible = baselines[
        (baselines["count"] >= min_player_games) & (baselines["mean"] >= min_baseline)
    ]
    pos = pos[pos["player_id"].isin(eligible.index)]
    if pos.empty:
        return {}

    pos = pos.merge(
        eligible["mean"].rename("_baseline"), left_on="player_id", right_index=True
    )
    pos["_rel"] = pos["_value"] / pos["_baseline"]

    out: Dict[str, Tuple[float, int]] = {}
    for opponent, grp in pos.groupby("opponent"):
        n = len(grp)
        if n < min_games:
            continue
        out[str(opponent)] = (_trimmed_mean(grp["_rel"].tolist(), trim_fraction), n)
    return out


def compute_multipliers_from_game_stats(
    stats: pd.DataFrame,
    *,
    min_games: int = 3,
    min_player_games: int = MIN_PLAYER_GAMES,
    min_baseline: float = MIN_BASELINE_YARDS,
    trim_fraction: float = TRIM_FRACTION,
    shrinkage_k: float = SHRINKAGE_K,
    clamp_bounds: Tuple[float, float] = CLAMP_BOUNDS,
) -> Dict[Tuple[str, str, str], float]:
    """Pure, DB-free multiplier computation (see module docstring for the pipeline).

    Args:
        stats: One row per player-game with columns ``player_id``, ``position``,
            ``opponent`` and every stat column in COMPUTED_MATCHUPS.
        min_games: Minimum player-games observed against a defense to emit a key.

    Returns:
        Dict like ``{('CIN', 'RB', 'rushing_yards'): 1.05, ...}``. Only
        COMPUTED_MATCHUPS groups are emitted; NEUTRAL_MATCHUPS resolve to 1.0
        in ``get_defense_multiplier``.

    Raises:
        ValueError: If required columns are missing or a group degenerates
            (fail loud at the boundary).
    """
    required = {"player_id", "position", "opponent", *_STAT_COLUMNS}
    missing = sorted(required - set(stats.columns))
    if missing:
        raise ValueError(f"stats is missing required columns: {missing}")

    lo, hi = clamp_bounds
    multipliers: Dict[Tuple[str, str, str], float] = {}
    for position, stat_type in sorted(COMPUTED_MATCHUPS):
        group = _raw_group_multipliers(
            stats,
            position,
            stat_type,
            min_games=min_games,
            min_player_games=min_player_games,
            min_baseline=min_baseline,
            trim_fraction=trim_fraction,
        )
        if not group:
            continue

        group_mean = sum(raw for raw, _ in group.values()) / len(group)
        if group_mean <= 0:
            raise ValueError(
                f"Degenerate multiplier group for ({position}, {stat_type}): "
                f"mean raw multiplier {group_mean} is not positive"
            )

        for opponent, (raw, n) in group.items():
            normalized = raw / group_mean
            shrunk = 1.0 + (normalized - 1.0) * (n / (n + shrinkage_k))
            multipliers[(opponent, position, stat_type)] = max(lo, min(hi, shrunk))

    logger.info(f"Computed {len(multipliers)} relative defense vs position multipliers")
    return multipliers


def compute_defense_vs_position_multipliers(
    season: int,
    through_week: int,
    min_games: int = 3
) -> Dict[Tuple[str, str, str], float]:
    """
    Compute defense multipliers based on RELATIVE PERFORMANCE vs player's baseline.

    DB-reading entry point: loads the schedule and player_stats_enhanced, maps
    opponents, then delegates all math to ``compute_multipliers_from_game_stats``.

    Returns dict like {('CIN', 'RB', 'rushing_yards'): 1.05, ...}
    """
    season = int(season)
    through_week = int(through_week)

    schedule = _load_schedule(season)
    if schedule.empty:
        logger.warning("No schedule available for defense adjustments")
        return {}

    stats = read_dataframe(f'''
        SELECT season, week, player_id, name, team, position,
               rushing_yards, receiving_yards, passing_yards
        FROM player_stats_enhanced
        WHERE season = {season} AND week <= {through_week}
        AND position IN ('QB', 'RB', 'WR', 'TE')
    ''')

    if stats.empty:
        return {}

    # Add opponent
    def _get_opp(row):
        game = schedule[
            (schedule['week'] == row['week']) &
            ((schedule['home_team'] == row['team']) | (schedule['away_team'] == row['team']))
        ]
        if game.empty:
            return None
        r = game.iloc[0]
        return r['away_team'] if r['home_team'] == row['team'] else r['home_team']

    stats['opponent'] = stats.apply(_get_opp, axis=1)
    stats = stats.dropna(subset=['opponent'])
    if stats.empty:
        return {}

    return compute_multipliers_from_game_stats(stats, min_games=min_games)


def get_defense_multiplier(
    opponent: str,
    position: str,
    stat_type: str,
    season: int,
    through_week: int
) -> float:
    """
    Get the defense adjustment multiplier for a specific matchup.

    Args:
        opponent: Defense team (e.g., 'CIN')
        position: Player position (e.g., 'RB')
        stat_type: Stat type (e.g., 'rushing_yards')
        season: Season year
        through_week: Use data through this week

    Returns:
        Multiplier (1.0 = league average, >1.0 = weak defense, <1.0 = strong defense)

    Raises:
        ValueError: If (position, stat_type) is not a supported matchup — an
            unknown stat_type must never silently return 1.0.
    """
    if (position, stat_type) not in SUPPORTED_MATCHUPS:
        raise ValueError(
            f"Unsupported defense matchup: position={position!r}, "
            f"stat_type={stat_type!r}. Supported: {sorted(SUPPORTED_MATCHUPS)}"
        )

    if (position, stat_type) in NEUTRAL_MATCHUPS:
        # Deliberately neutral (zero-inflated stat, no per-defense signal) —
        # see module docstring. Not a silent gap: the pair is documented.
        return 1.0

    multipliers = compute_defense_vs_position_multipliers(season, through_week)

    key = (opponent, position, stat_type)
    if key in multipliers:
        return multipliers[key]

    # Data-availability fallback (computed matchup, but this defense has too
    # few observed player-games): group mean, which is ~1.0 by construction.
    group = [
        v for (_, pos, stat), v in multipliers.items()
        if pos == position and stat == stat_type
    ]
    if group:
        return sum(group) / len(group)

    logger.info(
        "No defense multipliers computed for position=%s stat_type=%s "
        "(season=%s through_week=%s); defaulting to 1.0",
        position, stat_type, season, through_week,
    )
    return 1.0


def print_defense_rankings(season: int, through_week: int) -> None:
    """Print defense rankings for analysis."""
    multipliers = compute_defense_vs_position_multipliers(season, through_week)

    if not multipliers:
        print("No defense data available")
        return

    # Group by stat type
    for stat_type in _STAT_COLUMNS:
        print(f"\n=== Defense vs {stat_type.replace('_', ' ').title()} ===")
        relevant = {k: v for k, v in multipliers.items() if k[2] == stat_type}

        # Sort by multiplier (worst defenses first)
        sorted_defs = sorted(relevant.items(), key=lambda x: -x[1])

        print(f"{'Team':<6} {'Position':<6} {'Multiplier':<12} {'Rating'}")
        print("-" * 40)
        for (opp, pos, _), mult in sorted_defs[:10]:
            rating = "Weak D" if mult > 1.1 else "Strong D" if mult < 0.9 else "Average"
            print(f"{opp:<6} {pos:<6} {mult:.2f}x        {rating}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_defense_rankings(2025, 12)
