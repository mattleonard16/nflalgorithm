"""
🛡️ Defense vs Position Adjustments
===================================
Calculates how players perform RELATIVE to their own average against specific defenses.
Used to boost/reduce projections based on matchup strength.

Example:
- If RBs typically get 30% MORE than their average vs CIN → 1.30x multiplier (weak D)
- If QBs typically get 10% LESS than their average vs DEN → 0.90x multiplier (strong D)
"""

import logging
from functools import lru_cache
from typing import Dict, Optional, Tuple

import pandas as pd

try:
    import nflreadpy as nfl
except ImportError:
    nfl = None

from utils.db import read_dataframe

logger = logging.getLogger(__name__)


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


def get_opponent(team: str, week: int, season: int) -> Optional[str]:
    """Get opponent for a team in a specific week."""
    schedule = _load_schedule(season)
    if schedule.empty:
        return None
    
    game = schedule[
        (schedule['week'] == week) & 
        ((schedule['home_team'] == team) | (schedule['away_team'] == team))
    ]
    if game.empty:
        return None
    
    row = game.iloc[0]
    return row['away_team'] if row['home_team'] == team else row['home_team']


def compute_defense_vs_position_multipliers(
    season: int, 
    through_week: int,
    min_games: int = 3
) -> Dict[Tuple[str, str, str], float]:
    """
    Compute defense multipliers based on RELATIVE PERFORMANCE vs player's baseline.
    
    Instead of raw yards (which unfairly weights star players), we calculate:
    - Each player's season average
    - How they performed vs that average against each defense
    - The defense multiplier is the average relative performance
    
    Example: If players typically get 95% of their average against CIN,
    the multiplier is 0.95 (good defense). If they get 110%, it's 1.10 (weak defense).
    
    Returns dict like {('CIN', 'RB', 'rushing_yards'): 1.05, ...}
    """
    schedule = _load_schedule(season)
    if schedule.empty:
        logger.warning("No schedule available for defense adjustments")
        return {}
    
    stats = read_dataframe(f'''
        SELECT season, week, player_id, name, team, position, rushing_yards, receiving_yards
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
    
    # Calculate each player's season average (their baseline)
    player_avgs = stats.groupby(['player_id', 'position']).agg({
        'rushing_yards': 'mean',
        'receiving_yards': 'mean',
        'week': 'count'  # games played
    }).reset_index()
    player_avgs.columns = ['player_id', 'position', 'avg_rush', 'avg_rec', 'games']
    
    # Only use players with enough games for reliable baseline
    player_avgs = player_avgs[player_avgs['games'] >= 3]
    
    # Merge player averages back to get relative performance per game
    stats = stats.merge(player_avgs[['player_id', 'avg_rush', 'avg_rec']], on='player_id', how='left')
    
    # Calculate relative performance (actual / expected)
    # Add small epsilon to avoid division by zero
    stats['rush_rel_perf'] = stats['rushing_yards'] / (stats['avg_rush'] + 0.1)
    stats['rec_rel_perf'] = stats['receiving_yards'] / (stats['avg_rec'] + 0.1)
    
    multipliers = {}
    
    # Calculate defense adjustment based on relative performance
    stat_configs = [
        ('RB', 'rushing_yards', 'rush_rel_perf'),
        ('WR', 'receiving_yards', 'rec_rel_perf'),
        ('TE', 'receiving_yards', 'rec_rel_perf'),
        ('QB', 'rushing_yards', 'rush_rel_perf'),
    ]
    
    for position, stat_col, rel_col in stat_configs:
        pos_stats = stats[
            (stats['position'] == position) & 
            (stats[rel_col].notna()) & 
            (stats[rel_col] < 10)  # Filter outliers (10x normal is likely noise)
        ]
        
        for opponent in pos_stats['opponent'].unique():
            opp_games = pos_stats[pos_stats['opponent'] == opponent]
            
            if len(opp_games) < min_games:
                continue
            
            # Average relative performance against this defense
            avg_rel_perf = opp_games[rel_col].mean()
            
            # Clamp to reasonable bounds (0.7 to 1.3)
            # A defense can make players perform 30% worse or 30% better than usual
            multiplier = max(0.7, min(1.3, avg_rel_perf))
            
            multipliers[(opponent, position, stat_col)] = multiplier
    
    logger.info(f"Computed {len(multipliers)} relative defense vs position multipliers")
    return multipliers


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
    """
    multipliers = compute_defense_vs_position_multipliers(season, through_week)
    
    key = (opponent, position, stat_type)
    if key in multipliers:
        return multipliers[key]
    
    # Fallback: try just the position
    for k, v in multipliers.items():
        if k[1] == position and k[2] == stat_type:
            # Return average of all defenses for this position/stat
            relevant = [v for k2, v in multipliers.items() if k2[1] == position and k2[2] == stat_type]
            if relevant:
                return sum(relevant) / len(relevant)
    
    return 1.0  # Default: no adjustment


def print_defense_rankings(season: int, through_week: int) -> None:
    """Print defense rankings for analysis."""
    multipliers = compute_defense_vs_position_multipliers(season, through_week)
    
    if not multipliers:
        print("No defense data available")
        return
    
    # Group by stat type
    for stat_type in ['rushing_yards', 'receiving_yards']:
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

