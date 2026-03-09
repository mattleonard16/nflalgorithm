"""Running Back specific model with position-focused features.

Uses only real data columns from player_stats_enhanced. No fabricated estimates.
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor

from .base_model import BasePositionModel


class RBModel(BasePositionModel):
    """Running back specific model using real data columns only."""

    def __init__(self):
        super().__init__('RB')

    def get_position_features(self) -> List[str]:
        """Return RB-specific feature columns (real data only)."""
        return [
            # Core usage metrics (real data from player_stats_enhanced)
            'rushing_attempts',
            'carries_per_game',
            'rush_share',
            'receptions',
            'receptions_per_game',
            'targets',
            'target_share',

            # Efficiency metrics (derived from real data)
            'yards_per_attempt',
            'yards_per_touch',
            'total_touches',

            # Game context (real data)
            'game_script',
            'snap_count',
            'snap_percentage',
            'age',

            # Historical rolling features (real data with shift(1))
            'rushing_yards_prev',
            'rushing_attempts_prev',
            'rushing_yards_3game_avg',
            'rushing_attempts_3game_avg',
            'receiving_yards_prev',
        ]

    def get_target_columns(self) -> List[str]:
        """Return target columns for RB model."""
        return ['rushing_yards', 'receiving_yards', 'total_touches']

    def engineer_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer RB-specific features from real data columns."""
        features = df.copy()

        # Core usage features (derived from real data)
        games_denom = features.get('games_played', pd.Series(1, index=features.index)) + 1
        features['carries_per_game'] = features['rushing_attempts'] / games_denom
        features['receptions_per_game'] = features['receptions'] / games_denom
        features['total_touches'] = features['rushing_attempts'] + features['receptions']

        # Team rush share (real data from groupby)
        team_rush_attempts = features.groupby(
            ['team', 'season', 'week']
        )['rushing_attempts'].transform('sum')
        features['rush_share'] = features['rushing_attempts'] / (team_rush_attempts + 1)

        # Efficiency metrics from real data
        features['yards_per_attempt'] = (
            features['rushing_yards'] / (features['rushing_attempts'] + 1)
        )
        features['yards_per_touch'] = (
            (features['rushing_yards'] + features['receiving_yards'])
            / (features['total_touches'] + 1)
        )

        # Career averages from real history
        features = self._add_career_averages(features)

        # Lag features from real history
        features = self._add_lag_features(features)

        return features

    def _add_career_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add expanding career average features (no leakage via shift)."""
        df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

        career_cols = ['rushing_yards', 'rushing_attempts', 'total_touches', 'yards_per_attempt']
        for col in career_cols:
            if col in df.columns:
                career_avg = (
                    df.groupby('player_id')[col]
                    .expanding()
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                df[f'{col}_career_avg'] = career_avg.shift(1).fillna(0.0)

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features from real game history (shift(1) prevents leakage)."""
        df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

        lag_cols = ['rushing_yards', 'rushing_attempts', 'total_touches', 'yards_per_attempt',
                    'receiving_yards']
        for col in lag_cols:
            if col in df.columns:
                df[f'{col}_prev'] = df.groupby('player_id')[col].shift(1).fillna(0.0)

        # 3-game rolling averages (with shift to prevent lookahead)
        for col in ['rushing_yards', 'rushing_attempts']:
            if col in df.columns:
                rolling_avg = df.groupby('player_id')[col].transform(
                    lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
                )
                df[f'{col}_3game_avg'] = rolling_avg.fillna(0.0)

        return df

    def _create_model(self) -> BaseEstimator:
        """Create the underlying ML model for RB performance."""
        return GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )
