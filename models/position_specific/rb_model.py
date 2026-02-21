"""
Running Back specific model with position-focused features.
"""

from typing import List
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator

from .base_model import BasePositionModel

class RBModel(BasePositionModel):
    """Running back specific model optimized for rushing performance."""
    
    def __init__(self):
        super().__init__('RB')
    
    def get_position_features(self) -> List[str]:
        """Return RB-specific feature columns."""
        return [
            # Core usage metrics
            'rushing_attempts', 'carries_per_game', 'rush_share',
            'red_zone_carries', 'goal_line_carries',
            
            # Efficiency metrics
            'yards_per_attempt', 'yards_after_contact', 'breakaway_runs',
            'first_down_rate', 'success_rate',
            
            # Game script & situation
            'game_script', 'negative_script_usage', 'positive_script_usage',
            'early_down_usage', 'late_down_usage',
            
            # Team context
            'oline_strength', 'offensive_strength', 'pace_rank',
            'red_zone_efficiency', 'turnover_differential',
            
            # Physical & context
            'age', 'age_squared', 'is_prime', 'is_veteran',
            'health_score', 'injury_probability',
            
            # Weather (for outdoor games)
            'cold_weather', 'bad_weather', 'is_dome',
            
            # Historical performance
            'rushing_yards_prev', 'rushing_attempts_prev',
            'rushing_yards_career_avg', 'carries_career_avg',
            
            # Receiving component
            'receptions_per_game', 'target_share', 'receiving_yards_prev'
        ]
    
    def get_target_columns(self) -> List[str]:
        """Return target columns for RB model."""
        return ['rushing_yards', 'receiving_yards', 'total_touches']
    
    def engineer_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer RB-specific features."""
        features = df.copy()
        
        # Core usage features
        features['carries_per_game'] = features['rushing_attempts'] / (features['games_played'] + 1)
        features['receptions_per_game'] = features['receptions'] / (features['games_played'] + 1)
        features['total_touches'] = features['rushing_attempts'] + features['receptions']
        features['touches_per_game'] = features['total_touches'] / (features['games_played'] + 1)
        
        # Team share metrics
        team_rush_attempts = features.groupby(['team', 'season', 'week'])['rushing_attempts'].transform('sum')
        features['rush_share'] = features['rushing_attempts'] / (team_rush_attempts + 1)
        
        # Efficiency metrics
        features['yards_per_attempt'] = features['rushing_yards'] / (features['rushing_attempts'] + 1)
        features['yards_per_touch'] = (features['rushing_yards'] + features['receiving_yards']) / (features['total_touches'] + 1)
        
        # Red zone usage (estimated)
        features['red_zone_carries'] = features['rushing_attempts'] * 0.15  # Estimate
        features['goal_line_carries'] = features['rushing_attempts'] * 0.05  # Estimate
        
        # Success rate (simplified)
        features['success_rate'] = np.where(
            features['yards_per_attempt'] > 4.0, 0.6,
            np.where(features['yards_per_attempt'] > 3.0, 0.4, 0.2)
        )
        
        # Breakaway runs (estimated)
        features['breakaway_runs'] = features['rushing_yards'] / 100  # Rough estimate
        
        # Game script features
        features['game_script'] = features.get('game_script', 0)  # Would come from team data
        features['negative_script_usage'] = features['rush_share'] * (features['game_script'] < -3).astype(int)
        features['positive_script_usage'] = features['rush_share'] * (features['game_script'] > 3).astype(int)
        
        # Down and distance (simplified)
        features['early_down_usage'] = features['rushing_attempts'] * 0.7  # Estimate
        features['late_down_usage'] = features['rushing_attempts'] * 0.3   # Estimate
        
        # Yards after contact (estimated)
        features['yards_after_contact'] = features['rushing_yards'] * 0.6  # Rough estimate
        
        # First down rate
        features['first_down_rate'] = np.where(
            features['yards_per_attempt'] > 3.5, 0.5,
            np.where(features['yards_per_attempt'] > 2.5, 0.3, 0.1)
        )
        
        # Career averages
        features = self._add_career_averages(features)
        
        # Lag features
        features = self._add_lag_features(features)
        
        return features
    
    def _add_career_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add career average features."""
        df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)
        
        # Career averages (expanding mean)
        career_cols = ['rushing_yards', 'rushing_attempts', 'total_touches', 'yards_per_attempt']
        for col in career_cols:
            if col in df.columns:
                career_avg = df.groupby('player_id')[col].expanding().mean().reset_index(level=0, drop=True)
                df[f'{col}_career_avg'] = career_avg.shift(1).fillna(df[col].mean())
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features for recent performance."""
        df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)
        
        # Previous game stats
        lag_cols = ['rushing_yards', 'rushing_attempts', 'total_touches', 'yards_per_attempt']
        for col in lag_cols:
            if col in df.columns:
                df[f'{col}_prev'] = df.groupby('player_id')[col].shift(1).fillna(0)
        
        # 3-game rolling averages
        for col in lag_cols:
            if col in df.columns:
                rolling_avg = df.groupby('player_id')[col].rolling(window=3, min_periods=1).mean()
                df[f'{col}_3game_avg'] = rolling_avg.reset_index(level=0, drop=True).shift(1).fillna(0)
        
        return df
    
    def _create_model(self) -> BaseEstimator:
        """Create the underlying ML model optimized for RB performance."""
        return GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    
 