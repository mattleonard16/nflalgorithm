#!/usr/bin/env python3
"""
Enhanced Cross-Season Validation for NFL Algorithm
K-fold validation across 2021-2024 seasons with markdown leaderboard output.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedCrossSeasonValidator:
    """Enhanced cross-season validator targeting professional-grade performance."""
    
    def __init__(self, db_path: str = "nfl_data.db"):
        self.db_path = db_path
        self.target_mae = config.model.target_mae
        self.results = []
        
    def load_enhanced_data(self) -> pd.DataFrame:
        """Load enhanced dataset with all features."""
        conn = sqlite3.connect(self.db_path)
        
        # Try enhanced table first, fall back to basic
        try:
            query = """
            SELECT * FROM player_stats_enhanced 
            WHERE season BETWEEN 2021 AND 2024
            ORDER BY season, week, player_id
            """
            df = pd.read_sql_query(query, conn)
            logger.info(f"Loaded {len(df)} enhanced records")
        except:
            # Fallback to basic table
            query = """
            SELECT * FROM player_stats 
            WHERE season BETWEEN 2021 AND 2024
            ORDER BY season, player_id
            """
            df = pd.read_sql_query(query, conn)
            logger.info(f"Loaded {len(df)} basic records")
        
        conn.close()
        return df
    
    def engineer_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer enhanced features for better performance."""
        features = df.copy()
        
        # Basic derived features
        features['yards_per_attempt'] = features['rushing_yards'] / (features['rushing_attempts'] + 1)
        features['receptions_per_target'] = features['receptions'] / (features['targets'] + 1)
        features['total_touches'] = features['rushing_attempts'] + features['receptions']
        features['total_yards'] = features['rushing_yards'] + features['receiving_yards']
        features['yards_per_touch'] = features['total_yards'] / (features['total_touches'] + 1)
        
        # Position encoding
        position_map = {'RB': 1, 'WR': 2, 'TE': 3, 'QB': 4, 'FB': 5}
        features['position_encoded'] = features['position'].map(position_map).fillna(0)
        
        # Enhanced age features
        features['age_squared'] = features['age'] ** 2
        features['age_cubed'] = features['age'] ** 3
        features['is_prime'] = ((features['age'] >= 24) & (features['age'] <= 28)).astype(int)
        features['is_veteran'] = (features['age'] >= 30).astype(int)
        features['is_rookie'] = (features['age'] <= 22).astype(int)
        
        # Usage rate features
        features['rush_usage'] = features['rushing_attempts'] / (features['games_played'] + 1)
        features['target_usage'] = features['targets'] / (features['games_played'] + 1)
        features['touch_usage'] = features['total_touches'] / (features['games_played'] + 1)
        
        # Efficiency tiers
        features['high_efficiency'] = (features['yards_per_touch'] > 6.0).astype(int)
        features['low_efficiency'] = (features['yards_per_touch'] < 3.0).astype(int)
        
        # Team context (if available)
        if 'offensive_rank' in features.columns:
            features['offensive_strength'] = (33 - features['offensive_rank']) / 32
        else:
            features['offensive_strength'] = 0.5
        
        # Sort for lag features
        features = features.sort_values(['player_id', 'season']).reset_index(drop=True)
        
        # Enhanced lag features
        lag_cols = ['rushing_yards', 'receiving_yards', 'rushing_attempts', 'receptions', 'total_touches']
        for col in lag_cols:
            if col in features.columns:
                # Previous season
                features[f'{col}_prev'] = features.groupby('player_id')[col].shift(1).fillna(0)
                
                # Career averages (expanding mean)
                career_avg = features.groupby('player_id')[col].expanding().mean().reset_index(level=0, drop=True)
                features[f'{col}_career_avg'] = career_avg.shift(1).fillna(features[col].mean())
                
                # Momentum (recent trend)
                rolling_avg = features.groupby('player_id')[col].rolling(window=2, min_periods=1).mean()
                features[f'{col}_momentum'] = (features[col] - rolling_avg.reset_index(level=0, drop=True).shift(1)).fillna(0)
        
        # Interaction features
        features['age_x_usage'] = features['age'] * features['touch_usage']
        features['efficiency_x_volume'] = features['yards_per_touch'] * features['total_touches']
        features['prime_x_efficiency'] = features['is_prime'] * features['yards_per_touch']
        
        return features
    
    def get_enhanced_feature_columns(self) -> List[str]:
        """Get enhanced feature set for modeling."""
        return [
            # Core features
            'age', 'age_squared', 'age_cubed', 'is_prime', 'is_veteran', 'is_rookie',
            'games_played', 'position_encoded',
            
            # Usage features
            'rushing_attempts', 'yards_per_attempt', 'rush_usage',
            'receptions', 'targets', 'receptions_per_target', 'target_usage',
            'total_touches', 'touch_usage',
            
            # Efficiency features
            'yards_per_touch', 'high_efficiency', 'low_efficiency',
            'efficiency_x_volume',
            
            # Historical features
            'rushing_yards_prev', 'receiving_yards_prev', 'total_touches_prev',
            'rushing_yards_career_avg', 'receiving_yards_career_avg', 'total_touches_career_avg',
            'rushing_yards_momentum', 'receiving_yards_momentum',
            
            # Team context
            'offensive_strength',
            
            # Interaction features
            'age_x_usage', 'prime_x_efficiency'
        ]
    
    def create_ensemble_model(self, optimized_params: Optional[Dict] = None) -> StackingRegressor:
        """Create optimized ensemble model."""
        
        # Base models with optimized parameters
        if optimized_params:
            gb_params = optimized_params.get('gradient_boosting', {})
            rf_params = optimized_params.get('random_forest', {})
        else:
            # Default optimized parameters
            gb_params = {
                'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.08,
                'subsample': 0.85, 'min_samples_split': 5, 'min_samples_leaf': 2
            }
            rf_params = {
                'n_estimators': 250, 'max_depth': 15, 'min_samples_split': 5,
                'min_samples_leaf': 2, 'max_features': 'sqrt'
            }
        
        base_models = [
            ('gb', GradientBoostingRegressor(random_state=42, **gb_params)),
            ('rf', RandomForestRegressor(random_state=42, **rf_params)),
            ('lr', LinearRegression())
        ]
        
        # Meta-learner
        meta_learner = LinearRegression()
        
        return StackingRegressor(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=3
        )
    
    def validate_k_fold_seasons(self, k: int = 5) -> List[Dict]:
        """Perform k-fold validation across seasons."""
        df = self.load_enhanced_data()
        
        if df.empty:
            logger.error("No data loaded for validation")
            return []
        
        features_df = self.engineer_enhanced_features(df)
        feature_cols = self.get_enhanced_feature_columns()
        
        # Available feature columns
        available_features = [col for col in feature_cols if col in features_df.columns]
        logger.info(f"Using {len(available_features)} features")
        
        seasons = sorted(features_df['season'].unique())
        results = []
        
        # K-fold cross-validation across seasons
        n_splits = min(k, len(seasons) - 1)  # Ensure we don't exceed available seasons
        tscv = TimeSeriesSplit(n_splits=n_splits)
        season_indices = np.arange(len(seasons))
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(season_indices)):
            train_seasons = [seasons[i] for i in train_idx]
            test_seasons = [seasons[i] for i in test_idx]
            
            logger.info(f"Fold {fold + 1}: Train on {train_seasons}, Test on {test_seasons}")
            
            # Split data
            train_data = features_df[features_df['season'].isin(train_seasons)]
            test_data = features_df[features_df['season'].isin(test_seasons)]
            
            if train_data.empty or test_data.empty:
                continue
            
            # Prepare features and targets
            X_train = train_data[available_features].fillna(0)
            X_test = test_data[available_features].fillna(0)
            
            y_rush_train = train_data['rushing_yards']
            y_rush_test = test_data['rushing_yards']
            y_rec_train = train_data['receiving_yards']
            y_rec_test = test_data['receiving_yards']
            
            # Train models
            rush_model = self.create_ensemble_model()
            rec_model = self.create_ensemble_model()
            
            rush_model.fit(X_train, y_rush_train)
            rec_model.fit(X_train, y_rec_train)
            
            # Make predictions
            rush_pred = rush_model.predict(X_test)
            rec_pred = rec_model.predict(X_test)
            
            # Calculate metrics
            rush_metrics = {
                'mae': mean_absolute_error(y_rush_test, rush_pred),
                'rmse': np.sqrt(mean_squared_error(y_rush_test, rush_pred)),
                'r2': r2_score(y_rush_test, rush_pred)
            }
            
            rec_metrics = {
                'mae': mean_absolute_error(y_rec_test, rec_pred),
                'rmse': np.sqrt(mean_squared_error(y_rec_test, rec_pred)),
                'r2': r2_score(y_rec_test, rec_pred)
            }
            
            result = {
                'fold': fold + 1,
                'train_seasons': train_seasons,
                'test_seasons': test_seasons,
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'rushing_metrics': rush_metrics,
                'receiving_metrics': rec_metrics,
                'meets_target': rush_metrics['mae'] <= self.target_mae
            }
            
            results.append(result)
            
            logger.info(f"Fold {fold + 1} Results:")
            logger.info(f"  Rushing MAE: {rush_metrics['mae']:.3f} (Target: ≤{self.target_mae})")
            logger.info(f"  Receiving MAE: {rec_metrics['mae']:.3f}")
        
        self.results = results
        return results
    
    def generate_leaderboard_markdown(self) -> str:
        """Generate markdown leaderboard report using available result fields."""
        if not self.results:
            return "No validation results available."

        rush_maes = [r['rushing_metrics']['mae'] for r in self.results]
        rec_maes = [r['receiving_metrics']['mae'] for r in self.results]
        rush_r2s = [r['rushing_metrics']['r2'] for r in self.results]
        rec_r2s = [r['receiving_metrics']['r2'] for r in self.results]

        avg_rush_mae = float(np.mean(rush_maes))
        avg_rec_mae = float(np.mean(rec_maes))
        avg_rush_r2 = float(np.mean(rush_r2s))
        avg_rec_r2 = float(np.mean(rec_r2s))

        target_met = avg_rush_mae <= self.target_mae
        folds_meeting_target = int(sum(r['meets_target'] for r in self.results))

        report: List[str] = []
        report.append("# NFL Algorithm Cross-Season Validation Report\n\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        report.append("## Executive Summary\n\n")
        report.append(f"- Target MAE (Rushing): ≤ {self.target_mae:.1f}\n")
        report.append(f"- Average Rushing MAE: {avg_rush_mae:.3f}\n")
        report.append(f"- Average Receiving MAE: {avg_rec_mae:.3f}\n")
        report.append(f"- Average Rushing R²: {avg_rush_r2:.3f}\n")
        report.append(f"- Average Receiving R²: {avg_rec_r2:.3f}\n")
        report.append(f"- Folds Meeting Target: {folds_meeting_target}/{len(self.results)}\n\n")

        report.append("## Fold Results\n\n")
        report.append("| Fold | Test Seasons | Rushing MAE | Receiving MAE | Rushing R² | Receiving R² | Meets Target |\n")
        report.append("|------|--------------|-------------|---------------|------------|--------------|--------------|\n")
        for r in self.results:
            test_seasons = ", ".join(map(str, r['test_seasons']))
            report.append(
                f"| {r['fold']} | {test_seasons} | {r['rushing_metrics']['mae']:.3f} | {r['receiving_metrics']['mae']:.3f} | {r['rushing_metrics']['r2']:.3f} | {r['receiving_metrics']['r2']:.3f} | {'YES' if r['meets_target'] else 'NO'} |\n"
            )

        report.append("\n## Distribution\n\n")
        report.append(f"- Best Rushing MAE: {min(rush_maes):.3f}\n")
        report.append(f"- Worst Rushing MAE: {max(rush_maes):.3f}\n")
        report.append(f"- Rushing MAE StdDev: {np.std(rush_maes):.3f}\n")

        report.append("\n---\n")
        report.append("Report generated by Enhanced NFL Algorithm v2.0\n")
        return "".join(report)
    
    def save_results(self, output_dir: str = "logs") -> None:
        """Save validation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save markdown report
        markdown_report = self.generate_leaderboard_markdown()
        with open(output_path / "validation_leaderboard.md", "w") as f:
            f.write(markdown_report)
        
        # Save detailed results as CSV
        if self.results:
            results_df = pd.DataFrame([
                {
                    'fold': r['fold'],
                    'train_seasons': ', '.join(map(str, r['train_seasons'])),
                    'test_seasons': ', '.join(map(str, r['test_seasons'])),
                    'train_samples': r['train_samples'],
                    'test_samples': r['test_samples'],
                    'rushing_mae': r['rushing_metrics']['mae'],
                    'rushing_rmse': r['rushing_metrics']['rmse'],
                    'rushing_r2': r['rushing_metrics']['r2'],
                    'receiving_mae': r['receiving_metrics']['mae'],
                    'receiving_rmse': r['receiving_metrics']['rmse'],
                    'receiving_r2': r['receiving_metrics']['r2'],
                    'meets_target': r['meets_target']
                }
                for r in self.results
            ])
            
            results_df.to_csv(output_path / "validation_results.csv", index=False)
        
        logger.info(f"Results saved to {output_path}/")

def main():
    """Run enhanced cross-season validation."""
    print("Enhanced NFL Algorithm Cross-Season Validation")
    print("=" * 60)
    print("Targeting professional-grade performance (MAE ≤ 3.0)")
    
    validator = EnhancedCrossSeasonValidator()
    
    # Run k-fold validation
    print("\nRunning K-Fold Cross-Season Validation...")
    results = validator.validate_k_fold_seasons(k=5)
    
    if results:
        # Generate and display report
        markdown_report = validator.generate_leaderboard_markdown()
        print("\n" + markdown_report)
        
        # Save results
        validator.save_results()
        
        # Check if target met
        avg_mae = np.mean([r['rushing_metrics']['mae'] for r in results])
        if avg_mae <= validator.target_mae:
            print(f"\nSUCCESS: Target MAE ≤ {validator.target_mae} ACHIEVED!")
            print(f"   Average MAE: {avg_mae:.3f}")
            print("   Model ready for professional deployment!")
        else:
            print(f"\nTARGET NOT MET: Average MAE {avg_mae:.3f} > {validator.target_mae}")
            print("   Model needs further optimization.")
    
    else:
        print("❌ No validation results generated")

if __name__ == "__main__":
    main() 