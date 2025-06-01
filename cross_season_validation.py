#!/usr/bin/env python3
"""
Cross-Season Validation for NFL Algorithm
Train on historical seasons to predict future performance
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import sqlite3
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossSeasonValidator:
    def __init__(self, db_path="nfl_data.db"):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        
    def load_data(self):
        """Load all available data"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM player_stats ORDER BY season", conn)
        conn.close()
        
        logger.info(f"Loaded {len(df)} records across {df['season'].nunique()} seasons")
        return df
    
    def prepare_advanced_features(self, df):
        """Create advanced features for better predictions"""
        features = df.copy()
        
        # Basic derived features
        features['yards_per_attempt'] = features['rushing_yards'] / (features['rushing_attempts'] + 1)
        features['receptions_per_target'] = features['receptions'] / (features['targets'] + 1)
        features['total_touches'] = features['rushing_attempts'] + features['receptions']
        features['total_yards'] = features['rushing_yards'] + features['receiving_yards']
        
        # Position encoding
        position_map = {'RB': 1, 'WR': 2, 'TE': 3, 'QB': 4, 'FB': 5}
        features['position_encoded'] = features['position'].map(position_map).fillna(0)
        
        # Historical performance features (for players with multiple seasons)
        features = features.sort_values(['player_id', 'season']).reset_index(drop=True)
        
        # Previous season stats (lag features) - simplified approach
        lag_cols = ['rushing_yards', 'receiving_yards', 'rushing_attempts', 'receptions']
        for col in lag_cols:
            features[f'{col}_prev'] = features.groupby('player_id')[col].shift(1).fillna(0)
        
        # Career averages (simplified) - use expanding mean but handle index issues
        for col in lag_cols:
            # Calculate expanding mean and shift, then fill NaN with 0
            career_avg = features.groupby('player_id')[col].expanding().mean().reset_index(level=0, drop=True)
            features[f'{col}_career_avg'] = career_avg.shift(1).fillna(0)
        
        # Age-based features
        features['age_squared'] = features['age'] ** 2
        features['is_prime'] = ((features['age'] >= 24) & (features['age'] <= 28)).astype(int)
        features['is_veteran'] = (features['age'] >= 30).astype(int)
        
        # Usage rate features
        features['rush_usage'] = features['rushing_attempts'] / (features['games_played'] + 1)
        features['target_usage'] = features['targets'] / (features['games_played'] + 1)
        
        return features
    
    def get_feature_columns(self):
        """Define which features to use for modeling"""
        return [
            'age', 'age_squared', 'is_prime', 'is_veteran',
            'games_played', 'position_encoded',
            'rushing_attempts', 'yards_per_attempt', 'rush_usage',
            'receptions', 'targets', 'receptions_per_target', 'target_usage',
            'rushing_yards_prev', 'receiving_yards_prev',
            'rushing_attempts_prev', 'receptions_prev',
            'rushing_yards_career_avg', 'receiving_yards_career_avg'
        ]
    
    def train_test_by_season(self, df, train_seasons, test_season):
        """Train on specific seasons and test on another"""
        
        # Prepare features
        features_df = self.prepare_advanced_features(df)
        feature_cols = self.get_feature_columns()
        
        # Split by season
        train_data = features_df[features_df['season'].isin(train_seasons)].copy()
        test_data = features_df[features_df['season'] == test_season].copy()
        
        if train_data.empty or test_data.empty:
            logger.warning(f"No data for training seasons {train_seasons} or test season {test_season}")
            return None
        
        # Prepare training data
        X_train = train_data[feature_cols].fillna(0)
        y_rush_train = train_data['rushing_yards']
        y_rec_train = train_data['receiving_yards']
        
        # Prepare test data
        X_test = test_data[feature_cols].fillna(0)
        y_rush_test = test_data['rushing_yards']
        y_rec_test = test_data['receiving_yards']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=6),
            'Linear': LinearRegression()
        }
        
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Training {model_name} model...")
            
            # Train rushing model
            if model_name == 'Linear':
                rush_model = model.__class__()
                rec_model = model.__class__()
                rush_model.fit(X_train_scaled, y_rush_train)
                rec_model.fit(X_train_scaled, y_rec_train)
            else:
                rush_model = model.__class__(**model.get_params())
                rec_model = model.__class__(**model.get_params())
                rush_model.fit(X_train, y_rush_train)
                rec_model.fit(X_train, y_rec_train)
            
            # Make predictions
            if model_name == 'Linear':
                rush_pred = rush_model.predict(X_test_scaled)
                rec_pred = rec_model.predict(X_test_scaled)
            else:
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
            
            results[model_name] = {
                'rushing': rush_metrics,
                'receiving': rec_metrics,
                'rush_model': rush_model,
                'rec_model': rec_model,
                'scaler': scaler,
                'predictions': {
                    'rush_actual': y_rush_test,
                    'rush_pred': rush_pred,
                    'rec_actual': y_rec_test,
                    'rec_pred': rec_pred
                }
            }
        
        return results
    
    def comprehensive_validation(self):
        """Run comprehensive cross-season validation"""
        
        df = self.load_data()
        seasons = sorted(df['season'].unique())
        
        print("Cross-Season Validation Results")
        print("=" * 60)
        print(f"Available seasons: {seasons}")
        
        all_results = {}
        
        # Test different training/validation scenarios
        scenarios = [
            {
                'name': '2021→2022 Prediction',
                'train_seasons': [2021],
                'test_season': 2022
            },
            {
                'name': '2022→2023 Prediction', 
                'train_seasons': [2022],
                'test_season': 2023
            },
            {
                'name': '2021-2022→2023 Prediction',
                'train_seasons': [2021, 2022],
                'test_season': 2023
            }
        ]
        
        for scenario in scenarios:
            print(f"\n{scenario['name']}")
            print("-" * 40)
            
            results = self.train_test_by_season(
                df, 
                scenario['train_seasons'], 
                scenario['test_season']
            )
            
            if results:
                all_results[scenario['name']] = results
                
                # Display results for each model
                for model_name, model_results in results.items():
                    print(f"\n{model_name} Results:")
                    print(f"  Rushing - MAE: {model_results['rushing']['mae']:.1f}, R²: {model_results['rushing']['r2']:.3f}")
                    print(f"  Receiving - MAE: {model_results['receiving']['mae']:.1f}, R²: {model_results['receiving']['r2']:.3f}")
        
        return all_results
    
    def train_production_model(self):
        """Train final model on all available data for 2024/2025 predictions"""
        
        df = self.load_data()
        features_df = self.prepare_advanced_features(df)
        feature_cols = self.get_feature_columns()
        
        # Use all available data for training
        X = features_df[feature_cols].fillna(0)
        y_rush = features_df['rushing_yards']
        y_rec = features_df['receiving_yards']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train ensemble model (best performing from validation)
        rush_model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=12)
        rec_model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=12)
        
        rush_model.fit(X, y_rush)
        rec_model.fit(X, y_rec)
        
        # Store models
        self.models['production'] = {
            'rush_model': rush_model,
            'rec_model': rec_model,
            'scaler': scaler,
            'feature_cols': feature_cols
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'rush_importance': rush_model.feature_importances_,
            'rec_importance': rec_model.feature_importances_
        })
        feature_importance['avg_importance'] = (
            feature_importance['rush_importance'] + feature_importance['rec_importance']
        ) / 2
        feature_importance = feature_importance.sort_values('avg_importance', ascending=False)
        
        print(f"\nProduction Model Trained on {len(df)} records")
        print("=" * 50)
        print("Top 10 Most Important Features:")
        for _, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:<25} | Avg: {row['avg_importance']:.3f}")
        
        return feature_importance
    
    def predict_2025_template(self):
        """Create template for 2024/2025 season predictions"""
        
        if 'production' not in self.models:
            logger.error("Production model not trained yet. Run train_production_model() first.")
            return
        
        print(f"\n2024/2025 Season Prediction Framework")
        print("=" * 50)
        print("To make predictions for 2024/2025 season:")
        print("1. Collect current season data (games played, stats so far)")
        print("2. Use this model to predict remaining season totals")
        print("3. Compare with sportsbook lines for betting opportunities")
        
        # Example prediction (using 2023 data as template)
        df = self.load_data()
        sample_players = df[df['season'] == 2023].head(5)
        
        print(f"\nExample Predictions (using 2023 data as template):")
        print("-" * 60)
        
        for _, player in sample_players.iterrows():
            print(f"{player['name']} ({player['position']}, {player['team']}):")
            print(f"  2023 Actual: {player['rushing_yards']} rush yds, {player['receiving_yards']} rec yds")
            print(f"  Model would predict based on mid-season stats...")

def main():
    validator = CrossSeasonValidator()
    
    # Run comprehensive validation
    validation_results = validator.comprehensive_validation()
    
    # Train production model
    print(f"\n" + "="*60)
    print("TRAINING PRODUCTION MODEL FOR 2024/2025 PREDICTIONS")
    print("="*60)
    
    feature_importance = validator.train_production_model()
    
    # Show prediction framework
    validator.predict_2025_template()
    
    print(f"\n" + "="*60)
    print("CROSS-SEASON VALIDATION COMPLETE")
    print("✅ Model trained on all historical data")
    print("🎯 Ready for 2024/2025 season predictions")
    print("📊 Next: Collect current 2024 stats and make predictions")

if __name__ == "__main__":
    main() 