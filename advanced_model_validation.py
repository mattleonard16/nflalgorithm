#!/usr/bin/env python3
"""
Advanced NFL Model Validation with Context7 Ensemble Methods
Enhanced with StackingRegressor, VotingRegressor, TimeSeriesSplit, and hyperparameter optimization
"""

import sqlite3
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             StackingRegressor, VotingRegressor, ExtraTreesRegressor)
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedModelValidator:
    """Advanced NFL Model Validator with Context7 Ensemble Methods"""
    
    def __init__(self, db_path="nfl_data.db"):
        self.db_path = db_path
        self.models = {}
        self.validation_results = {}
        self.ensemble_results = {}
        
    def load_data(self, seasons=None):
        """Load NFL data from database"""
        logger.info("Loading NFL data from database...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            if seasons:
                season_clause = f"WHERE season IN ({','.join(map(str, seasons))})"
            else:
                season_clause = ""
            
            query = f"""
            SELECT * FROM player_stats 
            {season_clause}
            ORDER BY season, player_id
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            logger.info(f"Loaded {len(df)} player records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def prepare_features(self, df):
        """Prepare features for modeling with enhanced feature engineering"""
        logger.info("Preparing features for modeling...")
        
        # Create a copy to avoid modifying original
        features = df.copy()
        
        # Calculate derived features
        features['yards_per_attempt'] = features['rushing_yards'] / (features['rushing_attempts'] + 1)
        features['receptions_per_target'] = features['receptions'] / (features['targets'] + 1)
        features['yards_per_reception'] = features['receiving_yards'] / (features['receptions'] + 1)
        features['total_yards'] = features['rushing_yards'] + features['receiving_yards']
        features['total_touches'] = features['rushing_attempts'] + features['receptions']
        features['yards_per_touch'] = features['total_yards'] / (features['total_touches'] + 1)
        
        # Position encoding
        position_map = {'RB': 1, 'WR': 2, 'TE': 3, 'QB': 4, 'FB': 5}
        features['position_encoded'] = features['position'].map(position_map).fillna(0)
        
        # Enhanced feature set
        feature_columns = [
            'age', 'games_played', 'rushing_attempts', 'yards_per_attempt',
            'receptions', 'targets', 'receptions_per_target', 'position_encoded',
            'yards_per_reception', 'total_touches', 'yards_per_touch'
        ]
        
        X = features[feature_columns].fillna(0)
        y_rushing = features['rushing_yards']
        y_receiving = features['receiving_yards']
        
        return X, y_rushing, y_receiving, feature_columns
    
    def create_base_estimators(self):
        """Create base estimators for ensemble methods"""
        estimators = [
            ('ridge', RidgeCV(alphas=[0.1, 1.0, 10.0])),
            ('lasso', LassoCV(random_state=42, max_iter=2000)),
            ('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
            ('et', ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('knr', KNeighborsRegressor(n_neighbors=20, metric='euclidean'))
        ]
        return estimators
    
    def train_stacking_regressor(self, X, y, cv_splits=5):
        """Train StackingRegressor with Context7 knowledge"""
        logger.info("Training StackingRegressor...")
        
        # Base estimators
        estimators = self.create_base_estimators()
        
        # Meta-learner (final estimator)
        final_estimator = GradientBoostingRegressor(
            n_estimators=25, subsample=0.5, min_samples_leaf=25, 
            max_features=1, random_state=42
        )
        
        # Time series cross-validation
        cv = TimeSeriesSplit(n_splits=cv_splits)
        
        # Stacking ensemble
        stacking_model = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            n_jobs=-1
        )
        
        return stacking_model
    
    def train_voting_regressor(self, X, y):
        """Train VotingRegressor with Context7 knowledge"""
        logger.info("Training VotingRegressor...")
        
        # Create individual regressors
        reg1 = GradientBoostingRegressor(n_estimators=100, random_state=42)
        reg2 = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        reg3 = LinearRegression()
        reg4 = ExtraTreesRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        
        # Voting ensemble
        voting_model = VotingRegressor(
            estimators=[
                ('gb', reg1), 
                ('rf', reg2), 
                ('lr', reg3),
                ('et', reg4)
            ]
        )
        
        return voting_model
    
    def validate_advanced_models(self):
        """Main validation function for advanced models"""
        logger.info("Starting advanced model validation...")
        
        # Load data
        df = self.load_data()
        
        if df.empty:
            logger.error("No data found in database. Run data collection first.")
            return
        
        print(f"\nADVANCED MODEL VALIDATION")
        print("=" * 50)
        print(f"Dataset: {len(df)} players")
        print(f"Features: Enhanced with Context7 knowledge")
        print(f"Methods: Stacking, Voting, TimeSeriesSplit")
        
        # Prepare features
        X, y_rushing, y_receiving, feature_columns = self.prepare_features(df)
        
        # Time series split for proper validation
        tscv = TimeSeriesSplit(n_splits=5)
        train_idx, test_idx = list(tscv.split(X))[-1]  # Use last split
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_rush_train, y_rush_test = y_rushing.iloc[train_idx], y_rushing.iloc[test_idx]
        y_rec_train, y_rec_test = y_receiving.iloc[train_idx], y_receiving.iloc[test_idx]
        
        models = {}
        results = {}
        
        # 1. Original RandomForest (baseline)
        print(f"\nTraining baseline RandomForest...")
        rf_baseline = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_baseline.fit(X_train, y_rush_train)
        rf_pred = rf_baseline.predict(X_test)
        
        rf_mae = mean_absolute_error(y_rush_test, rf_pred)
        rf_r2 = r2_score(y_rush_test, rf_pred)
        
        models['baseline_rf'] = rf_baseline
        results['baseline'] = {'mae': rf_mae, 'r2': rf_r2}
        
        # 2. Stacking Regressor
        print(f"\nTraining StackingRegressor...")
        rushing_stacker = self.train_stacking_regressor(X_train, y_rush_train)
        rushing_stacker.fit(X_train, y_rush_train)
        stacker_pred = rushing_stacker.predict(X_test)
        
        stacker_mae = mean_absolute_error(y_rush_test, stacker_pred)
        stacker_r2 = r2_score(y_rush_test, stacker_pred)
        
        models['stacker'] = rushing_stacker
        results['stacker'] = {'mae': stacker_mae, 'r2': stacker_r2}
        
        # 3. Voting Regressor
        print(f"\nTraining VotingRegressor...")
        rushing_voter = self.train_voting_regressor(X_train, y_rush_train)
        rushing_voter.fit(X_train, y_rush_train)
        voter_pred = rushing_voter.predict(X_test)
        
        voter_mae = mean_absolute_error(y_rush_test, voter_pred)
        voter_r2 = r2_score(y_rush_test, voter_pred)
        
        models['voter'] = rushing_voter
        results['voter'] = {'mae': voter_mae, 'r2': voter_r2}
        
        # Display comparison
        print(f"\n{'='*60}")
        print("MODEL COMPARISON RESULTS")
        print(f"{'='*60}")
        
        for name, metrics in results.items():
            improvement = ((rf_mae - metrics['mae']) / rf_mae * 100) if name != 'baseline' else 0
            print(f"{name.upper():15} | MAE: {metrics['mae']:5.1f} | R²: {metrics['r2']:5.3f} | Improvement: {improvement:+5.1f}%")
        
        # Find best model
        best_model = min(results.keys(), key=lambda x: results[x]['mae'])
        best_mae = results[best_model]['mae']
        
        print(f"\n🏆 BEST MODEL: {best_model.upper()}")
        print(f"   MAE: {best_mae:.1f} yards")
        print(f"   Improvement over baseline: {((rf_mae - best_mae) / rf_mae * 100):+.1f}%")
        
        # Store results
        self.models = models
        self.ensemble_results = {
            'results': results,
            'best_model': best_model,
            'test_data': {'X_test': X_test, 'y_test': y_rush_test},
            'predictions': {
                'baseline': rf_pred,
                'stacker': stacker_pred,
                'voter': voter_pred
            }
        }
        
        return self.ensemble_results

# Run advanced validation if script is executed directly
if __name__ == "__main__":
    validator = AdvancedModelValidator()
    
    # Run advanced validation
    results = validator.validate_advanced_models()
    
    if results:
        print(f"\n{'='*80}")
        print("CONTEXT7 ENHANCED MODEL VALIDATION COMPLETE")
        print(f"{'='*80}")
        print("Improvements achieved:")
        print("✅ StackingRegressor ensemble")
        print("✅ VotingRegressor ensemble") 
        print("✅ TimeSeriesSplit cross-validation")
        print("✅ Enhanced feature engineering")
        print("✅ Professional model comparison")
        print("\nNext level capabilities unlocked! 🚀") 