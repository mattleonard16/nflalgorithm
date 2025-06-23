#!/usr/bin/env python3
"""
Advanced NFL Model Validation with Context7 Ensemble Methods (Fixed)
Enhanced with StackingRegressor, VotingRegressor, and proper validation
"""

import sqlite3
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             StackingRegressor, VotingRegressor, ExtraTreesRegressor)
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedModelValidatorFixed:
    """Advanced NFL Model Validator with Context7 Ensemble Methods (Fixed)"""
    
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
        
        # Calculate derived features (Context7 enhanced)
        features['yards_per_attempt'] = features['rushing_yards'] / (features['rushing_attempts'] + 1)
        features['receptions_per_target'] = features['receptions'] / (features['targets'] + 1)
        features['yards_per_reception'] = features['receiving_yards'] / (features['receptions'] + 1)
        features['total_yards'] = features['rushing_yards'] + features['receiving_yards']
        features['total_touches'] = features['rushing_attempts'] + features['receptions']
        features['yards_per_touch'] = features['total_yards'] / (features['total_touches'] + 1)
        features['efficiency_score'] = (features['yards_per_attempt'] + features['receptions_per_target']) / 2
        
        # Position encoding
        position_map = {'RB': 1, 'WR': 2, 'TE': 3, 'QB': 4, 'FB': 5}
        features['position_encoded'] = features['position'].map(position_map).fillna(0)
        
        # Enhanced feature set
        feature_columns = [
            'age', 'games_played', 'rushing_attempts', 'yards_per_attempt',
            'receptions', 'targets', 'receptions_per_target', 'position_encoded',
            'yards_per_reception', 'total_touches', 'yards_per_touch', 'efficiency_score'
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
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
            ('et', ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
            ('knr', KNeighborsRegressor(n_neighbors=10, metric='euclidean'))
        ]
        return estimators
    
    def train_stacking_regressor(self, X, y):
        """Train StackingRegressor with Context7 knowledge (Fixed)"""
        logger.info("Training StackingRegressor...")
        
        # Base estimators
        estimators = self.create_base_estimators()
        
        # Meta-learner (final estimator)
        final_estimator = GradientBoostingRegressor(
            n_estimators=25, subsample=0.5, min_samples_leaf=25, 
            max_features=1, random_state=42
        )
        
        # Stacking ensemble with regular CV (fixes the TimeSeriesSplit issue)
        stacking_model = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,  # Use regular 5-fold CV instead of TimeSeriesSplit
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
        
        # Voting ensemble (averages predictions)
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
        
        print(f"\nADVANCED MODEL VALIDATION WITH CONTEXT7")
        print("=" * 60)
        print(f"Dataset: {len(df)} players")
        print(f"Features: Enhanced with Context7 knowledge")
        print(f"Methods: Stacking, Voting, Enhanced Features")
        
        # Prepare features
        X, y_rushing, y_receiving, feature_columns = self.prepare_features(df)
        
        # Use train_test_split for proper data splitting
        X_train, X_test, y_rush_train, y_rush_test = train_test_split(
            X, y_rushing, test_size=0.2, random_state=42
        )
        
        models = {}
        results = {}
        
        print(f"\nTraining Models...")
        print("-" * 40)
        
        # 1. Original RandomForest (baseline)
        print(f"1. Training baseline RandomForest...")
        rf_baseline = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_baseline.fit(X_train, y_rush_train)
        rf_pred = rf_baseline.predict(X_test)
        
        rf_mae = mean_absolute_error(y_rush_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_rush_test, rf_pred))
        rf_r2 = r2_score(y_rush_test, rf_pred)
        
        models['baseline_rf'] = rf_baseline
        results['baseline'] = {'mae': rf_mae, 'rmse': rf_rmse, 'r2': rf_r2}
        
        # 2. Enhanced RandomForest
        print(f"2. Training enhanced RandomForest...")
        rf_enhanced = RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_split=5,
            max_features='sqrt', random_state=42, n_jobs=-1
        )
        rf_enhanced.fit(X_train, y_rush_train)
        rf_enh_pred = rf_enhanced.predict(X_test)
        
        rf_enh_mae = mean_absolute_error(y_rush_test, rf_enh_pred)
        rf_enh_rmse = np.sqrt(mean_squared_error(y_rush_test, rf_enh_pred))
        rf_enh_r2 = r2_score(y_rush_test, rf_enh_pred)
        
        models['enhanced_rf'] = rf_enhanced
        results['enhanced'] = {'mae': rf_enh_mae, 'rmse': rf_enh_rmse, 'r2': rf_enh_r2}
        
        # 3. Stacking Regressor
        print(f"3. Training StackingRegressor...")
        rushing_stacker = self.train_stacking_regressor(X_train, y_rush_train)
        rushing_stacker.fit(X_train, y_rush_train)
        stacker_pred = rushing_stacker.predict(X_test)
        
        stacker_mae = mean_absolute_error(y_rush_test, stacker_pred)
        stacker_rmse = np.sqrt(mean_squared_error(y_rush_test, stacker_pred))
        stacker_r2 = r2_score(y_rush_test, stacker_pred)
        
        models['stacker'] = rushing_stacker
        results['stacker'] = {'mae': stacker_mae, 'rmse': stacker_rmse, 'r2': stacker_r2}
        
        # 4. Voting Regressor
        print(f"4. Training VotingRegressor...")
        rushing_voter = self.train_voting_regressor(X_train, y_rush_train)
        rushing_voter.fit(X_train, y_rush_train)
        voter_pred = rushing_voter.predict(X_test)
        
        voter_mae = mean_absolute_error(y_rush_test, voter_pred)
        voter_rmse = np.sqrt(mean_squared_error(y_rush_test, voter_pred))
        voter_r2 = r2_score(y_rush_test, voter_pred)
        
        models['voter'] = rushing_voter
        results['voter'] = {'mae': voter_mae, 'rmse': voter_rmse, 'r2': voter_r2}
        
        # 5. Gradient Boosting
        print(f"5. Training GradientBoosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
        )
        gb_model.fit(X_train, y_rush_train)
        gb_pred = gb_model.predict(X_test)
        
        gb_mae = mean_absolute_error(y_rush_test, gb_pred)
        gb_rmse = np.sqrt(mean_squared_error(y_rush_test, gb_pred))
        gb_r2 = r2_score(y_rush_test, gb_pred)
        
        models['gradient_boost'] = gb_model
        results['gradient_boost'] = {'mae': gb_mae, 'rmse': gb_rmse, 'r2': gb_r2}
        
        # Display comparison
        print(f"\n{'='*80}")
        print("CONTEXT7 ENHANCED MODEL COMPARISON RESULTS")
        print(f"{'='*80}")
        
        print(f"{'Model':<15} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Improvement':<12}")
        print("-" * 65)
        
        baseline_mae = results['baseline']['mae']
        
        for name, metrics in results.items():
            improvement = ((baseline_mae - metrics['mae']) / baseline_mae * 100) if name != 'baseline' else 0
            print(f"{name:<15} {metrics['mae']:<8.1f} {metrics['rmse']:<8.1f} "
                  f"{metrics['r2']:<8.3f} {improvement:<12.1f}%")
        
        # Find best model
        best_model = min(results.keys(), key=lambda x: results[x]['mae'])
        best_mae = results[best_model]['mae']
        best_improvement = ((baseline_mae - best_mae) / baseline_mae * 100)
        
        print(f"\n🏆 BEST MODEL: {best_model.upper()}")
        print(f"   MAE: {best_mae:.1f} yards")
        print(f"   Improvement over baseline: {best_improvement:.1f}%")
        
        # Feature importance analysis
        if hasattr(models[best_model], 'feature_importances_'):
            print(f"\n📊 FEATURE IMPORTANCE ({best_model}):")
            importances = models[best_model].feature_importances_
            feature_imp = list(zip(feature_columns, importances))
            feature_imp.sort(key=lambda x: x[1], reverse=True)
            
            for feature, importance in feature_imp[:5]:
                print(f"   {feature:<20}: {importance:.3f}")
        
        # Cross-validation scores
        print(f"\n🔄 CROSS-VALIDATION SCORES:")
        cv_scores = cross_val_score(models[best_model], X, y_rushing, cv=5, 
                                   scoring='neg_mean_absolute_error')
        print(f"   CV MAE: {-cv_scores.mean():.1f} (+/- {cv_scores.std() * 2:.1f})")
        
        # Store results
        self.models = models
        self.ensemble_results = {
            'results': results,
            'best_model': best_model,
            'test_data': {'X_test': X_test, 'y_test': y_rush_test},
            'predictions': {
                'baseline': rf_pred,
                'enhanced': rf_enh_pred,
                'stacker': stacker_pred,
                'voter': voter_pred,
                'gradient_boost': gb_pred
            },
            'cv_score': -cv_scores.mean()
        }
        
        return self.ensemble_results
    
    def betting_simulation(self):
        """Advanced betting simulation with best model"""
        if not self.ensemble_results:
            logger.error("Run model validation first")
            return
        
        print(f"\n{'='*60}")
        print("BETTING STRATEGY SIMULATION")
        print(f"{'='*60}")
        
        best_model = self.ensemble_results['best_model']
        best_pred = self.ensemble_results['predictions'][best_model]
        y_test = self.ensemble_results['test_data']['y_test']
        
        # Betting simulation
        betting_threshold = 100
        confidence_margin = 15
        
        total_bets = 0
        successful_bets = 0
        
        for actual, predicted in zip(y_test, best_pred):
            if abs(predicted - betting_threshold) > confidence_margin:
                total_bets += 1
                
                if predicted > betting_threshold + confidence_margin and actual > betting_threshold:
                    successful_bets += 1
                elif predicted < betting_threshold - confidence_margin and actual < betting_threshold:
                    successful_bets += 1
        
        if total_bets > 0:
            hit_rate = successful_bets / total_bets
            roi = (hit_rate - 0.525) * 100
            
            print(f"Best Model: {best_model.upper()}")
            print(f"MAE: {self.ensemble_results['results'][best_model]['mae']:.1f} yards")
            print(f"Total Bets: {total_bets}")
            print(f"Successful Bets: {successful_bets}")
            print(f"Hit Rate: {hit_rate:.1%}")
            print(f"Estimated ROI: {roi:.1f}%")
            
            if hit_rate > 0.55:
                print(f"🎯 PROFITABLE STRATEGY DETECTED!")
            else:
                print(f"⚠️  Strategy needs refinement")
        
        return hit_rate if total_bets > 0 else 0

# Run advanced validation if script is executed directly
if __name__ == "__main__":
    validator = AdvancedModelValidatorFixed()
    
    # Run advanced validation
    results = validator.validate_advanced_models()
    
    if results:
        # Run betting simulation
        validator.betting_simulation()
        
        print(f"\n{'='*80}")
        print("CONTEXT7 ENHANCED MODEL VALIDATION COMPLETE")
        print(f"{'='*80}")
        print("✅ StackingRegressor ensemble (fixed)")
        print("✅ VotingRegressor ensemble") 
        print("✅ Enhanced feature engineering")
        print("✅ Professional model comparison")
        print("✅ Cross-validation analysis")
        print("✅ Betting strategy simulation")
        print("\nNext level capabilities unlocked! 🚀") 