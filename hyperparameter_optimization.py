#!/usr/bin/env python3
"""
Hyperparameter Optimization for NFL Algorithm using Context7 Knowledge
Implements RandomizedSearchCV and GridSearchCV with TimeSeriesSplit
"""

import sqlite3
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """NFL Algorithm Hyperparameter Optimizer with Context7 Knowledge"""
    
    def __init__(self, db_path="nfl_data.db"):
        self.db_path = db_path
        self.best_models = {}
        self.optimization_results = {}
        
    def load_data(self):
        """Load NFL data from database"""
        logger.info("Loading NFL data for hyperparameter optimization...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT * FROM player_stats ORDER BY season, player_id"
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            logger.info(f"Loaded {len(df)} player records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        features = df.copy()
        
        # Enhanced feature engineering from Context7 knowledge
        features['yards_per_attempt'] = features['rushing_yards'] / (features['rushing_attempts'] + 1)
        features['receptions_per_target'] = features['receptions'] / (features['targets'] + 1)
        features['yards_per_reception'] = features['receiving_yards'] / (features['receptions'] + 1)
        features['total_yards'] = features['rushing_yards'] + features['receiving_yards']
        features['total_touches'] = features['rushing_attempts'] + features['receptions']
        features['yards_per_touch'] = features['total_yards'] / (features['total_touches'] + 1)
        
        # Position encoding
        position_map = {'RB': 1, 'WR': 2, 'TE': 3, 'QB': 4, 'FB': 5}
        features['position_encoded'] = features['position'].map(position_map).fillna(0)
        
        feature_columns = [
            'age', 'games_played', 'rushing_attempts', 'yards_per_attempt',
            'receptions', 'targets', 'receptions_per_target', 'position_encoded',
            'yards_per_reception', 'total_touches', 'yards_per_touch'
        ]
        
        X = features[feature_columns].fillna(0)
        y_rushing = features['rushing_yards']
        
        return X, y_rushing
    
    def optimize_random_forest(self, X, y, search_type='randomized'):
        """Optimize RandomForest hyperparameters using Context7 knowledge"""
        logger.info(f"Optimizing RandomForest with {search_type} search...")
        
        # Parameter grid from Context7 documentation
        param_grid = {
            'n_estimators': [200, 500, 1000],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        # Base model
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Time series cross-validation (Context7 best practice)
        cv = TimeSeriesSplit(n_splits=5)
        
        if search_type == 'randomized':
            search = RandomizedSearchCV(
                rf, param_grid, cv=cv,
                scoring='neg_mean_absolute_error',
                n_iter=50, random_state=42, n_jobs=-1,
                verbose=1
            )
        else:
            # For demonstration - GridSearch would be too slow with full grid
            reduced_grid = {
                'n_estimators': [200, 500],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'max_features': ['sqrt', None]
            }
            search = GridSearchCV(
                rf, reduced_grid, cv=cv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1, verbose=1
            )
        
        search.fit(X, y)
        
        return search
    
    def optimize_gradient_boosting(self, X, y):
        """Optimize GradientBoosting hyperparameters using Context7 knowledge"""
        logger.info("Optimizing GradientBoosting with RandomizedSearch...")
        
        # Parameter grid from Context7 documentation
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        gb = GradientBoostingRegressor(random_state=42)
        cv = TimeSeriesSplit(n_splits=5)
        
        search = RandomizedSearchCV(
            gb, param_grid, cv=cv,
            scoring='neg_mean_absolute_error',
            n_iter=50, random_state=42, n_jobs=-1,
            verbose=1
        )
        
        search.fit(X, y)
        
        return search
    
    def optimize_extra_trees(self, X, y):
        """Optimize ExtraTrees hyperparameters using Context7 knowledge"""
        logger.info("Optimizing ExtraTrees with RandomizedSearch...")
        
        param_grid = {
            'n_estimators': [200, 500, 1000],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        et = ExtraTreesRegressor(random_state=42, n_jobs=-1)
        cv = TimeSeriesSplit(n_splits=5)
        
        search = RandomizedSearchCV(
            et, param_grid, cv=cv,
            scoring='neg_mean_absolute_error',
            n_iter=50, random_state=42, n_jobs=-1,
            verbose=1
        )
        
        search.fit(X, y)
        
        return search
    
    def optimize_linear_models(self, X, y):
        """Optimize Ridge and Lasso hyperparameters"""
        logger.info("Optimizing linear models...")
        
        # Ridge optimization
        ridge_params = {
            'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
        }
        
        ridge = Ridge(random_state=42)
        cv = TimeSeriesSplit(n_splits=5)
        
        ridge_search = GridSearchCV(
            ridge, ridge_params, cv=cv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        ridge_search.fit(X, y)
        
        # Lasso optimization
        lasso_params = {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            'max_iter': [1000, 2000, 5000]
        }
        
        lasso = Lasso(random_state=42)
        lasso_search = GridSearchCV(
            lasso, lasso_params, cv=cv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        lasso_search.fit(X, y)
        
        return ridge_search, lasso_search
    
    def compare_optimized_models(self, X, y):
        """Compare all optimized models"""
        logger.info("Comparing all optimized models...")
        
        # Split data using TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        train_idx, test_idx = list(tscv.split(X))[-1]
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        results = {}
        
        # Optimize each model
        print(f"\n{'='*80}")
        print("HYPERPARAMETER OPTIMIZATION IN PROGRESS")
        print(f"{'='*80}")
        
        # RandomForest
        rf_search = self.optimize_random_forest(X_train, y_train)
        rf_pred = rf_search.best_estimator_.predict(X_test)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        
        results['RandomForest'] = {
            'search': rf_search,
            'mae': rf_mae,
            'r2': rf_r2,
            'best_params': rf_search.best_params_,
            'best_score': -rf_search.best_score_
        }
        
        # GradientBoosting
        gb_search = self.optimize_gradient_boosting(X_train, y_train)
        gb_pred = gb_search.best_estimator_.predict(X_test)
        gb_mae = mean_absolute_error(y_test, gb_pred)
        gb_r2 = r2_score(y_test, gb_pred)
        
        results['GradientBoosting'] = {
            'search': gb_search,
            'mae': gb_mae,
            'r2': gb_r2,
            'best_params': gb_search.best_params_,
            'best_score': -gb_search.best_score_
        }
        
        # ExtraTrees
        et_search = self.optimize_extra_trees(X_train, y_train)
        et_pred = et_search.best_estimator_.predict(X_test)
        et_mae = mean_absolute_error(y_test, et_pred)
        et_r2 = r2_score(y_test, et_pred)
        
        results['ExtraTrees'] = {
            'search': et_search,
            'mae': et_mae,
            'r2': et_r2,
            'best_params': et_search.best_params_,
            'best_score': -et_search.best_score_
        }
        
        # Linear models
        ridge_search, lasso_search = self.optimize_linear_models(X_train, y_train)
        
        ridge_pred = ridge_search.best_estimator_.predict(X_test)
        ridge_mae = mean_absolute_error(y_test, ridge_pred)
        ridge_r2 = r2_score(y_test, ridge_pred)
        
        lasso_pred = lasso_search.best_estimator_.predict(X_test)
        lasso_mae = mean_absolute_error(y_test, lasso_pred)
        lasso_r2 = r2_score(y_test, lasso_pred)
        
        results['Ridge'] = {
            'search': ridge_search,
            'mae': ridge_mae,
            'r2': ridge_r2,
            'best_params': ridge_search.best_params_,
            'best_score': -ridge_search.best_score_
        }
        
        results['Lasso'] = {
            'search': lasso_search,
            'mae': lasso_mae,
            'r2': lasso_r2,
            'best_params': lasso_search.best_params_,
            'best_score': -lasso_search.best_score_
        }
        
        return results
    
    def display_optimization_results(self, results):
        """Display comprehensive optimization results"""
        print(f"\n{'='*100}")
        print("HYPERPARAMETER OPTIMIZATION RESULTS")
        print(f"{'='*100}")
        
        # Sort by MAE (best first)
        sorted_models = sorted(results.items(), key=lambda x: x[1]['mae'])
        
        print(f"{'Model':<15} {'Test MAE':<10} {'Test R²':<10} {'CV MAE':<10} {'Improvement':<12}")
        print("-" * 80)
        
        baseline_mae = sorted_models[-1][1]['mae']  # Worst model as baseline
        
        for model_name, metrics in sorted_models:
            improvement = ((baseline_mae - metrics['mae']) / baseline_mae * 100)
            print(f"{model_name:<15} {metrics['mae']:<10.1f} {metrics['r2']:<10.3f} "
                  f"{metrics['best_score']:<10.1f} {improvement:<12.1f}%")
        
        # Best model details
        best_model, best_metrics = sorted_models[0]
        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"   Test MAE: {best_metrics['mae']:.1f} yards")
        print(f"   Test R²: {best_metrics['r2']:.3f}")
        print(f"   CV MAE: {best_metrics['best_score']:.1f} yards")
        
        print(f"\n📋 BEST HYPERPARAMETERS:")
        for param, value in best_metrics['best_params'].items():
            print(f"   {param}: {value}")
        
        return best_model, best_metrics
    
    def run_optimization(self):
        """Run complete hyperparameter optimization"""
        logger.info("Starting comprehensive hyperparameter optimization...")
        
        # Load and prepare data
        df = self.load_data()
        if df.empty:
            logger.error("No data found. Run data collection first.")
            return
        
        X, y = self.prepare_features(df)
        
        print(f"\nHYPERPARAMETER OPTIMIZATION")
        print("=" * 50)
        print(f"Dataset: {len(df)} players")
        print(f"Features: {X.shape[1]} enhanced features")
        print(f"Target: Rushing yards prediction")
        print(f"Validation: TimeSeriesSplit (5 folds)")
        
        # Run optimization
        results = self.compare_optimized_models(X, y)
        
        # Display results
        best_model, best_metrics = self.display_optimization_results(results)
        
        # Store results
        self.best_models = {model: metrics['search'].best_estimator_ 
                           for model, metrics in results.items()}
        self.optimization_results = results
        
        print(f"\n{'='*80}")
        print("CONTEXT7 HYPERPARAMETER OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        print("✅ RandomizedSearchCV implemented")
        print("✅ TimeSeriesSplit cross-validation")
        print("✅ Multiple model comparison")
        print("✅ Best hyperparameters identified")
        print(f"✅ Best model: {best_model} (MAE: {best_metrics['mae']:.1f})")
        
        return results

# Run optimization if script is executed directly
if __name__ == "__main__":
    optimizer = HyperparameterOptimizer()
    results = optimizer.run_optimization() 