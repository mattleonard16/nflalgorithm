#!/usr/bin/env python3
"""
Enhanced NFL Algorithm Hyperparameter Optimization with Optuna
Professional-grade optimization targeting MAE ≤ 3.0
"""

import os
import sys
import sqlite3
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
import shap
from optuna.storages import RDBStorage
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import StackingRegressor
# import tensorflow as tf  # Commented out for now due to Python 3.13 compatibility

from config import config
from models.position_specific.rb_model import RBModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptunaOptimizer:
    """Optuna-based hyperparameter optimizer for NFL models."""
    
    def __init__(self):
        # Create Optuna storage
        storage_url = f"sqlite:///{config.project_root}/optuna.db"
        self.storage = RDBStorage(storage_url)
        
    def optimize_gradient_boosting(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize GradientBoosting hyperparameters."""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
            
            model = GradientBoostingRegressor(random_state=42, **params)
            
            # Use TimeSeriesSplit for time-aware validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
            
            return -scores.mean()  # Minimize MAE
        
        study = optuna.create_study(
            direction='minimize',
            storage=self.storage,
            study_name=f'gb_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best GradientBoosting MAE: {study.best_value:.3f}")
        logger.info(f"Best params: {study.best_params}")
        
        return {
            'best_params': study.best_params,
            'best_mae': study.best_value,
            'n_trials': len(study.trials),
            'study_name': study.study_name
        }
    
    def optimize_random_forest(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize RandomForest hyperparameters."""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
            
            model = RandomForestRegressor(random_state=42, **params)
            
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
            
            return -scores.mean()
        
        study = optuna.create_study(
            direction='minimize',
            storage=self.storage,
            study_name=f'rf_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best RandomForest MAE: {study.best_value:.3f}")
        logger.info(f"Best params: {study.best_params}")
        
        return {
            'best_params': study.best_params,
            'best_mae': study.best_value,
            'n_trials': len(study.trials),
            'study_name': study.study_name
        }
    
    def optimize_lstm(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> Dict[str, Any]:
        """Optimize LSTM hyperparameters."""
        
        def objective(trial):
            # LSTM hyperparameters
            lstm_units = trial.suggest_int('lstm_units', 32, 256)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            # Prepare sequential data (simplified)
            sequence_length = 8
            X_seq, y_seq = self._prepare_sequential_data(X, y, sequence_length)
            
            if len(X_seq) < 50:  # Need minimum data
                return float('inf')
            
            # Build model
            # model = tf.keras.Sequential([
            #     tf.keras.layers.LSTM(lstm_units, dropout=dropout_rate, return_sequences=False),
            #     tf.keras.layers.Dense(64, activation='relu'),
            #     tf.keras.layers.Dropout(dropout_rate),
            #     tf.keras.layers.Dense(1)
            # ])
            
            # model.compile(
            #     optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            #     loss='mean_absolute_error'
            # )
            
            # Train with early stopping
            # early_stopping = tf.keras.callbacks.EarlyStopping(
            #     monitor='val_loss', patience=10, restore_best_weights=True
            # )
            
            # Split for validation
            split_idx = int(0.8 * len(X_seq))
            X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
            
            # history = model.fit(
            #     X_train, y_train,
            #     validation_data=(X_val, y_val),
            #     epochs=50,
            #     batch_size=batch_size,
            #     callbacks=[early_stopping],
            #     verbose=0
            # )
            
            # Return validation MAE
            # val_mae = min(history.history['val_loss'])
            # return val_mae
            return float('inf') # Commented out TensorFlow LSTM optimization
        
        study = optuna.create_study(
            direction='minimize',
            storage=self.storage,
            study_name=f'lstm_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        
        study.optimize(objective, n_trials=n_trials, timeout=3600)  # 1 hour timeout
        
        logger.info(f"Best LSTM MAE: {study.best_value:.3f}")
        logger.info(f"Best params: {study.best_params}")
        
        return {
            'best_params': study.best_params,
            'best_mae': study.best_value,
            'n_trials': len(study.trials),
            'study_name': study.study_name
        }
    
    def optimize_lightgbm(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize LightGBM with monotonic constraints for breakout prediction."""
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'verbose': -1,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
                'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
                'lambda_l2': trial.suggest_float('lambda_l2', 0, 5),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7)
            }
            
            # Monotonic constraints (e.g., positive for usage_delta, negative for age_squared)
            mono_constraints = [1 if feat in ['usage_delta', 'preseason_buzz'] else -1 if feat in ['age_squared', 'injury_games_missed'] else 0 for feat in X.columns]
            params['monotone_constraints'] = mono_constraints
            
            dtrain = lgb.Dataset(X, label=y)
            
            pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'l1')
            
            cv_results = lgb.cv(
                params, dtrain, nfold=5, stratified=False, 
                callbacks=[pruning_callback], seed=42
            )
            
            return np.min(cv_results['l1-mean'])
        
        study = optuna.create_study(
            direction='minimize',
            storage=self.storage,
            study_name=f'lgbm_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best LightGBM MAE: {study.best_value:.3f}")
        logger.info(f"Best params: {study.best_params}")
        
        return {
            'best_params': study.best_params,
            'best_mae': study.best_value,
            'n_trials': len(study.trials),
            'study_name': study.study_name
        }
    
    def optimize_ensemble(self, X: pd.DataFrame, y: pd.Series, base_models: Dict) -> Dict[str, Any]:
        """Optimize stacking ensemble with LightGBM meta-learner."""
        estimators = [(name, model) for name, model in base_models.items()]
        
        stack = StackingRegressor(
            estimators=estimators,
            final_estimator=lgb.LGBMRegressor(objective='regression', metric='mae')
        )
        
        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(stack, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        
        mae = -scores.mean()
        
        logger.info(f"Ensemble MAE: {mae:.3f}")
        
        return {'best_mae': mae}
    
    def explain_with_shap(self, model, X: pd.DataFrame) -> pd.DataFrame:
        """Generate SHAP explanations for breakout predictions."""
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        
        # Summarize feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(shap_values.values).mean(0)
        }).sort_values('importance', ascending=False)
        
        logger.info("Top breakout drivers: " + str(importance.head(5)))
        
        return importance
    
    def _prepare_sequential_data(self, X: pd.DataFrame, y: pd.Series, sequence_length: int):
        """Prepare sequential data for LSTM."""
        # Sort by player and time
        if 'player_id' in X.columns and 'week' in X.columns:
            combined = pd.concat([X, y], axis=1).sort_values(['player_id', 'week'])
        else:
            combined = pd.concat([X, y], axis=1)
        
        sequences_X = []
        sequences_y = []
        
        # Create sequences per player
        if 'player_id' in X.columns:
            for player_id in combined['player_id'].unique():
                player_data = combined[combined['player_id'] == player_id]
                
                if len(player_data) >= sequence_length + 1:
                    for i in range(len(player_data) - sequence_length):
                        seq_X = player_data.iloc[i:i+sequence_length][X.columns].values
                        seq_y = player_data.iloc[i+sequence_length][y.name]
                        
                        sequences_X.append(seq_X)
                        sequences_y.append(seq_y)
        
        return np.array(sequences_X), np.array(sequences_y)
    
    def optimize_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, Any]]:
        """Optimize all including LightGBM and ensemble."""
        logger.info("Starting comprehensive hyperparameter optimization...")
        
        results = {}
        
        # Optimize GradientBoosting
        logger.info("Optimizing GradientBoosting...")
        results['gradient_boosting'] = self.optimize_gradient_boosting(X, y, n_trials=100)
        
        # Optimize RandomForest
        logger.info("Optimizing RandomForest...")
        results['random_forest'] = self.optimize_random_forest(X, y, n_trials=100)
        
        # Optimize LSTM (if enough data)
        if len(X) > 500:
            logger.info("Optimizing LSTM...")
            results['lstm'] = self.optimize_lstm(X, y, n_trials=30)
        
        # Optimize LightGBM
        logger.info("Optimizing LightGBM...")
        results['lightgbm'] = self.optimize_lightgbm(X, y, n_trials=100)
        
        # Ensemble stacking
        base_models = {
            'gb': GradientBoostingRegressor(**results['gradient_boosting']['best_params']),
            'rf': RandomForestRegressor(**results['random_forest']['best_params']),
            'lgbm': lgb.LGBMRegressor(**results['lightgbm']['best_params'])
        }
        results['ensemble'] = self.optimize_ensemble(X, y, base_models)
        
        # Find best overall model
        best_model = min(results.keys(), key=lambda k: results[k]['best_mae'])
        best_mae = results[best_model]['best_mae']
        
        logger.info(f"Best overall model: {best_model} with MAE: {best_mae:.3f}")
        
        # Generate SHAP for best model
        if best_model != 'lstm':
            model = base_models.get(best_model, GradientBoostingRegressor())
            model.fit(X, y)
            self.explain_with_shap(model, X)
        
        return {
            'results': results,
            'best_model': best_model,
            'best_mae': best_mae
        }
    
    def load_optimization_history(self, study_name: str) -> pd.DataFrame:
        """Load optimization history for analysis."""
        study = optuna.load_study(study_name=study_name, storage=self.storage)
        
        trials_df = study.trials_dataframe()
        return trials_df
    
    def generate_optimization_report(self) -> str:
        """Generate optimization report in markdown format."""
        
        # Get all studies
        studies = optuna.get_all_study_summaries(storage=self.storage)
        
        report = ["# Hyperparameter Optimization Report\n"]
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if not studies:
            report.append("No optimization studies found.\n")
            return "".join(report)
        
        report.append("## Study Summary\n\n")
        report.append("| Study Name | Direction | Best Value | Trials |\n")
        report.append("|------------|-----------|------------|--------|\n")
        
        for study_summary in studies:
            study = optuna.load_study(study_name=study_summary.study_name, storage=self.storage)
            report.append(f"| {study_summary.study_name} | {study_summary.direction.name} | "
                         f"{study.best_value:.4f} | {len(study.trials)} |\n")
        
        report.append("\n## Best Parameters by Model\n\n")
        
        for study_summary in studies:
            study = optuna.load_study(study_name=study_summary.study_name, storage=self.storage)
            if study.best_params:
                report.append(f"### {study_summary.study_name}\n\n")
                report.append(f"**Best Value:** {study.best_value:.4f}\n\n")
                report.append("**Best Parameters:**\n")
                for param, value in study.best_params.items():
                    report.append(f"- {param}: {value}\n")
                report.append("\n")
        
        return "".join(report)

if __name__ == "__main__":
    optimizer = OptunaOptimizer()
    
    # Load sample data for testing
    from data_pipeline import DataPipeline
    pipeline = DataPipeline()
    
    # Would load real data here
    logger.info("Optimization complete") 