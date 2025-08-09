"""
Base model class for position-specific NFL player performance prediction.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logger = logging.getLogger(__name__)

class BasePositionModel(ABC):
    """Abstract base class for position-specific models."""
    
    def __init__(self, position: str):
        self.position = position
        self.models: Dict[str, BaseEstimator] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        self.is_fitted = False
        
    @abstractmethod
    def get_position_features(self) -> List[str]:
        """Return position-specific feature columns."""
        pass
    
    @abstractmethod
    def get_target_columns(self) -> List[str]:
        """Return target columns for this position."""
        pass
    
    @abstractmethod
    def engineer_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer position-specific features."""
        pass
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for modeling."""
        # Filter for position
        position_data = df[df['position'] == self.position].copy()
        
        if position_data.empty:
            logger.warning(f"No data found for position {self.position}")
            return position_data
        
        # Engineer position-specific features
        features_df = self.engineer_position_features(position_data)
        
        # Get feature columns
        self.feature_columns = self.get_position_features()
        self.target_columns = self.get_target_columns()
        
        return features_df
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'BasePositionModel':
        """Fit the model to training data."""
        # Prepare features
        features_df = self.prepare_features(X)
        
        if features_df.empty:
            raise ValueError(f"No data available for position {self.position}")
        
        X_features = features_df[self.feature_columns].fillna(0)
        
        # Train separate models for each target
        for target in self.target_columns:
            if target not in y.columns:
                logger.warning(f"Target {target} not found in data")
                continue
            
            y_target = y[target]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_features)
            
            # Create and fit model
            model = self._create_model()
            model.fit(X_scaled, y_target)
            
            # Store model and scaler
            self.models[target] = model
            self.scalers[target] = scaler
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare features
        features_df = self.prepare_features(X)
        
        if features_df.empty:
            return pd.DataFrame()
        
        X_features = features_df[self.feature_columns].fillna(0)
        
        predictions = {}
        
        for target in self.target_columns:
            if target not in self.models:
                continue
            
            # Scale features
            X_scaled = self.scalers[target].transform(X_features)
            
            # Make prediction
            pred = self.models[target].predict(X_scaled)
            predictions[target] = pred
        
        return pd.DataFrame(predictions, index=X_features.index)
    
    def predict_with_uncertainty(self, X: pd.DataFrame, confidence_level: float = 0.8) -> pd.DataFrame:
        """Make predictions with uncertainty intervals."""
        predictions = self.predict(X)
        
        # Simple approach: use historical residuals for uncertainty
        # In practice, you'd use quantile regression or similar
        uncertainty = {}
        for target in predictions.columns:
            std_pred = np.std(predictions[target])
            margin = std_pred * 1.28  # 80% confidence interval
            uncertainty[f"{target}_lower"] = predictions[target] - margin
            uncertainty[f"{target}_upper"] = predictions[target] + margin
        
        result = predictions.copy()
        for key, values in uncertainty.items():
            result[key] = values
        
        return result
    
    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Evaluate model performance."""
        predictions = self.predict(X)
        
        results = {}
        for target in self.target_columns:
            if target not in predictions.columns or target not in y.columns:
                continue
            
            y_true = y[target]
            y_pred = predictions[target]
            
            results[target] = {
                'mae': mean_absolute_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred),
                'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            }
        
        return results
    
    def feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance for tree-based models."""
        importance = {}
        
        for target, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance[target] = model.feature_importances_
            else:
                logger.warning(f"Model for {target} does not support feature importance")
        
        return importance
    
    def save(self, filepath: str) -> None:
        """Save model to disk."""
        model_data = {
            'position': self.position,
            'models': self.models,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BasePositionModel':
        """Load model from disk."""
        model_data = joblib.load(filepath)
        
        # Create instance
        instance = cls(model_data['position'])
        instance.models = model_data['models']
        instance.scalers = model_data['scalers']
        instance.feature_columns = model_data['feature_columns']
        instance.target_columns = model_data['target_columns']
        instance.is_fitted = model_data['is_fitted']
        
        return instance
    
    @abstractmethod
    def _create_model(self) -> BaseEstimator:
        """Create the underlying ML model."""
        pass 