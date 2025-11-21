"""
Basic test suite for NFL algorithm components.
"""

import os
import sqlite3
import tempfile
from contextlib import contextmanager
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import config
from data_pipeline import DataPipeline
from value_betting_engine import ValueBettingEngine


@contextmanager
def pipeline_for_tests():
    """Yield a DataPipeline backed by an isolated SQLite database."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original_path = config.database.path
    original_backend = config.database.backend
    env_backend = os.environ.get("DB_BACKEND")
    env_sqlite_path = os.environ.get("SQLITE_DB_PATH")
    try:
        pipeline = DataPipeline(db_path=tmp.name)
        yield pipeline
    finally:
        config.database.path = original_path
        config.database.backend = original_backend
        if env_backend is not None:
            os.environ["DB_BACKEND"] = env_backend
        else:
            os.environ.pop("DB_BACKEND", None)
        if env_sqlite_path is not None:
            os.environ["SQLITE_DB_PATH"] = env_sqlite_path
        else:
            os.environ.pop("SQLITE_DB_PATH", None)
        os.unlink(tmp.name)

class TestDataPipeline:
    """Test data pipeline functionality."""
    
    def test_database_setup(self):
        """Test database schema creation."""
        with pipeline_for_tests() as pipeline:
            pipeline.setup_enhanced_database()

            from utils.db import get_connection

            with get_connection() as conn:
                if isinstance(conn, sqlite3.Connection):
                    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = {row[0] for row in cursor.fetchall()}
                else:  # MySQL path
                    with conn.cursor() as cursor:
                        cursor.execute("SHOW TABLES")
                        tables = {row[0] for row in cursor.fetchall()}

            expected_tables = {
                'player_stats_enhanced',
                'weather_data',
                'injury_data',
                'odds_data',
                'team_context',
                'clv_tracking'
            }

            missing = expected_tables - tables
            assert not missing, f"Missing tables: {missing}"
    
    def test_feature_engineering(self):
        """Test basic feature engineering."""
        # Create sample data
        data = {
            'player_id': ['P1', 'P2'],
            'season': [2023, 2023],
            'week': [1, 1],
            'position': ['RB', 'WR'],
            'age': [25, 27],
            'games_played': [16, 16],
            'rushing_yards': [100, 0],
            'rushing_attempts': [20, 0],
            'receiving_yards': [50, 80],
            'receptions': [5, 8],
            'targets': [7, 12]
        }
        
        df = pd.DataFrame(data)

        with pipeline_for_tests() as pipeline:
            try:
                result = pipeline._engineer_weather_features(df)
                assert 'cold_weather' in result.columns
                assert 'windy_conditions' in result.columns
                assert 'bad_weather' in result.columns
            except Exception as e:
                pytest.fail(f"Feature engineering failed: {e}")

class TestValueBettingEngine:
    """Test value betting engine."""
    
    def test_kelly_calculation(self):
        """Test Kelly criterion calculation."""
        engine = ValueBettingEngine()
        
        # Test positive expected value scenario
        kelly = engine.calculate_fractional_kelly(win_prob=0.6, odds=-110, fraction=0.5)
        assert 0 < kelly <= 0.25, f"Kelly calculation out of range: {kelly}"
        
        # Test negative expected value scenario
        kelly_neg = engine.calculate_fractional_kelly(win_prob=0.4, odds=-110, fraction=0.5)
        assert kelly_neg <= 0, f"Negative EV should give zero Kelly: {kelly_neg}"
    
    def test_value_bet_creation(self):
        """Test value bet object creation."""
        # Create sample prediction data
        predictions = pd.DataFrame({
            'player_id': ['P1'],
            'player_name': ['Test Player'],
            'position': ['RB'],
            'team': ['TEST'],
            'prediction': [85.5],
            'prediction_lower': [80.0],
            'prediction_upper': [91.0],
            'confidence': [0.8]
        })
        
        engine = ValueBettingEngine()
        
        # Test that find_value_opportunities doesn't crash
        try:
            value_bets = engine.find_value_opportunities(predictions)
            assert isinstance(value_bets, list)
        except Exception as e:
            pytest.fail(f"Value bet finding failed: {e}")

class TestConfiguration:
    """Test configuration system."""
    
    def test_config_structure(self):
        """Test that configuration is properly structured."""
        assert hasattr(config, 'database')
        assert hasattr(config, 'api')
        assert hasattr(config, 'model')
        assert hasattr(config, 'betting')
        assert hasattr(config, 'pipeline')
        
        # Test target values
        assert config.model.target_mae == 3.0
        assert config.betting.min_edge_threshold == 0.08
        assert config.betting.min_confidence == 0.75

class TestCrossSeasonValidation:
    """Test cross-season validation components."""
    
    def test_feature_columns(self):
        """Test that feature columns are properly defined."""
        from cross_season_validation import EnhancedCrossSeasonValidator
        
        validator = EnhancedCrossSeasonValidator()
        features = validator.get_enhanced_feature_columns()
        
        assert isinstance(features, list)
        assert len(features) > 10  # Should have substantial feature set
        assert 'age' in features
        assert 'position_encoded' in features

# Integration tests
class TestIntegration:
    """Integration tests for complete workflows."""
    
    @patch('data_pipeline.requests.get')
    def test_pipeline_integration(self, mock_requests):
        """Test basic pipeline integration."""
        # Mock API responses
        mock_response = Mock()
        mock_response.json.return_value = {'main': {'temp': 70}, 'wind': {'speed': 5}}
        mock_response.raise_for_status.return_value = None
        mock_requests.return_value = mock_response
        
        with pipeline_for_tests() as pipeline:
            try:
                pipeline.setup_enhanced_database()
            except Exception as e:
                pytest.fail(f"Pipeline integration failed: {e}")

# Performance tests
class TestPerformance:
    """Performance and benchmark tests."""
    
    def test_feature_engineering_speed(self):
        """Test that feature engineering completes in reasonable time."""
        import time
        
        # Create larger dataset
        n_samples = 1000
        data = {
            'player_id': [f'P{i}' for i in range(n_samples)],
            'season': [2023] * n_samples,
            'week': [1] * n_samples,
            'position': ['RB'] * n_samples,
            'age': np.random.randint(22, 35, n_samples),
            'games_played': [16] * n_samples,
            'rushing_yards': np.random.randint(0, 200, n_samples),
            'rushing_attempts': np.random.randint(0, 30, n_samples),
            'receiving_yards': np.random.randint(0, 100, n_samples),
            'receptions': np.random.randint(0, 10, n_samples),
            'targets': np.random.randint(0, 15, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        from cross_season_validation import EnhancedCrossSeasonValidator
        validator = EnhancedCrossSeasonValidator()
        
        start_time = time.time()
        result = validator.engineer_enhanced_features(df)
        elapsed = time.time() - start_time
        
        # Should complete within 5 seconds for 1000 samples
        assert elapsed < 5.0, f"Feature engineering too slow: {elapsed:.2f}s"
        assert len(result) == n_samples, "Data loss during feature engineering"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
