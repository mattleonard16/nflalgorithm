"""
Enhanced Data Validation Pipeline for NFL Algorithm
===============================================

Implements comprehensive data quality validation, error handling,
and recovery mechanisms for robust data integration.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from config import config

logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Custom exception for data validation failures."""
    pass

class DataValidator:
    """Comprehensive data quality validation system."""
    
    def __init__(self):
        self.validation_rules = self._define_validation_rules()
        self.validation_history = []
        
    def _define_validation_rules(self) -> Dict[str, Dict]:
        """Define comprehensive validation rules for different data types."""
        return {
            'player_stats': {
                'required_columns': [
                    'player_id', 'season', 'week', 'name', 'team', 'position',
                    'rushing_yards', 'receiving_yards', 'targets', 'snap_percentage'
                ],
                'numeric_ranges': {
                    'rushing_yards': {'min': 0, 'max': 400},
                    'receiving_yards': {'min': 0, 'max': 400},
                    'targets': {'min': 0, 'max': 25},
                    'snap_percentage': {'min': 0, 'max': 100},
                    'rushing_attempts': {'min': 0, 'max': 50},
                    'receptions': {'min': 0, 'max': 25}
                },
                'categorical_values': {
                    'position': ['RB', 'WR', 'TE', 'QB', 'FB', 'FLEX']
                }
            },
            'odds_data': {
                'required_columns': [
                    'player_id', 'market', 'sportsbook', 'line', 'price',
                    'season', 'week'
                ],
                'numeric_ranges': {
                    'line': {'min': 0.0, 'max': 5000.0},
                    'price': {'min': -5000, 'max': 5000}
                },
                'categorical_values': {
                    'market': ['rushing_yards', 'receiving_yards', 'passing_yards'],
                    'sportsbook': ['DraftKings', 'FanDuel', 'BetMGM', 'Caesars']
                }
            },
            'weather_data': {
                'required_columns': [
                    'game_id', 'temperature', 'wind_speed', 'humidity'
                ],
                'numeric_ranges': {
                    'temperature': {'min': -50, 'max': 120},
                    'wind_speed': {'min': 0, 'max': 100},
                    'humidity': {'min': 0, 'max': 100}
                }
            },
            'injury_data': {
                'required_columns': [
                    'player_id', 'season', 'week', 'status'
                ],
                'categorical_values': {
                    'status': ['ACTIVE', 'DOUBTFUL', 'QUESTIONABLE', 'OUT'],
                    'practice_participation': ['FULL', 'LIMITED', 'DNP']
                }
            }
        }
    
    def validate_dataframe(self, df: pd.DataFrame, data_type: str) -> Tuple[bool, List[str]]:
        """
        Validate a DataFrame against defined rules.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if df.empty:
            return False, ["DataFrame is empty"]
        
        errors = []
        rules = self.validation_rules.get(data_type, {})
        
        # Check required columns
        if 'required_columns' in rules:
            missing_cols = set(rules['required_columns']) - set(df.columns)
            if missing_cols:
                errors.append(f"Missing required columns: {missing_cols}")
        
        # Check numeric ranges
        if 'numeric_ranges' in rules:
            for col, ranges in rules['numeric_ranges'].items():
                if col in df.columns:
                    min_val, max_val = ranges['min'], ranges['max']
                    invalid_values = df[(df[col] < min_val) | (df[col] > max_val)]
                    if not invalid_values.empty:
                        errors.append(
                            f"{col} has {len(invalid_values)} invalid values outside range [{min_val}, {max_val}]"
                        )
        
        # Check categorical values
        if 'categorical_values' in rules:
            for col, valid_values in rules['categorical_values'].items():
                if col in df.columns:
                    invalid_cat = df[~df[col].isin(valid_values)]
                    if not invalid_cat.empty:
                        errors.append(
                            f"{col} has {len(invalid_cat)} invalid categories: {invalid_cat[col].unique()}"
                        )
        
        # Check for null values in critical columns
        critical_cols = rules.get('required_columns', [])
        null_counts = df[critical_cols].isnull().sum()
        critical_nulls = null_counts[null_counts > 0]
        if not critical_nulls.empty:
            errors.append(f"Critical null values found: {critical_nulls.to_dict()}")
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            errors.append(f"Found {duplicates} duplicate rows")
        
        is_valid = len(errors) == 0
        
        # Log validation results
        validation_result = {
            'timestamp': datetime.now().isoformat(),
            'data_type': data_type,
            'rows': len(df),
            'is_valid': is_valid,
            'errors': errors
        }
        self.validation_history.append(validation_result)
        
        if not is_valid:
            logger.warning(f"Data validation failed for {data_type}: {errors}")
        else:
            logger.info(f"Data validation passed for {data_type} ({len(df)} rows)")
        
        return is_valid, errors
    
    def clean_player_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and correct common player stats issues."""
        if df.empty:
            return df
        
        cleaned = df.copy()
        
        # Fix common data entry issues
        cleaned['position'] = cleaned['position'].fillna('FLEX')
        
        # Normalize position names
        position_mapping = {
            'RB': 'RB', 'WR': 'WR', 'TE': 'TE', 'QB': 'QB', 'FB': 'FB',
            'RUNNING BACK': 'RB', 'WIDE RECEIVER': 'WR', 'TIGHT END': 'TE',
            'QUARTERBACK': 'QB', 'FULLBACK': 'FB'
        }
        cleaned['position'] = cleaned['position'].str.upper().map(position_mapping).fillna('FLEX')
        
        # Fix numeric issues (negative values, extreme outliers)
        numeric_cols = ['rushing_yards', 'receiving_yards', 'targets', 'snap_percentage']
        for col in numeric_cols:
            if col in cleaned.columns:
                # Clip values to reasonable ranges
                if col in ['rushing_yards', 'receiving_yards']:
                    cleaned[col] = cleaned[col].clip(lower=0, upper=400)
                elif col == 'targets':
                    cleaned[col] = cleaned[col].clip(lower=0, upper=25)
                elif col == 'snap_percentage':
                    cleaned[col] = cleaned[col].clip(lower=0, upper=100)
        
        # Fill missing numeric values with sensible defaults
        cleaned['targets'] = cleaned['targets'].fillna(0)
        cleaned['snap_percentage'] = cleaned['snap_percentage'].fillna(0)
        
        return cleaned
    
    def validate_apis_response(self, response_data: Dict, api_type: str) -> Tuple[bool, List[str]]:
        """Validate API response data."""
        errors = []
        
        if not isinstance(response_data, dict):
            return False, ["Response data is not a dictionary"]
        
        # Odds API validation
        if api_type == 'odds':
            if 'data' not in response_data:
                errors.append("Missing 'data' key in odds response")
            elif not isinstance(response_data['data'], list):
                errors.append("'data' should be a list")
            elif len(response_data['data']) == 0:
                errors.append("Empty data array in odds response")
        
        # Weather API validation
        elif api_type == 'weather':
            required_weather_keys = ['main', 'wind', 'name']
            missing_keys = set(required_weather_keys) - set(response_data.keys())
            if missing_keys:
                errors.append(f"Missing weather keys: {missing_keys}")
        
        return len(errors) == 0, errors
    
    def check_data_freshness(self, feed_name: str, last_update: datetime, 
                           max_age_minutes: int) -> Tuple[bool, str]:
        """Check if data feed is fresh enough."""
        age_minutes = (datetime.now() - last_update).total_seconds() / 60
        is_fresh = age_minutes <= max_age_minutes
        
        status = "FRESH" if is_fresh else f"STALE ({age_minutes:.1f} minutes old)"
        return is_fresh, status
    
    def generate_validation_report(self) -> str:
        """Generate markdown report of recent validation results."""
        if not self.validation_history:
            return "# No validation history available\n"
        
        recent_validations = self.validation_history[-10:]  # Last 10 validations
        
        report = ["# Data Validation Report\n"]
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"Total validations: {len(self.validation_history)}\n\n")
        
        report.append("## Recent Validations\n\n")
        report.append("| Timestamp | Data Type | Rows | Status | Errors |\n")
        report.append("|-----------|------------|------|--------|--------|\n")
        
        for validation in recent_validations:
            timestamp = validation['timestamp'][:19]  # Remove milliseconds
            data_type = validation['data_type']
            rows = validation['rows']
            status = "✅ PASS" if validation['is_valid'] else "❌ FAIL"
            errors = len(validation['errors'])
            
            report.append(f"| {timestamp} | {data_type} | {rows} | {status} | {errors} |\n")
        
        # Show recent errors
        recent_errors = [v for v in recent_validations if not v['is_valid']]
        if recent_errors:
            report.append("\n## Recent Errors\n\n")
            for validation in recent_errors[-5:]:  # Last 5 error validations
                report.append(f"### {validation['data_type']} - {validation['timestamp']}\n")
                for error in validation['errors']:
                    report.append(f"- {error}\n")
                report.append("\n")
        
        return "".join(report)
    
    def save_validation_report(self) -> None:
        """Save validation report to logs directory."""
        report = self.generate_validation_report()
        report_path = config.logs_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Validation report saved to {report_path}")


class APIErrorHandler:
    """Enhanced error handling and recovery for API failures."""
    
    def __init__(self):
        self.failure_count = {}
        self.last_failure = {}
        self.circuit_breakers = {}
        
    def handle_api_failure(self, api_type: str, endpoint: str, error: Exception) -> Dict[str, Any]:
        """
        Handle API failure with circuit breaker and retry logic.
        
        Returns:
            Dict with handling strategy and next steps
        """
        api_key = f"{api_type}:{endpoint}"
        
        # Track failures
        self.failure_count[api_key] = self.failure_count.get(api_key, 0) + 1
        self.last_failure[api_key] = datetime.now()
        
        # Circuit breaker logic
        failure_threshold = 5
        timeout_minutes = 30
        
        if self.failure_count[api_key] >= failure_threshold:
            self.circuit_breakers[api_key] = datetime.now()
            logger.warning(f"Circuit breaker opened for {api_key} after {failure_threshold} failures")
        
        # Check if circuit breaker should be reset
        if api_key in self.circuit_breakers:
            time_since_open = (datetime.now() - self.circuit_breakers[api_key]).total_seconds() / 60
            if time_since_open > timeout_minutes:
                del self.circuit_breakers[api_key]
                del self.failure_count[api_key]
                logger.info(f"Circuit breaker reset for {api_key}")
        
        return {
            'should_retry': self._should_retry(api_key, error),
            'use_fallback': api_key in self.circuit_breakers,
            'wait_time': self._calculate_wait_time(api_key),
            'failure_count': self.failure_count[api_key],
            'circuit_open': api_key in self.circuit_breakers
        }
    
    def _should_retry(self, api_key: str, error: Exception) -> bool:
        """Determine if request should be retried based on error type and failure history."""
        if api_key in self.circuit_breakers:
            return False
        
        # Don't retry on certain error types
        non_retryable_errors = ['400', '401', '403', '404']
        error_str = str(error)
        for code in non_retryable_errors:
            if code in error_str:
                return False
        
        # Retry on other errors but limit attempts
        return self.failure_count.get(api_key, 0) < 3
    
    def _calculate_wait_time(self, api_key: str) -> int:
        """Calculate exponential backoff wait time."""
        failures = self.failure_count.get(api_key, 0)
        if failures == 0:
            return 0
        return min(300, 30 * (2 ** (failures - 1)))  # Cap at 5 minutes
    
    def get_api_health_status(self) -> Dict[str, Any]:
        """Get overall API health status."""
        total_apis = len(self.failure_count)
        healthy_apis = total_apis - len(self.circuit_breakers)
        
        return {
            'total_endpoints': total_apis,
            'healthy_endpoints': healthy_apis,
            'circuit_breakers_open': len(self.circuit_breakers),
            'api_health_score': healthy_apis / max(1, total_apis),
            'last_updated': datetime.now().isoformat(),
            'failed_endpoints': list(self.circuit_breakers.keys())
        }


# Global instances
data_validator = DataValidator()
api_error_handler = APIErrorHandler()
