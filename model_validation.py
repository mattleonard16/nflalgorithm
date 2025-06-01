import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelValidator:
    def __init__(self, db_path="nfl_data.db"):
        self.db_path = db_path
        self.validation_results = {}
        self.models = {}
    
    def load_data(self, seasons=None):
        """Load player data from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM player_stats"
        if seasons:
            placeholders = ','.join(['?' for _ in seasons])
            query += f" WHERE season IN ({placeholders})"
            df = pd.read_sql_query(query, conn, params=seasons)
        else:
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        logger.info(f"Loaded {len(df)} player records")
        return df
    
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        # Basic feature engineering
        features = df.copy()
        
        # Create derived features
        features['yards_per_attempt'] = features['rushing_yards'] / (features['rushing_attempts'] + 1)
        features['receptions_per_target'] = features['receptions'] / (features['targets'] + 1)
        features['total_touches'] = features['rushing_attempts'] + features['receptions']
        features['total_yards'] = features['rushing_yards'] + features['receiving_yards']
        
        # Position encoding (simplified)
        position_map = {'RB': 1, 'WR': 2, 'TE': 3, 'QB': 4, 'FB': 5}
        features['position_encoded'] = features['position'].map(position_map).fillna(0)
        
        # Select features for modeling
        feature_columns = [
            'age', 'games_played', 'rushing_attempts', 'yards_per_attempt',
            'receptions', 'targets', 'receptions_per_target', 'position_encoded'
        ]
        
        X = features[feature_columns].fillna(0)
        y_rushing = features['rushing_yards']
        y_receiving = features['receiving_yards']
        
        return X, y_rushing, y_receiving, feature_columns
    
    def train_models(self, X, y_rushing, y_receiving):
        """Train machine learning models"""
        logger.info("Training machine learning models...")
        
        # Split data
        X_train, X_test, y_rush_train, y_rush_test = train_test_split(
            X, y_rushing, test_size=0.2, random_state=42
        )
        _, _, y_rec_train, y_rec_test = train_test_split(
            X, y_receiving, test_size=0.2, random_state=42
        )
        
        # Train rushing yards model
        rushing_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rushing_model.fit(X_train, y_rush_train)
        
        # Train receiving yards model
        receiving_model = RandomForestRegressor(n_estimators=100, random_state=42)
        receiving_model.fit(X_train, y_rec_train)
        
        # Store models
        self.models['rushing'] = rushing_model
        self.models['receiving'] = receiving_model
        
        # Predictions
        rush_pred = rushing_model.predict(X_test)
        rec_pred = receiving_model.predict(X_test)
        
        # Calculate metrics
        rushing_metrics = {
            'mae': mean_absolute_error(y_rush_test, rush_pred),
            'rmse': np.sqrt(mean_squared_error(y_rush_test, rush_pred)),
            'r2': r2_score(y_rush_test, rush_pred)
        }
        
        receiving_metrics = {
            'mae': mean_absolute_error(y_rec_test, rec_pred),
            'rmse': np.sqrt(mean_squared_error(y_rec_test, rec_pred)),
            'r2': r2_score(y_rec_test, rec_pred)
        }
        
        return rushing_metrics, receiving_metrics, X_test, y_rush_test, y_rec_test, rush_pred, rec_pred
    
    def feature_importance_analysis(self, feature_columns):
        """Analyze feature importance from trained models"""
        logger.info("Analyzing feature importance...")
        
        if 'rushing' not in self.models or 'receiving' not in self.models:
            logger.error("Models not trained yet")
            return {}
        
        rushing_importance = self.models['rushing'].feature_importances_
        receiving_importance = self.models['receiving'].feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'rushing_importance': rushing_importance,
            'receiving_importance': receiving_importance
        })
        
        importance_df['avg_importance'] = (importance_df['rushing_importance'] + 
                                         importance_df['receiving_importance']) / 2
        importance_df = importance_df.sort_values('avg_importance', ascending=False)
        
        print("\nFeature Importance Analysis:")
        print("-" * 40)
        for _, row in importance_df.iterrows():
            print(f"{row['feature']:<20} | Avg: {row['avg_importance']:.3f} | "
                  f"Rush: {row['rushing_importance']:.3f} | Rec: {row['receiving_importance']:.3f}")
        
        return importance_df
    
    def validate_model_performance(self):
        """Main validation function using real data"""
        logger.info("Starting model validation with real data...")
        
        # Load data
        df = self.load_data()
        
        if df.empty:
            logger.error("No data found in database. Run data collection first.")
            return
        
        print(f"\nModel Validation Results")
        print("=" * 50)
        print(f"Dataset: {len(df)} players from 2023 season")
        
        # Prepare features
        X, y_rushing, y_receiving, feature_columns = self.prepare_features(df)
        
        # Train models
        rushing_metrics, receiving_metrics, X_test, y_rush_test, y_rec_test, rush_pred, rec_pred = self.train_models(
            X, y_rushing, y_receiving
        )
        
        # Display results
        print(f"\nRushing Yards Prediction:")
        print(f"  Mean Absolute Error: {rushing_metrics['mae']:.1f} yards")
        print(f"  Root Mean Square Error: {rushing_metrics['rmse']:.1f} yards")
        print(f"  R² Score: {rushing_metrics['r2']:.3f}")
        
        print(f"\nReceiving Yards Prediction:")
        print(f"  Mean Absolute Error: {receiving_metrics['mae']:.1f} yards")
        print(f"  Root Mean Square Error: {receiving_metrics['rmse']:.1f} yards")
        print(f"  R² Score: {receiving_metrics['r2']:.3f}")
        
        # Feature importance
        importance_df = self.feature_importance_analysis(feature_columns)
        
        # Store results
        self.validation_results = {
            'rushing_metrics': rushing_metrics,
            'receiving_metrics': receiving_metrics,
            'feature_importance': importance_df,
            'predictions': {
                'rush_actual': y_rush_test,
                'rush_predicted': rush_pred,
                'rec_actual': y_rec_test,
                'rec_predicted': rec_pred
            }
        }
        
        return self.validation_results
    
    def simulate_betting_strategy(self):
        """Simulate a basic betting strategy"""
        if not self.validation_results:
            logger.error("Run model validation first")
            return
        
        print(f"\nBetting Strategy Simulation")
        print("-" * 30)
        
        # Get predictions vs actual
        rush_actual = self.validation_results['predictions']['rush_actual']
        rush_pred = self.validation_results['predictions']['rush_predicted']
        
        # Simple strategy: bet when model predicts significantly different from a threshold
        betting_threshold = 100  # yards
        confidence_margin = 20   # yards difference needed to bet
        
        bets_made = 0
        successful_bets = 0
        
        for actual, predicted in zip(rush_actual, rush_pred):
            # Simulate betting over/under on rushing yards
            if abs(predicted - betting_threshold) > confidence_margin:
                bets_made += 1
                
                # Bet over if predicted > threshold + margin
                if predicted > betting_threshold + confidence_margin:
                    if actual > betting_threshold:
                        successful_bets += 1
                # Bet under if predicted < threshold - margin
                elif predicted < betting_threshold - confidence_margin:
                    if actual < betting_threshold:
                        successful_bets += 1
        
        if bets_made > 0:
            hit_rate = successful_bets / bets_made
            print(f"Simulated Betting Results:")
            print(f"  Total Bets: {bets_made}")
            print(f"  Successful Bets: {successful_bets}")
            print(f"  Hit Rate: {hit_rate:.1%}")
            print(f"  Estimated ROI: {(hit_rate - 0.55) * 100:.1f}%")  # Assuming -110 odds
        else:
            print("No bets would have been made with current strategy")

# Run validation if script is executed directly
if __name__ == "__main__":
    validator = ModelValidator()
    
    # Run validation
    results = validator.validate_model_performance()
    
    if results:
        # Run betting simulation
        validator.simulate_betting_strategy()
        
        print(f"\n{'='*60}")
        print("MODEL VALIDATION COMPLETE")
        print("Next steps:")
        print("1. Collect more historical seasons for better training")
        print("2. Add more sophisticated features (team stats, weather, etc.)")
        print("3. Implement real betting line integration")
        print("4. Add cross-validation and hyperparameter tuning")