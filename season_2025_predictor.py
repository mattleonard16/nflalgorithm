#!/usr/bin/env python3
"""
2024/2025 NFL Season Predictor
Use trained model to predict player performance and find betting value
"""

import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Season2025Predictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.train_model()
    
    def train_model(self):
        """Train the production model on all historical data"""
        logger.info("Training production model for 2024/2025 predictions...")
        
        # Load historical data
        conn = sqlite3.connect("nfl_data.db")
        df = pd.read_sql_query("SELECT * FROM player_stats ORDER BY season", conn)
        conn.close()
        
        # Prepare features (same as cross_season_validation.py)
        features_df = self.prepare_features(df)
        
        self.feature_cols = [
            'age', 'age_squared', 'is_prime', 'is_veteran',
            'games_played', 'position_encoded',
            'rushing_attempts', 'yards_per_attempt', 'rush_usage',
            'receptions', 'targets', 'receptions_per_target', 'target_usage',
            'rushing_yards_prev', 'receiving_yards_prev',
            'rushing_attempts_prev', 'receptions_prev',
            'rushing_yards_career_avg', 'receiving_yards_career_avg'
        ]
        
        X = features_df[self.feature_cols].fillna(0)
        y_rush = features_df['rushing_yards']
        y_rec = features_df['receiving_yards']
        
        # Train models
        self.rush_model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=12)
        self.rec_model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=12)
        
        self.rush_model.fit(X, y_rush)
        self.rec_model.fit(X, y_rec)
        
        logger.info(f"Model trained on {len(df)} historical records")
    
    def prepare_features(self, df):
        """Prepare features for prediction (same logic as training)"""
        features = df.copy()
        
        # Basic derived features
        features['yards_per_attempt'] = features['rushing_yards'] / (features['rushing_attempts'] + 1)
        features['receptions_per_target'] = features['receptions'] / (features['targets'] + 1)
        features['total_touches'] = features['rushing_attempts'] + features['receptions']
        features['total_yards'] = features['rushing_yards'] + features['receiving_yards']
        
        # Position encoding
        position_map = {'RB': 1, 'WR': 2, 'TE': 3, 'QB': 4, 'FB': 5}
        features['position_encoded'] = features['position'].map(position_map).fillna(0)
        
        # Sort for lag features
        features = features.sort_values(['player_id', 'season']).reset_index(drop=True)
        
        # Previous season stats
        lag_cols = ['rushing_yards', 'receiving_yards', 'rushing_attempts', 'receptions']
        for col in lag_cols:
            features[f'{col}_prev'] = features.groupby('player_id')[col].shift(1).fillna(0)
        
        # Career averages
        for col in lag_cols:
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
    
    def predict_player_season(self, player_data):
        """Predict full season stats for a player"""
        
        # Convert single player to DataFrame
        if isinstance(player_data, dict):
            player_df = pd.DataFrame([player_data])
        else:
            player_df = player_data.copy()
        
        # Prepare features
        features_df = self.prepare_features(player_df)
        X = features_df[self.feature_cols].fillna(0)
        
        # Make predictions
        rush_pred = self.rush_model.predict(X)[0]
        rec_pred = self.rec_model.predict(X)[0]
        
        return {
            'predicted_rushing_yards': max(0, int(rush_pred)),
            'predicted_receiving_yards': max(0, int(rec_pred)),
            'total_predicted_yards': max(0, int(rush_pred + rec_pred))
        }
    
    def create_2025_projections(self):
        """Create projections for top 2024 players (using 2023 as baseline)"""
        
        # Load 2023 data as baseline for 2024 projections
        conn = sqlite3.connect("nfl_data.db")
        df_2023 = pd.read_sql_query("SELECT * FROM player_stats WHERE season = 2023", conn)
        conn.close()
        
        # Project these players into 2024 (age them by 1 year)
        projections = []
        
        # Focus on key fantasy players
        key_players = df_2023[
            (df_2023['rushing_yards'] > 200) | 
            (df_2023['receiving_yards'] > 400) |
            (df_2023['position'].isin(['RB', 'WR', 'TE']))
        ].copy()
        
        key_players['age'] += 1  # Age players by 1 year for 2024
        key_players['season'] = 2024  # Project to 2024
        
        for _, player in key_players.iterrows():
            player_dict = player.to_dict()
            
            try:
                prediction = self.predict_player_season(player_dict)
                
                projections.append({
                    'name': player['name'],
                    'position': player['position'],
                    'team': player['team'],
                    'age_2024': player['age'],
                    '2023_rush_yds': player['rushing_yards'],
                    '2023_rec_yds': player['receiving_yards'],
                    '2024_proj_rush': prediction['predicted_rushing_yards'],
                    '2024_proj_rec': prediction['predicted_receiving_yards'],
                    '2024_proj_total': prediction['total_predicted_yards'],
                    'rush_change': prediction['predicted_rushing_yards'] - player['rushing_yards'],
                    'rec_change': prediction['predicted_receiving_yards'] - player['receiving_yards']
                })
            except Exception as e:
                logger.warning(f"Could not predict for {player['name']}: {e}")
                continue
        
        return pd.DataFrame(projections)
    
    def find_betting_opportunities(self, projections_df):
        """Identify potential betting opportunities"""
        
        print("\n🎯 2024 BETTING OPPORTUNITIES")
        print("=" * 60)
        
        # Players with significant projected increases
        big_increasers = projections_df[
            (projections_df['rush_change'] > 200) | 
            (projections_df['rec_change'] > 200)
        ].sort_values('rush_change', ascending=False)
        
        if not big_increasers.empty:
            print("\n📈 PROJECTED BREAKOUT CANDIDATES (Rushing):")
            for _, player in big_increasers.head(5).iterrows():
                print(f"  {player['name']} ({player['position']}, {player['team']})")
                print(f"    2023: {player['2023_rush_yds']} rush yds → 2024 proj: {player['2024_proj_rush']} (+{player['rush_change']})")
        
        # Players with significant projected decreases (fade candidates)
        big_decreasers = projections_df[
            (projections_df['rush_change'] < -200) | 
            (projections_df['rec_change'] < -200)
        ].sort_values('rush_change', ascending=True)
        
        if not big_decreasers.empty:
            print("\n📉 PROJECTED REGRESSION CANDIDATES:")
            for _, player in big_decreasers.head(5).iterrows():
                print(f"  {player['name']} ({player['position']}, {player['team']})")
                print(f"    2023: {player['2023_rush_yds']} rush yds → 2024 proj: {player['2024_proj_rush']} ({player['rush_change']})")
        
        # Top total yards projections
        top_producers = projections_df.nlargest(10, '2024_proj_total')
        print(f"\n🏆 TOP 10 PROJECTED TOTAL YARDS (2024):")
        for i, (_, player) in enumerate(top_producers.iterrows(), 1):
            print(f"  {i:2d}. {player['name']:<20} ({player['position']}) - {player['2024_proj_total']} total yards")
    
    def save_predictions(self, projections_df):
        """Save predictions to CSV for analysis"""
        filename = "2024_nfl_projections.csv"
        projections_df.to_csv(filename, index=False)
        print(f"\n💾 Projections saved to {filename}")

def main():
    print("2024/2025 NFL Season Predictor")
    print("=" * 50)
    
    predictor = Season2025Predictor()
    
    # Generate projections
    print("\n🔮 Generating 2024 season projections...")
    projections = predictor.create_2025_projections()
    
    print(f"✅ Generated projections for {len(projections)} players")
    
    # Find betting opportunities
    predictor.find_betting_opportunities(projections)
    
    # Save results
    predictor.save_predictions(projections)
    
    print(f"\n" + "="*60)
    print("2024 SEASON PREDICTIONS COMPLETE")
    print("🎯 Use these projections to:")
    print("  1. Compare against sportsbook season totals")
    print("  2. Identify undervalued players in fantasy drafts")
    print("  3. Find betting edges on player props")
    print("  4. Track actual performance vs predictions")

if __name__ == "__main__":
    main() 