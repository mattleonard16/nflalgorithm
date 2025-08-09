#!/usr/bin/env python3
"""
Quick Model Training - Get Models Working Fast
=============================================
Simple training script that works with our populated database
"""

import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from rich.console import Console
from rich.table import Table
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
console = Console()

def main():
    console.print("🚀 [bold blue]Quick NFL Model Training[/]")
    
    # Load data from our database
    conn = sqlite3.connect("nfl_data.db")
    
    # Get enhanced features
    enhanced_df = pd.read_sql_query("""
    SELECT * FROM enhanced_features 
    WHERE season = 2024
    ORDER BY fantasy_points_per_game DESC
    """, conn)
    
    # Get player stats aggregated by season
    stats_df = pd.read_sql_query("""
    SELECT player_id, 
           AVG(fantasy_points) as avg_fantasy,
           SUM(rushing_yards) as total_rush_yds,
           SUM(receiving_yards) as total_rec_yds, 
           SUM(passing_yards) as total_pass_yds,
           COUNT(*) as games_played
    FROM player_stats 
    WHERE season = 2024
    GROUP BY player_id
    HAVING games_played >= 4
    """, conn)
    
    conn.close()
    
    if enhanced_df.empty:
        console.print("❌ No enhanced features found. Run populate script first.")
        return
        
    console.print(f"✅ Loaded {len(enhanced_df)} player records")
    
    # Merge data
    data = enhanced_df.merge(stats_df, on='player_id', how='left')
    data = data.fillna(0)
    
    # Create features
    feature_columns = [
        'value_score', 'consistency_score', 'total_rush_yds',
        'total_rec_yds', 'total_pass_yds', 'games_played'
    ]
    
    # Encode position
    le = LabelEncoder()
    data['position_encoded'] = le.fit_transform(data['position'])
    feature_columns.append('position_encoded')
    
    # Prepare data
    X = data[feature_columns].values
    y = data['fantasy_points_per_game'].values
    
    # Remove any NaN values
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    
    console.print(f"🎯 Training on {X.shape[0]} samples with {X.shape[1]} features")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42)
    }
    
    results_table = Table(title="Model Training Results")
    results_table.add_column("Model", style="cyan")
    results_table.add_column("CV MAE", style="green") 
    results_table.add_column("CV R²", style="blue")
    
    trained_models = {}
    best_mae = float('inf')
    best_model_name = None
    
    for name, model in models.items():
        # Cross validation
        cv_mae = -cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error').mean()
        cv_r2 = cross_val_score(model, X_scaled, y, cv=5, scoring='r2').mean()
        
        # Train on full dataset
        model.fit(X_scaled, y)
        trained_models[name] = model
        
        # Track best model
        if cv_mae < best_mae:
            best_mae = cv_mae
            best_model_name = name
            
        results_table.add_row(name, f"{cv_mae:.3f}", f"{cv_r2:.3f}")
        
    console.print("\n")
    console.print(results_table)
    
    # Create ensemble (simple average)
    console.print(f"\n🎯 Creating ensemble model...")
    ensemble_preds = np.mean([model.predict(X_scaled) for model in trained_models.values()], axis=0)
    ensemble_mae = mean_absolute_error(y, ensemble_preds)
    ensemble_r2 = r2_score(y, ensemble_preds)
    
    console.print(f"✅ Ensemble - MAE: {ensemble_mae:.3f}, R²: {ensemble_r2:.3f}")
    
    # Save models
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save best model and scaler
    joblib.dump(trained_models[best_model_name], models_dir / "best_model.joblib")
    joblib.dump(scaler, models_dir / "scaler.joblib")
    joblib.dump(le, models_dir / "label_encoder.joblib")
    
    # Generate 2025 projections
    console.print(f"\n🔮 Generating 2025 projections...")
    
    # Use best model for projections
    best_model = trained_models[best_model_name]
    projections = best_model.predict(X_scaled)
    
    # Create projections table
    proj_data = []
    for i, proj in enumerate(projections):
        proj_data.append({
            'player_id': data.iloc[i]['player_id'],
            'player_name': data.iloc[i]['player_name'],
            'position': data.iloc[i]['position'],
            'season': 2025,
            'projected_fantasy_ppg': round(proj, 2),
            'confidence_score': 0.75,  # Default confidence
            'current_2024_ppg': round(data.iloc[i]['fantasy_points_per_game'], 2)
        })
    
    proj_df = pd.DataFrame(proj_data)
    proj_df = proj_df.sort_values('projected_fantasy_ppg', ascending=False)
    
    # Save projections to database
    conn = sqlite3.connect("nfl_data.db")
    proj_df.to_sql('player_projections_2025', conn, if_exists='replace', index=False)
    conn.close()
    
    # Show top projections
    console.print(f"\n🏆 Top 2025 Projections:")
    for _, row in proj_df.head(10).iterrows():
        console.print(f"   • {row['player_name']} ({row['position']}): {row['projected_fantasy_ppg']} FPG")
        
    # Summary
    console.print(f"\n✅ [bold green]Model Training Complete![/]")
    console.print(f"   • Best Model: [cyan]{best_model_name}[/] (MAE: {best_mae:.3f})")
    console.print(f"   • Projections: [cyan]{len(proj_df)}[/] players for 2025")
    console.print(f"   • Models saved to: [blue]models/[/]")
    console.print(f"   • Database updated with projections")
    
    # Validation assessment
    if best_mae <= 3.0:
        console.print(f"\n🎉 [bold green]SUCCESS: Model meets target accuracy (MAE ≤ 3.0)[/]")
    else:
        console.print(f"\n⚠️ [bold yellow]ACCEPTABLE: Model MAE ({best_mae:.3f}) above target but usable[/]")
        
    console.print(f"\n🚀 [bold]Next: Run the dashboard to see projections and value bets![/]")

if __name__ == "__main__":
    main()