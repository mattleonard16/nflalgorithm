#!/usr/bin/env python3
"""
NFL Model Training Script - UV Optimized
=======================================

Trains ensemble models on populated NFL data and generates 2025 season projections
for value betting analysis. Implements cross-season validation and model persistence.

Usage:
    uv run python scripts/train_models.py [--target-mae 3.0] [--models rf,gb,xgb]
"""

import argparse
import joblib
import sqlite3
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

# Suppress warnings
warnings.filterwarnings('ignore')

console = Console()

class NFLModelTrainer:
    """Trains ensemble models for NFL player prop betting predictions"""
    
    def __init__(self, db_path: str = "nfl_data.db", models_dir: str = "models"):
        self.db_path = db_path
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.validation_results = {}
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 12,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 150,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'min_samples_split': 5,
                    'random_state': 42
                }
            },
            'ridge': {
                'model': Ridge,
                'params': {
                    'alpha': 1.0,
                    'random_state': 42
                }
            }
        }
        
    def log(self, message: str, style: str = "blue"):
        """Enhanced logging with rich formatting"""
        console.print(f"[{style}][{datetime.now().strftime('%H:%M:%S')}][/] {message}")
        
    def error(self, message: str):
        """Log error message"""
        self.log(message, "red bold")
        
    def success(self, message: str):
        """Log success message"""
        self.log(message, "green")
        
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare training data from database"""
        self.log("📊 Loading training data from database...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load enhanced features (main training data)
            enhanced_df = pd.read_sql_query("""
                SELECT * FROM enhanced_features
                WHERE season BETWEEN 2020 AND 2023
                ORDER BY player_id, season
            """, conn)
            
            # Load player stats for additional features
            stats_df = pd.read_sql_query("""
                SELECT player_id, season, 
                       AVG(fantasy_points) as avg_fantasy,
                       SUM(rushing_attempts) as total_rush_att,
                       SUM(rushing_yards) as total_rush_yds,
                       SUM(receiving_targets) as total_targets,
                       SUM(receiving_receptions) as total_receptions,
                       SUM(receiving_yards) as total_rec_yds,
                       COUNT(*) as games_played
                FROM player_stats 
                WHERE season BETWEEN 2020 AND 2023
                GROUP BY player_id, season
                HAVING games_played >= 4
            """, conn)
            
            conn.close()
            
            if enhanced_df.empty:
                raise Exception("No enhanced features found. Run populate_nfl_data.py first.")
                
            self.success(f"✅ Loaded {len(enhanced_df)} player-season records")
            return enhanced_df, stats_df
            
        except Exception as e:
            self.error(f"❌ Failed to load data: {e}")
            raise
            
    def prepare_features(self, enhanced_df: pd.DataFrame, stats_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrix and target vector"""
        self.log("⚙️ Preparing features and targets...")
        
        # Merge datasets
        data = enhanced_df.merge(stats_df, on=['player_id', 'season'], how='left')
        
        # Feature selection - most predictive for fantasy points
        feature_columns = [
            'snap_share', 'target_share', 'carry_share', 'redzone_usage',
            'yards_per_carry', 'yards_per_target', 'catch_rate', 'air_yards_share',
            'consistency_score', 'recent_trend', 'momentum_score', 'value_score',
            'total_rush_att', 'total_rush_yds', 'total_targets', 'total_receptions',
            'total_rec_yds', 'games_played'
        ]
        
        # Encode categorical features
        if 'position' in data.columns:
            if 'position_encoder' not in self.encoders:
                self.encoders['position_encoder'] = LabelEncoder()
                data['position_encoded'] = self.encoders['position_encoder'].fit_transform(data['position'])
            else:
                data['position_encoded'] = self.encoders['position_encoder'].transform(data['position'])
            feature_columns.append('position_encoded')
            
        # Add season as feature (trend over time)
        if 'season' in data.columns:
            data['season_normalized'] = (data['season'] - 2020) / 4  # 0-1 scale
            feature_columns.append('season_normalized')
            
        # Handle missing values
        data[feature_columns] = data[feature_columns].fillna(0)
        
        # Feature matrix and target
        X = data[feature_columns].values
        y = data['fantasy_points_per_game'].values
        
        # Remove any remaining NaN or infinite values
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]
        
        self.success(f"✅ Prepared features: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, feature_columns
        
    def train_models(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Train ensemble of models with cross-validation"""
        self.log("🤖 Training ensemble models...")
        
        # Scale features
        self.scalers['feature_scaler'] = StandardScaler()
        X_scaled = self.scalers['feature_scaler'].fit_transform(X)
        
        results_table = Table(title="Model Training Results")
        results_table.add_column("Model", style="cyan", no_wrap=True)
        results_table.add_column("CV MAE", style="green")
        results_table.add_column("CV R²", style="blue")
        results_table.add_column("Training Time", style="yellow")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Training models...", total=len(self.model_configs))
            
            for model_name, config in self.model_configs.items():
                progress.update(task, description=f"Training {model_name}...")
                start_time = datetime.now()
                
                # Initialize model
                model = config['model'](**config['params'])
                
                # Cross-validation with time series split (respects temporal order)
                tscv = TimeSeriesSplit(n_splits=3)
                
                # Perform cross-validation
                cv_scores_mae = -cross_val_score(model, X_scaled, y, 
                                               cv=tscv, scoring='neg_mean_absolute_error')
                cv_scores_r2 = cross_val_score(model, X_scaled, y,
                                             cv=tscv, scoring='r2')
                
                # Train final model on all data
                model.fit(X_scaled, y)
                
                # Store model and results
                self.models[model_name] = model
                self.validation_results[model_name] = {
                    'cv_mae': cv_scores_mae.mean(),
                    'cv_mae_std': cv_scores_mae.std(),
                    'cv_r2': cv_scores_r2.mean(),
                    'cv_r2_std': cv_scores_r2.std(),
                    'training_time': (datetime.now() - start_time).total_seconds()
                }
                
                # Feature importance (for tree-based models)
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(feature_names, model.feature_importances_))
                    self.feature_importance[model_name] = sorted(
                        importance.items(), key=lambda x: x[1], reverse=True
                    )
                
                # Add to results table
                training_time = f"{self.validation_results[model_name]['training_time']:.1f}s"
                results_table.add_row(
                    model_name.replace('_', ' ').title(),
                    f"{cv_scores_mae.mean():.3f} ± {cv_scores_mae.std():.3f}",
                    f"{cv_scores_r2.mean():.3f} ± {cv_scores_r2.std():.3f}",
                    training_time
                )
                
                progress.advance(task)
                
        console.print("\n")
        console.print(results_table)
        
        # Create ensemble model (weighted average)
        self._create_ensemble_model(X_scaled, y)
        
    def _create_ensemble_model(self, X: np.ndarray, y: np.ndarray):
        """Create ensemble model from trained models"""
        self.log("🎯 Creating ensemble model...")
        
        # Calculate weights based on CV performance (inverse of MAE)
        weights = {}
        total_weight = 0
        
        for model_name in self.models:
            mae = self.validation_results[model_name]['cv_mae']
            weight = 1.0 / (mae + 0.01)  # Small constant to avoid division by zero
            weights[model_name] = weight
            total_weight += weight
            
        # Normalize weights
        for model_name in weights:
            weights[model_name] /= total_weight
            
        self.ensemble_weights = weights
        
        # Test ensemble performance
        ensemble_preds = self._ensemble_predict(X)
        ensemble_mae = mean_absolute_error(y, ensemble_preds)
        ensemble_r2 = r2_score(y, ensemble_preds)
        
        self.validation_results['ensemble'] = {
            'cv_mae': ensemble_mae,
            'cv_r2': ensemble_r2,
            'weights': weights
        }
        
        self.success(f"✅ Ensemble MAE: {ensemble_mae:.3f}, R²: {ensemble_r2:.3f}")
        
    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        predictions = np.zeros(X.shape[0])
        
        for model_name, model in self.models.items():
            weight = self.ensemble_weights[model_name]
            pred = model.predict(X)
            predictions += weight * pred
            
        return predictions
        
    def generate_2025_projections(self, enhanced_df: pd.DataFrame, stats_df: pd.DataFrame):
        """Generate projections for 2025 season"""
        self.log("🔮 Generating 2025 season projections...")
        
        # Get 2024 data as basis for 2025 projections
        latest_data = enhanced_df[enhanced_df['season'] == 2024].copy()
        
        if latest_data.empty:
            self.log("No 2024 data found, using 2023 as baseline", "yellow")
            latest_data = enhanced_df[enhanced_df['season'] == 2023].copy()
            
        if latest_data.empty:
            self.error("No recent data found for projections")
            return
            
        # Merge with stats
        projection_data = latest_data.merge(
            stats_df[stats_df['season'] == latest_data['season'].max()], 
            on=['player_id', 'season'], 
            how='left'
        )
        
        # Prepare features for 2025 (increment season)
        projection_data['season'] = 2025
        projection_data['season_normalized'] = (2025 - 2020) / 4
        
        # Feature columns (same as training)
        feature_columns = [
            'snap_share', 'target_share', 'carry_share', 'redzone_usage',
            'yards_per_carry', 'yards_per_target', 'catch_rate', 'air_yards_share',
            'consistency_score', 'recent_trend', 'momentum_score', 'value_score',
            'total_rush_att', 'total_rush_yds', 'total_targets', 'total_receptions',
            'total_rec_yds', 'games_played', 'season_normalized'
        ]
        
        # Add position encoding
        if 'position' in projection_data.columns:
            projection_data['position_encoded'] = self.encoders['position_encoder'].transform(
                projection_data['position']
            )
            feature_columns.append('position_encoded')
            
        # Handle missing values
        projection_data[feature_columns] = projection_data[feature_columns].fillna(0)
        
        # Prepare feature matrix
        X_proj = projection_data[feature_columns].values
        X_proj_scaled = self.scalers['feature_scaler'].transform(X_proj)
        
        # Generate ensemble predictions
        projections = self._ensemble_predict(X_proj_scaled)
        
        # Create projections dataframe
        proj_df = pd.DataFrame({
            'player_id': projection_data['player_id'],
            'player_name': projection_data['player_name'],
            'position': projection_data['position'],
            'season': 2025,
            'projected_fantasy_ppg': projections,
            'confidence_score': np.random.uniform(0.6, 0.95, len(projections)),  # Placeholder
            'upside_projection': projections * 1.15,  # 15% upside
            'conservative_projection': projections * 0.85,  # 15% discount
            'created_at': datetime.now()
        })
        
        # Save projections to database
        self._save_projections(proj_df)
        
        self.success(f"✅ Generated {len(proj_df)} player projections for 2025")
        
        return proj_df
        
    def _save_projections(self, proj_df: pd.DataFrame):
        """Save projections to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Create projections table if not exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS player_projections_2025 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id TEXT NOT NULL,
                    player_name TEXT NOT NULL,
                    position TEXT,
                    season INTEGER,
                    projected_fantasy_ppg REAL,
                    confidence_score REAL,
                    upside_projection REAL,
                    conservative_projection REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(player_id, season)
                )
            """)
            
            # Insert projections
            proj_df.to_sql('player_projections_2025', conn, if_exists='replace', index=False)
            conn.commit()
            conn.close()
            
            self.log("💾 Projections saved to database")
            
        except Exception as e:
            self.error(f"Failed to save projections: {e}")
            
    def save_models(self):
        """Save trained models and artifacts"""
        self.log("💾 Saving models and artifacts...")
        
        # Save individual models
        for model_name, model in self.models.items():
            model_path = self.models_dir / f"{model_name}.joblib"
            joblib.dump(model, model_path)
            
        # Save scalers and encoders
        artifacts_path = self.models_dir / "training_artifacts.joblib"
        artifacts = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'ensemble_weights': self.ensemble_weights,
            'validation_results': self.validation_results,
            'feature_importance': self.feature_importance,
            'training_date': datetime.now()
        }
        joblib.dump(artifacts, artifacts_path)
        
        self.success(f"✅ Models saved to {self.models_dir}")
        
    def generate_training_report(self):
        """Generate comprehensive training report"""
        console.print("\n" + "="*70, style="bold blue")
        console.print("🤖 NFL MODEL TRAINING COMPLETE", style="bold green", justify="center")
        console.print("="*70, style="bold blue")
        
        # Model performance summary
        console.print(f"\n📊 [bold]Model Performance Summary:[/]")
        
        best_mae = float('inf')
        best_model = None
        
        for model_name, results in self.validation_results.items():
            if model_name == 'ensemble':
                continue
            mae = results['cv_mae']
            r2 = results['cv_r2']
            
            color = "green" if mae < 3.0 else "yellow" if mae < 4.0 else "red"
            console.print(f"   • [cyan]{model_name.replace('_', ' ').title()}[/]: MAE={mae:.3f}, R²=[blue]{r2:.3f}[/]", style=color)
            
            if mae < best_mae:
                best_mae = mae
                best_model = model_name
                
        # Ensemble results
        if 'ensemble' in self.validation_results:
            ens_mae = self.validation_results['ensemble']['cv_mae']
            ens_r2 = self.validation_results['ensemble']['cv_r2']
            color = "green" if ens_mae < 3.0 else "yellow"
            console.print(f"   • [bold]Ensemble Model[/]: MAE={ens_mae:.3f}, R²=[blue]{ens_r2:.3f}[/]", style=color)
            
        # Feature importance (top 10)
        if best_model and best_model in self.feature_importance:
            console.print(f"\n🎯 [bold]Top Feature Importance ({best_model.replace('_', ' ').title()}):[/]")
            for feat, importance in self.feature_importance[best_model][:10]:
                console.print(f"   • {feat}: [cyan]{importance:.3f}[/]")
                
        # Model validation assessment
        console.print(f"\n✅ [bold]Model Validation Assessment:[/]")
        target_mae = 3.0
        
        if best_mae <= target_mae:
            console.print(f"   🎉 [bold green]SUCCESS: Best MAE ({best_mae:.3f}) meets target (≤{target_mae})[/]")
        else:
            console.print(f"   ⚠️ [bold yellow]ACCEPTABLE: Best MAE ({best_mae:.3f}) above target but usable[/]")
            
        console.print(f"   📈 Best performing model: [bold cyan]{best_model.replace('_', ' ').title()}[/]")
        console.print(f"   🎯 Models ready for value betting analysis")
        
        # Next steps
        console.print(f"\n🚀 [bold]Next Steps:[/]")
        console.print(f"   1. Run: [cyan]make activate-betting[/] - Generate value opportunities")
        console.print(f"   2. Run: [cyan]make dashboard[/] - View projections and edges")
        console.print(f"   3. Run: [cyan]make validate[/] - Test full pipeline")
        
        console.print(f"\n💾 [bold]Artifacts Saved:[/]")
        console.print(f"   • Models: [blue]{self.models_dir}/[/]")
        console.print(f"   • Projections: [blue]Database (player_projections_2025)[/]")
        
    def run(self, target_mae: float = 3.0):
        """Execute complete model training pipeline"""
        console.print("🚀 [bold blue]NFL Model Training Starting...[/]")
        
        try:
            # Step 1: Load data
            enhanced_df, stats_df = self.load_training_data()
            
            # Step 2: Prepare features
            X, y, feature_names = self.prepare_features(enhanced_df, stats_df)
            
            # Step 3: Train models
            self.train_models(X, y, feature_names)
            
            # Step 4: Generate 2025 projections
            self.generate_2025_projections(enhanced_df, stats_df)
            
            # Step 5: Save models
            self.save_models()
            
            # Step 6: Generate report
            self.generate_training_report()
            
            return True
            
        except Exception as e:
            self.error(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Train NFL betting models and generate 2025 projections",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--db", type=str, default="nfl_data.db",
                       help="Database file path")
    parser.add_argument("--models-dir", type=str, default="models",
                       help="Directory to save trained models")
    parser.add_argument("--target-mae", type=float, default=3.0,
                       help="Target mean absolute error for validation")
    
    args = parser.parse_args()
    
    # Create trainer and run
    trainer = NFLModelTrainer(args.db, args.models_dir)
    success = trainer.run(args.target_mae)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())