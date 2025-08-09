"""Configuration settings for NFL algorithm system."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

@dataclass
class DatabaseConfig:
    path: str = "nfl_data.db"
    backup_path: str = "data/backups/"
    max_connections: int = 10

@dataclass
class CacheConfig:
    # HTTP Cache Settings (requests-cache)
    http_cache_backend: str = "filesystem"
    http_cache_dir: str = "cache/http"
    http_cache_expire_after: int = 1800  # 30 minutes default
    
    # Database Cache TTL (seconds)
    odds_cache_ttl_season: int = 1800      # 30 minutes during season
    odds_cache_ttl_offseason: int = 21600  # 6 hours off-season
    weather_cache_ttl: int = 3600          # 60 minutes
    weather_cache_ttl_dome: int = 21600    # 360 minutes for dome stadiums
    player_cache_ttl: int = 14400          # 4 hours
    injury_cache_ttl: int = 1800           # 30 minutes
    
    # Stale-While-Revalidate (serve stale data during API failures)
    stale_while_revalidate_window: int = 86400  # 24 hours
    
    # Rate Limiting (Token Bucket)
    rate_limit_tokens_per_minute: int = 60
    rate_limit_burst_capacity: int = 10
    
    # Cache Warm-up
    cache_warm_enabled: bool = True
    cache_warm_popular_markets: List[str] = None
    
    def __post_init__(self):
        if self.cache_warm_popular_markets is None:
            self.cache_warm_popular_markets = [
                "player_props", "spreads", "totals", "moneylines"
            ]

@dataclass
class APIConfig:
    odds_api_key: str = os.getenv("ODDS_API_KEY", "")
    weather_api_key: str = os.getenv("WEATHER_API_KEY", "")
    fantasy_pros_api_key: str = os.getenv("FANTASY_PROS_API_KEY", "")
    rate_limit_requests_per_minute: int = 60
    retry_max_attempts: int = 3
    retry_backoff_factor: float = 2.0
    
    # API Cache Integration
    enable_caching: bool = True
    cache_offline_mode: bool = False
    force_cache_refresh: bool = False

@dataclass
class ModelConfig:
    target_mae: float = 3.0
    min_training_samples: int = 100
    cross_validation_folds: int = 5
    feature_importance_threshold: float = 0.01
    prediction_interval_coverage: float = 0.8

@dataclass
class BettingConfig:
    min_edge_threshold: float = 0.08  # 8%
    min_confidence: float = 0.75
    max_kelly_fraction: float = 0.25
    min_expected_roi: float = 0.12
    sportsbooks: List[str] = None
    
    def __post_init__(self):
        if self.sportsbooks is None:
            self.sportsbooks = ["DraftKings", "FanDuel", "BetMGM", "Caesars"]

@dataclass
class PipelineConfig:
    update_interval_minutes: int = 15
    weather_update_hours: int = 1
    injury_update_minutes: int = 30
    odds_update_minutes: int = 5
    max_concurrent_requests: int = 5

@dataclass
class Config:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    betting: BettingConfig = field(default_factory=BettingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    
    # Paths
    project_root: Path = Path(__file__).parent
    data_dir: Path = project_root / "data"
    models_dir: Path = project_root / "models"
    logs_dir: Path = project_root / "logs"
    reports_dir: Path = project_root / "reports"
    reports_img_dir: Path = project_root / "reports" / "img"
    cache_dir: Path = project_root / "cache"
    
    def __post_init__(self):
        # Create directories if they don't exist
        for path in [self.data_dir, self.models_dir, self.logs_dir, 
                    self.reports_dir, self.reports_img_dir, self.cache_dir]:
            path.mkdir(exist_ok=True)
        
        # Create HTTP cache directory
        http_cache_path = self.cache_dir / "http"
        http_cache_path.mkdir(exist_ok=True)

# Global config instance
config = Config() 