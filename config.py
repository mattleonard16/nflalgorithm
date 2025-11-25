"""Configuration settings for NFL algorithm system."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv not installed, try to load manually
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

@dataclass
class DatabaseConfig:
    path: str = os.getenv("SQLITE_DB_PATH", "nfl_data.db")
    backup_path: str = "data/backups/"
    max_connections: int = 10
    # Force load from env if not set correctly by default
    backend: str = field(default_factory=lambda: os.getenv("DB_BACKEND", "sqlite"))
    db_url: str = field(default_factory=lambda: os.getenv("DB_URL", ""))

@dataclass
class CacheConfig:
    # HTTP Cache Settings (requests-cache)
    http_cache_backend: str = "filesystem"
    http_cache_dir: str = "cache/http"
    http_cache_expire_after: int = 1800  # 30 minutes default
    
    # Database Cache TTL (seconds) - AGGRESSIVE CACHING to preserve API credits
    odds_cache_ttl_season: int = 172800    # 48 hours during season (preserve API credits)
    odds_cache_ttl_offseason: int = 604800  # 7 days off-season
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
    max_bankroll_fraction: float = 0.02
    kelly_fraction_default: float = 0.5
    daily_loss_stop: float = 0.06
    per_market_unit_cap: float = 0.75
    
    def __post_init__(self):
        if self.sportsbooks is None:
            self.sportsbooks = ["DraftKings", "FanDuel", "BetMGM", "Caesars"]

@dataclass
class FreshnessConfig:
    odds_minutes: int = 2
    injuries_minutes: int = 10
    weather_minutes: int = 30
    player_stats_minutes: int = 60
    projections_minutes: int = 30

@dataclass
class SyntheticConfig:
    # WR synthetic odds tuning
    min_wr_baseline: float = 35.0
    min_wr_line: float = 25.0
    max_wr_line: float = 115.0
    targets_to_yards_factor: float = 8.0
    wr_targets_threshold: float = 3.0
    wr_avg_rec_threshold: float = 12.0
    max_simbook_pwin: float = 0.75
    max_simbook_edge: float = 0.20


@dataclass
class IntegrationConfig:
    # Match tier filtering for join_odds_projections
    # Tiers: 1=player_id exact, 2=name+team match, 3=name-only match
    min_match_tier_real: int = 2       # For real sportsbooks, keep tier<=2 (player_id or name+team)
    min_match_tier_synthetic: int = 3  # For SimBook, allow all tiers including fuzzy name
    # WR-specific team mismatch tolerance
    allow_wr_team_mismatch: bool = True  # Allow WRs to match even if team differs (traded players)

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
    freshness: FreshnessConfig = field(default_factory=FreshnessConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    
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
# config = Config()

def get_config():
    return Config()

config = get_config() 
