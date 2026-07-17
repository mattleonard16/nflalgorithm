"""Tracked environment-driven runtime configuration defaults."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


database = SimpleNamespace(
    backend=os.getenv("DB_BACKEND", "sqlite"),
    path=os.getenv("SQLITE_DB_PATH", str(PROJECT_ROOT / "nfl_data.db")),
    db_url=os.getenv("DB_URL", ""),
)

api = SimpleNamespace(
    odds_api_key=os.getenv("ODDS_API_KEY", ""),
    weather_api_key=os.getenv("WEATHER_API_KEY", ""),
    host=os.getenv("API_HOST", "0.0.0.0"),
    port=int(os.getenv("API_PORT", "8000")),
    enable_caching=env_flag("API_ENABLE_CACHING", True),
    cache_offline_mode=env_flag("API_CACHE_OFFLINE_MODE"),
    force_cache_refresh=env_flag("API_FORCE_CACHE_REFRESH"),
)

cache = SimpleNamespace(
    enabled=True,
    ttl_seconds=3600,
    max_size=500,
    directory=str(PROJECT_ROOT / "cache"),
    http_cache_dir="http_cache",
    http_cache_backend="sqlite",
    http_cache_expire_after=3600,
    rate_limit_burst_capacity=60,
    rate_limit_tokens_per_minute=60,
    odds_cache_ttl_season=30,
    odds_cache_ttl_offseason=300,
    weather_cache_ttl=60,
    weather_cache_ttl_dome=120,
    player_cache_ttl=120,
    cache_warm_enabled=False,
    stale_while_revalidate_window=300,
)

model = SimpleNamespace(
    target_mae=3.0,
    n_estimators=500,
    max_depth=12,
    learning_rate=0.05,
    min_samples_split=8,
    model_dir=str(PROJECT_ROOT / "models"),
    playoff_volume_factor=1.07,
    wildcard_factor=1.03,
    divisional_factor=1.05,
    conference_factor=1.07,
    superbowl_factor=1.10,
    context_sensitivity_high_volume_weight=0.60,
    context_sensitivity_low_sample_threshold=4,
)

betting = SimpleNamespace(
    min_edge_threshold=0.08,
    min_confidence=0.75,
    kelly_fraction=0.25,
    max_kelly=0.10,
    bankroll=1000.0,
    min_odds=-200,
    max_odds=200,
    target_share_min_threshold=0.08,
    volatility_penalty_weight=0.15,
)

pipeline = SimpleNamespace(
    default_seasons=[2024, 2025],
    data_sources=["nflreadpy"],
    update_interval_minutes=15,
    weather_update_interval_minutes=60,
    injury_update_interval_minutes=30,
    odds_max_age_seconds=env_int("NFL_ODDS_MAX_AGE_SECONDS", 300),
    odds_min_event_coverage=env_float("NFL_ODDS_MIN_EVENT_COVERAGE", 1.0),
    odds_min_market_coverage=env_float("NFL_ODDS_MIN_MARKET_COVERAGE", 1.0),
    odds_min_sportsbooks_per_event_market=env_int(
        "NFL_ODDS_MIN_SPORTSBOOKS_PER_EVENT_MARKET", 2
    ),
    odds_required_markets=tuple(
        value.strip()
        for value in os.getenv(
            "NFL_ODDS_REQUIRED_MARKETS",
            "player_pass_yds,player_rush_yds,player_rec_yds",
        ).split(",")
        if value.strip()
    ),
)

integration = SimpleNamespace(
    tier1_confidence=0.95,
    tier2_confidence=0.90,
    tier3_confidence=0.85,
    wr_team_mismatch_tolerance=True,
    ewma_decay=0.65,
    role_priors={"alpha": 75, "secondary": 55, "slot": 45, "fringe": 30},
)

risk = SimpleNamespace(
    max_team_exposure=0.30,
    max_game_exposure=0.40,
    max_player_exposure=0.15,
    monte_carlo_iterations=1000,
    max_drawdown_threshold=0.20,
)

te_market_bias = SimpleNamespace(
    enabled=False,
    adjustment_yards=-3.5,
    playoff_weeks_only=True,
    min_sample_size=10,
    significance_threshold=0.10,
)

confidence = SimpleNamespace(
    edge_weight=0.35,
    stability_weight=0.25,
    volume_weight=0.20,
    volatility_weight=0.20,
    min_tier=2,
)

features = SimpleNamespace(
    no_vig_enabled=env_flag("NFL_FEATURE_NO_VIG"),
    kelly_cap_enabled=env_flag("NFL_FEATURE_KELLY_CAP"),
)

config = SimpleNamespace(
    database=database,
    api=api,
    cache=cache,
    model=model,
    betting=betting,
    pipeline=pipeline,
    integration=integration,
    risk=risk,
    te_market_bias=te_market_bias,
    confidence=confidence,
    features=features,
    project_root=PROJECT_ROOT,
    cache_dir=PROJECT_ROOT / "cache",
    logs_dir=PROJECT_ROOT / "logs",
    reports_dir=PROJECT_ROOT / "reports",
    reports_img_dir=PROJECT_ROOT / "reports" / "images",
)
