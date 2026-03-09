"""Example configuration — copy to config.py and fill in values."""
from types import SimpleNamespace

config = SimpleNamespace(
    database=SimpleNamespace(
        backend="sqlite",
        path="nfl_data.db",
        url="",  # MySQL URL for production
    ),
    api=SimpleNamespace(
        odds_api_key="",  # Get from https://the-odds-api.com
        host="0.0.0.0",
        port=8000,
    ),
    model=SimpleNamespace(
        target_mae=3.0,
        ewma_decay=0.65,
    ),
    betting=SimpleNamespace(
        min_edge_threshold=0.08,
        min_confidence=0.75,
        kelly_fraction=0.25,
    ),
    integration=SimpleNamespace(
        ewma_decay=0.65,
    ),
)
