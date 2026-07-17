"""Pytest configuration for deterministic database backend."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so tests can import top-level modules
# like schema_migrations, config, utils, scripts, etc.
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_private_algorithm_files = (
    Path(_project_root) / "data_pipeline.py",
    Path(_project_root) / "prop_integration.py",
    Path(_project_root) / "value_betting_engine.py",
    Path(_project_root) / "models" / "position_specific" / "weekly.py",
)
_private_algorithm_tests = [
    "test_augmentation_wr.py",
    "test_backtest_replay.py",
    "test_basic.py",
    "test_constraint_handling.py",
    "test_dry_run_validation.py",
    "test_kelly_cap.py",
    "test_market_mu_wr.py",
    "test_merge_suffix_handling.py",
    "test_nfl_weekly_model.py",
    "test_no_vig_probability.py",
    "test_prop_integration_matching.py",
    "test_prop_integration_season_week.py",
    "test_prop_integration_wr.py",
    "test_qb_decomposition.py",
    "test_synthetic_odds_wr.py",
    "test_value_engine_side.py",
    "test_weekly_pipeline.py",
]
collect_ignore = []
if not all(path.is_file() for path in _private_algorithm_files):
    collect_ignore.extend(_private_algorithm_tests)

_private_api_server = Path(_project_root) / "api" / "server.py"
_private_api_tests = [
    "test_api_contract.py",
    "test_export_api.py",
    "test_nba_api.py",
    "test_nba_api_contract.py",
    "test_pipeline_run_api.py",
    "test_record_bet_api.py",
    "test_risk_api.py",
]
if not _private_api_server.is_file():
    collect_ignore.extend(_private_api_tests)

# Force tests to run against a local SQLite database regardless of .env settings.
TEST_DB_DIR = Path(__file__).parent / "_tmp"
TEST_DB_DIR.mkdir(exist_ok=True)
TEST_DB_PATH = TEST_DB_DIR / "test_suite.db"

# Reset the file each session to avoid stale data bleeding into tests.
TEST_DB_PATH.unlink(missing_ok=True)
TEST_DB_PATH.touch()

os.environ["DB_BACKEND"] = "sqlite"
os.environ["SQLITE_DB_PATH"] = str(TEST_DB_PATH)

# Apply the schema deliberately instead of relying on a test-specific
# MigrationManager call to mutate this shared database as a side effect.
from schema_migrations import MigrationManager

MigrationManager(TEST_DB_PATH).run()


# Shared fixture for clearing NBA cache before tests that use TestClient
def clear_nba_cache():
    """Clear NBA endpoint cache to avoid cross-test pollution."""
    from api.cache import nba_cache

    nba_cache.invalidate_all()
