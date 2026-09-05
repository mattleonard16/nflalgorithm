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

# Tests are fenced per module, not as one group: publishing a single private
# module must unlock exactly its own tests. A file needed by two modules is
# skipped when either is absent.
_private_module_tests = {
    Path(_project_root) / "data_pipeline.py": (
        "test_augmentation_wr.py",
        "test_basic.py",
        "test_market_mu_wr.py",
        "test_qb_decomposition.py",
        "test_qb_gating.py",
        "test_synthetic_odds_wr.py",
        "test_weekly_pipeline.py",
    ),
    Path(_project_root) / "value_betting_engine.py": (
        "test_backtest_replay.py",
        "test_basic.py",
        "test_constraint_handling.py",
        "test_dry_run_validation.py",
        "test_kelly_cap.py",
        "test_no_vig_probability.py",
        "test_value_engine_side.py",
        "test_weekly_pipeline.py",
    ),
    Path(_project_root) / "models" / "position_specific" / "weekly.py": (
        "test_nfl_weekly_model.py",
        "test_qb_gating.py",
    ),
}
collect_ignore = sorted(
    {test for path, tests in _private_module_tests.items() if not path.is_file() for test in tests}
)

# The suite shares one process and one TestClient address, so the API rate
# limiter would 429 unrelated tests once the global budget is spent. Disable
# both tiers here; tests/test_rate_limit.py constructs its middleware with
# explicit limits and is unaffected.
os.environ.setdefault("RATE_LIMIT_AUTH_PER_MIN", "0")
os.environ.setdefault("RATE_LIMIT_GLOBAL_PER_MIN", "0")

# The normal suite is deterministic SQLite. The focused database-integration
# matrix opts into a real MySQL service with TEST_DB_BACKEND=mysql.
TEST_DB_BACKEND = os.getenv("TEST_DB_BACKEND", "sqlite").lower()
TEST_DB_DIR = Path(__file__).parent / "_tmp"
TEST_DB_DIR.mkdir(exist_ok=True)
TEST_DB_PATH = TEST_DB_DIR / "test_suite.db"

if TEST_DB_BACKEND == "mysql":
    test_db_url = os.getenv("TEST_DB_URL")
    if not test_db_url:
        raise RuntimeError("TEST_DB_URL is required when TEST_DB_BACKEND=mysql")
    os.environ["DB_BACKEND"] = "mysql"
    os.environ["DB_URL"] = test_db_url
else:
    # Reset the file each session to avoid stale data bleeding into tests.
    TEST_DB_PATH.unlink(missing_ok=True)
    TEST_DB_PATH.touch()
    os.environ["DB_BACKEND"] = "sqlite"
    os.environ["SQLITE_DB_PATH"] = str(TEST_DB_PATH)

# Apply the schema deliberately instead of relying on a test-specific
# MigrationManager call to mutate this shared database as a side effect.
from schema_migrations import MigrationManager

MigrationManager(TEST_DB_PATH if TEST_DB_BACKEND == "sqlite" else "unused-mysql-path").run()


# Shared fixture for clearing NBA cache before tests that use TestClient
def clear_nba_cache():
    """Clear NBA endpoint cache to avoid cross-test pollution."""
    from api.cache import nba_cache

    nba_cache.invalidate_all()
