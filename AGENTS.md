# Repository Guidelines

## Project Structure & Module Organization

| Location | Purpose |
|----------|---------|
| Root scripts | `data_pipeline.py`, `value_betting_engine.py`, `materialized_value_view.py` |
| `models/` | Position-specific ML models |
| `utils/` | Helpers (`player_id_utils`, `defense_adjustments`, `db`) |
| `scripts/` | Populate, train, report, ingest scripts |
| `dashboard/` | Streamlit UI |
| `tests/` | pytest test suite |
| `data/` | CSVs, JSON, odds cache |
| `*.db` | Local SQLite caches (dev only) |

## Build, Test, and Development Commands

```bash
# Setup
make install        # Smart env setup (UV or venv)
make dev-setup      # Install + format + lint

# Testing
make test           # pytest with coverage
make lint           # mypy type checking
make format         # black + isort (100 cols)

# Data
make ingest-nfl     # Fetch 2024+2025 NFL data via nflreadpy

# Weekly Workflow
make week-predict SEASON=2025 WEEK=13
make week-materialize SEASON=2025 WEEK=13
make dashboard      # Streamlit on :8501
```

## Coding Style & Naming Conventions

- **Python**: 3.13+ with type hints
- **Formatting**: `make format` before commits (black + isort, 100 cols)
- **Files**: `snake_case.py`
- **Functions**: verbs (`load_week_data`, `compute_mu`)
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE`

## Testing Guidelines

- Framework: **pytest** (`test_*.py`, `test_*` functions)
- Coverage: Add tests for new code paths
- Fixtures: Use `tmp_path` for temp files
- Pre-commit: Run `make test` before pushing

## Commit & Pull Request Guidelines

```bash
# Commit prefixes
feat:   # New feature
fix:    # Bug fix
chore:  # Maintenance
docs:   # Documentation
test:   # Test additions
```

- Keep diffs focused
- Separate formatting from logic changes
- Link related issues

## Configuration & Data Safety

| Item | Guideline |
|------|-----------|
| `.env` | Copy from `.env.example`, never commit |
| `*.db` | Disposable caches, back up before migrations |
| `archive/` | Store large exports here |
| Secrets | API keys in `.env` only |

## Database Backends

```bash
# SQLite (local dev - default)
DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db

# MySQL (production)
DB_BACKEND=mysql DB_URL="mysql://user:pass@host/db"
```
