# Repository Guidelines

## Project Structure & Module Organization
- Core scripts in the root: `data_pipeline.py`, `value_betting_engine.py`, `cross_season_validation.py`, `materialized_value_view.py`.
- Key folders: `models/` (position models), `utils/` (helpers), `scripts/` (populate/train/report), `dashboard/` (Streamlit UI), `templates/` (reports), `docs/`, `reports/`, `logs/`, `data/` (CSVs/JSON/cache), `tests/` (pytest).
- Local databases (`nfl_data.db`, `nfl_prop_lines.db`, `optuna.db`) live at the root; treat as dev caches, not production sources.

## Build, Test, and Development Commands
- `make install` — smart env setup (prefers `uv`, falls back to `venv`).
- `make dev-setup` — install + format + lint for a clean workspace.
- `make test` — pytest with coverage (HTML in `htmlcov/`); `make lint` — mypy; `make format` — black + isort (line length 100).
- `make validate` — cross-season validation baseline; `make optimize` — Optuna search.
- `make dashboard` — Streamlit on port 8501; `make report` or `make enhanced-report` — build weekly artifacts in `reports/`.
- Weekly helpers: `make week-update SEASON=2025 WEEK=10`, `make week-predict ...`, `make week-materialize ...`, `make mini-backtest ...`.

## Coding Style & Naming Conventions
- Python 3.13; run `make format` before pushing. Black/isort enforce 100-column width.
- Use type hints where practical; mypy is configured with `--ignore-missing-imports`.
- Files/modules: snake_case; tests mirror sources (e.g., `tests/test_value_betting_engine.py`). Functions use verbs (`load_week_data`); classes PascalCase; constants UPPER_SNAKE.

## Testing Guidelines
- Framework: pytest; discovery set in `pyproject.toml` (`test_*.py`, `Test*` classes, `test_*` functions).
- Cover new code paths; add regression tests for every bug fix.
- Prefer temporary fixtures (`tmp_path`) over writing to `data/`.
- Run `make test` before commits; avoid checking in `htmlcov/`.

## Commit & Pull Request Guidelines
- Commit messages: concise, imperative; optional prefixes (`feat:`, `fix:`, `chore:`) match current history.
- PRs should list intent, key commands run (tests/linters), and artifacts for UI/report changes (e.g., `reports/week_10_enhanced_dashboard.html`); link issues when relevant.
- Keep diffs focused; separate formatting-only changes from logic.

## Configuration & Data Safety
- Copy `.env.example` to `.env`; add DB and odds API credentials locally. Never commit secrets.
- Treat `.db` files as disposable caches; back up with `make backup` before migrations.
- Store large exports in `archive/` or `reports/`; keep the repo root clean.
