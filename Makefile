# NFL Algorithm Professional Pipeline Makefile - UV Enhanced
# Supports both UV and traditional venv for seamless transition

.PHONY: help list-targets install install-uv install-venv runtime-preflight doctor doctor-production migrate test lint format validate optimize dashboard api-preflight api-serve api api-prod-serve api-prod pipeline-worker pipeline-worker-once frontend-install frontend-dev frontend-build fullstack clean report validate-report backfill-accuracy run-agents ingest-nfl ingest-nba nba-train nba-predict nba-odds nba-value nba-risk nba-agents nba-full nba-train-pts nba-train-reb nba-train-ast nba-train-fg3m nba-grade nba-injuries nba-learn nba-report nba-tune nfl-train nfl-tune demo nba-importance nba-drift nba-calibrate nba-backtest week week-update week-predict week-refresh week-materialize week-grade production-run health health-check

# Load a Make-compatible local environment file without adding a dotenv dependency.
ENV_FILE ?= .env
ifneq (,$(wildcard $(ENV_FILE)))
include $(ENV_FILE)
ENV_KEYS := $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' $(ENV_FILE))
export $(ENV_KEYS)
endif

# Environment detection - defaults to UV if available
ENV_TYPE ?= $(shell command -v uv >/dev/null 2>&1 && [ -f "pyproject.toml" ] && echo "uv" || echo "venv")

# SEASON/WEEK must be supplied by the caller for any target that runs the
# weekly prop pipeline (T0 #5: no hardcoded defaults).
# Example: make week SEASON=2026 WEEK=1
SEASON ?=
WEEK ?=
WEEKS ?=
NFL_SEASONS ?=
HISTORY_SEASONS ?=
REFRESH_HISTORY ?= 0
THROUGH_WEEK ?= 18
MIGRATION_BACKUP_DIR ?= migration_backup

# Guard used by weekly targets — fails loud if SEASON or WEEK is empty.
define require_season_week
	@if [ -z "$(SEASON)" ] || [ -z "$(WEEK)" ]; then \
		echo "ERROR: SEASON and WEEK are required. Example: make $@ SEASON=2026 WEEK=1" >&2; \
		exit 2; \
	fi
endef

# Guard used by season-scoped targets — fails loud if SEASON is empty.
define require_season
	@if [ -z "$(SEASON)" ]; then \
		echo "ERROR: SEASON is required. Example: make $@ SEASON=2026" >&2; \
		exit 2; \
	fi
endef

# Guard used by historical evaluation targets. WEEKS is a space-separated list.
define require_season_weeks
	@if [ -z "$(SEASON)" ] || [ -z "$(WEEKS)" ]; then \
		echo "ERROR: SEASON and WEEKS are required. Example: make $@ SEASON=2025 WEEKS='1 2 3'" >&2; \
		exit 2; \
	fi
endef

# Runtime defaults for local development
DB_BACKEND ?= sqlite
SQLITE_DB_PATH ?= nfl_data.db
API_HOST ?= 0.0.0.0
API_PORT ?= 8000
DB_ENV := DB_BACKEND=$(DB_BACKEND) SQLITE_DB_PATH=$(SQLITE_DB_PATH)

# Conditional commands based on ENV_TYPE
ifeq ($(ENV_TYPE),uv)
    PYTHON := uv run python
    PIP_INSTALL := uv pip install
    PIP_UNINSTALL := uv pip uninstall
    ENV_SETUP := uv venv --python 3.13 && uv pip install -r requirements.txt
else
    PYTHON := venv/bin/python
    PIP_INSTALL := venv/bin/pip install
    PIP_UNINSTALL := venv/bin/pip uninstall
    ENV_SETUP := python3.13 -m venv venv && venv/bin/pip install --upgrade pip && venv/bin/pip install -r requirements.txt
endif

# Default target
help:
	@echo "NFL Algorithm developer commands"
	@echo "================================"
	@echo "Environment: $(ENV_TYPE) | Database: $(DB_BACKEND)"
	@echo ""
	@echo "Setup and diagnostics:"
	@echo "  make install             Install Python dependencies (UV preferred, venv fallback)"
	@echo "  make frontend-install    Install locked frontend dependencies"
	@echo "  make migrate             Back up and migrate the local SQLite database"
	@echo "  make doctor              Validate tools, config, database, migrations, keys, and modules"
	@echo "  make doctor-production   Require live-odds key and private execution modules"
	@echo ""
	@echo "Local applications:"
	@echo "  make fullstack           Supervise worker, API (:8000), and frontend (:3000)"
	@echo "  make api                 Migrate and start the development API"
	@echo "  make pipeline-worker     Start the durable pipeline worker"
	@echo "  make frontend-dev        Start the Next.js frontend"
	@echo "  make dashboard           Start the legacy Streamlit dashboard (:8501)"
	@echo ""
	@echo "Weekly NFL workflow (requires SEASON and WEEK):"
	@echo "  make week-refresh SEASON=2026 WEEK=1"
	@echo "  make production-run SEASON=2026 WEEK=1"
	@echo "  make pipeline-worker-once"
	@echo "  make health SEASON=2026 WEEK=1"
	@echo ""
	@echo "Quality: make test | make lint | make format"
	@echo "Evaluation: make validate SEASON=2025 WEEKS='1 2 3'"
	@echo "All target names: make list-targets"

list-targets:
	@$(MAKE) -qp 2>/dev/null | awk -F: '/^[A-Za-z0-9][A-Za-z0-9_.-]*:([^=]|$$)/ {print $$1}' | sort -u

# Smart installation - auto-detects best environment
install:
	@if command -v uv >/dev/null 2>&1 && [ -f "pyproject.toml" ]; then \
		echo "Installing with UV (detected)..."; \
		$(MAKE) install-uv; \
	elif [ -d "venv" ]; then \
		echo "Installing with venv (detected)..."; \
		$(MAKE) install-venv; \
	elif command -v python3.13 >/dev/null 2>&1; then \
		echo "UV not detected; installing with Python 3.13 venv..."; \
		$(MAKE) install-venv; \
	else \
		echo "ERROR: install requires either UV or Python 3.13." >&2; \
		echo "Install UV from https://docs.astral.sh/uv/ or install Python 3.13." >&2; \
		exit 1; \
	fi

# UV installation (10-100x faster)
install-uv:
	@echo "Installing dependencies with UV (10-100x faster)..."
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Installing UV..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		export PATH="$$HOME/.cargo/bin:$$PATH"; \
	fi
	uv python pin 3.13
	uv venv --python 3.13
	@start_time=$$(date +%s); \
	uv pip install -r requirements.txt; \
	end_time=$$(date +%s); \
	duration=$$((end_time - start_time)); \
	echo "Creating directories..."; \
	mkdir -p data logs models dashboard tests scripts; \
	echo "UV installation complete in $${duration}s!"

# Traditional venv installation
install-venv:
	@echo "Installing dependencies with venv..."
	@start_time=$$(date +%s); \
	python3.13 -m venv venv; \
	venv/bin/pip install --upgrade pip; \
	venv/bin/pip install -r requirements.txt; \
	end_time=$$(date +%s); \
	duration=$$((end_time - start_time)); \
	echo "Creating directories..."; \
	mkdir -p data logs models dashboard tests scripts; \
	echo "venv installation complete in $${duration}s!"

# Lightning-fast UV sync (only works with UV)
fast-sync:
	@if command -v uv >/dev/null 2>&1; then \
		echo "Lightning-fast UV dependency sync..."; \
		time uv pip sync requirements.txt; \
		echo "Sync complete!"; \
	else \
		echo "UV not available. Run 'make install-uv' first."; \
		exit 1; \
	fi

# Testing with environment detection and timing
test:
	@echo "Running test suite with $(ENV_TYPE)..."
	@start_time=$$(date +%s); \
	$(PYTHON) -m pytest tests/ -v --cov=. --cov-report=html; \
	end_time=$$(date +%s); \
	duration=$$((end_time - start_time)); \
	echo "Tests complete in $${duration}s! Coverage report in htmlcov/"

# Linting with environment detection
lint:
	@echo "Running mypy type checking with $(ENV_TYPE)..."
	@start_time=$$(date +%s); \
	$(PYTHON) -m mypy . --ignore-missing-imports; \
	end_time=$$(date +%s); \
	duration=$$((end_time - start_time)); \
	echo "Linting complete in $${duration}s!"

# Code formatting with environment detection
format:
	@echo "Formatting code with $(ENV_TYPE)..."
	$(PYTHON) -m black . --line-length 100
	$(PYTHON) -m isort . --profile black
	@echo "Formatting complete!"

# Evaluate persisted production projections against point-in-time outcomes.
validate:
	$(call require_season_weeks)
	@echo "Evaluating persisted NFL projections with $(ENV_TYPE)..."
	@start_time=$$(date +%s); \
	$(DB_ENV) $(PYTHON) -m scripts.evaluate_nfl_projections evaluate \
		--season $(SEASON) --weeks $(WEEKS) \
		--output logs/metrics/nfl-projection-evaluation-$(SEASON).json; \
	end_time=$$(date +%s); \
	duration=$$((end_time - start_time)); \
	echo "Evaluation complete in $${duration}s."

# Hyperparameter optimization with environment detection
optimize:
	@echo "Running Optuna hyperparameter optimization with $(ENV_TYPE)..."
	$(PYTHON) optuna_optimization.py
	@echo "Optimization complete! Check optuna.db for results"


# Launch dashboard with environment detection
dashboard:
	@echo "Launching Streamlit dashboard with $(ENV_TYPE) ($(DB_BACKEND))..."
	$(DB_ENV) $(PYTHON) -m streamlit run dashboard/main_dashboard.py --server.port 8501

# Upgrade the local SQLite schema before the API starts. Production MySQL
# migrations remain an explicit deployment concern.
api-preflight:
	@if [ "$(DB_BACKEND)" = "sqlite" ]; then \
		if [ -f "$(SQLITE_DB_PATH)" ]; then \
			mkdir -p "$(MIGRATION_BACKUP_DIR)"; \
			backup_path="$(MIGRATION_BACKUP_DIR)/$$(basename "$(SQLITE_DB_PATH)").$$(date -u +%Y%m%dT%H%M%SZ).$$$$.bak"; \
			echo "Backing up $(SQLITE_DB_PATH) to $$backup_path..."; \
			cp "$(SQLITE_DB_PATH)" "$$backup_path"; \
		fi; \
		echo "Applying SQLite schema migrations to $(SQLITE_DB_PATH)..."; \
		$(DB_ENV) $(PYTHON) -m scripts.run_migrations --database "$(SQLITE_DB_PATH)"; \
	else \
		echo "Skipping SQLite API preflight for DB_BACKEND=$(DB_BACKEND)"; \
	fi

migrate: api-preflight

runtime-preflight:
	$(DB_ENV) $(PYTHON) -m scripts.preflight --check-schema

# Validate a migrated local environment. Warnings identify optional live/private features.
doctor:
	$(DB_ENV) $(PYTHON) -m scripts.preflight --check-schema --check-frontend

doctor-production:
	$(DB_ENV) $(PYTHON) -m scripts.preflight --check-schema --check-frontend --require-live-odds --require-private-modules

# Launch FastAPI backend after callers complete any required preflight.
api-serve:
	@echo "Starting FastAPI backend on $(API_HOST):$(API_PORT)..."
	$(DB_ENV) $(PYTHON) -m uvicorn api.application:app --host $(API_HOST) --port $(API_PORT) --reload

api: api-preflight
	@$(MAKE) runtime-preflight
	@$(MAKE) api-serve

api-prod-serve:
	@echo "Starting FastAPI backend (production) on $(API_HOST):$(API_PORT)..."
	$(DB_ENV) $(PYTHON) -m uvicorn api.application:app --host $(API_HOST) --port $(API_PORT)

api-prod: api-preflight
	@$(MAKE) runtime-preflight
	@$(MAKE) api-prod-serve

# Frontend commands
frontend-install:
	@echo "Installing locked frontend dependencies..."
	cd frontend && npm ci

frontend-dev:
	@echo "Starting Next.js frontend on port 3000..."
	cd frontend && npm run dev

frontend-build:
	@echo "Building frontend for production..."
	cd frontend && npm run build

# Full stack - supervised startup with readiness waiting and graceful cleanup.
fullstack: api-preflight
	@echo "Starting worker, API, and frontend..."
	$(DB_ENV) $(PYTHON) -m scripts.run_local_services

# Weekly report with timing
report:
	$(call require_season_week)
	@echo "Running weekly report pipeline (season=$(SEASON) week=$(WEEK)) with $(ENV_TYPE)..."
	@start_time=$$(date +%s); \
	$(PYTHON) -m scripts.run_prop_update --season $(SEASON) --week $(WEEK); \
	$(PYTHON) -m scripts.enhanced_visualizer --season $(SEASON) --week $(WEEK); \
	end_time=$$(date +%s); \
	duration=$$((end_time - start_time)); \
	echo "Reports generated in $${duration}s! Available in reports/"
	@echo "Enhanced outputs: reports/enhanced_dashboard.html, reports/value_bets_enhanced.csv, reports/quick_picks.md"

# Enhanced report only
enhanced-report:
	$(call require_season_week)
	@echo "Building enhanced report (season=$(SEASON) week=$(WEEK)) with $(ENV_TYPE)..."
	$(PYTHON) -m scripts.enhanced_visualizer --season $(SEASON) --week $(WEEK); \
	if command -v open >/dev/null 2>&1; then open reports/enhanced_dashboard.html || true; fi

# Weekly flow for specific week/season
week:
	$(call require_season_week)
	@echo "Running weekly report for week $(WEEK) season $(SEASON) with $(ENV_TYPE)..."
	$(PYTHON) -m scripts.run_prop_update --week $(WEEK) --season $(SEASON)
	$(PYTHON) -m scripts.enhanced_visualizer --season $(SEASON) --week $(WEEK)
	@echo "Weekly artifacts in reports/: week_$(WEEK)_*.{csv,json,md,html} and enhanced files"

week-open:
	@echo "Opening weekly enhanced dashboard for week $(WEEK)"
	@open reports/week_$(WEEK)_enhanced_dashboard.html 2>/dev/null || true

week-update:
	$(call require_season_week)
	@echo "Running canonical pregame refresh for season $(SEASON), week $(WEEK)..."
	$(DB_ENV) $(PYTHON) -m scripts.prepare_nfl_week --season $(SEASON) --week $(WEEK)

week-predict:
	$(call require_season_week)
	@echo "Running canonical roster-backed prediction refresh for season $(SEASON), week $(WEEK)..."
	$(DB_ENV) $(PYTHON) -m scripts.prepare_nfl_week --season $(SEASON) --week $(WEEK)

week-refresh:
	$(call require_season_week)
	@echo "Refreshing roster, schedule, history, and predictions for $(SEASON) week $(WEEK)..."
	$(DB_ENV) $(PYTHON) -m scripts.prepare_nfl_week --season $(SEASON) --week $(WEEK) $(if $(strip $(HISTORY_SEASONS)),--history-seasons $(HISTORY_SEASONS),) $(if $(filter 1,$(REFRESH_HISTORY)),--refresh-history,)

week-materialize:
	$(call require_season_week)
	@echo "Materializing value view for season $(SEASON), week $(WEEK)..."
	$(DB_ENV) $(PYTHON) -m scripts.materialize_value_view --season $(SEASON) --week $(WEEK)

week-grade:
	$(call require_season_week)
	@echo "📊 Grading bets for season $(SEASON), week $(WEEK)..."
	$(DB_ENV) $(PYTHON) -m scripts.record_outcomes --season $(SEASON) --week $(WEEK)

backfill-accuracy:
	@echo "Running historical line accuracy backfill..."
	$(DB_ENV) $(PYTHON) -m scripts.backfill_line_accuracy --seasons 2024,2025 --persist

run-agents:
	$(call require_season_week)
	@echo "Running agent coordinator for season $(SEASON), week $(WEEK)..."
	$(DB_ENV) $(PYTHON) -m agents.coordinator --season $(SEASON) --week $(WEEK)

risk-check:
	$(call require_season_week)
	@echo "Running risk check for season $(SEASON), week $(WEEK)..."
	$(DB_ENV) $(PYTHON) -m risk_manager --season $(SEASON) --week $(WEEK)

dry-run:
	$(call require_season)
	@echo "Running dry-run validation for season $(SEASON)..."
	$(DB_ENV) $(PYTHON) -m scripts.dry_run_validation --season $(SEASON)

production-run:
	$(call require_season_week)
	@echo "Queueing production pipeline for season $(SEASON), week $(WEEK)..."
	$(DB_ENV) $(PYTHON) -m scripts.production_runner --season $(SEASON) --week $(WEEK)

pipeline-worker: runtime-preflight
	@echo "Starting durable NFL pipeline worker..."
	$(DB_ENV) $(PYTHON) -m pipeline_jobs.worker

pipeline-worker-once: runtime-preflight
	@echo "Processing at most one durable NFL pipeline job..."
	$(DB_ENV) $(PYTHON) -m pipeline_jobs.worker --once

learn:
	$(call require_season_week)
	@echo "Running learning loop for season $(SEASON), week $(WEEK)..."
	$(DB_ENV) $(PYTHON) -m learning_loop learn --season $(SEASON) --week $(WEEK)

learning-report:
	$(call require_season)
	@echo "Generating learning report for season $(SEASON)..."
	$(DB_ENV) $(PYTHON) -m learning_loop report --season $(SEASON)

mini-backtest:
	$(call require_season_week)
	@echo "Running mini backtest for season $(SEASON), week $(WEEK)..."
	$(DB_ENV) $(PYTHON) -m scripts.backtest_replay --season $(SEASON) --weeks $(WEEK) --dry-run

health:
	$(call require_season_week)
	@echo "Checking feed freshness for season $(SEASON), week $(WEEK)..."
	$(DB_ENV) $(PYTHON) -m scripts.health_check --season $(SEASON) --week $(WEEK)

# TE market bias analysis
te-bias-analysis:
	@echo "Running TE market bias analysis with $(ENV_TYPE)..."
	$(DB_ENV) $(PYTHON) -m scripts.te_market_bias --output reports
	@echo "TE bias report written to reports/te_market_bias_report.json"

# Ingest real NFL data from nflverse (configured history plus the current season)
ingest-nfl:
	@echo "Ingesting real NFL data via nflreadpy..."
	$(DB_ENV) $(PYTHON) scripts/ingest_real_nfl_data.py $(if $(strip $(NFL_SEASONS)),--seasons $(NFL_SEASONS),) --through-week $(THROUGH_WEEK)

# ============================================================
# NBA Targets
# ============================================================

# Ingest NBA player game logs for current and prior season
ingest-nba:
	@echo "Ingesting NBA player game logs via nba_api..."
	$(DB_ENV) $(PYTHON) scripts/ingest_nba_data.py

# Train NBA models for all markets
nba-train:
	@echo "Training NBA models for all markets..."
	$(DB_ENV) $(PYTHON) models/nba/stat_model.py --train --market all

# Generate NBA projections for all markets (use NBA_DATE=YYYY-MM-DD for historical)
NBA_DATE ?= $(shell date +%Y-%m-%d)
nba-predict:
	@echo "Generating NBA projections for all markets for $(NBA_DATE)..."
	$(DB_ENV) $(PYTHON) models/nba/stat_model.py --predict --market all --date $(NBA_DATE)

# Scrape NBA player prop odds from The Odds API
nba-odds:
	@echo "Scraping NBA player prop odds for $(NBA_DATE)..."
	$(DB_ENV) $(PYTHON) scripts/scrape_nba_odds.py --date $(NBA_DATE)

# Compute NBA value bets and materialise to nba_materialized_value_view
nba-value:
	@echo "Computing NBA value bets for $(NBA_DATE)..."
	$(DB_ENV) $(PYTHON) nba_value_engine.py --date $(NBA_DATE)

# Train NBA calibration model
nba-calibrate:
	@echo "Training NBA calibration model..."
	$(DB_ENV) $(PYTHON) scripts/train_nba_calibration.py

# Run NBA risk assessment
nba-risk:
	@echo "Running NBA risk assessment for $(NBA_DATE)..."
	$(DB_ENV) $(PYTHON) nba_risk_manager.py --date $(NBA_DATE)

# Run NBA agent coordinator
nba-agents:
	@echo "Running NBA agent coordinator for $(NBA_DATE)..."
	$(DB_ENV) $(PYTHON) -m agents.nba_coordinator --date $(NBA_DATE)

# Ingest NBA injury / DNP data for a given date
nba-injuries:
	@echo "Ingesting NBA injury data for $(NBA_DATE)..."
	$(DB_ENV) $(PYTHON) scripts/ingest_nba_injuries.py --date $(NBA_DATE)

# Grade NBA bets against actual results
GAME_DATE ?= $(shell date +%Y-%m-%d)
nba-grade:
	@echo "Grading NBA bets for $(GAME_DATE)..."
	$(DB_ENV) $(PYTHON) scripts/record_nba_outcomes.py --game-date $(GAME_DATE)

# Run NBA learning loop for a game date
nba-learn:
	@echo "Running NBA learning loop for $(GAME_DATE)..."
	$(DB_ENV) $(PYTHON) nba_learning_loop.py learn --date $(GAME_DATE)

# Generate NBA learning report for a date range
nba-report:
	@echo "Generating NBA learning report..."
	$(DB_ENV) $(PYTHON) nba_learning_loop.py report --start-date $(NBA_DATE) --end-date $(GAME_DATE)

# Grade NBA bets and compute line accuracy for a given date
nba-accuracy:
	$(DB_ENV) uv run python scripts/record_nba_outcomes.py --game-date $(GAME_DATE)

# Run full NBA production pipeline for a given date
nba-run:
	$(DB_ENV) uv run python scripts/nba_production_runner.py --date $(NBA_DATE)

# Full NBA refresh: ingest -> train -> predict -> injuries -> odds -> value -> risk -> agents
nba-full:
	@echo "Running full NBA pipeline..."
	$(MAKE) ingest-nba
	$(MAKE) nba-train
	$(MAKE) nba-predict
	$(MAKE) nba-injuries
	$(MAKE) nba-odds
	$(MAKE) nba-value
	$(MAKE) nba-risk
	$(MAKE) nba-agents

# ============================================================
# NFL Model Training Targets
# ============================================================

# Train NFL weekly models for all markets (StackingRegressor ensemble)
nfl-train:
	@echo "Training NFL weekly models for all markets..."
	$(DB_ENV) $(PYTHON) -c "from models.position_specific.weekly import train_weekly_models; from utils.db import read_dataframe; df = read_dataframe('SELECT DISTINCT season, week FROM player_stats_enhanced ORDER BY season DESC, week DESC LIMIT 20'); train_weekly_models(list(df.itertuples(index=False, name=None)))"

# Optuna hyperparameter tuning for NFL stat models (writes best_params_{market}.json)
nfl-tune:
	$(DB_ENV) uv run python scripts/nfl_optuna_tuning.py --market all

# Optuna hyperparameter tuning for NBA stat models (writes best_params_{market}.json)
nba-tune:
	$(DB_ENV) uv run python scripts/nba_optuna_tuning.py --market all

# Individual market training targets
nba-train-pts:
	@echo "Training NBA points (PTS) model..."
	$(DB_ENV) $(PYTHON) models/nba/stat_model.py --train --market pts

nba-train-reb:
	@echo "Training NBA rebounds (REB) model..."
	$(DB_ENV) $(PYTHON) models/nba/stat_model.py --train --market reb

nba-train-ast:
	@echo "Training NBA assists (AST) model..."
	$(DB_ENV) $(PYTHON) models/nba/stat_model.py --train --market ast

nba-train-fg3m:
	@echo "Training NBA 3-pointers made (FG3M) model..."
	$(DB_ENV) $(PYTHON) models/nba/stat_model.py --train --market fg3m

nba-defense:
	$(DB_ENV) uv run python -c "from scripts.ingest_nba_data import ingest_defensive_stats; ingest_defensive_stats()"

# Compute NBA feature importance for all trained markets
nba-importance:
	@echo "Computing NBA feature importance..."
	$(DB_ENV) $(PYTHON) scripts/run_nba_importance.py

# Run NBA drift detection checks for a given date
nba-drift:
	@echo "Running NBA drift detection..."
	$(DB_ENV) $(PYTHON) scripts/run_nba_drift.py

# Run NBA walk-forward backtest over a date range
nba-backtest:
	@echo "Running NBA walk-forward backtest..."
	$(DB_ENV) $(PYTHON) scripts/run_nba_backtest.py --start-date $(NBA_DATE) --end-date $(GAME_DATE)

# Demo mode: install, migrate schema, seed synthetic data
demo:
	@echo "Setting up demo environment..."
	$(MAKE) install
	$(DB_ENV) $(PYTHON) -c "from schema_migrations import MigrationManager; MigrationManager('nfl_data.db').run()"
	$(DB_ENV) $(PYTHON) scripts/seed_demo_data.py
	@echo "Demo data loaded! Run: make fullstack"

# Validation report
validate-report:
	$(call require_season_weeks)
	@$(MAKE) validate SEASON=$(SEASON) WEEKS="$(WEEKS)"
	@echo "Saved JSON: logs/metrics/nfl-projection-evaluation-$(SEASON).json"

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	# Avoid touching virtualenvs
	find . \( -path "./venv" -o -path "./.venv" \) -prune -o -type f -name "*.pyc" -delete
	find . \( -path "./venv" -o -path "./.venv" \) -prune -o -type d -name "__pycache__" -delete
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	@echo "Clean complete!"

# Clean root clutter files (exports, logs, old DBs)
clean-root:
	@echo "Archiving root clutter files..."
	@mkdir -p archive/exports archive/databases archive/logs_old
	@if [ -f "export10.csv" ]; then mv export10.csv archive/exports/; echo "Moved export10.csv"; fi
	@if [ -f "week102.csv" ]; then mv week102.csv archive/exports/; echo "Moved week102.csv"; fi
	@if [ -f "prop_update.log" ]; then mv prop_update.log archive/logs_old/; echo "Moved prop_update.log"; fi
	@if [ -f "nfl_data.db" ]; then cp nfl_data.db archive/databases/; echo "Copied nfl_data.db to archive/databases/"; fi
	@if [ -f "nfl_prop_lines.db" ]; then cp nfl_prop_lines.db archive/databases/; echo "Copied nfl_prop_lines.db to archive/databases/"; fi
	@if [ -f "optuna.db" ]; then cp optuna.db archive/databases/; echo "Copied optuna.db to archive/databases/"; fi
	@echo "Root cleanup complete!"

# Research loop (requires dev-browser on :9222)
research-loop:
	@echo "Run: npx tsx scripts/research_loop.ts --question '...' --urls 'url1,url2' --output docs/research/slug.md"
	@echo "Start dev-browser first: ~/.claude/plugins/cache/n-skills/dev-browser/1.0.0/skills/dev-browser/server.sh --headless &"
	@echo "Example: make research-run QUESTION='Playwright rate limit' URLS='https://playwright.dev' OUTPUT=docs/research/playwright-rate-limit.md"

research-run:
	npx tsx scripts/research_loop.ts --question "$(QUESTION)" --urls "$(URLS)" --output "$(OUTPUT)"

# Development workflow
dev-setup: install format lint
	@echo "Development environment ready!"

# Production deployment check
production-check: lint test validate
	@echo "Production readiness check complete!"
	@echo "Review validation results before deployment."

# Backup data
backup:
	@echo "Backing up data..."
	mkdir -p data/backups/$(shell date +%Y%m%d)
	cp data/nfl.db data/backups/$(shell date +%Y%m%d)/ 2>/dev/null || true
	cp logs/*.csv data/backups/$(shell date +%Y%m%d)/ 2>/dev/null || true
	@echo "Backup complete in data/backups/$(shell date +%Y%m%d)/"

# Archive current week's reports
archive-week:
	@echo "Archiving current week's reports..."
	@WEEK_DIR=archive/reports/$(shell date +%Y-%m-%d-week-$(WEEK)); \
	mkdir -p $$WEEK_DIR; \
	if [ -d "reports" ]; then \
		cp -r reports/* $$WEEK_DIR/ 2>/dev/null || true; \
		echo "Archived reports to $$WEEK_DIR"; \
	else \
		echo "No reports directory found"; \
	fi

# Archive everything (exports, logs, DBs, reports)
archive-all: archive-week clean-root
	@echo "Full archive complete!"
	@echo "Archived to: archive/exports/, archive/databases/, archive/logs_old/, archive/reports/"

# Database maintenance
db-maintenance:
	@echo "Running database maintenance..."
	# python -c "import sqlite3; conn = sqlite3.connect('data/nfl.db'); conn.execute('VACUUM'); conn.close()"
	@echo "Database optimization not supported for remote MySQL yet."
 
# ============================================================================
# UV ENHANCED TARGETS - 10-100x faster dependency management
# ============================================================================

# Environment information
env-info:
	@echo "Environment Information"
	@echo "========================="
	@echo "Environment Type: $(ENV_TYPE)"
	@echo "Python Command: $(PYTHON)"
	@echo "Python Version: $(shell $(PYTHON) --version 2>/dev/null || echo 'Not available')"
	@echo "UV Available: $(shell command -v uv >/dev/null 2>&1 && echo 'Yes' || echo 'No')"
	@echo "UV Version: $(shell uv --version 2>/dev/null || echo 'Not installed')"
	@echo "VEnv Available: $(shell [ -d 'venv' ] && echo 'Yes' || echo 'No')"
	@echo "PyProject.toml: $(shell [ -f 'pyproject.toml' ] && echo 'Yes' || echo 'No')"
	@echo "Project Root: $(shell pwd)"

# Comprehensive health check
health-check: env-info
	@echo ""
	@echo "🔍 Health Check Results:"
	@if command -v uv >/dev/null 2>&1; then \
		echo "UV is available"; \
		uv pip list | head -5 2>/dev/null || echo "   (No packages listed)"; \
	else \
		echo "UV not available"; \
	fi
	@if [ -d "venv" ]; then \
		echo "venv is available"; \
	else \
		echo "venv not available"; \
	fi

# Quick environment reset (UV only - seconds instead of minutes)
quick-reset:
	@if command -v uv >/dev/null 2>&1; then \
		echo "Quick environment reset with UV..."; \
		rm -rf .venv/; \
		time uv venv --python 3.13; \
		time uv pip install -r requirements.txt; \
		echo "Environment reset complete\!"; \
	else \
		echo "UV not available for quick reset"; \
	fi

# Add new dependency (smart detection)
add-dep:
	@if [ -z "$(PKG)" ]; then \
		echo "Usage: make add-dep PKG=package_name"; \
		exit 1; \
	fi
	@if command -v uv >/dev/null 2>&1 && [ -f "pyproject.toml" ]; then \
		echo "Adding $(PKG) with UV..."; \
		uv pip install $(PKG); \
		uv pip freeze > requirements.txt; \
		echo "Added $(PKG) and updated requirements.txt"; \
	else \
		echo "Adding $(PKG) with pip..."; \
		$(PIP_INSTALL) $(PKG); \
		$(PYTHON) -m pip freeze > requirements.txt; \
		echo "Added $(PKG) and updated requirements.txt"; \
	fi
