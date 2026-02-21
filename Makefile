# NFL Algorithm Professional Pipeline Makefile - UV Enhanced
# Supports both UV and traditional venv for seamless transition

.PHONY: help install install-uv install-venv test lint format validate optimize dashboard api api-prod frontend-dev frontend-build fullstack start_pipeline stop_pipeline clean report validate-report backfill-accuracy run-agents ingest-nba nba-train nba-predict nba-odds nba-value nba-risk nba-agents nba-full nba-train-pts nba-train-reb nba-train-ast nba-train-fg3m nba-grade nba-injuries nba-learn nba-report

# Environment detection - defaults to UV if available
ENV_TYPE ?= $(shell command -v uv >/dev/null 2>&1 && [ -f "pyproject.toml" ] && echo "uv" || echo "venv")

SEASON ?= 2025
WEEK ?= 13

# Database backend - default to SQLite for local development
DB_BACKEND ?= sqlite
SQLITE_DB_PATH ?= nfl_data.db
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
	@echo "NFL Algorithm Professional Pipeline - UV Enhanced"
	@echo "==============================================="
	@echo ""
	@echo "Current Environment: $(ENV_TYPE) $(shell [ "$(ENV_TYPE)" = "uv" ] && echo "(UV)" || echo "(venv)")"
	@echo ""
	@echo "Available targets:"
	@echo "  install        - Smart install (auto-detects UV/venv)"
	@echo "  install-uv     - Force UV installation"
	@echo "  install-venv   - Force venv installation"
	@echo "  fast-sync      - Lightning-fast UV dependency sync"
	@echo "  test          - Run test suite"
	@echo "  lint          - Run linting (mypy)"
	@echo "  format        - Format code (black + isort)"
	@echo "  validate      - Run cross-season validation"
	@echo "  optimize      - Run hyperparameter optimization"
	@echo "  dashboard     - Launch Streamlit dashboard"
	@echo "  report        - Run weekly prop update and generate shareable reports"
	@echo "  enhanced-report - Build enhanced HTML/CSV/MD report and open HTML"
	@echo "  validate-report - Run validation and print leaderboard"
	@echo "  start_pipeline - Start complete automated pipeline"
	@echo "  stop_pipeline - Stop automated pipeline"
	@echo "  clean         - Clean temporary files (caches, coverage) (safe for venv/.venv)"
	@echo "  clean-root    - Archive root clutter (exports, logs, backup DB copies)"
	@echo "  archive-week  - Archive current week's reports (use WEEK=N)"
	@echo "  archive-all   - Full archive (reports + root cleanup)"
	@echo "  migrate-to-uv (deprecated) - See UV_MIGRATION_NOTES.md"
	@echo "  env-info      - Show environment information"
	@echo "  health-check  - Comprehensive environment health check"
	@echo ""
	@echo "Data Ingestion:"
	@echo "  ingest-nfl    - Ingest real NFL data (2024+2025) via nflreadpy"

# Smart installation - auto-detects best environment
install:
	@if command -v uv >/dev/null 2>&1 && [ -f "pyproject.toml" ]; then \
		echo "Installing with UV (detected)..."; \
		$(MAKE) install-uv; \
	elif [ -d "venv" ]; then \
		echo "Installing with venv (detected)..."; \
		$(MAKE) install-venv; \
	else \
		echo "❓ No environment detected. Recommendations:"; \
		echo "  • For 10x faster builds: make migrate-to-uv"; \
		echo "  • For traditional setup: make install-venv"; \
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

# Cross-season validation with timing
validate:
	@echo "Running enhanced cross-season validation with $(ENV_TYPE)..."
	@start_time=$$(date +%s); \
	$(PYTHON) cross_season_validation.py; \
	end_time=$$(date +%s); \
	duration=$$((end_time - start_time)); \
	echo "Validation complete in $${duration}s! Check logs/validation_leaderboard.md"

# Hyperparameter optimization with environment detection
optimize:
	@echo "Running Optuna hyperparameter optimization with $(ENV_TYPE)..."
	$(PYTHON) optuna_optimization.py
	@echo "Optimization complete! Check optuna.db for results"


# Launch dashboard with environment detection
dashboard:
	@echo "Launching Streamlit dashboard with $(ENV_TYPE) ($(DB_BACKEND))..."
	$(DB_ENV) $(PYTHON) -m streamlit run dashboard/main_dashboard.py --server.port 8501

# Launch FastAPI backend
api:
	@echo "Starting FastAPI backend on port 8000..."
	$(DB_ENV) $(PYTHON) -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

api-prod:
	@echo "Starting FastAPI backend (production) on port 8000..."
	$(DB_ENV) $(PYTHON) -m uvicorn api.server:app --host 0.0.0.0 --port 8000

# Frontend commands
frontend-install:
	@echo "Installing frontend dependencies..."
	cd frontend && npm install

frontend-dev:
	@echo "Starting Next.js frontend on port 3000..."
	cd frontend && npm run dev

frontend-build:
	@echo "Building frontend for production..."
	cd frontend && npm run build

# Full stack - run both API and frontend
fullstack:
	@echo "Starting API and frontend..."
	@make api &
	@sleep 2
	@make frontend-dev

# Weekly report with timing
report:
	@echo "Running weekly report pipeline with $(ENV_TYPE)..."
	@start_time=$$(date +%s); \
	$(PYTHON) -m scripts.run_prop_update; \
	$(PYTHON) -m scripts.enhanced_visualizer; \
	end_time=$$(date +%s); \
	duration=$$((end_time - start_time)); \
	echo "Reports generated in $${duration}s! Available in reports/"
	@echo "Enhanced outputs: reports/enhanced_dashboard.html, reports/value_bets_enhanced.csv, reports/quick_picks.md"

# Enhanced report only
enhanced-report:
	@echo "Building enhanced report with $(ENV_TYPE)..."
	$(PYTHON) -m scripts.enhanced_visualizer; \
	if command -v open >/dev/null 2>&1; then open reports/enhanced_dashboard.html || true; fi

# Weekly flow for specific week/season
week:
	@echo "Running weekly report for week $(WEEK) season $(SEASON) with $(ENV_TYPE)..."
	$(PYTHON) -m scripts.run_prop_update --week $(WEEK) --season $(SEASON)
	$(PYTHON) -m scripts.enhanced_visualizer
	@echo "Weekly artifacts in reports/: week_$(WEEK)_*.{csv,json,md,html} and enhanced files"

week-open:
	@echo "Opening weekly enhanced dashboard for week $(WEEK)"
	@open reports/week_$(WEEK)_enhanced_dashboard.html 2>/dev/null || true

week-update:
	@echo "Updating data for season $(SEASON), week $(WEEK)..."
	$(DB_ENV) $(PYTHON) -c "from data_pipeline import update_week; update_week(int('$(SEASON)'), int('$(WEEK)'))"

week-predict:
	@echo "Generating projections for season $(SEASON), week $(WEEK)..."
	$(DB_ENV) $(PYTHON) -c "from models.position_specific import predict_week; predict_week(int('$(SEASON)'), int('$(WEEK)'))"

week-materialize:
	@echo "Materializing value view for season $(SEASON), week $(WEEK)..."
	$(DB_ENV) $(PYTHON) -m scripts.materialize_value_view --season $(SEASON) --week $(WEEK)

week-grade:
	@echo "📊 Grading bets for season $(SEASON), week $(WEEK)..."
	$(DB_ENV) $(PYTHON) -m scripts.record_outcomes --season $(SEASON) --week $(WEEK)

backfill-accuracy:
	@echo "Running historical line accuracy backfill..."
	$(DB_ENV) $(PYTHON) -m scripts.backfill_line_accuracy --seasons 2024,2025 --persist

run-agents:
	@echo "Running agent coordinator for season $(SEASON), week $(WEEK)..."
	$(DB_ENV) $(PYTHON) -m agents.coordinator --season $(SEASON) --week $(WEEK)

risk-check:
	@echo "Running risk check for season $(SEASON), week $(WEEK)..."
	$(DB_ENV) $(PYTHON) -m risk_manager --season $(SEASON) --week $(WEEK)

dry-run:
	@echo "Running dry-run validation for season $(SEASON)..."
	$(DB_ENV) $(PYTHON) -m scripts.dry_run_validation --season $(SEASON)

production-run:
	@echo "Running production pipeline for season $(SEASON), week $(WEEK)..."
	$(DB_ENV) $(PYTHON) -m scripts.production_runner --season $(SEASON) --week $(WEEK)

learn:
	@echo "Running learning loop for season $(SEASON), week $(WEEK)..."
	$(DB_ENV) $(PYTHON) -m learning_loop learn --season $(SEASON) --week $(WEEK)

learning-report:
	@echo "Generating learning report for season $(SEASON)..."
	$(DB_ENV) $(PYTHON) -m learning_loop report --season $(SEASON)

mini-backtest:
	@echo "Running mini backtest for season $(SEASON), week $(WEEK)..."
	$(DB_ENV) $(PYTHON) -m scripts.backtest_replay --season $(SEASON) --weeks $(WEEK) --dry-run

health:
	@echo "Checking feed freshness for season $(SEASON), week $(WEEK)..."
	$(DB_ENV) $(PYTHON) -m scripts.health_check --season $(SEASON) --week $(WEEK)

# Activate betting: scrape props, compute edges, persist to DB
activate-betting:
	@echo "Activating value betting pipeline with $(ENV_TYPE)..."
	@start_time=$$(date +%s); \
	$(DB_ENV) $(PYTHON) scripts/quick_activate.py; \
	end_time=$$(date +%s); \
	duration=$$((end_time - start_time)); \
	echo "Activation complete in $${duration}s!"

# Full system activation: populate data, train models, activate betting
activate-all:
	@echo "Full system activation..."
	@$(MAKE) populate-data || true
	@$(MAKE) train-models || true
	@$(MAKE) activate-betting
	@echo "System operational! Run: make dashboard"

# TE market bias analysis
te-bias-analysis:
	@echo "Running TE market bias analysis with $(ENV_TYPE)..."
	$(DB_ENV) $(PYTHON) -m scripts.te_market_bias --output reports
	@echo "TE bias report written to reports/te_market_bias_report.json"

# Ingest real NFL data from nflverse (2024+2025 by default)
ingest-nfl:
	@echo "Ingesting real NFL data via nflreadpy..."
	$(DB_ENV) $(PYTHON) scripts/ingest_real_nfl_data.py --seasons 2024,2025 --through-week 18

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

# Populate historical data (helper target)
populate-data:
	@echo "Populating NFL database with UV speed..."
	$(DB_ENV) $(PYTHON) scripts/quick_populate.py

# Train models (helper target)
train-models:
	@echo "🤖 Training models..."
	$(DB_ENV) $(PYTHON) scripts/quick_train.py

# Validation report
validate-report:
	@echo "Running enhanced cross-season validation and printing leaderboard..."
	$(PYTHON) cross_season_validation.py | cat
	@echo "---"
	@echo "Saved markdown: logs/validation_leaderboard.md"

# Start complete pipeline
start_pipeline: install
	@echo "Starting NFL Algorithm Professional Pipeline..."
	@echo "Installing dependencies..."
	@$(PIP_INSTALL) -r requirements.txt
	@echo "Setting up database..."
	@$(PYTHON) data_pipeline.py
	@echo "Running validation..."
	@$(PYTHON) cross_season_validation.py
	@echo "Starting pipeline scheduler..."
	@$(PYTHON) -m scripts.pipeline_scheduler &
	@echo "Launching dashboard..."
	@$(PYTHON) -m streamlit run dashboard/main_dashboard.py &
	@echo "Pipeline started successfully!"
	@echo "   Dashboard: http://localhost:8501"

# Stop pipeline
stop_pipeline:
	@echo "Stopping NFL Algorithm Pipeline..."
	pkill -f "pipeline_scheduler.py" || true
	pkill -f "streamlit run dashboard/main_dashboard.py" || true
	@echo "Pipeline stopped!"

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

# Migration to UV (10-100x faster dependency management)
migrate-to-uv:
	@echo "migrate-to-uv is deprecated."
	@echo "See UV_MIGRATION_NOTES.md for historical details if needed."

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

