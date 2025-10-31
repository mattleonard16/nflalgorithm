# NFL Algorithm Professional Pipeline Makefile - UV Enhanced
# Supports both UV and traditional venv for seamless transition

.PHONY: help install install-uv install-venv test lint format validate optimize dashboard start_pipeline stop_pipeline clean report validate-report

# Environment detection - defaults to UV if available
ENV_TYPE ?= $(shell command -v uv >/dev/null 2>&1 && [ -f "pyproject.toml" ] && echo "uv" || echo "venv")

SEASON ?= 2023
WEEK ?= 1

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
	@echo "  clean         - Clean temporary files"
	@echo "  migrate-to-uv  - Migrate from venv to UV (10-100x faster!)"
	@echo "  env-info      - Show environment information"
	@echo "  health-check  - Comprehensive environment health check"
	@echo ""
	@echo "Cache Management:"
	@echo "  cache-stats   - Show comprehensive cache statistics"
	@echo "  cache-warm    - Pre-populate cache with popular endpoints"
	@echo "  cache-test    - Test cache functionality and performance"
	@echo "  cache-clean   - Clean expired cache entries"

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
	@echo "Launching Streamlit dashboard with $(ENV_TYPE)..."
	$(PYTHON) -m streamlit run dashboard/main_dashboard.py --server.port 8501

# Weekly report with timing
report:
	@echo "Running weekly report pipeline with $(ENV_TYPE)..."
	@start_time=$$(date +%s); \
	$(PYTHON) run_prop_update.py; \
	$(PYTHON) enhanced_visualizer.py; \
	end_time=$$(date +%s); \
	duration=$$((end_time - start_time)); \
	echo "Reports generated in $${duration}s! Available in reports/"
	@echo "Enhanced outputs: reports/enhanced_dashboard.html, reports/value_bets_enhanced.csv, reports/quick_picks.md"

# Enhanced report only
enhanced-report:
	@echo "Building enhanced report with $(ENV_TYPE)..."
	$(PYTHON) enhanced_visualizer.py; \
	if command -v open >/dev/null 2>&1; then open reports/enhanced_dashboard.html || true; fi

# Weekly flow for specific week/season
week:
	@echo "Running weekly report for week $(W) season $(S) with $(ENV_TYPE)..."
	$(PYTHON) run_prop_update.py --week $(W) --season $(S)
	$(PYTHON) enhanced_visualizer.py
	@echo "Weekly artifacts in reports/: week_$(W)_*.{csv,json,md,html} and enhanced files"

week-open:
	@echo "Opening weekly enhanced dashboard for week $(W)"
	@open reports/week_$(W)_enhanced_dashboard.html 2>/dev/null || true

week-update:
	@echo "Updating data for season $(SEASON), week $(WEEK)..."
	$(PYTHON) -c "from data_pipeline import update_week; update_week(int('$(SEASON)'), int('$(WEEK)'))"

week-predict:
	@echo "Generating projections for season $(SEASON), week $(WEEK)..."
	$(PYTHON) -c "from models.position_specific import predict_week; predict_week(int('$(SEASON)'), int('$(WEEK)'))"

week-materialize:
	@echo "Materializing value view for season $(SEASON), week $(WEEK)..."
	$(PYTHON) scripts/materialize_value_view.py --season $(SEASON) --week $(WEEK)

mini-backtest:
	@echo "Running mini backtest for season $(SEASON), week $(WEEK)..."
	$(PYTHON) backtest_replay.py --season $(SEASON) --weeks $(WEEK) --dry-run

health:
	@echo "Checking feed freshness for season $(SEASON), week $(WEEK)..."
	$(PYTHON) health_check.py --season $(SEASON) --week $(WEEK)

# Activate betting: scrape props, compute edges, persist to DB
activate-betting:
	@echo "Activating value betting pipeline with $(ENV_TYPE)..."
	@start_time=$$(date +%s); \
	$(PYTHON) scripts/quick_activate.py; \
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

# Populate historical data (helper target)
populate-data:
	@echo "Populating NFL database with UV speed..."
	$(PYTHON) scripts/quick_populate.py

# Train models (helper target)
train-models:
	@echo "🤖 Training models..."
	$(PYTHON) scripts/quick_train.py

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
	@pip install -r requirements.txt
	@echo "Setting up database..."
	@python data_pipeline.py
	@echo "Running validation..."
	@python cross_season_validation.py
	@echo "Starting pipeline scheduler..."
	@python pipeline_scheduler.py &
	@echo "Launching dashboard..."
	@streamlit run dashboard/main_dashboard.py &
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
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	@echo "Clean complete!"

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
	cp data/nfl.db data/backups/$(shell date +%Y%m%d)/
	cp logs/*.csv data/backups/$(shell date +%Y%m%d)/ 2>/dev/null || true
	@echo "Backup complete in data/backups/$(shell date +%Y%m%d)/"

# Database maintenance
db-maintenance:
	@echo "Running database maintenance..."
	python -c "import sqlite3; conn = sqlite3.connect('data/nfl.db'); conn.execute('VACUUM'); conn.close()"
	@echo "Database optimized!" 
# ============================================================================
# UV ENHANCED TARGETS - 10-100x faster dependency management
# ============================================================================

# Migration to UV (10-100x faster dependency management)
migrate-to-uv:
	@echo "Migrating to UV for 10-100x faster dependency management..."
	python migrate_to_uv.py
	@echo "Migration complete\! Use 'make install-uv' or set ENV_TYPE=uv"

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

# ============================================================================
# CACHE MANAGEMENT TARGETS - Comprehensive API caching system
# ============================================================================

# Show cache statistics and performance metrics
cache-stats:
	@echo "Showing cache statistics with $(ENV_TYPE)..."
	$(PYTHON) simple_cache_test.py

# Pre-populate cache with popular endpoints  
cache-warm:
	@echo "Warming cache with popular endpoints using $(ENV_TYPE)..."
	@echo "   This pre-loads common API calls to improve performance"
	$(PYTHON) -c "from cache_cli import CacheCLI; CacheCLI().warm_cache()"

# Test cache functionality and performance
cache-test:
	@echo "Testing cache functionality and performance with $(ENV_TYPE)..."
	$(PYTHON) simple_cache_test.py

# Clean expired cache entries
cache-clean:
	@echo "🧹 Cleaning expired cache entries with $(ENV_TYPE)..."
	$(PYTHON) -c "from cache_cli import CacheCLI; CacheCLI().cleanup()"

# Reset all cache data (use with caution)
cache-reset:
	@echo "Resetting all cache data with $(ENV_TYPE)..."
	$(PYTHON) -c "from cache_cli import CacheCLI; CacheCLI().reset_cache()"

# Cache offline mode test
cache-offline-test:
	@echo "Testing offline mode functionality with $(ENV_TYPE)..."
	$(PYTHON) -c "from cache_cli import CacheCLI; CacheCLI().offline_mode_test()"
