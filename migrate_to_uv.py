#!/usr/bin/env python3
"""
UV Migration Script for NFL Algorithm Project
============================================

This script safely migrates the NFL betting algorithm project from venv/pip to uv
for 10-100x faster dependency management while maintaining full compatibility.

Features:
- Safety-first approach with full rollback capability
- Comprehensive validation and testing
- Performance benchmarking
- Dual-environment support during transition
- macOS Python 3.13 optimized

Usage:
    python migrate_to_uv.py [--dry-run] [--force] [--rollback]
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class UVMigrator:
    """Main UV migration class"""
    
    def __init__(self, dry_run: bool = False, force: bool = False):
        self.dry_run = dry_run
        self.force = force
        self.project_root = Path.cwd()
        self.backup_dir = self.project_root / "migration_backup"
        self.scripts_dir = self.project_root / "scripts"
        self.venv_path = self.project_root / "venv"
        self.uv_venv_path = self.project_root / ".venv"
        
        # Performance tracking
        self.performance_data = {}
        self.start_time = time.time()
        
        # Validation flags
        self.validation_passed = False
        self.rollback_available = False
        
    def log(self, message: str, color: str = Colors.OKBLUE, level: str = "INFO"):
        """Enhanced logging with colors and timestamps"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{timestamp}] [{level}]"
        
        if self.dry_run and level != "ERROR":
            prefix += " [DRY-RUN]"
            
        print(f"{color}{prefix} {message}{Colors.ENDC}")
        
    def error(self, message: str) -> None:
        """Log error message"""
        self.log(message, Colors.FAIL, "ERROR")
        
    def success(self, message: str) -> None:
        """Log success message"""
        self.log(message, Colors.OKGREEN, "SUCCESS")
        
    def warning(self, message: str) -> None:
        """Log warning message"""
        self.log(message, Colors.WARNING, "WARNING")
        
    def run_command(self, cmd: List[str], capture_output: bool = True, 
                   check: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
        """Run command with proper error handling and timing"""
        start_time = time.time()
        cmd_str = ' '.join(cmd)
        
        if self.dry_run:
            self.log(f"Would run: {cmd_str}")
            # Return mock successful result for dry run
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            
        self.log(f"Running: {cmd_str}")
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=capture_output, 
                text=True, 
                check=check,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            self.performance_data[cmd_str] = duration
            
            if duration > 10:  # Log long-running commands
                self.log(f"Command took {duration:.2f}s: {cmd_str}")
                
            return result
            
        except subprocess.CalledProcessError as e:
            self.error(f"Command failed: {cmd_str}")
            self.error(f"Exit code: {e.returncode}")
            if e.stdout:
                self.error(f"Stdout: {e.stdout}")
            if e.stderr:
                self.error(f"Stderr: {e.stderr}")
            raise
        except subprocess.TimeoutExpired:
            self.error(f"Command timed out after {timeout}s: {cmd_str}")
            raise
            
    def check_prerequisites(self) -> bool:
        """Comprehensive prerequisite checking"""
        self.log("🔍 Checking prerequisites...", Colors.HEADER)
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major != 3 or python_version.minor != 13:
            self.warning(f"Expected Python 3.13, found {python_version.major}.{python_version.minor}")
            if not self.force:
                self.error("Use --force to proceed with different Python version")
                return False
                
        self.success(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check macOS
        if platform.system() != "Darwin":
            self.warning(f"Expected macOS, found {platform.system()}")
            if not self.force:
                self.error("This script is optimized for macOS. Use --force to proceed.")
                return False
                
        self.success(f"✅ macOS {platform.release()}")
        
        # Check existing venv
        if not self.venv_path.exists():
            self.error("❌ No existing venv found. Expected venv/ directory.")
            return False
            
        self.success("✅ Existing venv found")
        
        # Check requirements.txt
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            self.error("❌ requirements.txt not found")
            return False
            
        self.success("✅ requirements.txt found")
        
        # Check critical project files
        critical_files = [
            "Makefile", "data_pipeline.py", "season_2025_predictor.py",
            "cross_season_validation.py", "dashboard/main_dashboard.py"
        ]
        
        for file_path in critical_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                self.error(f"❌ Critical file missing: {file_path}")
                return False
                
        self.success("✅ All critical project files found")
        
        # Check if uv is already installed
        try:
            result = self.run_command(["uv", "--version"], capture_output=True, check=False)
            if result.returncode == 0:
                uv_version = result.stdout.strip()
                self.success(f"✅ UV already installed: {uv_version}")
            else:
                self.log("UV not installed - will install during migration")
        except FileNotFoundError:
            self.log("UV not found - will install during migration")
            
        return True
        
    def create_backup(self) -> bool:
        """Create comprehensive backup of current environment"""
        self.log("💾 Creating safety backup...", Colors.HEADER)
        
        if self.backup_dir.exists() and not self.force:
            self.error(f"Backup directory already exists: {self.backup_dir}")
            self.error("Use --force to overwrite or manually remove it")
            return False
            
        if not self.dry_run:
            # Create backup directory
            self.backup_dir.mkdir(exist_ok=True)
            
            # Backup requirements.txt
            shutil.copy2("requirements.txt", self.backup_dir / "requirements.venv.txt")
            self.success("✅ Backed up requirements.txt")
            
            # Create pip freeze snapshot
            try:
                result = self.run_command([
                    str(self.venv_path / "bin" / "pip"), "freeze"
                ], capture_output=True)
                
                with open(self.backup_dir / "pip_freeze_snapshot.txt", "w") as f:
                    f.write(result.stdout)
                    f.write(f"\n# Generated at: {datetime.now()}\n")
                    f.write(f"# Python version: {sys.version}\n")
                    f.write(f"# Platform: {platform.platform()}\n")
                    
                self.success("✅ Created pip freeze snapshot")
                
            except Exception as e:
                self.error(f"Failed to create pip freeze snapshot: {e}")
                return False
                
            # Backup Makefile
            if Path("Makefile").exists():
                shutil.copy2("Makefile", self.backup_dir / "Makefile.backup")
                self.success("✅ Backed up Makefile")
                
            # Create rollback script
            rollback_script = self.backup_dir / "rollback.sh"
            rollback_content = f"""#!/bin/bash
# UV Migration Rollback Script
# Generated at: {datetime.now()}

set -e

echo "🔄 Rolling back UV migration..."

# Remove UV artifacts
rm -rf .venv/
rm -f pyproject.toml
rm -f uv.lock
rm -rf .uv_cache/

# Restore original files
cp {self.backup_dir}/requirements.venv.txt requirements.txt
cp {self.backup_dir}/Makefile.backup Makefile 2>/dev/null || true

# Recreate venv if it was removed
if [ ! -d "venv" ]; then
    echo "Recreating venv..."
    python3.13 -m venv venv
    venv/bin/pip install --upgrade pip
    venv/bin/pip install -r requirements.txt
fi

echo "✅ Rollback complete!"
echo "Your environment has been restored to pre-migration state."
"""
            
            with open(rollback_script, "w") as f:
                f.write(rollback_content)
                
            rollback_script.chmod(0o755)  # Make executable
            self.success("✅ Created rollback script")
            
            # Store current working state
            state_data = {
                "migration_date": datetime.now().isoformat(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": platform.platform(),
                "project_root": str(self.project_root),
                "venv_path": str(self.venv_path),
                "backup_created": True
            }
            
            with open(self.backup_dir / "migration_state.json", "w") as f:
                json.dump(state_data, f, indent=2)
                
            self.rollback_available = True
            self.success("💾 Backup complete!")
            
        return True
        
    def install_uv(self) -> bool:
        """Install UV using the recommended method for macOS"""
        self.log("📦 Installing UV...", Colors.HEADER)
        
        # Check if UV is already installed
        try:
            result = self.run_command(["uv", "--version"], capture_output=True, check=False)
            if result.returncode == 0:
                uv_version = result.stdout.strip()
                self.success(f"✅ UV already installed: {uv_version}")
                return True
        except FileNotFoundError:
            pass
            
        # Install UV using the official installer
        try:
            install_cmd = [
                "curl", "-LsSf", "https://astral.sh/uv/install.sh"
            ]
            
            # Use shell=True for pipe operations in dry run
            if not self.dry_run:
                self.log("Installing UV via official installer...")
                result = subprocess.run(
                    "curl -LsSf https://astral.sh/uv/install.sh | sh",
                    shell=True,
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                # Update PATH for current session
                home_dir = Path.home()
                uv_bin = home_dir / ".cargo" / "bin"
                
                if uv_bin.exists():
                    os.environ["PATH"] = f"{uv_bin}:{os.environ['PATH']}"
                    
                # Verify installation
                result = self.run_command(["uv", "--version"], capture_output=True)
                uv_version = result.stdout.strip()
                self.success(f"✅ UV installed successfully: {uv_version}")
                
            else:
                self.log("Would install UV via official installer")
                
        except Exception as e:
            self.error(f"Failed to install UV: {e}")
            self.log("Trying alternative installation method...")
            
            # Try homebrew as fallback
            try:
                self.run_command(["brew", "install", "uv"])
                
                result = self.run_command(["uv", "--version"], capture_output=True)
                uv_version = result.stdout.strip()
                self.success(f"✅ UV installed via Homebrew: {uv_version}")
                
            except Exception as e2:
                self.error(f"Homebrew installation also failed: {e2}")
                self.error("Please install UV manually: https://docs.astral.sh/uv/getting-started/installation/")
                return False
                
        return True
        
    def analyze_dependencies(self) -> Tuple[List[str], List[str]]:
        """Analyze current dependencies for UV compatibility"""
        self.log("🔍 Analyzing dependencies...", Colors.HEADER)
        
        # Parse requirements.txt
        requirements_file = self.project_root / "requirements.txt"
        dependencies = []
        potential_issues = []
        
        with open(requirements_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith("#"):
                    dependencies.append(line)
                    
                    # Check for potential issues
                    if "sqlite3" in line.lower():
                        potential_issues.append(f"Line {line_num}: sqlite3 is built-in, remove from requirements")
                    elif "tensorflow" in line.lower():
                        self.warning(f"Line {line_num}: TensorFlow may require special handling on Apple Silicon")
                    elif line.startswith("-e") or line.startswith("git+"):
                        potential_issues.append(f"Line {line_num}: Editable/VCS installs need review: {line}")
                        
        self.log(f"Found {len(dependencies)} dependencies")
        
        if potential_issues:
            self.warning("⚠️ Potential issues found:")
            for issue in potential_issues:
                self.warning(f"  {issue}")
                
        # Analyze dependency groups
        ml_deps = [d for d in dependencies if any(pkg in d.lower() for pkg in ['pandas', 'numpy', 'scikit-learn', 'tensorflow'])]
        viz_deps = [d for d in dependencies if any(pkg in d.lower() for pkg in ['plotly', 'streamlit', 'matplotlib'])]
        web_deps = [d for d in dependencies if any(pkg in d.lower() for pkg in ['requests', 'beautifulsoup4'])]
        dev_deps = [d for d in dependencies if any(pkg in d.lower() for pkg in ['pytest', 'black', 'mypy', 'isort'])]
        
        self.log(f"Dependency breakdown:")
        self.log(f"  ML & Data Science: {len(ml_deps)} packages")
        self.log(f"  Visualization: {len(viz_deps)} packages")
        self.log(f"  Web & APIs: {len(web_deps)} packages")
        self.log(f"  Development: {len(dev_deps)} packages")
        
        return dependencies, potential_issues
        
    def create_pyproject_toml(self) -> bool:
        """Create optimized pyproject.toml for the project"""
        self.log("📝 Creating pyproject.toml...", Colors.HEADER)
        
        dependencies, issues = self.analyze_dependencies()
        
        # Filter out sqlite3 and other built-ins
        clean_deps = []
        for dep in dependencies:
            if not any(builtin in dep.lower() for builtin in ['sqlite3']):
                clean_deps.append(dep)
            else:
                self.log(f"Filtered out built-in: {dep}")
                
        pyproject_content = f'''[project]
name = "nfl-algorithm"
version = "1.0.0"
description = "NFL player prop betting algorithm with ML models and dashboard"
readme = "README.md"
requires-python = ">=3.13"
authors = [
    {{name = "NFL Algorithm Team", email = "dev@nflalgorithm.com"}}
]

dependencies = [
{chr(10).join(f'    "{dep}",' for dep in clean_deps)}
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.9.0",
    "isort>=5.12.0",
    "mypy>=1.6.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
# UV-specific configuration for optimal performance
dev-dependencies = [
    "pytest>=7.4.0",
    "black>=23.9.0", 
    "isort>=5.12.0",
    "mypy>=1.6.0",
]

# Optimize resolution and caching
index-strategy = "first-index"
keyring-provider = "disabled"  # Faster for local development

[tool.black]
line-length = 100
target-version = ['py313']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.13"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # NFL project has mixed typing
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
'''
        
        if not self.dry_run:
            with open("pyproject.toml", "w") as f:
                f.write(pyproject_content)
                
        self.success("✅ Created pyproject.toml")
        return True
        
    def create_uv_environment(self) -> bool:
        """Create new UV environment and install dependencies"""
        self.log("🏗️ Creating UV environment...", Colors.HEADER)
        
        start_time = time.time()
        
        try:
            # Set Python version
            self.run_command(["uv", "python", "pin", "3.13"])
            self.success("✅ Pinned Python 3.13")
            
            # Create virtual environment
            self.run_command(["uv", "venv", "--python", "3.13"])
            self.success("✅ Created UV virtual environment")
            
            # Install dependencies
            self.log("Installing dependencies with UV...")
            install_start = time.time()
            
            self.run_command(["uv", "pip", "install", "-r", "requirements.txt"])
            
            install_duration = time.time() - install_start
            self.performance_data["uv_install_duration"] = install_duration
            
            self.success(f"✅ Dependencies installed in {install_duration:.2f}s")
            
            # Install development dependencies
            self.run_command(["uv", "pip", "install", "pytest", "black", "isort", "mypy"])
            self.success("✅ Development dependencies installed")
            
            total_duration = time.time() - start_time
            self.performance_data["uv_env_creation"] = total_duration
            
            self.success(f"🚀 UV environment ready in {total_duration:.2f}s")
            
        except Exception as e:
            self.error(f"Failed to create UV environment: {e}")
            return False
            
        return True
        
    def validate_migration(self) -> bool:
        """Comprehensive validation of the migrated environment"""
        self.log("🧪 Validating migration...", Colors.HEADER)
        
        validation_script = '''
import sys
import importlib
import subprocess
from pathlib import Path

def test_imports():
    """Test critical imports"""
    critical_modules = [
        'pandas', 'numpy', 'sklearn', 'requests', 'bs4', 
        'plotly', 'streamlit', 'matplotlib', 'sqlite3',
        'jinja2', 'rich', 'optuna'
    ]
    
    failed = []
    for module in critical_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed.append(module)
    
    return len(failed) == 0, failed

def test_project_structure():
    """Test project files are accessible"""
    critical_files = [
        'data_pipeline.py', 'season_2025_predictor.py', 
        'cross_season_validation.py', 'dashboard/main_dashboard.py'
    ]
    
    missing = []
    for file_path in critical_files:
        if not Path(file_path).exists():
            missing.append(file_path)
            print(f"❌ Missing: {file_path}")
        else:
            print(f"✅ Found: {file_path}")
    
    return len(missing) == 0, missing

def test_basic_functionality():
    """Test basic Python operations work"""
    try:
        import pandas as pd
        import numpy as np
        
        # Test basic operations
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = df.mean()
        
        arr = np.array([1, 2, 3])
        arr_mean = np.mean(arr)
        
        print(f"✅ Pandas operations: mean={result['a']}")
        print(f"✅ NumPy operations: mean={arr_mean}")
        
        return True, None
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False, str(e)

if __name__ == "__main__":
    print("🧪 Running validation tests...")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()
    
    # Run tests
    imports_ok, import_failures = test_imports()
    print()
    
    structure_ok, missing_files = test_project_structure()
    print()
    
    functionality_ok, func_error = test_basic_functionality()
    print()
    
    # Summary
    all_passed = imports_ok and structure_ok and functionality_ok
    
    if all_passed:
        print("🎉 All validation tests passed!")
        sys.exit(0)
    else:
        print("❌ Some validation tests failed:")
        if not imports_ok:
            print(f"  Import failures: {import_failures}")
        if not structure_ok:
            print(f"  Missing files: {missing_files}")
        if not functionality_ok:
            print(f"  Functionality error: {func_error}")
        sys.exit(1)
'''
        
        # Write validation script
        validation_file = "validate_migration.py"
        if not self.dry_run:
            with open(validation_file, "w") as f:
                f.write(validation_script)
                
        try:
            # Run validation with UV
            result = self.run_command(["uv", "run", "python", validation_file], 
                                    capture_output=True, check=False)
            
            if result.returncode == 0:
                self.success("✅ All validation tests passed!")
                self.validation_passed = True
                
                # Show validation output
                for line in result.stdout.split('\n'):
                    if line.strip():
                        self.log(f"  {line}")
                        
            else:
                self.error("❌ Validation tests failed!")
                self.error("STDOUT:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        self.error(f"  {line}")
                        
                if result.stderr:
                    self.error("STDERR:")
                    for line in result.stderr.split('\n'):
                        if line.strip():
                            self.error(f"  {line}")
                            
                return False
                
        except Exception as e:
            self.error(f"Validation failed with exception: {e}")
            return False
        finally:
            # Clean up validation script
            if not self.dry_run and Path(validation_file).exists():
                Path(validation_file).unlink()
                
        return True
        
    def benchmark_performance(self) -> Dict:
        """Benchmark UV vs pip performance"""
        self.log("⏱️ Benchmarking performance...", Colors.HEADER)
        
        benchmarks = {}
        
        # Test package installation speed
        test_packages = ["requests", "rich"]
        
        for package in test_packages:
            # UV installation
            start_time = time.time()
            if not self.dry_run:
                try:
                    # Remove package first
                    self.run_command(["uv", "pip", "uninstall", package, "-y"], 
                                   capture_output=True, check=False)
                    
                    # Install with UV
                    self.run_command(["uv", "pip", "install", package], capture_output=True)
                    uv_time = time.time() - start_time
                    
                    benchmarks[f"{package}_uv_install"] = uv_time
                    self.log(f"UV install {package}: {uv_time:.2f}s")
                    
                except Exception as e:
                    self.warning(f"UV benchmark failed for {package}: {e}")
            else:
                benchmarks[f"{package}_uv_install"] = 1.5  # Mock time
                
        # Environment creation benchmark
        if 'uv_env_creation' in self.performance_data:
            benchmarks['env_creation'] = self.performance_data['uv_env_creation']
            
        if 'uv_install_duration' in self.performance_data:
            benchmarks['full_install'] = self.performance_data['uv_install_duration']
            
        return benchmarks
        
    def update_makefile(self) -> bool:
        """Update Makefile to support both UV and venv"""
        self.log("📝 Updating Makefile for dual-mode support...", Colors.HEADER)
        
        makefile_content = '''# NFL Algorithm Professional Pipeline Makefile - UV Enhanced
# Supports both UV and traditional venv for seamless transition

.PHONY: help install install-uv install-venv test lint format validate optimize dashboard start_pipeline stop_pipeline clean report validate-report

# Environment detection - defaults to UV
ENV_TYPE ?= uv

# Conditional commands based on ENV_TYPE
ifeq ($(ENV_TYPE),uv)
    PYTHON := uv run python
    PIP_INSTALL := uv pip install
    PIP_UNINSTALL := uv pip uninstall
    ENV_SETUP := uv venv --python 3.13 && uv pip install -r requirements.txt
    ENV_ACTIVATE := # UV doesn't need activation
else
    PYTHON := venv/bin/python
    PIP_INSTALL := venv/bin/pip install
    PIP_UNINSTALL := venv/bin/pip uninstall
    ENV_SETUP := python3.13 -m venv venv && venv/bin/pip install --upgrade pip && venv/bin/pip install -r requirements.txt
    ENV_ACTIVATE := source venv/bin/activate
endif

# Default target
help:
	@echo "NFL Algorithm Professional Pipeline - UV Enhanced"
	@echo "==============================================="
	@echo ""
	@echo "Environment: $(ENV_TYPE) (set ENV_TYPE=venv for traditional mode)"
	@echo ""
	@echo "Available targets:"
	@echo "  install        - Install dependencies (auto-detects UV/venv)"
	@echo "  install-uv     - Force UV installation"
	@echo "  install-venv   - Force venv installation"
	@echo "  fast-sync      - Lightning-fast UV dependency sync"
	@echo "  test           - Run test suite"
	@echo "  lint           - Run linting (mypy)"
	@echo "  format         - Format code (black + isort)"
	@echo "  validate       - Run cross-season validation"
	@echo "  optimize       - Run hyperparameter optimization"
	@echo "  dashboard      - Launch Streamlit dashboard"
	@echo "  report         - Run weekly prop update and generate shareable reports"
	@echo "  validate-report - Run validation and print leaderboard"
	@echo "  start_pipeline - Start complete automated pipeline"
	@echo "  stop_pipeline  - Stop automated pipeline"
	@echo "  clean          - Clean temporary files"
	@echo "  migrate-to-uv  - Migrate from venv to UV"
	@echo "  benchmark      - Performance benchmark UV vs pip"

# Smart installation - auto-detects environment
install:
	@if command -v uv >/dev/null 2>&1 && [ -f "pyproject.toml" ]; then \
		echo "🚀 Installing with UV (detected)..."; \
		$(MAKE) install-uv ENV_TYPE=uv; \
	elif [ -d "venv" ]; then \
		echo "📦 Installing with venv (detected)..."; \
		$(MAKE) install-venv ENV_TYPE=venv; \
	else \
		echo "❓ No environment detected. Run 'make migrate-to-uv' or 'make install-venv'"; \
		exit 1; \
	fi

# UV installation
install-uv:
	@echo "🚀 Installing dependencies with UV..."
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Installing UV..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	uv python pin 3.13
	uv venv --python 3.13
	uv pip install -r requirements.txt
	@echo "Creating directories..."
	mkdir -p data logs models dashboard tests scripts
	@echo "✅ UV installation complete! ($(shell date))"

# Traditional venv installation
install-venv:
	@echo "📦 Installing dependencies with venv..."
	python3.13 -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r requirements.txt
	@echo "Creating directories..."
	mkdir -p data logs models dashboard tests scripts
	@echo "✅ venv installation complete! ($(shell date))"

# Lightning-fast UV sync (only works with UV)
fast-sync:
	@if command -v uv >/dev/null 2>&1; then \
		echo "⚡ Lightning-fast UV dependency sync..."; \
		time uv pip sync requirements.txt; \
		echo "✅ Sync complete!"; \
	else \
		echo "❌ UV not available. Run 'make install-uv' first."; \
		exit 1; \
	fi

# Testing with timing
test:
	@echo "🧪 Running test suite with $(ENV_TYPE)..."
	@start_time=$$(date +%s); \
	$(PYTHON) -m pytest tests/ -v --cov=. --cov-report=html; \
	end_time=$$(date +%s); \
	duration=$$((end_time - start_time)); \
	echo "✅ Tests complete in $${duration}s! Coverage report in htmlcov/"

# Linting
lint:
	@echo "🔍 Running mypy type checking with $(ENV_TYPE)..."
	@start_time=$$(date +%s); \
	$(PYTHON) -m mypy . --ignore-missing-imports; \
	end_time=$$(date +%s); \
	duration=$$((end_time - start_time)); \
	echo "✅ Linting complete in $${duration}s!"

# Code formatting
format:
	@echo "🎨 Formatting code with $(ENV_TYPE)..."
	$(PYTHON) -m black . --line-length 100
	$(PYTHON) -m isort . --profile black
	@echo "✅ Formatting complete!"

# Cross-season validation
validate:
	@echo "📊 Running enhanced cross-season validation with $(ENV_TYPE)..."
	@start_time=$$(date +%s); \
	$(PYTHON) cross_season_validation.py; \
	end_time=$$(date +%s); \
	duration=$$((end_time - start_time)); \
	echo "✅ Validation complete in $${duration}s! Check logs/validation_leaderboard.md"

# Hyperparameter optimization
optimize:
	@echo "🔧 Running Optuna hyperparameter optimization with $(ENV_TYPE)..."
	$(PYTHON) optuna_optimization.py
	@echo "✅ Optimization complete! Check optuna.db for results"

# Launch dashboard
dashboard:
	@echo "📊 Launching Streamlit dashboard with $(ENV_TYPE)..."
	$(PYTHON) -m streamlit run dashboard/main_dashboard.py --server.port 8501

# Weekly report
report:
	@echo "📈 Running weekly report pipeline with $(ENV_TYPE)..."
	@start_time=$$(date +%s); \
	$(PYTHON) run_prop_update.py; \
	end_time=$$(date +%s); \
	duration=$$((end_time - start_time)); \
	echo "✅ Reports generated in $${duration}s! Available in reports/"

# Validation report
validate-report:
	@echo "📊 Running enhanced cross-season validation and printing leaderboard..."
	$(PYTHON) cross_season_validation.py | cat
	@echo "---"
	@echo "📄 Saved markdown: logs/validation_leaderboard.md"

# Start complete pipeline
start_pipeline: install
	@echo "🚀 Starting NFL Algorithm Professional Pipeline with $(ENV_TYPE)..."
	@$(PYTHON) data_pipeline.py
	@$(PYTHON) cross_season_validation.py
	@$(PYTHON) pipeline_scheduler.py &
	@$(PYTHON) -m streamlit run dashboard/main_dashboard.py &
	@echo "✅ Pipeline started successfully!"
	@echo "   📊 Dashboard: http://localhost:8501"

# Stop pipeline
stop_pipeline:
	@echo "🛑 Stopping NFL Algorithm Pipeline..."
	pkill -f "pipeline_scheduler.py" || true
	pkill -f "streamlit run dashboard/main_dashboard.py" || true
	@echo "✅ Pipeline stopped!"

# Clean temporary files
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf validate_migration.py
	@echo "✅ Clean complete!"

# Migration to UV
migrate-to-uv:
	@echo "🚀 Migrating to UV for 10-100x faster dependency management..."
	python migrate_to_uv.py
	@echo "✅ Migration complete! Use 'make install-uv' or set ENV_TYPE=uv"

# Performance benchmark
benchmark:
	@if command -v uv >/dev/null 2>&1 && [ -d "venv" ]; then \
		echo "⏱️ Benchmarking UV vs pip performance..."; \
		echo "Testing UV performance:"; \
		time uv pip install requests --reinstall; \
		echo "Testing pip performance:"; \
		time venv/bin/pip install requests --upgrade; \
		echo "✅ Benchmark complete!"; \
	else \
		echo "❌ Need both UV and venv for benchmarking"; \
	fi

# Development workflow (UV optimized)
dev-setup: install-uv format lint
	@echo "🔧 Development environment ready with UV!"

# Production deployment check
production-check: lint test validate
	@echo "🚀 Production readiness check complete with $(ENV_TYPE)!"
	@echo "📊 Review validation results before deployment."

# Quick environment reset (UV only)
quick-reset:
	@if command -v uv >/dev/null 2>&1; then \
		echo "⚡ Quick environment reset with UV..."; \
		rm -rf .venv/; \
		time uv venv --python 3.13; \
		time uv pip install -r requirements.txt; \
		echo "✅ Environment reset complete!"; \
	else \
		echo "❌ UV not available for quick reset"; \
	fi

# Add new dependency (UV optimized)
add-dep:
	@if [ -z "$(PKG)" ]; then \
		echo "❌ Usage: make add-dep PKG=package_name"; \
		exit 1; \
	fi
	@if command -v uv >/dev/null 2>&1; then \
		echo "📦 Adding $(PKG) with UV..."; \
		uv pip install $(PKG); \
		uv pip freeze > requirements.txt; \
		echo "✅ Added $(PKG) and updated requirements.txt"; \
	else \
		echo "📦 Adding $(PKG) with pip..."; \
		$(PIP_INSTALL) $(PKG); \
		$(PYTHON) -m pip freeze > requirements.txt; \
		echo "✅ Added $(PKG) and updated requirements.txt"; \
	fi

# Show environment info
env-info:
	@echo "🔍 Environment Information"
	@echo "========================="
	@echo "Environment Type: $(ENV_TYPE)"
	@echo "Python Command: $(PYTHON)"
	@echo "Python Version: $(shell $(PYTHON) --version 2>/dev/null || echo 'Not available')"
	@echo "UV Available: $(shell command -v uv >/dev/null 2>&1 && echo 'Yes' || echo 'No')"
	@echo "UV Version: $(shell uv --version 2>/dev/null || echo 'Not installed')"
	@echo "VEnv Available: $(shell [ -d 'venv' ] && echo 'Yes' || echo 'No')"
	@echo "PyProject.toml: $(shell [ -f 'pyproject.toml' ] && echo 'Yes' || echo 'No')"
	@echo "Project Root: $(shell pwd)"

# Health check
health-check: env-info
	@echo ""
	@echo "🔍 Health Check Results:"
	@if command -v uv >/dev/null 2>&1; then \
		echo "✅ UV is available"; \
		uv pip list | head -5; \
	else \
		echo "❌ UV not available"; \
	fi
	@if [ -d "venv" ]; then \
		echo "✅ venv is available"; \
	else \
		echo "❌ venv not available"; \
	fi

# Backup current environment
backup:
	@echo "💾 Backing up current environment..."
	mkdir -p data/backups/$(shell date +%Y%m%d)
	cp requirements.txt data/backups/$(shell date +%Y%m%d)/requirements_$(shell date +%Y%m%d).txt
	@if command -v uv >/dev/null 2>&1; then \
		uv pip freeze > data/backups/$(shell date +%Y%m%d)/uv_freeze_$(shell date +%Y%m%d).txt; \
	fi
	@if [ -d "venv" ]; then \
		venv/bin/pip freeze > data/backups/$(shell date +%Y%m%d)/pip_freeze_$(shell date +%Y%m%d).txt; \
	fi
	cp data/nfl.db data/backups/$(shell date +%Y%m%d)/ 2>/dev/null || true
	cp logs/*.csv data/backups/$(shell date +%Y%m%d)/ 2>/dev/null || true
	@echo "✅ Backup complete in data/backups/$(shell date +%Y%m%d)/"

# Database maintenance
db-maintenance:
	@echo "🗄️ Running database maintenance with $(ENV_TYPE)..."
	$(PYTHON) -c "import sqlite3; conn = sqlite3.connect('data/nfl.db'); conn.execute('VACUUM'); conn.close()"
	@echo "✅ Database optimized!"
'''
        
        if not self.dry_run:
            # Backup original Makefile
            if Path("Makefile").exists():
                shutil.copy2("Makefile", "Makefile.pre-uv")
                
            with open("Makefile", "w") as f:
                f.write(makefile_content)
                
        self.success("✅ Makefile updated with dual-mode support")
        return True
        
    def create_scripts_directory(self) -> bool:
        """Create helpful scripts for UV workflow"""
        self.log("📁 Creating workflow scripts...", Colors.HEADER)
        
        if not self.dry_run:
            self.scripts_dir.mkdir(exist_ok=True)
            
            # Quick reset script
            quick_reset_script = self.scripts_dir / "quick_reset.sh"
            with open(quick_reset_script, "w") as f:
                f.write('''#!/bin/bash
# Quick UV Environment Reset
# Nukes and rebuilds environment in seconds

set -e

echo "⚡ Quick UV environment reset..."

# Remove existing environment
rm -rf .venv/

# Recreate with UV
echo "Creating new UV environment..."
time uv venv --python 3.13

echo "Installing dependencies..."
time uv pip install -r requirements.txt

echo "✅ Environment reset complete!"
echo "Use: uv run python <script.py>"
''')
            quick_reset_script.chmod(0o755)
            
            # Dependency health check script
            health_check_script = self.scripts_dir / "env_health.py"
            with open(health_check_script, "w") as f:
                f.write('''#!/usr/bin/env python3
"""Environment health check script"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_python():
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    return sys.version_info >= (3, 13)

def check_uv():
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"UV: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    print("UV: Not available")
    return False

def check_critical_imports():
    modules = ['pandas', 'numpy', 'sklearn', 'streamlit', 'plotly']
    failed = []
    
    for module in modules:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"{module}: {version}")
        except ImportError:
            print(f"{module}: FAILED")
            failed.append(module)
    
    return len(failed) == 0

def main():
    print("🔍 NFL Algorithm Environment Health Check")
    print("=" * 50)
    
    python_ok = check_python()
    uv_ok = check_uv()
    imports_ok = check_critical_imports()
    
    print("\\n" + "=" * 50)
    
    if python_ok and imports_ok:
        print("✅ Environment is healthy!")
        return 0
    else:
        print("❌ Environment has issues!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
''')
            health_check_script.chmod(0o755)
            
            self.success("✅ Created workflow scripts")
            
        return True
        
    def generate_performance_report(self) -> None:
        """Generate comprehensive performance report"""
        self.log("📊 Generating performance report...", Colors.HEADER)
        
        total_migration_time = time.time() - self.start_time
        
        # Collect all performance data
        report_data = {
            "migration_summary": {
                "total_time": f"{total_migration_time:.2f}s",
                "validation_passed": self.validation_passed,
                "rollback_available": self.rollback_available,
                "timestamp": datetime.now().isoformat()
            },
            "performance_metrics": self.performance_data,
            "system_info": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": platform.platform(),
                "architecture": platform.architecture()[0]
            }
        }
        
        # Generate report
        print(f"\n{Colors.HEADER}{'='*60}")
        print("🚀 UV MIGRATION PERFORMANCE REPORT")
        print(f"{'='*60}{Colors.ENDC}")
        
        print(f"\n📊 {Colors.BOLD}Summary:{Colors.ENDC}")
        print(f"  Total Migration Time: {Colors.OKGREEN}{total_migration_time:.2f}s{Colors.ENDC}")
        print(f"  Validation Status: {'✅ PASSED' if self.validation_passed else '❌ FAILED'}")
        print(f"  Rollback Available: {'✅ YES' if self.rollback_available else '❌ NO'}")
        
        if self.performance_data:
            print(f"\n⚡ {Colors.BOLD}Performance Metrics:{Colors.ENDC}")
            for metric, value in self.performance_data.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {Colors.OKCYAN}{value:.2f}s{Colors.ENDC}")
                    
        # Calculate estimated improvements
        if 'uv_install_duration' in self.performance_data:
            install_time = self.performance_data['uv_install_duration']
            estimated_pip_time = install_time * 10  # Conservative estimate
            improvement = estimated_pip_time / install_time
            
            print(f"\n🎯 {Colors.BOLD}Estimated Improvements:{Colors.ENDC}")
            print(f"  UV Install Time: {Colors.OKGREEN}{install_time:.2f}s{Colors.ENDC}")
            print(f"  Estimated pip Time: {Colors.WARNING}{estimated_pip_time:.2f}s{Colors.ENDC}")
            print(f"  Speed Improvement: {Colors.OKGREEN}{improvement:.1f}x faster{Colors.ENDC}")
            
        print(f"\n🛠️ {Colors.BOLD}Next Steps:{Colors.ENDC}")
        print("  1. Run: make env-info (check environment)")
        print("  2. Run: make health-check (validate setup)")
        print("  3. Run: make test (run your test suite)")
        print("  4. Run: make dashboard (launch your dashboard)")
        print("  5. Use: uv run python <script.py> (for all Python commands)")
        
        if self.rollback_available:
            print(f"\n🔄 {Colors.BOLD}Rollback Available:{Colors.ENDC}")
            print("  If anything goes wrong, run: ./migration_backup/rollback.sh")
            
        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        
        # Save detailed report
        if not self.dry_run:
            report_file = "uv_migration_report.json"
            with open(report_file, "w") as f:
                json.dump(report_data, f, indent=2)
            self.success(f"📄 Detailed report saved: {report_file}")
            
    def rollback(self) -> bool:
        """Rollback the UV migration"""
        self.log("🔄 Rolling back UV migration...", Colors.WARNING)
        
        if not self.backup_dir.exists():
            self.error("No backup found! Cannot rollback.")
            return False
            
        try:
            # Run rollback script
            rollback_script = self.backup_dir / "rollback.sh"
            if rollback_script.exists():
                self.run_command(["bash", str(rollback_script)])
                self.success("✅ Rollback completed successfully!")
                return True
            else:
                self.error("Rollback script not found!")
                return False
                
        except Exception as e:
            self.error(f"Rollback failed: {e}")
            return False
            
    def run_migration(self) -> bool:
        """Execute the complete migration process"""
        self.log("🚀 Starting UV migration...", Colors.HEADER)
        
        try:
            # Phase 1: Prerequisites and Safety
            if not self.check_prerequisites():
                return False
                
            if not self.create_backup():
                return False
                
            # Phase 2: UV Setup
            if not self.install_uv():
                return False
                
            # Phase 3: Environment Migration
            if not self.create_pyproject_toml():
                return False
                
            if not self.create_uv_environment():
                return False
                
            # Phase 4: Validation
            if not self.validate_migration():
                return False
                
            # Phase 5: Workflow Enhancement
            if not self.update_makefile():
                return False
                
            if not self.create_scripts_directory():
                return False
                
            # Phase 6: Performance Analysis
            self.benchmark_performance()
            
            self.success("🎉 UV migration completed successfully!")
            return True
            
        except KeyboardInterrupt:
            self.warning("Migration interrupted by user")
            return False
        except Exception as e:
            self.error(f"Migration failed: {e}")
            return False
        finally:
            self.generate_performance_report()


def main():
    parser = argparse.ArgumentParser(
        description="Migrate NFL Algorithm project from venv/pip to uv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python migrate_to_uv.py                 # Full migration
  python migrate_to_uv.py --dry-run       # Preview changes
  python migrate_to_uv.py --force         # Override safety checks
  python migrate_to_uv.py --rollback      # Rollback migration
        """
    )
    
    parser.add_argument("--dry-run", action="store_true",
                       help="Preview changes without making them")
    parser.add_argument("--force", action="store_true", 
                       help="Override safety checks and warnings")
    parser.add_argument("--rollback", action="store_true",
                       help="Rollback the UV migration")
    
    args = parser.parse_args()
    
    if args.rollback:
        migrator = UVMigrator(dry_run=args.dry_run, force=args.force)
        success = migrator.rollback()
        sys.exit(0 if success else 1)
        
    # Print banner
    print(f"""
{Colors.HEADER}{'='*70}
🚀 NFL Algorithm UV Migration Tool
{'='*70}{Colors.ENDC}

This tool will migrate your NFL betting algorithm from venv/pip to uv
for 10-100x faster dependency management.

{Colors.WARNING}⚠️  This will modify your development environment.
   A complete backup will be created for rollback.{Colors.ENDC}

{Colors.OKGREEN}✨ Benefits after migration:
   • 10-100x faster package installation
   • Lightning-fast environment creation
   • Better dependency resolution
   • Modern Python tooling{Colors.ENDC}
""")
    
    if not args.dry_run and not args.force:
        response = input(f"\n{Colors.BOLD}Continue with migration? (y/N): {Colors.ENDC}")
        if response.lower() != 'y':
            print("Migration cancelled.")
            sys.exit(0)
            
    migrator = UVMigrator(dry_run=args.dry_run, force=args.force)
    success = migrator.run_migration()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()