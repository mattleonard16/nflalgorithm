# File Organization Guide

This document explains the project's file organization structure and how to maintain it.

## Directory Structure

```
nflalgorithm/
├── archive/              # Historical artifacts (git-tracked)
│   ├── databases/       # Old database backups
│   ├── docs/            # Legacy documentation
│   ├── exports/         # Old CSV exports from root
│   ├── logs_old/        # Old log files from root
│   ├── misc/            # Miscellaneous archived files
│   └── reports/         # Weekly report snapshots
│
├── data/                # Active data files
│   ├── backups/         # Database backups (YYYYMMDD/)
│   ├── odds/            # Raw odds JSON files
│   └── *.csv            # Active CSV data
│
├── dashboard/           # Streamlit dashboard code
├── docs/                # Current documentation
├── logs/                # Active log files
│   └── metrics/         # Weekly metrics JSON
│
├── models/              # Trained model files (.joblib)
│   ├── position_specific/
│   └── weekly/
│
├── reports/             # Current week's reports (HTML/CSV/MD)
├── scripts/             # Utility scripts
├── tests/               # Test suite
├── utils/               # Shared utilities
│
└── Root files:
    ├── *.py             # Main application code
    ├── Makefile         # Build commands
    ├── README.md        # Documentation
    ├── requirements.txt # Dependencies
    └── pyproject.toml   # UV/poetry config
```

## File Categories

### ✅ Keep in Root (Active Code)
- `*.py` - Application code
- `Makefile` - Build commands
- `README.md`, `*.md` docs - Documentation
- `requirements.txt`, `pyproject.toml` - Dependencies
- `config.py`, `.env` - Configuration

### 📦 Archive Candidates (Root Clutter)

**Export CSVs:**
- `export10.csv`, `week102.csv` → `archive/exports/`

**Log Files:**
- `prop_update.log` → `archive/logs_old/`
- Active logs stay in `logs/` directory

**Database Files:**
- `nfl_data.db` (if duplicate) → `archive/databases/`
- `nfl_prop_lines.db` → `archive/databases/`
- `optuna.db` → `archive/databases/`
- Active DB should be in `data/` or configured path

### 📁 Archive Structure

```
archive/
├── databases/           # Old .db files
├── exports/             # Old CSV exports
├── logs_old/            # Old .log files  
├── misc/                # Miscellaneous files
├── reports/             # Weekly snapshots
│   └── YYYY-MM-DD-week-N/
└── docs/                # Legacy docs
```

## Maintenance Commands

### Clean Root Clutter
```bash
make clean-root
```
Moves root clutter files to appropriate archive subdirectories.

### Archive Weekly Reports
```bash
make archive-week WEEK=5
```
Creates a snapshot of current reports in `archive/reports/YYYY-MM-DD-week-5/`.

### Full Archive
```bash
make archive-all WEEK=5
```
Runs both `archive-week` and `clean-root` to fully organize files.

### Clean Temporary Files
```bash
make clean
```
Removes caches, coverage files, `.pyc` files, and `__pycache__` directories.

## Git Tracking

The `archive/` directory is **tracked in git** so you can:
- Keep historical snapshots
- Review past reports
- Restore old files if needed

**Do not track:**
- `*.db` files (unless needed for tests)
- `*.log` files
- `__pycache__/`, `.pytest_cache/`
- `htmlcov/`, `.coverage`

## Best Practices

1. **After generating reports:** Run `make archive-week WEEK=N`
2. **Weekly cleanup:** Run `make clean-root` to move stray files
3. **Before commits:** Run `make clean` to remove temporary files
4. **Monthly:** Review `archive/` and remove very old files if disk space is tight

## Active vs Archived

| Location | Purpose | Frequency | Size |
|----------|---------|-----------|------|
| `reports/` | Current week's outputs | Weekly refresh | ~10MB |
| `archive/reports/` | Historical snapshots | Permanent | Growing |
| `logs/` | Active logs | Daily rotation | <1MB |
| `archive/logs_old/` | Old logs | Reference only | Small |
| `data/` | Active datasets | Updated regularly | ~100MB |
| `archive/databases/` | Old DB backups | Reference only | ~500MB |

