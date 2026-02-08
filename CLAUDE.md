# NFL Algorithm - Claude Session Notes

## Quick Start (Fresh Setup)

No .env file needed — SQLite is the default local dev database.

### Steps:
1. Install dependencies: `make install`
2. Run schema migrations:
   ```bash
   DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python -c "from schema_migrations import MigrationManager; MigrationManager('nfl_data.db').run()"
   ```
3. Ingest real NFL data: `make ingest-nfl`
4. Run tests: `make test`
5. Generate projections: `make week-predict SEASON=2025 WEEK=13`
6. Materialize for dashboard: `make week-materialize SEASON=2025 WEEK=13`
7. Launch full stack: `make fullstack`

---

## Environment Configuration

- **Database**: SQLite for local dev (`DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db`)
- All Makefile targets automatically set DB env vars via `$(DB_ENV)`
- MySQL available for production via `DB_URL` env var
- `ODDS_API_KEY` needed only for live odds scraping (not required for dev)

---

## Proprietary Files (.gitignored)

These files are excluded from version control:

| File | Purpose |
|------|---------|
| `config.py` | Centralized configuration (database, API, model, betting settings) |
| `data_pipeline.py` | Data ingestion, feature engineering, EWMA market mu computation |
| `value_betting_engine.py` | Kelly criterion, probability calculations, value ranking |
| `prop_integration.py` | 3-tier player matching (odds to projections) |
| `models/position_specific/weekly.py` | Weekly model training and prediction |
| `api/server.py` | FastAPI REST API for frontend dashboard |
| `scripts/record_outcomes.py` | Bet grading and outcome recording |

---

## Key Configuration Values

From `config.py`:

- `config.model.target_mae = 3.0` (professional-grade target)
- `config.betting.min_edge_threshold = 0.08` (8% minimum edge)
- `config.betting.min_confidence = 0.75`
- `config.integration.ewma_decay = 0.65`
- WR role priors: alpha=75, secondary=55, slot=45, fringe=30
- Minimum mu floor: 15.0

---

## Common Commands

```bash
# Install
make install

# Ingest data (2024+2025 seasons)
make ingest-nfl

# Run tests
make test

# Weekly workflow
make week-predict SEASON=2025 WEEK=13
make week-materialize SEASON=2025 WEEK=13
make week-grade SEASON=2025 WEEK=13

# Launch services
make api          # FastAPI on :8000
make frontend-dev # Next.js on :3000
make fullstack    # Both
make dashboard    # Streamlit on :8501
```

---

## nflverse/nflreadpy Reference

### Available Functions
```python
import nflreadpy as nfl

# Core data
nfl.load_player_stats([2025])      # Weekly player stats
nfl.load_pbp([2025])               # Play-by-play with EPA
nfl.load_snap_counts([2025])       # Snap counts
nfl.load_schedules([2025])         # Game schedule
nfl.load_rosters([2025])           # Current rosters
nfl.load_depth_charts([2025])      # Depth charts
nfl.load_ftn_charting([2025])      # Route/target data
```

### Update Cadence
- Player/team stats: Nightly after games
- Schedules: Every 5 minutes in-season
- Depth charts: Daily 07:00 UTC
- Snap counts: 4x/day
- FTN charting: Every 6 hours

### Data Notes
- Returns Polars DataFrames (convert with `.to_pandas()`)
- Uses `team` column (not `recent_team`)
- License: CC-BY-4.0 (FTN is CC-BY-SA-4.0)
- Pull Thursday AM UTC for corrected "clean" data

---

## Architecture

```
nflreadpy -> ingest_real_nfl_data.py -> player_stats_enhanced
                                              |
                                    weekly.py (train/predict)
                                              |
                                     weekly_projections
                                              |
Odds API -> prop_line_scraper.py -> weekly_odds
                                              |
                                prop_integration.py (3-tier match)
                                              |
                              value_betting_engine.py (Kelly + CLV)
                                              |
                           materialized_value_view.py (dashboard layer)
                                              |
                             api/server.py -> React Dashboard
```

---

## Data Status

| Season | Source | Notes |
|--------|--------|-------|
| 2024 | nflreadpy | Full season (weeks 1-18) |
| 2025 | nflreadpy | Through latest available week |

**Data Source**: All data ingested via `scripts/ingest_real_nfl_data.py` using nflverse/nflreadpy.

### Verify Data
```bash
DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python -c "
from utils.db import read_dataframe
print(read_dataframe('SELECT season, COUNT(*) as rows, COUNT(DISTINCT player_id) as players FROM player_stats_enhanced GROUP BY season'))
"
```

---

## Key Features

### Defense Adjustments
- Relative performance vs defense multipliers
- Applied during feature engineering in `data_pipeline.py`

### WR-Specific Enhancements
- EWMA with decay=0.65 for market mu computation
- Role-based cluster priors (alpha/secondary/slot/fringe)
- Blended weighting (55% hist, 30% targets, 15% role)

### Player Matching (3-Tier)
- Tier 1: player_id exact match
- Tier 2: name + team match
- Tier 3: name only match (WR team mismatch tolerance for trades)
- Implemented in `prop_integration.py`

### Dashboard Features
- "Best Line Only" toggle
- Multiple sportsbook comparison
- Value ranking by edge/CLV
- Real-time projection updates

---

## Testing

Run full test suite:
```bash
make test
```

Key test files:
- `tests/test_market_mu_wr.py` - EWMA and role priors
- `tests/test_prop_integration_wr.py` - 3-tier player matching
- `tests/test_projection_accuracy.py` - MAE validation
- `tests/test_value_betting.py` - Kelly criterion and edge calculation

---

## Notes

- Database migrations are managed by `schema_migrations.py`
- All proprietary logic is in .gitignored files
- Use `make fullstack` for complete local development environment
- Front-end dashboard is in `/frontend` (Next.js + TypeScript)
- Legacy Streamlit dashboard available via `make dashboard`
