# Claude Session Notes

## Session: November 25, 2025 (Updated)

### Summary
1. **Data Source**: Switched to `nflreadpy` (nflverse) - has real-time 2025 data
2. **Historical Data**: Loaded 2024 + 2025 seasons (9,896 player-week rows)
3. **Defense Adjustments**: Added relative performance vs defense multipliers
4. **Dashboard**: Added "Best Line Only" toggle, fixed column order
5. **Opponent Data**: Fixed incorrect matchups (PHI vs CHI, etc.)

---

## Key Accomplishments

### 1. Data Source Migration
- **Problem**: `nfl_data_py` was returning 2024 season data (last year), not current 2025 season
- **Solution**: Switched to `nflreadpy` which has up-to-date 2025 data through Week 12
- **Installed**: `uv add nflreadpy` (uses Polars DataFrames, faster than pandas)

### 2. Updated Ingestion Script
**File**: `scripts/ingest_real_nfl_data.py`

Changes made:
- Replaced `nfl_data_py` with `nflreadpy` 
- Updated `fetch_weekly_stats()` to use `nfl.load_player_stats()` with Polars-to-pandas conversion
- Fixed column name differences (`team` vs `recent_team`)
- Made script SQLite-compatible (was MySQL-only before)
- Fixed deprecated `datetime.utcnow()` to `datetime.now(timezone.utc)`

### 3. Database Configuration
- Per `AGENTS.md`: Local SQLite databases (`nfl_data.db`, `nfl_prop_lines.db`) are dev caches
- Run with: `DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db`
- Remote MySQL (Kinsta) was timing out; use local SQLite for development

### 4. Data Verification
Week 12 2025 data now matches real stats:
- Justin Jefferson: 4 rec, 48 yds, 6 targets (MIN vs GB)
- Christian Watson: 5 rec, 49 yds, 7 targets (GB vs MIN)

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

## Commands

### Ingest Real Data
```bash
# Ingest BOTH 2024 and 2025 seasons (default)
DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python scripts/ingest_real_nfl_data.py

# Or use Makefile target:
make ingest-nfl

# Single season only:
DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python scripts/ingest_real_nfl_data.py --season 2025 --through-week 12
```

### Verify Data
```bash
DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python -c "
from utils.db import read_dataframe
print(read_dataframe('SELECT season, COUNT(*) as rows, COUNT(DISTINCT player_id) as players FROM player_stats_enhanced GROUP BY season'))
"
```

---

## Current Data Status (as of Nov 25, 2025)

| Season | Rows  | Players | Weeks    |
|--------|-------|---------|----------|
| 2024   | 5,920 | 618     | 1-18     |
| 2025   | 3,976 | 587     | 1-12     |

**Total**: 9,896 player-week rows for training/prediction

**Note**: Old stale data was cleaned up. Only fresh nflreadpy data remains in `player_stats_enhanced`.

---

## Issues 3 & 4 Status (from previous session)

Both completed and tested:

### Issue 3: Enhanced `_compute_market_mu` for WR
- Added EWMA with decay=0.65
- Role-based cluster priors (alpha/secondary/slot/fringe)
- Blended weighting (55% hist, 30% targets, 15% role)
- Tests in `tests/test_market_mu_wr.py`

### Issue 4: WR-Resilient Matching
- 3-tier matching: player_id to name+team to name only
- WR team mismatch tolerance for trades
- `IntegrationConfig` in `config.py`
- Tests in `tests/test_prop_integration_wr.py`

---

## Next Steps
1. Generate Week 13 projections using 2025 Week 1-12 data
2. Fetch/synthesize Week 13 odds
3. Run `make week-materialize SEASON=2025 WEEK=13`
4. Launch dashboard to verify predictions

---

## Files Modified This Session
- `scripts/ingest_real_nfl_data.py` - Switched to nflreadpy, SQLite support, **default to 2024+2025**
- `pyproject.toml` - Added nflreadpy dependency
- `Makefile` - Added `ingest-nfl` target

## Files Created
- `claude.md` - This file

---

## Quick Reference

```bash
# Ingest both seasons (recommended)
make ingest-nfl

# Generate Week 13 projections
make week-predict SEASON=2025 WEEK=13

# Materialize for dashboard
make week-materialize SEASON=2025 WEEK=13

# Launch dashboard
make dashboard
```
