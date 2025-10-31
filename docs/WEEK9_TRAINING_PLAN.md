# Week 9 Prediction Training Plan - Complete Context

## 🎯 OBJECTIVE
Collect data from weeks 1-8 of the current season (2025) plus historical data from previous year (2024) to train models and predict Week 9 of 2025 value betting opportunities.

---

## 📊 CURRENT DATA STATUS

### Available Data in Database:
- **2025 Season**: Weeks 1, 5 (partial - current season)
- **2024 Season**: Weeks 1-4 (partial - previous year)
- **2023 Season**: Weeks 4-17 (older data)

### Data Needed:
- **2025 Season Weeks 1-8**: Collect all weeks (missing: 2, 3, 4, 6, 7, 8)
- **2024 Season Full Season**: Use as historical training data (all weeks 1-18)
- **2025 Week 9 Predictions**: Generate after training on weeks 1-8 + 2024 data

---

## 🔧 SYSTEM ARCHITECTURE OVERVIEW

### Data Flow:
1. **Data Collection** → `update_week(season, week)` → Stores to `player_stats_enhanced`, `games`, `weather_data`, `injury_data`
2. **Feature Engineering** → `compute_week_features(season, week)` → Creates feature frames with rolling averages, matchups, etc.
3. **Model Training** → `train_weekly_models([(season, week), ...])` → Trains on historical weeks, saves models
4. **Predictions** → `predict_week(season, week)` → Uses trained models to predict Week 9
5. **Value Calculation** → `rank_weekly_value(season, week, min_edge)` → Compares predictions to odds

### Key Files:
- `data_pipeline.py`: Data collection and feature engineering
- `models/position_specific/weekly.py`: Model training and prediction
- `value_betting_engine.py`: Edge calculation and bet recommendations
- `prop_line_scraper.py`: Fetches live odds from The Odds API

### Key Database Tables:
- `player_stats_enhanced`: Historical player stats with engineered features
- `games`: Game information (teams, dates, venues)
- `weather_data`: Weather conditions for games
- `injury_data`: Player injury status
- `weekly_odds`: Sportsbook lines and odds
- `weekly_projections`: ML model predictions (`mu`, `sigma`)
- `materialized_value_view`: Calculated edges and ROI for dashboard

---

## 📋 STEP-BY-STEP PLAN

### PHASE 1: Collect Weeks 1-8 Data (Current Season 2025)

#### Step 1.1: Update Each Week Sequentially
```bash
# For each week 1-8 in 2025 (current season)
for week in 1 2 3 4 5 6 7 8; do
  make week-update SEASON=2025 WEEK=$week
done
```

**What `week-update` does:**
- Calls `update_week(season, week)` from `data_pipeline.py`
- `prepare_weekly_bundle()`:
  - Loads baseline projections (`2024_nfl_projections.csv` - note: file name is historical, but used for baseline player data)
  - Generates weekly games based on team matchups
  - Synthesizes player stats from baseline (scaled by week)
  - Creates team context (offensive/defensive rankings)
  - Synthesizes injuries and weather data
  - Fetches real odds from The Odds API (if available)
- `apply_weekly_bundle()`:
  - Persists all data to `nfl_data.db`
  - Updates `player_stats_enhanced`, `games`, `weather_data`, `injury_data`, `weekly_odds`

#### Step 1.2: Fetch Real Odds for Each Week (Optional but Recommended)
```bash
# Fetch odds for weeks 1-8 of 2025 (if still available from API)
for week in 1 2 3 4 5 6 7 8; do
  python run_prop_update.py --week $week --season 2025
done
```

**What `run_prop_update.py` does:**
- Calls `NFLPropScraper.get_upcoming_week_props(week, season)`
- Fetches odds from The Odds API for that week
- Saves to `nfl_prop_lines.db` → `prop_lines` table
- Integrates with existing projections via `PropIntegration`

#### Step 1.3: Verify Data Collection
```bash
# Check what weeks are in database for 2025
sqlite3 nfl_data.db "SELECT DISTINCT season, week FROM player_stats_enhanced WHERE season=2025 ORDER BY week;"

# Check player stats count
sqlite3 nfl_data.db "SELECT COUNT(*) FROM player_stats_enhanced WHERE season=2025;"
```

---

### PHASE 2: Prepare Previous Year Data (2024 Season - Full Historical Data)

#### Step 2.1: Identify Available 2024 Weeks
```bash
# Check what 2024 data exists
sqlite3 nfl_data.db "SELECT DISTINCT week FROM player_stats_enhanced WHERE season=2024 ORDER BY week;"
```

**Goal**: Collect all 2024 weeks (1-18) for comprehensive historical training data

#### Step 2.2: Collect All 2024 Weeks
```bash
# Collect all weeks from 2024 season (previous year)
for week in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18; do
  make week-update SEASON=2024 WEEK=$week
done
```

**Why 2024 Full Season Data Matters:**
- Provides large training set for models (18 weeks)
- Shows full season player performance patterns
- Includes playoff weeks (17-18) for late-season trends
- Helps identify breakout candidates and year-over-year trends
- Improves model accuracy for Week 9 predictions (mid-season patterns)

#### Step 2.3: Enhance 2024 Data with Real Stats (If Available)
If you have historical NFL stats APIs or data files:
- Load actual 2024 player stats (not synthetic)
- Update `player_stats_enhanced` table with real data
- This significantly improves model training quality
- More accurate historical data = better Week 9 predictions

---

### PHASE 3: Train Models on Collected Data

#### Step 3.1: Prepare Training Data List
The `train_weekly_models()` function requires a list of `(season, week)` tuples:

**For Week 9 prediction, use:**
- **2025 Weeks 1-8**: Current season data (most relevant - what we're trying to predict)
- **2024 Weeks 1-18**: Previous year full season data (historical patterns)
- **Optional**: 2024 Week 9 specifically (same week from previous year for context)

#### Step 3.2: Train Models
```python
from models.position_specific.weekly import train_weekly_models

# Current season weeks (2025) - weeks 1-8
current_season_weeks = [(2025, 1), (2025, 2), (2025, 3), (2025, 4), 
                        (2025, 5), (2025, 6), (2025, 7), (2025, 8)]

# Previous year full season (2024) - all 18 weeks
previous_year_weeks = [(2024, w) for w in range(1, 19)]  # Weeks 1-18

# Combine training data
training_weeks = current_season_weeks + previous_year_weeks

print(f"Training on {len(training_weeks)} weeks:")
print(f"  - 2025 weeks 1-8: {len(current_season_weeks)} weeks")
print(f"  - 2024 full season: {len(previous_year_weeks)} weeks")

# Train models
trained_models = train_weekly_models(training_weeks)
```

**What `train_weekly_models()` does:**
1. For each `(season, week)` tuple:
   - Calls `update_week()` to ensure data exists
   - Calls `compute_week_features()` to create feature frames
   - Groups features by market (rushing_yards, receiving_yards, passing_yards)

2. For each market:
   - Concatenates all feature frames
   - Trains Ridge Regression model:
     - Features: `line`, `targets`, `rolling_targets`, `offensive_rank`, `defensive_rank`, `temperature`, `wind_speed`, `injury_indicator`, `weather_penalty`
     - Target: `mu_prior` (baseline prediction from historical stats)
   - Calculates `sigma_default` (uncertainty estimate)
   - Saves model to `models/weekly/{market}_model.joblib`

**Model Files Created:**
- `models/weekly/rushing_yards_model.joblib`
- `models/weekly/receiving_yards_model.joblib`
- `models/weekly/passing_yards_model.joblib`

---

### PHASE 4: Generate Week 9 Predictions (2025 Season)

#### Step 4.1: Prepare Week 9 Data
```bash
# Update Week 9 data for 2025 (creates games, player stats, weather, injuries)
make week-update SEASON=2025 WEEK=9
```

#### Step 4.2: Fetch Week 9 Odds
```bash
# Get current Week 9 odds from sportsbooks for 2025
python run_prop_update.py --week 9 --season 2025
```

#### Step 4.3: Generate Week 9 Predictions
```bash
# Generate predictions using trained models
make week-predict SEASON=2025 WEEK=9
```

**What `predict_week()` does:**
1. Calls `update_week(2025, 9)` to ensure data exists
2. Calls `compute_week_features(2025, 9)` to create feature frames
3. For each market:
   - Loads trained model from `models/weekly/{market}_model.joblib`
   - If model exists: Uses ML model to predict `mu` (mean projection)
   - If no model: Uses `mu_prior` (baseline from historical stats)
   - Calculates `sigma` (uncertainty) = `max(sigma_default, |mu| * 0.3)`
4. Persists predictions to `weekly_projections` table

**Prediction Output:**
- `mu`: Model's predicted mean (e.g., 112.5 receiving yards)
- `sigma`: Uncertainty/standard deviation (e.g., 7.5 yards)
- `model_version`: Which model was used
- Stored in `weekly_projections` table

---

### PHASE 5: Calculate Value Bets for Week 9 (2025)

#### Step 5.1: Materialize Value View
```bash
# Calculate edges, ROI, Kelly stakes for Week 9 of 2025
make week-materialize SEASON=2025 WEEK=9
```

**What `materialize_week()` does:**
1. Calls `rank_weekly_value(2025, 9, min_edge=0.05)`:
   - Joins `weekly_projections` (predictions) with `prop_lines` (odds)
   - For each player/market/odds combination:
     - Calculates `p_win`: Probability of beating the line
       - Formula: Uses normal distribution with `mu` and `sigma` vs `line`
     - Calculates `edge_percentage`: `p_win - implied_probability_from_odds`
     - Calculates `expected_roi`: `(decimal_odds * p_win) - 1`
     - Calculates `kelly_fraction`: Fractional Kelly Criterion stake size
     - Calculates `stake`: Recommended bet size in units
     - Sets `recommendation`: 'BET' if `edge >= min_edge`, else 'PASS'
2. Persists to `materialized_value_view` table

#### Step 5.2: View Results in Dashboard
```bash
# Launch dashboard
make dashboard
```

**In Dashboard:**
1. Select Season: **2025** (current season)
2. Select Week: **9**
3. Set Minimum Edge: **5.0%** (or desired threshold)
4. Click **"Refresh Data"** button
5. View value bets in table

---

## 🔍 KEY FUNCTIONS EXPLAINED

### `update_week(season, week)` - Data Collection
**Location**: `data_pipeline.py` line 1300

**Purpose**: Collects and stores all data needed for a specific week

**Process:**
1. `prepare_weekly_bundle(season, week)`:
   - Loads baseline projections from `2024_nfl_projections.csv`
   - Generates game matchups (home/away teams)
   - Synthesizes player stats (from baseline, scaled by week factor)
   - Creates team context (offensive/defensive rankings)
   - Synthesizes injuries and weather
   - Fetches real odds from API (if available)

2. `apply_weekly_bundle(bundle)`:
   - Upserts data to database tables
   - Stores: `games`, `player_stats_enhanced`, `team_context`, `injury_data`, `weather_data`, `weekly_odds`

**Database Tables Updated:**
- `games`: Game information
- `player_stats_enhanced`: Player statistics with engineered features
- `team_context`: Team rankings and metrics
- `injury_data`: Player injury status
- `weather_data`: Weather conditions
- `weekly_odds`: Sportsbook lines

---

### `compute_week_features(season, week)` - Feature Engineering
**Location**: `data_pipeline.py` line 913

**Purpose**: Creates modeling features from collected data

**Process:**
1. Loads data from database:
   - `player_stats_enhanced`: Stats, rolling averages
   - `games`: Matchups
   - `weather_data`: Conditions
   - `injury_data`: Player status
   - `weekly_odds`: Sportsbook lines

2. For each player/market combination:
   - Creates feature row with:
     - **Baseline stats**: `targets`, `rushing_yards`, `receiving_yards`
     - **Rolling averages**: `rolling_targets`, `rolling_routes`, `rolling_air_yards`
     - **Team context**: `offensive_rank`, `defensive_rank`, `pace_rank`
     - **Matchup factors**: `oline_rank`, `pass_rush_rank`
     - **Weather**: `temperature`, `wind_speed`, `is_dome`, `weather_penalty`
     - **Injuries**: `injury_indicator`, `injury_status`
     - **Odds**: `line`, `price` (from sportsbooks)
     - **Prior**: `mu_prior` (baseline prediction from stats)

3. Returns DataFrame with features for model training/prediction

**Feature Columns** (BASE_FEATURE_COLUMNS):
- `line`: Sportsbook line
- `price`: American odds
- `mu_prior`: Baseline prediction
- `snap_percentage`: Snap share
- `targets`: Targets in that week
- `rolling_targets`: 3-week rolling average
- `rolling_routes`: Routes run average
- `rolling_air_yards`: Air yards average
- `usage_delta`: Change in usage
- `breakout_percentile`: Breakout potential
- `offensive_rank`: Team offensive ranking
- `defensive_rank`: Opponent defensive ranking
- `pace_rank`: Team pace ranking
- `red_zone_efficiency`: Red zone conversion rate
- `oline_rank`: Offensive line ranking
- `pass_rush_rank`: Pass rush ranking
- `temperature`: Game temperature
- `wind_speed`: Wind conditions
- `injury_indicator`: 0=active, 1=injured
- `weather_penalty`: Bad weather flag

---

### `train_weekly_models(season_weeks)` - Model Training
**Location**: `models/position_specific/weekly.py` line 34

**Purpose**: Trains ML models on historical weeks

**Process:**
1. For each `(season, week)` in training list:
   - Ensures data exists: `update_week(season, week)`
   - Computes features: `compute_week_features(season, week)`
   - Groups by market (rushing, receiving, passing)

2. For each market:
   - Concatenates all feature frames
   - Trains Ridge Regression:
     - Features: All columns in BASE_FEATURE_COLUMNS
     - Target: `mu_prior` (baseline from historical stats)
   - Calculates `sigma_default` (residual standard deviation)
   - Saves model artifact to `models/weekly/{market}_model.joblib`

**Model Artifact Contains:**
- `model`: Trained scikit-learn Pipeline
- `feature_columns`: Which features were used
- `market`: Market name (rushing_yards, receiving_yards, passing_yards)
- `model_version`: Timestamp
- `sigma_default`: Uncertainty estimate

---

### `predict_week(season, week)` - Generate Predictions
**Location**: `models/position_specific/weekly.py` line 95

**Purpose**: Uses trained models to predict Week 9

**Process:**
1. Ensures data exists: `update_week(season, week)`
2. Computes features: `compute_week_features(season, week)`
3. For each market:
   - Loads trained model from `models/weekly/{market}_model.joblib`
   - If model exists:
     - Uses ML model to predict `mu` (mean projection)
     - Uses `sigma_default` from model for uncertainty
   - If no model:
     - Uses `mu_prior` (baseline from stats)
     - Uses default `sigma = 7.5`
4. Persists to `weekly_projections` table

**Prediction Columns:**
- `season`: 2024
- `week`: 9
- `player_id`: Player identifier
- `market`: rushing_yards, receiving_yards, passing_yards
- `mu`: Model's predicted mean (e.g., 112.5 yards)
- `sigma`: Uncertainty/standard deviation (e.g., 7.5 yards)
- `model_version`: Model version used
- `generated_at`: Timestamp

---

### `rank_weekly_value(season, week, min_edge)` - Calculate Edges
**Location**: `value_betting_engine.py` line 99

**Purpose**: Compares predictions to odds and calculates betting edges

**Process:**
1. Joins `weekly_projections` (predictions) with `prop_lines` (odds):
   - Uses `prop_integration.join_odds_projections(season, week)`
   - Matches by: `player_id`, `market`

2. For each player/market/odds combination:
   - Calculates `p_win`: Probability of beating the line
     - Uses normal distribution: `P(actual > line | mu, sigma)`
   - Calculates `implied_prob`: From odds (e.g., -110 = 52.38%)
   - Calculates `edge_percentage`: `p_win - implied_prob`
   - Calculates `expected_roi`: `(decimal_odds * p_win) - 1`
   - Calculates `kelly_fraction`: `(b * p - q) / b` where:
     - `b` = decimal_odds - 1
     - `p` = win probability
     - `q` = 1 - p
   - Calculates `stake`: `kelly_fraction * bankroll`
   - Sets `recommendation`: 'BET' if `edge >= min_edge`, else 'PASS'

3. Returns DataFrame sorted by edge (highest first)

**Output Columns:**
- `player_id`, `player_name`, `team`, `opponent`, `position`
- `market`: rushing_yards, receiving_yards, passing_yards
- `sportsbook`: Caesars, FanDuel, BetMGM, DraftKings
- `line`: Sportsbook line (e.g., 71.5 yards)
- `price`: American odds (e.g., -110)
- `mu`: Model prediction (e.g., 112.5 yards)
- `sigma`: Uncertainty (e.g., 7.5 yards)
- `p_win`: Win probability (e.g., 0.91 = 91%)
- `edge_percentage`: Edge over implied probability (e.g., 0.37 = 37%)
- `expected_roi`: Expected return (e.g., 0.66 = 66%)
- `kelly_fraction`: Kelly stake percentage
- `stake`: Recommended bet size in units
- `recommendation`: 'BET' or 'PASS'

---

## 📊 COMPLETE EXECUTION PLAN

### Script 1: Collect Weeks 1-8 Data (2025 Season)
```python
#!/usr/bin/env python3
"""Collect weeks 1-8 data for 2025 season (current season)"""
from data_pipeline import update_week

for week in range(1, 9):
    print(f"Updating Week {week} of 2025...")
    update_week(2025, week)
    print(f"✅ Week {week} complete")
```

### Script 2: Collect Full 2024 Season Data (Historical)
```python
#!/usr/bin/env python3
"""Collect all weeks from 2024 season for historical training data"""
from data_pipeline import update_week

for week in range(1, 19):  # Weeks 1-18 (full season including playoffs)
    print(f"Updating Week {week} of 2024...")
    update_week(2024, week)
    print(f"✅ Week {week} complete")
```

### Script 3: Fetch Odds for Weeks 1-8 (2025)
```python
#!/usr/bin/env python3
"""Fetch odds for weeks 1-8 of 2025 season"""
import subprocess

for week in range(1, 9):
    print(f"Fetching odds for Week {week} of 2025...")
    subprocess.run(["python", "run_prop_update.py", "--week", str(week), "--season", "2025"])
    print(f"✅ Week {week} odds fetched")
```

### Script 4: Train Models on Collected Data
```python
#!/usr/bin/env python3
"""Train models on 2025 weeks 1-8 + 2024 full season data"""
from models.position_specific.weekly import train_weekly_models

# Current season weeks (2025) - weeks 1-8
current_weeks = [(2025, w) for w in range(1, 9)]

# Previous year full season (2024) - weeks 1-18
previous_year_weeks = [(2024, w) for w in range(1, 19)]

# Combine training data
training_weeks = current_weeks + previous_year_weeks

print(f"Training on {len(training_weeks)} weeks:")
print(f"  - 2025 weeks 1-8: {len(current_weeks)} weeks")
print(f"  - 2024 full season: {len(previous_year_weeks)} weeks")

trained_models = train_weekly_models(training_weeks)

print("✅ Models trained:")
for market, path in trained_models.items():
    print(f"  - {market}: {path}")
```

### Script 5: Generate Week 9 Predictions (2025)
```python
#!/usr/bin/env python3
"""Generate Week 9 predictions for 2025 season"""
from models.position_specific.weekly import predict_week
from data_pipeline import update_week

# Ensure Week 9 data exists
print("Updating Week 9 data for 2025...")
update_week(2025, 9)

# Generate predictions
print("Generating Week 9 predictions...")
projections = predict_week(2025, 9)

print(f"✅ Generated {len(projections)} projections")
print(f"Markets: {projections['market'].unique().tolist()}")
```

### Script 6: Calculate Week 9 Value Bets (2025)
```python
#!/usr/bin/env python3
"""Calculate Week 9 value bets for 2025 season"""
from materialized_value_view import materialize_week
import pandas as pd

# Materialize value view
print("Materializing Week 9 value view for 2025...")
value_bets = materialize_week(2025, 9, min_edge=0.05)

# Filter for BET recommendations
bet_opportunities = value_bets[value_bets['recommendation'] == 'BET']

print(f"✅ Found {len(bet_opportunities)} value bets")
print(f"Average edge: {bet_opportunities['edge_percentage'].mean()*100:.2f}%")
print(f"Average ROI: {bet_opportunities['expected_roi'].mean()*100:.2f}%")

# Show top opportunities
top = bet_opportunities.nlargest(10, 'edge_percentage')
print("\nTop 10 Value Opportunities:")
for _, row in top.iterrows():
    print(f"  {row['player_name']} ({row['position']}): {row['market']} | "
          f"Line: {row['line']:.1f} | Model: {row['mu']:.1f} | "
          f"Edge: {row['edge_percentage']*100:.1f}% | ROI: {row['expected_roi']*100:.1f}%")
```

---

## 🗂️ DATABASE SCHEMA REFERENCE

### `player_stats_enhanced` Table
**Purpose**: Stores player statistics with engineered features

**Key Columns:**
- `player_id`, `name`, `team`, `position`, `season`, `week`
- `rushing_yards`, `rushing_attempts`
- `receiving_yards`, `receptions`, `targets`
- `snap_percentage`, `snap_count`
- `rolling_targets`, `rolling_routes`, `rolling_air_yards` (3-week averages)
- `usage_delta`, `breakout_percentile`
- `age`, `games_played`
- `game_id`, `opponent`

### `weekly_projections` Table
**Purpose**: Stores ML model predictions

**Key Columns:**
- `season`, `week`, `player_id`, `team`, `opponent`, `position`
- `market`: rushing_yards, receiving_yards, passing_yards
- `mu`: Predicted mean (e.g., 112.5 yards)
- `sigma`: Uncertainty/standard deviation (e.g., 7.5 yards)
- `model_version`: Model identifier
- `generated_at`: Timestamp

### `materialized_value_view` Table
**Purpose**: Stores calculated betting edges and recommendations

**Key Columns:**
- `season`, `week`, `player_id`, `event_id`, `market`, `sportsbook`
- `line`: Sportsbook line
- `price`: American odds
- `mu`: Model prediction
- `sigma`: Uncertainty
- `p_win`: Win probability
- `edge_percentage`: Edge over implied probability
- `expected_roi`: Expected return on investment
- `kelly_fraction`: Kelly Criterion stake percentage
- `stake`: Recommended bet size in units
- `generated_at`: Timestamp

---

## ✅ VERIFICATION CHECKLIST

### After Phase 1 (Collect 2024 Full Season):
- [ ] All 18 weeks exist in `player_stats_enhanced` table for 2024
- [ ] Games table has Week 1-18 games for 2024
- [ ] Weather and injury data populated for 2024

### After Phase 2 (Collect 2025 Weeks 1-8):
- [ ] All 8 weeks exist in `player_stats_enhanced` table for 2025
- [ ] Games table has Week 1-8 games for 2025
- [ ] Weather and injury data populated for 2025
- [ ] Odds data fetched for 2025 weeks 1-8 (if available)

### After Phase 3 (Train Models):
- [ ] Model files exist: `models/weekly/rushing_yards_model.joblib`, etc.
- [ ] Models were trained on multiple weeks
- [ ] Model version recorded

### After Phase 5 (Week 9 Predictions - 2025):
- [ ] Week 9 data exists in `player_stats_enhanced` for 2025
- [ ] Week 9 odds fetched from API for 2025
- [ ] Predictions exist in `weekly_projections` table for 2025 Week 9
- [ ] Predictions have `mu` and `sigma` values

### After Phase 7 (Value Bets - 2025 Week 9):
- [ ] Value view materialized in `materialized_value_view` table
- [ ] Bets have `edge_percentage` and `expected_roi` calculated
- [ ] Dashboard shows 2025 Week 9 value bets

---

## 🚀 QUICK START COMMANDS

```bash
# 1. Collect 2024 full season data (historical training data)
for week in {1..18}; do
  make week-update SEASON=2024 WEEK=$week
done

# 2. Collect 2025 weeks 1-8 data (current season)
for week in {1..8}; do
  make week-update SEASON=2025 WEEK=$week
done

# 3. Fetch odds for 2025 weeks 1-8
for week in {1..8}; do
  python run_prop_update.py --week $week --season 2025
done

# 4. Train models on 2025 weeks 1-8 + 2024 full season
python scripts/train_weekly_for_week9.py

# 5. Generate Week 9 predictions (2025)
make week-predict SEASON=2025 WEEK=9

# 6. Fetch Week 9 odds (2025)
python run_prop_update.py --week 9 --season 2025

# 7. Materialize value view for Week 9 (2025)
make week-materialize SEASON=2025 WEEK=9

# 8. Launch dashboard
make dashboard
# Then select Season=2025, Week=9, click "Refresh Data"
```

---

## 📝 PYTHON SCRIPT TEMPLATE

```python
#!/usr/bin/env python3
"""
Complete Week 9 Training and Prediction Script
Collects weeks 1-8, trains models, predicts Week 9
"""
from data_pipeline import update_week, compute_week_features
from models.position_specific.weekly import train_weekly_models, predict_week
from materialized_value_view import materialize_week
from prop_line_scraper import NFLPropScraper
import pandas as pd

def main():
    # PHASE 1: Collect 2024 Full Season (Historical Data)
    print("📊 PHASE 1: Collecting 2024 full season data (historical)...")
    for week in range(1, 19):  # Weeks 1-18
        print(f"  2024 Week {week}...")
        update_week(2024, week)
    print("✅ 2024 full season collected")
    
    # PHASE 2: Collect 2025 Weeks 1-8 (Current Season)
    print("\n📊 PHASE 2: Collecting 2025 weeks 1-8 data (current season)...")
    for week in range(1, 9):
        print(f"  2025 Week {week}...")
        update_week(2025, week)
    print("✅ 2025 weeks 1-8 collected")
    
    # PHASE 3: Fetch Odds for 2025 Weeks 1-8
    print("\n📊 PHASE 3: Fetching odds for 2025 weeks 1-8...")
    scraper = NFLPropScraper()
    for week in range(1, 9):
        print(f"  2025 Week {week} odds...")
        try:
            scraper.get_upcoming_week_props(week, 2025)
        except:
            print(f"    ⚠️  Week {week} odds unavailable")
    
    # PHASE 4: Train Models
    print("\n🤖 PHASE 4: Training models...")
    current_weeks = [(2025, w) for w in range(1, 9)]  # 2025 weeks 1-8
    previous_year_weeks = [(2024, w) for w in range(1, 19)]  # 2024 full season
    training_weeks = current_weeks + previous_year_weeks
    
    print(f"  Training on {len(training_weeks)} weeks:")
    print(f"    - 2025 weeks 1-8: {len(current_weeks)} weeks")
    print(f"    - 2024 full season: {len(previous_year_weeks)} weeks")
    
    trained_models = train_weekly_models(training_weeks)
    print(f"✅ Models trained: {list(trained_models.keys())}")
    
    # PHASE 5: Predict Week 9 (2025)
    print("\n🔮 PHASE 5: Generating Week 9 predictions for 2025...")
    update_week(2025, 9)  # Ensure Week 9 data exists
    projections = predict_week(2025, 9)
    print(f"✅ Generated {len(projections)} projections")
    
    # PHASE 6: Fetch Week 9 Odds (2025)
    print("\n💰 PHASE 6: Fetching Week 9 odds for 2025...")
    scraper.get_upcoming_week_props(9, 2025)
    
    # PHASE 7: Calculate Value Bets
    print("\n🎯 PHASE 7: Calculating Week 9 value bets for 2025...")
    value_bets = materialize_week(2025, 9, min_edge=0.05)
    bet_opportunities = value_bets[value_bets['recommendation'] == 'BET']
    
    print(f"\n✅ COMPLETE!")
    print(f"   Found {len(bet_opportunities)} value betting opportunities")
    print(f"   Average edge: {bet_opportunities['edge_percentage'].mean()*100:.2f}%")
    print(f"   Average ROI: {bet_opportunities['expected_roi'].mean()*100:.2f}%")
    
    # Show top 5
    top = bet_opportunities.nlargest(5, 'edge_percentage')
    print("\n📊 Top 5 Value Opportunities:")
    for _, row in top.iterrows():
        print(f"   {row['player_name']} ({row['position']}): {row['market']} | "
              f"Edge: {row['edge_percentage']*100:.1f}% | ROI: {row['expected_roi']*100:.1f}%")

if __name__ == "__main__":
    main()
```

---

## 🎯 EXPECTED OUTCOMES

After completing all phases:
1. **Data Collected**: 
   - 2024 full season (weeks 1-18) - historical training data
   - 2025 weeks 1-8 - current season training data
2. **Models Trained**: 3 models (rushing, receiving, passing yards)
   - Trained on 26 total weeks (8 from 2025 + 18 from 2024)
3. **Week 9 Predictions**: ML-based predictions for 2025 Week 9
4. **Value Bets Identified**: Opportunities with 5%+ edge for 2025 Week 9
5. **Dashboard Ready**: 2025 Week 9 value bets visible in dashboard

---

## 📝 KEY CORRECTIONS

**Important**: The plan uses:
- **2025 Season**: Current season (weeks 1-8 to train, week 9 to predict)
- **2024 Season**: Previous year full season (weeks 1-18) as historical training data

This provides:
- **26 weeks of training data** (8 current + 18 historical)
- **Current season context** from 2025 weeks 1-8
- **Full season patterns** from 2024
- **Week 9 predictions** for 2025 season

---

**This document provides complete context for collecting 2025 weeks 1-8 data and 2024 full season data to predict 2025 Week 9. Share with LLM to execute the plan.**


