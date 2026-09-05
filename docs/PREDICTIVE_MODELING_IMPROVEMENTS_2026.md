# Predictive Modeling & Feature Engineering Improvements (2026)

This document provides a comprehensive technical overview of the predictive modeling and feature engineering improvements implemented under the plan `docs/plans/2026-09-02-001-feat-predictive-modeling-feature-engineering-plan.md`.

---

## Executive Summary

Prior to this work, the NFL betting algorithm suffered from systematic projection biases and gaps in market coverage:
1. **Unconsumed Schedule Context**: Schedule data (point spreads, implied team totals, weather conditions, indoor dome indicators, and divisional game flags) was persisted in the database but unused by the predictive models.
2. **Hardcoded Neutral Game Script**: `game_script` and `expected_game_script` were hardcoded to `0.0` across all historical stats and pregame snapshots, blinding the models to game-flow pacing and lead/trailing effects.
3. **QB Passing Volume Over-Projection**: Quarterbacks exhibited a headline **+13.2 yard** passing bias decomposing into:
   - **Slate breadth**: Backup QBs projected as full starters when given historical volume (+18 to +31 yard bias on backups).
   - **Starter baseline hot**: Starting QBs ran hot at 33.2 predicted pass attempts vs. 30.9 actual attempts (+8.3 yard bias), aggravated by an inverted game-script multiplier.
4. **Unmodeled Prop Markets**: High-liquidity prop betting markets—specifically **player receptions** and **anytime touchdowns**—were not modeled, priced, or graded.
5. **Stale Receiver Role Priors**: Early-season WR role priors were hardcoded at uncalibrated levels (75/55/45/30 yards), overestimating empirical production by 30% to 200%.

All five areas have been addressed with end-to-end integration across data ingestion, feature engineering, predictive models, value engines, and grading pipelines.

---

## Detailed Improvement Breakdown

### 1. Schedule Game Context Features (Unit 1)

#### Problem
The `games` table captured rich pregame schedule data via `utils/game_context.py`, but the predictive models in `models/position_specific/weekly.py` did not consume any of these columns. Projections were unable to adjust for Vegas game script expectations, projected shootouts, extreme weather, or indoor dome environments.

#### Solution
- Implemented `attach_game_context_to_player_frame()` in `utils/game_context.py` to join schedule context with player frames for any target week.
- Defined 7 canonical context features:
  - `spread_margin`: Point spread margin from the player's team perspective (positive = favored, negative = underdog).
  - `implied_team_total`: Expected team points derived from game total and spread \((T/2 \pm S/2)\).
  - `game_total`: Over/under line for the game.
  - `wind_speed`: Wind speed in mph (imputed to 0.0 for indoor/dome games).
  - `temperature`: Temperature in Fahrenheit (imputed to 70.0 for indoor/dome games).
  - `is_indoor`: Binary flag (1 if dome or closed retractable roof; 0 otherwise).
  - `div_game`: Binary divisional matchup flag (1 if divisional rivalry; 0 otherwise).
- Added robust `_safe_float` coercion to handle `None`, `NaN`, string representations (`"none"`, `"nan"`), or corrupt weather data without throwing runtime exceptions. Players on bye weeks cleanly receive `GAME_CONTEXT_DEFAULTS`.
- Integrated `_GAME_CONTEXT_COLS` into `get_nfl_feature_cols()` across all markets. In `_engineer_rolling_features()`, these features are treated as static pregame features (bypassing rolling EWM calculations) to preserve true causal bounds without lookahead leakage.

#### Files Modified
- `utils/game_context.py`
- `models/position_specific/weekly.py`
- `tests/test_game_context.py`
- `tests/test_nfl_weekly_model.py`

---

### 2. Empirical Game Script Ingestion (Unit 2)

#### Problem
Historical player games in `player_stats_enhanced` and pregame player snapshots in `nfl_player_context_snapshots` had `game_script = 0.0` and `expected_game_script = 0.0` hardcoded. Consequently, models could never learn how actual game flow altered volume distribution between the pass and run.

#### Solution
- **Actual Game Script**: Implemented `_compute_actual_game_script()` in `scripts/ingest_real_nfl_data.py`.
  - Primary source: Average `score_differential` from play-by-play (PBP) data across all plays the player's team ran.
  - Secondary fallback: When PBP is unavailable or insufficient, derives game script directly from schedule final score margins:
    \[
    \text{game\_script}_{\text{home}} = \frac{\text{home\_score} - \text{away\_score}}{2.0}, \quad
    \text{game\_script}_{\text{away}} = -\text{game\_script}_{\text{home}}
    \]
  - Integrated into `transform_to_enhanced_stats()` for continuous pipeline ingestion.
- **Expected Game Script**: Updated `build_player_context_snapshots()` to take a `schedule` DataFrame. Derived expected game script from the closing Vegas `spread_line`:
  \[
  \text{expected\_script}_{\text{favored}} = +\frac{|\text{spread}|}{2.0}, \quad
  \text{expected\_script}_{\text{underdog}} = -\frac{|\text{spread}|}{2.0}
  \]
- **Backfill Utility**: Added `backfill_game_script_and_expected_script()` to safely update historical records in `player_stats_enhanced` and `nfl_player_context_snapshots` from stored `games` scores and spreads.

#### Files Modified
- `scripts/ingest_real_nfl_data.py`
- `tests/test_ingest_game_script.py`

---

### 3. QB Passing Volume Calibration & Starter Gating (Unit 3)

#### Problem
A walk-forward backtest analysis revealed an overall **+13.2 yard** bias in quarterback passing yards projections. This had two distinct root causes:
1. **Inverted Sign Convention & Inflated Baseline**: Baseline starter attempts were assumed to be 34.0 (actual clear-starter volume is 30.9). Furthermore, the game script multiplier was inverted (`1.0 + game_script * 0.05`), erroneously projecting *more* passes when leading.
2. **Backup Projection Pollution**: Backup QBs who had started in previous weeks (or games) inherited full starter attempt baselines and projected for 200+ yards despite holding `depth_rank = 2` or `3`, skewing backtest metrics and card generation.

#### Solution
- **Baseline Calibration**: Reset `QB_BASELINE_ATTEMPTS = 31.0` in `data_pipeline.py`.
- **Canonical Script Factor Convention**: Corrected the script factor formula so that positive game script (leading) suppresses pass volume:
  \[
  \text{script\_factor} = 1.0 - (\text{game\_script} \times 0.04), \quad \text{clamped to } [0.75, 1.25]
  \]
- **Starter Probability Gating (\(p_{\text{start}}\))**:
  - Computed starter probability in `models/position_specific/weekly.py:_build_roster_week_data()` based on depth chart rank and injury report status:
    - Starter (`depth_rank == 1`):
      - Healthy / Probable: \(p = 1.00\)
      - Questionable: \(p = 0.70\)
      - Doubtful: \(p = 0.25\)
      - Out: \(p = 0.00\)
    - Backup (`depth_rank > 1`):
      - If starter is Out / Doubtful: inherits elevated starter likelihood.
      - Default backup: \(p = 0.02\)
  - Expected passing attempts are discounted: \(\text{expected\_attempts} = \text{base\_attempts} \times p_{\text{start}}\).
  - Fixed an issue in `_prefer_richer_role_estimate` where historical starter volume was overriding snapshot backup discounts.
  - Backups projecting \(\le 12.0\) attempts now fail `_eligible_role_mask` and are excluded from passing yards line generation.

#### Files Modified
- `data_pipeline.py`
- `models/position_specific/weekly.py`
- `tests/test_qb_decomposition.py`
- `tests/test_qb_gating.py`

---

### 4. Receptions & Anytime Touchdown Models (Unit 4)

#### Problem
High-volume betting markets for player receptions and anytime touchdowns could not be traded:
- `receptions` was registered in some areas but lacked trained predictive models, sigma calibration, and pipeline integration.
- `anytime_touchdown` was unmodeled, and binary/count props were at risk of being priced using Gaussian normal distributions, which severely distort near-zero tail probabilities.

#### Solution
- **Market Registration**: Registered `receptions` (unit: receptions, positions: WR/TE/RB) and `anytime_touchdown` (unit: touchdowns, positions: RB/WR/TE/QB, stat column: `anytime_td`) in `sports/markets.py` and `sports/nfl.py`.
- **Volume Floors & Sigma Defaults**:
  - `MARKET_MIN_EXPECTED_VOLUME`: 1.5 receptions; 0.5 red zone touches for anytime touchdown.
  - Added sigma floors and defaults in `utils/nfl_sigma.py` (e.g., reception sigma floor 1.4, default 2.2; anytime TD floor 0.35, default 0.48).
- **Stacking Regressor Models**:
  - Added `receptions` and `anytime_touchdown` configurations to `MARKET_CONFIGS` and `_MARKET_STATS` in `models/position_specific/weekly.py`.
  - Synthesized `anytime_td` on-the-fly from `rushing_tds` and `receiving_tds`
    via the shared `utils.nfl_markets.synthesize_anytime_td` helper:
    \[
    \text{anytime\_td} = \text{rushing\_tds} + \text{receiving\_tds}
    \]
    (integer count — the expectation \(\mu\) this trains is the Poisson rate
    the pricing engine consumes, not a binary flag).
- **Poisson Pricing Engine**:
  - In `value_betting_engine.py:prob_over()`, binary anytime touchdown props (line == 0.5) are priced using a **Poisson survival function** rather than a Gaussian CDF:
    \[
    P(\text{Over } 0.5) = 1.0 - e^{-\mu}
    \]
    where \(\mu\) represents the projected expected touchdowns. Continuous markets (yards, receptions) continue using Gaussian CDFs with player-specific \(\sigma\).
- **Outcome Grading**:
  - Updated `utils/nfl_markets.py:melt_actuals` and `scripts/record_outcomes.py` to synthesize `anytime_td` from `rushing_tds` and `receiving_tds` so bets on anytime touchdown props grade accurately against recorded game actuals.

#### Files Modified
- `sports/markets.py`
- `sports/nfl.py`
- `utils/nfl_sigma.py`
- `models/position_specific/weekly.py`
- `value_betting_engine.py`
- `utils/nfl_markets.py`
- `scripts/record_outcomes.py`
- `tests/test_sports_markets.py`
- `tests/test_internal_lines.py`

---

### 5. Multi-Season Empirical WR Role Priors (Unit 5)

#### Problem
Early in the season or when sample size is low, receiver projections blend player history with role priors. The priors in `config.py` were static guesses from early iterations:
- Alpha: 75.0 yards
- Secondary: 55.0 yards
- Slot: 45.0 yards
- Fringe: 30.0 yards

In 2024–2025 actuals, an average starting alpha wide receiver averaged ~58 yards per game, and fringe receivers averaged under 10 yards. The 75-yard alpha prior and 30-yard fringe prior caused massive over-projection on low-snap wide receivers.

#### Solution
- Implemented `calibrate_wr_role_priors()` in `utils/season_priors.py` to calculate empirical mean receiving yards by snap share bracket across recent NFL seasons.
- Calibrated new empirical defaults (`DEFAULT_WR_ROLE_PRIORS`):
  - **Alpha** (\(\ge 80\%\) snaps): **58.0 yards** (reduced from 75.0)
  - **Secondary** (\(\ge 60\%\) snaps): **43.0 yards** (reduced from 55.0)
  - **Slot** (\(\ge 40\%\) snaps): **30.0 yards** (reduced from 45.0)
  - **Fringe** (\(< 40\%\) snaps): **10.0 yards** (reduced from 30.0)
- Updated static and runtime configuration:
  - `config.py`: `IntegrationConfig.role_priors`
  - `config/runtime.py`: `integration.role_priors`
  - `data_pipeline.py`: `WR_ROLE_PRIORS` and `_wr_role_prior()` (normalized snap percentages between 0.0 and 1.0)
  - `scripts/ingest_real_nfl_data.py`: `assign_wr_role_prior()`

#### Files Modified
- `utils/season_priors.py`
- `config.py`
- `config/runtime.py`
- `data_pipeline.py`
- `scripts/ingest_real_nfl_data.py`
- `tests/test_season_priors.py`

---

## Verification & Test Evidence

### Full Regression Suite
The entire project test suite was executed:
```bash
make test
```
**Results**:
- **2,147 tests passed**, 0 failed, 1 skipped.
- Total run time: ~4 minutes 17 seconds.

### Targeted Component Tests
All unit and integration test suites covering the changes pass cleanly:
```bash
uv run pytest \
  tests/test_game_context.py \
  tests/test_ingest_game_script.py \
  tests/test_qb_decomposition.py \
  tests/test_qb_gating.py \
  tests/test_sports_markets.py \
  tests/test_internal_lines.py \
  tests/test_value_engine_side.py \
  tests/test_market_mu_wr.py \
  tests/test_season_priors.py \
  tests/test_nfl_weekly_model.py \
  tests/test_learning_loop.py \
  tests/test_nfl_usage_floors.py \
  tests/test_nfl_projection_evaluation.py
```
**Results**:
- **197 targeted tests passed** (including all MAE position gate checks).

---

## Operational Guide

### 1. Backfilling Existing Database Records
To populate empirical game script and expected script on existing historical rows:
```bash
uv run python -c "
from scripts.ingest_real_nfl_data import backfill_game_script_and_expected_script
results = backfill_game_script_and_expected_script(seasons=[2023, 2024, 2025, 2026])
print('Backfill results:', results)
"
```

### 2. Training Models for All Supported Markets
To train models including receptions and anytime touchdown:
```bash
make nfl-train
```
Or for specific markets:
```bash
uv run python -m models.position_specific.weekly train --market rushing_yards
uv run python -m models.position_specific.weekly train --market receiving_yards
uv run python -m models.position_specific.weekly train --market passing_yards
uv run python -m models.position_specific.weekly train --market receptions
uv run python -m models.position_specific.weekly train --market anytime_touchdown
```

### 3. Running Weekly Predictions & Cards
To run projections and value materialization with the updated pipeline:
```bash
make week-refresh SEASON=2026 WEEK=1
make week-predict SEASON=2026 WEEK=1
make week-materialize SEASON=2026 WEEK=1
make mae-gate SEASON=2026 WEEK=1
```
