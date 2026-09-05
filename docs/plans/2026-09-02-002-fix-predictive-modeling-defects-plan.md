---
title: Fix Predictive Modeling & Feature Engineering Defects
type: fix
date: 2026-09-02
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
---

# Fix Predictive Modeling & Feature Engineering Defects

## Goal Capsule

Eliminate runtime SQL failures, test collection errors in CI, and model projection inaccuracies across the predictive modeling and feature engineering pipeline on `feat/predictive-modeling-feature-engineering`. Resolve the six confirmed defects identified during code review: synthesize anytime touchdown actuals dynamically from tracked rushing and receiving touchdown counts without modifying database schema, isolate gitignored test dependencies so clean CI checkouts pass test collection, wire pregame expected game script into QB volume decomposition instead of last week's realized score margin, lower the anytime touchdown volume eligibility floor to retain active skill-position players, propagate Poisson survival pricing and 0.5 fixed line minting while protecting yardage MAE gates from count dilution, and eliminate DataFrame index realignment bugs in game context joins.

Authority order:

1. The Product Contract in this document defines the behavior of projections, grading, pricing, and evaluation gates.
2. The causal pregame prediction contract in `models/position_specific/weekly.py` defines feature lag and leakage prevention rules.
3. Market schemas in `sports/markets.py` and `sports/nfl.py` define canonical sport and prop types.
4. Repository testing conventions in `tests/conftest.py` define CI collection boundaries between tracked and proprietary modules.
5. Existing evaluation metrics in `scripts/evaluate_nfl_projections.py` (`mae-gate`) govern model quality ceilings.

Execution profile: six dependency-ordered implementation units covering data access decoupling, test environment isolation, QB context wiring, volume eligibility calibration, pricing and evaluation alignment, and DataFrame join safety.

Stop conditions:

- Stop if any proposed fix requires modifying database migrations or adding a stored `anytime_td` column to `player_stats_enhanced`.
- Stop if test isolation breaks local execution of private algorithm test suites.
- Stop if QB passing attempts decomposition introduces lookahead leakage by reading post-kickoff game margins.
- Stop if position MAE gate alterations allow yardage regressions to pass undetected.

Tail ownership: the executor owns implementation, unit tests, walk-forward backtest verification, and documentation synchronization.

---

## Product Contract

### Summary

The recent predictive modeling feature expansion introduced six verified defects spanning runtime execution, test suite portability, and model calibration:
- `sports/markets.py` registered `anytime_touchdown` with `stat_column="anytime_td"`, but `player_stats_enhanced` possesses only `rushing_tds` and `receiving_tds`. Queries selecting `MARKET_TO_STAT.values()` crash in backtest runners and evaluation scripts with `sqlite3.OperationalError: no such column: anytime_td`, while synthesis logic was duplicated across three separate files with an unhandled scalar `AttributeError`.
- Clean CI environments lacking proprietary gitignored files fail test collection because `tests/test_sports_markets.py` imports `prob_over` from gitignored `value_betting_engine.py` and `tests/test_qb_gating.py` is omitted from `tests/conftest.py:collect_ignore`.
- QB passing volume decomposition in `models/position_specific/weekly.py` inadvertently reads `game_script` (last week's realized score differential from player history) instead of `expected_game_script` (pregame Vegas spread expectations from snapshots).
- The volume floor `MARKET_MIN_EXPECTED_VOLUME["anytime_touchdown"] = 2.5` red zone touches per game excludes ~96% of active WRs and TEs, leaving the anytime touchdown market virtually unpopulated.
- Poisson survival pricing was restricted to `value_betting_engine.py` while tracked callers (`utils/clv.py`, `scripts/backtest_replay.py`, `scripts/dry_run_validation.py`) fell back to Gaussian normal CDFs. `utils/internal_lines.py` rounded $\mu$ to line increments (minting 0.0 or 1.0) rather than fixing anytime TD at line 0.5, and position MAE evaluation diluted yardage error ceilings by averaging in 0.3-scale TD errors.
- `utils/game_context.py:attach_game_context_to_player_frame` reassigned `df["team"] = orig_team` after an index-resetting merge, corrupting or wiping the `team` column to `NaN` whenever the input frame possessed a non-default index.

This plan addresses all six defects systematically.

### Problem Frame

The predictive modeling engine cannot be deployed or verified reliably in CI while test collection crashes on missing private modules and runtime backtests crash on non-existent database columns. Furthermore, pricing anytime touchdown props with normal CDFs and rounding their lines to 0.0 or 1.0 breaks fair-odds and edge calculations. Finally, misaligning player teams and feeding historical score margins into pregame QB attempt projections degrades projection quality and compromises downstream betting decisions.

### Key Decisions

- **Synthesize anytime touchdown actuals dynamically at read time via a shared tracked helper.** (session-settled: user-directed — chosen over a stored database column and schema migration: computing `anytime_td = (rushing_tds + receiving_tds)` on the fly avoids schema mutations, keeps the database schema clean, and eliminates duplicate synthesis blocks across modules.) Governs R1.
- **Isolate proprietary test dependencies to preserve clean CI test collection.** (session-settled: user-directed — chosen over mocking private modules in CI: moving `prob_over` to a tracked utility allows public market tests to run natively everywhere, while registering `test_qb_gating.py` in `tests/conftest.py` ensures clean checkouts skip private tests gracefully.) Governs R2.
- **Wire pregame expected game script into QB passing attempts decomposition.** (session-settled: user-directed — chosen over realized prior-week game script: passing volume must reflect the upcoming matchup spread rather than the previous game's final margin.) Governs R3.
- **Calibrate anytime touchdown volume eligibility floor to 0.5 expected red zone touches.** (session-settled: user-directed — chosen over keeping the 2.5 floor or switching to general touches: 0.5 expected red zone touches per game captures primary and secondary skill players who realistically score touchdowns without opening the market to pure non-contributors.) Governs R4.
- **Propagate Poisson pricing, fix anytime TD lines at 0.5, and isolate count markets from yardage MAE gates.** (session-settled: user-directed — chosen over uncalibrated normal CDFs and mixed MAE metrics: anytime TD props are binary over/under 0.5 bets priced via Poisson survival $1 - e^{-\mu}$, and evaluating them separately prevents yardage error ceilings from being artificially deflated.) Governs R5.
- **Preserve input DataFrame integrity during game context joins without index-dependent reassignment.** (session-settled: user-directed — chosen over restoring series by index: dropping redundant columns or using numpy arrays prevents pandas index misalignment from corrupting team identifiers.) Governs R6.

### Requirements

- R1. All SQL queries and DataFrame reshaping operations that access player actuals must successfully execute without requesting a physical `anytime_td` column from `player_stats_enhanced`. Anytime touchdown counts must be synthesized via a single tracked helper function in `utils/nfl_markets.py` that handles missing columns gracefully.
- R2. `pytest tests/` must collect and pass cleanly in an environment where all gitignored proprietary algorithm files are absent. Public tests in `tests/test_sports_markets.py` must import pricing math from tracked modules, and private tests in `tests/test_qb_gating.py` must be included in `tests/conftest.py:collect_ignore`.
- R3. In `models/position_specific/weekly.py:_enrich_with_decomposition`, the QB attempt decomposition must read `expected_game_script` from the player context snapshot, falling back to 0.0 only when absent.
- R4. `sports/nfl.py:MARKET_MIN_EXPECTED_VOLUME["anytime_touchdown"]` must be set to `0.5`, enabling realistic WR, TE, and RB market eligibility across weekly slates.
- R5. Market-aware probability calculation supporting Poisson survival for `anytime_touchdown` must be accessible to all projection, line minting, CLV, replay, and validation callers. `utils/internal_lines.py` must mint anytime touchdown props strictly at line `0.5`, and `scripts/evaluate_nfl_projections.py:check_position_mae` must evaluate position MAE strictly over yardage markets.
- R6. `utils/game_context.py:attach_game_context_to_player_frame` must preserve player `team` values and context columns accurately regardless of the input DataFrame's index structure.

### Scope Boundaries

- **In Scope**:
  - Fixing findings 1, 2, 3, 5, 6, and 8 from the code review.
  - Absorbing finding 7 (scalar `fillna` bug) into the shared anytime TD synthesis helper.
  - Adding unit tests for each corrected behavior.
  - Verifying full test suite passes and walk-forward backtest succeeds.
- **Out of Scope**:
  - Finding 4: Slate eligibility filter for unnamed markets in `utils/slate_eligibility.py` (explicitly excluded by user).
  - Finding 9: Air yards and YAC rolling decay rate unification (explicitly excluded by user).
  - Finding 10: Roster merge duplicate key safety in `weekly.py` (explicitly excluded by user).
  - Adding database schema migrations or physical columns to SQLite tables.
  - Frontend UI redesigns or odds scraper modifications.
- **Deferred to Follow-Up Work**:
  - Extending Poisson survival pricing to other discrete count markets such as defensive sacks or field goals.
  - Position-specific MAE thresholds dedicated specifically to touchdown count markets.

---

## Planning Contract

### Key Technical Decisions

- **KTD1: Tracked Read-Time TD Synthesis Helper.** `utils/nfl_markets.py` exports `synthesize_anytime_td(df: pd.DataFrame) -> pd.DataFrame` which computes integer touchdown counts as `(rush + rec).astype(int)` using safe `pd.Series` conversion with `.fillna(0.0)`. A new export `DATABASE_STAT_COLUMNS` lists all physical columns present in `player_stats_enhanced` needed across all markets, mapping `anytime_touchdown` to `("rushing_tds", "receiving_tds")`. All SQL queries in `scripts/run_nfl_backtest.py` and `scripts/evaluate_nfl_projections.py` select `DATABASE_STAT_COLUMNS` instead of raw `MARKET_TO_STAT.values()`.
- **KTD2: Separation of Tracked Math and Private Test Fencing.** The core probability evaluation function `prob_over(mu, sigma, line, market=None)` is placed in `utils/nfl_markets.py` as a tracked helper, supporting Gaussian normal CDF for continuous markets and Poisson survival $P(\text{Over } 0.5) = 1 - e^{-\max(0, \mu)}$ for anytime touchdown props. `value_betting_engine.py` re-exports this function. `tests/test_sports_markets.py` imports from `utils/nfl_markets.py`, and `tests/test_qb_gating.py` is added to `_private_algorithm_tests` in `tests/conftest.py`.
- **KTD3: Pregame Script Forwarding for QB Passing Decomposition.** In `models/position_specific/weekly.py`, `_enrich_with_decomposition` extracts `expected_game_script` from the incoming row. This guarantees that QB volume estimates reflect the pregame Vegas spread margin rather than the previous week's realized score margin.
- **KTD4: Skill-Position Red Zone Volume Threshold Alignment.** The eligibility threshold `MARKET_MIN_EXPECTED_VOLUME["anytime_touchdown"]` is reduced from 2.5 to 0.5 expected red zone touches in `sports/nfl.py`. This aligns with the empirical reality that high-value receivers and goal-line backs average between 0.5 and 1.8 red zone touches per game.
- **KTD5: Anytime TD Market Pricing, Line Minting & MAE Evaluation.** `utils/internal_lines.py:_assemble` assigns `line = 0.5` whenever `market == "anytime_touchdown"`, bypassing continuous increment rounding. `utils/clv.py`, `scripts/backtest_replay.py`, and `scripts/dry_run_validation.py` pass `market` into `prob_over`. In `scripts/evaluate_nfl_projections.py`, `by_position` MAE aggregation is restricted to continuous yardage markets (`{"rushing_yards", "receiving_yards", "passing_yards"}`), preventing fractional touchdown errors from diluting yardage error gates.
- **KTD6: Vectorized Team Preservation in Game Context Join.** In `utils/game_context.py:attach_game_context_to_player_frame`, the unsafe index-based assignment `df["team"] = orig_team` is removed. Because `suffixes=("", "_context_dup")` is specified during the merge, `df["team"]` already retains the player's team, and dropping `team_context_dup` cleanly avoids index corruption.

### High-Level Technical Design

```mermaid
flowchart TD
    subgraph DataAccess [Data Access & Synthesis (U1, U6)]
        PSE[player_stats_enhanced: rushing_tds, receiving_tds] --> LoadActuals[scripts/_load_actuals: queries DATABASE_STAT_COLUMNS]
        LoadActuals --> SynthTD[utils/nfl_markets.py: synthesize_anytime_td]
        SynthTD --> MeltActuals[melt_actuals: single shared touchdown count]
        GameContextJoin[utils/game_context.py: attach_game_context_to_player_frame] --> CleanTeam[Safe merge without index corruption]
    end

    subgraph FeatureEngineering [Feature & Decomposition Wiring (U3, U4)]
        Snapshots[nfl_player_context_snapshots: expected_game_script] --> WeeklyRoster[weekly.py: roster merge]
        WeeklyRoster --> QBDecomp[weekly.py: _enrich_with_decomposition reads expected_game_script]
        QBDecomp --> QBProjections[QB pass_attempts_predicted]
        NFLConfig[sports/nfl.py: min_volume = 0.5] --> RoleMask[weekly.py: _eligible_role_mask includes skill players]
    end

    subgraph PricingAndEval [Pricing, Minting & MAE Gate (U2, U5)]
        TrackedPricing[utils/nfl_markets.py: prob_over with Poisson survival] --> CLV[utils/clv.py: _fair_prob with market]
        TrackedPricing --> Replay[scripts/backtest_replay.py]
        TrackedPricing --> DryRun[scripts/dry_run_validation.py]
        InternalLines[utils/internal_lines.py: _assemble] --> FixedLine[Fixed line = 0.5 for anytime_touchdown]
        EvalScript[scripts/evaluate_nfl_projections.py] --> YardageMAE[by_position groups strictly on yardage markets]
    end

    subgraph TestPortability [CI Portability (U2)]
        Conftest[tests/conftest.py: collect_ignore] --> TestQBGating[Skips test_qb_gating.py when private files absent]
        TrackedPricing --> TestSportsMarkets[tests/test_sports_markets.py runs in pure CI]
    end
```

### Core Interfaces & Workflows

1. **Actuals Loading Workflow**:
   - `scripts/run_nfl_backtest.py` and `scripts/evaluate_nfl_projections.py` invoke `_load_actuals`.
   - The query formats `DATABASE_STAT_COLUMNS` from `utils.nfl_markets`.
   - `melt_actuals` calls `synthesize_anytime_td`, producing valid touchdown counts under `actual` with `market = "anytime_touchdown"`.
2. **QB Projection Workflow**:
   - `models/position_specific/weekly.py:predict_week` calls `_build_roster_week_data`.
   - Pregame `expected_game_script` is merged from context snapshots onto `roster`.
   - `_enrich_with_decomposition` extracts `expected_game_script` and passes it to `decompose_qb_passing`.
3. **Bet Pricing & Line Minting Workflow**:
   - `utils/internal_lines.py` creates projection lines, assigning `line = 0.5` for anytime touchdown.
   - `utils/clv.py`, `scripts/backtest_replay.py`, and `scripts/dry_run_validation.py` compute fair probability via `prob_over(mu, sigma, line, market=market)`.
   - For `anytime_touchdown`, `prob_over` executes $1 - e^{-\mu}$, while yardage markets execute the normal CDF.

### Failure Modes & Edge Cases

- **Missing Red Zone Touches in Input Frame**: If `expected_red_zone_touches` is missing or null, `pd.to_numeric` fills 0.0, and players falling below 0.5 are excluded as expected.
- **Missing Touchdown Columns in Actuals**: If only `rushing_tds` or only `receiving_tds` is present, `synthesize_anytime_td` constructs a 0.0 Series for the missing column, preventing `AttributeError` or key errors.
- **Non-RangeIndex in Player DataFrames**: If callers pass DataFrames with arbitrary, negative, or duplicated indices into `attach_game_context_to_player_frame`, the merge preserves the original team column without index-based overwrite.
- **Extreme $\mu$ Values in Poisson Survival**: If $\mu < 0$ due to model anomalies, `max(0.0, mu)` ensures $1 - e^0 = 0.0$ rather than negative probabilities.

---

## Implementation Units

### U1. Shared Tracked Anytime-TD Synthesis & Actuals Query Decoupling

- **Goal**: Resolve SQL crashes and consolidate duplicate anytime TD synthesis logic into a robust, shared tracked helper.
- **Requirements**: R1.
- **Dependencies**: None.
- **Files**:
  - `utils/nfl_markets.py`
  - `scripts/run_nfl_backtest.py`
  - `scripts/evaluate_nfl_projections.py`
  - `scripts/record_outcomes.py`
  - `learning_loop.py`
  - `models/position_specific/weekly.py`
  - `tests/test_sports_markets.py`
- **Approach**:
  1. In `utils/nfl_markets.py`:
     - Define `DATABASE_STAT_COLUMNS`: sorted list of all physical stat columns in `player_stats_enhanced` needed across all markets. Replace `"anytime_td"` with `"rushing_tds"` and `"receiving_tds"`.
     - Implement `synthesize_anytime_td(df: pd.DataFrame) -> pd.DataFrame`: checks if `"anytime_td"` is already present; if not, constructs `rush` and `rec` Series safely using `pd.to_numeric` with `.fillna(0.0)`, and assigns `df["anytime_td"] = (rush + rec).astype(int)`.
     - Refactor `melt_actuals` to invoke `synthesize_anytime_td(df_actuals)` directly, eliminating the bug where a missing column returns integer 0 and triggers `AttributeError: 'int' object has no attribute 'fillna'`.
  2. In `scripts/run_nfl_backtest.py:_load_actuals`:
     - Replace `stat_columns = ", ".join(sorted(set(MARKET_TO_STAT.values())))` with `stat_columns = ", ".join(DATABASE_STAT_COLUMNS)`.
  3. In `scripts/evaluate_nfl_projections.py:480`:
     - Replace `actual_columns = ", ".join(sorted(set(MARKET_TO_STAT.values())))` with `actual_columns = ", ".join(DATABASE_STAT_COLUMNS)`.
  4. In `scripts/record_outcomes.py`:
     - Replace lines 93-96 with `actuals = synthesize_anytime_td(actuals)`.
  5. In `learning_loop.py`:
     - In `_load_actual_stat`: handle `market == "anytime_touchdown"` by querying `(COALESCE(rushing_tds, 0) + COALESCE(receiving_tds, 0)) AS anytime_td`.
     - In the aggregation loop at line 498: handle `anytime_touchdown` with SQL expression `(COALESCE(s.rushing_tds, 0) + COALESCE(s.receiving_tds, 0))`.
  6. In `models/position_specific/weekly.py`:
     - In lines 165-178: replace inline synthesis with `df = synthesize_anytime_td(df)`.
- **Patterns to follow**: `utils/nfl_markets.py` pure function helpers with defensive type coercion.
- **Test scenarios**:
  - Input DataFrame has only `rushing_tds`: `synthesize_anytime_td` creates `anytime_td` without throwing `AttributeError`.
  - Input DataFrame has only `receiving_tds`: `synthesize_anytime_td` creates `anytime_td` correctly.
  - Input DataFrame has neither column: `synthesize_anytime_td` leaves frame unchanged or fills zeros cleanly.
  - Player with 2 rushing TDs and 1 receiving TD produces `anytime_td = 3` (discrete count).
  - SQL query generation in backtest and evaluation scripts selects existing physical columns without raising SQLite OperationalError.
- **Verification**: `pytest tests/test_sports_markets.py tests/test_learning_loop.py` passes cleanly.

---

### U2. CI Test Collection Isolation & Tracked Pricing Exposure

- **Goal**: Allow the test suite to be collected and executed in a clean CI environment lacking proprietary algorithm files.
- **Requirements**: R2.
- **Dependencies**: U1.
- **Files**:
  - `utils/nfl_markets.py`
  - `value_betting_engine.py`
  - `tests/test_sports_markets.py`
  - `tests/conftest.py`
- **Approach**:
  1. In `utils/nfl_markets.py`:
     - Implement `prob_over(mu: float, sigma: float, line: float, market: str | None = None) -> float`: computes Poisson survival $1 - e^{-\max(0, \mu)}$ when `market == "anytime_touchdown"` or `(line == 0.5 and market is not None and "touchdown" in market)`, and normal CDF $1 - \Phi((line - \mu)/\sigma)$ otherwise.
  2. In `value_betting_engine.py`:
     - Re-export `prob_over` from `utils.nfl_markets` to maintain backward compatibility.
  3. In `tests/test_sports_markets.py`:
     - Change import from `from value_betting_engine import prob_over` to `from utils.nfl_markets import prob_over`.
  4. In `tests/conftest.py`:
     - Add `"test_qb_gating.py"` to `_private_algorithm_tests` so that `collect_ignore` skips it when `data_pipeline.py` or `models/position_specific/weekly.py` are absent.
- **Patterns to follow**: `tests/conftest.py:collect_ignore` registration pattern.
- **Test scenarios**:
  - When proprietary files are removed or simulated as missing, `pytest --collect-only` completes with 0 collection errors.
  - `tests/test_sports_markets.py` executes successfully in pure isolation without importing any gitignored files.
  - `prob_over` produces identical numerical values for continuous and count props across both import paths.
- **Verification**: `pytest tests/test_sports_markets.py` passes, and `pytest --collect-only tests/test_qb_gating.py` honors collection rules.

---

### U3. QB Pregame Expected Game Script Wiring

- **Goal**: Ensure QB passing attempts predictions use pregame spread expectations rather than previous week's realized score margin.
- **Requirements**: R3.
- **Dependencies**: None.
- **Files**:
  - `models/position_specific/weekly.py`
  - `tests/test_qb_gating.py`
- **Approach**:
  1. In `models/position_specific/weekly.py:_enrich_with_decomposition`:
     - At line 956, replace `game_script = float(row.get("game_script", 0) or 0)` with:
       `game_script = float(row.get("expected_game_script", 0) or 0)`.
  2. Verify that `expected_game_script` is selected from `nfl_player_context_snapshots` (line 693), merged into `roster` (line 722), and preserved through `frame` (line 755).
- **Patterns to follow**: Pregame causal feature selection rules in `docs/PREDICTIVE_MODELING_IMPROVEMENTS_2026.md`.
- **Test scenarios**:
  - A starting QB whose previous week's game had `game_script = -14.0` (trailing blowout) but whose upcoming matchup has `expected_game_script = +7.0` (favored) receives attempt projections based on +7.0, suppressing attempts.
  - When `expected_game_script` is `None` or `NaN`, default fallback to `0.0` operates safely without raising TypeError.
- **Verification**: `pytest tests/test_qb_gating.py` passes and confirms that leading pregame script suppresses passing volume.

---

### U4. Anytime Touchdown Red Zone Volume Gate Floor Alignment

- **Goal**: Lower the anytime touchdown red zone volume threshold to include primary and secondary offensive skill players.
- **Requirements**: R4.
- **Dependencies**: None.
- **Files**:
  - `sports/nfl.py`
  - `tests/test_sports_markets.py`
  - `docs/PREDICTIVE_MODELING_IMPROVEMENTS_2026.md`
  - `docs/MODEL_CARD.md`
- **Approach**:
  1. In `sports/nfl.py`:
     - Update `MARKET_MIN_EXPECTED_VOLUME["anytime_touchdown"] = 0.5` (lowered from 2.5).
  2. In `tests/test_sports_markets.py`:
     - Update assertion in `test_market_min_volume_floors` from `2.5` to `0.5`.
  3. In `docs/PREDICTIVE_MODELING_IMPROVEMENTS_2026.md` and `docs/MODEL_CARD.md`:
     - Update documented volume floor for anytime touchdown props from 2.5 to 0.5 expected red zone touches.
- **Patterns to follow**: Volume floor dictionary conventions in `sports/nfl.py`.
- **Test scenarios**:
  - WR with 0.8 expected red zone touches is marked eligible under `_eligible_role_mask(df, "anytime_touchdown")`.
  - Fringe player with 0.2 expected red zone touches is filtered out.
  - `test_market_min_volume_floors` validates that 0.5 is the active threshold.
- **Verification**: `pytest tests/test_sports_markets.py` passes with the updated threshold.

---

### U5. Poisson Pricing Propagation, Fixed Line Minting & MAE Gate Separation

- **Goal**: Ensure consistent Poisson pricing across all consumers, fix minted anytime TD lines at 0.5, and prevent touchdown errors from diluting yardage MAE gates.
- **Requirements**: R5.
- **Dependencies**: U1, U2.
- **Files**:
  - `utils/internal_lines.py`
  - `utils/clv.py`
  - `scripts/backtest_replay.py`
  - `scripts/dry_run_validation.py`
  - `scripts/evaluate_nfl_projections.py`
  - `tests/test_internal_lines.py`
  - `tests/test_clv.py`
  - `tests/test_nfl_projection_evaluation.py`
- **Approach**:
  1. In `utils/internal_lines.py:_assemble`:
     - When minting lines, check the market:
       `lines["line"] = [0.5 if m == "anytime_touchdown" else round_to_line(val, increment) for m, val in zip(lines["market"], lines["mu"])]`.
  2. In `utils/clv.py`:
     - Update `_fair_prob` to accept `market: str | None = None`.
     - In `_fair_prob`, pass `market=market` to `prob_over(mu, sigma, float(line), market=market)`.
     - In `score_clv_row`, extract `market = entry.get("market") or close.get("market")` and pass to `_fair_prob`.
  3. In `scripts/backtest_replay.py`:
     - At line 211, pass `market=row.get("market")` into `prob_over`.
  4. In `scripts/dry_run_validation.py`:
     - At line 155, pass `market=pick.get("market")` into `prob_over`.
  5. In `scripts/evaluate_nfl_projections.py:evaluate_projections`:
     - Restrict `by_position` grouping to yardage markets (`{"rushing_yards", "receiving_yards", "passing_yards"}`), so count markets with $\approx 0.3$ errors do not deflate position yardage MAE.
     - Add `by_market_position` in metrics to retain position-level visibility for non-yardage markets without compromising the gate.
- **Patterns to follow**: Market-specific line minting and metric segregation in evaluation reports.
- **Test scenarios**:
  - Minting internal lines for `anytime_touchdown` always produces `line = 0.5` regardless of whether `mu` is 0.22, 0.48, or 0.85.
  - `_fair_prob` in `utils/clv.py` yields Poisson survival probability for anytime touchdown rows and normal CDF for yardage rows.
  - `evaluate_projections` position MAE for RB matches the average rushing yard error and is not deflated by anytime TD errors.
- **Verification**: `pytest tests/test_internal_lines.py tests/test_clv.py tests/test_nfl_projection_evaluation.py` passes cleanly.

---

### U6. Game Context Index Alignment Safety

- **Goal**: Prevent player team identifiers from being scrambled or overwritten with `NaN` when attaching game context to player frames with non-default indices.
- **Requirements**: R6.
- **Dependencies**: None.
- **Files**:
  - `utils/game_context.py`
  - `tests/test_game_context.py`
- **Approach**:
  1. In `utils/game_context.py:attach_game_context_to_player_frame`:
     - Remove the unsafe `df["team"] = orig_team` assignment at line 300.
     - Verify that `df.merge(context_df, ... suffixes=("", "_context_dup"))` retains the original `team` column under `"team"`, while `team_context_dup` is dropped at line 302.
     - Alternatively, if re-assignment is needed for casing or formatting, assign via `.to_numpy()` to decouple from pandas index alignment.
  2. In `tests/test_game_context.py`:
     - Add a unit test verifying `attach_game_context_to_player_frame` on a DataFrame with non-standard index (e.g. `index=[5, 9, 42]`), asserting `team` values match the input exactly.
- **Patterns to follow**: Index-agnostic DataFrame transformations in `utils/game_context.py`.
- **Test scenarios**:
  - Input DataFrame with `index=[10, 20]` preserves `team = ["KC", "BUF"]` and attaches valid `spread_margin` without introducing `NaN`.
  - Input DataFrame with single-row filtered index preserves team identity.
- **Verification**: `pytest tests/test_game_context.py` passes with the new index test.

---

## Verification Contract

### Automated Verification

Execute targeted unit test suites for each modified module:

```bash
# 1. Verify market registration, pricing, and touchdown synthesis
.venv/bin/pytest tests/test_sports_markets.py -v

# 2. Verify game context index alignment and feature attachments
.venv/bin/pytest tests/test_game_context.py -v

# 3. Verify QB volume calibration and starter probability gating
.venv/bin/pytest tests/test_qb_gating.py -v

# 4. Verify internal line minting (0.5 for anytime TD)
.venv/bin/pytest tests/test_internal_lines.py -v

# 5. Verify CLV calculation and fair probability with market parameter
.venv/bin/pytest tests/test_clv.py -v

# 6. Verify MAE gate position separation
.venv/bin/pytest tests/test_nfl_projection_evaluation.py -v

# 7. Run complete test suite to ensure zero regressions
.venv/bin/pytest
```

### Manual & Operational Verification

1. **Test Collection Isolation Verification**:
   - Temporarily simulate a clean checkout without private files (e.g. `PYTHONPATH=. pytest --collect-only tests/`) and confirm that test collection succeeds without `ModuleNotFoundError`.
2. **Walk-Forward Backtest Verification**:
   - Run walk-forward backtest execution:
     ```bash
     .venv/bin/python scripts/run_nfl_backtest.py run --season 2025 --weeks 1 2 3 --label defect_fix_eval
     ```
   - Verify that `_load_actuals` successfully loads stats for all markets including `anytime_touchdown` without SQLite column errors.
3. **MAE Gate Check**:
   - Run projection evaluation:
     ```bash
     .venv/bin/python scripts/evaluate_nfl_projections.py mae-gate --season 2025 --week 1
     ```
   - Verify that position MAEs evaluate only yardage markets and do not fail unexpectedly.

---

## Definition of Done

- [x] All six implementation units (U1–U6) are implemented in the codebase. (2026-09-03 — see Execution Log.)
- [x] No database schema changes or SQLite migrations are introduced. (No migration touched; verified by `git status` showing no schema files changed.)
- [x] `utils/nfl_markets.py:synthesize_anytime_td` handles missing touchdown columns without raising `AttributeError` and provides integer touchdown counts. (U1, landed 2026-09-03.)
- [x] `DATABASE_STAT_COLUMNS` is used by all SQL actuals loaders, preventing `no such column: anytime_td` failures. (U1, landed 2026-09-03.)
- [x] `tests/test_sports_markets.py` imports `prob_over` from tracked `utils.nfl_markets` and runs in clean CI without proprietary files. (U2, landed 2026-09-03 — see Execution Log.)
- [x] `tests/test_qb_gating.py` is ignored in `tests/conftest.py` when proprietary files are absent. (U2, landed 2026-09-03 — see Execution Log.)
- [x] `_enrich_with_decomposition` consumes `expected_game_script` from context snapshots. (U3, landed 2026-09-03 — local-only, gitignored file, CI-unverifiable by construction.)
- [x] `MARKET_MIN_EXPECTED_VOLUME["anytime_touchdown"]` is set to `0.5`. (U4, landed 2026-09-03.)
- [x] Anytime touchdown internal lines mint at `0.5`, and Poisson survival is passed to all `prob_over` callers. (U5, landed 2026-09-03.)
- [x] `check_position_mae` aggregates position MAE over yardage markets only. (U5, landed 2026-09-03 — via yardage-only `by_position` in `evaluate_projections`.)
- [x] `attach_game_context_to_player_frame` preserves `team` on DataFrames with arbitrary indices. (U6, landed 2026-09-03.)
- [x] All automated tests pass (2,148+ tests). (2169 passed, 1 skipped on 2026-09-03 — see Execution Log.)

---

## Execution Log

### 2026-09-03 — U2 landed (CI Test Collection Isolation & Tracked Pricing Exposure)

**Implement first, alone, per maintainer direction** ("U2 first, then reassess" — separable contributor-facing commit).

**Changes (tracked, committable):**

- `utils/nfl_markets.py`: new tracked `prob_over(mu, sigma, line, market=None)` — Poisson
  survival `1 - exp(-max(0, mu))` for `anytime_touchdown`, Gaussian normal CDF otherwise.
  Numerically identical to the engine version on all sampled inputs (verified by parity script).
- `tests/test_sports_markets.py`: import moved from gitignored `value_betting_engine` to
  tracked `utils.nfl_markets`. This test now runs natively in clean CI checkouts.
- `tests/conftest.py`: `"test_qb_gating.py"` added to `_private_algorithm_tests`, so clean
  checkouts (no `data_pipeline.py` / `weekly.py`) skip it instead of collection-erroring.

**Local-only change (gitignored, not committable):**

- `value_betting_engine.py`: local `prob_over` definition replaced with
  `from utils.nfl_markets import prob_over` re-export. Backward compatible — all existing
  callers (`scripts/backtest_replay.py`, `scripts/dry_run_validation.py`, `utils/clv.py`,
  private tests) keep working unchanged.

**Dead-test audit (2026-09-03):** grepped every `tests/`, `scripts/`, `utils/` file for
top-level imports of gitignored modules (`value_betting_engine`, `data_pipeline`,
`prop_integration`, `models.position_specific.weekly`). Every test file with such an import
is now in `tests/conftest.py`'s ignore lists except `tests/test_sports_markets.py`, which no
longer needs fencing. No test asserts the old import location, so nothing was deleted —
U2 changed code location only, not behavior. Remaining `from value_betting_engine import …`
users are scripts (not pytest-collected) and fenced private tests.

**Verification (all on the merged tree + U2 changes):**

- Parity: `utils.nfl_markets.prob_over` vs `value_betting_engine.prob_over` identical on
  TD / yardage / symmetric / negative-mu inputs.
- `pytest tests/test_sports_markets.py tests/test_qb_gating.py` → 10 passed.
- Fresh-clone simulation (moved the 4 gitignored algorithm files aside, ran
  `pytest tests/ --collect-only`): **1985 tests collected, 0 collection errors**
  (pre-fix: 2 collection errors — `test_sports_markets.py` via `value_betting_engine`,
  `test_qb_gating.py` via `data_pipeline`/`weekly`). Files restored afterward.
- `pytest tests/test_sports_markets.py tests/test_clv.py tests/test_weekly_pipeline.py` →
  36 passed (engine re-export causes no regressions for existing callers).
- Full suite `pytest tests/` → **2156 passed, 1 skipped**, identical to the post-merge
  baseline. Zero regressions from U2.

**Remaining:** U1, U3, U4, U5, U6 per the units above. Suggested order per review: U1, U6, U4
(all tracked, CI-verifiable), then U3 and U5 last (touch gitignored files; U3's
`weekly.py` fix can never be verified by CI — candidate for moving the decomposition math
into a tracked module per the `utils/clv.py` pattern).

### 2026-09-03 — U1, U6, U4, U3, U5 landed (rest of the plan)

Executed in dependency order U1 → U6 → U4 → U3 → U5, per maintainer direction
("implement rest of plan"; uncertain parts skipped — none were uncertain).

**U1 — Shared tracked anytime-TD synthesis & actuals query decoupling (R1):**

- `utils/nfl_markets.py`: new `DATABASE_STAT_COLUMNS` (all `MARKET_TO_STAT` values
  with virtual `anytime_td` replaced by physical `rushing_tds` + `receiving_tds`) and
  new `synthesize_anytime_td(df)` helper (integer **counts**, safe Series construction —
  absorbs finding 7's scalar-`fillna` crash; no-op when neither TD column exists;
  coerces a pre-existing column). `melt_actuals` now delegates to it.
- `scripts/run_nfl_backtest.py`, `scripts/evaluate_nfl_projections.py`: actuals queries
  select `DATABASE_STAT_COLUMNS` — no more `no such column: anytime_td`.
- `scripts/record_outcomes.py`: inline binary-flag block replaced with the helper.
- `learning_loop.py` (tracked): `_load_actual_stat` and the MAE-trend aggregation loop
  synthesize via `COALESCE(rushing_tds,0)+COALESCE(receiving_tds,0)` SQL expressions.
- `models/position_specific/weekly.py` (gitignored, local-only): inline block replaced
  with the helper import.
- Deliberate semantic note: the helper produces integer **counts**
  (`rush + rec`), not the old binary flag (`> 0`). The count is the Poisson rate
  the pricing engine consumes, so training target and pricing now agree.
  `docs/PREDICTIVE_MODELING_IMPROVEMENTS_2026.md`'s stale binary-indicator formula
  was updated to match. Grading over/under 0.5 is unaffected (count ≥ 1 ⟺ flag 1).
- Tests: 5 new tests in `tests/test_sports_markets.py` (physical-only columns, count
  semantics, missing-column tolerance, untouched frames, melt counts); actuals query
  executed against a scratch SQLite table to prove it runs.

**U6 — Game context index-alignment safety (R6):**

- `utils/game_context.py`: removed `orig_team` save/restore around the merge. The merge
  joins on `_clean_*` helpers, so the player's own `team` column is never a join key
  and survives intact; restoring it from a series saved under the caller's index
  realigned by label post-merge (fresh RangeIndex) and wiped teams to NaN. Added an
  explanatory comment at the site so the line is not reintroduced.
- Test: `test_non_default_index_preserves_team_identity` (index `[10, 20]` → teams
  intact, no NaNs, correct spreads, no `team_context_dup`).

**U4 — Anytime-TD volume floor 2.5 → 0.5 (R4):**

- `sports/nfl.py`: floor is now `0.5` expected red-zone touches.
- `tests/test_sports_markets.py` floor assertion updated; eligibility verified live:
  0.8 in / 0.2 out / 0.5 boundary in / null out.
- Docs: floor updated in `docs/PREDICTIVE_MODELING_IMPROVEMENTS_2026.md`.
  `docs/MODEL_CARD.md` states no floor value, so no change needed there.
- **Stale-test fallout (the "bad tests" the review predicted):** two tests encoded the
  old 2.5 floor and failed after the change —
  `test_nfl_weekly_model.py::...::test_role_eligibility_requires_rotation_level_volume`
  and `test_nfl_usage_floors.py::test_usage_floors_keep_rotation_players...`. Both
  updated to the 0.5 contract (the former's fixture now uses 0.2 red-zone touches to
  keep exercising the below-floor branch). No test was deleted; both assert the new
  intended behavior.

**U3 — QB pregame expected game script wiring (R3, gitignored `weekly.py`, local-only):**

- One-line change in `_enrich_with_decomposition`: reads `expected_game_script`
  (pregame Vegas expectation merged from `nfl_player_context_snapshots`) instead of
  `game_script` (last week's realized margin from history).
- **Found while verifying:** `float(row.get(...) or 0)` lets NaN through (NaN is
  truthy), so an explicit NaN → 0.0 guard was added — the plan's fallback scenario.
- Verified locally (CI cannot: file is gitignored): pregame +7 → 23.25 attempts vs
  fallback 0.0 → 31.0; None/NaN/missing all equal the 0.0 path. `test_qb_gating.py`
  passes. Open follow-up unchanged: move this math into a tracked module so CI can
  verify it (`utils/clv.py` pattern).

**U5 — Poisson propagation, fixed 0.5 lines, yardage-only MAE gate (R5):**

- `utils/internal_lines.py:_assemble`: `anytime_touchdown` mints line `0.5` for any
  mu (tested at 0.22 / 0.48 / 0.85).
- `utils/clv.py`: `_fair_prob` accepts `market` and forwards it to `prob_over`;
  `compute_clv` derives market from entry-or-close with a str-or-None normalization
  (a NaN cell must not reach `"touchdown" in market`). No-vig path untouched.
- `scripts/backtest_replay.py:211`, `scripts/dry_run_validation.py:155`: pass
  `market=` into `prob_over` (Gaussian callers now explicitly Poisson-aware for TD).
- `scripts/evaluate_nfl_projections.py`: new `YARDAGE_MARKETS` constant;
  `by_position` aggregates yardage markets only (gate ceilings are yard-scale);
  new `by_market_position` (`{market: {position: metrics}}`) preserves TD/count
  visibility without touching the gate. `check_position_mae` itself unchanged.
- Tests: fixed-line minting, `_fair_prob` Poisson-vs-Gaussian split, end-to-end
  `compute_clv` TD plumbing, and `by_position`-excludes-TD-errors with
  `by_market_position` retention — 80 passed across the three touched test files,
  plus 50 passed for replay/dry-run/backtest-harness suites.

**Verification (merged tree + U1–U6):**

- Fresh-clone collection simulation: **1998 collected, 0 errors**.
- Full suite `pytest tests/`: **2169 passed, 1 skipped** (baseline at merge: 2156/1;
  +13 net new tests, zero regressions after the two stale-floor updates above).
- No schema migrations; `git status` shows only the intended tracked files plus the
  pre-existing U7/U8 work-in-progress, which this work did not touch.

---

## Addendum (2026-09-03): Maintainer Experience, CI, and Security

**Context.** An outside contributor cloned the public repository, ran `make doctor`, and got
`[FAIL] private_api` because `api/server.py` and its `PUBLIC_VALUE_VISIBILITY_CONTRACT` do not
exist in a public clone. Nothing was broken: that file is gitignored and only the API server and
full-stack launcher need it. But the preflight said FAIL, and `docs/TROUBLESHOOTING.md` did not
cover it, so the contributor could not tell whether it was their mistake or ours. These units are
branch-independent and were implemented on `claude/nfl-algorithm-improvements-0qz81b`; U1–U6
above target `feat/predictive-modeling-feature-engineering` and their files do not exist here.

### U7. Preflight and contributor docs

- **Goal**: A public clone passes `make doctor` with WARN rows, and the docs say so before anyone has to ask.
- **Files**: `scripts/preflight.py`, `tests/test_preflight.py`, `Makefile`, `docs/TROUBLESHOOTING.md`, `CONTRIBUTING.md`, `README.md`, `docs/DEPLOYMENT_MANIFEST.md`.
- **Approach**:
  1. `check_private_api(root, *, required=False)`: missing file is `warn` unless required; a present-but-stale contract stays `fail`.
  2. New `--require-private-api` flag. Passed by `runtime-preflight` (so `make api` still refuses to start without the file), `runtime-production-preflight`, and every `doctor-*` target. Plain `make doctor` does not pass it.
  3. Troubleshooting gets a "Public clone vs deployment checkout" table and rows for the WARN, the FAIL, and the stale-contract messages.
  4. `CONTRIBUTING.md` rewritten: what is private, what works without it, setup, PR checklist, what CI runs, how to report a security issue.
- **Tests**: `test_missing_private_api_warns_on_public_clone`, `test_missing_private_api_blocks_api_startup_when_required`, stale-contract test now passes `required=False` and still expects `fail`.
- **Verification**: `pytest tests/test_preflight.py`; `git archive HEAD` into a scratch dir, then `python -m scripts.run_migrations` and `python -m scripts.preflight --check-schema` exit 0 with `private_api` / `private_modules` at `warn`.

### U8. CI and security gates

- **Goal**: Contributors cannot break the public-clone path or the test suite without CI saying so, and dependency and code security issues are caught on every PR.
- **Files**: `.github/workflows/ci.yml`, `.github/workflows/security.yml`, `.github/dependabot.yml`, `.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/setup-problem.md`.
- **Approach**:
  1. `ci.yml` gains `fresh-clone-setup`: migrate an empty DB, run preflight with `--json`, assert `ok` and that `private_api` / `private_modules` are `warn`, assert `pytest --collect-only` is clean, assert no proprietary file is tracked. Adds `permissions: contents: read` and cancel-in-progress concurrency.
  2. `security.yml`: `pip-audit` on `requirements.txt` (blocking, clean today), `bandit -lll` on tracked Python (blocking, clean today; the one HIGH finding is in gitignored `weekly.py`), gitleaks plus the tracked-private-file guard (blocking), `npm audit --audit-level=high` (report-only: 6 high advisories exist in `frontend/package-lock.json` today), CodeQL for Python and TypeScript. Weekly schedule so new advisories surface without a PR.
  3. Dependabot weekly grouped updates for pip, npm, and Actions.
- **Not gated, deliberately**: `black --check`, `isort --check`, and `mypy` all fail on the current tree (mypy: 169 errors in 49 files). Gating them would block every PR until a repo-wide cleanup lands. Follow-up: clean up, then add a `lint` job.
- **Verification**: YAML parses; the fresh-clone steps were run locally against a `git archive` of HEAD; `bandit -lll` and `pip-audit` run locally with no findings.
