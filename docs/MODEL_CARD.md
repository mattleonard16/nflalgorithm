# Weekly Model Card

**Model family:** `StackingRegressor` ensemble per market — RandomForest + GradientBoosting base
estimators (XGBoost added when the package is available), with `Ridge(alpha=1.0)` as the final
estimator. Built in `_build_nfl_model` (`models/position_specific/weekly.py`). CV folds are chosen
from the sample count, since stacking requires `n_samples >= cv`.

**Markets:** `rushing_yards`, `receiving_yards`, `passing_yards`. One model per market; there is no
per-position split — the orphaned `RBModel` subclass was deleted, and `weekly.py` is the single
production model path.

**Training data:** Weekly feature frames from the data pipeline across a rolling list of
`(season, week)` tuples. Features include odds line, targets, rolling EWM usage, team ranks,
weather flags, and injury indicators.

**Causality:** Outcome-time values (`snap_count`, `snap_percentage`, `game_script`, `target_share`,
…) are excluded from target-week rows via `_OUTCOME_CONTEXT_COLS`. The feature builder exposes only
their lagged pregame estimates, so a target week never consumes its own results.

**Targets:** Market-specific mean projection (`mu`) for rushing, receiving, and passing yards.

**Versioning:** Artifacts under `models/weekly/<market>_model.joblib` with metadata:
- `model_version`: UTC timestamp string (`weekly-YYYYMMDDTHHMMSSZ`)
- `feature_columns`: columns used during fit
- `featureset_hash`: emitted per projection row for reproducibility

**Inference pipeline:**
1. Run `make week-update` then `make week-predict` for the desired week.
2. `models.position_specific.predict_week(season, week)` writes `weekly_projections` rows.
3. Output includes `mu`, `sigma`, `model_version`, and `featureset_hash`.

**Calibration:** `sigma` is per-player, not a single residual constant. `compute_player_sigma`
(`utils/nfl_sigma.py`) takes an EWMA-weighted standard deviation over the player's game history
(decay 0.65), floored by a per-market minimum and falling back to a per-market default when history
is too thin. An uncertainty multiplier scales it further at predict time.

**Evaluation:** `scripts/evaluate_nfl_projections.py` reports MAE/RMSE overall and by market, model
version, and position. `make mae-gate` enforces absolute per-position ceilings (QB 65.0, RB 26.0,
WR 29.0, TE 27.0) and exits non-zero on breach. Each ceiling is ~10% above that position's worst
single-week MAE in the 2025 walk-forward baseline, so a normal bad week passes and a broken model
trips the gate. Positions with fewer than 30 projections are reported as skipped rather than
passed. `config.model.target_mae = 3.0` is the aspirational target, not the gate threshold.

**Known limitations:**
- The gate's real-data path is blocked only for legacy 2025 projection rows, whose `team` is
  unpopulated (546 of 568), so the join to `games` finds no kickoff there. The current
  roster-backed path populates `team` on every row (2026 W1: 0 empty of 1,396); the gate becomes
  verifiable on real data once 2026 actuals land. Gate logic is unit-tested.
- Passing yard market infers mean via a rolling air-yards proxy.
- Injury and weather feeds fall back to heuristics when upstream data is missing.
- EWMA decay 0.65 is uniform across markets and untuned.
- Final estimator is Ridge; LightGBM or isotonic calibration is an open improvement.
