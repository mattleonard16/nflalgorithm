# Weekly Model Card

**Model family:** `StackingRegressor` ensemble per market — RandomForest + GradientBoosting base
estimators (XGBoost added when the package is available), with `Ridge(alpha=1.0)` as the final
estimator. Built in `_build_nfl_model` (`models/position_specific/weekly.py`). CV folds are chosen
from the sample count, since stacking requires `n_samples >= cv`.

**Markets:** `rushing_yards`, `receiving_yards`, `passing_yards`, `receptions`, `anytime_touchdown`.
One model per market; there is no per-position split — the orphaned `RBModel` subclass was deleted,
and `weekly.py` is the single production model path.

**Training data:** Weekly feature frames from the data pipeline across a rolling list of
`(season, week)` tuples. Features include odds line, targets, rolling EWM usage, team ranks,
weather flags, injury indicators, pregame schedule context (`spread_margin`, `implied_team_total`,
`game_total`, `wind_speed`, `temperature`, `is_indoor`, `div_game`), and empirical game script.

**Causality:** Outcome-time values (`snap_count`, `snap_percentage`, `game_script`, `target_share`,
…) are excluded from target-week rows via `_OUTCOME_CONTEXT_COLS`. The feature builder exposes only
their lagged pregame estimates (e.g. `expected_game_script` derived from schedule `spread_line`),
so a target week never consumes its own results.

**Targets:** Market-specific mean projection (`mu`) for rushing yards, receiving yards, passing
yards, receptions count, and anytime touchdown expected scores (`anytime_td`).

**Probability & Pricing Engine:** (`utils/nfl_markets.prob_over`)
- Yardage markets (rushing, receiving, passing; `GAMMA_MARKETS`) evaluate win probability from gamma
  survival, parameterized by method of moments from the same `mu` and calibrated player-specific
  `sigma`: shape \((\mu/\sigma)^2\), scale \(\sigma^2/\mu\). Mean and variance are unchanged, so only
  the shape of the curve differs from the normal it replaced. Weekly yardage is right-skewed, and a
  normal centred on a conditional mean prices every over at 50% when a player clears his own mean
  closer to 39% of the time. Measured on the 2025 walk-forward rows
  (`reports/nfl_backtest_2025_sigma_v2_rows.csv`, 25,585 priced lines), the normal curve ran
  +10.84pp long on receiving yards, +7.53pp on rushing and +4.43pp on passing; gamma cut those to
  +1.46pp, -0.92pp and +0.69pp and improved Brier score in all six market-position buckets.
- `receptions` still uses the Gaussian normal with calibrated `sigma`. It is a count, and the
  walk-forward output carries no reception rows, so no replacement has been measured against it.
- Discrete binary count props (`anytime_touchdown`, line == 0.5) evaluate win probability using
  Poisson survival probability: \(P(\text{Over } 0.5) = 1.0 - e^{-\mu}\), avoiding distorted
  Gaussian tail probabilities near zero.
- A non-positive `sigma` prices as a point mass at `mu` (1.0 above the line, 0.0 at or below) rather
  than the NaN `scipy` returns for a zero scale. `compute_player_sigma` floors every bucket, so this
  guards direct callers and bad input, not the production path.

**Market Combination:** (`utils/market_blend.py`)
- The priced `mu` is the model's projection averaged with the mean the book's own price implies, at
  `config.betting.market_blend_weight` (0.5, env `NFL_MARKET_BLEND_WEIGHT`). Without it the engine
  never compares its mean to the market's, so a projection far from the line reads as a large edge
  whether the disagreement is information or model error. On the 2026 week-1 slate (12 games, 6
  books, 915 two-sided quotes matched to a projection) the unblended engine flagged 118 distinct
  bets at a 19.3% average edge; blending cut that to 76 at 14.0%.
- The book's price is converted to a *mean* before blending, by solving for the `mu` that makes
  `prob_over` reproduce the de-vigged probability at that line and sigma. A line sits near the
  median and `mu` is a mean, and for a right-skewed stat the mean is higher, so blending into the
  raw line biases every projection down: done that way the same slate came out 87% unders. On the
  mean it is 27 overs to 49 unders. The same slate's books implied a mean of 38.6 receiving yards
  against a median line of 29.5.
- Books post different lines for the same player, so the recovered means are pooled across books by
  median per (player, market). Pooling the lines themselves would compare different bets.
- 0.5 is the equal-weight baseline from the M4 and M5 forecasting competitions, not a fitted value.
  Fitting needs stored market lines sitting next to actuals. `clv_weekly` now holds 449 rows from
  2026 week 1, which is one week and not enough to fit a weight against. Revisit after a few more.
- `skew_is_reliable` screens out rows whose blended `mu` is at or below `sigma`. That puts the gamma
  shape under 1 and piles the curve against zero: on the 2025 walk-forward rows the actual cleared
  the gamma's own median 64.4% of the time where shape < 0.5 and 55.0% where shape was 0.5 to 1.0,
  against the 50% a correct median requires, and 23.9% of rows had `sigma >= mu`. The normal curve
  is wrong the other way in the same bucket (35.0%), so this is a reason to skip the row, not to
  switch curves. It screened 96 of 915 quotes on the 2026 week-1 slate.
- `model_has_skill` screens out a market whose projection carries no information that early in a
  season. Only `passing_yards` in weeks 1 to 4 qualifies today. On the 2025 walk-forward rows that
  bucket correlates 0.21 with the actual and its MAE (73.4) is worse than predicting the league mean
  (71.9), because the projections have a standard deviation of 32 yards against real passing days at
  94. From week 5 the same market correlates 0.51 and beats the flat baseline by 12.5%. Receiving and
  rushing correlate 0.45 to 0.63 in every week of 2025, week 1 included, so the screen names the one
  market rather than muting the card. It needs a `week` column on the quote frame; without one it
  passes every row and logs a warning. It screened 72 of 915 quotes on the 2026 week-1 slate, which
  removed the last 3 passing bets from the card.
- **Measured against the real 2026 week-1 card (all 16 games final, 447 graded bets).** Re-pricing
  that slate through this module and re-grading it: the shipped card returned +0.2% ROI, gamma
  re-pricing alone returned +0.8%, adding `skew_is_reliable` changed nothing (it screens 96 of 915
  quotes but none that reached the card), and adding `model_has_skill` returned **-6.6%**. The
  screen's own target market, `passing_yards`, was the only profitable one on the real card: 69
  bets, 72.5%, +35.9% ROI. The 2025-backtest correlation the screen is argued from did not predict
  that. Treat `model_has_skill` as refuted by out-of-sample data and do not wire it in; the case
  for removing it is stronger than the case for keeping it. One week is not proof, but it is the
  only real evidence either way, and it points against the screen.

- **Integration is not complete.** `value_betting_engine.rank_weekly_value` is gitignored and absent
  from public checkouts, so it must be changed on the deployment machine to call `blend_quotes` and
  price off `mu_blended` instead of `mu`, filtering on `skew_reliable` only. Until that lands, the
  module ships tested but unused. See `docs/DEPLOYMENT_MANIFEST.md`.

**QB Volume Calibration & Starter Gating:**
- Baseline starter pass attempts are calibrated to 31.0 attempts (from 34.0).
- Pass attempt volume incorporates game script using the canonical leading-reduces-volume convention:
  \(\text{script\_factor} = 1.0 - (\text{game\_script} \times 0.04)\), clamped to \([0.75, 1.25]\).
- Starter probability gating (\(p_{\text{start}}\)) is computed from depth chart rank and injury
  status (starter = 1.0; questionable = 0.70; doubtful = 0.25; out = 0.0; backup = 0.02). Non-starter
  expected attempts scale by \(p_{\text{start}}\), preventing backup QBs from inheriting starter volume.

**Empirical Role Priors:**
- Early-season receiver priors are calibrated empirically from multi-season snap percentages:
  Alpha (\(\ge 80\%\) snaps): 58.0 receiving yards; Secondary (\(\ge 60\%\)): 43.0;
  Slot (\(\ge 40\%\)): 30.0; Fringe (\(< 40\%\)): 10.0 yards. Overrides legacy uncalibrated priors
  (75/55/45/30).

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
