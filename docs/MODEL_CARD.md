# Weekly Model Card

**Model family:** Ridge regression per-market with standardized feature stack.

**Training data:** Synthesized weekly feature frames built from `DataPipeline.compute_week_feature_frame` across a rolling list of `(season, week)` tuples. Features include odds line, targets, rolling usage, team ranks, weather flags, and injury indicators.

**Targets:** Market-specific mean projection (`mu_prior`) for rushing, receiving, and passing yards.

**Versioning:** Model artifacts saved under `models/weekly/<market>_model.joblib` with metadata:
- `model_version`: UTC timestamp string (`weekly-YYYYMMDDTHHMMSSZ`)
- `feature_columns`: columns used during fit
- `sigma_default`: residual standard deviation for baseline uncertainty

**Inference pipeline:**
1. Ensure `make week-update` and `make week-predict` have run for desired week.
2. Call `models.position_specific.predict_week(season, week)` to emit `weekly_projections` rows.
3. Output includes `mu`, `sigma`, `model_version`, and `featureset_hash` for reproducibility.

**Calibration:** Residual standard deviation drives `sigma_default`; live predictions clamp `sigma` to `max(sigma_default, |mu| * 0.3)`.

**Known limitations:**
- Synthetic training data bootstrapped from season priors; replace with historical stats when available.
- Passing yard market infers mean via rolling air yards proxy.
- Injury and weather feeds default to heuristics when upstream data missing.
