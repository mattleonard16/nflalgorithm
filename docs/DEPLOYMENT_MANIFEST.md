# Deployment manifest — changes that are NOT in git

Some production modules are gitignored (see the Proprietary Files table in `CLAUDE.md`). Changes to
them live on one machine and reach no commit, so **a fresh clone plus a deployment copy of the
private modules will silently lack them**. Git will not warn anyone.

This file records what the tracked code now *expects* the private modules to do. When deploying, or
when the private modules are restored from a backup or another machine, verify each item below and
re-apply anything missing.

Last verified: 2026-08-16, including live-odds selection, kickoff-aware production CLV, and season
priors. The context-factors section below was added 2026-09-02 from the tracked side only and is
unverified.

The market-mean blend section under `value_betting_engine.py` was added 2026-09-12, also from
the tracked side only, and is unverified.

`api/server.py` and `prop_integration.py` left this file on 2026-09-05. Both are tracked now, so
git verifies them and nothing here needs to.

---

## Why this matters

Most fixes below are **split across the git boundary**. The tracked half is committed; the wiring
that activates it is not. A deploy that takes only the committed half gets the new code sitting
inert, with no error and no warning, just the old behavior. The last column says what breaks when
the private half is missing, so start there.

| Fix | Tracked (in git) | Local-only (NOT in git) | Effect if the private half is missing |
|---|---|---|---|
| Kelly cap default | `config/runtime.py` default `True` | `config.py:193` `_flag("NFL_FEATURE_KELLY_CAP", True)` | `config.py` wins at runtime. An older copy defaults **False**, so per-bet Kelly capping is **off** and full-Kelly fractions (measured up to 0.743) size the bets. |
| Portfolio stake cap | `risk_manager.normalize_portfolio_stakes`, called from `materialized_value_view.py:105` | — (fully tracked) | None. This one is safe. |
| Position-keyed sigma | `utils/nfl_sigma.py` | `weekly.py:977` passes `position=position` | Falls back to the `(market, None)` legacy floors. Dispersion silently reverts to the old miscalibrated values (WR/TE rushing ~2.5x too wide). |
| Live-odds stale filter | `utils/live_odds.py` | `value_betting_engine.rank_weekly_value` | Ranking takes SQL `MAX(as_of)` and can price an in-game quote. |
| Market-mean blend | `utils/market_blend.py`, `config/runtime.py` (`betting.market_blend_weight`) | `value_betting_engine.rank_weekly_value` | Prices off the raw model `mu`, so a projection far from the line reads as a large edge. Measured on the 2026 W1 slate: 118 flagged bets at 19.3% average edge instead of 68 at 14.1%. |
| Kickoff-aware production CLV | `utils/clv.py`, `utils/live_odds.py` | `scripts/record_outcomes.py` `compute_and_save_clv` | Closing line is `MAX(as_of)`, including post-kickoff scrapes. |
| Early-season 70/30 role prior | `utils/season_priors.py` | `weekly.py` `_engineer_rolling_features` and `get_nfl_feature_cols` | Week 1 expected_* stays last-6 EWM; last_season_*_pg features are missing so a restored private weekly.py ignores the new helper. |

## Required state of each private module

### `config.py`
```python
kelly_cap_enabled=_flag("NFL_FEATURE_KELLY_CAP", True),   # default ON
```
Verify: `uv run python -c "from config import config; print(config.features.kelly_cap_enabled)"`
must print `True` with no env var set.

### `models/position_specific/weekly.py`
- Imports `compute_player_sigma` from `utils.nfl_sigma` and `get_defense_multiplier` from
  `utils.defense_adjustments` (module level, ~lines 24-25).
- The sigma call passes position:
  `compute_player_sigma(history, market=market, position=position) * uncertainty_multiplier`
- The per-row defense multiplier call passes `position=position`.
- **Writes `volatility_score`.** Imports `volatility_score_or_none` from `utils.volatility_scoring`,
  calls it on the same per-player weekly series the sigma uses, and carries the result through
  `_write_predictions` into the `weekly_projections` INSERT — column list, placeholder count, and
  **both** the SQLite `ON CONFLICT` and MySQL `ON DUPLICATE KEY` clauses. The value is passed
  through `_optional_float` so "not measured" survives as NULL rather than becoming NaN.

  Note the deliberate asymmetry: `volatility_score` is **not** in the
  `for col in (...)` default loop next to it, because that loop fills 0.0, and 0.0 means "measured
  as perfectly steady" — a claim — where None means "not measured".

  If this write is missing, the column is all-NULL, `apply_volatility_widening` reports every row
  unscored, and sigma widening silently does nothing. That is safer than the old behavior it
  replaced, but the feature is inert.

Verify: `command grep -n "position=position" models/position_specific/weekly.py` returns the sigma
call site, and after a predict run:

```bash
DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python -c "
from utils.db import read_dataframe
print(read_dataframe('SELECT COUNT(*) n, COUNT(volatility_score) scored FROM weekly_projections WHERE season=2026 AND week=1'))
"
```
`scored` must be non-zero. Measured on the 2026 W1 slate: 784 of 1396 scored, widening multiplier
spanning 1.0000-1.1500 across 472 distinct values (it was a constant 1.075 on every row before).

### `value_betting_engine.py`
- The per-bet Kelly cap is gated on `config.features.kelly_cap_enabled` (~line 266).
- Sigma widening calls `apply_volatility_widening(df["sigma"], df.get("volatility_score"))` from
  `utils.volatility_scoring` and logs the unscored row count. **An older copy uses
  `fillna(50.0)` + `widen_sigma_for_volatility`, which inflates every NFL sigma by a flat 7.5%
  because `volatility_score` is never written (0 of 1964 rows).** This is the one item here whose
  absence silently changes every price on the card, so check it first.
- Three unused `__init__` attributes (`bankroll`, `default_fraction`, `min_edge`) were removed. This
  is cosmetic; their presence breaks nothing.
- `rank_weekly_value` loads raw `weekly_odds`, applies `utils.live_odds.select_live_odds` with
  `kickoffs_from_games`, then joins projections. It must **not** take SQL `MAX(as_of)` before the
  stale filter — that would keep a post-kickoff scrape and drop the last live line.
- **Market-mean blend, added 2026-09-12. Tracked side only, NOT yet applied to a private
  checkout.** `rank_weekly_value` must route its quote frame through
  `utils.market_blend.blend_quotes` before pricing. Contract:
  1. Build the frame it already builds (live odds joined to projections), and add a
     `p_market_over` column holding the **de-vigged** over probability, from
     `implied_probability_no_vig(price, under_price)` in this same module. A raw implied
     probability carries the book's margin and pushes the recovered mean up.
  2. Call `blend_quotes(frame)`. Required columns are `player_id`, `market`, `line`, `mu`,
     `sigma`, `p_market_over`. It returns a copy with `market_mean`,
     `market_mean_consensus`, `mu_blended` and `skew_reliable` added.
  3. Price off `mu_blended`, not `mu`, in every `prob_over` call and in anything that
     reports the projection next to the line.
  4. Drop rows where `mu_blended` is null (the book's price implied no positive mean) and
     rows where `skew_reliable` is false (gamma shape under 1, where the curve's median was
     measured 5 to 14 points off). On the 2026 W1 slate the skew screen dropped 96 of 915
     quotes, none of which had reached the graded card.
  5. **Do not filter on `model_skilled`.** The function exists and is tested, but re-grading
     the real 2026 week-1 card with it applied returned -6.6% ROI against +0.8% for the blend
     alone, and the market it screens (`passing_yards`, weeks 1 to 4) was the only profitable
     market on that card at +35.9% ROI over 69 bets. It is argued from 2025 backtest
     correlation that the first week of real results contradicted. Leave it unwired pending
     more weeks, or delete it. See `docs/MODEL_CARD.md` under **Market Combination**.
  The weight lives in `config.betting.market_blend_weight` (0.5, env `NFL_MARKET_BLEND_WEIGHT`)
  and needs no `config.py` change: `_fill_missing_settings` fills tracked defaults into the
  private override. Reasoning and measurements are in `docs/MODEL_CARD.md` under
  **Market Combination**.

Verify: `command grep -n "apply_volatility_widening" value_betting_engine.py` returns the call site,
and `command grep -n "fillna(50.0)" value_betting_engine.py` returns nothing.
`command grep -n "select_live_odds" value_betting_engine.py` must return the ranking call site.
`command grep -n "blend_quotes\|mu_blended\|skew_reliable" value_betting_engine.py` must return
the blend call site and the pricing and filter uses; no output means the card is still built off
the raw `mu`. `command grep -n "model_skilled" value_betting_engine.py` must return nothing, per
step 5 above.

### `scripts/record_outcomes.py`
- `compute_and_save_clv` loads `games.game_id` / `kickoff_utc` for the week and passes
  `kickoffs_from_games(...)` into `resolve_closing_lines`. An older copy calls
  `resolve_closing_lines(odds)` with no kickoffs, so CLV grades `MAX(as_of)` including in-game quotes.

Verify: `command grep -n "kickoffs_from_games" scripts/record_outcomes.py` returns the CLV call site.

### `data_pipeline.py`
- `compute_player_volatility` deleted (was never wired to anything).
- The unused `import requests` removed. **If restoring an older copy, the import may return — it is
  harmless, but `tests/test_basic.py` no longer patches it.**

### Context factors (`NFL_FEATURE_CONTEXT_FACTORS`) — added 2026-09-02, NOT yet verified on a production checkout

Tracked: `utils/context_factors.py`, `config/runtime.py` (`features.context_factors_enabled`, default
OFF). The Wednesday cron (`make week-auto`) sets `NFL_FEATURE_CONTEXT_FACTORS=1`, so the private
half must exist for that to do anything:

- `config.py` mirrors the flag:
  `context_factors_enabled=_flag("NFL_FEATURE_CONTEXT_FACTORS", False)`.
- `models/position_specific/weekly.py`, at the mu site (the same block that applies the defense
  multiplier once, ~lines 1003-1015), when `config.features.context_factors_enabled` is true: call
  `utils.context_factors.context_factor_lookup(season, week, market, players=<the frame being
  predicted>)` **once per market**, then multiply each row's mu by `lookup.get(player_id, 1.0)`.
  The 1.0 default is the contract — an unknown player is never a silent zero.
- It must **not** add its own opponent-strength term; the module deliberately omits defense
  because mu already carries `get_defense_multiplier` exactly once.

Verify:
```bash
command grep -n "context_factor_lookup\|context_factors_enabled" models/position_specific/weekly.py config.py
```
Both files must match. If `weekly.py` has no call site, the flag is inert: the cron's projections
are unadjusted and `make nfl-backtest CONTEXT_FACTORS=on` measures nothing (its report will still
say `"features": {"context_factors_enabled": true}` — that records the flag, not the wiring).

### Trained model artifacts

`models/weekly/{passing,receiving,rushing}_yards_model.joblib` are **also gitignored** (`.gitignore`
line 102, `*.joblib`). They are not code, but the same reasoning applies: a fresh deploy has no
models until something trains them.

Local artifacts are dated 2026-08-03. Confirm before Week 1 whether a deploy ships these files or
retrains on arrival, and whether the featureset hash stored in the artifact still matches what the
current code computes — a mismatch forces a retrain, which is a batch job, not something to
discover on game day.

---

## Note on `command grep`

Plain `grep` in some agent environments respects `.gitignore` and will report zero matches inside
these files even when the code is present. Use `command grep -rn ... . --include="*.py"` for any
check that has to be trustworthy.

## Standing recommendation

Every item above exists because logic lives in a module CI cannot see. The durable fix is to keep
moving verifiable math into tracked modules (`utils/clv.py`, `utils/matching.py`,
`utils/nfl_sigma.py`, `utils/defense_adjustments.py`, `utils/live_odds.py` are all precedents) and leave only thin wiring
in the private ones. The thinner the private layer, the smaller this file gets.
