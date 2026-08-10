# Deployment manifest — changes that are NOT in git

Some production modules are gitignored (see the Proprietary Files table in `CLAUDE.md`). Changes to
them live on one machine and reach no commit, so **a fresh clone plus a deployment copy of the
private modules will silently lack them**. Git will not warn anyone.

This file records what the tracked code now *expects* the private modules to do. When deploying, or
when the private modules are restored from a backup or another machine, verify each item below and
re-apply anything missing.

Last verified: 2026-08-09, against the Tier A money-path work (`aae89c6`..`cce88e0`).

---

## Why this matters

Tier A shipped four money-path fixes. Three of them are **split across the git boundary**: the
tracked half is committed, the wiring that activates it is not. A deploy that takes only the
committed half gets the new modules sitting inert — no error, no warning, just the old behavior.

| Fix | Tracked (in git) | Local-only (NOT in git) | Effect if the private half is missing |
|---|---|---|---|
| Kelly cap default | `config/runtime.py` default `True` | `config.py:193` `_flag("NFL_FEATURE_KELLY_CAP", True)` | `config.py` wins at runtime. An older copy defaults **False**, so per-bet Kelly capping is **off** and full-Kelly fractions (measured up to 0.743) size the bets. |
| Portfolio stake cap | `risk_manager.normalize_portfolio_stakes`, called from `materialized_value_view.py:105` | — (fully tracked) | None. This one is safe. |
| Position-keyed sigma | `utils/nfl_sigma.py` | `weekly.py:977` passes `position=position` | Falls back to the `(market, None)` legacy floors. Dispersion silently reverts to the old miscalibrated values (WR/TE rushing ~2.5x too wide). |
| Matching guards | `utils/matching.py` | `prop_integration.py:36-42` imports and wiring | Guards never run. Cross-position name collisions (QB/DB Lamar Jackson, WR/LB Justin Jefferson) merge again, and stale odds snapshots fan out into duplicate rows. |

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

Verify: `command grep -n "position=position" models/position_specific/weekly.py` returns the sigma
call site.

### `prop_integration.py`
- Imports `latest_snapshot_per_key`, `name_variants`, `positions_compatible`, `strip_name_suffix`,
  `suffix_conflict` from `utils.matching`.
- Dedups odds via `latest_snapshot_per_key` at load.
- Sources positions and raw names from `nfl_roster_players` (NOT `player_stats_enhanced`, which is
  empty pregame and contains no defensive players — the guards are inert if it is used).

Verify: `command grep -n "from utils.matching import" prop_integration.py`

### `value_betting_engine.py`
- The per-bet Kelly cap is gated on `config.features.kelly_cap_enabled` (~line 266).
- Sigma widening calls `apply_volatility_widening(df["sigma"], df.get("volatility_score"))` from
  `utils.volatility_scoring` and logs the unscored row count. **An older copy uses
  `fillna(50.0)` + `widen_sigma_for_volatility`, which inflates every NFL sigma by a flat 7.5%
  because `volatility_score` is never written (0 of 1964 rows).** This is the one item here whose
  absence silently changes every price on the card, so check it first.
- Three unused `__init__` attributes (`bankroll`, `default_fraction`, `min_edge`) were removed. This
  is cosmetic; their presence breaks nothing.

Verify: `command grep -n "apply_volatility_widening" value_betting_engine.py` returns the call site,
and `command grep -n "fillna(50.0)" value_betting_engine.py` returns nothing.

### `data_pipeline.py`
- `compute_player_volatility` deleted (was never wired to anything).
- The unused `import requests` removed. **If restoring an older copy, the import may return — it is
  harmless, but `tests/test_basic.py` no longer patches it.**

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
`utils/nfl_sigma.py`, `utils/defense_adjustments.py` are all precedents) and leave only thin wiring
in the private ones. The thinner the private layer, the smaller this file gets.
