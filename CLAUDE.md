# Project Context: NFL Algorithm

Ground truth for anyone working in this repo. This is a **brownfield** Python project for NFL prop-bet projections and value betting.

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

Excluded from version control. This is the complete set — verify with
`git check-ignore -v <file>` rather than assuming.

| File | Purpose |
|------|---------|
| `config.py` | Optional local override for configuration. Not required: see the note below. |
| `data_pipeline.py` | Data ingestion, feature engineering, EWMA market mu computation |
| `value_betting_engine.py` | Kelly criterion, probability calculations, value ranking |
| `models/position_specific/weekly.py` | Weekly model training and prediction |

**Four files that are not on this list, though older versions of it said otherwise.**
`materialized_value_view.py` and `scripts/record_outcomes.py` are tracked with long commit
histories and never were gitignored. `api/server.py` and `prop_integration.py` were published on
2026-09-05: neither held modeling edge, and hiding the API kept 196 tests out of CI. See
`docs/plans/2026-09-05-003-open-the-public-contribution-boundary-plan.md`.

**Consequence for the gitignored set**: edits to those files live only on the local machine and in
no commit — a fresh clone gets whatever the deployment supplies. When a change spans a gitignored
module and a tracked one, only the tracked half reaches git. Say so explicitly instead of letting a
reviewer read the commit as the whole change.

Because CI has no access to these modules, logic that CI must verify belongs in a tracked module.
`utils/clv.py` exists for exactly this reason: `scripts/record_outcomes.py` can call it, and the
math stays testable without the private code.

---

## Key Configuration Values

A fresh clone reads these from tracked `config/runtime.py`. A gitignored top-level `config.py`
is an *optional* override: `config/__init__.py` loads it by path when present and its values
win, with `_fill_missing_settings` filling any gaps from the tracked defaults. So `import config`
always resolves to the tracked package, never to `config.py` directly, and a public clone runs
without that file. This is the pattern the other private modules should follow.

- `config.model.target_mae = 3.0` (professional-grade target)
- `config.betting.min_edge_threshold = 0.08` (8% minimum edge)
- `config.betting.min_confidence = 0.75`
- WR role priors: alpha=58, secondary=43, slot=30, fringe=10 (empirically calibrated)
- Minimum mu floor: 15.0

---

## Common Commands

```bash
# Install
make install

# Ingest data
make ingest-nfl

# Run tests
make test

# Weekly workflow (local/manual — see docs/OPERATIONS.md for the production run)
make week-predict SEASON=2025 WEEK=13
make week-materialize SEASON=2025 WEEK=13
make week-grade SEASON=2025 WEEK=13      # grades bets, records CLV

# Quality gate — non-zero exit when a position regresses past its MAE ceiling
make mae-gate SEASON=2025 WEEK=13

# Durable production path
make migrate
make doctor
make production-run SEASON=2026 WEEK=1
make pipeline-worker

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
                              value_betting_engine.py (Kelly + no-vig edge)
                                              |
                           materialized_value_view.py (dashboard layer)
                                              |
                             api/server.py -> React Dashboard

Grading loop (after results land):
  weekly_odds + materialized_value_view
        -> utils/clv.py (closing line, points + no-vig bp)
        -> scripts/record_outcomes.py
        -> bet_outcomes, clv_weekly, weekly_performance.clv_avg
```

This is the data-flow view. For the durable job/worker execution architecture that actually runs
production — FastAPI enqueues, a separate worker owns the fail-closed pipeline — see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/OPERATIONS.md](docs/OPERATIONS.md).
`make week-*` targets are the local/manual path and are **not** the production publication path.

---

## Data Status

Local `nfl_data.db` as of 2026-07-24 (`player_stats_enhanced`):

| Season | Rows | Players | Weeks |
|--------|------|---------|-------|
| 2023 | 6,167 | 604 | 1–22 |
| 2024 | 6,407 | 639 | 1–22 |
| 2025 | 6,558 | 663 | 1–21 |

Week numbers run past 18 because postseason weeks are included. Re-run the verification query
below rather than trusting these counts — they drift with every ingest.

**2026 season prep (as of 2026-08-03)**: the full 2026 schedule is loaded (`games`: 272 games, all
with `kickoff_utc`), 2026 rosters are in `nfl_roster_players` (2,924 players, 32 teams), week-1
context snapshots exist, and `make week-refresh SEASON=2026 WEEK=1` has produced 1,396 week-1
projections (872 players) with `team` populated on every row. Feeds keyed to the stats year
(weekly stats, injuries, weekly rosters) are unpublished until the season starts; the ingest skips
them with a warning instead of crashing (`_is_missing_feed_error` treats nflreadpy's season-range
`ValueError` as an unpublished feed for optional seasons only — history seasons still fail loud).

**Data Source**: All data ingested via `scripts/ingest_real_nfl_data.py` using nflverse/nflreadpy.

**Known gap**: `weekly_projections.team` is empty on legacy 2025 rows (546 of 568), so evaluation
joins to `games` find no kickoff for those weeks and `make mae-gate` fails loud with
`missing_kickoff` there. The current roster-backed prediction path populates `team` (2026 W1: 0
empty of 1,396), so the gate becomes verifiable on real data once 2026 actuals land. Do not
re-run `make week-refresh` for a past 2025 week to "fix" those rows — it would overwrite pregame
evidence (and the pre-kickoff guard refuses anyway).

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
- `tests/test_nfl_projection_evaluation.py` - evaluation metrics and the per-position MAE gate
- `tests/test_clv.py` - closing line value math (points and no-vig basis points)
- `tests/test_two_sided_odds.py` - over/under pairing at the same line (no-vig depends on it)
- `tests/test_odds_snapshot.py` - scraper quote to `weekly_odds` row, incl. `under_price`
- `tests/test_game_context.py` - schedule spread/total/weather extraction and its conventions, including the
  kickoff-aware closing definition
- `tests/test_event_keys.py` - odds → game key resolution; the contract the gitignored writers honor
- `tests/test_odds_quality.py` - screening unjoinable and circular snapshots out of grading
- `tests/test_kelly_cap.py` - Kelly fraction capping
- `tests/test_value_engine_side.py` - over/under side handling
- `tests/test_weekly_pipeline.py` - end-to-end ingest → train → predict → materialize. Seeds its
  own `games` rows: odds are keyed by game, so a club with no scheduled game gets no line and
  every later assertion would pass vacuously.

`tests/conftest.py` uses `collect_ignore` to skip tests that import gitignored modules, so the
suite runs in CI without the private code. Tests for logic CI must cover therefore need to import
from tracked modules only.

---

## Notes

- Database migrations are managed by `schema_migrations.py`. `_ensure_indexes` has a **MySQL branch
  that returns before the SQLite index list** — an index added to only one branch silently does not
  exist on the other. Add to both. `materialized_value_view` also has a near-duplicate `CREATE`
  inside `_rebuild_mvv_pk_if_needed`; schema changes must land in both copies.
- Most proprietary logic is gitignored, but not all — see the Proprietary Files section for the
  exact set and why it matters for CI.
- Use `make fullstack` for complete local development environment
- Front-end dashboard is in `/frontend` (Next.js + TypeScript)
- Legacy Streamlit dashboard available via `make dashboard`
- Further docs: [ARCHITECTURE](docs/ARCHITECTURE.md) (durable job pipeline),
  [OPERATIONS](docs/OPERATIONS.md) (weekly runbook),
  [TROUBLESHOOTING](docs/TROUBLESHOOTING.md), [MODEL_CARD](docs/MODEL_CARD.md)

---

## Active 2026-Prep Punch List

A 5-agent audit identified blockers and high-impact fixes for the 2026 season. Use this as the source of work when picking up a fix item.

### Tier 0 — BLOCKERS (broken code) — ALL RESOLVED
1. [RESOLVED] Snap counts dead merge — `scripts/ingest_real_nfl_data.py:100-111`. Real snap counts now merged; no `snap_percentage=50.0` fallback.
2. [RESOLVED] Auth endpoints all TypeError — `api/server.py` vs `api/auth.py`. Signatures aligned and password hashing moved to bcrypt.
3. [RESOLVED] `user_bets` INSERT broken — column list and placeholder count now match.
4. [RESOLVED] Side hardcoded "over" — over/under both supported end to end (`side` column on `materialized_value_view`).
5. [RESOLVED] Hardcoded season/week — `prop_integration.py` now requires explicit season/week.
6. [RESOLVED] Hardcoded fields — real `age` and `game_date` are ingested rather than defaulted.

### Tier 1 — HIGH IMPACT (MAE + ROI)
7. Premium features dropped in `_CONTEXTUAL_COLS` (weekly.py:44).
8. [RESOLVED] No vig removal — `implied_probability_no_vig` now lives in `value_betting_engine.py`
   and is what `utils/clv.py` uses for probability-space CLV. Both inputs are now actually
   captured: `utils/two_sided_odds.py` pairs an Over with the Under **at the same line** (the
   scraper previously matched on player name alone, which crosses alternate lines — a
   `DraftKings_Alt` book is already in the DB), and `utils/odds_snapshot.build_snapshot_row`
   carries `under_price` into storage. It had been NULL on every row because the writer in
   `data_pipeline._fetch_real_weekly_odds` built a row with `price` and no under column at all.
   Synthetic `SimBook` rows store NULL explicitly — there is no second side to de-vig against.
9. [RESOLVED] CLV never captured — `utils/clv.py` computes it; `scripts/record_outcomes.py` writes
   per-bet rows to `clv_weekly` and aggregates into `weekly_performance.clv_avg`. Closing line is
   now the **last snapshot at or before kickoff** when `resolve_closing_lines` is given a
   `kickoffs` frame; omitting it preserves the old `MAX(as_of)` behavior exactly. Degradation is
   per key, not wholesale — a key with no schedule row or an unparseable kickoff keeps
   `MAX(as_of)`, and a key whose every snapshot is post-kickoff yields no closing row rather than
   one graded off a stale in-game quote. A key with a single snapshot reports
   `insufficient_snapshots`, never a silent 0.

   **`weekly_odds.event_id` is now a real game key.** It previously held per-player strings
   (`2025_W22_NE_a_hooper`) and The Odds API's opaque provider ids, both of which joined to zero
   `games` rows — which is why kickoff was unreachable. `utils/event_keys.py` (tracked) mints the
   canonical nflverse form `{season}_{week:02d}_{away}_{home}`, and all three writers resolve
   through it; a row that cannot be tied to a game is dropped rather than stored under a key that
   looks joinable. `utils/odds_quality.py` screens the two disqualifiers — unjoinable keys and
   circular `SimBook` rows — out of the value/CLV path.

   **The 89 pre-existing snapshots are not backfillable and were deliberately left in place.**
   `describe_excluded` reports `{total: 89, unjoinable: 89, synthetic: 72, gradeable: 0}`. Week 10
   rows are the `alpha_receiver` test fixture; week 22 has zero scheduled games. `clv_weekly` is
   empty, so nothing was ever computed from them. Screened, not deleted.
10. [RESOLVED] NFL walk-forward backtest — `utils/nfl_backtest.py` (tracked, CI-tested with a stub
    model in `tests/test_nfl_backtest.py`) plus runner `scripts/run_nfl_backtest.py` and
    `make nfl-backtest SEASON=2025`. Retrains the production weekly model per week on history
    strictly before that week; model artifacts go to a temp dir and `_write_predictions` is
    no-oped, so production bundles and stored pregame evidence are untouched. The report includes
    `by_market_position` (the granularity `utils/nfl_sigma.py` buckets use) and `--rows-output`
    dumps the per-row scored frame for calibration analysis. A `compare` subcommand refuses to
    compare runs with mismatched scope. 2025 full-season baseline (5,117 predictions, 18/18 weeks,
    zero problems): overall MAE 26.88, bias +2.81. The item-25 mae-gate ceilings are now
    calibrated from this run's per-week worst MAE (see item 25); re-derive them from the latest
    `--rows-output` CSV whenever the model or slate changes materially.
11. [RESOLVED — by deletion] Universal model, no position split. Decision: the orphaned `RBModel` subclass was deleted rather than revived; `models/position_specific/weekly.py` is the single production model path. `BasePositionModel` is retained as the shared base. Revisit per-position splits as new work against weekly.py, not the old subclass.
12. [RESOLVED] nflreadpy sources unused — rosters, weekly rosters, schedules, depth charts,
    injuries, and pbp red-zone touches are all ingested by `scripts/ingest_real_nfl_data.py` and
    feed `games`, `nfl_roster_players`, and `nfl_player_context_snapshots`. The schedule's pregame
    context (`spread_line`, `total_line`, `temp`, `wind`, `roof`, `surface`, `div_game`) is
    extracted by `utils/game_context.py`, persisted on `games`, and attached directly to player
    frames in `models/position_specific/weekly.py` as static pregame features. Actual `game_script`
    is computed from pbp score differential or schedule margin (`_compute_actual_game_script`), and
    `expected_game_script` is computed from schedule `spread_line` — neither is hardcoded 0.0 any
    more. Touchdown columns are now kept through `transform_to_enhanced_stats` `final_cols`, so
    receptions and anytime touchdown markets are registered in `sports/markets.py`, modeled in
    `weekly.py`, priced (Poisson for anytime TD) in `value_betting_engine.py`, and graded in
    `utils/nfl_markets.py` / `scripts/record_outcomes.py`. Separately, `utils/context_factors.py`
    consumes `spread_line`/`total_line` as a game-script multiplier behind
    `NFL_FEATURE_CONTEXT_FACTORS` (`config/runtime.py:144`, set by `week-auto`) — see item 31 for
    its validation status. Still unused: FTN charting; pbp EPA.
13. [RESOLVED] Kelly cap in ranking path — enforced at both levels. Per-bet: gitignored
    `value_betting_engine.py:273` caps at `config.betting.max_kelly` (0.10) behind
    `config.features.kelly_cap_enabled` (`NFL_FEATURE_KELLY_CAP`, default ON per
    `config/runtime.py`). Portfolio: tracked `materialized_value_view.py:140` scales the whole
    card to the bankroll via `risk_manager.normalize_portfolio_stakes`, so persisted stakes
    never sum past `config.betting.bankroll` even when per-bet caps individually pass.
31. [RESOLVED] QB passing volume over-projection (diagnosed 2026-09 from the 2025 walk-forward rows CSV).
    The headline +13.2 passing_yards bias has been eliminated via: (a) QB baseline attempts calibrated
    from 34.0 to 31.0 in `data_pipeline.py` based on clear-starter actuals; (b) script factor updated to
    the canonical convention where leading suppresses attempts (`1.0 - game_script * 0.04`); and (c) dual-factor
    starter gating in `models/position_specific/weekly.py`, where backup QBs (`depth_rank > 1`) have expected
    attempts scaled by starter probability `p_start` (0.02 for healthy backups, 0.70 for questionable, etc.),
    preventing non-starters from inheriting starter attempt baselines.

### Tier 2 — MEDIUM (correctness/ops)
14. [RESOLVED] EWMA decay 0.65 / sigma calibration: the market-mu EWMA path in
    `DataPipeline._compute_market_mu` (`data_pipeline.py:74`) is **dead code in production** — it
    is exercised only by `tests/test_market_mu_wr.py`. Production mu is
    `model.predict() x defense_mult` (`weekly.py:1003-1015`); that path never touches EWMA at all.
    The live tuning surface was the **sigma** tables in `utils/nfl_sigma.py`, now recalibrated
    per (market, position) bucket from the 2025 walk-forward backtest (item 10's harness): each
    bucket deviating >= 2 SE from nominal 68.3% coverage got the 68.27th percentile of |z| as its
    EWMA multiplier, floors/defaults scaled by the same factor. Result: overall 1-sigma coverage
    66.3% -> 68.2%; QB passing (previously unvalidated, 58.2%) landed at 68.5%. Factors are
    in-sample for 2025 — validate against 2026 walk-forward before re-tuning, via
    `make nfl-backtest` + `compare`. The unread config knobs (`config.integration.ewma_decay` in
    both `config/runtime.py` and `config.py` — zero readers, confirmed by grep) have been
    removed; do not reintroduce a config knob for `_compute_market_mu`'s constant, since nothing
    in production reads it.
15. [RESOLVED — claim was stale] Defense multiplier double-apply: mu gets the multiplier exactly
    once, at `weekly.py:1003-1015`. The QB decomposition's `opp_factor` (`weekly.py:693-703`)
    writes only metadata columns, never mu — it does not double-apply anything. Do not "fix" this
    by removing the single legitimate application at 1003-1015.
16. [RESOLVED] SQLite/MySQL parity — upserts in `schema_migrations.py` and
    `scripts/record_outcomes.py` carry both dialects; `tests/test_grading_upsert_parity.py` and the
    CI MySQL matrix (`tests/test_pipeline_database_matrix.py`) hold the line.
17. [RESOLVED] SQLite WAL — `utils/db.py` sets `journal_mode=WAL` and `synchronous=NORMAL`;
    `make doctor` fails on a non-WAL database (`tests/test_db_pragmas.py`). No pool by design: the
    worker is the single writer.
18. [RESOLVED — by design] Migrations are forward-only and idempotent, gated on introspection
    rather than a version table; recovery from any partial run is `make migrate` again. Crash
    safety of the PK-widening rebuilds is covered by `tests/test_migration_crash_recovery.py`. See
    docs/OPERATIONS.md "Migrations: re-run forward, no rollback".
19. [Documented non-blocker] Opening vs closing line: opening is derivable as the first stored
    snapshot (`MIN(as_of)` per key, `scripts/backfill_line_accuracy.py:load_opening_lines`,
    consumed by `make backfill-accuracy`), bounded by scrape cadence rather than true market-open.
    CLV (`utils/clv.py`) intentionally grades entry-vs-close, not open-vs-close — that is by
    design, not a gap. `backfill_line_accuracy.py` compares `as_of` as SQLite TEXT (lexical, not
    chronological) — fine for consistently-formatted UTC timestamps but unlike `utils/clv.py` and
    `utils/matching.py`, which parse with pandas for true chronological ordering. Latent gap, not
    fixed here.
20. [RESOLVED] Tier-3 name-only match: position-guarded via `utils/matching.positions_compatible`
    at both tier-3 paths (`prop_integration.py:276-277`, `:315-316`) and the tier-3b fuzzy path;
    suffix handling is non-destructive (`utils/matching.py:91-132`, roster-verified tier 2.5 at
    `prop_integration.py:254-263`). Full truth table in `tests/test_matching.py`; end-to-end
    wiring covered by `tests/test_prop_integration_matching.py`
    (`test_join_blocks_cross_position_name_collision`,
    `test_join_blocks_suffix_conflict_but_matches_dropped_suffix`).
21. [RESOLVED] Stale-line filter: `filter_stale_snapshots` is wired via `utils/live_odds.select_live_odds`
    into `rank_weekly_value`; a schedule row with a null kickoff degrades per-key to
    unjoinable rather than crashing the run (`utils/live_odds.kickoffs_from_games`). It still filters
    nothing on the 89 legacy pre-existing rows, since none carry a joinable key.
22. [RESOLVED] `ALLOWED_ORIGINS` env drives CORS (preflight warns when unset);
    `api/rate_limit.py` token-bucket middleware (`RATE_LIMIT_*` in `config/runtime.py`); production
    worker count is explicit in the Makefile.
23. [RESOLVED] Structured logging (`LOG_FORMAT=json`), `utils/error_tracking.py`, and safe API
    error responses (`utils/api_exceptions.py`, `tests/test_api_exception_handler.py`).

### Tier 3 — LOWER (polish)
24. Property tests for Kelly/edge/vig math.
25. [RESOLVED] CI gate on per-position MAE — `check_position_mae` in
    `scripts/evaluate_nfl_projections.py`, exposed as the `mae-gate` subcommand and `make mae-gate`.
    Two ways to set the ceilings, and both survive in the merged code.
    **Absolute (default)**: QB 65.0, RB 26.0, WR 29.0, TE 27.0 — each ~10% above that position's
    worst single-week MAE in the 2025 walk-forward baseline (QB 59.5, RB 24.0, WR 26.0, TE 24.6
    from `reports/nfl_backtest_2025_baseline_rows.csv`), so a normal bad week passes and a broken
    model trips the gate. The original guessed ceilings (QB 18/RB 12/WR 12/TE 9) predated any
    measurement and would have failed every position on real data; they are gone.
    **Regression mode (optional)**: `make mae-gate SEASON=2026 WEEK=1 BASELINE=<walk-forward
    report> [TOLERANCE_PCT=10]` derives each ceiling from that baseline's per-position MAE plus the
    tolerance (`thresholds_from_backtest`), falling back to the absolute table for small-sample
    positions. Use it to catch drift against a specific run rather than against a fixed number.
    A position under 30 projections is reported as skipped, never silently passed. CI runs the
    gate's unit tests only (no projection data in CI); the real-data run is the Makefile target.
    See the Data Status note — the real-data path currently fails on `missing_kickoff`.
26. Perf regression budgets.
27. Stacking final estimator Ridge → LightGBM or isotonic calibration.
28. [RESOLVED] WR role priors stale — recalibrated in `utils/season_priors.py`, `config.py`,
    `config/runtime.py`, and `data_pipeline.py` using empirical 2023–2025 receiving yard distributions
    across snap percentage tiers: alpha (>=80% snaps) = 58.0 yards (down from 75.0); secondary (>=60%) =
    43.0 (down from 55.0); slot (>=40%) = 30.0 (down from 45.0); fringe (<40%) = 10.0 (down from 30.0).
29. Cache stale-while-revalidate no in-flight dedup.
30. [RESOLVED] `materialized_value_view` ranking index — composite `(season, week, edge_percentage)` on `idx_materialized_value_view_lookup`, created on both the SQLite and MySQL branches of `_ensure_indexes`. It supersedes the former `(season, week)` index.

### Season-start checks (added 2026-09-02)
31. **Context factors are unvalidated but switched on by the cron.** `utils/context_factors.py`
    (game script, matchup history, usage trend; composite clipped to [0.85, 1.15]) was committed
    with its flag defaulting OFF "until a walk-forward backtest validates it" — but `make week-auto`
    sets `NFL_FEATURE_CONTEXT_FACTORS=1` unconditionally. No backtest has measured it. Run, on the
    machine that has the private model:
    `make nfl-backtest SEASON=2025 CONTEXT_FACTORS=off OUTPUT=logs/metrics/bt-2025-off.json` and
    the same with `CONTEXT_FACTORS=on LABEL=ctx OUTPUT=logs/metrics/bt-2025-on.json`, then
    `uv run python -m scripts.run_nfl_backtest compare logs/metrics/bt-2025-off.json
    logs/metrics/bt-2025-on.json`. If MAE does not improve, drop the `NFL_FEATURE_CONTEXT_FACTORS=1`
    from `week-auto`. The `--off` report is also the `BASELINE` for item 25's gate.
32. **The private wiring for context factors is not in `docs/DEPLOYMENT_MANIFEST.md`'s verified
    set.** What the tracked code expects of `weekly.py` is recorded there now; verify it on the
    production checkout before week 1.
