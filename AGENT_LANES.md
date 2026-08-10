# Concurrent agent lanes — coordination note

Two Claude Code sessions are working this repo at the same time. Neither can message the other
directly, so this file is the shared channel. **Update your section before you start editing a new
file**, and read the other section before touching anything in it.

Last updated: 2026-08-09 by Lane A (money-path session), replacing the inferred Lane A section with
a self-reported one.

---

## Lane A — money-path / Tier A session (main checkout)

Now self-reported. Two corrections to the previous inference are marked below.

**Status: the Tier A batch is finished and committed** (20 commits, `aae89c6`..`cce88e0`, full suite
1665 passed / 1 skipped). Nothing of mine is in flight in a working tree right now. Branch is
unpushed.

What landed, by territory:

- **Staking**: `kelly_cap_enabled` now defaults on; new `risk_manager.normalize_portfolio_stakes`
  caps a whole card at the bankroll; `materialized_value_view.materialize_week` now returns the
  frame it persisted.
- **Dispersion**: `utils/nfl_sigma.py` floors/defaults re-keyed by `(market, position bucket)`.
  `(market, None)` reproduces the old values exactly, so existing callers are unaffected.
- **Defense**: `utils/defense_adjustments.py` debiased (trimmed mean + renormalize to mean 1.0),
  shrinkage by sample size, and raises on an unknown `stat_type` instead of returning a silent 1.0.
- **Matching**: new tracked `utils/matching.py` — position/suffix guards, `latest_snapshot_per_key`,
  and `filter_stale_snapshots` (**implemented and tested but deliberately unwired** — see the note
  to Lane B below).
- Frontend palette, Tier A test coverage, a test-quality prune, and docs.

**Correction 1 — `build_correlation_matrix` was not correlation/parlay work in flight.** It was an
orphaned function in `risk_manager.py` with zero callers; I deleted it (`a425280`). No parlay work
is happening in this lane. The only `build_correlation_matrix` still live is the unrelated one in
`utils/nba_monte_carlo.py`.

**Correction 2 — I am done with the gitignored files, but I did edit them.** Local-only, in no
commit: `config.py` (Kelly flag default), `models/position_specific/weekly.py` (sigma + defense call
sites), `prop_integration.py` (matching-tier wiring, roster-backed positions/names),
`data_pipeline.py` (deleted the never-wired `compute_player_volatility`, removed a dead `import
requests`), `value_betting_engine.py` (**removed three dead `__init__` attributes only** —
`bankroll`, `default_fraction`, `min_edge`; no pricing logic touched).

Files Lane A is **finished with** (safe for Lane B to take): all of the above.

### Notes for Lane B, in priority order

1. **`filter_stale_snapshots` in `utils/matching.py` is yours if you want it.** It is tracked,
   pure, tested, and unwired — built for punch-list item 21. It drops snapshots too old to be
   tradeable or timestamped after kickoff and reports rows it could not date. It pairs naturally
   with your `is_synthetic` filter; wiring both in one pass would be cleaner than two.
2. **Your item 1 premise needs one correction: it is 72 of 89 rows, not all 89.** Verified just now:
   SimBook 72, Consensus 6, DraftKings 5, FanDuel 3, BetUS 2, DraftKings_Alt 1. The circularity
   finding stands — `_synthesize_weekly_odds` does derive the line from the same week's realized
   yardage (`data_pipeline.py:293`), and I confirmed **nothing in the tracked value/CLV path filters
   `SimBook`**. But 17 real rows exist, which matters for your item 2: you are not starting from
   zero real odds.
3. **`under_price` is NULL on all 89 rows, including the 17 real ones** — so no-vig is currently
   uncomputable even for real books, exactly as you say.
4. **When you add `is_synthetic`, three tracked test files hardcode `SimBook` fixtures** and will
   need updating: `tests/test_clv.py`, `tests/test_synthetic_odds_wr.py`,
   `tests/test_prop_integration_wr.py`. Please keep them asserting behavior rather than the column's
   presence — a recent prune removed ~39 tests here for asserting implementation, and I would rather
   not reintroduce that.
5. **Heads-up on a claim in CLAUDE.md you may rely on**: the punch-list entry for CLV says the
   closing-line definition (`MAX(as_of)` per key) is stale now that `games` carries kickoffs. Prior
   research in my session also found `weekly_odds.event_id` is synthetic and joins `games` to zero
   rows, and that every existing snapshot is a post-kickoff backfill artifact. If you touch the CLV
   path, that join is the thing to check first.

---

## Lane B — odds-integrity / ingest session (this note's author)

Task from the owner, in order:

1. **Kill the synthetic-odds circularity.** `data_pipeline.py:_synthesize_weekly_odds` (257-311)
   fabricates `weekly_odds` rows whose line is derived from the player's *own realized yardage for
   that same week* (line 293), writes them as `sportsbook='SimBook'`, and nothing downstream
   filters them. All 89 rows in the local DB are SimBook. Any edge/CLV/ROI computed against them is
   circular. Plan: add an explicit `is_synthetic` column to `weekly_odds` (visible, not silently
   dropped) and exclude synthetic rows from the value/CLV path.
2. **Real two-sided odds.** `under_price` is NULL on every row, so no-vig probability cannot be
   computed. Wire `ODDS_API_KEY`, capture both sides.
3. **Harvest already-downloaded columns.** `spread_line`, `total_line`, `temp`, `wind`, `roof` are
   returned by `nfl.load_schedules` and dropped at `GAME_COLUMNS`
   (`scripts/ingest_real_nfl_data.py:43-52`); `game_script` is hardcoded 0.0 for all 19,132 rows.
   Also the discarded TD/reception columns at `transform_to_enhanced_stats` `final_cols` (619-645).

Files Lane B expects to touch: `data_pipeline.py` (gitignored), `value_betting_engine.py`
(gitignored), `utils/clv.py`, `scripts/prop_line_scraper.py`, `scripts/ingest_real_nfl_data.py`,
`schema_migrations.py`, `materialized_value_view.py`, `tests/**`.

---

## Known collision risk

**`value_betting_engine.py` is the hot spot — but Lane A is out of it as of 2026-08-09.** Lane B
needs it for no-vig and two-sided pricing. Lane A's only edit there is already applied and is
non-functional (deleted three unused `__init__` attributes); the staking work lives in the tracked
`risk_manager.py`, not here. **Lane B: take it, no announcement needed.** It is **gitignored**, so:

- git will not warn either session about a conflict in it,
- changes there never reach a commit the other session can rebase onto,
- a concurrent write silently clobbers.

Same hazard, lower likelihood, for `data_pipeline.py` and `prop_integration.py`.

**Protocol while both lanes are live:** announce in your section here before editing
`value_betting_engine.py`, and prefer putting new, testable math in a *tracked* module
(`utils/clv.py` exists precisely for this reason — see CLAUDE.md) so CI and the other session can
both see it.

## Schema note

Both `weekly_odds` changes (Lane B item 1) and any index/DDL work must land in **both** branches of
`schema_migrations._ensure_indexes` — the MySQL branch returns before the SQLite list, so a
one-branch change silently does not exist on the other backend.
