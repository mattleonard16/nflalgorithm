# Concurrent agent lanes — coordination note

Two Claude Code sessions are working this repo at the same time. Neither can message the other
directly, so this file is the shared channel. **Update your section before you start editing a new
file**, and read the other section before touching anything in it.

Last updated: 2026-08-09 by the odds/ingest session.

---

## Lane A — "Fable Advisor NFL/NBA Roadmap" session (main checkout)

Inferred from commit history, not self-reported. Correct this section if it is wrong.

Recent commits on `improvement-cycle-clv-mae-gate`:

- `cce88e0` docs(model-card): record the sigma, defense and staking changes
- `cda3646` docs: refresh commands, architecture and the 2026-prep punch list
- `bea0c68` docs: correct the proprietary-files list and spell out its consequences
- `50efd85` refactor(sigma): delete get_sigma_or_default
- `9a07865` test(defense): assert hand-computed multipliers
- `7374298` feat(matching): guard player matching by position, suffix and snapshot age
- `2b8e58a` fix(sigma): key dispersion floors by position, not market alone
- `126e01a` fix(defense): debias, shrink and fail loud on defense multipliers

Apparent territory: sigma/dispersion floors, defense multipliers, player matching guards, frontend
palette, Tier A test coverage, model card. In-flight at the time of writing: something involving an
orphaned `build_correlation_matrix` (correlation/parlay work).

Files believed in use by Lane A: `models/position_specific/weekly.py` (gitignored),
`prop_integration.py` (gitignored), `docs/MODEL_CARD.md`, `frontend/**`, `tests/**`.

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

**`value_betting_engine.py` is the hot spot.** Lane B needs it for no-vig and two-sided pricing;
Lane A's staking/defense work may also touch it. It is **gitignored**, so:

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
