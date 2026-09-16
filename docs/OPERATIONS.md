# Weekly Operations

## Startup

```bash
make migrate
make doctor
make fullstack
```

`make fullstack` supervises the worker, API, and frontend. It waits for `/readyz`; `/livez` only verifies that the API process is running. Use `LOG_FORMAT=json` in deployments.

Production MySQL must be Oracle MySQL 8.0 or newer. Startup rejects MySQL 5.7 and MariaDB rather
than silently weakening `SKIP LOCKED` claim and fencing behavior.

The worker is the only process allowed to execute stages or publish cards. A heartbeat exception or
zero-row renewal means lease loss; the production worker exits immediately with code 75, including
while a non-cooperative stage handler is running. The service supervisor must restart it. Failed
attempt history is append-only and visible through the authenticated pipeline API and metrics.

Automatic retries are fail-closed. Every executed stage must declare `retry_safe=true`; an unknown
runner exception or undeclared side effect becomes terminal so the worker cannot repeat an external
publication whose acknowledgement may have been lost. Crash recovery applies the same policy to
persisted stage history: empty history or any unaudited stage blocks automatic requeue and records
the reason on the job, run, and interrupted stage.

1. Run migrations and preflight after pulling updates:
   ```bash
   make migrate
   make doctor
   ```
2. Confirm 2024-2025 history is team-scoped before live odds exist:
   ```bash
   make ingest-nfl NFL_SEASONS=2024,2025 THROUGH_WEEK=22
   make doctor-preseason SEASON=2026 WEEK=1
   ```
3. Update a specific NFL week (idempotent upserts):
   ```bash
   make week-refresh SEASON=2026 WEEK=1
   make doctor-season SEASON=2026 WEEK=1
   ```
4. Train or refresh rolling models if needed:
   ```bash
   make nfl-train
   ```
5. Queue the durable production run and let the worker execute it:
   ```bash
   make production-run SEASON=2026 WEEK=1
   make pipeline-worker-once
   ```
6. Verify persisted worker evidence and execute sanity checks:
   ```bash
   make doctor-season SEASON=2026 WEEK=1 SEASON_PHASE=post-run
   make health SEASON=2026 WEEK=1
   ```
7. Launch dashboard and monitor feeds:
   ```bash
   make dashboard
   ```

## Week 1 go-live checklist (2026)

Run on the production checkout, in this order. Everything before the first arrow is tracked and
was rehearsed on a fresh clone on 2026-09-02 (migrate, doctor, frontend build); everything after
needs the private modules and real data.

1. `make migrate && make doctor` — 47 tables, WAL on, no `[FAIL]` rows. `private_modules`
   must PASS here, unlike on a tracked-only clone.
2. Confirm the private wiring listed in `docs/DEPLOYMENT_MANIFEST.md`, especially the
   context-factors section (added 2026-09-02, unverified) — the cron turns that flag on.
3. `make ingest-nfl NFL_SEASONS=2024,2025 THROUGH_WEEK=22` then
   `make doctor-preseason SEASON=2026 WEEK=1`.
4. Validate context factors before the cron uses them (CLAUDE.md item 31):
   ```bash
   make nfl-backtest SEASON=2025 CONTEXT_FACTORS=off OUTPUT=logs/metrics/bt-2025-off.json
   make nfl-backtest SEASON=2025 CONTEXT_FACTORS=on LABEL=ctx OUTPUT=logs/metrics/bt-2025-on.json
   uv run python -m scripts.run_nfl_backtest compare logs/metrics/bt-2025-off.json logs/metrics/bt-2025-on.json
   ```
   No improvement means remove `NFL_FEATURE_CONTEXT_FACTORS=1` from `week-auto` before Wednesday.
5. `make week-refresh SEASON=2026 WEEK=1` and `make doctor-season SEASON=2026 WEEK=1`.
6. After week 1 results land, pull the actuals and then grade:
   ```bash
   make ingest-nfl NFL_SEASONS=2026 THROUGH_WEEK=1
   make week-grade SEASON=2026 WEEK=1
   make mae-gate SEASON=2026 WEEK=1 BASELINE=logs/metrics/bt-2025-off.json
   ```
   The ingest has to run first. Grading reads `player_stats_enhanced`, and week 1 is empty there
   until nflverse publishes and the ingest pulls it.

   Watch the `Matched N of M stat rows to bets` line. A 0 there means grading is about to
   record the whole week as pushes. `player_stats_enhanced` and `nfl_roster_players` mint
   `player_id` from different spellings of the same name, so `utils.grading.align_actuals_to_bets`
   re-keys the stats onto the bets' ids through nflverse's `gsis_id`; a zero match means the
   roster is missing for that season.

   Grading is safe to run mid-week and safe to re-run. `utils/game_completion.classify_games`
   settles only games whose kickoff is at least four hours past *and* whose stats have published;
   everything else is listed as pending with a reason and skipped. Re-running after the rest of
   the slate lands grades the new games and overwrites the old rows in place, since `bet_id` is a
   hash of the bet's natural key. A week that spans Thursday to Monday therefore takes two or three
   runs, which is expected rather than a failure.

   Before the gate existed, a mid-week run marked every unplayed bet as a `push` with zero profit,
   which is indistinguishable from a settled push once it reaches `bet_outcomes` and
   `weekly_performance.clv_avg`. `--include-unfinished` restores the old behavior and exists only
   for backfilling a historical week whose schedule rows are incomplete.

   `make mae-gate` without `BASELINE` uses absolute ceilings 2-3x below the measured baseline and
   will block every position.
7. `make week-auto` is the Wednesday entrypoint from week 2 on; it grades the previous week and
   writes its research memo, degrading to a warning so a results hiccup never blocks new lines.

## Migrations: re-run forward, no rollback

There is no down-migration. Every migration is idempotent and gated on introspection — it checks
what the schema currently looks like and applies only what is missing — so the recovery for any
interrupted or partial migration is to run `make migrate` again, never to roll back.

Table rebuilds that widen a primary key run inside an explicit transaction, so a crash leaves the
original table intact and the re-run replays cleanly. One case still needs a human: if migration
aborts with a stranded `_pipeline_stage_runs_old` table, it means a swap died between renaming the
original and recreating it. Migration then refuses to continue rather than build an empty table
beside your stage history. Recover with:

```bash
sqlite3 <db> 'DROP TABLE IF EXISTS pipeline_stage_runs;'
sqlite3 <db> 'ALTER TABLE _pipeline_stage_runs_old RENAME TO pipeline_stage_runs;'
make migrate
```

`make doctor` additionally verifies the SQLite database is in WAL mode; a `sqlite_wal` failure means
the file never went through `make migrate`.

All commands are idempotent; rerun if data feeds update. For common database, migration, API-key, private-module, CORS, frontend, and port failures, see [Troubleshooting](TROUBLESHOOTING.md).

Do not use `make week-materialize` as a production publication path. Durable runs stage candidate
rows and atomically promote them only while their job attempt still owns its lease, has not been
cancelled, and has a persisted valid odds snapshot.
