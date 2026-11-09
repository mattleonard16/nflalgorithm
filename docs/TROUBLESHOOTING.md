# Troubleshooting

Start with the preflight report:

```bash
make doctor
# Deployment-equivalent requirements (live odds + private NFL modules):
make doctor-production
```

`PASS` checks are ready, `WARN` checks disable an optional capability, and `FAIL` checks block startup. The commands never print database credentials or API-key values.

## Public clone vs deployment checkout

Three files are gitignored and never published: `data_pipeline.py`, `value_betting_engine.py`,
and `models/position_specific/weekly.py`. They hold the model and the pricing logic. A fresh clone
does not have them, and `make doctor` reports that as `WARN`, not `FAIL`. Nothing is broken.

A clone runs tracked public baselines in their place, the same way `config/` falls back to
`config/runtime.py` when the optional `config.py` override is absent:

- `models/position_specific/baseline.py` projects each player from their last six games. No trained
  model, no defense adjustment. Its rows carry `model_version = public_baseline_ewma_v1`, and
  `make week-predict` prints the version it used.
- `utils/value_ranking.py` prices the card: live quotes only, no-vig fair odds, both sides of
  every line, capped Kelly stakes.

So the whole local workflow runs on a clone: `make week-predict`, `make week-materialize`,
`make nfl-backtest`, `make mae-gate`, `make fullstack`. A card needs an `ODDS_API_KEY`; without
one it builds empty, which is correct. On the 2025 walk-forward the baseline scores yardage MAE
28.78 against 26.50 for the private model, so it is a fair floor to measure a change against.

Still needs the private files: `make doctor-production`, `make doctor-season`,
`make doctor-preseason`, and any deployment that sets
`NFL_REQUIRE_PRIVATE_MODELS=1`. That setting turns a missing private file into a failed run
instead of a baseline card. `render.yaml` sets it.

Tests that import a private module skip themselves. `tests/conftest.py` fences them per module, so
each one is skipped only when the module it actually needs is absent.

If a `[FAIL]` row names one of those files on a public clone, that is a bug in the repository.
Please open an issue with the `make doctor` output.

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `SQLite database does not exist` | The configured database has not been created | Check `SQLITE_DB_PATH`, then run `make migrate` |
| `Required tables are missing` or `/readyz` returns 503 | Migrations were not applied to the selected database | Run `make migrate` for local SQLite; run `python -m scripts.run_migrations` in the deployment environment |
| `DB_BACKEND=mysql requires DB_URL` | MySQL was selected without credentials | Set `DB_URL=mysql+pymysql://user:pass@host:3306/database` in the secret store; do not commit it |
| `Database connection failed` | Bad credentials, unreachable host, missing database, or unwritable SQLite path | Verify `DB_BACKEND`, `DB_URL`/`SQLITE_DB_PATH`, DNS/firewall, and filesystem permissions |
| `ODDS_API_KEY is not configured` | Live odds cannot be fetched | Create a key at https://the-odds-api.com/ and set `ODDS_API_KEY` in `.env` or the deployment secret store. The market keys must be The Odds API's exact names (`player_pass_yds`, `player_rush_yds`, `player_reception_yds`, `player_receptions`); a misspelled key returns HTTP 422 for every event and the run fails coverage. Each event-market pair costs one credit per request, so a full week is roughly 16 events x 4 markets = 64 credits. The live pipeline fails closed without a usable key |
| `nfl_history_team_scope` or `nfl_history_franchises` fail | Historical stats have empty `team` (legacy `LA` Rams rows) or fewer than 32 clubs | `make ingest-nfl NFL_SEASONS=2024,2025 THROUGH_WEEK=22` then `make doctor-preseason SEASON=2026 WEEK=1`. Do not use `doctor-season` until a live-odds key exists |
| `[FAIL] api_server: api/server.py is missing` | The file is tracked, so this means an incomplete checkout, not a public clone | Run `git status` to find the deletion, or re-clone |
| `api/server.py does not implement the current public visibility contract` | `PUBLIC_VALUE_VISIBILITY_CONTRACT` in that file is stale | Set it to `publication-safe-v1`. A stale value serves legacy unjoinable and `SimBook` rows as if they were real |
| `[WARN] private_modules: Private NFL execution modules are unavailable` | Expected on a public clone: the model and pricing modules are gitignored | Nothing to do locally; projections and cards use the public baselines. A production deployment must install the private modules and set `NFL_REQUIRE_PRIVATE_MODELS=1` |
| `Node.js ... is too old` | Next.js 16 requires Node 20.9+ | Upgrade Node, then run `make frontend-install` |
| `Frontend dependencies are not installed` | `frontend/node_modules` is absent | Run `make frontend-install` (`npm ci`) |
| `Required local ports are already in use` | Another API/frontend process owns 8000 or 3000 | Stop it (`lsof -i :8000`, `lsof -i :3000`) or change `API_PORT`/`FRONTEND_PORT` in `.env` |
| Browser cannot reach the API | Public API URL or CORS origin is wrong | Set `NEXT_PUBLIC_API_URL` to a browser-reachable URL and rebuild the frontend; set `ALLOWED_ORIGINS` on the API |
| Compose frontend remains blocked | API readiness failed | Run `docker compose logs api`; query `/livez` and `/readyz`; do not use `/api/health` as a process probe |

## Probe semantics

```bash
curl -i http://localhost:8000/livez   # Process liveness only
curl -i http://localhost:8000/readyz  # DB connectivity + required migrations
curl -i http://localhost:8000/api/health  # Feed freshness; not a startup probe
```

`/readyz` returns HTTP 503 until the configured database is reachable and required tables exist.

## Local startup

```bash
cp .env.example .env
make install
make frontend-install
make migrate
make doctor
make fullstack
```

`make fullstack` starts the worker and API, waits for `/readyz`, then starts Next.js. Ctrl-C or any child-process failure stops the remaining processes.

## Deployment logs

Set `LOG_FORMAT=json` and `LOG_LEVEL=INFO`. Service-entrypoint logs then include stable `timestamp`, `level`, `service`, `logger`, `message`, and allow-listed operational context such as `event`, `run_id`, `job_id`, `season`, `week`, and `stage`. Secret-like arbitrary fields are not serialized.
