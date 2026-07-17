# Troubleshooting

Start with the preflight report:

```bash
make doctor
# Deployment-equivalent requirements (live odds + private NFL modules):
make doctor-production
```

`PASS` checks are ready, `WARN` checks disable an optional capability, and `FAIL` checks block startup. The commands never print database credentials or API-key values.

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `SQLite database does not exist` | The configured database has not been created | Check `SQLITE_DB_PATH`, then run `make migrate` |
| `Required tables are missing` or `/readyz` returns 503 | Migrations were not applied to the selected database | Run `make migrate` for local SQLite; run `python -m scripts.run_migrations` in the deployment environment |
| `DB_BACKEND=mysql requires DB_URL` | MySQL was selected without credentials | Set `DB_URL=mysql+pymysql://user:pass@host:3306/database` in the secret store; do not commit it |
| `Database connection failed` | Bad credentials, unreachable host, missing database, or unwritable SQLite path | Verify `DB_BACKEND`, `DB_URL`/`SQLITE_DB_PATH`, DNS/firewall, and filesystem permissions |
| `ODDS_API_KEY is not configured` | Live odds cannot be fetched | Set the key in `.env` locally or the deployment secret store. The NFL live pipeline intentionally fails closed without it |
| `Private NFL execution modules are unavailable` | This checkout lacks deployment-supplied model modules | Read-only API/UI work remains available. Install the private modules before starting production workers |
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
