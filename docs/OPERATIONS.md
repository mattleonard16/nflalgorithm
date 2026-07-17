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
publication whose acknowledgement may have been lost.

1. Run migrations and preflight after pulling updates:
   ```bash
   make migrate
   make doctor
   ```
2. Update a specific NFL week (idempotent upserts):
   ```bash
   make week-update SEASON=2023 WEEK=12
   ```
3. Train or refresh rolling models if needed:
   ```bash
   python -c "from models.position_specific import train_weekly_models; train_weekly_models([(2023, 10), (2023, 11), (2023, 12)])"
   ```
4. Generate projections and materialize dashboard view:
   ```bash
   make week-predict SEASON=2023 WEEK=12
   make week-materialize SEASON=2023 WEEK=12
   ```
5. Execute sanity checks before publishing:
   ```bash
   make mini-backtest SEASON=2023 WEEK=12
   make health SEASON=2023 WEEK=12
   ```
6. Launch dashboard and monitor feeds:
   ```bash
   make dashboard
   ```

All commands are idempotent; rerun if data feeds update. For common database, migration, API-key, private-module, CORS, frontend, and port failures, see [Troubleshooting](TROUBLESHOOTING.md).

Do not use `make week-materialize` as a production publication path. Durable runs stage candidate
rows and atomically promote them only while their job attempt still owns its lease, has not been
cancelled, and has a persisted valid odds snapshot.
