# Weekly Operations

## Startup

```bash
make migrate
make doctor
make fullstack
```

`make fullstack` supervises the worker, API, and frontend. It waits for `/readyz`; `/livez` only verifies that the API process is running. Use `LOG_FORMAT=json` in deployments.

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
