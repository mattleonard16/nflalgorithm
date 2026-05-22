# Render Deploy

Scaffold only. No remote project created yet.

## Layout

`render.yaml` defines two Docker services:

| Service | Dockerfile | Port | Disk |
|---|---|---|---|
| `nflalgorithm-api` | `Dockerfile.api` | 8000 | 1 GB at `/data` (SQLite) |
| `nflalgorithm-frontend` | `frontend/Dockerfile` | 3000 | none |

SQLite lives on the persistent disk. Survives redeploys but is **not** backed up — re-ingest from nflreadpy if lost.

## First-time setup

1. Push branch to GitHub (Render reads `render.yaml` from the repo).
2. Render dashboard -> **New** -> **Blueprint** -> pick this repo.
3. Render creates both services from `render.yaml`.
4. Set env vars (marked `sync: false`) in each service's dashboard. See `.env.render.example`.
   - API: `ODDS_API_KEY`, `ALLOWED_ORIGINS`
   - Frontend: `NEXT_PUBLIC_API_URL` (the API service's public URL)
5. First deploy builds both images. API healthcheck: `GET /api/health`.

## Seed the database

Fresh disk = empty SQLite. After first deploy, run migrations + ingest via Render shell:

```bash
# In Render shell for nflalgorithm-api
python -c "from schema_migrations import MigrationManager; MigrationManager('/data/nfl_data.db').run()"
python scripts/ingest_real_nfl_data.py
```

## CORS

`api/server.py` reads `ALLOWED_ORIGINS` (comma-separated). Unset = falls back to `localhost:3000/3001`. Set it to the frontend Render URL on the API service before the frontend can reach the API.

## Known gaps before going live

- **Auth endpoints broken** — punch-list #2/#3. Do not expose publicly until fixed.
- **No rate limits** — punch-list #22.

## Costs

Two `starter` services + 1 GB disk. Check Render pricing; free tier spins down on idle (cold starts ~30s).
