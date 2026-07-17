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
5. First deploy builds both images. API readiness probe: `GET /readyz`.

## Seed the database

The API container applies migrations before starting the API and worker. A fresh disk still needs data ingestion via the Render shell:

```bash
# In Render shell for nflalgorithm-api
python scripts/ingest_real_nfl_data.py
python -m scripts.preflight --check-schema --require-live-odds --require-private-modules
```

`/livez` checks only the API process. `/readyz` returns 503 when the database is unavailable or required migrations are missing. Set `LOG_FORMAT=json` for structured service-entrypoint logs.

## CORS

The deployment-supplied `api/server.py` reads `ALLOWED_ORIGINS` (comma-separated); tracked startup goes through `api/application.py` to attach operational probes. Unset CORS origins fall back to `localhost:3000/3001`. Set the frontend Render URL before the frontend can reach the API.

## Known gaps before going live

- **Auth endpoints broken** — punch-list #2/#3. Do not expose publicly until fixed.
- **No rate limits** — punch-list #22.

## Costs

Two `starter` services + 1 GB disk. Check Render pricing; free tier spins down on idle (cold starts ~30s).
