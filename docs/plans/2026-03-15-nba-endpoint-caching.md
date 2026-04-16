# NBA Endpoint Caching Implementation Plan

> **For Claude:** Tasks marked "Depends on: none" form Wave 1 and can run in
> parallel. Tasks with dependencies wait for their prerequisites to complete.
> REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** Add in-memory TTL caching to all NBA API endpoints using the existing `EndpointCache` infrastructure, eliminating redundant 44K-row aggregations on every request.

**Architecture:** Create a dedicated `nba_cache` instance (separate from NFL's `value_bets_cache`). Wrap each NBA endpoint's DB query with cache-check-before / cache-set-after pattern. Add a `/api/nba/cache/invalidate` admin endpoint for manual cache busting. Expose cache stats on the existing `/api/nba/health` endpoint.

**Tech Stack:** Python, FastAPI, existing `api/cache.py` (EndpointCache, make_cache_key)

---

## Wave visualization

```
Wave 1: [Task 1]          ← depends_on: none — create nba_cache + invalidation endpoint
Wave 2: [Task 2, Task 3]  ← depends_on: Task 1 — cache /meta and /players (parallel)
Wave 3: [Task 4]          ← depends_on: Task 1 — cache /value-bets, /performance, /projections
Wave 4: [Task 5]          ← depends_on: Task 2, 3, 4 — cache stats on /health + integration test
```

---

### Task 1: Create NBA cache instance and invalidation endpoint
**Depends on:** none

**Files:**
- Modify: `api/cache.py` (add `nba_cache` singleton)
- Modify: `api/nba_router.py` (add invalidation endpoint)
- Test: `tests/test_nba_cache.py`

**Step 1: Write the failing test**

```python
# tests/test_nba_cache.py
"""Tests for NBA endpoint caching."""

from api.cache import nba_cache, make_cache_key


class TestNbaCacheInstance:
    def test_nba_cache_exists(self):
        """nba_cache should be a separate instance from value_bets_cache."""
        from api.cache import value_bets_cache
        assert nba_cache is not value_bets_cache

    def test_nba_cache_default_ttl(self):
        """NBA cache should have 600s (10-min) default TTL."""
        assert nba_cache._default_ttl == 600

    def test_nba_cache_max_size(self):
        """NBA cache should support 200 entries."""
        assert nba_cache._max_size == 200
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/mleonard/sandbox/nflalgorithm && uv run pytest tests/test_nba_cache.py::TestNbaCacheInstance -v`
Expected: FAIL with `ImportError: cannot import name 'nba_cache'`

**Step 3: Write minimal implementation**

In `api/cache.py`, add after line 82 (after `value_bets_cache`):

```python
# NBA-specific cache (10-minute TTL, 200 entry max — NBA data changes less frequently)
nba_cache = EndpointCache(default_ttl=600, max_size=200)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/mleonard/sandbox/nflalgorithm && uv run pytest tests/test_nba_cache.py::TestNbaCacheInstance -v`
Expected: PASS

**Step 5: Commit**

```bash
git add api/cache.py tests/test_nba_cache.py
git commit -m "feat: add nba_cache instance with 10-min TTL for NBA endpoints"
```

---

### Task 2: Cache `/meta` endpoint (highest TTL — data changes once per pipeline run)
**Depends on:** Task 1

**Files:**
- Modify: `api/nba_router.py:266-289` (wrap `nba_meta` with cache)
- Test: `tests/test_nba_cache.py` (add meta cache test)

**Step 1: Write the failing test**

Append to `tests/test_nba_cache.py`:

```python
import importlib
from unittest.mock import patch, MagicMock

from api.cache import nba_cache


class TestNbaMetaCache:
    def setup_method(self):
        nba_cache.invalidate_all()

    def test_meta_cache_hit(self, monkeypatch, tmp_path):
        """Second call to /meta should use cache, not DB."""
        db_path = str(tmp_path / "test.db")
        monkeypatch.setenv("DB_BACKEND", "sqlite")
        monkeypatch.setenv("SQLITE_DB_PATH", db_path)

        import config as cfg
        monkeypatch.setattr(cfg.config.database, "backend", "sqlite")
        monkeypatch.setattr(cfg.config.database, "path", db_path)

        from utils.db import execute_sql
        execute_sql("CREATE TABLE IF NOT EXISTS nba_player_game_logs (season INT, game_date TEXT, player_id INT, game_id TEXT)")
        execute_sql("INSERT INTO nba_player_game_logs VALUES (2025, '2026-03-10', 1, 'G1')")
        execute_sql("INSERT INTO nba_player_game_logs VALUES (2025, '2026-03-10', 2, 'G1')")

        from fastapi.testclient import TestClient
        from api.nba_router import router
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        # First call — populates cache
        r1 = client.get("/api/nba/meta")
        assert r1.status_code == 200
        assert r1.json()["total_players"] == 2

        # Delete data — cache should still serve old result
        execute_sql("DELETE FROM nba_player_game_logs")

        r2 = client.get("/api/nba/meta")
        assert r2.status_code == 200
        assert r2.json()["total_players"] == 2  # cached!
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/mleonard/sandbox/nflalgorithm && uv run pytest tests/test_nba_cache.py::TestNbaMetaCache -v`
Expected: FAIL — second call returns `total_players=0` (no cache)

**Step 3: Write minimal implementation**

In `api/nba_router.py`, add import at top:

```python
from api.cache import nba_cache, make_cache_key
```

Replace the `nba_meta` function body with cache wrapping:

```python
@router.get("/meta", response_model=NbaMeta)
def nba_meta() -> NbaMeta:
    """Return available seasons and summary counts."""
    cache_key = make_cache_key("nba-meta")
    cached = nba_cache.get(cache_key)
    if cached is not None:
        return cached

    seasons_rows = fetchall(
        "SELECT DISTINCT season FROM nba_player_game_logs ORDER BY season"
    )
    seasons = [r[0] for r in seasons_rows]

    latest = fetchone(
        "SELECT MAX(game_date) FROM nba_player_game_logs"
    )
    latest_date = latest[0] if latest else None

    totals = fetchone(
        "SELECT COUNT(DISTINCT player_id), COUNT(DISTINCT game_id) "
        "FROM nba_player_game_logs"
    )

    result = NbaMeta(
        available_seasons=seasons,
        latest_game_date=latest_date,
        total_players=totals[0] if totals else 0,
        total_games=totals[1] if totals else 0,
    )
    nba_cache.set(cache_key, result)
    return result
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/mleonard/sandbox/nflalgorithm && uv run pytest tests/test_nba_cache.py::TestNbaMetaCache -v`
Expected: PASS

**Step 5: Commit**

```bash
git add api/nba_router.py tests/test_nba_cache.py
git commit -m "perf: cache /api/nba/meta endpoint (3 SQL queries → 0 on hit)"
```

---

### Task 3: Cache `/players` endpoint (heaviest aggregation — 44K row GROUP BY)
**Depends on:** Task 1

**Files:**
- Modify: `api/nba_router.py:413-467` (wrap `nba_players` with cache)
- Test: `tests/test_nba_cache.py` (add players cache test)

**Step 1: Write the failing test**

Append to `tests/test_nba_cache.py`:

```python
class TestNbaPlayersCache:
    def setup_method(self):
        nba_cache.invalidate_all()

    def test_players_cache_keyed_by_params(self):
        """Different query params should produce different cache keys."""
        from api.cache import make_cache_key
        k1 = make_cache_key("nba-players", season=2025, sort="pts", team=None, search=None, limit=100)
        k2 = make_cache_key("nba-players", season=2025, sort="reb", team=None, search=None, limit=100)
        k3 = make_cache_key("nba-players", season=2025, sort="pts", team="LAL", search=None, limit=100)
        assert k1 != k2
        assert k1 != k3

    def test_players_cache_hit(self, monkeypatch, tmp_path):
        """Second call to /players with same params should use cache."""
        db_path = str(tmp_path / "test.db")
        monkeypatch.setenv("DB_BACKEND", "sqlite")
        monkeypatch.setenv("SQLITE_DB_PATH", db_path)

        import config as cfg
        monkeypatch.setattr(cfg.config.database, "backend", "sqlite")
        monkeypatch.setattr(cfg.config.database, "path", db_path)

        from utils.db import execute_sql
        execute_sql(
            "CREATE TABLE IF NOT EXISTS nba_player_game_logs "
            "(season INT, player_id INT, player_name TEXT, team_abbreviation TEXT, "
            "pts REAL, reb REAL, ast REAL, fg3m REAL, min REAL, game_date TEXT, game_id TEXT)"
        )
        execute_sql(
            "INSERT INTO nba_player_game_logs VALUES "
            "(2025, 1, 'Test Player', 'LAL', 25.0, 5.0, 7.0, 3.0, 35.0, '2026-03-10', 'G1')"
        )

        from fastapi.testclient import TestClient
        from api.nba_router import router
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        r1 = client.get("/api/nba/players?season=2025")
        assert r1.status_code == 200
        assert r1.json()["total"] == 1

        execute_sql("DELETE FROM nba_player_game_logs")

        r2 = client.get("/api/nba/players?season=2025")
        assert r2.status_code == 200
        assert r2.json()["total"] == 1  # cached!
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/mleonard/sandbox/nflalgorithm && uv run pytest tests/test_nba_cache.py::TestNbaPlayersCache -v`
Expected: FAIL — second call returns `total=0`

**Step 3: Write minimal implementation**

Replace the `nba_players` function body:

```python
@router.get("/players", response_model=NbaPlayersResponse)
def nba_players(
    season: int = Query(2025),
    team: str | None = Query(None),
    search: str | None = Query(None, description="Partial player name search"),
    sort: str = Query("pts", description="Sort by market: pts, reb, ast, fg3m"),
    limit: int = Query(100, ge=1, le=500),
) -> NbaPlayersResponse:
    """Return players with season averages."""
    cache_key = make_cache_key("nba-players", season=season, sort=sort, team=team, search=search, limit=limit)
    cached = nba_cache.get(cache_key)
    if cached is not None:
        return cached

    where_clauses = ["season = ?"]
    params: list[Any] = [season]

    if team:
        where_clauses.append("team_abbreviation = ?")
        params.append(team.upper())

    if search:
        where_clauses.append("LOWER(player_name) LIKE ?")
        params.append(f"%{search.lower()}%")

    where = " AND ".join(where_clauses)
    params.append(limit)

    order_col = _PLAYERS_SORT_COLS.get(sort, "avg_pts")

    df = read_dataframe(
        "SELECT player_id, player_name, team_abbreviation as team, "
        "COUNT(*) as games_played, "
        "ROUND(AVG(pts), 1) as avg_pts, "
        "ROUND(AVG(reb), 1) as avg_reb, "
        "ROUND(AVG(ast), 1) as avg_ast, "
        "ROUND(AVG(fg3m), 1) as avg_fg3m, "
        "ROUND(AVG(min), 1) as avg_min "
        "FROM nba_player_game_logs WHERE " + where + " "
        "GROUP BY player_id, player_name, team_abbreviation "
        f"ORDER BY {order_col} DESC LIMIT ?",
        params,
    )

    players = [
        NbaPlayerSummary(
            player_id=int(row["player_id"]),
            player_name=str(row["player_name"]),
            team=str(row["team"]),
            games_played=int(row["games_played"]),
            avg_pts=float(row["avg_pts"] or 0),
            avg_reb=float(row["avg_reb"] or 0),
            avg_ast=float(row["avg_ast"] or 0),
            avg_fg3m=float(row["avg_fg3m"] or 0),
            avg_min=float(row["avg_min"] or 0),
        )
        for _, row in df.iterrows()
    ]

    result = NbaPlayersResponse(players=players, season=season, total=len(players))
    nba_cache.set(cache_key, result)
    return result
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/mleonard/sandbox/nflalgorithm && uv run pytest tests/test_nba_cache.py::TestNbaPlayersCache -v`
Expected: PASS

**Step 5: Commit**

```bash
git add api/nba_router.py tests/test_nba_cache.py
git commit -m "perf: cache /api/nba/players endpoint (44K-row GROUP BY → 0 on hit)"
```

---

### Task 4: Cache `/value-bets`, `/performance`, and `/projections` endpoints
**Depends on:** Task 1

**Files:**
- Modify: `api/nba_router.py:470-502, 600-654, 331-410` (wrap 3 endpoints)
- Test: `tests/test_nba_cache.py` (add value-bets cache test)

**Step 1: Write the failing test**

Append to `tests/test_nba_cache.py`:

```python
class TestNbaValueBetsCache:
    def setup_method(self):
        nba_cache.invalidate_all()

    def test_value_bets_cache_keyed_by_params(self):
        """Different filters should produce different cache keys."""
        from api.cache import make_cache_key
        k1 = make_cache_key("nba-value-bets", game_date="2026-03-10", market="pts", min_edge=0.0, sportsbook=None, limit=50, best_line_only=False, include_why=False)
        k2 = make_cache_key("nba-value-bets", game_date="2026-03-10", market="reb", min_edge=0.0, sportsbook=None, limit=50, best_line_only=False, include_why=False)
        assert k1 != k2

    def test_value_bets_cache_hit(self, monkeypatch, tmp_path):
        """Second call to /value-bets should use cache."""
        db_path = str(tmp_path / "test.db")
        monkeypatch.setenv("DB_BACKEND", "sqlite")
        monkeypatch.setenv("SQLITE_DB_PATH", db_path)

        import config as cfg
        monkeypatch.setattr(cfg.config.database, "backend", "sqlite")
        monkeypatch.setattr(cfg.config.database, "path", db_path)

        from utils.db import execute_sql
        execute_sql(
            "CREATE TABLE IF NOT EXISTS nba_materialized_value_view "
            "(game_date TEXT, player_id INT, player_name TEXT, team TEXT, event_id TEXT, "
            "market TEXT, sportsbook TEXT, line REAL, over_price INT, under_price INT, "
            "mu REAL, sigma REAL, p_win REAL, edge_percentage REAL, expected_roi REAL, "
            "kelly_fraction REAL, confidence REAL, generated_at TEXT, side TEXT)"
        )
        execute_sql(
            "INSERT INTO nba_materialized_value_view VALUES "
            "('2026-03-10', 1, 'Test Player', 'LAL', 'E1', 'pts', 'DraftKings', "
            "20.5, -110, 110, 22.0, 3.0, 0.65, 0.12, 0.15, 0.05, 0.8, '2026-03-10T08:00:00', 'over')"
        )

        from fastapi.testclient import TestClient
        from api.nba_router import router
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        r1 = client.get("/api/nba/value-bets?game_date=2026-03-10&market=pts")
        assert r1.status_code == 200
        assert r1.json()["total"] == 1

        execute_sql("DELETE FROM nba_materialized_value_view")

        r2 = client.get("/api/nba/value-bets?game_date=2026-03-10&market=pts")
        assert r2.status_code == 200
        assert r2.json()["total"] == 1  # cached!
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/mleonard/sandbox/nflalgorithm && uv run pytest tests/test_nba_cache.py::TestNbaValueBetsCache -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Wrap `nba_value_bets` — add cache check at start and cache set before return:

```python
@router.get("/value-bets", response_model=NbaValueBetsResponse)
def nba_value_bets(
    game_date: str | None = Query(None),
    market: str = Query("pts"),
    min_edge: float = Query(0.0, ge=0.0),
    best_line_only: bool = Query(False),
    sportsbook: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    include_why: bool = Query(False),
) -> NbaValueBetsResponse:
    if market not in VALID_MARKETS:
        raise HTTPException(status_code=400, detail=f"Invalid market '{market}'...")

    game_date = _resolve_game_date(game_date)

    cache_key = make_cache_key(
        "nba-value-bets", game_date=game_date, market=market,
        min_edge=min_edge, sportsbook=sportsbook, limit=limit,
        best_line_only=best_line_only, include_why=include_why,
    )
    cached = nba_cache.get(cache_key)
    if cached is not None:
        return cached

    # ... existing query logic unchanged ...

    result = NbaValueBetsResponse(bets=bets, total=len(bets), game_date=game_date, filters=applied_filters)
    nba_cache.set(cache_key, result)
    return result
```

Same pattern for `nba_performance` (key: `nba-performance`, params: `season`) and `nba_projections` (key: `nba-projections`, params: `game_date, market, min_confidence, limit`).

**Step 4: Run test to verify it passes**

Run: `cd /Users/mleonard/sandbox/nflalgorithm && uv run pytest tests/test_nba_cache.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add api/nba_router.py tests/test_nba_cache.py
git commit -m "perf: cache /value-bets, /performance, /projections NBA endpoints"
```

---

### Task 5: Add cache stats to `/health`, invalidation endpoint, and integration test
**Depends on:** Task 2, Task 3, Task 4

**Files:**
- Modify: `api/nba_router.py:815-837` (add cache stats to health)
- Modify: `api/nba_router.py` (add POST invalidation endpoint)
- Test: `tests/test_nba_cache.py` (add integration + invalidation tests)

**Step 1: Write the failing test**

Append to `tests/test_nba_cache.py`:

```python
class TestNbaCacheInvalidation:
    def setup_method(self):
        nba_cache.invalidate_all()

    def test_invalidate_endpoint(self, monkeypatch, tmp_path):
        """POST /api/nba/cache/invalidate should clear NBA cache."""
        db_path = str(tmp_path / "test.db")
        monkeypatch.setenv("DB_BACKEND", "sqlite")
        monkeypatch.setenv("SQLITE_DB_PATH", db_path)

        import config as cfg
        monkeypatch.setattr(cfg.config.database, "backend", "sqlite")
        monkeypatch.setattr(cfg.config.database, "path", db_path)

        from utils.db import execute_sql
        execute_sql("CREATE TABLE IF NOT EXISTS nba_player_game_logs (season INT, game_date TEXT, player_id INT, game_id TEXT)")
        execute_sql("INSERT INTO nba_player_game_logs VALUES (2025, '2026-03-10', 1, 'G1')")

        from fastapi.testclient import TestClient
        from api.nba_router import router
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        # Populate cache
        client.get("/api/nba/meta")
        assert nba_cache.size() > 0

        # Invalidate
        r = client.post("/api/nba/cache/invalidate")
        assert r.status_code == 200
        assert r.json()["cleared"] is True
        assert nba_cache.size() == 0


class TestNbaHealthCacheStats:
    def setup_method(self):
        nba_cache.invalidate_all()

    def test_health_includes_cache_stats(self, monkeypatch, tmp_path):
        """GET /api/nba/health should include cache_entries count."""
        db_path = str(tmp_path / "test.db")
        monkeypatch.setenv("DB_BACKEND", "sqlite")
        monkeypatch.setenv("SQLITE_DB_PATH", db_path)

        import config as cfg
        monkeypatch.setattr(cfg.config.database, "backend", "sqlite")
        monkeypatch.setattr(cfg.config.database, "path", db_path)

        from utils.db import execute_sql
        execute_sql("CREATE TABLE IF NOT EXISTS nba_player_game_logs (season INT, game_date TEXT, player_id INT, game_id TEXT)")
        execute_sql("CREATE TABLE IF NOT EXISTS nba_projections (game_date TEXT, player_id INT, player_name TEXT, team TEXT, market TEXT, projected_value REAL, confidence REAL)")

        from fastapi.testclient import TestClient
        from api.nba_router import router
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        r = client.get("/api/nba/health")
        assert r.status_code == 200
        assert "cache" in r.json()
        assert "entries" in r.json()["cache"]
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/mleonard/sandbox/nflalgorithm && uv run pytest tests/test_nba_cache.py::TestNbaCacheInvalidation tests/test_nba_cache.py::TestNbaHealthCacheStats -v`
Expected: FAIL — no POST endpoint, no `cache` key in health

**Step 3: Write minimal implementation**

Add invalidation endpoint to `api/nba_router.py`:

```python
@router.post("/cache/invalidate")
def nba_cache_invalidate() -> dict:
    """Clear all NBA endpoint caches."""
    nba_cache.invalidate_all()
    return {"cleared": True, "message": "NBA cache invalidated"}
```

Update `/health` to include cache stats:

```python
@router.get("/health")
def nba_health() -> dict:
    """Return NBA data freshness info."""
    latest = fetchone(
        "SELECT MAX(game_date), COUNT(*), COUNT(DISTINCT player_id) "
        "FROM nba_player_game_logs"
    )
    proj_latest = fetchone(
        "SELECT MAX(game_date), COUNT(*) FROM nba_projections"
    )

    return {
        "status": "ok",
        "game_logs": {
            "latest_game_date": latest[0] if latest else None,
            "total_rows": latest[1] if latest else 0,
            "total_players": latest[2] if latest else 0,
        },
        "projections": {
            "latest_date": proj_latest[0] if proj_latest else None,
            "total_rows": proj_latest[1] if proj_latest else 0,
        },
        "cache": {
            "entries": nba_cache.size(),
            "max_size": nba_cache._max_size,
            "ttl_seconds": nba_cache._default_ttl,
        },
    }
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/mleonard/sandbox/nflalgorithm && uv run pytest tests/test_nba_cache.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add api/nba_router.py tests/test_nba_cache.py
git commit -m "feat: add NBA cache invalidation endpoint and health cache stats"
```
