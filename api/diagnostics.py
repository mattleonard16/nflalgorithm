"""Dependency diagnostics exposed independently of the private API implementation."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from utils.db import get_connection, is_sqlite_connection, table_exists

logger = logging.getLogger(__name__)
router = APIRouter(tags=["operations"])
REQUIRED_API_TABLES = (
    "feed_freshness",
    "materialized_value_view",
    "pipeline_jobs",
    "pipeline_runs",
    "users",
)


@router.get("/livez")
async def get_liveness() -> dict[str, str]:
    """Report that the API process is running without testing dependencies."""
    return {"status": "ok", "service": "api"}


@router.get("/readyz")
async def get_readiness() -> dict[str, object]:
    """Report database connectivity and required migration state."""
    try:
        with get_connection() as connection:
            if is_sqlite_connection(connection):
                connection.execute("SELECT 1")
            else:
                cursor = connection.cursor()
                try:
                    cursor.execute("SELECT 1")
                finally:
                    cursor.close()
            missing_tables = [
                table for table in REQUIRED_API_TABLES if not table_exists(table, conn=connection)
            ]
    except Exception:
        logger.exception(
            "Database readiness check failed",
            extra={"event": "health.readiness_failed"},
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Database unavailable. Verify DB_BACKEND, DB_URL/SQLITE_DB_PATH, "
                "network access, and credentials."
            ),
        )

    if missing_tables:
        logger.error(
            "Readiness blocked by missing database tables: %s",
            ", ".join(missing_tables),
            extra={"event": "health.migrations_missing"},
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"Database migrations are incomplete; missing tables: {', '.join(missing_tables)}. "
                "Run `make migrate` or the deployment migration command."
            ),
        )

    return {
        "status": "ready",
        "service": "api",
        "checks": {"database": "ok", "migrations": "ok"},
    }
