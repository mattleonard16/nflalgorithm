"""Tracked ASGI wrapper around the deployment-supplied API application."""

from __future__ import annotations

try:
    from api.server import app
except ModuleNotFoundError as exc:
    if exc.name != "api.server":
        raise
    raise RuntimeError(
        "Private API module `api.server` is unavailable. Install the deployment-supplied "
        "API module before starting the service."
    ) from exc

from api.diagnostics import router as diagnostics_router

app.include_router(diagnostics_router)

__all__ = ["app"]
