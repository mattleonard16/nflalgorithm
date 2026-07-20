"""Tracked ASGI wrapper around the deployment-supplied API application."""

from __future__ import annotations

from typing import Any

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
from api.pipeline_router import router as pipeline_router


def _replace_legacy_routes(app: Any, replacement_router: Any) -> None:
    """Remove deployment-supplied routes superseded by tracked controllers."""
    replacement_keys = {
        (route.path, frozenset(route.methods or set()))
        for route in replacement_router.routes
        if getattr(route, "path", None)
    }
    app.router.routes[:] = [
        route
        for route in app.router.routes
        if (
            getattr(route, "path", None),
            frozenset(getattr(route, "methods", None) or set()),
        )
        not in replacement_keys
    ]


_replace_legacy_routes(app, pipeline_router)
app.include_router(pipeline_router)
app.include_router(diagnostics_router)

__all__ = ["app"]
