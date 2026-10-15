"""Tracked ASGI wrapper around the deployment-supplied API application."""

from __future__ import annotations

from typing import Any

from utils.error_tracking import init_error_tracking
from utils.logging_config import configure_logging

# Runs before importing api.server so that module's own logging calls (and any
# side effects at import time) go through the structured stream instead of
# Python's lastResort handler.
configure_logging("api")
init_error_tracking("api")

try:
    from api.server import app
except ModuleNotFoundError as exc:
    if exc.name != "api.server":
        raise
    raise RuntimeError(
        "Private API module `api.server` is unavailable. Install the deployment-supplied "
        "API module before starting the service."
    ) from exc

from fastapi.middleware.cors import CORSMiddleware

from api.diagnostics import router as diagnostics_router
from api.pipeline_router import router as pipeline_router
from api.projections_router import router as projections_router
from api.rate_limit import RateLimitMiddleware
from utils.api_exceptions import install_exception_handlers


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
app.include_router(projections_router)
install_exception_handlers(app)
# Applied here, not in the deployment-supplied module, so every deployment gets
# the limiter regardless of which private api.server it ships. Limits are read
# from the environment at construction; invalid values fail startup loud.
app.add_middleware(RateLimitMiddleware)
# add_middleware wraps outermost, which would put the limiter outside CORS and
# send 429s without Access-Control-* headers (browsers then report a CORS error
# instead of the rate limit). Move CORS back to the outside.
_cors = [m for m in app.user_middleware if m.cls is CORSMiddleware]
for _middleware in _cors:
    app.user_middleware.remove(_middleware)
    app.user_middleware.insert(0, _middleware)

__all__ = ["app"]
