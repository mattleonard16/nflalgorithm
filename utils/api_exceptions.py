"""Global FastAPI exception handling.

Registers a catch-all handler so uncaught exceptions are logged through the
structured logging stream (see ``utils/logging_config.py``) instead of falling
through to uvicorn's default stderr traceback, and so clients never see raw
exception text.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def _handle_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """Log the exception with request context and return a generic 500.

    The response body never includes ``str(exc)`` or a traceback: the caller
    only learns that something failed server-side. Query params are excluded
    from the logged context because they may carry tokens or other secrets.
    """
    logger.exception(
        "Unhandled exception during request",
        extra={"event": "api.unhandled_exception", "method": request.method, "path": request.url.path},
    )
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


def install_exception_handlers(app: FastAPI) -> None:
    """Register the catch-all handler on ``app``."""
    app.add_exception_handler(Exception, _handle_unhandled_exception)
