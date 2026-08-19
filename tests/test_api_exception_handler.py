"""Tests for the global FastAPI exception handler."""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from utils.api_exceptions import install_exception_handlers


def _build_app() -> FastAPI:
    app = FastAPI()
    install_exception_handlers(app)

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("sensitive internal detail: db password is hunter2")

    @app.get("/ok")
    def ok() -> dict:
        return {"status": "fine"}

    return app


def test_unhandled_exception_returns_generic_500_body() -> None:
    client = TestClient(_build_app(), raise_server_exceptions=False)

    response = client.get("/boom")

    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error"}
    assert "hunter2" not in response.text
    assert "RuntimeError" not in response.text


def test_unhandled_exception_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    client = TestClient(_build_app(), raise_server_exceptions=False)

    with caplog.at_level(logging.ERROR, logger="utils.api_exceptions"):
        client.get("/boom")

    records = [r for r in caplog.records if r.name == "utils.api_exceptions"]
    assert records, "expected the handler to log the unhandled exception"
    assert records[0].exc_info is not None
    assert getattr(records[0], "path", None) == "/boom"
    assert getattr(records[0], "method", None) == "GET"


def test_healthy_route_unaffected() -> None:
    client = TestClient(_build_app())

    response = client.get("/ok")

    assert response.status_code == 200
    assert response.json() == {"status": "fine"}
