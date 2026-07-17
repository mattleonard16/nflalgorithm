"""Minimum database version required by durable MySQL queue semantics."""

from __future__ import annotations

import pytest

from utils.db import validate_mysql_server_version


@pytest.mark.parametrize("version", ["8.0.36", "8.4.3", "9.5.0-commercial"])
def test_mysql_8_or_newer_is_supported(version: str) -> None:
    assert validate_mysql_server_version(version)[0] >= 8


@pytest.mark.parametrize("version", ["5.7.44", "10.11.8-MariaDB", "unknown"])
def test_unsupported_mysql_versions_fail_startup(version: str) -> None:
    with pytest.raises(RuntimeError, match="MySQL|parse"):
        validate_mysql_server_version(version)
