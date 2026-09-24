"""Account, login and session rules in api/auth.py.

Runs the module's functions against a throwaway SQLite database. Hashing and
the HTTP endpoints are covered in tests/test_auth_t0_2.py; this file covers
what they leave out: registration limits, failed logins, session expiry and
logout, and stored hashes that cannot be parsed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from api.auth import (
    UserCreate,
    UserLogin,
    UserPreferences,
    authenticate_user,
    create_user,
    get_user_preferences,
    logout_user,
    update_user_preferences,
    validate_session,
)
from config import config
from schema_migrations import MigrationManager
from utils.db import execute, fetchone

PASSWORD = "correct-horse-1"


@pytest.fixture()
def db(tmp_path, monkeypatch) -> str:
    db_path = str(tmp_path / "auth.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)
    monkeypatch.setattr(config.database, "backend", "sqlite")
    monkeypatch.setattr(config.database, "path", db_path)
    MigrationManager(db_path).run()
    return db_path


def _register(email: str = "bettor@example.com") -> str:
    user = create_user(UserCreate(email=email, password=PASSWORD, name="Bettor"))
    assert user is not None
    return user.id


def _login(email: str = "bettor@example.com", password: str = PASSWORD) -> dict | None:
    return authenticate_user(UserLogin(email=email, password=password))


def _count(table: str) -> int:
    row = fetchone(f"SELECT COUNT(*) FROM {table}")
    assert row is not None
    return int(row[0])


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_a_password_under_eight_characters_is_rejected(db):
    with pytest.raises(ValueError, match="at least 8"):
        create_user(UserCreate(email="bettor@example.com", password="short1"))

    assert _count("users") == 0


def test_a_password_over_72_bytes_is_rejected_even_when_under_72_characters(db):
    # bcrypt ignores everything past byte 72, so 37 two-byte characters would
    # let a login with only the first 36 of them through.
    with pytest.raises(ValueError, match="72 bytes"):
        create_user(UserCreate(email="bettor@example.com", password="é" * 37))

    assert _count("users") == 0


def test_a_second_account_on_the_same_email_is_rejected(db):
    _register()

    with pytest.raises(ValueError, match="already registered"):
        _register()

    assert _count("users") == 1


def test_a_new_account_starts_with_default_preferences(db):
    user_id = _register()

    assert get_user_preferences(user_id) == UserPreferences()


# ---------------------------------------------------------------------------
# Login and sessions
# ---------------------------------------------------------------------------


def test_login_with_an_unregistered_email_fails(db):
    assert _login(email="nobody@example.com") is None
    assert _count("user_sessions") == 0


def test_a_wrong_password_fails_and_opens_no_session(db):
    _register()

    assert _login(password="wrong-password") is None
    assert _count("user_sessions") == 0


def test_a_login_session_resolves_to_the_user_who_logged_in(db):
    user_id = _register()

    session = _login()

    assert validate_session(session["session_id"]).id == user_id


def test_an_unknown_session_id_is_rejected(db):
    assert validate_session("not-a-session") is None


def test_an_expired_session_is_rejected_and_deleted(db):
    user_id = _register()
    now = datetime.now(timezone.utc)
    execute(
        "INSERT INTO user_sessions (session_id, user_id, expires_at, created_at) "
        "VALUES (?, ?, ?, ?)",
        ("stale", user_id, (now - timedelta(minutes=1)).isoformat(), now.isoformat()),
    )

    assert validate_session("stale") is None
    assert _count("user_sessions") == 0


def test_logout_ends_only_the_session_it_names(db):
    _register()
    phone = _login()["session_id"]
    laptop = _login()["session_id"]

    logout_user(phone)

    assert validate_session(phone) is None
    assert validate_session(laptop) is not None


@pytest.mark.parametrize("stored_hash", ["$2b$not-a-real-hash", "no-dollar-sign"])
def test_an_unparseable_stored_hash_fails_login_instead_of_raising(db, stored_hash):
    now = datetime.now(timezone.utc).isoformat()
    execute(
        "INSERT INTO users (id, email, password_hash, created_at, updated_at) "
        "VALUES ('usr_broken', 'broken@example.com', ?, ?, ?)",
        (stored_hash, now, now),
    )

    assert _login(email="broken@example.com") is None
    assert _count("user_sessions") == 0


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------


def test_saved_preferences_read_back_unchanged(db):
    user_id = _register()
    # Every flag flipped from its default, so a 1/0 round trip that drops a
    # value cannot pass by landing on the default.
    prefs = UserPreferences(
        default_min_edge=0.08,
        default_kelly_fraction=0.5,
        default_max_stake=0.05,
        best_line_only=False,
        show_synthetic_odds=True,
        defense_multipliers=False,
        weather_adjustments=False,
        injury_weighting=False,
        preferred_sportsbooks="DraftKings,FanDuel",
        preferred_markets="receiving_yards",
    )

    update_user_preferences(user_id, prefs)

    assert get_user_preferences(user_id) == prefs
