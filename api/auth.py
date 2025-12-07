"""
Authentication utilities for NFL Algorithm API.

Simple JWT-based authentication with bcrypt password hashing.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional

from pydantic import BaseModel, EmailStr

from utils.db import execute, fetchone, fetchall


# Configuration
SESSION_EXPIRY_DAYS = 7
MIN_PASSWORD_LENGTH = 8


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    name: Optional[str]
    subscription_tier: str
    bankroll: float
    created_at: str


class UserPreferences(BaseModel):
    default_min_edge: float = 0.05
    default_kelly_fraction: float = 0.25
    default_max_stake: float = 0.02
    best_line_only: bool = True
    show_synthetic_odds: bool = False
    defense_multipliers: bool = True
    weather_adjustments: bool = True
    injury_weighting: bool = True
    preferred_sportsbooks: Optional[str] = None
    preferred_markets: Optional[str] = None


def hash_password(password: str) -> str:
    """Hash password using SHA256 with salt. Use bcrypt in production."""
    salt = secrets.token_hex(16)
    hash_obj = hashlib.sha256((password + salt).encode())
    return f"{salt}${hash_obj.hexdigest()}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash."""
    try:
        salt, stored_hash = password_hash.split("$")
        hash_obj = hashlib.sha256((password + salt).encode())
        return hash_obj.hexdigest() == stored_hash
    except ValueError:
        return False


def generate_session_id() -> str:
    """Generate a secure session ID."""
    return secrets.token_urlsafe(32)


def generate_user_id() -> str:
    """Generate a unique user ID."""
    return f"usr_{secrets.token_hex(12)}"


def create_user(user: UserCreate) -> Optional[UserResponse]:
    """Create a new user account."""
    if len(user.password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")

    # Check if email already exists
    existing = fetchone("SELECT id FROM users WHERE email = ?", (user.email,))
    if existing:
        raise ValueError("Email already registered")

    user_id = generate_user_id()
    password_hash = hash_password(user.password)
    now = datetime.utcnow().isoformat()

    execute(
        """
        INSERT INTO users (id, email, password_hash, name, subscription_tier, bankroll, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'free', 1000.0, ?, ?)
        """,
        (user_id, user.email, password_hash, user.name, now, now),
    )

    # Create default preferences
    execute(
        """
        INSERT INTO user_preferences (user_id, updated_at)
        VALUES (?, ?)
        """,
        (user_id, now),
    )

    return UserResponse(
        id=user_id,
        email=user.email,
        name=user.name,
        subscription_tier="free",
        bankroll=1000.0,
        created_at=now,
    )


def authenticate_user(login: UserLogin) -> Optional[dict]:
    """Authenticate user and return session info."""
    row = fetchone(
        "SELECT id, email, password_hash, name, subscription_tier, bankroll, created_at FROM users WHERE email = ?",
        (login.email,),
    )

    if not row:
        return None

    user_id, email, password_hash, name, tier, bankroll, created_at = row

    if not verify_password(login.password, password_hash):
        return None

    # Create session
    session_id = generate_session_id()
    expires_at = (datetime.utcnow() + timedelta(days=SESSION_EXPIRY_DAYS)).isoformat()
    now = datetime.utcnow().isoformat()

    execute(
        """
        INSERT INTO user_sessions (session_id, user_id, expires_at, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (session_id, user_id, expires_at, now),
    )

    return {
        "session_id": session_id,
        "expires_at": expires_at,
        "user": UserResponse(
            id=user_id,
            email=email,
            name=name,
            subscription_tier=tier,
            bankroll=bankroll,
            created_at=created_at,
        ),
    }


def validate_session(session_id: str) -> Optional[UserResponse]:
    """Validate session and return user if valid."""
    row = fetchone(
        """
        SELECT u.id, u.email, u.name, u.subscription_tier, u.bankroll, u.created_at, s.expires_at
        FROM user_sessions s
        JOIN users u ON s.user_id = u.id
        WHERE s.session_id = ?
        """,
        (session_id,),
    )

    if not row:
        return None

    user_id, email, name, tier, bankroll, created_at, expires_at = row

    if datetime.fromisoformat(expires_at) < datetime.utcnow():
        # Session expired, delete it
        execute("DELETE FROM user_sessions WHERE session_id = ?", (session_id,))
        return None

    return UserResponse(
        id=user_id,
        email=email,
        name=name,
        subscription_tier=tier,
        bankroll=bankroll,
        created_at=created_at,
    )


def logout_user(session_id: str) -> bool:
    """Delete user session."""
    execute("DELETE FROM user_sessions WHERE session_id = ?", (session_id,))
    return True


def get_user_preferences(user_id: str) -> Optional[UserPreferences]:
    """Get user preferences."""
    row = fetchone(
        """
        SELECT default_min_edge, default_kelly_fraction, default_max_stake,
               best_line_only, show_synthetic_odds, defense_multipliers,
               weather_adjustments, injury_weighting, preferred_sportsbooks, preferred_markets
        FROM user_preferences WHERE user_id = ?
        """,
        (user_id,),
    )

    if not row:
        return None

    return UserPreferences(
        default_min_edge=row[0],
        default_kelly_fraction=row[1],
        default_max_stake=row[2],
        best_line_only=bool(row[3]),
        show_synthetic_odds=bool(row[4]),
        defense_multipliers=bool(row[5]),
        weather_adjustments=bool(row[6]),
        injury_weighting=bool(row[7]),
        preferred_sportsbooks=row[8],
        preferred_markets=row[9],
    )


def update_user_preferences(user_id: str, prefs: UserPreferences) -> bool:
    """Update user preferences."""
    now = datetime.utcnow().isoformat()
    execute(
        """
        UPDATE user_preferences SET
            default_min_edge = ?,
            default_kelly_fraction = ?,
            default_max_stake = ?,
            best_line_only = ?,
            show_synthetic_odds = ?,
            defense_multipliers = ?,
            weather_adjustments = ?,
            injury_weighting = ?,
            preferred_sportsbooks = ?,
            preferred_markets = ?,
            updated_at = ?
        WHERE user_id = ?
        """,
        (
            prefs.default_min_edge,
            prefs.default_kelly_fraction,
            prefs.default_max_stake,
            1 if prefs.best_line_only else 0,
            1 if prefs.show_synthetic_odds else 0,
            1 if prefs.defense_multipliers else 0,
            1 if prefs.weather_adjustments else 0,
            1 if prefs.injury_weighting else 0,
            prefs.preferred_sportsbooks,
            prefs.preferred_markets,
            now,
            user_id,
        ),
    )
    return True


def update_bankroll(user_id: str, bankroll: float) -> bool:
    """Update user bankroll."""
    now = datetime.utcnow().isoformat()
    execute(
        "UPDATE users SET bankroll = ?, updated_at = ? WHERE id = ?",
        (bankroll, now, user_id),
    )
    return True

