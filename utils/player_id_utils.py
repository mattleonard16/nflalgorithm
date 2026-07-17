"""Shared helpers for generating and parsing player identifiers."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
VALID_NFL_TEAMS = {
    'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
    'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
    'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
    'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
}

TEAM_TYPO_FIXES = {
    'jacson': 'JAX',
    'ptd': 'DAL',
}

TEAM_ALIASES = {
    "ARZ": "ARI",
    "JAC": "JAX",
    "KAN": "KC",
    "KCC": "KC",
    "GBP": "GB",
    "GNB": "GB",
    "NOS": "NO",
    "NOLA": "NO",
    "OAK": "LV",
    "LVR": "LV",
    "SD": "LAC",
    "SND": "LAC",
    "LACHARGERS": "LAC",
    "LOSANGELESCHARGERS": "LAC",
    "LACHARGER": "LAC",
    "LARAMS": "LAR",
    "LOSANGELESRAMS": "LAR",
    "STL": "LAR",
    "WSH": "WAS",
    "WFT": "WAS",
    "HST": "HOU",
    "HOUSTON": "HOU",
}


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_name(name: Optional[str]) -> str:
    """Normalize a raw player name for identifier construction."""
    if not name:
        return ""
    text = _strip_accents(str(name)).lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [token for token in text.split() if token and token not in NAME_SUFFIXES]

    collapsed = []
    buffer = []
    for token in tokens:
        if len(token) == 1:
            buffer.append(token)
        else:
            if buffer:
                collapsed.append("".join(buffer))
                buffer = []
            collapsed.append(token)
    if buffer:
        collapsed.append("".join(buffer))
    return " ".join(collapsed)


def canonicalize_team(team: Optional[str]) -> str:
    """Normalize a team string to uppercase abbreviation."""
    if not team:
        return ""
    token = re.sub(r"[^a-zA-Z0-9]", "", str(team)).upper()
    if not token:
        return ""
    
    # Check typo fixes first
    if token.lower() in TEAM_TYPO_FIXES:
        token = TEAM_TYPO_FIXES[token.lower()]
    
    # Apply aliases
    token = TEAM_ALIASES.get(token, token)
    
    # Validate against known teams
    if token not in VALID_NFL_TEAMS:
        # Could be a position code or garbage - return empty
        return ""
    
    return token


def make_player_id(name: Optional[str], team: Optional[str]) -> str:
    """Create a canonical player identifier of the form TEAM_normalizedname."""
    normalized_name = normalize_name(name).replace(" ", "_")
    team_token = canonicalize_team(team)
    if not team_token:  # Handle invalid/missing teams
        return normalized_name if normalized_name else ""
    return f"{team_token}_{normalized_name}" if normalized_name else team_token
