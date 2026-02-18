#!/usr/bin/env python3
"""
NBA Player Prop Odds Scraper
Retrieves current NBA player prop odds from The Odds API (basketball_nba).
"""

import argparse
import logging
import os
import random
import sys
import time
import unicodedata
from datetime import datetime, date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.db import execute, get_connection, read_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_nba"
MAX_EVENTS = 15

# Odds API market key -> internal market name
MARKET_MAP: Dict[str, str] = {
    "player_points": "pts",
    "player_rebounds": "reb",
    "player_assists": "ast",
    "player_threes": "fg3m",
    "player_blocks": "blk",
    "player_steals": "stl",
    "player_turnovers": "tov",
}

_SUFFIX_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v"}

_MARKET_BASES = {
    "pts": 22.5,
    "reb": 6.5,
    "ast": 5.5,
    "fg3m": 2.5,
    "blk": 1.5,
    "stl": 1.5,
    "tov": 2.5,
}

_FALLBACK_PLAYERS = [
    ("LeBron James", "LAL"),
    ("Stephen Curry", "GSW"),
    ("Kevin Durant", "PHX"),
    ("Nikola Jokic", "DEN"),
    ("Giannis Antetokounmpo", "MIL"),
    ("Luka Doncic", "DAL"),
    ("Joel Embiid", "PHI"),
    ("Jayson Tatum", "BOS"),
    ("Devin Booker", "PHX"),
    ("Damian Lillard", "MIL"),
]

_FALLBACK_BOOKS = ["DraftKings", "FanDuel", "BetMGM", "Caesars"]


# ── Name normalization ────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """Lowercase, strip accents, remove suffixes, periods, apostrophes."""
    nfd = unicodedata.normalize("NFD", name)
    ascii_name = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    ascii_name = ascii_name.replace("'", " ").replace(".", "").lower()
    tokens = [t for t in ascii_name.split() if t not in _SUFFIX_TOKENS]
    return " ".join(tokens)


# ── Player lookup / matching ──────────────────────────────────────────────────

def _build_player_lookup() -> Dict[str, int]:
    """
    Return {normalized_name: player_id} from nba_player_game_logs.
    Reads all seasons; later seasons win on duplicate normalized names.
    """
    try:
        df = read_dataframe(
            "SELECT DISTINCT player_id, player_name FROM nba_player_game_logs "
            "ORDER BY season ASC"
        )
        lookup: Dict[str, int] = {}
        for row in df.to_dict("records"):
            key = _normalize_name(row["player_name"])
            lookup[key] = int(row["player_id"])
        return lookup
    except Exception as exc:
        logger.warning("Could not build player lookup: %s", exc)
        return {}


def _match_player(raw_name: str, lookup: Dict[str, int]) -> Optional[int]:
    """
    Return player_id for raw_name, or None if unmatched.

    Tier 1: exact normalized name match in lookup dict.
    Tier 2: fuzzy SequenceMatcher ratio >= 0.85 across all keys.
    """
    normalized = _normalize_name(raw_name)

    # Tier 1 – exact
    if normalized in lookup:
        return lookup[normalized]

    # Tier 2 – fuzzy
    best_ratio = 0.0
    best_key = None
    for key in lookup:
        ratio = SequenceMatcher(None, normalized, key).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_key = key

    if best_ratio >= 0.85 and best_key is not None:
        logger.debug(
            "Fuzzy match: '%s' -> '%s' (ratio=%.2f)", raw_name, best_key, best_ratio
        )
        return lookup[best_key]

    logger.warning("Unmatched player: '%s' (best_ratio=%.2f)", raw_name, best_ratio)
    return None


# ── Season derivation ─────────────────────────────────────────────────────────

def _season_from_date(game_date: date) -> int:
    """NBA season starts in October: Oct 2024 game -> season 2024."""
    return game_date.year if game_date.month >= 10 else game_date.year - 1


# ── HTTP layer (patchable) ────────────────────────────────────────────────────

def _fetch_odds_api(game_date: str, api_key: str) -> List[Dict]:
    """
    Fetch raw event + odds data from The Odds API for a given date string.

    Returns a list of event dicts, each already augmented with bookmaker odds,
    ready for _parse_events(). Uses a requests.Session with rate limiting and
    up to 3 retry attempts with exponential backoff.

    This function is intentionally isolated so tests can patch it:
        patch("scripts.scrape_nba_odds._fetch_odds_api")
    """
    session = requests.Session()
    results: List[Dict] = []

    # Step 1 – list events
    events = _get_with_retry(
        session,
        f"{BASE_URL}/sports/{SPORT_KEY}/events",
        {"apiKey": api_key},
    )
    if not events:
        logger.error("Failed to fetch NBA events.")
        return []

    logger.info("Found %d upcoming NBA events", len(events))
    time.sleep(0.3)

    # Step 2 – per-event, per-market odds
    markets = list(MARKET_MAP.keys())
    for event in events[:MAX_EVENTS]:
        event_id = event.get("id", "")
        augmented = dict(event)
        augmented["bookmakers"] = []

        for api_market in markets:
            payload = _get_with_retry(
                session,
                f"{BASE_URL}/sports/{SPORT_KEY}/events/{event_id}/odds",
                {
                    "apiKey": api_key,
                    "regions": "us",
                    "markets": api_market,
                    "oddsFormat": "american",
                    "dateFormat": "iso",
                },
            )
            time.sleep(0.3)
            if payload:
                for bm in payload.get("bookmakers", []):
                    augmented["bookmakers"].append(bm)

        results.append(augmented)

    return results


def _get_with_retry(
    session: requests.Session,
    url: str,
    params: Dict,
    max_attempts: int = 3,
) -> Optional[Dict]:
    """GET with exponential backoff. Returns JSON or None."""
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            if attempt >= max_attempts:
                logger.error("Request failed after %d attempts: %s — %s", max_attempts, url, exc)
                return None
            backoff = 0.5 * (2 ** (attempt - 1))
            logger.warning(
                "Attempt %d/%d failed (%s). Retrying in %.1fs",
                attempt,
                max_attempts,
                exc,
                backoff,
            )
            time.sleep(backoff)
    return None


# ── Parsing ───────────────────────────────────────────────────────────────────

def _parse_events(events: List[Dict], season: int) -> List[Dict]:
    """
    Parse raw Odds API event list into rows matching the nba_odds schema.

    Each row has: event_id, season, game_date, player_id, player_name,
    team, market, sportsbook, line, over_price, under_price, as_of.
    """
    if not events:
        return []

    lookup = _build_player_lookup()
    as_of = datetime.utcnow().isoformat()
    rows: List[Dict] = []

    for event in events:
        event_id = event.get("id", "")
        commence_time = event.get("commence_time", "")

        try:
            game_date = datetime.fromisoformat(
                commence_time.replace("Z", "+00:00")
            ).strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            game_date = datetime.utcnow().strftime("%Y-%m-%d")

        seen: set = set()

        for bookmaker in event.get("bookmakers", []):
            sportsbook = bookmaker.get("title", "Unknown")

            for market_data in bookmaker.get("markets", []):
                api_market = market_data.get("key", "")
                internal_market = MARKET_MAP.get(api_market)
                if internal_market is None:
                    continue

                outcomes = market_data.get("outcomes", [])
                player_descs = {
                    o.get("description", "") for o in outcomes if o.get("description")
                }

                for player_desc in player_descs:
                    dedup_key = (player_desc, internal_market, sportsbook)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    over_outcome = next(
                        (
                            o for o in outcomes
                            if o.get("description") == player_desc and o.get("name") == "Over"
                        ),
                        None,
                    )
                    under_outcome = next(
                        (
                            o for o in outcomes
                            if o.get("description") == player_desc and o.get("name") == "Under"
                        ),
                        None,
                    )

                    if over_outcome is None:
                        continue

                    line = over_outcome.get("point") or (
                        under_outcome.get("point") if under_outcome else None
                    )
                    if line is None:
                        continue

                    player_id = _match_player(player_desc, lookup)
                    over_price = int(over_outcome["price"])
                    under_price = int(under_outcome["price"]) if under_outcome else None

                    rows.append({
                        "event_id": event_id,
                        "season": season,
                        "game_date": game_date,
                        "player_id": player_id,
                        "player_name": player_desc,
                        "team": None,
                        "market": internal_market,
                        "sportsbook": sportsbook,
                        "line": float(line),
                        "over_price": over_price,
                        "under_price": under_price,
                        "as_of": as_of,
                    })

    logger.info("Parsed %d NBA odds rows from %d events", len(rows), len(events))
    return rows


# ── Synthetic fallback ────────────────────────────────────────────────────────

def generate_synthetic_odds(game_date: str, season: int) -> List[Dict]:
    """
    Generate plausible NBA prop rows without a live API call.

    Reads player names from nba_player_game_logs when available.
    All rows have sportsbook="synthetic". Returns [] when no game logs exist.
    """
    as_of = datetime.utcnow().isoformat()
    markets = list(_MARKET_BASES.keys())

    # Try real players from DB
    try:
        df = read_dataframe(
            "SELECT DISTINCT player_name, team_abbreviation FROM nba_player_game_logs "
            "WHERE season = ? LIMIT 50",
            params=(season,),
        )
        pool = [(r["player_name"], r["team_abbreviation"]) for r in df.to_dict("records")]
    except Exception:
        pool = []

    if not pool:
        logger.warning(
            "No game logs found for season=%s. generate_synthetic_odds returning [].", season
        )
        return []

    rng = random.Random(42)
    rows: List[Dict] = []

    for _ in range(30):
        player_name, team = rng.choice(pool)
        market = rng.choice(markets)
        base = _MARKET_BASES[market]
        line = round(base + rng.uniform(-3.0, 3.0) * 2, 1)
        over_price = rng.choice([-120, -115, -110, -105, 100, 105])
        under_price = rng.choice([-120, -115, -110, -105, 100, 105])
        rows.append({
            "event_id": f"synthetic_{rng.randint(1000, 9999)}",
            "season": season,
            "game_date": game_date,
            "player_id": None,
            "player_name": player_name,
            "team": team,
            "market": market,
            "sportsbook": "synthetic",
            "line": line,
            "over_price": over_price,
            "under_price": under_price,
            "as_of": as_of,
        })

    logger.info("Generated %d synthetic NBA prop rows", len(rows))
    return rows


# ── Top-level entry point ─────────────────────────────────────────────────────

def scrape_nba_odds(game_date: str, season: int) -> List[Dict]:
    """
    Fetch NBA player prop odds for *game_date* (YYYY-MM-DD) and *season*.

    Reads ODDS_API_KEY env var. If missing or empty, returns synthetic data
    via generate_synthetic_odds(). Otherwise calls _fetch_odds_api() then
    _parse_events().
    """
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        try:
            from config import config
            api_key = getattr(getattr(config, "api", None), "odds_api_key", "") or ""
        except ImportError:
            api_key = ""

    if not api_key:
        logger.warning("No Odds API key found. Returning synthetic fallback data.")
        return generate_synthetic_odds(game_date, season)

    events = _fetch_odds_api(game_date, api_key)
    return _parse_events(events, season)


# ── Database write ────────────────────────────────────────────────────────────

def upsert_odds_rows(rows: List[Dict]) -> int:
    """
    INSERT OR REPLACE rows into nba_odds. Returns count of rows processed.
    upsert_odds_rows([]) returns 0 (noop).
    """
    if not rows:
        return 0

    written = 0
    with get_connection() as conn:
        for row in rows:
            try:
                execute(
                    """
                    INSERT OR REPLACE INTO nba_odds
                        (event_id, season, game_date, player_id, player_name,
                         team, market, sportsbook, line, over_price, under_price, as_of)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["event_id"],
                        row["season"],
                        row["game_date"],
                        row.get("player_id"),
                        row["player_name"],
                        row.get("team"),
                        row["market"],
                        row["sportsbook"],
                        row["line"],
                        row["over_price"],
                        row.get("under_price"),
                        row["as_of"],
                    ),
                    conn=conn,
                )
                written += 1
            except Exception as exc:
                logger.warning("Failed to save row for '%s': %s", row.get("player_name"), exc)
        try:
            conn.commit()
        except Exception:
            pass

    logger.info("Saved %d/%d NBA odds rows to nba_odds.", written, len(rows))
    return written


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape NBA player prop odds from The Odds API.")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target game date (YYYY-MM-DD). Defaults to today.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    target_date_str = args.date or date.today().isoformat()
    try:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    except ValueError:
        logger.error("Invalid date format: %s. Use YYYY-MM-DD.", target_date_str)
        sys.exit(1)

    season = _season_from_date(target_date)
    logger.info("Scraping NBA odds for date=%s, season=%d", target_date_str, season)

    rows = scrape_nba_odds(target_date_str, season)
    count = upsert_odds_rows(rows)
    logger.info("Done. %d rows written to nba_odds.", count)


if __name__ == "__main__":
    main()
