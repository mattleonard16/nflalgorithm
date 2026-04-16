"""Seed demo data for all dashboard pages.

Run with DEMO_MODE=true or directly:
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db python scripts/seed_demo_data.py
"""

from __future__ import annotations

import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.db import get_connection

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")

NFL_PLAYERS = [
    ("patrick-mahomes", "Patrick Mahomes", "QB", "KC"),
    ("josh-allen", "Josh Allen", "QB", "BUF"),
    ("jalen-hurts", "Jalen Hurts", "QB", "PHI"),
    ("tyreek-hill", "Tyreek Hill", "WR", "MIA"),
    ("stefon-diggs", "Stefon Diggs", "WR", "HOU"),
    ("davante-adams", "Davante Adams", "WR", "NYJ"),
    ("travis-kelce", "Travis Kelce", "TE", "KC"),
    ("sam-laporta", "Sam LaPorta", "TE", "DET"),
    ("christian-mccaffrey", "Christian McCaffrey", "RB", "SF"),
    ("saquon-barkley", "Saquon Barkley", "RB", "PHI"),
    ("derrick-henry", "Derrick Henry", "RB", "BAL"),
    ("deandre-hopkins", "DeAndre Hopkins", "WR", "TEN"),
    ("ceedee-lamb", "CeeDee Lamb", "WR", "DAL"),
    ("justin-jefferson", "Justin Jefferson", "WR", "MIN"),
    ("tee-higgins", "Tee Higgins", "WR", "CIN"),
    ("lamar-jackson", "Lamar Jackson", "QB", "BAL"),
    ("joe-burrow", "Joe Burrow", "QB", "CIN"),
    ("jaylen-waddle", "Jaylen Waddle", "WR", "MIA"),
    ("george-kittle", "George Kittle", "TE", "SF"),
    ("tony-pollard", "Tony Pollard", "RB", "TEN"),
]

NBA_PLAYERS = [
    ("lebron-james", "LeBron James", "F", "LAL"),
    ("stephen-curry", "Stephen Curry", "G", "GSW"),
    ("giannis-antetokounmpo", "Giannis Antetokounmpo", "F", "MIL"),
    ("nikola-jokic", "Nikola Jokic", "C", "DEN"),
    ("luka-doncic", "Luka Doncic", "G", "DAL"),
    ("jayson-tatum", "Jayson Tatum", "F", "BOS"),
    ("devin-booker", "Devin Booker", "G", "PHX"),
    ("joel-embiid", "Joel Embiid", "C", "PHI"),
    ("kevin-durant", "Kevin Durant", "F", "PHX"),
    ("damian-lillard", "Damian Lillard", "G", "MIL"),
    ("anthony-davis", "Anthony Davis", "C", "LAL"),
    ("bam-adebayo", "Bam Adebayo", "C", "MIA"),
    ("pascal-siakam", "Pascal Siakam", "F", "IND"),
    ("tyrese-haliburton", "Tyrese Haliburton", "G", "IND"),
    ("shai-gilgeous-alexander", "Shai Gilgeous-Alexander", "G", "OKC"),
    ("donovan-mitchell", "Donovan Mitchell", "G", "CLE"),
    ("de-aaron-fox", "De'Aaron Fox", "G", "SAC"),
    ("anthony-edwards", "Anthony Edwards", "G", "MIN"),
    ("darius-garland", "Darius Garland", "G", "CLE"),
    ("karl-anthony-towns", "Karl-Anthony Towns", "C", "NYK"),
]

NFL_MARKETS = ["passing_yards", "rushing_yards", "receiving_yards", "receptions", "passing_tds"]
NBA_MARKETS = ["pts", "reb", "ast", "fg3m"]
SPORTSBOOKS = ["DraftKings", "FanDuel", "BetMGM", "Caesars"]
NFL_OPPONENTS = ["DAL", "NYG", "PHI", "WAS", "CHI", "GB", "MIN", "DET", "SF", "LAR", "SEA", "ARI"]


def _market_defaults(position: str, market: str) -> tuple[float, float]:
    """Return (mu, sigma) defaults for a market/position combination."""
    defaults = {
        ("QB", "passing_yards"): (265.0, 42.0),
        ("QB", "rushing_yards"): (28.0, 14.0),
        ("QB", "passing_tds"): (1.85, 0.9),
        ("WR", "receiving_yards"): (68.0, 22.0),
        ("WR", "receptions"): (5.2, 2.1),
        ("TE", "receiving_yards"): (52.0, 18.0),
        ("TE", "receptions"): (4.1, 1.8),
        ("RB", "rushing_yards"): (72.0, 28.0),
        ("RB", "receiving_yards"): (28.0, 14.0),
        ("RB", "receptions"): (3.0, 1.5),
    }
    return defaults.get((position, market), (45.0, 15.0))


def _nba_market_defaults(market: str) -> tuple[float, float]:
    return {
        "pts": (22.0, 6.0),
        "reb": (7.5, 2.5),
        "ast": (5.0, 2.0),
        "fg3m": (2.2, 1.2),
    }.get(market, (10.0, 3.0))


def seed_player_stats_enhanced(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_stats_enhanced (
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            player_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            position TEXT NOT NULL,
            team TEXT NOT NULL,
            passing_yards REAL DEFAULT 0,
            rushing_yards REAL DEFAULT 0,
            receiving_yards REAL DEFAULT 0,
            receptions REAL DEFAULT 0,
            passing_tds REAL DEFAULT 0,
            rushing_tds REAL DEFAULT 0,
            receiving_tds REAL DEFAULT 0,
            targets REAL DEFAULT 0,
            snap_pct REAL DEFAULT 0,
            PRIMARY KEY (season, week, player_id)
        )
        """
    )
    rows = []
    for season in (2024, 2025):
        for week in range(1, 19):
            for pid, name, pos, team in NFL_PLAYERS:
                mu_pass, _ = _market_defaults(pos, "passing_yards")
                mu_rush, _ = _market_defaults(pos, "rushing_yards")
                mu_recv, _ = _market_defaults(pos, "receiving_yards")
                mu_rec, _ = _market_defaults(pos, "receptions")
                mu_ptd, _ = _market_defaults(pos, "passing_tds")
                rows.append(
                    (
                        season,
                        week,
                        pid,
                        name,
                        pos,
                        team,
                        round(random.gauss(mu_pass, 40), 1) if pos == "QB" else 0.0,
                        round(max(0, random.gauss(mu_rush, 15)), 1),
                        round(max(0, random.gauss(mu_recv, 20)), 1) if pos != "QB" else 0.0,
                        round(max(0, random.gauss(mu_rec, 2)), 1) if pos != "QB" else 0.0,
                        round(max(0, random.gauss(mu_ptd, 0.8)), 1) if pos == "QB" else 0.0,
                        0.0,
                        0.0,
                        round(max(0, random.gauss(mu_rec + 2, 2)), 1) if pos != "QB" else 0.0,
                        round(min(1.0, max(0.4, random.gauss(0.75, 0.1))), 2),
                    )
                )
    conn.executemany(
        """
        INSERT OR REPLACE INTO player_stats_enhanced
        (season, week, player_id, player_name, position, team,
         passing_yards, rushing_yards, receiving_yards, receptions,
         passing_tds, rushing_tds, receiving_tds, targets, snap_pct)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )


def seed_weekly_projections(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS weekly_projections (
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            player_id TEXT NOT NULL,
            team TEXT NOT NULL,
            opponent TEXT NOT NULL,
            market TEXT NOT NULL,
            mu REAL NOT NULL,
            sigma REAL NOT NULL,
            context_sensitivity REAL DEFAULT 0,
            pass_attempts_predicted REAL DEFAULT 0,
            yards_per_attempt_predicted REAL DEFAULT 0,
            model_version TEXT NOT NULL,
            featureset_hash TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            PRIMARY KEY (season, week, player_id, market)
        )
        """
    )
    rows = []
    season, week = 2025, 13
    for pid, _name, pos, team in NFL_PLAYERS:
        opponent = random.choice([o for o in NFL_OPPONENTS if o != team])
        for market in NFL_MARKETS:
            base_mu, base_sigma = _market_defaults(pos, market)
            mu = round(max(0, random.gauss(base_mu, base_sigma * 0.15)), 2)
            sigma = round(base_sigma * random.uniform(0.85, 1.15), 2)
            rows.append(
                (
                    season,
                    week,
                    pid,
                    team,
                    opponent,
                    market,
                    mu,
                    sigma,
                    round(random.uniform(0.1, 0.9), 3),
                    round(random.uniform(30, 45), 1) if pos == "QB" else 0.0,
                    round(random.uniform(6.5, 8.5), 2) if pos == "QB" else 0.0,
                    "demo_v1",
                    "demo_hash",
                    datetime.now(timezone.utc).isoformat(),
                )
            )
    conn.executemany(
        """
        INSERT OR REPLACE INTO weekly_projections
        (season, week, player_id, team, opponent, market, mu, sigma,
         context_sensitivity, pass_attempts_predicted, yards_per_attempt_predicted,
         model_version, featureset_hash, generated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )


def seed_weekly_odds(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS weekly_odds (
            event_id TEXT NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            player_id TEXT NOT NULL,
            market TEXT NOT NULL,
            sportsbook TEXT NOT NULL,
            line REAL NOT NULL,
            price INTEGER NOT NULL,
            as_of TEXT NOT NULL,
            PRIMARY KEY (event_id, player_id, market, sportsbook, as_of)
        )
        """
    )
    rows = []
    season, week = 2025, 13
    as_of = datetime.now(timezone.utc).isoformat()
    for pid, _name, pos, _team in NFL_PLAYERS:
        for market in NFL_MARKETS:
            base_mu, base_sigma = _market_defaults(pos, market)
            for book in SPORTSBOOKS:
                line = round(base_mu + random.gauss(0, base_sigma * 0.1), 1)
                price = random.choice([-115, -110, -105, 100, 105, 110])
                event_id = f"demo_{season}_{week}_{pid}_{market}_{book}"
                rows.append((event_id, season, week, pid, market, book, line, price, as_of))
    conn.executemany(
        """
        INSERT OR REPLACE INTO weekly_odds
        (event_id, season, week, player_id, market, sportsbook, line, price, as_of)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )


def seed_materialized_value_view(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS materialized_value_view (
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            player_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            position TEXT NOT NULL,
            team TEXT NOT NULL,
            opponent TEXT NOT NULL,
            market TEXT NOT NULL,
            sportsbook TEXT NOT NULL,
            event_id TEXT NOT NULL,
            line REAL NOT NULL,
            price INTEGER NOT NULL,
            mu REAL NOT NULL,
            sigma REAL NOT NULL,
            edge REAL NOT NULL,
            kelly_fraction REAL NOT NULL,
            confidence REAL NOT NULL,
            bet_direction TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            PRIMARY KEY (season, week, player_id, market, sportsbook)
        )
        """
    )
    rows = []
    season, week = 2025, 13
    for pid, name, pos, team in NFL_PLAYERS:
        opponent = random.choice([o for o in NFL_OPPONENTS if o != team])
        for market in NFL_MARKETS:
            base_mu, base_sigma = _market_defaults(pos, market)
            mu = round(max(0, random.gauss(base_mu, base_sigma * 0.1)), 2)
            sigma = round(base_sigma * random.uniform(0.9, 1.1), 2)
            book = random.choice(SPORTSBOOKS)
            line = round(mu + random.gauss(0, base_sigma * 0.15), 1)
            price = random.choice([-115, -110, -105, 100, 105])
            edge = round(random.uniform(0.05, 0.25), 4)
            kelly = round(edge * random.uniform(0.2, 0.35), 4)
            confidence = round(random.uniform(0.60, 0.92), 3)
            direction = "over" if mu > line else "under"
            event_id = f"demo_{season}_{week}_{pid}_{market}_{book}"
            rows.append(
                (
                    season,
                    week,
                    pid,
                    name,
                    pos,
                    team,
                    opponent,
                    market,
                    book,
                    event_id,
                    line,
                    price,
                    mu,
                    sigma,
                    edge,
                    kelly,
                    confidence,
                    direction,
                    datetime.now(timezone.utc).isoformat(),
                )
            )
    conn.executemany(
        """
        INSERT OR REPLACE INTO materialized_value_view
        (season, week, player_id, player_name, position, team, opponent,
         market, sportsbook, event_id, line, price, mu, sigma, edge,
         kelly_fraction, confidence, bet_direction, generated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )


def seed_nba_player_game_logs(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS nba_player_game_logs (
            game_id TEXT NOT NULL,
            game_date TEXT NOT NULL,
            season_year INTEGER NOT NULL,
            player_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            team_abbreviation TEXT NOT NULL,
            matchup TEXT NOT NULL,
            pts REAL DEFAULT 0,
            reb REAL DEFAULT 0,
            ast REAL DEFAULT 0,
            fg3m REAL DEFAULT 0,
            min_played REAL DEFAULT 0,
            PRIMARY KEY (game_id, player_id)
        )
        """
    )
    rows = []
    base_date = datetime(2025, 1, 1)
    for game_num in range(60):
        game_date = (base_date + timedelta(days=game_num)).strftime("%Y-%m-%d")
        for pid, name, _pos, team in NBA_PLAYERS:
            mu_pts, sig_pts = _nba_market_defaults("pts")
            mu_reb, sig_reb = _nba_market_defaults("reb")
            mu_ast, sig_ast = _nba_market_defaults("ast")
            mu_fg3, sig_fg3 = _nba_market_defaults("fg3m")
            game_id = f"demo_nba_{game_date}_{pid}"
            opponent = random.choice(["LAL", "GSW", "BOS", "MIL", "PHX", "DEN", "MIA", "PHI"])
            rows.append(
                (
                    game_id,
                    game_date,
                    2025,
                    pid,
                    name,
                    team,
                    f"{team} vs. {opponent}",
                    round(max(0, random.gauss(mu_pts, sig_pts)), 1),
                    round(max(0, random.gauss(mu_reb, sig_reb)), 1),
                    round(max(0, random.gauss(mu_ast, sig_ast)), 1),
                    round(max(0, random.gauss(mu_fg3, sig_fg3)), 1),
                    round(random.uniform(28, 38), 1),
                )
            )
    conn.executemany(
        """
        INSERT OR REPLACE INTO nba_player_game_logs
        (game_id, game_date, season_year, player_id, player_name,
         team_abbreviation, matchup, pts, reb, ast, fg3m, min_played)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )


def seed_nba_projections(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS nba_projections (
            game_date TEXT NOT NULL,
            player_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            team_abbreviation TEXT NOT NULL,
            market TEXT NOT NULL,
            mu REAL NOT NULL,
            sigma REAL NOT NULL,
            model_version TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            PRIMARY KEY (game_date, player_id, market)
        )
        """
    )
    rows = []
    game_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for pid, name, _pos, team in NBA_PLAYERS:
        for market in NBA_MARKETS:
            base_mu, base_sigma = _nba_market_defaults(market)
            mu = round(max(0, random.gauss(base_mu, base_sigma * 0.1)), 2)
            sigma = round(base_sigma * random.uniform(0.85, 1.15), 2)
            rows.append(
                (game_date, pid, name, team, market, mu, sigma, "demo_v1", datetime.now(timezone.utc).isoformat())
            )
    conn.executemany(
        """
        INSERT OR REPLACE INTO nba_projections
        (game_date, player_id, player_name, team_abbreviation, market, mu, sigma, model_version, generated_at)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )


def seed_nba_odds(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS nba_odds (
            event_id TEXT NOT NULL,
            game_date TEXT NOT NULL,
            player_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            market TEXT NOT NULL,
            sportsbook TEXT NOT NULL,
            line REAL NOT NULL,
            price INTEGER NOT NULL,
            as_of TEXT NOT NULL,
            PRIMARY KEY (event_id, player_id, market, sportsbook)
        )
        """
    )
    rows = []
    game_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    as_of = datetime.now(timezone.utc).isoformat()
    for pid, name, _pos, _team in NBA_PLAYERS:
        for market in NBA_MARKETS:
            base_mu, base_sigma = _nba_market_defaults(market)
            for book in SPORTSBOOKS:
                line = round(base_mu + random.gauss(0, base_sigma * 0.1), 1)
                price = random.choice([-115, -110, -105, 100, 105, 110])
                event_id = f"demo_nba_{game_date}_{pid}_{market}_{book}"
                rows.append((event_id, game_date, pid, name, market, book, line, price, as_of))
    conn.executemany(
        """
        INSERT OR REPLACE INTO nba_odds
        (event_id, game_date, player_id, player_name, market, sportsbook, line, price, as_of)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )


def seed_nba_materialized_value_view(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS nba_materialized_value_view (
            game_date TEXT NOT NULL,
            player_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            team_abbreviation TEXT NOT NULL,
            market TEXT NOT NULL,
            sportsbook TEXT NOT NULL,
            event_id TEXT NOT NULL,
            line REAL NOT NULL,
            price INTEGER NOT NULL,
            mu REAL NOT NULL,
            sigma REAL NOT NULL,
            edge REAL NOT NULL,
            kelly_fraction REAL NOT NULL,
            confidence REAL NOT NULL,
            bet_direction TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            PRIMARY KEY (game_date, player_id, market, sportsbook)
        )
        """
    )
    rows = []
    game_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for pid, name, _pos, team in NBA_PLAYERS:
        for market in NBA_MARKETS:
            base_mu, base_sigma = _nba_market_defaults(market)
            mu = round(max(0, random.gauss(base_mu, base_sigma * 0.1)), 2)
            sigma = round(base_sigma * random.uniform(0.9, 1.1), 2)
            book = random.choice(SPORTSBOOKS)
            line = round(mu + random.gauss(0, base_sigma * 0.15), 1)
            price = random.choice([-115, -110, -105, 100, 105])
            edge = round(random.uniform(0.06, 0.22), 4)
            kelly = round(edge * random.uniform(0.2, 0.3), 4)
            confidence = round(random.uniform(0.62, 0.90), 3)
            direction = "over" if mu > line else "under"
            event_id = f"demo_nba_{game_date}_{pid}_{market}_{book}"
            rows.append(
                (
                    game_date,
                    pid,
                    name,
                    team,
                    market,
                    book,
                    event_id,
                    line,
                    price,
                    mu,
                    sigma,
                    edge,
                    kelly,
                    confidence,
                    direction,
                    datetime.now(timezone.utc).isoformat(),
                )
            )
    conn.executemany(
        """
        INSERT OR REPLACE INTO nba_materialized_value_view
        (game_date, player_id, player_name, team_abbreviation, market, sportsbook,
         event_id, line, price, mu, sigma, edge, kelly_fraction, confidence,
         bet_direction, generated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )


def main() -> None:
    random.seed(42)
    print("Seeding demo data...")
    with get_connection() as conn:
        seed_player_stats_enhanced(conn)
        print("  player_stats_enhanced: done")
        seed_weekly_projections(conn)
        print("  weekly_projections: done")
        seed_weekly_odds(conn)
        print("  weekly_odds: done")
        seed_materialized_value_view(conn)
        print("  materialized_value_view: done")
        seed_nba_player_game_logs(conn)
        print("  nba_player_game_logs: done")
        seed_nba_projections(conn)
        print("  nba_projections: done")
        seed_nba_odds(conn)
        print("  nba_odds: done")
        seed_nba_materialized_value_view(conn)
        print("  nba_materialized_value_view: done")
        conn.commit()
    print("Demo data seeded successfully.")


if __name__ == "__main__":
    main()
