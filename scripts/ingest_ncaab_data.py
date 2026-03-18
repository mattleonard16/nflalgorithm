"""Ingest KenPom college basketball ratings from CSV export.

Parses the user's KenPom export, computes derived metrics (Pythagorean win%,
Trapezoid of Excellence, composite rating), and upserts to ncaab_team_ratings.

Usage:
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python scripts/ingest_ncaab_data.py --file data/kenpom_2026.csv --season 2026
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.db import executemany, read_dataframe
from utils.ncaab_ratings import composite_rating, compute_trapezoid_score, pyth_win_pct

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _parse_kenpom_csv(filepath: str) -> list[dict]:
    """Parse cleaned KenPom CSV into list of team dicts."""
    teams = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            teams.append({
                "kenpom_rank": int(row["rank"]),
                "team_name": row["team"].strip(),
                "seed": int(row["seed"]) if row.get("seed") else None,
                "conf": row["conf"].strip(),
                "wins": int(row["wins"]),
                "losses": int(row["losses"]),
                "adj_em": float(row["adj_em"]),
                "adj_oe": float(row["adj_oe"]),
                "adj_oe_rank": int(row["adj_oe_rank"]),
                "adj_de": float(row["adj_de"]),
                "adj_de_rank": int(row["adj_de_rank"]),
                "adj_t": float(row["adj_t"]),
                "luck": float(row["luck"]),
                "luck_rank": int(row["luck_rank"]) if row.get("luck_rank") else None,
                "sos_adj_em": float(row["sos_adj_em"]),
                "sos_adj_em_rank": int(row["sos_adj_em_rank"]) if row.get("sos_adj_em_rank") else None,
                "sos_oe": float(row["sos_oe"]) if row.get("sos_oe") else None,
                "sos_de": float(row["sos_de"]) if row.get("sos_de") else None,
                "ncsos_adj_em": float(row["ncsos_adj_em"]) if row.get("ncsos_adj_em") else None,
            })
    return teams


def _compute_derived_metrics(team: dict) -> dict:
    """Compute pyth_win, trapezoid_score, composite_rating for a team."""
    pw = pyth_win_pct(team["adj_oe"], team["adj_de"])

    trap = compute_trapezoid_score(
        adj_oe_rank=team["adj_oe_rank"],
        adj_de_rank=team["adj_de_rank"],
        sos_rank=team.get("sos_adj_em_rank") or 200,
        adj_em_rank=team["kenpom_rank"],
    )

    seed = team["seed"] if team["seed"] else 16
    cr = composite_rating(
        adj_em=team["adj_em"],
        adj_oe=team["adj_oe"],
        adj_de=team["adj_de"],
        pyth_win=pw,
        sos_adj_em=team.get("sos_adj_em") or 0.0,
        luck=team["luck"],
        seed=seed,
        trapezoid_score=trap,
    )

    return {**team, "pyth_win": pw, "trapezoid_score": trap, "composite_rating": cr}


def ingest(filepath: str, season: int) -> int:
    """Parse KenPom CSV and upsert to ncaab_team_ratings. Returns row count."""
    teams = _parse_kenpom_csv(filepath)
    log.info("Parsed %d teams from %s", len(teams), filepath)

    enriched = [_compute_derived_metrics(t) for t in teams]
    now = datetime.utcnow().isoformat()

    rows = [
        (
            t["team_name"], season, t["seed"], None, t["conf"],
            t["wins"], t["losses"], t["adj_em"], t["adj_oe"], t["adj_oe_rank"],
            t["adj_de"], t["adj_de_rank"], t["adj_t"], t["luck"],
            t.get("sos_adj_em"), t.get("sos_oe"), t.get("sos_de"),
            t.get("ncsos_adj_em"), t["pyth_win"], t["trapezoid_score"],
            t["composite_rating"], t["kenpom_rank"], now,
        )
        for t in enriched
    ]

    executemany(
        """INSERT OR REPLACE INTO ncaab_team_ratings
        (team_name, season, seed, region, conf,
         wins, losses, adj_em, adj_oe, adj_oe_rank,
         adj_de, adj_de_rank, adj_t, luck,
         sos_adj_em, sos_oe, sos_de, ncsos_adj_em,
         pyth_win, trapezoid_score, composite_rating, kenpom_rank, scraped_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    log.info("Upserted %d teams to ncaab_team_ratings", len(rows))

    # Print top-10 composite ratings
    df = read_dataframe(
        "SELECT team_name, seed, adj_em, composite_rating, trapezoid_score "
        "FROM ncaab_team_ratings WHERE season = ? ORDER BY composite_rating DESC LIMIT 10",
        (season,),
    )
    log.info("Top 10 by composite rating:\n%s", df.to_string(index=False))
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest KenPom CSV into ncaab_team_ratings")
    parser.add_argument("--file", required=True, help="Path to KenPom CSV")
    parser.add_argument("--season", type=int, default=2026, help="Season year (default: 2026)")
    args = parser.parse_args()

    from schema_migrations import MigrationManager
    import config as cfg
    MigrationManager(cfg.config.database.path).run()

    count = ingest(args.file, args.season)
    log.info("Done: %d teams ingested for season %d", count, args.season)


if __name__ == "__main__":
    main()
