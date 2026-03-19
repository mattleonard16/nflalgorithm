"""Ingest supplemental modifier data for NCAAB bracket prediction.

Parses BartTorvik, coaching, experience, and momentum CSVs, computes
modifier factors, and updates ncaab_team_ratings with enhanced_rating.

Usage:
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python scripts/ingest_ncaab_modifiers.py --season 2026
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.db import execute, read_dataframe
from utils.ncaab_modifiers import (
    barttorvik_factor,
    coaching_factor,
    enhanced_composite_rating,
    experience_factor,
    momentum_factor,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


def _parse_csv(filepath: str) -> list[dict]:
    """Parse a CSV file, returning list of row dicts. Returns [] if file missing or empty."""
    path = Path(filepath)
    if not path.exists():
        log.warning("File not found, skipping: %s", filepath)
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_existing_ratings(season: int) -> dict[str, dict]:
    """Load current ncaab_team_ratings as dict keyed by team_name."""
    df = read_dataframe(
        "SELECT team_name, kenpom_rank, composite_rating, adj_t FROM ncaab_team_ratings WHERE season = ?",
        (season,),
    )
    result = {}
    for _, row in df.iterrows():
        result[row["team_name"]] = row.to_dict()
    return result


def ingest_modifiers(
    season: int,
    barttorvik_path: str | None = None,
    coaching_path: str | None = None,
    experience_path: str | None = None,
    momentum_path: str | None = None,
) -> int:
    """Ingest all modifier CSVs and update ncaab_team_ratings."""
    existing = _load_existing_ratings(season)
    if not existing:
        log.error("No existing ratings for season %d. Run ingest-ncaab first.", season)
        return 0

    # Build modifier lookup dicts per team
    bt_data: dict[str, dict] = {}
    coach_data: dict[str, dict] = {}
    exp_data: dict[str, dict] = {}
    mom_data: dict[str, dict] = {}

    if barttorvik_path:
        for row in _parse_csv(barttorvik_path):
            bt_data[row["team"].strip()] = row
        log.info("Loaded %d BartTorvik entries", len(bt_data))

    if coaching_path:
        for row in _parse_csv(coaching_path):
            coach_data[row["team"].strip()] = row
        log.info("Loaded %d coaching entries", len(coach_data))

    if experience_path:
        for row in _parse_csv(experience_path):
            exp_data[row["team"].strip()] = row
        log.info("Loaded %d experience entries", len(exp_data))

    if momentum_path:
        for row in _parse_csv(momentum_path):
            mom_data[row["team"].strip()] = row
        log.info("Loaded %d momentum entries", len(mom_data))

    updated = 0
    for team_name, ratings in existing.items():
        kp_rank = ratings.get("kenpom_rank") or 100
        base_cr = ratings.get("composite_rating") or 0.30

        # Compute each factor (default 1.0 if no data)
        bt = bt_data.get(team_name)
        bt_rank = int(bt["barttorvik_rank"]) if bt else kp_rank
        bt_em = float(bt["adj_em"]) if bt else None
        bt_f = barttorvik_factor(kp_rank, bt_rank)

        coach = coach_data.get(team_name)
        coach_wr = float(coach["win_rate"]) if coach else 0.500
        coach_wins = int(coach["tournament_wins"]) if coach else 0
        coach_f = coaching_factor(coach_wr)

        exp = exp_data.get(team_name)
        ret_min = float(exp["returning_minutes_pct"]) if exp else 0.50
        avg_yrs = float(exp["avg_years_experience"]) if exp else 2.0
        seniors = int(exp["seniors_count"]) if exp else 2
        exp_f = experience_factor(ret_min)

        mom = mom_data.get(team_name)
        l10w = int(mom["last_10_wins"]) if mom else 5
        l10l = int(mom["last_10_losses"]) if mom else 5
        conf_result = mom["conf_tourney_result"].strip() if mom else "quarterfinal"
        streak = int(mom["winning_streak"]) if mom else 0
        mom_f = momentum_factor(l10w, conf_result, streak)

        enhanced = enhanced_composite_rating(base_cr, bt_f, coach_f, exp_f, mom_f)

        execute(
            """UPDATE ncaab_team_ratings SET
                barttorvik_rank = ?, bt_adj_em = ?,
                coaching_win_rate = ?, coaching_tourney_wins = ?,
                returning_minutes_pct = ?, avg_years_experience = ?, seniors_count = ?,
                last_10_wins = ?, last_10_losses = ?,
                conf_tourney_result = ?, winning_streak = ?,
                bt_factor = ?, coaching_factor = ?,
                experience_factor = ?, momentum_factor = ?,
                enhanced_rating = ?
            WHERE team_name = ? AND season = ?""",
            (
                bt_rank, bt_em,
                coach_wr, coach_wins,
                ret_min, avg_yrs, seniors,
                l10w, l10l, conf_result, streak,
                round(bt_f, 4), round(coach_f, 4),
                round(exp_f, 4), round(mom_f, 4),
                round(enhanced, 6),
                team_name, season,
            ),
        )
        updated += 1

        # Log teams where modifiers shift rating significantly
        delta = enhanced - base_cr
        if abs(delta) > 0.01:
            direction = "+" if delta > 0 else ""
            log.info(
                "  %s: %.4f -> %.4f (%s%.4f) BT:%.3f Coach:%.3f Exp:%.3f Mom:%.3f",
                team_name, base_cr, enhanced, direction, delta, bt_f, coach_f, exp_f, mom_f,
            )

    log.info("Updated %d teams with modifier data", updated)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NCAAB modifier data")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--barttorvik", default="data/barttorvik_2026.csv")
    parser.add_argument("--coaching", default="data/coaching_tournament_2026.csv")
    parser.add_argument("--experience", default="data/team_experience_2026.csv")
    parser.add_argument("--momentum", default="data/team_momentum_2026.csv")
    args = parser.parse_args()

    from schema_migrations import MigrationManager
    import config as cfg
    MigrationManager(cfg.config.database.path).run()

    ingest_modifiers(
        season=args.season,
        barttorvik_path=args.barttorvik,
        coaching_path=args.coaching,
        experience_path=args.experience,
        momentum_path=args.momentum,
    )


if __name__ == "__main__":
    main()
