"""Define the 2026 NCAA tournament bracket structure.

Inserts all 65 games (2 First Four + 63 main bracket) into ncaab_bracket.
Also updates ncaab_team_ratings.region for tournament teams.

Usage:
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python scripts/define_ncaab_bracket.py --season 2026
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.db import execute, executemany, read_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _region_games(region: str, prefix: str, matchups: list[tuple]) -> list[tuple]:
    """Build all games for a region (R1 through R4).

    matchups: list of 8 tuples (team_a, seed_a, team_b, seed_b) for R1.
    Standard NCAA bracket wiring:
      R1 Slot 1 (1v16) + R1 Slot 2 (8v9) -> R2 Slot 1
      R1 Slot 3 (5v12) + R1 Slot 4 (4v13) -> R2 Slot 2
      R1 Slot 5 (6v11) + R1 Slot 6 (3v14) -> R2 Slot 3
      R1 Slot 7 (7v10) + R1 Slot 8 (2v15) -> R2 Slot 4
      R2 Slot 1 + R2 Slot 2 -> R3 Slot 1 (Sweet 16)
      R2 Slot 3 + R2 Slot 4 -> R3 Slot 2 (Sweet 16)
      R3 Slot 1 + R3 Slot 2 -> R4 Slot 1 (Elite 8)
    """
    games = []

    # Round 1: 8 games
    for i, (ta, sa, tb, sb) in enumerate(matchups, 1):
        r2_slot = (i + 1) // 2
        games.append((
            f"{prefix}_R1_{i}", region, 1, i,
            ta, sa, tb, sb,
            None, None, f"{prefix}_R2_{r2_slot}",
        ))

    # Round 2: 4 games
    for slot in range(1, 5):
        r1_a = slot * 2 - 1
        r1_b = slot * 2
        r3_slot = (slot + 1) // 2
        games.append((
            f"{prefix}_R2_{slot}", region, 2, slot,
            None, None, None, None,
            f"{prefix}_R1_{r1_a}", f"{prefix}_R1_{r1_b}",
            f"{prefix}_R3_{r3_slot}",
        ))

    # Round 3 (Sweet 16): 2 games
    for slot in range(1, 3):
        r2_a = slot * 2 - 1
        r2_b = slot * 2
        games.append((
            f"{prefix}_R3_{slot}", region, 3, slot,
            None, None, None, None,
            f"{prefix}_R2_{r2_a}", f"{prefix}_R2_{r2_b}",
            f"{prefix}_R4_1",
        ))

    # Round 4 (Elite 8): 1 game
    games.append((
        f"{prefix}_R4_1", region, 4, 1,
        None, None, None, None,
        f"{prefix}_R3_1", f"{prefix}_R3_2",
        None,  # feeds_game_id set below for Final Four
    ))

    return games


def define_bracket(season: int) -> int:
    """Insert the full 2026 bracket. Returns game count."""

    # ---- First Four (Round 0) ----
    first_four = [
        # FF0_1: PVAM vs Lehigh -> winner to South R1 Slot 1 as 16 seed
        (
            "FF0_1", "First Four", 0, 1,
            "Prairie View A&M", 16, "Lehigh", 16,
            None, None, "S_R1_1",
        ),
        # FF0_2: Miami OH vs SMU -> winner to Midwest R1 Slot 5 as 11 seed
        (
            "FF0_2", "First Four", 0, 2,
            "Miami OH", 11, "SMU", 11,
            None, None, "MW_R1_5",
        ),
    ]

    # ---- East Region ----
    east_matchups = [
        ("Duke", 1, "Siena", 16),
        ("Ohio State", 8, "TCU", 9),
        ("St. John's", 5, "Northern Iowa", 12),
        ("Kansas", 4, "Cal Baptist", 13),
        ("Louisville", 6, "South Florida", 11),
        ("Michigan State", 3, "North Dakota State", 14),
        ("UCLA", 7, "UCF", 10),
        ("UConn", 2, "Furman", 15),
    ]
    east_games = _region_games("East", "E", east_matchups)

    # ---- West Region ----
    west_matchups = [
        ("Arizona", 1, "LIU", 16),
        ("Villanova", 8, "Utah State", 9),
        ("Wisconsin", 5, "High Point", 12),
        ("Arkansas", 4, "Hawaii", 13),
        ("BYU", 6, "Texas", 11),
        ("Gonzaga", 3, "Kennesaw State", 14),
        ("Miami FL", 7, "Missouri", 10),
        ("Purdue", 2, "Queens", 15),
    ]
    west_games = _region_games("West", "W", west_matchups)

    # ---- South Region ----
    # Note: South R1 Slot 1 has Florida(1) vs First Four winner (16)
    south_matchups = [
        ("Florida", 1, None, 16),  # team_b filled by FF0_1 winner
        ("Clemson", 8, "Iowa", 9),
        ("Vanderbilt", 5, "McNeese", 12),
        ("Nebraska", 4, "Troy", 13),
        ("North Carolina", 6, "VCU", 11),
        ("Illinois", 3, "Penn", 14),
        ("Saint Mary's", 7, "Texas A&M", 10),
        ("Houston", 2, "Idaho", 15),
    ]
    south_games = _region_games("South", "S", south_matchups)
    # Fix S_R1_1: team_b comes from FF0_1 winner
    for i, g in enumerate(south_games):
        if g[0] == "S_R1_1":
            south_games[i] = g[:8] + (None, "FF0_1") + g[10:]
            break

    # ---- Midwest Region ----
    # Note: Midwest R1 Slot 5 has Tennessee(6) vs First Four winner (11)
    midwest_matchups = [
        ("Michigan", 1, "Howard", 16),
        ("Georgia", 8, "Saint Louis", 9),
        ("Texas Tech", 5, "Akron", 12),
        ("Alabama", 4, "Hofstra", 13),
        ("Tennessee", 6, None, 11),  # team_b filled by FF0_2 winner
        ("Virginia", 3, "Wright State", 14),
        ("Kentucky", 7, "Santa Clara", 10),
        ("Iowa State", 2, "Tennessee State", 15),
    ]
    midwest_games = _region_games("Midwest", "MW", midwest_matchups)
    # Fix MW_R1_5: team_b comes from FF0_2 winner
    for i, g in enumerate(midwest_games):
        if g[0] == "MW_R1_5":
            midwest_games[i] = g[:8] + (None, "FF0_2") + g[10:]
            break

    # ---- Final Four (Round 5) ----
    # East winner vs West winner
    final_four_games = [
        (
            "FF_1", "Final Four", 5, 1,
            None, None, None, None,
            "E_R4_1", "W_R4_1", "CHAMP",
        ),
        (
            "FF_2", "Final Four", 5, 2,
            None, None, None, None,
            "S_R4_1", "MW_R4_1", "CHAMP",
        ),
    ]

    # ---- Championship (Round 6) ----
    championship = [
        (
            "CHAMP", "Championship", 6, 1,
            None, None, None, None,
            "FF_1", "FF_2", None,
        ),
    ]

    # Update Elite 8 feeds_game_id to point to Final Four
    for games in [east_games, west_games, south_games, midwest_games]:
        for i, g in enumerate(games):
            if g[0].endswith("_R4_1"):
                prefix = g[0].split("_")[0]
                ff_game = "FF_1" if prefix in ("E", "W") else "FF_2"
                games[i] = g[:10] + (ff_game,)

    all_games = (
        first_four
        + east_games
        + west_games
        + south_games
        + midwest_games
        + final_four_games
        + championship
    )

    # Add season to each row
    rows = [g + (season,) for g in all_games]

    executemany(
        """INSERT OR REPLACE INTO ncaab_bracket
        (game_id, region, round, slot, team_a, seed_a, team_b, seed_b,
         prev_game_a, prev_game_b, feeds_game_id, season)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    log.info("Inserted %d bracket games for season %d", len(rows), season)

    # Update team regions in ncaab_team_ratings
    region_map = {
        "East": [m[0] for m in east_matchups] + [m[2] for m in east_matchups if m[2]],
        "West": [m[0] for m in west_matchups] + [m[2] for m in west_matchups if m[2]],
        "South": [m[0] for m in south_matchups] + [m[2] for m in south_matchups if m[2]],
        "Midwest": [m[0] for m in midwest_matchups] + [m[2] for m in midwest_matchups if m[2]],
    }
    # First Four teams go to their destination region
    region_map["South"].extend(["Prairie View A&M", "Lehigh"])
    region_map["Midwest"].extend(["Miami OH", "SMU"])

    for region_name, teams in region_map.items():
        for team in teams:
            execute(
                "UPDATE ncaab_team_ratings SET region = ? WHERE team_name = ? AND season = ?",
                (region_name, team, season),
            )

    # Verify
    df = read_dataframe(
        "SELECT region, round, COUNT(*) as games FROM ncaab_bracket "
        "WHERE season = ? GROUP BY region, round ORDER BY region, round",
        (season,),
    )
    log.info("Bracket summary:\n%s", df.to_string(index=False))

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Define NCAA tournament bracket")
    parser.add_argument("--season", type=int, default=2026)
    args = parser.parse_args()

    from schema_migrations import MigrationManager
    import config as cfg
    MigrationManager(cfg.config.database.path).run()

    count = define_bracket(args.season)
    log.info("Done: %d games defined for season %d", count, args.season)


if __name__ == "__main__":
    main()
