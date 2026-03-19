"""March Madness bracket predictor.

Loads ratings + bracket from SQLite, simulates round-by-round using log5,
persists predictions, and renders CLI (rich) + HTML (jinja2) output.

Usage:
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python models/ncaab/bracket_predictor.py --season 2026 --cli --html
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.db import executemany, read_dataframe
from utils.ncaab_ratings import confidence_tier, log5
import json
from utils.ncaab_modifiers import blend_with_seed_prior, blend_with_vegas, tempo_factor as compute_tempo_factor, seed_matchup_prior

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROUND_NAMES = {
    0: "First Four",
    1: "First Round",
    2: "Second Round",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Championship",
}


def load_ratings(season: int) -> dict[str, dict]:
    """Load team ratings keyed by team_name."""
    df = read_dataframe(
        "SELECT * FROM ncaab_team_ratings WHERE season = ?", (season,)
    )
    ratings = {}
    for _, row in df.iterrows():
        ratings[row["team_name"]] = row.to_dict()
    return ratings


def load_bracket(season: int) -> list[dict]:
    """Load bracket games sorted by round, slot."""
    df = read_dataframe(
        "SELECT * FROM ncaab_bracket WHERE season = ? ORDER BY round, slot",
        (season,),
    )
    return [row.to_dict() for _, row in df.iterrows()]


def _load_vegas_lines(filepath: str) -> dict[str, dict]:
    """Load Vegas lines CSV. Returns empty dict if file missing/empty."""
    import csv
    path = Path(filepath)
    if not path.exists():
        return {}
    lines = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            game_id = row.get("game_id", "").strip()
            if not game_id:
                continue
            ml_a = row.get("moneyline_a")
            ml_b = row.get("moneyline_b")
            implied_p_a = None
            if ml_a and ml_b:
                from utils.ncaab_modifiers import vegas_implied_prob
                p_a = vegas_implied_prob(int(ml_a))
                p_b = vegas_implied_prob(int(ml_b))
                total = p_a + p_b
                implied_p_a = p_a / total if total > 0 else None
            lines[game_id] = {"implied_p_a": implied_p_a}
    return lines


def _team_modifier_summary(team_name: str, ratings: dict[str, dict]) -> dict:
    """Build modifier breakdown dict for a team (for transparency/audit)."""
    r = ratings.get(team_name, {})
    return {
        "bt_factor": r.get("bt_factor"),
        "coaching_factor": r.get("coaching_factor"),
        "experience_factor": r.get("experience_factor"),
        "momentum_factor": r.get("momentum_factor"),
        "composite_rating": r.get("composite_rating"),
        "enhanced_rating": r.get("enhanced_rating"),
    }


def _predict_game(
    game: dict,
    ratings: dict[str, dict],
    winner_by_game: dict[str, dict],
    vegas_lines: dict[str, dict] | None = None,
) -> dict:
    """Predict a single game, resolving teams from prior rounds if needed."""
    team_a = game["team_a"]
    seed_a = game["seed_a"]
    team_b = game["team_b"]
    seed_b = game["seed_b"]

    # Resolve teams from previous games if needed
    if team_a is None and game["prev_game_a"]:
        prev = winner_by_game[game["prev_game_a"]]
        team_a = prev["team"]
        seed_a = prev["seed"]

    if team_b is None and game["prev_game_b"]:
        prev = winner_by_game[game["prev_game_b"]]
        team_b = prev["team"]
        seed_b = prev["seed"]

    seed_a = int(seed_a) if seed_a is not None else 16
    seed_b = int(seed_b) if seed_b is not None else 16

    # Use enhanced_rating if available, fall back to composite_rating
    r_a = ratings.get(team_a, {}).get("enhanced_rating") or ratings.get(team_a, {}).get("composite_rating", 0.30)
    r_b = ratings.get(team_b, {}).get("enhanced_rating") or ratings.get(team_b, {}).get("composite_rating", 0.30)

    p_raw = log5(r_a, r_b)

    # Game-level modifier: Signal 2 — Historical seed matchup prior
    p_after_seed = blend_with_seed_prior(p_raw, seed_a, seed_b)

    # Game-level modifier: Signal 3 — Vegas calibration
    vegas_p = None
    if vegas_lines:
        vl = vegas_lines.get(game["game_id"])
        if vl:
            vegas_p = vl.get("implied_p_a")
    p_after_vegas = blend_with_vegas(p_after_seed, vegas_p)

    # Game-level modifier: Signal 8 — Tempo matchup
    adj_t_a = ratings.get(team_a, {}).get("adj_t") or 67.0
    adj_t_b = ratings.get(team_b, {}).get("adj_t") or 67.0
    is_a_underdog = p_after_vegas < 0.50
    tf = compute_tempo_factor(adj_t_a, adj_t_b, is_a_underdog)
    if is_a_underdog:
        p_final = min(0.99, p_after_vegas * tf)
    else:
        p_final = max(0.01, p_after_vegas / tf) if tf > 1.0 else p_after_vegas

    p_a = p_final

    if p_a >= 0.50:
        winner, loser = team_a, team_b
        winner_seed, loser_seed = seed_a, seed_b
        p_winner = p_a
    else:
        winner, loser = team_b, team_a
        winner_seed, loser_seed = seed_b, seed_a
        p_winner = 1.0 - p_a

    is_upset = 1 if winner_seed > loser_seed else 0
    tier = confidence_tier(p_winner, is_upset=bool(is_upset))
    margin = abs(r_a - r_b)

    return {
        "game_id": game["game_id"],
        "season": game["season"],
        "round": game["round"],
        "region": game["region"],
        "team_a": team_a,
        "seed_a": seed_a,
        "team_b": team_b,
        "seed_b": seed_b,
        "rating_a": r_a,
        "rating_b": r_b,
        "p_a_wins": p_a,
        "predicted_winner": winner,
        "predicted_loser": loser,
        "winner_seed": winner_seed,
        "loser_seed": loser_seed,
        "is_upset": is_upset,
        "confidence_tier": tier,
        "margin": margin,
        # Transparency fields
        "enhanced_rating_a": r_a,
        "enhanced_rating_b": r_b,
        "p_raw_log5": round(p_raw, 6),
        "seed_historical_p": round(seed_matchup_prior(seed_a, seed_b), 6),
        "vegas_implied_p": vegas_p,
        "tempo_factor": round(tf, 6),
        "final_p_a": round(p_final, 6),
        "modifier_json_a": json.dumps(_team_modifier_summary(team_a, ratings)),
        "modifier_json_b": json.dumps(_team_modifier_summary(team_b, ratings)),
    }


def simulate_bracket(
    ratings: dict[str, dict],
    bracket: list[dict],
    vegas_lines: dict[str, dict] | None = None,
) -> list[dict]:
    """Simulate all bracket games round-by-round. Pure, deterministic."""
    games_by_round: dict[int, list[dict]] = {}
    for g in bracket:
        rd = g["round"]
        games_by_round.setdefault(rd, []).append(g)

    winner_by_game: dict[str, dict] = {}
    predictions: list[dict] = []

    for rd in sorted(games_by_round.keys()):
        for game in games_by_round[rd]:
            pred = _predict_game(game, ratings, winner_by_game, vegas_lines)
            predictions.append(pred)
            winner_by_game[game["game_id"]] = {
                "team": pred["predicted_winner"],
                "seed": pred["winner_seed"],
            }

    return predictions


def persist_predictions(predictions: list[dict]) -> None:
    """Write predictions to ncaab_bracket_predictions."""
    now = datetime.utcnow().isoformat()
    rows = [
        (
            p["game_id"], p["season"], p["round"], p["region"],
            p["team_a"], p["seed_a"], p["team_b"], p["seed_b"],
            p["rating_a"], p["rating_b"], p["p_a_wins"],
            p["predicted_winner"], p["predicted_loser"],
            p["winner_seed"], p["loser_seed"],
            p["is_upset"], p["confidence_tier"], p["margin"], now,
            p.get("enhanced_rating_a"), p.get("enhanced_rating_b"),
            p.get("p_raw_log5"), p.get("seed_historical_p"),
            p.get("vegas_implied_p"), p.get("tempo_factor"),
            p.get("final_p_a"),
            p.get("modifier_json_a"), p.get("modifier_json_b"),
        )
        for p in predictions
    ]
    executemany(
        """INSERT OR REPLACE INTO ncaab_bracket_predictions
        (game_id, season, round, region,
         team_a, seed_a, team_b, seed_b,
         rating_a, rating_b, p_a_wins,
         predicted_winner, predicted_loser,
         winner_seed, loser_seed,
         is_upset, confidence_tier, margin, generated_at,
         enhanced_rating_a, enhanced_rating_b,
         p_raw_log5, seed_historical_p,
         vegas_implied_p, tempo_factor,
         final_p_a,
         modifier_json_a, modifier_json_b)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    log.info("Persisted %d predictions", len(rows))


def _trap_display(score: int) -> str:
    """Trapezoid score as filled/empty blocks."""
    if score is None:
        score = 0
    return "\u2588" * score + "\u2591" * (4 - score)


def _team_kenpom_line(name: str, ratings: dict[str, dict]) -> str:
    """Format KenPom context for a team: AdjEM, AdjDE, Trap score."""
    r = ratings.get(name, {})
    em = r.get("adj_em", 0)
    de = r.get("adj_de", 0)
    trap = int(r.get("trapezoid_score", 0) or 0)
    rank = r.get("kenpom_rank", "?")
    return f"KP#{rank} EM:{em:+.1f} DE:{de:.1f} {_trap_display(trap)}"


def render_cli(predictions: list[dict], ratings: dict[str, dict]) -> None:
    """Render bracket predictions to terminal using rich."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.columns import Columns
    except ImportError:
        log.warning("rich not installed, falling back to plain text")
        _render_plain(predictions, ratings)
        return

    console = Console()

    # --- Header: Methodology ---
    console.print()
    console.print(
        Panel(
            "[bold cyan]Base Rating: KenPom Composite[/]\n"
            "  40% AdjEM  |  20% AdjDE  |  15% Pyth Win%\n"
            "  10% SOS    |  15% Anti-Luck (regression adj)\n\n"
            "[bold cyan]Pre-Game Modifiers (on composite rating)[/]\n"
            "  BartTorvik agreement  |  Coaching experience\n"
            "  Roster continuity     |  Momentum / hot streak\n\n"
            "[bold cyan]Game-Level Adjustments (on win probability)[/]\n"
            "  Historical seed matchup prior (30% blend)\n"
            "  Vegas line calibration (when available)\n"
            "  Tempo matchup factor (slow underdogs)",
            title="[bold]MODEL: 8-Signal Smart Modifiers[/]",
            border_style="cyan",
        )
    )

    # --- Champion ---
    champ = [p for p in predictions if p["game_id"] == "CHAMP"]
    if champ:
        c = champ[0]
        w_info = _team_kenpom_line(c["predicted_winner"], ratings)
        console.print(
            Panel(
                f"[bold yellow]{c['predicted_winner']}[/] ({c['winner_seed']} seed)\n"
                f"  {w_info}\n"
                f"Defeated {c['predicted_loser']} ({c['loser_seed']} seed)\n"
                f"Win probability: {max(c['p_a_wins'], 1 - c['p_a_wins']):.1%}",
                title="[bold]PREDICTED CHAMPION[/]",
                border_style="yellow",
            )
        )

    # --- Trapezoid of Excellence Leaders ---
    trap_leaders: dict[int, list[str]] = {}
    for name, r in ratings.items():
        score = r.get("trapezoid_score", 0)
        if score is not None and score >= 1:
            trap_leaders.setdefault(int(score), []).append(name)

    if trap_leaders:
        console.print()
        console.print(
            Panel(
                "\n".join(
                    f"  {_trap_display(score)} ({score}/4): "
                    + ", ".join(sorted(trap_leaders[score]))
                    for score in sorted(trap_leaders.keys(), reverse=True)
                ),
                title="[bold]TRAPEZOID OF EXCELLENCE[/]",
                subtitle="Top-40 AdjOE + Top-40 AdjDE + Top-40 SOS + Top-15 AdjEM",
                border_style="green",
            )
        )

    # --- Round-by-round results, grouped by region within each round ---
    for rd in sorted(set(p["round"] for p in predictions)):
        rd_preds = [p for p in predictions if p["round"] == rd]
        rd_name = ROUND_NAMES.get(rd, f"Round {rd}")

        # Group by region for rounds 1-4
        if rd <= 4:
            regions_in_round: dict[str, list[dict]] = {}
            for p in rd_preds:
                regions_in_round.setdefault(p["region"], []).append(p)

            for region_name in ["East", "West", "South", "Midwest"]:
                region_preds = regions_in_round.get(region_name, [])
                if not region_preds:
                    continue

                table = Table(
                    title=f"\n{rd_name} — {region_name} Region",
                    title_style="bold",
                )
                table.add_column("Matchup", min_width=18)
                table.add_column("Winner", style="bold green", min_width=16)
                table.add_column("KenPom (Winner)", style="cyan", min_width=28)
                table.add_column("Prob", justify="right")
                table.add_column("Tier")

                for p in region_preds:
                    p_win = max(p["p_a_wins"], 1.0 - p["p_a_wins"])
                    tier_color = {
                        "lock": "green", "likely": "yellow",
                        "toss-up": "white", "upset_pick": "red",
                    }.get(p["confidence_tier"], "white")

                    matchup = f"({p['seed_a']}) vs ({p['seed_b']})"
                    upset_tag = " [red bold]UPSET[/]" if p["is_upset"] else ""

                    table.add_row(
                        matchup,
                        f"({p['winner_seed']}) {p['predicted_winner']}{upset_tag}",
                        _team_kenpom_line(p["predicted_winner"], ratings),
                        f"{p_win:.1%}",
                        f"[{tier_color}]{p['confidence_tier']}[/]",
                    )
                console.print(table)
        else:
            # Final Four / First Four / Championship — single table
            table = Table(title=f"\n{rd_name}", title_style="bold")
            table.add_column("Matchup", min_width=22)
            table.add_column("Winner", style="bold green", min_width=16)
            table.add_column("KenPom (Winner)", style="cyan", min_width=28)
            table.add_column("Prob", justify="right")
            table.add_column("Tier")

            for p in rd_preds:
                p_win = max(p["p_a_wins"], 1.0 - p["p_a_wins"])
                tier_color = {
                    "lock": "green", "likely": "yellow",
                    "toss-up": "white", "upset_pick": "red",
                }.get(p["confidence_tier"], "white")

                matchup = (
                    f"({p['seed_a']}) {p['team_a']} vs "
                    f"({p['seed_b']}) {p['team_b']}"
                )
                upset_tag = " [red bold]UPSET[/]" if p["is_upset"] else ""

                table.add_row(
                    matchup,
                    f"({p['winner_seed']}) {p['predicted_winner']}{upset_tag}",
                    _team_kenpom_line(p["predicted_winner"], ratings),
                    f"{p_win:.1%}",
                    f"[{tier_color}]{p['confidence_tier']}[/]",
                )
            console.print(table)

    # --- Upset summary ---
    upsets = [p for p in predictions if p["is_upset"]]
    if upsets:
        console.print()
        console.print(f"[bold red]Upset Picks: {len(upsets)}[/]")
        for u in upsets:
            p_win = max(u["p_a_wins"], 1.0 - u["p_a_wins"])
            console.print(
                f"  ({u['winner_seed']}) {u['predicted_winner']} over "
                f"({u['loser_seed']}) {u['predicted_loser']} "
                f"[dim]({p_win:.1%}, {ROUND_NAMES.get(u['round'], '')})[/]"
            )
    else:
        console.print()
        console.print(
            "[dim]No upsets predicted — model favors KenPom composite in "
            "all matchups. Watch toss-up games for potential bracket busters.[/]"
        )


def _render_plain(predictions: list[dict], ratings: dict[str, dict]) -> None:
    """Fallback plain text rendering."""
    for rd in sorted(set(p["round"] for p in predictions)):
        rd_preds = [p for p in predictions if p["round"] == rd]
        print(f"\n=== {ROUND_NAMES.get(rd, f'Round {rd}')} ===")
        for p in rd_preds:
            p_win = max(p["p_a_wins"], 1.0 - p["p_a_wins"])
            upset = " *UPSET*" if p["is_upset"] else ""
            print(
                f"  ({p['winner_seed']}) {p['predicted_winner']:20s} over "
                f"({p['loser_seed']}) {p['predicted_loser']:20s} "
                f"{p_win:.1%} [{p['confidence_tier']}]{upset}"
            )


def render_html(
    predictions: list[dict], ratings: dict[str, dict], season: int
) -> str:
    """Render bracket to HTML file. Returns output path."""
    from jinja2 import Environment, FileSystemLoader

    # Organize data for template
    regions: dict[str, list[dict]] = {}
    final_four = []
    championship = None

    for p in predictions:
        if p["round"] == 5:
            final_four.append(p)
        elif p["round"] == 6:
            championship = p
        else:
            regions.setdefault(p["region"], []).append(p)

    champion = championship["predicted_winner"] if championship else "TBD"
    champion_seed = championship["winner_seed"] if championship else 0

    # Trapezoid leaders
    trap_leaders: dict[int, list[str]] = {}
    for name, r in ratings.items():
        score = r.get("trapezoid_score", 0)
        if score is not None:
            trap_leaders.setdefault(int(score), []).append(name)

    upsets = [p for p in predictions if p["is_upset"]]

    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("ncaab_bracket.html.j2")

    html = template.render(
        predictions=predictions,
        regions=regions,
        final_four=final_four,
        championship=championship,
        champion=champion,
        champion_seed=champion_seed,
        trapezoid_leaders=trap_leaders,
        upsets=upsets,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        season=season,
        round_names=ROUND_NAMES,
    )

    out_path = f"reports/ncaab_bracket_{season}.html"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html)
    log.info("HTML bracket written to %s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate NCAA tournament bracket")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--cli", action="store_true", help="Render CLI output")
    parser.add_argument("--html", action="store_true", help="Render HTML output")
    args = parser.parse_args()

    ratings = load_ratings(args.season)
    if not ratings:
        log.error("No ratings found for season %d. Run ingest first.", args.season)
        sys.exit(1)

    bracket = load_bracket(args.season)
    if not bracket:
        log.error("No bracket found for season %d. Run define_bracket first.", args.season)
        sys.exit(1)

    log.info("Loaded %d team ratings and %d bracket games", len(ratings), len(bracket))

    vegas_lines = _load_vegas_lines("data/vegas_lines_2026.csv")
    if vegas_lines:
        log.info("Loaded %d Vegas lines", len(vegas_lines))

    predictions = simulate_bracket(ratings, bracket, vegas_lines=vegas_lines)
    persist_predictions(predictions)
    log.info("Simulation complete: %d predictions", len(predictions))

    if args.cli:
        render_cli(predictions, ratings)

    if args.html:
        out_path = render_html(predictions, ratings, args.season)
        # Auto-open on macOS
        import subprocess
        try:
            subprocess.run(["open", out_path], check=False)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
