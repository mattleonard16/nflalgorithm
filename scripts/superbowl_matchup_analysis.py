#!/usr/bin/env python3
"""
Super Bowl LX Matchup Analysis: New England Patriots vs Seattle Seahawks
========================================================================
Leverages the NFL Algorithm's defense adjustments, EWMA projections,
and volatility scoring to identify prop betting edges.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from utils.db import read_dataframe
from utils.defense_adjustments import compute_defense_vs_position_multipliers

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 60)

SEASON = 2025
THROUGH_WEEK = 18
EWMA_DECAY = 0.65  # from config


def ewma_projection(weekly_values: list[float], decay: float = EWMA_DECAY) -> float:
    """Compute EWMA projection matching the algorithm's approach."""
    if not weekly_values:
        return 0.0
    weights = [(1 - decay) * (decay ** i) for i in range(len(weekly_values))]
    weights.reverse()
    total_w = sum(weights)
    if total_w == 0:
        return float(np.mean(weekly_values))
    return sum(v * w for v, w in zip(weekly_values, weights)) / total_w


def compute_volatility(weekly_values: list[float]) -> float:
    """Compute volatility score (coefficient of variation)."""
    if len(weekly_values) < 3:
        return 1.0
    mean = np.mean(weekly_values)
    std = np.std(weekly_values, ddof=1)
    if mean == 0:
        return 1.0
    return std / mean


def get_player_weekly(player_id: str) -> pd.DataFrame:
    """Get weekly stats for a player."""
    return read_dataframe(f"""
        SELECT week, rushing_yards, receiving_yards, passing_yards,
               targets, receptions, target_share
        FROM player_stats_enhanced
        WHERE season={SEASON} AND player_id='{player_id}'
        ORDER BY week
    """)


def analyze_player(player_id: str, name: str, position: str,
                   team: str, opponent: str, multipliers: dict) -> dict:
    """Full player analysis with defense-adjusted projections."""
    weekly = get_player_weekly(player_id)
    if weekly.empty:
        return {}

    result = {"name": name, "position": position, "team": team, "games": len(weekly)}

    # Determine primary stat
    if position == "QB":
        stat_col, stat_name = "passing_yards", "passing_yards"
        weekly_vals = weekly["passing_yards"].tolist()
    elif position == "RB":
        stat_col, stat_name = "rushing_yards", "rushing_yards"
        weekly_vals = weekly["rushing_yards"].tolist()
    else:
        stat_col, stat_name = "receiving_yards", "receiving_yards"
        weekly_vals = weekly["receiving_yards"].tolist()

    season_avg = float(np.mean(weekly_vals))
    ewma_proj = ewma_projection(weekly_vals)
    volatility = compute_volatility(weekly_vals)

    # Defense adjustment
    def_key = (opponent, position, stat_name)
    def_mult = multipliers.get(def_key, 1.0)
    adjusted_proj = ewma_proj * def_mult

    # Recent form (last 4)
    recent_vals = weekly_vals[-4:] if len(weekly_vals) >= 4 else weekly_vals
    recent_avg = float(np.mean(recent_vals))

    # Floor/ceiling
    floor = float(np.percentile(weekly_vals, 10)) if len(weekly_vals) >= 5 else min(weekly_vals)
    ceiling = float(np.percentile(weekly_vals, 90)) if len(weekly_vals) >= 5 else max(weekly_vals)

    result.update({
        "primary_stat": stat_name.replace("_", " ").title(),
        "season_avg": round(season_avg, 1),
        "ewma_projection": round(ewma_proj, 1),
        "defense_multiplier": round(def_mult, 3),
        "adjusted_projection": round(adjusted_proj, 1),
        "recent_form_avg": round(recent_avg, 1),
        "volatility": round(volatility, 2),
        "floor_10th": round(floor, 1),
        "ceiling_90th": round(ceiling, 1),
        "trend": "UP" if recent_avg > season_avg * 1.05 else "DOWN" if recent_avg < season_avg * 0.95 else "STEADY",
    })

    # Secondary stat for RBs (receiving) and QBs (rushing)
    if position == "RB":
        rec_vals = weekly["receiving_yards"].tolist()
        result["secondary_stat"] = "Receiving Yards"
        result["secondary_avg"] = round(float(np.mean(rec_vals)), 1)
        result["secondary_ewma"] = round(ewma_projection(rec_vals), 1)
    elif position == "QB":
        rush_vals = weekly["rushing_yards"].tolist()
        result["secondary_stat"] = "Rushing Yards"
        result["secondary_avg"] = round(float(np.mean(rush_vals)), 1)
        result["secondary_ewma"] = round(ewma_projection(rush_vals), 1)

    return result


def main():
    print("=" * 80)
    print("  SUPER BOWL LX MATCHUP ANALYSIS")
    print("  New England Patriots vs Seattle Seahawks")
    print("  Algorithm-Powered Breakdown")
    print("=" * 80)

    # Load defense multipliers
    multipliers = compute_defense_vs_position_multipliers(SEASON, THROUGH_WEEK)

    # ── SECTION 1: TEAM PROFILES ──
    print("\n" + "─" * 80)
    print("  1. TEAM OFFENSIVE PROFILES (2025 Regular Season)")
    print("─" * 80)

    for team, label in [("NE", "NEW ENGLAND PATRIOTS"), ("SEA", "SEATTLE SEAHAWKS")]:
        totals = read_dataframe(f"""
            SELECT
                ROUND(SUM(rushing_yards), 0) as total_rush,
                ROUND(SUM(passing_yards), 0) as total_pass,
                ROUND(SUM(receiving_yards), 0) as total_rec,
                ROUND(SUM(targets), 0) as total_tgt,
                ROUND(SUM(receptions), 0) as total_rec_ct
            FROM player_stats_enhanced
            WHERE season={SEASON} AND team='{team}'
        """)
        r = totals.iloc[0]
        total_yds = r["total_rush"] + r["total_pass"]
        rush_pct = r["total_rush"] / total_yds * 100
        pass_pct = r["total_pass"] / total_yds * 100

        print(f"\n  {label}")
        print(f"    Rush: {r['total_rush']:.0f} yds ({rush_pct:.0f}%) | "
              f"Pass: {r['total_pass']:.0f} yds ({pass_pct:.0f}%)")
        print(f"    Per Game: {r['total_rush']/17:.1f} rush ypg | "
              f"{r['total_pass']/17:.1f} pass ypg")
        print(f"    Targets/Game: {r['total_tgt']/17:.1f} | "
              f"Receptions/Game: {r['total_rec_ct']/17:.1f}")

    # ── SECTION 2: DEFENSE MATCHUP MATRIX ──
    print("\n" + "─" * 80)
    print("  2. DEFENSE MATCHUP MATRIX (Relative Performance Multipliers)")
    print("─" * 80)
    print("    > 1.0 = EXPLOITABLE (players do better than avg vs this defense)")
    print("    < 1.0 = SUPPRESSIVE (players do worse than avg vs this defense)")

    for def_team, off_team in [("SEA", "NE"), ("NE", "SEA")]:
        print(f"\n  {off_team} Offense vs {def_team} Defense:")
        for pos in ["QB", "RB", "WR", "TE"]:
            for stat in ["rushing_yards", "receiving_yards"]:
                key = (def_team, pos, stat)
                if key in multipliers:
                    m = multipliers[key]
                    bar_len = int(abs(m - 1.0) * 50)
                    if m > 1.0:
                        bar = "+" * bar_len
                        label = "EDGE"
                    else:
                        bar = "-" * bar_len
                        label = "TOUGH"
                    print(f"    {pos:<3} {stat:<20} {m:.3f}x  {bar}  [{label}]")

    # ── SECTION 3: PLAYER PROJECTIONS ──
    print("\n" + "─" * 80)
    print("  3. KEY PLAYER PROJECTIONS (EWMA + Defense-Adjusted)")
    print("─" * 80)

    players_ne = [
        ("NE_d_maye", "Drake Maye", "QB"),
        ("NE_s_diggs", "Stefon Diggs", "WR"),
        ("NE_h_henry", "Hunter Henry", "TE"),
        ("NE_t_henderson", "TreVeyon Henderson", "RB"),
        ("NE_r_stevenson", "Rhamondre Stevenson", "RB"),
        ("NE_m_hollins", "Mack Hollins", "WR"),
        ("NE_k_boutte", "Kayshon Boutte", "WR"),
    ]
    players_sea = [
        ("SEA_s_darnold", "Sam Darnold", "QB"),
        ("SEA_j_smith_njigba", "Jaxon Smith-Njigba", "WR"),
        ("SEA_k_walker", "Kenneth Walker III", "RB"),
        ("SEA_z_charbonnet", "Zach Charbonnet", "RB"),
        ("SEA_c_kupp", "Cooper Kupp", "WR"),
        ("SEA_a_barner", "AJ Barner", "TE"),
    ]

    all_analyses = []
    for pid, name, pos in players_ne:
        a = analyze_player(pid, name, pos, "NE", "SEA", multipliers)
        if a:
            all_analyses.append(a)
    for pid, name, pos in players_sea:
        a = analyze_player(pid, name, pos, "SEA", "NE", multipliers)
        if a:
            all_analyses.append(a)

    for team_label, team_code in [("PATRIOTS", "NE"), ("SEAHAWKS", "SEA")]:
        print(f"\n  --- {team_label} ---")
        team_players = [a for a in all_analyses if a["team"] == team_code]
        for p in team_players:
            trend_arrow = {"UP": "^", "DOWN": "v", "STEADY": "="}[p["trend"]]
            vol_label = "LOW" if p["volatility"] < 0.4 else "HIGH" if p["volatility"] > 0.7 else "MED"
            print(f"\n  {p['name']} ({p['position']}) - {p['primary_stat']}")
            print(f"    Season Avg:  {p['season_avg']:>6.1f}  |  EWMA Proj:  {p['ewma_projection']:>6.1f}")
            print(f"    Def Mult:    {p['defense_multiplier']:>6.3f}  |  ADJ Proj:   {p['adjusted_projection']:>6.1f}")
            print(f"    Recent Form: {p['recent_form_avg']:>6.1f}  |  Trend: {p['trend']:>5} {trend_arrow}")
            print(f"    Floor (10%): {p['floor_10th']:>6.1f}  |  Ceiling (90%): {p['ceiling_90th']:>5.1f}")
            print(f"    Volatility:  {p['volatility']:>6.2f}  [{vol_label}]")
            if "secondary_stat" in p:
                print(f"    {p['secondary_stat']}: Avg {p['secondary_avg']:.1f} | EWMA {p['secondary_ewma']:.1f}")

    # ── SECTION 4: ALGORITHM EDGE DETECTION ──
    print("\n" + "─" * 80)
    print("  4. ALGORITHM EDGE DETECTION")
    print("─" * 80)
    print("  Where our model has the biggest advantages over typical market lines:")

    edges = []
    for p in all_analyses:
        # Identify discrepancies between season avg and defense-adjusted EWMA
        diff = p["adjusted_projection"] - p["season_avg"]
        diff_pct = diff / max(p["season_avg"], 1) * 100

        confidence = "HIGH" if p["volatility"] < 0.45 and p["games"] >= 14 else \
                     "MEDIUM" if p["volatility"] < 0.65 else "LOW"

        edges.append({
            "player": p["name"],
            "pos": p["position"],
            "team": p["team"],
            "stat": p["primary_stat"],
            "season_avg": p["season_avg"],
            "adj_proj": p["adjusted_projection"],
            "diff": round(diff, 1),
            "diff_pct": round(diff_pct, 1),
            "volatility": p["volatility"],
            "confidence": confidence,
            "trend": p["trend"],
            "defense_mult": p["defense_multiplier"],
        })

    edges_df = pd.DataFrame(edges)
    edges_df = edges_df.sort_values("diff_pct", key=abs, ascending=False)

    print("\n  Biggest Projection Deviations from Season Average:")
    print("  (Positive = our model projects OVER season avg; Negative = UNDER)")
    print()
    for _, e in edges_df.iterrows():
        direction = "OVER" if e["diff"] > 0 else "UNDER"
        sign = "+" if e["diff"] > 0 else ""
        print(f"  {e['player']:<22} {e['pos']:<3} {e['team']:<4} "
              f"{e['stat']:<18} "
              f"Avg:{e['season_avg']:>6.1f} -> Proj:{e['adj_proj']:>6.1f} "
              f"({sign}{e['diff']:>5.1f} / {sign}{e['diff_pct']:>5.1f}%)  "
              f"Vol:{e['volatility']:.2f} [{e['confidence']}] {e['trend']}")

    # ── SECTION 5: MATCHUP ADVANTAGES ──
    print("\n" + "─" * 80)
    print("  5. KEY MATCHUP ADVANTAGES")
    print("─" * 80)

    # Identify the biggest edges
    ne_advantages = edges_df[(edges_df["team"] == "NE") & (edges_df["defense_mult"] > 1.0)]
    sea_advantages = edges_df[(edges_df["team"] == "SEA") & (edges_df["defense_mult"] > 1.0)]
    ne_tough = edges_df[(edges_df["team"] == "NE") & (edges_df["defense_mult"] < 0.95)]
    sea_tough = edges_df[(edges_df["team"] == "SEA") & (edges_df["defense_mult"] < 0.95)]

    print("\n  PATRIOTS EXPLOITABLE MATCHUPS (vs SEA Defense):")
    if len(ne_advantages) > 0:
        for _, e in ne_advantages.iterrows():
            print(f"    + {e['player']} ({e['pos']}) - {e['stat']}: "
                  f"DEF mult {e['defense_mult']:.3f}x (players do {(e['defense_mult']-1)*100:+.1f}% vs SEA)")
    else:
        print("    No significant exploitable matchups found")

    print("\n  PATRIOTS TOUGH MATCHUPS (vs SEA Defense):")
    if len(ne_tough) > 0:
        for _, e in ne_tough.iterrows():
            print(f"    - {e['player']} ({e['pos']}) - {e['stat']}: "
                  f"DEF mult {e['defense_mult']:.3f}x (players do {(e['defense_mult']-1)*100:+.1f}% vs SEA)")
    else:
        print("    No significant suppressive matchups found")

    print("\n  SEAHAWKS EXPLOITABLE MATCHUPS (vs NE Defense):")
    if len(sea_advantages) > 0:
        for _, e in sea_advantages.iterrows():
            print(f"    + {e['player']} ({e['pos']}) - {e['stat']}: "
                  f"DEF mult {e['defense_mult']:.3f}x (players do {(e['defense_mult']-1)*100:+.1f}% vs NE)")
    else:
        print("    No significant exploitable matchups found")

    print("\n  SEAHAWKS TOUGH MATCHUPS (vs NE Defense):")
    if len(sea_tough) > 0:
        for _, e in sea_tough.iterrows():
            print(f"    - {e['player']} ({e['pos']}) - {e['stat']}: "
                  f"DEF mult {e['defense_mult']:.3f}x (players do {(e['defense_mult']-1)*100:+.1f}% vs NE)")
    else:
        print("    No significant suppressive matchups found")

    # ── SECTION 6: PROP BETTING ANGLES ──
    print("\n" + "─" * 80)
    print("  6. PROP BETTING ANGLES (Algorithm-Driven)")
    print("─" * 80)

    high_conf_edges = edges_df[
        (edges_df["confidence"].isin(["HIGH", "MEDIUM"])) &
        (edges_df["diff_pct"].abs() > 5)
    ].sort_values("diff_pct", key=abs, ascending=False)

    print("\n  High-Confidence Projection Edges (|diff| > 5%, Low-Med Volatility):")
    if len(high_conf_edges) > 0:
        for _, e in high_conf_edges.iterrows():
            direction = "OVER" if e["diff"] > 0 else "UNDER"
            sign = "+" if e["diff"] > 0 else ""
            print(f"    {'>>>' if e['confidence']=='HIGH' else '>>'} {e['player']} {e['stat']}: "
                  f"Lean {direction} ({sign}{e['diff']:.1f} yds from avg) "
                  f"[{e['confidence']} confidence, {e['trend']} trend]")
    else:
        print("    No high-confidence edges detected")

    # ── SECTION 7: VOLATILITY REPORT ──
    print("\n" + "─" * 80)
    print("  7. VOLATILITY REPORT (Projection Uncertainty)")
    print("─" * 80)
    print("  Lower volatility = more consistent = higher confidence in projections")

    for label in ["LOW (<0.40)", "MEDIUM (0.40-0.70)", "HIGH (>0.70)"]:
        if "LOW" in label:
            players = [a for a in all_analyses if a["volatility"] < 0.40]
        elif "MEDIUM" in label:
            players = [a for a in all_analyses if 0.40 <= a["volatility"] <= 0.70]
        else:
            players = [a for a in all_analyses if a["volatility"] > 0.70]

        print(f"\n  {label}:")
        for p in sorted(players, key=lambda x: x["volatility"]):
            print(f"    {p['name']:<22} ({p['position']:<2}) Vol: {p['volatility']:.2f}  "
                  f"Range: {p['floor_10th']:.0f}-{p['ceiling_90th']:.0f} yds")

    print("\n" + "=" * 80)
    print("  Analysis complete. Data source: nflverse via nflreadpy (2025 season)")
    print("  Defense multipliers computed from relative performance methodology")
    print("  Projections use EWMA (decay=0.65) with defense adjustments")
    print("=" * 80)


if __name__ == "__main__":
    main()
