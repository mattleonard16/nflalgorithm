"""Prop Integration Module.

Joins weekly projections with odds from multiple sportsbooks using tiered matching:
  Tier 1:   exact player_id + market (confidence 0.95)
  Tier 2:   normalized_name + market + canonicalized team (confidence 0.90)
  Tier 2.5: roster-verified suffix-bearing name + market (confidence 0.87),
            tried before the suffix-stripped tier 3. player_ids are built from
            already-stripped names, so suffix evidence comes ONLY from
            nfl_roster_players raw names; pairs without real suffix evidence
            fall through to tier 3 rather than being relabeled here.
  Tier 3:   normalized_name + market only / fuzzy name (confidence <= 0.85),
            guarded by positions_compatible and suffix_conflict
            (utils/matching.py). Positions come from nfl_roster_players first
            (covers defense and is available pregame), then
            player_stats_enhanced as fallback.

Odds are deduplicated to the latest snapshot per (event_id, player_id, market,
sportsbook) before matching, so superseded scrapes never fan out the join.

Provides PropIntegration class for generating value opportunity reports.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Tuple

import pandas as pd

from utils.db import read_dataframe, get_connection
from utils.matching import (
    latest_snapshot_per_key,
    name_variants,
    positions_compatible,
    strip_name_suffix,
    suffix_conflict,
)
from utils.player_id_utils import (
    canonicalize_team,
    name_from_player_id,
)


# ======================================================================
# Name normalisation
# ======================================================================

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def normalize_player_name(name: str) -> str:
    """Normalize a player name for matching.

    - Strip accents
    - Lowercase
    - Strip suffixes (Jr/Sr/II/III/IV/V)
    - Apostrophes -> space
    - Remove periods
    - Collapse whitespace
    """
    if not name:
        return ""
    text = _strip_accents(name).lower()
    text = text.replace("'", " ")
    text = text.replace(".", "")
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in _SUFFIXES]
    return " ".join(tokens)


# ======================================================================
# 3-tier matching
# ======================================================================


def join_odds_projections(season: int, week: int) -> pd.DataFrame:
    """Join projections with odds using 3-tier + fuzzy matching.

    Output columns include:
        player_id, player_id_odds, market, mu, sigma, line, price,
        event_id, sportsbook, match_tier, match_type, match_confidence,
        match_score, normalized_name, team, team_odds, team_match_flag,
        position, opponent, model_version, featureset_hash, generated_at,
        as_of, season, week, status, practice_participation
    """
    output_cols = [
        "player_id", "player_id_odds", "market", "mu", "sigma", "line",
        "price", "event_id", "sportsbook", "match_tier", "match_type",
        "match_confidence", "match_score", "normalized_name", "team",
        "team_odds", "team_match_flag", "position", "opponent",
        "model_version", "featureset_hash", "generated_at", "as_of",
        "season", "week", "status", "practice_participation",
    ]

    empty = pd.DataFrame(columns=output_cols)

    # ---- load tables ------------------------------------------------
    try:
        proj = read_dataframe(
            "SELECT * FROM weekly_projections WHERE season = ? AND week = ?",
            (season, week),
        )
    except Exception:
        proj = pd.DataFrame()
    if proj.empty:
        return empty

    try:
        odds = read_dataframe(
            "SELECT * FROM weekly_odds WHERE season = ? AND week = ?",
            (season, week),
        )
    except Exception:
        odds = pd.DataFrame()
    if odds.empty:
        return empty
    # One live quote per (event_id, player_id, market, sportsbook): drop
    # superseded scrape snapshots before matching. Fails loud on bad as_of.
    odds = latest_snapshot_per_key(odds)

    try:
        stats = read_dataframe(
            "SELECT DISTINCT player_id, name, position, team "
            "FROM player_stats_enhanced WHERE season = ? AND week = ?",
            (season, week),
        )
    except Exception:
        stats = pd.DataFrame()

    try:
        injuries = read_dataframe(
            "SELECT player_id, status, practice_participation "
            "FROM injury_data WHERE season = ? AND week = ?",
            (season, week),
        )
    except Exception:
        injuries = pd.DataFrame()

    # Roster is the only raw-name source (pids are suffix-stripped) and the
    # only pregame position source that covers defensive players.
    full_name_map, roster_pos = _load_roster_maps(season)

    # ---- prepare enrichment columns ---------------------------------
    # norm_name_full is the roster-verified suffix-preserving form, or ""
    # when the pid has no unambiguous roster name. NEVER derived from the
    # pid: pids are built from already-stripped names, so a pid-derived
    # "with suffix" form would silently equal the stripped form and turn
    # tier 2.5 into a relabeling of tier 3.
    proj = proj.copy()
    proj["norm_name"] = proj["player_id"].apply(
        lambda pid: normalize_player_name(name_from_player_id(pid))
    )
    proj["norm_name_full"] = proj["player_id"].map(full_name_map).fillna("")
    proj["team_canon"] = proj["team"].apply(canonicalize_team)

    odds = odds.copy()
    odds["norm_name"] = odds["player_id"].apply(
        lambda pid: normalize_player_name(name_from_player_id(pid))
    )
    odds["norm_name_full"] = odds["player_id"].map(full_name_map).fillna("")
    odds["odds_team_raw"] = odds["player_id"].apply(_team_from_pid)
    odds["odds_team_canon"] = odds["odds_team_raw"].apply(canonicalize_team)

    # Build a stats lookup: player_id -> (position, team, name)
    stats_map: dict = {}
    if not stats.empty:
        for _, sr in stats.iterrows():
            stats_map[sr["player_id"]] = {
                "position": sr.get("position", ""),
                "stats_team": sr.get("team", ""),
                "stats_name": sr.get("name", ""),
            }

    injury_map: dict = {}
    if not injuries.empty:
        for _, ir in injuries.iterrows():
            injury_map[ir["player_id"]] = {
                "status": ir.get("status", ""),
                "practice_participation": ir.get("practice_participation", ""),
            }

    # ---- Tier 1: exact player_id + market ---------------------------
    matched_rows: list[dict] = []
    proj_remaining = proj.copy()
    odds_remaining = odds.copy()

    tier1 = proj_remaining.merge(
        odds_remaining,
        on=["player_id", "market"],
        how="inner",
        suffixes=("_proj", "_odds"),
    )
    for _, r in tier1.iterrows():
        matched_rows.append(
            _build_row(r, tier=1, match_type="player_id", confidence=0.95,
                       stats_map=stats_map, injury_map=injury_map,
                       season=season, week=week)
        )

    matched_odds_pids = set()
    if not tier1.empty:
        matched_odds_pids = set(
            zip(tier1["player_id"], tier1["market"])
        )

    proj_remaining = proj_remaining[
        ~proj_remaining.apply(lambda r: (r["player_id"], r["market"]) in matched_odds_pids, axis=1)
    ]
    odds_remaining = odds_remaining[
        ~odds_remaining.apply(lambda r: (r["player_id"], r["market"]) in matched_odds_pids, axis=1)
    ]

    # ---- Tier 2: normalized_name + market + team_canon --------------
    if not proj_remaining.empty and not odds_remaining.empty:
        tier2 = proj_remaining.merge(
            odds_remaining,
            left_on=["norm_name", "market", "team_canon"],
            right_on=["norm_name", "market", "odds_team_canon"],
            how="inner",
            suffixes=("_proj", "_odds"),
        )
        t2_keys = set()
        for _, r in tier2.iterrows():
            matched_rows.append(
                _build_row(r, tier=2, match_type="normalized_name_team",
                           confidence=0.90, stats_map=stats_map,
                           injury_map=injury_map, season=season, week=week)
            )
            t2_keys.add((r.get("player_id_proj", r.get("player_id", "")), r["market"]))

        proj_remaining = proj_remaining[
            ~proj_remaining.apply(lambda r: (r["player_id"], r["market"]) in t2_keys, axis=1)
        ]
        # Remove odds rows already matched in tier 2
        t2_odds_keys = set()
        if not tier2.empty:
            for _, r in tier2.iterrows():
                opid = r.get("player_id_odds", r.get("player_id", ""))
                t2_odds_keys.add((opid, r["market"]))
        odds_remaining = odds_remaining[
            ~odds_remaining.apply(lambda r: (r["player_id"], r["market"]) in t2_odds_keys, axis=1)
        ]

    # ---- Tier 2.5: roster-verified suffix-bearing name matches ------
    # Runs before the suffix-stripped tier 3; only pairs whose roster raw
    # names BOTH carry the same suffix land here.
    if not proj_remaining.empty and not odds_remaining.empty:
        t25_rows, proj_remaining, odds_remaining = _suffix_tier_matches(
            proj_remaining, odds_remaining, stats_map=stats_map,
            roster_pos=roster_pos, injury_map=injury_map,
            season=season, week=week,
        )
        matched_rows.extend(t25_rows)

    # ---- Tier 3: normalized_name + market (no team) -----------------
    if not proj_remaining.empty and not odds_remaining.empty:
        tier3 = proj_remaining.merge(
            odds_remaining,
            on=["norm_name", "market"],
            how="inner",
            suffixes=("_proj", "_odds"),
        )
        t3_keys = set()
        t3_odds_keys = set()
        for _, r in tier3.iterrows():
            p_pos, o_pos = _row_positions(r, stats_map, roster_pos)
            if not positions_compatible(p_pos, o_pos, r["market"]):
                # Name collision across positions (e.g. Lamar Jackson QB vs
                # DB): not the same player, do not consume either row.
                continue
            if suffix_conflict(r.get("norm_name_full_proj", ""),
                               r.get("norm_name_full_odds", "")):
                # Both roster names known and they differ only by suffix
                # (David Long Jr. TEN vs David Long IND): different players.
                continue
            mtype = ("normalized_name" if (p_pos or o_pos)
                     else "normalized_name_unverified_position")
            matched_rows.append(
                _build_row(r, tier=3, match_type=mtype,
                           confidence=0.85, stats_map=stats_map,
                           injury_map=injury_map, season=season, week=week)
            )
            t3_keys.add((r.get("player_id_proj", r.get("player_id", "")), r["market"]))
            t3_odds_keys.add((r.get("player_id_odds", r.get("player_id", "")), r["market"]))

        proj_remaining = proj_remaining[
            ~proj_remaining.apply(lambda r: (r["player_id"], r["market"]) in t3_keys, axis=1)
        ]
        odds_remaining = odds_remaining[
            ~odds_remaining.apply(lambda r: (r["player_id"], r["market"]) in t3_odds_keys, axis=1)
        ]

    # ---- Tier 3b: fuzzy name + market (for typos) -------------------
    if not proj_remaining.empty and not odds_remaining.empty:
        for _, pr in proj_remaining.iterrows():
            p_norm = pr["norm_name"]
            p_market = pr["market"]
            p_pos = _pid_position(pr["player_id"], stats_map, roster_pos)
            best_score = 0.0
            best_odds_row = None
            for _, orow in odds_remaining.iterrows():
                if orow["market"] != p_market:
                    continue
                o_pos = _pid_position(orow["player_id"], stats_map, roster_pos)
                if not positions_compatible(p_pos, o_pos, p_market):
                    continue
                if suffix_conflict(pr["norm_name_full"], orow["norm_name_full"]):
                    # Pairs blocked at tier 3 reappear here with ratio 1.0;
                    # the same suffix evidence must block them again.
                    continue
                ratio = SequenceMatcher(None, p_norm, orow["norm_name"]).ratio()
                if ratio > best_score and ratio >= 0.80:
                    best_score = ratio
                    best_odds_row = orow
            if best_odds_row is not None:
                # Build a merged-style dict manually
                merged = {}
                for c in pr.index:
                    merged[f"{c}_proj"] = pr[c]
                for c in best_odds_row.index:
                    merged[f"{c}_odds"] = best_odds_row[c]
                merged["norm_name"] = p_norm
                merged["market"] = p_market
                conf = min(0.85, best_score)
                o_pos = _pid_position(best_odds_row["player_id"], stats_map, roster_pos)
                mtype = ("fuzzy_name" if (p_pos or o_pos)
                         else "fuzzy_name_unverified_position")
                matched_rows.append(
                    _build_row(merged, tier=3, match_type=mtype,
                               confidence=conf, stats_map=stats_map,
                               injury_map=injury_map, season=season,
                               week=week, is_dict=True)
                )

    if not matched_rows:
        return empty

    result = pd.DataFrame(matched_rows)
    # Ensure types
    result["match_tier"] = result["match_tier"].astype(int)
    result["match_confidence"] = result["match_confidence"].astype(float)
    result["match_score"] = result["match_score"].astype(float)

    # Select only expected columns (drop extras, fill missing)
    for c in output_cols:
        if c not in result.columns:
            result[c] = None
    return result[output_cols].reset_index(drop=True)


# ======================================================================
# Matching-guard helpers (roster maps, positions, suffix tier)
# ======================================================================


def _load_roster_maps(season: int) -> Tuple[dict, dict]:
    """Roster-derived per-pid maps: suffix-preserving names and positions.

    Returns ``(full_name_map, roster_pos)``:
    - ``full_name_map``: pid -> normalized raw roster name WITH suffix
      preserved ("david long jr"), only for pids with exactly one distinct
      roster name that season; ambiguous pids are omitted (= unknown).
    - ``roster_pos``: pid -> position (QB/RB/WR/TE/DB/LB/...), only for pids
      with exactly one distinct position; covers defensive players and is
      available pregame, unlike player_stats_enhanced.
    """
    try:
        roster = read_dataframe(
            "SELECT DISTINCT player_id, player_name, position "
            "FROM nfl_roster_players WHERE season = ?",
            (season,),
        )
    except Exception:
        roster = pd.DataFrame()
    full_name_map: dict = {}
    roster_pos: dict = {}
    if roster.empty:
        return full_name_map, roster_pos
    for pid, grp in roster.groupby("player_id"):
        names = {
            name_variants(n).with_suffix
            for n in grp["player_name"]
            if isinstance(n, str) and n.strip()
        }
        if len(names) == 1:
            full_name_map[pid] = names.pop()
        positions = {
            str(p).strip().upper()
            for p in grp["position"]
            if isinstance(p, str) and p.strip()
        }
        if len(positions) == 1:
            roster_pos[pid] = positions.pop()
    return full_name_map, roster_pos


def _pid_position(pid, stats_map: dict, roster_pos: dict) -> str:
    """Best-known position for a pid: roster first, then weekly stats."""
    return roster_pos.get(pid) or stats_map.get(pid, {}).get("position", "")


def _row_positions(r, stats_map: dict, roster_pos: dict) -> Tuple[str, str]:
    """Positions for the projection and odds sides of a merged row."""
    ppid = r.get("player_id_proj", r.get("player_id", ""))
    opid = r.get("player_id_odds", r.get("player_id", ""))
    return (
        _pid_position(ppid, stats_map, roster_pos),
        _pid_position(opid, stats_map, roster_pos),
    )


def _drop_consumed(proj_remaining, odds_remaining, accepted_rows):
    """Remove proj/odds rows consumed by accepted merge results."""
    if not accepted_rows:
        return proj_remaining, odds_remaining
    proj_keys = set()
    odds_keys = set()
    for r in accepted_rows:
        proj_keys.add((r.get("player_id_proj", r.get("player_id", "")), r["market"]))
        odds_keys.add((r.get("player_id_odds", r.get("player_id", "")), r["market"]))
    proj_remaining = proj_remaining[
        ~proj_remaining.apply(lambda row: (row["player_id"], row["market"]) in proj_keys, axis=1)
    ]
    odds_remaining = odds_remaining[
        ~odds_remaining.apply(lambda row: (row["player_id"], row["market"]) in odds_keys, axis=1)
    ]
    return proj_remaining, odds_remaining


def _suffix_bearing(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows whose roster-verified name actually carries a suffix."""
    mask = frame["norm_name_full"].map(
        lambda s: bool(s) and strip_name_suffix(s) != s
    )
    return frame[mask]


def _suffix_tier_matches(
    proj_remaining, odds_remaining, *, stats_map, roster_pos, injury_map,
    season, week
):
    """Tier 2.5: roster-verified suffix-bearing name + market (0.87).

    Only pairs where BOTH sides' roster raw names carry the same generational
    suffix land here — that is real verification the pid-derived stripped name
    cannot provide. Everything else (suffix-less or roster-unknown names)
    falls through to tier 3 at its lower confidence instead of being
    relabeled as suffix-verified. Team-less, so same-name different-team
    collisions remain possible — guarded by positions_compatible like tier 3.

    (A with-suffix + team tier would be a strict subset of tier 2's
    stripped-name + team merge, which already ran: intentionally absent.)
    """
    rows: list[dict] = []
    merged = _suffix_bearing(proj_remaining).merge(
        _suffix_bearing(odds_remaining),
        on=["norm_name_full", "market"],
        how="inner",
        suffixes=("_proj", "_odds"),
    )
    accepted = []
    for _, r in merged.iterrows():
        p_pos, o_pos = _row_positions(r, stats_map, roster_pos)
        if not positions_compatible(p_pos, o_pos, r["market"]):
            continue
        mtype = ("normalized_name_suffix" if (p_pos or o_pos)
                 else "normalized_name_suffix_unverified_position")
        rows.append(
            _build_row(r, tier=3, match_type=mtype, confidence=0.87,
                       stats_map=stats_map, injury_map=injury_map,
                       season=season, week=week)
        )
        accepted.append(r)
    proj_remaining, odds_remaining = _drop_consumed(proj_remaining, odds_remaining, accepted)

    return rows, proj_remaining, odds_remaining


# ======================================================================
# Row builder
# ======================================================================

def _build_row(
    r, *, tier: int, match_type: str, confidence: float,
    stats_map: dict, injury_map: dict, season: int, week: int,
    is_dict: bool = False,
) -> dict:
    """Build an output row from a merged pandas row or dict."""
    def _g(key: str, default=""):
        """Get value trying suffixed then plain keys."""
        if is_dict:
            for k in (f"{key}_proj", key, f"{key}_odds"):
                if k in r and r[k] is not None and str(r[k]) != "":
                    return r[k]
            return default
        for k in (f"{key}_proj", key, f"{key}_odds"):
            try:
                v = r[k]
                if v is not None and str(v) != "":
                    return v
            except (KeyError, AttributeError):
                continue
        return default

    pid = _g("player_id")
    pid_odds = _g("player_id_odds", _g("player_id", ""))
    # For tier 1 merges, player_id_odds won't exist as a separate column
    if tier == 1:
        pid_odds = pid

    # If merged produced player_id_proj / player_id_odds
    if not is_dict:
        try:
            pid = r["player_id_proj"] if "player_id_proj" in r.index else r["player_id"]
        except (KeyError, AttributeError):
            pass
        try:
            pid_odds = r["player_id_odds"] if "player_id_odds" in r.index else pid
        except (KeyError, AttributeError):
            pid_odds = pid

    if is_dict:
        if "player_id_proj" in r:
            pid = r["player_id_proj"]
        if "player_id_odds" in r:
            pid_odds = r["player_id_odds"]

    # Team from stats takes priority
    si = stats_map.get(pid, {})
    team = si.get("stats_team") or _g("team")
    position = si.get("position") or _g("position", "")

    odds_team_raw = _g("odds_team_raw", _g("odds_team_canon", ""))
    if is_dict:
        odds_team_raw = r.get("odds_team_raw_odds", r.get("odds_team_canon_odds", ""))
    team_odds = canonicalize_team(str(odds_team_raw)) if odds_team_raw else ""

    team_canon = canonicalize_team(str(team)) if team else ""
    team_match = team_canon == team_odds if (team_canon and team_odds) else False

    inj = injury_map.get(pid, {})

    return {
        "player_id": pid,
        "player_id_odds": pid_odds,
        "market": _g("market"),
        "mu": _g("mu", 0),
        "sigma": _g("sigma", 0),
        "line": _g("line", 0),
        "price": _g("price", -110),
        "event_id": _g("event_id", ""),
        "sportsbook": _g("sportsbook", ""),
        "match_tier": tier,
        "match_type": match_type,
        "match_confidence": confidence,
        "match_score": confidence,
        "normalized_name": _g("norm_name", ""),
        "team": team,
        "team_odds": team_odds,
        "team_match_flag": team_match,
        "position": position,
        "opponent": _g("opponent", ""),
        "model_version": _g("model_version", ""),
        "featureset_hash": _g("featureset_hash", ""),
        "generated_at": _g("generated_at", ""),
        "as_of": _g("as_of", ""),
        "season": season,
        "week": week,
        "status": inj.get("status", ""),
        "practice_participation": inj.get("practice_participation", ""),
    }


def _team_from_pid(pid: str) -> str:
    if not pid or "_" not in pid:
        return ""
    return pid.split("_", 1)[0].upper()


# ======================================================================
# PropIntegration class
# ======================================================================


class PropIntegration:
    """Integrates projections with odds to identify value opportunities.

    Stateless: callers MUST pass season and week to every method. No defaults
    (T0 #5 fix — previously pinned 2025/13).
    """

    def get_best_value_opportunities(
        self,
        season: int,
        week: int,
        min_edge_threshold: float = 8.0,
    ) -> pd.DataFrame:
        joined = join_odds_projections(season, week)
        if joined.empty:
            return pd.DataFrame(columns=[
                "player", "team", "position", "book", "stat", "line",
                "model_prediction", "edge_yards", "edge_percentage",
                "over_odds", "under_odds", "value_rating",
            ])

        joined["edge_yards"] = joined["mu"].astype(float) - joined["line"].astype(float)
        joined["edge_percentage"] = (joined["edge_yards"] / joined["line"].astype(float)) * 100
        joined["over_odds"] = joined.apply(
            lambda r: r["price"] if float(r["edge_yards"]) > 0 else None, axis=1
        )
        joined["under_odds"] = joined.apply(
            lambda r: r["price"] if float(r["edge_yards"]) < 0 else None, axis=1
        )
        joined["value_rating"] = joined["edge_percentage"].abs() * joined["match_confidence"]
        opps = joined[joined["edge_percentage"].abs() >= min_edge_threshold].copy()
        opps["player"] = opps["player_id"].apply(name_from_player_id)
        out = opps.rename(columns={"sportsbook": "book", "market": "stat", "mu": "model_prediction"})
        cols = [
            "player", "team", "position", "book", "stat", "line",
            "model_prediction", "edge_yards", "edge_percentage",
            "over_odds", "under_odds", "value_rating",
        ]
        return out[[c for c in cols if c in out.columns]].sort_values(
            "value_rating", ascending=False
        ).reset_index(drop=True)

    def generate_value_report(
        self,
        season: int,
        week: int,
        min_edge_threshold: float = 8.0,
    ) -> str:
        opps = self.get_best_value_opportunities(
            season=season, week=week, min_edge_threshold=min_edge_threshold
        )
        report = f"# NFL Prop Value Report (season={season}, week={week})\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"**Opportunities:** {len(opps)}\n\n---\n\n"
        if opps.empty:
            report += "No opportunities found.\n"
        else:
            for _, row in opps.iterrows():
                report += f"- {row.get('player','?')} ({row.get('team','?')}): "
                report += f"{row.get('stat','?')} line={row.get('line',0)} "
                report += f"edge={row.get('edge_percentage',0):.1f}%\n"
        return report

    def render_html_from_markdown(self, md_text: str, output_path) -> str:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        html = f"<html><body><pre>{md_text}</pre></body></html>"
        p.write_text(html, encoding="utf-8")
        return str(p)

    def export_opportunities_csv(self, opps: pd.DataFrame, path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        opps.to_csv(p, index=False)
        return str(p)

    def export_opportunities_json(self, opps: pd.DataFrame, path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(opps.to_dict(orient="records"), f, indent=2, default=str)
        return str(p)

    def update_real_time_value_finder(
        self,
        season: int,
        week: int,
        min_edge_threshold: float = 5.0,
    ) -> pd.DataFrame:
        """Update dashboard tables with current opportunities.

        Returns the opportunities DataFrame so callers don't recompute.
        """
        opps = self.get_best_value_opportunities(
            season=season, week=week, min_edge_threshold=min_edge_threshold
        )
        if opps.empty:
            return opps
        with get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS value_opportunities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    season INTEGER, week INTEGER, player TEXT, team TEXT,
                    position TEXT, book TEXT, stat TEXT, line REAL,
                    model_prediction REAL, edge_yards REAL,
                    edge_percentage REAL, value_rating REAL,
                    updated_at TIMESTAMP
                )
            """)
            conn.commit()
        return opps
