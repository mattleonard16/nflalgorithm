"""Matching guards and snapshot hygiene for the odds-to-projection join.

Pure functions only — no database access — so the logic stays testable in CI
even though the caller (``prop_integration.py``) is gitignored.

Three failure modes of name-based matching are addressed here:

- **Position collisions**: distinct players share a normalized name
  (Lamar Jackson BAL QB vs ATL DB; Justin Jefferson MIN WR vs CLE LB).
  :func:`positions_compatible` blocks cross-position merges and, when a
  position is unknown, falls back to market plausibility.
- **Destructive suffix stripping**: "David Long Jr." (TEN) and "David Long"
  (IND) collapse to the same stripped name. :func:`name_variants` exposes a
  suffix-preserving normalized form so callers can try exact-with-suffix
  matches before falling back to the stripped form, and
  :func:`suffix_conflict` blocks a stripped-name merge when two KNOWN raw
  names disagree only in their suffix. Suffix info must come from a raw-name
  source (e.g. roster names) — player_ids are built from already-stripped
  names and carry none.
- **Book spelling variants**: books write "JaMarr Chase" for the roster's
  "Ja'Marr Chase" and "Hollywood Brown" for "Marquise Brown".
  :func:`loose_name_key` deletes punctuation instead of spacing it and folds
  known first-name aliases (``FIRST_NAME_ALIASES``) to one form;
  :func:`resolve_roster_player` uses it to find the one rostered club when
  the exact player_id lookup misses.
- **Stale/duplicate snapshots**: ``weekly_odds`` accumulates one row per
  scrape. :func:`latest_snapshot_per_key` keeps only the newest row per
  quote key; :func:`filter_stale_snapshots` drops quotes that are ancient
  relative to kickoff or timestamped after it.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Container, Iterable, Mapping, NamedTuple

import pandas as pd

from utils.clv import SNAPSHOT_KEY
from utils.player_id_utils import canonicalize_team, make_player_id, name_from_player_id

NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})

# Positions that can plausibly hold a prop in each market. Used only when one
# side's position is unknown; when both are known, equality is required.
MARKET_POSITIONS: dict[str, frozenset[str]] = {
    "passing_yards": frozenset({"QB"}),
    "receiving_yards": frozenset({"WR", "TE", "RB"}),
    # QB scrambles and the occasional WR/TE carry are legal, if rare.
    "rushing_yards": frozenset({"RB", "QB", "WR", "TE"}),
}


def _clean_position(position: Any) -> str:
    """Uppercased position code, or '' when absent/NaN/blank (unknown)."""
    if position is None:
        return ""
    try:
        if pd.isna(position):
            return ""
    except (TypeError, ValueError):
        pass
    return str(position).strip().upper()


def positions_compatible(proj_position: Any, odds_position: Any, market: Any) -> bool:
    """Whether a projection row and an odds row can be the same player.

    - Both positions known: require exact equality (blocks QB-vs-DB
      Lamar Jackson, WR-vs-LB Justin Jefferson merges).
    - Exactly one known: require the known position to be plausible for the
      market (``MARKET_POSITIONS``); a market not modeled there cannot be
      disproven and passes.
    - Both unknown: returns True — callers should tag such matches as
      position-unverified rather than treat them as vetted.
    """
    proj = _clean_position(proj_position)
    odds = _clean_position(odds_position)
    if proj and odds:
        return proj == odds
    known = proj or odds
    if not known:
        return True
    plausible = MARKET_POSITIONS.get(str(market).strip().lower()) if market else None
    if plausible is None:
        return True
    return known in plausible


class NameVariants(NamedTuple):
    """Normalized name with the generational suffix preserved and stripped."""

    with_suffix: str
    stripped: str


def normalize_name(name: Any) -> str:
    """Normalize a player name for matching, PRESERVING any suffix.

    Same pipeline as the historical stripped normalization (accents removed,
    lowercased, apostrophes to spaces, periods dropped, whitespace collapsed)
    except that Jr/Sr/II/III/IV/V tokens survive, so "David Long Jr." and
    "David Long" stay distinct.
    """
    if not name or not isinstance(name, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in nfkd if not unicodedata.combining(ch)).lower()
    text = text.replace("'", " ").replace(".", "")
    return re.sub(r"\s+", " ", text).strip()


def strip_name_suffix(normalized: str) -> str:
    """Remove generational suffix tokens from an already-normalized name."""
    return " ".join(t for t in normalized.split() if t not in NAME_SUFFIXES)


def name_variants(name: Any) -> NameVariants:
    """Both normalized forms; try ``with_suffix`` matches before ``stripped``."""
    with_suffix = normalize_name(name)
    return NameVariants(with_suffix=with_suffix, stripped=strip_name_suffix(with_suffix))


def suffix_conflict(name_a: Any, name_b: Any) -> bool:
    """Whether two raw names are provably different players by suffix alone.

    True only when both names are known (non-empty after normalization),
    their stripped forms agree, and their suffix-preserving forms differ —
    "David Long Jr." (TEN) vs "David Long" (IND). An empty/unknown side is
    never a conflict: a name whose suffix status is unknown (e.g. derived
    from an already-stripped player_id) is not evidence the player lacks a
    suffix, and blocking on it would break legitimate traded-player matches.
    """
    a = normalize_name(name_a)
    b = normalize_name(name_b)
    if not a or not b or a == b:
        return False
    return strip_name_suffix(a) == strip_name_suffix(b)


# Nickname <-> legal first-name pairs that books and rosters disagree on.
# Each group folds to one form in :func:`loose_name_key`, so the lookup works
# in both directions. Only the FIRST name token is folded, and only these
# exact tokens: a broad nickname table ("Mike"/"Michael") would merge
# distinct players. Add a pair when a real book/roster disagreement is seen.
FIRST_NAME_ALIASES: tuple[frozenset[str], ...] = (
    frozenset({"marquise", "hollywood"}),
    frozenset({"bam", "zonovan"}),
    frozenset({"tank", "nathaniel"}),
    frozenset({"josh", "joshua"}),
    frozenset({"gabe", "gabriel"}),
    frozenset({"cam", "cameron"}),
    frozenset({"chig", "chigoziem"}),
    frozenset({"kenneth", "ken"}),
)
_FIRST_NAME_CANON: dict[str, str] = {
    token: min(group) for group in FIRST_NAME_ALIASES for token in group
}


def loose_name_key(name: Any) -> str:
    """A spelling-insensitive lookup key for one player's name.

    Accents, case, punctuation and generational suffixes are ignored, and
    apostrophes/periods are DELETED rather than turned into spaces, so
    "JaMarr Chase", "Ja'Marr Chase" and the player_id-derived "ja marr chase"
    agree, as do "A.J. Brown", "A. J. Brown" and "aj brown". The first token
    is folded through ``FIRST_NAME_ALIASES`` ("Hollywood Brown" ==
    "Marquise Brown"); tokens are then joined without separators.

    Returns "" for an empty or non-string name. This is a lookup key, never
    an identifier: it deliberately collides more than :func:`normalize_name`,
    so callers must still guard on team and position.
    """
    if not name or not isinstance(name, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in nfkd if not unicodedata.combining(ch)).lower()
    text = re.sub(r"['’`.]", "", text)
    tokens = [t for t in re.split(r"[^a-z0-9]+", text) if t and t not in NAME_SUFFIXES]
    if not tokens:
        return ""
    tokens[0] = _FIRST_NAME_CANON.get(tokens[0], tokens[0])
    return "".join(tokens)


LooseRosterIndex = Mapping[tuple[str, str], frozenset[str]]


def build_loose_roster_index(roster_player_ids: Iterable[str]) -> dict[tuple[str, str], frozenset[str]]:
    """Map ``(team, loose_name_key)`` to the roster player_ids carrying it.

    Roster ids are ``TEAM_name`` (see ``make_player_id``); an id without a
    valid team prefix is skipped. More than one id under one key means the
    key is ambiguous for that club and must not resolve.
    """
    index: dict[tuple[str, str], set[str]] = {}
    for pid in roster_player_ids:
        if not isinstance(pid, str) or "_" not in pid:
            continue
        team = canonicalize_team(pid.split("_", 1)[0])
        key = loose_name_key(name_from_player_id(pid))
        if team and key:
            index.setdefault((team, key), set()).add(pid)
    return {key: frozenset(pids) for key, pids in index.items()}


class RosterResolution(NamedTuple):
    """The club and roster player_id a book name resolved to ("" if none)."""

    team: str
    player_id: str


def resolve_roster_player(
    name: Any,
    clubs: Iterable[str],
    roster_player_ids: Container[str],
    loose_index: LooseRosterIndex | None = None,
) -> RosterResolution:
    """Resolve a book's player name to exactly one rostered club in the game.

    Tries the exact player_id first (``make_player_id`` per club, the rule
    ``utils.player_id_utils.team_for_player`` applies); when that finds no
    club, retries on :func:`loose_name_key` against ``loose_index`` (build it
    once with :func:`build_loose_roster_index`). Each pass must find exactly
    one club holding exactly one player: a name on both rosters (Michael
    Carter ARI RB vs Michael Carter II PHI DB) stays unresolved rather than
    being assigned to either club.

    Returns the roster's own player_id, so a loose hit is stored under the id
    the projections carry ("CIN_ja_marr_chase", not "CIN_jamarr_chase").
    """
    teams = [team for team in dict.fromkeys(canonicalize_team(c) for c in clubs) if team]

    exact = [(team, make_player_id(name, team)) for team in teams]
    exact = [(team, pid) for team, pid in exact if pid in roster_player_ids]
    if len(exact) == 1:
        return RosterResolution(*exact[0])
    if len(exact) > 1:
        return RosterResolution("", "")

    key = loose_name_key(name)
    if not key or not loose_index:
        return RosterResolution("", "")
    loose = [(team, loose_index.get((team, key), frozenset())) for team in teams]
    loose = [(team, pids) for team, pids in loose if pids]
    if len(loose) == 1 and len(loose[0][1]) == 1:
        team, pids = loose[0]
        return RosterResolution(team, next(iter(pids)))
    return RosterResolution("", "")


def _parse_timestamps(series: pd.Series, column: str) -> pd.Series:
    """Parse a timestamp column, failing loud on any unparseable value."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    if parsed.isna().any():
        bad = series[parsed.isna()].unique()[:5]
        raise ValueError(f"{column} values are not parseable timestamps: {list(bad)}")
    return parsed


def latest_snapshot_per_key(odds_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per ``SNAPSHOT_KEY``: the latest parseable ``as_of``.

    ``weekly_odds`` keeps every scrape, so a naive join fans one quote out
    into N rows. This mirrors the closing-line selection in
    :func:`utils.clv.resolve_closing_lines` (idxmax on parsed timestamps —
    lexical max is not chronological max across mixed offsets).

    Raises:
        ValueError: when key/``as_of`` columns are missing or any ``as_of``
            fails to parse. Never silently drops rows.
    """
    if odds_df is None:
        return pd.DataFrame(columns=[*SNAPSHOT_KEY, "as_of"])
    if odds_df.empty:
        return odds_df.copy()

    missing = [c for c in [*SNAPSHOT_KEY, "as_of"] if c not in odds_df.columns]
    if missing:
        raise ValueError(f"odds_df missing required columns: {missing}")

    # Work on a positional index: with duplicate index labels, idxmax would
    # return an ambiguous label and .loc could yield >1 row per key.
    frame = odds_df.reset_index(drop=True)
    parsed = _parse_timestamps(frame["as_of"], "as_of")
    latest_idx = parsed.groupby(
        [frame[c] for c in SNAPSHOT_KEY], dropna=False
    ).idxmax()
    return frame.loc[sorted(latest_idx)].reset_index(drop=True)


class StaleFilterResult(NamedTuple):
    """Filtered odds plus how many rows had no kickoff to judge against."""

    frame: pd.DataFrame
    unjoinable_count: int


def filter_stale_snapshots(
    odds_df: pd.DataFrame,
    kickoffs_df: pd.DataFrame,
    max_age_hours: float = 240.0,
) -> StaleFilterResult:
    """Drop odds snapshots that are stale relative to their game's kickoff.

    A snapshot is dropped when it is older than ``max_age_hours`` before its
    kickoff, or timestamped after kickoff (a post-game scrape is not a live
    line). Rows whose ``event_id`` has no kickoff pass through untouched —
    silently dropping them would hide ingestion gaps — and their count is
    surfaced as ``unjoinable_count`` so callers can alert on it.

    Args:
        odds_df: Rows with at least ``event_id`` and ``as_of``.
        kickoffs_df: Rows with ``event_id`` and ``kickoff_utc``; conflicting
            kickoffs for one event raise.
        max_age_hours: Freshness window before kickoff; must be positive.
            Defaults to ten days: a Monday run sits 7.3 days before that
            week's Monday night game, and a shorter window emptied the
            card on every run before Friday. Fetch-time freshness is
            enforced separately by the odds validator.

    Raises:
        ValueError: on missing columns, unparseable timestamps, conflicting
            kickoffs, or a non-positive ``max_age_hours``.
    """
    if max_age_hours <= 0:
        raise ValueError(f"max_age_hours must be positive, got {max_age_hours}")
    if odds_df is None or odds_df.empty:
        empty = pd.DataFrame(columns=["event_id", "as_of"]) if odds_df is None else odds_df.copy()
        return StaleFilterResult(frame=empty, unjoinable_count=0)

    missing = [c for c in ("event_id", "as_of") if c not in odds_df.columns]
    if missing:
        raise ValueError(f"odds_df missing required columns: {missing}")

    as_of_ts = _parse_timestamps(odds_df["as_of"], "as_of")

    if kickoffs_df is None or kickoffs_df.empty:
        return StaleFilterResult(frame=odds_df.copy(), unjoinable_count=len(odds_df))

    missing = [c for c in ("event_id", "kickoff_utc") if c not in kickoffs_df.columns]
    if missing:
        raise ValueError(f"kickoffs_df missing required columns: {missing}")

    # Parse BEFORE conflict detection: two spellings of the same instant
    # ("...T18:00:00Z" vs "...T18:00:00+00:00") are not a conflict.
    kicks = pd.DataFrame(
        {
            "event_id": kickoffs_df["event_id"].reset_index(drop=True),
            "kickoff_parsed": _parse_timestamps(
                kickoffs_df["kickoff_utc"], "kickoff_utc"
            ).reset_index(drop=True),
        }
    ).drop_duplicates()
    if kicks["event_id"].duplicated().any():
        conflicted = kicks.loc[kicks["event_id"].duplicated(), "event_id"].unique()[:5]
        raise ValueError(f"conflicting kickoff_utc for event_ids: {list(conflicted)}")
    kickoff_by_event = kicks.set_index("event_id")["kickoff_parsed"]

    kickoff_ts = pd.to_datetime(odds_df["event_id"].map(kickoff_by_event), utc=True)
    unjoinable = kickoff_ts.isna()
    window = pd.Timedelta(hours=max_age_hours)
    stale = ~unjoinable & (
        (as_of_ts < kickoff_ts - window) | (as_of_ts > kickoff_ts)
    )
    return StaleFilterResult(
        frame=odds_df.loc[~stale].reset_index(drop=True),
        unjoinable_count=int(unjoinable.sum()),
    )
