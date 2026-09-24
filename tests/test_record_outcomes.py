"""Grading and CLV persistence for NFL bets (scripts/record_outcomes.py).

Each test builds a throwaway SQLite database, seeds the tables grading reads
(`materialized_value_view`, `player_stats_enhanced`, `nfl_roster_players`,
`games`, `weekly_odds`) and checks what `grade_bets`, `compute_and_save_clv`
and `save_outcomes` return or persist. `grade_bets` reads the real clock to
decide which games have finished, so kickoffs are set relative to now.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta, timezone

import pytest

from config import config
from schema_migrations import MigrationManager
from scripts.record_outcomes import compute_and_save_clv, grade_bets, save_outcomes
from utils.db import execute, fetchall, fetchone

SEASON = 2025
WEEK = 5
FINAL_GAME = "2025_05_NE_BUF"
PENDING_GAME = "2025_05_KC_LV"

NOW = datetime.now(timezone.utc)
# Two days back clears the four-hour settle margin in utils/game_completion.py.
PAST_KICKOFF = NOW - timedelta(days=2)
FUTURE_KICKOFF = NOW + timedelta(days=2)


@pytest.fixture()
def db(tmp_path, monkeypatch) -> str:
    db_path = str(tmp_path / "record-outcomes.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)
    monkeypatch.setattr(config.database, "backend", "sqlite")
    monkeypatch.setattr(config.database, "path", db_path)
    MigrationManager(db_path).run()
    return db_path


@pytest.fixture()
def final_game(db) -> str:
    _game(FINAL_GAME, PAST_KICKOFF)
    return FINAL_GAME


def _game(game_id: str, kickoff: datetime) -> None:
    _, _, away, home = game_id.split("_")
    execute(
        "INSERT INTO games (game_id, season, week, home_team, away_team, kickoff_utc, game_date) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (game_id, SEASON, WEEK, home, away, kickoff.isoformat(), kickoff.date().isoformat()),
    )


def _bet(
    player_id: str,
    *,
    event_id: str = FINAL_GAME,
    market: str = "receiving_yards",
    line: float = 64.5,
    price: int = -110,
    side: str = "over",
    edge: float = 0.10,
) -> None:
    execute(
        """
        INSERT INTO materialized_value_view (
            season, week, player_id, event_id, market, sportsbook, line, price, side,
            mu, sigma, p_win, edge_percentage, expected_roi, kelly_fraction, stake,
            generated_at
        ) VALUES (?, ?, ?, ?, ?, 'DraftKings', ?, ?, ?, 70.0, 20.0, 0.55, ?, 0.05, 0.02, 20.0, ?)
        """,
        (SEASON, WEEK, player_id, event_id, market, line, price, side, edge, NOW.isoformat()),
    )


def _stat(
    player_id: str,
    team: str,
    *,
    name: str = "Test Player",
    gsis_id: str | None = None,
    **stats: float,
) -> None:
    columns = ["player_id", "gsis_id", "season", "week", "name", "team", "position", *stats]
    values = (player_id, gsis_id, SEASON, WEEK, name, team, "WR", *stats.values())
    execute(
        f"INSERT INTO player_stats_enhanced ({', '.join(columns)}) "
        f"VALUES ({', '.join('?' for _ in columns)})",
        values,
    )


def _roster(gsis_id: str, player_id: str, team: str) -> None:
    execute(
        "INSERT INTO nfl_roster_players "
        "(season, gsis_id, player_id, player_name, team, position, updated_at) "
        "VALUES (?, ?, ?, 'Test Player', ?, 'QB', ?)",
        (SEASON, gsis_id, player_id, team, NOW.isoformat()),
    )


def _odds(
    player_id: str,
    line: float,
    as_of: datetime,
    *,
    price: int = -110,
    under_price: int | None = None,
) -> None:
    execute(
        """
        INSERT INTO weekly_odds (
            event_id, season, week, player_id, market, sportsbook,
            line, price, under_price, as_of
        ) VALUES (?, ?, ?, ?, 'receiving_yards', 'DraftKings', ?, ?, ?, ?)
        """,
        (FINAL_GAME, SEASON, WEEK, player_id, line, price, under_price, as_of.isoformat()),
    )


# ---------------------------------------------------------------------------
# grade_bets
# ---------------------------------------------------------------------------


def test_a_bet_in_a_finished_game_is_graded_against_the_stat_line(final_game):
    _bet("BUF_receiver_a", line=64.5, price=-110)
    _stat("BUF_receiver_a", "BUF", name="Receiver A", receiving_yards=71.0)

    [outcome] = grade_bets(SEASON, WEEK)

    assert outcome["result"] == "win"
    assert outcome["actual_result"] == pytest.approx(71.0)
    assert outcome["profit_units"] == pytest.approx(100 / 110)
    assert outcome["player_name"] == "Receiver A"


def test_bets_in_a_game_that_has_not_finished_are_skipped_not_pushed(final_game):
    _game(PENDING_GAME, FUTURE_KICKOFF)
    _bet("BUF_receiver_a")
    _bet("LV_receiver_b", event_id=PENDING_GAME)
    _stat("BUF_receiver_a", "BUF", receiving_yards=71.0)

    outcomes = grade_bets(SEASON, WEEK)

    assert [o["player_id"] for o in outcomes] == ["BUF_receiver_a"]


def test_a_week_with_no_finished_game_grades_nothing(db):
    _game(PENDING_GAME, FUTURE_KICKOFF)
    _bet("LV_receiver_b", event_id=PENDING_GAME)

    assert grade_bets(SEASON, WEEK) == []


def test_include_unfinished_grades_bets_regardless_of_game_state(db):
    _game(PENDING_GAME, FUTURE_KICKOFF)
    _bet("LV_receiver_b", event_id=PENDING_GAME)

    [outcome] = grade_bets(SEASON, WEEK, include_unfinished=True)

    assert outcome["player_id"] == "LV_receiver_b"
    assert outcome["result"] == "push"


def test_stat_rows_are_matched_to_bets_through_gsis_id(db):
    # Stats mint the id from nflverse's abbreviated name, bets from the roster's
    # full name. Only gsis_id ties the two together.
    _game("2025_05_SF_LAR", PAST_KICKOFF)
    _roster("00-0000001", "LAR_matthew_stafford", "LAR")
    _bet(
        "LAR_matthew_stafford",
        event_id="2025_05_SF_LAR",
        market="passing_yards",
        line=250.5,
    )
    _stat("LAR_m_stafford", "LAR", gsis_id="00-0000001", passing_yards=301.0)

    [outcome] = grade_bets(SEASON, WEEK)

    assert outcome["result"] == "win"
    assert outcome["actual_result"] == pytest.approx(301.0)


def test_a_player_without_a_stat_row_in_a_finished_game_is_a_push(final_game):
    _bet("BUF_receiver_a")
    _bet("BUF_inactive_receiver")
    _stat("BUF_receiver_a", "BUF", receiving_yards=71.0)

    outcomes = {o["player_id"]: o for o in grade_bets(SEASON, WEEK)}
    inactive = outcomes["BUF_inactive_receiver"]

    assert inactive["result"] == "push"
    assert inactive["profit_units"] == 0.0
    assert inactive["actual_result"] is None


def test_an_under_bet_wins_when_the_stat_lands_below_the_line(final_game):
    _bet("BUF_receiver_a", side="under", line=64.5)
    _stat("BUF_receiver_a", "BUF", receiving_yards=50.0)

    [outcome] = grade_bets(SEASON, WEEK)

    assert (outcome["side"], outcome["result"]) == ("under", "win")


@pytest.mark.parametrize(
    ("edge_fraction", "tier"),
    [(0.40, "HIGH"), (0.10, "MEDIUM"), (0.05, "LOW"), (0.01, "MINIMAL")],
)
def test_confidence_tier_reads_the_stored_edge_as_a_fraction(final_game, edge_fraction, tier):
    _bet("BUF_receiver_a", edge=edge_fraction)
    _stat("BUF_receiver_a", "BUF", receiving_yards=71.0)

    [outcome] = grade_bets(SEASON, WEEK)

    assert outcome["confidence_tier"] == tier


def test_a_bet_on_an_unknown_market_is_left_ungraded(final_game):
    _bet("BUF_receiver_a")
    _bet("BUF_receiver_a", market="longest_reception")
    _stat("BUF_receiver_a", "BUF", receiving_yards=71.0)

    assert [o["market"] for o in grade_bets(SEASON, WEEK)] == ["receiving_yards"]


def test_anytime_touchdown_counts_rushing_and_receiving_scores(final_game):
    _bet("BUF_back_a", market="anytime_touchdown", line=0.5, price=150)
    _stat("BUF_back_a", "BUF", rushing_tds=0, receiving_tds=1)

    [outcome] = grade_bets(SEASON, WEEK)

    assert outcome["result"] == "win"
    assert outcome["profit_units"] == pytest.approx(1.5)


def test_regrading_a_week_reuses_the_same_bet_ids(final_game):
    _bet("BUF_receiver_a")
    _bet("BUF_receiver_c", line=40.5)
    _stat("BUF_receiver_a", "BUF", receiving_yards=71.0)

    first = {o["bet_id"] for o in grade_bets(SEASON, WEEK)}
    second = {o["bet_id"] for o in grade_bets(SEASON, WEEK)}

    assert first == second


def test_over_and_under_on_the_same_prop_get_distinct_bet_ids(final_game):
    _bet("BUF_receiver_a", side="over")
    _bet("BUF_receiver_a", side="under")
    _stat("BUF_receiver_a", "BUF", receiving_yards=71.0)

    assert len({o["bet_id"] for o in grade_bets(SEASON, WEEK)}) == 2


# ---------------------------------------------------------------------------
# compute_and_save_clv
# ---------------------------------------------------------------------------


def _graded_receiver(line: float = 64.5) -> list[dict]:
    _bet("BUF_receiver_a", line=line)
    _stat("BUF_receiver_a", "BUF", receiving_yards=71.0)
    return grade_bets(SEASON, WEEK)


def test_clv_is_unknown_when_no_odds_were_scraped(final_game):
    outcomes = _graded_receiver()

    assert compute_and_save_clv(SEASON, WEEK, outcomes) is None
    assert fetchone("SELECT COUNT(*) FROM clv_weekly")[0] == 0


def test_a_prop_scraped_once_records_no_clv(final_game):
    outcomes = _graded_receiver()
    _odds("BUF_receiver_a", 64.5, PAST_KICKOFF - timedelta(hours=2))

    assert compute_and_save_clv(SEASON, WEEK, outcomes) is None
    assert fetchone("SELECT COUNT(*) FROM clv_weekly")[0] == 0


def test_an_unmoved_line_records_zero_clv(final_game):
    [outcome] = _graded_receiver(line=64.5)
    _odds("BUF_receiver_a", 64.5, PAST_KICKOFF - timedelta(days=1))
    _odds("BUF_receiver_a", 64.5, PAST_KICKOFF - timedelta(hours=1))

    average = compute_and_save_clv(SEASON, WEEK, [outcome])

    row = fetchone(
        "SELECT close_line, clv_bp FROM clv_weekly WHERE bet_id = ?", (outcome["bet_id"],)
    )
    assert average == 0.0
    assert tuple(row) == (64.5, 0.0)


def test_the_close_is_the_last_snapshot_before_kickoff(final_game):
    [outcome] = _graded_receiver()
    _odds("BUF_receiver_a", 64.5, PAST_KICKOFF - timedelta(days=1))
    _odds("BUF_receiver_a", 66.5, PAST_KICKOFF - timedelta(hours=1))
    # An in-game quote reflects the score, not the pregame market.
    _odds("BUF_receiver_a", 80.5, PAST_KICKOFF + timedelta(hours=1))

    compute_and_save_clv(SEASON, WEEK, [outcome])

    row = fetchone("SELECT close_line FROM clv_weekly WHERE bet_id = ?", (outcome["bet_id"],))
    assert row[0] == 66.5


def test_the_weekly_clv_average_leaves_out_bets_without_a_close(final_game):
    _bet("BUF_receiver_a", line=64.5)
    _bet("BUF_receiver_b", line=40.5)
    _stat("BUF_receiver_a", "BUF", receiving_yards=71.0)
    outcomes = grade_bets(SEASON, WEEK)
    _odds("BUF_receiver_a", 64.5, PAST_KICKOFF - timedelta(days=1))
    _odds("BUF_receiver_a", 66.5, PAST_KICKOFF - timedelta(hours=1))
    # One scrape is no close; averaging it in as 0 would halve the week's CLV.
    _odds("BUF_receiver_b", 40.5, PAST_KICKOFF - timedelta(hours=1))

    average = compute_and_save_clv(SEASON, WEEK, outcomes)

    [(stored_bp,)] = fetchall("SELECT clv_bp FROM clv_weekly")
    assert stored_bp != 0.0
    assert average == pytest.approx(stored_bp)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "utils/clv.py compute_clv scores line movement with the sign inverted: an over "
        "taken at 64.5 that closes at 67.5 holds the better number, but clv_points and "
        "the model-curve clv_bp both come out negative"
    ),
)
def test_an_over_taken_below_the_closing_line_beat_the_close(final_game):
    outcomes = _graded_receiver(line=64.5)
    _odds("BUF_receiver_a", 64.5, PAST_KICKOFF - timedelta(days=1))
    _odds("BUF_receiver_a", 67.5, PAST_KICKOFF - timedelta(hours=1))

    assert compute_and_save_clv(SEASON, WEEK, outcomes) > 0


@pytest.mark.xfail(
    importlib.util.find_spec("value_betting_engine") is None,
    raises=ModuleNotFoundError,
    strict=True,
    reason=(
        "utils/clv.py _market_fair_prob imports the gitignored value_betting_engine "
        "whenever the close carries both prices, even though a one-sided entry sends "
        "compute_clv down the model path, so CLV crashes in a public clone"
    ),
)
def test_a_two_sided_close_records_clv_for_a_one_sided_entry(final_game):
    outcomes = _graded_receiver(line=64.5)
    _odds("BUF_receiver_a", 64.5, PAST_KICKOFF - timedelta(days=1), under_price=-110)
    _odds("BUF_receiver_a", 64.5, PAST_KICKOFF - timedelta(hours=1), under_price=-110)

    assert compute_and_save_clv(SEASON, WEEK, outcomes) == 0.0


# ---------------------------------------------------------------------------
# save_outcomes
# ---------------------------------------------------------------------------


def _weekly_performance(columns: str) -> tuple:
    row = fetchone(
        f"SELECT {columns} FROM weekly_performance WHERE season = ? AND week = ?",
        (SEASON, WEEK),
    )
    assert row is not None, "save_outcomes wrote no weekly_performance row"
    return tuple(row)


def test_weekly_roi_counts_only_settled_bets_as_units_risked(final_game):
    _bet("BUF_receiver_a", line=64.5)
    _bet("BUF_receiver_b", line=64.5)
    _bet("BUF_receiver_c", line=65.0)
    _stat("BUF_receiver_a", "BUF", receiving_yards=71.0)
    _stat("BUF_receiver_b", "BUF", receiving_yards=50.0)
    _stat("BUF_receiver_c", "BUF", receiving_yards=65.0)

    save_outcomes(grade_bets(SEASON, WEEK))

    wins, losses, pushes, roi = _weekly_performance("wins, losses, pushes, roi_pct")
    assert (wins, losses, pushes) == (1, 1, 1)
    # (+0.909 - 1.0) units over the 2 bets that settled; the push risks nothing.
    assert roi == pytest.approx(-4.5455, abs=1e-4)


def test_a_regrade_without_clv_keeps_the_stored_weekly_clv(final_game):
    execute(
        "INSERT INTO weekly_performance (season, week, clv_avg, updated_at) VALUES (?, ?, ?, ?)",
        (SEASON, WEEK, 12.5, NOW.isoformat()),
    )

    save_outcomes(_graded_receiver())

    total_bets, clv_avg = _weekly_performance("total_bets, clv_avg")
    assert total_bets == 1
    assert clv_avg == pytest.approx(12.5)
