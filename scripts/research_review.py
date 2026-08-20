"""Memo sections that look back at the week that just completed.

Section 1 asks what the bets returned; section 2 asks how close the projections
landed. Both are keyed on ``(season, week)`` exactly as given -- nothing here
looks forward.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from scripts.research_format import _fmt, _markdown_table, _no_data, _read_optional, _text
from utils.context_factors import _bridge_history_ids
from utils.db import DBConnection, get_table_columns, read_dataframe
from utils.nfl_markets import MARKET_TO_STAT, melt_actuals

# The per-position MAE ceilings the CI gate enforces. Imported rather than
# restated so the memo and `make mae-gate` can never quote different numbers.
from scripts.evaluate_nfl_projections import MIN_POSITION_SAMPLE, POSITION_MAE_THRESHOLDS


def grading_recap(
    season: int, week: int, *, conn: Optional[DBConnection] = None
) -> Tuple[str, Dict[str, Any]]:
    """What the bets placed for the just-completed week actually returned."""
    title = "## 1. GRADING RECAP"
    section = "grading_recap"

    outcomes = _read_optional(
        "SELECT bet_id, player_name, market, sportsbook, side, line, price, "
        "actual_result, result, profit_units, confidence_tier, edge_at_placement "
        "FROM bet_outcomes WHERE season = ? AND week = ?",
        (season, week),
        table="bet_outcomes",
        conn=conn,
    )
    performance = _read_optional(
        "SELECT total_bets, wins, losses, pushes, profit_units, roi_pct, avg_edge, "
        "clv_avg, best_bet, worst_bet, updated_at FROM weekly_performance "
        "WHERE season = ? AND week = ?",
        (season, week),
        table="weekly_performance",
        conn=conn,
    )

    if outcomes.empty and performance.empty:
        return _no_data(
            title,
            f"no graded bets for {season} W{week:02d} "
            "(bet_outcomes and weekly_performance are both empty for this key)",
            section,
        )

    payload: Dict[str, Any] = {"section": section, "status": "ok", "notes": []}
    body: List[str] = [title, ""]

    if performance.empty:
        payload["notes"].append(
            "weekly_performance has no row for this key; the record below is "
            "derived from bet_outcomes"
        )
        body.append(
            "_weekly_performance has no row for this week; record derived from bet_outcomes._"
        )
        body.append("")
        summary: Dict[str, Any] = {}
    else:
        row = performance.iloc[0]
        summary = {
            "total_bets": int(row["total_bets"]),
            "wins": int(row["wins"]),
            "losses": int(row["losses"]),
            "pushes": int(row["pushes"]),
            "profit_units": float(row["profit_units"]),
            "roi_pct": float(row["roi_pct"]),
            "avg_edge": float(row["avg_edge"]),
            "clv_avg": float(row["clv_avg"]),
            "best_bet": _text(row["best_bet"]) if _text(row["best_bet"]) != "-" else None,
            "worst_bet": _text(row["worst_bet"]) if _text(row["worst_bet"]) != "-" else None,
        }
        body.append(
            _markdown_table(
                ["bets", "W-L-P", "units", "ROI %", "avg edge", "CLV avg"],
                [
                    [
                        str(summary["total_bets"]),
                        f"{summary['wins']}-{summary['losses']}-{summary['pushes']}",
                        _fmt(summary["profit_units"], 2),
                        _fmt(summary["roi_pct"], 2),
                        _fmt(summary["avg_edge"], 3),
                        _fmt(summary["clv_avg"], 2),
                    ]
                ],
            )
        )
        body.append("")
    payload["summary"] = summary

    result_counts: Dict[str, int] = {}
    if not outcomes.empty:
        result_counts = {
            _text(name): int(count)
            for name, count in outcomes["result"].fillna("ungraded").value_counts().items()
        }
        payload["bet_count"] = int(len(outcomes))
        payload["result_counts"] = result_counts
        body.append(
            "Recorded bets: "
            + ", ".join(f"{name} {count}" for name, count in sorted(result_counts.items()))
            + f" (total {len(outcomes)})."
        )
        body.append("")

        graded = summary.get("wins", 0) + summary.get("losses", 0) + summary.get("pushes", 0)
        if summary and graded != len(outcomes):
            note = (
                f"weekly_performance counts {graded} settled bets but bet_outcomes holds "
                f"{len(outcomes)} rows; the aggregate looks stale relative to the per-bet table"
            )
            payload["notes"].append(note)
            body.append(f"_{note}._")
            body.append("")

    if not outcomes.empty and outcomes["profit_units"].notna().any():
        ranked = outcomes.copy()
        ranked["profit_units"] = pd.to_numeric(ranked["profit_units"], errors="coerce")
        ranked = ranked.dropna(subset=["profit_units"]).sort_values(
            ["profit_units", "bet_id"], ascending=[False, True], kind="stable"
        )
        best = ranked.head(3)
        worst = ranked.tail(3).iloc[::-1]
        payload["best_bets"] = _bet_records(best)
        payload["worst_bets"] = _bet_records(worst)
        body.append("**Best bets**")
        body.append("")
        body.append(_bet_table(best))
        body.append("")
        body.append("**Worst bets**")
        body.append("")
        body.append(_bet_table(worst))
        body.append("")
    elif not outcomes.empty:
        note = "every bet_outcomes row for this week has a null profit_units (not settled yet)"
        payload["notes"].append(note)
        body.append(f"_{note}._")
        body.append("")

    clv_body, clv_payload = _clv_block(outcomes, conn=conn)
    payload["clv"] = clv_payload
    body.append(clv_body)

    return "\n".join(body).rstrip(), payload


def _bet_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {
            "bet_id": _text(row.bet_id),
            "player_name": _text(row.player_name),
            "market": _text(row.market),
            "side": _text(row.side),
            "line": None if pd.isna(row.line) else float(row.line),
            "price": None if pd.isna(row.price) else int(row.price),
            "actual_result": None if pd.isna(row.actual_result) else float(row.actual_result),
            "result": _text(row.result),
            "profit_units": float(row.profit_units),
        }
        for row in frame.itertuples(index=False)
    ]


def _bet_table(frame: pd.DataFrame) -> str:
    return _markdown_table(
        ["player", "market", "side", "line", "price", "actual", "result", "units"],
        [
            [
                _text(row.player_name),
                _text(row.market),
                _text(row.side),
                _fmt(row.line),
                _text(row.price),
                _fmt(row.actual_result),
                _text(row.result),
                _fmt(row.profit_units, 2),
            ]
            for row in frame.itertuples(index=False)
        ],
    )


def _clv_block(
    outcomes: pd.DataFrame, *, conn: Optional[DBConnection] = None
) -> Tuple[str, Dict[str, Any]]:
    """Closing-line value for this week's bets, from ``clv_weekly``."""
    if outcomes.empty or "bet_id" not in outcomes.columns:
        return ("_CLV: no bets to look up._", {"status": "no_data", "note": "no bets"})

    bet_ids = [str(value) for value in outcomes["bet_id"].dropna().unique()]
    if not bet_ids:
        return ("_CLV: no bets to look up._", {"status": "no_data", "note": "no bet ids"})

    placeholders = ",".join("?" for _ in bet_ids)
    clv = _read_optional(
        f"SELECT bet_id, close_line, close_price, clv_bp, closed_at FROM clv_weekly "
        f"WHERE bet_id IN ({placeholders})",
        tuple(bet_ids),
        table="clv_weekly",
        conn=conn,
    )
    if clv.empty:
        note = f"clv_weekly holds no row for any of this week's {len(bet_ids)} bets"
        return (f"_CLV: no data: {note}._", {"status": "no_data", "note": note})

    values = pd.to_numeric(clv["clv_bp"], errors="coerce").dropna()
    payload = {
        "status": "ok",
        "rows": int(len(clv)),
        "bets_with_clv": int(len(values)),
        "mean_clv_bp": float(values.mean()) if not values.empty else None,
        "median_clv_bp": float(values.median()) if not values.empty else None,
        "beat_close_count": int((values > 0).sum()),
    }
    text = (
        f"**CLV**: {payload['bets_with_clv']} of {len(bet_ids)} bets priced against a close; "
        f"mean {_fmt(payload['mean_clv_bp'], 1)} bp, median "
        f"{_fmt(payload['median_clv_bp'], 1)} bp, "
        f"{payload['beat_close_count']} beat the close."
    )
    return text, payload


# ---------------------------------------------------------------------------
# section 2: projection accuracy
# ---------------------------------------------------------------------------


def projection_accuracy(
    season: int, week: int, *, conn: Optional[DBConnection] = None
) -> Tuple[str, Dict[str, Any]]:
    """Per-position MAE for the week that just completed.

    This is descriptive, not a gate. ``evaluate_projections`` in
    ``scripts.evaluate_nfl_projections`` is deliberately NOT called: it requires
    a completed ``pipeline_runs`` row and a matching git SHA, and drops every
    projection that fails a freshness check, so on a research memo it would
    report "no data" for provenance reasons that have nothing to do with
    accuracy. What is shared with it -- and imported from it -- are the ceilings
    (``POSITION_MAE_THRESHOLDS``) and the minimum sample size, so the memo and
    ``make mae-gate`` cannot drift apart on what counts as too high or too few.
    """
    title = "## 2. PROJECTION ACCURACY"
    section = "projection_accuracy"

    projections = _read_optional(
        "SELECT season, week, player_id, market, mu, model_version "
        "FROM weekly_projections WHERE season = ? AND week = ?",
        (season, week),
        table="weekly_projections",
        conn=conn,
    )
    if projections.empty:
        return _no_data(
            title, f"weekly_projections has no rows for {season} W{week:02d}", section
        )

    stat_columns = _available_stat_columns(conn=conn)
    if not stat_columns:
        return _no_data(
            title,
            "player_stats_enhanced carries none of the market stat columns",
            section,
        )

    actuals = read_dataframe(
        f"SELECT season, week, player_id, position, {', '.join(stat_columns)} "
        "FROM player_stats_enhanced WHERE season = ? AND week = ?",
        (season, week),
        conn=conn,
    )
    if actuals.empty:
        return _no_data(
            title,
            f"player_stats_enhanced has no actuals for {season} W{week:02d} "
            f"({len(projections)} projections are still ungraded)",
            section,
        )

    # weekly_projections mints ids from the season roster (ARI_james_conner)
    # while player_stats_enhanced uses an abbreviated legacy form
    # (ARI_j_conner). This is the same gsis_id bridge the projection path uses;
    # it is a no-op for seasons whose two tables already agree.
    actuals = _bridge_history_ids(actuals, players=projections, season=season, conn=conn)

    melted = melt_actuals(actuals)
    positions = (
        actuals[["season", "week", "player_id", "position"]]
        .drop_duplicates(["season", "week", "player_id"], keep="last")
        .copy()
    )
    positions["position"] = positions["position"].astype("string").str.upper().fillna("UNKNOWN")

    scored = projections.merge(
        melted, on=["season", "week", "player_id", "market"], how="inner"
    ).merge(positions, on=["season", "week", "player_id"], how="left")
    scored["mu"] = pd.to_numeric(scored["mu"], errors="coerce")
    scored["actual"] = pd.to_numeric(scored["actual"], errors="coerce")
    scored = scored.dropna(subset=["mu", "actual"])

    if scored.empty:
        return _no_data(
            title,
            f"none of the {len(projections)} projections for {season} W{week:02d} joined to "
            "an actual (id namespaces did not reconcile, or the players did not play)",
            section,
        )

    scored["signed_error"] = scored["mu"] - scored["actual"]
    scored["abs_error"] = scored["signed_error"].abs()
    scored["position"] = scored["position"].fillna("UNKNOWN").astype(str)

    by_position: List[Dict[str, Any]] = []
    for position, group in scored.groupby("position", sort=True):
        threshold = POSITION_MAE_THRESHOLDS.get(position)
        mae = float(group["abs_error"].mean())
        by_position.append(
            {
                "position": position,
                "n": int(len(group)),
                "mae": mae,
                "mean_bias": float(group["signed_error"].mean()),
                "threshold": None if threshold is None else float(threshold),
                "over_threshold": bool(threshold is not None and mae > threshold),
                "below_min_sample": bool(len(group) < MIN_POSITION_SAMPLE),
            }
        )

    by_market: List[Dict[str, Any]] = [
        {
            "market": str(market),
            "n": int(len(group)),
            "mae": float(group["abs_error"].mean()),
            "mean_bias": float(group["signed_error"].mean()),
        }
        for market, group in scored.groupby("market", sort=True)
    ]

    payload: Dict[str, Any] = {
        "section": section,
        "status": "ok",
        "projection_rows": int(len(projections)),
        "scored_rows": int(len(scored)),
        "unmatched_rows": int(len(projections) - len(scored)),
        "overall_mae": float(scored["abs_error"].mean()),
        "overall_bias": float(scored["signed_error"].mean()),
        "min_position_sample": int(MIN_POSITION_SAMPLE),
        "by_position": by_position,
        "by_market": by_market,
        "model_versions": sorted({_text(v) for v in projections["model_version"]}),
    }

    body = [
        title,
        "",
        f"Scored {len(scored)} of {len(projections)} projections "
        f"({payload['unmatched_rows']} had no actual). "
        f"Overall MAE {payload['overall_mae']:.2f}, mean bias "
        f"{payload['overall_bias']:+.2f} (positive = the model projected too high).",
        "",
        _markdown_table(
            ["position", "n", "MAE", "ceiling", "bias", "note"],
            [
                [
                    row["position"],
                    str(row["n"]),
                    _fmt(row["mae"], 2),
                    _fmt(row["threshold"], 1),
                    f"{row['mean_bias']:+.2f}",
                    _position_note(row),
                ]
                for row in by_position
            ],
        ),
        "",
        _markdown_table(
            ["market", "n", "MAE", "bias"],
            [
                [
                    row["market"],
                    str(row["n"]),
                    _fmt(row["mae"], 2),
                    f"{row['mean_bias']:+.2f}",
                ]
                for row in by_market
            ],
        ),
        "",
        "_Ceilings are the `make mae-gate` thresholds. A position with no ceiling is "
        f"reported but not judged, and fewer than {MIN_POSITION_SAMPLE} projections is too "
        "few to read as a trend either way._",
    ]
    return "\n".join(body), payload


def _position_note(row: Mapping[str, Any]) -> str:
    notes = []
    if row["below_min_sample"]:
        notes.append(f"under {MIN_POSITION_SAMPLE} projections")
    if row["over_threshold"]:
        notes.append("OVER CEILING")
    if row["threshold"] is None:
        notes.append("no ceiling defined")
    return "; ".join(notes) if notes else "ok"


def _available_stat_columns(*, conn: Optional[DBConnection] = None) -> List[str]:
    """Market stat columns that actually exist on ``player_stats_enhanced``."""
    present = set(get_table_columns("player_stats_enhanced", conn=conn))
    return sorted(set(MARKET_TO_STAT.values()) & present)
