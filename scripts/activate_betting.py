#!/usr/bin/env python3
"""
Activate Value Betting - UV Optimized
====================================

Generates value betting opportunities by:
- Ensuring current prop lines exist (scrape or sample)
- Merging with projections to compute edges
- Persisting opportunities to the main database tables used by the dashboard
- Emitting report artifacts (CSV/JSON/Markdown/HTML)

Usage:
    uv run python scripts/activate_betting.py
"""

from __future__ import annotations

import sys
import os
import uuid
from datetime import datetime
from typing import List

import pandas as pd

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from scripts.prop_line_scraper import NFLPropScraper
from prop_integration import PropIntegration
from value_betting_engine import ValueBettingEngine
from utils.db import get_connection, column_exists, execute, executemany


def _ensure_column(conn, table: str, column: str, ddl: str) -> None:
    if not column_exists(table, column, conn=conn):
        execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}", conn=conn)


def ensure_tables(conn) -> None:
    """Create the dashboard tables if they do not already exist."""
    execute(
        """
        CREATE TABLE IF NOT EXISTS enhanced_value_bets (
            bet_id TEXT PRIMARY KEY,
            player_name TEXT NOT NULL,
            position TEXT,
            team TEXT,
            prop_type TEXT,
            sportsbook TEXT,
            line REAL,
            model_prediction REAL,
            model_confidence REAL,
            edge_yards REAL,
            edge_percentage REAL,
            kelly_fraction REAL,
            expected_roi REAL,
            risk_level TEXT,
            recommendation TEXT,
            bet_size_units REAL,
            correlation_risk TEXT,
            market_efficiency REAL,
            date_identified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        conn=conn,
    )

    # Backfill/migrate columns if an older schema exists
    for col, ddl in [
        ("bet_id", "TEXT"),
        ("player_name", "TEXT"),
        ("position", "TEXT"),
        ("team", "TEXT"),
        ("prop_type", "TEXT"),
        ("sportsbook", "TEXT"),
        ("line", "REAL"),
        ("model_prediction", "REAL"),
        ("model_confidence", "REAL"),
        ("edge_yards", "REAL"),
        ("edge_percentage", "REAL"),
        ("kelly_fraction", "REAL"),
        ("expected_roi", "REAL"),
        ("risk_level", "TEXT"),
        ("recommendation", "TEXT"),
        ("bet_size_units", "REAL"),
        ("correlation_risk", "TEXT"),
        ("market_efficiency", "REAL"),
        ("date_identified", "TIMESTAMP")
    ]:
        try:
            _ensure_column(conn, "enhanced_value_bets", col, ddl)
        except Exception:
            # Skip if table doesn't exist yet; it was just created above
            pass

    execute(
        """
        CREATE TABLE IF NOT EXISTS clv_tracking (
            bet_id TEXT PRIMARY KEY,
            player_id TEXT,
            prop_type TEXT,
            sportsbook TEXT,
            bet_line REAL,
            closing_line REAL,
            clv_percentage REAL,
            bet_result TEXT,
            roi REAL,
            date_placed DATE,
            date_settled DATE
        )
        """,
        conn=conn,
    )

    for col, ddl in [
        ("bet_id", "TEXT"),
        ("player_id", "TEXT"),
        ("prop_type", "TEXT"),
        ("sportsbook", "TEXT"),
        ("bet_line", "REAL"),
        ("closing_line", "REAL"),
        ("clv_percentage", "REAL"),
        ("bet_result", "TEXT"),
        ("roi", "REAL"),
        ("date_placed", "DATE"),
        ("date_settled", "DATE"),
    ]:
        try:
            _ensure_column(conn, "clv_tracking", col, ddl)
        except Exception:
            pass


def opportunities_to_value_bets(opps: pd.DataFrame) -> pd.DataFrame:
    """Transform PropIntegration opportunities into the enhanced_value_bets schema.

    Required input columns:
    - player, team, position, book, stat, line, model_prediction, edge_yards, edge_percentage,
      over_odds, under_odds, value_rating
    """
    if opps is None or opps.empty:
        return pd.DataFrame(
            columns=[
                "bet_id",
                "player_name",
                "position",
                "team",
                "prop_type",
                "sportsbook",
                "line",
                "model_prediction",
                "model_confidence",
                "edge_yards",
                "edge_percentage",
                "kelly_fraction",
                "expected_roi",
                "risk_level",
                "recommendation",
                "bet_size_units",
                "correlation_risk",
                "market_efficiency",
                "date_identified",
            ]
        )

    engine = ValueBettingEngine()

    rows: List[dict] = []
    for _, r in opps.iterrows():
        try:
            direction = "OVER" if float(r.get("edge_yards", 0) or 0) > 0 else "UNDER"
            odds = int(r["over_odds"]) if direction == "OVER" else int(r["under_odds"]) if pd.notna(r.get("under_odds")) else -110

            # Confidence heuristic from value rating / edge
            value_rating = str(r.get("value_rating", "NO_VALUE"))
            abs_edge_pct = abs(float(r.get("edge_percentage", 0) or 0))
            if value_rating == "HIGH_VALUE" or abs_edge_pct >= 15:
                model_confidence = 0.86
            elif value_rating == "MEDIUM_VALUE" or abs_edge_pct >= 8:
                model_confidence = 0.80
            elif abs_edge_pct >= 3:
                model_confidence = 0.76
            else:
                model_confidence = 0.72

            # Kelly and ROI
            kelly_fraction = engine.calculate_fractional_kelly(model_confidence, odds, 1.0)
            expected_roi = engine._calculate_expected_roi(model_confidence, odds)  # noqa: SLF001

            # Risk bucket mirrors value rating
            risk_level = value_rating if value_rating in {"HIGH_VALUE", "MEDIUM_VALUE", "LOW_VALUE"} else "NO_VALUE"

            rows.append(
                {
                    "bet_id": str(uuid.uuid4()),
                    "player_name": r.get("player"),
                    "position": r.get("position"),
                    "team": r.get("team"),
                    "prop_type": r.get("stat"),
                    "sportsbook": r.get("book"),
                    "line": float(r.get("line", 0) or 0),
                    "model_prediction": float(r.get("model_prediction", 0) or 0),
                    "model_confidence": float(model_confidence),
                    "edge_yards": float(r.get("edge_yards", 0) or 0),
                    "edge_percentage": float(r.get("edge_percentage", 0) or 0),
                    "kelly_fraction": float(kelly_fraction),
                    "expected_roi": float(expected_roi),
                    "risk_level": risk_level,
                    "recommendation": direction,
                    "bet_size_units": float(max(0.0, kelly_fraction) * 100.0),
                    "correlation_risk": "LOW",
                    "market_efficiency": 0.5,
                    "date_identified": datetime.now(),
                }
            )
        except Exception:
            # Skip malformed rows
            continue

    return pd.DataFrame(rows)


def persist_value_bets(df: pd.DataFrame) -> int:
    """Insert enhanced value bets into the main database and initialize CLV tracking."""
    if df is None or df.empty:
        return 0

    with get_connection() as conn:
        ensure_tables(conn)

        # Insert enhanced value bets
        # Convert datetime columns to ISO strings for SQLite
        if "date_identified" in df.columns:
            df = df.copy()
            df["date_identified"] = df["date_identified"].apply(lambda x: x.isoformat() if hasattr(x, "isoformat") else x)

        insert_sql = (
            """
            INSERT INTO enhanced_value_bets (
                bet_id, player_name, position, team, prop_type, sportsbook, line,
                model_prediction, model_confidence, edge_yards, edge_percentage, kelly_fraction,
                expected_roi, risk_level, recommendation, bet_size_units,
                correlation_risk, market_efficiency, date_identified
            ) VALUES (
                :bet_id, :player_name, :position, :team, :prop_type, :sportsbook, :line,
                :model_prediction, :model_confidence, :edge_yards, :edge_percentage, :kelly_fraction,
                :expected_roi, :risk_level, :recommendation, :bet_size_units,
                :correlation_risk, :market_efficiency, :date_identified
            )
            ON CONFLICT(bet_id) DO UPDATE SET
                player_name = excluded.player_name,
                position = excluded.position,
                team = excluded.team,
                prop_type = excluded.prop_type,
                sportsbook = excluded.sportsbook,
                line = excluded.line,
                model_prediction = excluded.model_prediction,
                model_confidence = excluded.model_confidence,
                edge_yards = excluded.edge_yards,
                edge_percentage = excluded.edge_percentage,
                kelly_fraction = excluded.kelly_fraction,
                expected_roi = excluded.expected_roi,
                risk_level = excluded.risk_level,
                recommendation = excluded.recommendation,
                bet_size_units = excluded.bet_size_units,
                correlation_risk = excluded.correlation_risk,
                market_efficiency = excluded.market_efficiency,
                date_identified = excluded.date_identified
            """
        )
        executemany(insert_sql, df.to_dict(orient="records"), conn=conn)

        # Initialize CLV tracking rows
        today_str = datetime.now().date().isoformat()
        clv_rows = [
            {
                "bet_id": r["bet_id"],
                "player_id": None,
                "prop_type": r["prop_type"],
                "sportsbook": r["sportsbook"],
                "bet_line": r["line"],
                "date_placed": today_str,
            }
            for _, r in df.iterrows()
        ]
        executemany(
            """
            INSERT INTO clv_tracking
                (bet_id, player_id, prop_type, sportsbook, bet_line, date_placed)
            VALUES
                (:bet_id, :player_id, :prop_type, :sportsbook, :bet_line, :date_placed)
            ON CONFLICT(bet_id)
            DO UPDATE SET
                player_id = excluded.player_id,
                prop_type = excluded.prop_type,
                sportsbook = excluded.sportsbook,
                bet_line = excluded.bet_line,
                date_placed = excluded.date_placed
            """,
            clv_rows,
            conn=conn,
        )

        conn.commit()
        return len(df)


def main() -> int:
    print("🚀 Activating value betting pipeline...")

    # 1) Ensure current prop lines
    scraper = NFLPropScraper()
    prop_df = scraper.run_weekly_update()
    print(f"   ✅ Prop lines ready: {len(prop_df)} rows")

    # 2) Merge with projections and compute opportunities
    integrator = PropIntegration()

    # Use config threshold (convert from fraction to percentage expected by integrator)
    min_edge_pct = max(1.0, config.betting.min_edge_threshold * 100.0)
    opps = integrator.get_best_value_opportunities(min_edge_threshold=min_edge_pct)

    # Always export artifacts (empty-safe)
    md_text = integrator.generate_value_report()
    config.reports_dir.mkdir(exist_ok=True)
    config.reports_img_dir.mkdir(parents=True, exist_ok=True)
    (config.reports_dir / "weekly_value_report.md").write_text(md_text, encoding="utf-8")
    integrator.render_html_from_markdown(md_text, config.reports_dir / "weekly_value_report.html")
    csv_path = integrator.export_opportunities_csv(opps, config.reports_dir / "value_bets.csv")
    json_path = integrator.export_opportunities_json(opps, config.reports_dir / "value_bets.json")

    num_opps = int(len(opps)) if isinstance(opps, pd.DataFrame) else 0
    print(f"   ✅ Opportunities computed: {num_opps}")
    print(f"   📄 Artifacts: {csv_path}, {json_path}")

    # 3) Persist to dashboard tables
    value_bets_df = opportunities_to_value_bets(opps)
    inserted = persist_value_bets(value_bets_df)
    print(f"   💾 Persisted enhanced value bets: {inserted}")

    # 4) Update real-time finder table for compatibility
    integrator.update_real_time_value_finder()

    # 5) Final summary
    print("\n📊 ACTIVATION SUMMARY")
    print("---------------------")
    print(f"Prop lines: {len(prop_df)}")
    print(f"Value opportunities: {num_opps}")
    print(f"Enhanced bets persisted: {inserted}")
    print("✅ System operational! Launch the dashboard: http://localhost:8501")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
