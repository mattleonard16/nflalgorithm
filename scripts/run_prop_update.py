#!/usr/bin/env python3
"""
Weekly prop line update script
Run this weekly to refresh all prop lines and value opportunities
"""

import argparse
import logging
import sys
from datetime import datetime

import pandas as pd

from config import config
from prop_integration import PropIntegration
from scripts.prepare_nfl_week import prepare_week
from scripts.prop_line_scraper import NFLPropScraper

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("prop_update.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def refresh_pregame_inputs(
    season: int,
    week: int,
    *,
    scraper: NFLPropScraper | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Run canonical preparation followed by a live-only odds refresh."""
    prepare_summary = prepare_week(season, week)
    active_scraper = scraper or NFLPropScraper()
    prop_df = active_scraper.run_weekly_update(
        week,
        season,
        allow_synthetic=False,
    )
    if prop_df.empty:
        raise RuntimeError("Live odds refresh returned no rows")
    return prepare_summary, prop_df


def main():
    """Run complete prop line update process. Season and week are required."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, required=True, help="NFL week number (1-22)")
    parser.add_argument("--season", type=int, required=True, help="NFL season year (e.g., 2025)")
    args = parser.parse_args()

    print("NFL PROP LINE WEEKLY UPDATE")
    print("=" * 50)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Step 1: Run the sole validated pregame data/projection workflow.
        scraper = NFLPropScraper()
        print(
            f"\n1. Preparing causal roster-backed projections for season={args.season} week={args.week}..."
        )
        prepare_summary, prop_df = refresh_pregame_inputs(
            args.season,
            args.week,
            scraper=scraper,
        )
        print(f"   ✅ Prepared {prepare_summary['predictions']} projections")

        # Step 2: Scrape live current prop lines.
        print(f"\n2. Scraping live weekly prop lines for season={args.season} week={args.week}...")

        print(f"   ✅ Retrieved {len(prop_df)} prop lines")

        # Step 3: Integrate with existing predictions
        print("\n3. Integrating with player projections...")
        integrator = PropIntegration()
        opportunities = integrator.get_best_value_opportunities(
            season=args.season,
            week=args.week,
        )
        # Also refresh dashboard table.
        integrator.update_real_time_value_finder(season=args.season, week=args.week)

        num_opps = len(opportunities) if isinstance(opportunities, pd.DataFrame) else 0
        print(f"   ✅ Found {num_opps} value opportunities")

        # Step 4: Generate summary report
        print("\n4. Generating value report...")
        md_report = integrator.generate_value_report(season=args.season, week=args.week)
        # Ensure directories
        config.reports_dir.mkdir(exist_ok=True)
        config.reports_img_dir.mkdir(parents=True, exist_ok=True)

        # Save Markdown
        md_path = config.reports_dir / "weekly_value_report.md"
        md_path.write_text(md_report, encoding="utf-8")

        # Render HTML
        html_path = integrator.render_html_from_markdown(
            md_report, config.reports_dir / "weekly_value_report.html"
        )

        # Export CSV/JSON
        csv_path = integrator.export_opportunities_csv(
            opportunities, config.reports_dir / "value_bets.csv"
        )
        json_path = integrator.export_opportunities_json(
            opportunities, config.reports_dir / "value_bets.json"
        )

        # Charts
        chart_paths = integrator.generate_charts(opportunities)

        print(
            f"   ✅ Artifacts saved:\n     - {md_path}\n     - {html_path}\n     - {csv_path}\n     - {json_path}"
        )
        for key, p in chart_paths.items():
            print(f"     - {p}")

        # Step 5: Display summary
        print("\nUPDATE SUMMARY")
        print("-" * 30)
        print(f"Prop lines scraped: {len(prop_df)}")
        print(f"Value opportunities: {num_opps}")
        try:
            print(f"Books covered: {prop_df['book'].nunique()}")
            print(f"Players covered: {prop_df['player'].nunique()}")
        except Exception:
            pass

        # Show suspicious lines if any (guard against missing column)
        try:
            prop_df = scraper.flag_suspicious_lines(prop_df)
        except Exception:
            pass
        suspicious_mask = (
            prop_df["suspicious_line"]
            if "suspicious_line" in prop_df.columns
            else pd.Series([False] * len(prop_df))
        )
        suspicious_lines = prop_df[suspicious_mask]
        if len(suspicious_lines) > 0:
            print(f"\n⚠️  SUSPICIOUS LINES ({len(suspicious_lines)}):")
            for _, row in suspicious_lines.iterrows():
                print(
                    f"  {row['player']} - {row['stat']}: {row['line']} ({row.get('suspicious_reason', 'Unknown')})"
                )

        # Show top opportunities as pretty table
        if isinstance(opportunities, pd.DataFrame) and len(opportunities) > 0:
            print(f"\nTOP VALUE OPPORTUNITIES:")
            top = opportunities.copy()
            # Normalize edge% to percentage if in fraction
            if "edge_percentage" in top.columns and top["edge_percentage"].max() <= 1.0:
                top["edge_pct_display"] = top["edge_percentage"] * 100.0
            else:
                top["edge_pct_display"] = top["edge_percentage"]
            cols = ["player", "stat", "line", "model_prediction", "edge_pct_display", "book"]
            cols = [c for c in cols if c in top.columns]
            # Add recommendation
            if "edge_yards" in top.columns:
                top["rec"] = top["edge_yards"].apply(
                    lambda x: "OVER" if pd.notna(x) and x > 0 else "UNDER"
                )
                cols = cols + ["rec"]
            top = top.reindex(
                top["edge_pct_display"].abs().sort_values(ascending=False).index
            ).head(10)
            # Print header
            header = " | ".join([h.replace("_", " ").title() for h in cols])
            print(header)
            print("-" * len(header))
            for _, r in top[cols].iterrows():
                out = []
                for c in cols:
                    v = r[c]
                    if c == "edge_pct_display" and pd.notna(v):
                        out.append(f"{v:+.1f}%")
                    elif c in ("line", "model_prediction") and pd.notna(v):
                        out.append(f"{float(v):.1f}")
                    else:
                        out.append(str(v))
                print(" | ".join(out))
        else:
            print("\nNo value opportunities found.")

        print(f"\nUPDATE COMPLETE")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        logger.error(f"Error during prop line update: {e}")
        print(f"\nUPDATE FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
