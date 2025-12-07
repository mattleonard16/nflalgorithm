#!/usr/bin/env python3
"""
Weekly prop line update script
Run this weekly to refresh all prop lines and value opportunities
"""

import sys
from datetime import datetime
import logging
import argparse

from scripts.prop_line_scraper import NFLPropScraper
from prop_integration import PropIntegration
from config import config
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prop_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run complete prop line update process (season-long or weekly via --week)."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--week', type=int, default=None)
    parser.add_argument('--season', type=int, default=datetime.now().year)
    args = parser.parse_args()
    
    print("NFL PROP LINE WEEKLY UPDATE")
    print("=" * 50)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Scrape current prop lines
        scraper = NFLPropScraper()
        if args.week:
            print(f"\n1. Scraping weekly prop lines for week {args.week}...")
            rows = scraper.get_upcoming_week_props(args.week, args.season)
            prop_df = pd.DataFrame(rows)
            scraper.save_weekly_prop_lines(rows, args.week, args.season)
            # also write CSV for convenience
            from pathlib import Path
            out = Path('reports') / f"week_{args.week}_prop_lines.csv"
            out.parent.mkdir(exist_ok=True)
            prop_df.to_csv(out, index=False)
        else:
            print("\n1. Scraping current prop lines...")
            # Fallback: if no week specified, run a demo weekly scrape for week 1
            prop_df = scraper.run_weekly_update(1, args.season)
        
        print(f"   ✅ Retrieved {len(prop_df)} prop lines")
        
        # Step 2: Integrate with existing predictions
        print("\n2. Integrating with player projections...")
        integrator = PropIntegration()
        if args.week:
            opportunities = integrator.export_weekly_value_bets(args.week)
        else:
            opportunities = integrator.update_real_time_value_finder()
        
        num_opps = len(opportunities) if isinstance(opportunities, pd.DataFrame) else 0
        print(f"   ✅ Found {num_opps} value opportunities")
        
        # Step 3: Generate summary report
        print("\n3. Generating value report...")
        md_report = integrator.generate_value_report() if not args.week else integrator.generate_markdown_report(opportunities)
        # Ensure directories
        config.reports_dir.mkdir(exist_ok=True)
        config.reports_img_dir.mkdir(parents=True, exist_ok=True)

        # Save Markdown
        md_path = config.reports_dir / 'weekly_value_report.md'
        md_path.write_text(md_report, encoding='utf-8')

        # Render HTML
        html_path = integrator.render_html_from_markdown(md_report, config.reports_dir / 'weekly_value_report.html')

        # Export CSV/JSON
        csv_path = integrator.export_opportunities_csv(opportunities, config.reports_dir / 'value_bets.csv')
        json_path = integrator.export_opportunities_json(opportunities, config.reports_dir / 'value_bets.json')

        # Charts
        chart_paths = integrator.generate_charts(opportunities)

        print(f"   ✅ Artifacts saved:\n     - {md_path}\n     - {html_path}\n     - {csv_path}\n     - {json_path}")
        for key, p in chart_paths.items():
            print(f"     - {p}")
        
        # Step 4: Display summary
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
        suspicious_mask = prop_df['suspicious_line'] if 'suspicious_line' in prop_df.columns else pd.Series([False]*len(prop_df))
        suspicious_lines = prop_df[suspicious_mask]
        if len(suspicious_lines) > 0:
            print(f"\n⚠️  SUSPICIOUS LINES ({len(suspicious_lines)}):")
            for _, row in suspicious_lines.iterrows():
                print(f"  {row['player']} - {row['stat']}: {row['line']} ({row.get('suspicious_reason', 'Unknown')})")
        
        # Show top opportunities as pretty table
        if isinstance(opportunities, pd.DataFrame) and len(opportunities) > 0:
            print(f"\nTOP VALUE OPPORTUNITIES:")
            top = opportunities.copy()
            # Normalize edge% to percentage if in fraction
            if 'edge_percentage' in top.columns and top['edge_percentage'].max() <= 1.0:
                top['edge_pct_display'] = top['edge_percentage'] * 100.0
            else:
                top['edge_pct_display'] = top['edge_percentage']
            cols = ['player', 'stat', 'line', 'model_prediction', 'edge_pct_display', 'book']
            cols = [c for c in cols if c in top.columns]
            # Add recommendation
            if 'edge_yards' in top.columns:
                top['rec'] = top['edge_yards'].apply(lambda x: 'OVER' if pd.notna(x) and x > 0 else 'UNDER')
                cols = cols + ['rec']
            top = top.reindex(top['edge_pct_display'].abs().sort_values(ascending=False).index).head(10)
            # Print header
            header = " | ".join([h.replace('_', ' ').title() for h in cols])
            print(header)
            print("-" * len(header))
            for _, r in top[cols].iterrows():
                out = []
                for c in cols:
                    v = r[c]
                    if c == 'edge_pct_display' and pd.notna(v):
                        out.append(f"{v:+.1f}%")
                    elif c in ('line', 'model_prediction') and pd.notna(v):
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
