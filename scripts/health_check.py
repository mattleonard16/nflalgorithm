"""Enhanced health check with comprehensive monitoring."""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict
import json

from config import config
from scripts.monitoring import system_monitor, daily_reporter


THRESHOLDS = {
    'odds': timedelta(minutes=2),
    'injuries': timedelta(minutes=30),
    'weather': timedelta(minutes=60)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate feed freshness for weekly pipeline")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    return parser.parse_args()


def load_feed_freshness(season: int, week: int) -> Dict[str, datetime]:
    with sqlite3.connect(config.database.path) as conn:
        rows = conn.execute(
            "SELECT feed, as_of FROM feed_freshness WHERE season = ? AND week = ?",
            (season, week)
        ).fetchall()

    freshness: Dict[str, datetime] = {}
    for feed, as_of in rows:
        try:
            freshness[feed] = datetime.fromisoformat(as_of)
        except ValueError:
            continue
    return freshness


def check_freshness(freshness: Dict[str, datetime]) -> Dict[str, float]:
    now = datetime.now(timezone.utc)
    ages = {}
    for feed, timestamp in freshness.items():
        ages[feed] = (now - timestamp).total_seconds() / 60.0
    return ages


def main() -> None:
    args = parse_args()
    
    # Run traditional checks
    freshness = load_feed_freshness(args.season, args.week)
    if not freshness:
        print(f"No feed freshness data for season {args.season} week {args.week}")
    
    ages = check_freshness(freshness)
    failures = []
    for feed, threshold in THRESHOLDS.items():
        age_minutes = ages.get(feed)
        if age_minutes is None:
            failures.append(f"missing feed '{feed}'")
            continue
        if age_minutes > threshold.total_seconds() / 60.0:
            failures.append(f"{feed} stale {age_minutes:.1f}m > {threshold.total_seconds()/60:.0f}m")

    # Run comprehensive monitoring
    print("🏥 NFL Algorithm Health Check")
    print("=" * 50)
    
    comprehensive_report = system_monitor.generate_comprehensive_report()
    
    # Display system status
    status_emoji = {'HEALTHY': '✅', 'WARNING': '⚠️', 'CRITICAL': '🚨', 'ERROR': '❌'}.get(
        comprehensive_report['system_status'], '❓'
    )
    print(f"Overall Status: {status_emoji} {comprehensive_report['system_status']}")
    
    # Display component statuses
    for component_name, component_data in comprehensive_report['components'].items():
        status = component_data['status']
        emoji = {'HEALTHY': '✅', 'WARNING': '⚠️', 'CRITICAL': '🚨', 'ERROR': '❌'}.get(status, '❓')
        print(f"\n{component_name.replace('_', ' ').title()}: {emoji} {status}")
        
        if component_data.get('alerts'):
            for alert in component_data['alerts'][:2]:  # Show top 2 alerts
                print(f"  ⚠️  {alert}")
        
        # Show key metrics
        if 'metrics' in component_data:
            for metric_name, metric_value in component_data['metrics'].items():
                if metric_value is not None:
                    if isinstance(metric_value, float):
                        print(f"  📊 {metric_name.replace('_', ' ').title()}: {metric_value:.2f}")
                    else:
                        print(f"  📊 {metric_name.replace('_', ' ').title()}: {metric_value}")
    
    # Traditional feed freshness results
    print(f"\n📋 Feed Freshness (Season {args.season}, Week {args.week}):")
    if freshness:
        for feed, age in ages.items():
            status_emoji = "✅" if age <= THRESHOLDS.get(feed, timedelta(hours=1)).total_seconds() / 60 else "❌"
            print(f"  {status_emoji} {feed}: {age:.1f} minutes old")
    else:
        print("  ❌ No freshness data available")
    
    # Show any failures from traditional checks
    if failures:
        print(f"\n❌ Traditional Health Check Failed:")
        for failure in failures:
            print(f"  • {failure}")
    else:
        print("\n✅ Traditional Health Check Passed")
    
    # Save comprehensive report
    report_path = system_monitor.save_monitoring_report(comprehensive_report)
    print(f"\n💾 Comprehensive report saved to: {report_path}")
    
    # Generate and save daily report
    if datetime.now().hour == 8:  # Run at 8 AM
        daily_report_path = daily_reporter.save_daily_report()
        print(f"📄 Daily report saved to: {daily_report_path}")
    
    # Determine exit code
    should_exit_with_error = (
        comprehensive_report['system_status'] in ['CRITICAL', 'ERROR'] or
        len(failures) > 0
    )
    
    if should_exit_with_error:
        print(f"\n🚨 System health issues detected - please review alerts above")
        raise SystemExit("Health check failed")
    else:
        print(f"\n✅ All systems operational")


if __name__ == "__main__":
    main()
