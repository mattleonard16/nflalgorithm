"""
Enhanced Monitoring and Alerting System for NFL Algorithm
=========================================================

Provides comprehensive system health monitoring, performance metrics,
automated alerts, and professional-grade reporting capabilities.
"""

import logging
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import numpy as np
import pandas as pd
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from config import config

logger = logging.getLogger(__name__)

class SystemMonitor:
    """Comprehensive system monitoring and alerting."""
    
    def __init__(self):
        self.alert_thresholds = self._define_alert_thresholds()
        self.metrics_history = []
        
    def _define_alert_thresholds(self) -> Dict[str, Dict]:
        """Define alert thresholds for different metrics."""
        return {
            'model_performance': {
                'rushing_mae_critical': 4.5,
                'rushing_mae_warning': 4.0,
                'receiving_mae_critical': 5.0,
                'receiving_mae_warning': 4.5,
                'prediction_count_min': 50  # Minimum predictions per week
            },
            'data_freshness': {
                'odds_stale_minutes': 15,
                'injuries_stale_minutes': 60,
                'weather_stale_minutes': 120,
                'last_update_max_hours': 24
            },
            'system_health': {
                'disk_usage_critical': 90,  # %
                'disk_usage_warning': 80,
                'memory_critical': 90,
                'memory_warning': 75,
                'cache_hit_rate_min': 70,  # %
                'api_success_rate_min': 85  # %
            },
            'value_betting': {
                'active_bets_warning': 5,  # Minimum active bets
                'edge_average_min': 5.0,    # %
                'roi_daily_min': 1.0,       # %
                'max_correlation_bets': 8   # High correlation risk
            }
        }
    
    def check_model_performance(self, mae_scores: Dict[str, float], 
                              prediction_count: int) -> Dict[str, Any]:
        """Check model performance against thresholds."""
        alerts = []
        status = "HEALTHY"
        
        # Check MAE scores
        rushing_mae = mae_scores.get('rushing_mae', 0)
        receiving_mae = mae_scores.get('receiving_mae', 0)
        
        if rushing_mae >= self.alert_thresholds['model_performance']['rushing_mae_critical']:
            alerts.append(f"CRITICAL: Rushing MAE ({rushing_mae:.2f}) above threshold")
            status = "CRITICAL"
        elif rushing_mae >= self.alert_thresholds['model_performance']['rushing_mae_warning']:
            alerts.append(f"WARNING: Rushing MAE ({rushing_mae:.2f}) approaching threshold")
            status = "WARNING" if status != "CRITICAL" else "CRITICAL"
        
        if receiving_mae >= self.alert_thresholds['model_performance']['receiving_mae_critical']:
            alerts.append(f"CRITICAL: Receiving MAE ({receiving_mae:.2f}) above threshold")
            status = "CRITICAL"
        elif receiving_mae >= self.alert_thresholds['model_performance']['receiving_mae_warning']:
            alerts.append(f"WARNING: Receiving MAE ({receiving_mae:.2f}) approaching threshold")
            status = "WARNING" if status != "CRITICAL" else "CRITICAL"
        
        # Check prediction count
        if prediction_count < self.alert_thresholds['model_performance']['prediction_count_min']:
            alerts.append(f"WARNING: Low prediction count ({prediction_count})")
            status = "WARNING" if status == "HEALTHY" else status
        
        return {
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'alerts': alerts,
            'metrics': {
                'rushing_mae': rushing_mae,
                'receiving_mae': receiving_mae,
                'prediction_count': prediction_count
            }
        }
    
    def check_data_freshness(self, feed_updates: Dict[str, datetime]) -> Dict[str, Any]:
        """Check data freshness across all feeds."""
        alerts = []
        status = "HEALTHY"
        fresh_status = {}
        
        thresholds = self.alert_thresholds['data_freshness']
        
        for feed_name, last_update in feed_updates.items():
            if isinstance(last_update, str):
                last_update = datetime.fromisoformat(last_update)
            
            age_minutes = (datetime.now() - last_update).total_seconds() / 60
            
            if feed_name == 'odds' and age_minutes > thresholds['odds_stale_minutes']:
                alerts.append(f"WARNING: Odds data stale ({age_minutes:.1f} minutes old)")
                status = "WARNING" if status == "HEALTHY" else status
                fresh_status[feed_name] = "STALE"
            elif feed_name == 'injuries' and age_minutes > thresholds['injuries_stale_minutes']:
                alerts.append(f"WARNING: Injury data stale ({age_minutes:.1f} minutes old)")
                status = "WARNING" if status == "HEALTHY" else status
                fresh_status[feed_name] = "STALE"
            elif feed_name == 'weather' and age_minutes > thresholds['weather_stale_minutes']:
                alerts.append(f"WARNING: Weather data stale ({age_minutes:.1f} minutes old)")
                status = "WARNING" if status == "HEALTHY" else status
                fresh_status[feed_name] = "STALE"
            elif age_minutes > thresholds['last_update_max_hours'] * 60:
                alerts.append(f"CRITICAL: {feed_name} data very stale ({age_minutes:.1f} minutes old)")
                status = "CRITICAL"
                fresh_status[feed_name] = "CRITICAL_STALE"
            else:
                fresh_status[feed_name] = "FRESH"
        
        return {
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'alerts': alerts,
            'feed_status': fresh_status,
            'feed_updates': {k: v.isoformat() for k, v in feed_updates.items()}
        }
    
    def check_system_health(self, cache_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Check overall system health metrics."""
        alerts = []
        status = "HEALTHY"
        
        # Check cache hit rate
        hit_rate = cache_stats.get('hit_rate_percent', 0)
        if hit_rate < self.alert_thresholds['system_health']['cache_hit_rate_min']:
            alerts.append(f"WARNING: Low cache hit rate ({hit_rate:.1f}%)")
            status = "WARNING"
        
        # Check disk usage (simplified)
        try:
            disk_usage = self._get_disk_usage()
            if disk_usage >= self.alert_thresholds['system_health']['disk_usage_critical']:
                alerts.append(f"CRITICAL: Disk usage at {disk_usage:.1f}%")
                status = "CRITICAL"
            elif disk_usage >= self.alert_thresholds['system_health']['disk_usage_warning']:
                alerts.append(f"WARNING: Disk usage at {disk_usage:.1f}%")
                status = "WARNING" if status == "HEALTHY" else status
        except Exception as e:
            logger.warning(f"Could not check disk usage: {e}")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'alerts': alerts,
            'metrics': {
                'cache_hit_rate': hit_rate,
                'disk_usage': disk_usage if 'disk_usage' in locals() else None
            }
        }
    
    def check_value_betting_health(self, betting_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check value betting system health."""
        alerts = []
        status = "HEALTHY"
        
        # Check active bets count
        active_bets = betting_metrics.get('active_bets', 0)
        if active_bets < self.alert_thresholds['value_betting']['active_bets_warning']:
            alerts.append(f"WARNING: Low active bet count ({active_bets})")
            status = "WARNING"
        
        # Check average edge
        avg_edge = betting_metrics.get('average_edge', 0)
        if avg_edge < self.alert_thresholds['value_betting']['edge_average_min']:
            alerts.append(f"WARNING: Low average edge ({avg_edge:.1f}%)")
            status = "WARNING"
        
        # Check daily ROI
        daily_roi = betting_metrics.get('daily_roi', 0)
        if daily_roi < self.alert_thresholds['value_betting']['roi_daily_min']:
            alerts.append(f"WARNING: Low daily ROI ({daily_roi:.1f}%)")
            status = "WARNING"
        
        return {
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'alerts': alerts,
            'metrics': betting_metrics
        }
    
    def _get_disk_usage(self) -> float:
        """Get current disk usage percentage."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(config.project_root)
            return (used / total) * 100
        except Exception:
            return 0.0
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'system_status': 'HEALTHY',
            'components': {}
        }
        
        try:
            # Model Performance
            model_metrics = self._get_latest_model_metrics()
            model_status = self.check_model_performance(
                model_metrics['mae_scores'],
                model_metrics['prediction_count']
            )
            report['components']['model_performance'] = model_status
            
            # Data Freshness
            feed_updates = self._get_feed_update_status()
            data_status = self.check_data_freshness(feed_updates)
            report['components']['data_freshness'] = data_status
            
            # System Health
            cache_stats = self._get_cache_statistics()
            system_status = self.check_system_health(cache_stats)
            report['components']['system_health'] = system_status
            
            # Value Betting
            betting_metrics = self._get_betting_metrics()
            betting_status = self.check_value_betting_health(betting_metrics)
            report['components']['value_betting'] = betting_status
            
            # Determine overall status
            statuses = [comp['status'] for comp in report['components'].values()]
            if 'CRITICAL' in statuses:
                report['system_status'] = 'CRITICAL'
            elif 'WARNING' in statuses:
                report['system_status'] = 'WARNING'
            else:
                report['system_status'] = 'HEALTHY'
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            report['error'] = str(e)
            report['system_status'] = 'ERROR'
        
        # Store in history
        self.metrics_history.append({
            'timestamp': report['generated_at'],
            'status': report['system_status'],
            'report': report
        })
        
        return report
    
    def _get_latest_model_metrics(self) -> Dict[str, Any]:
        """Get latest model performance metrics."""
        try:
            conn = sqlite3.connect(config.database.path)
            
            # Get recent validation results
            query = """
            SELECT metric_type, metric_value, recorded_at 
            FROM model_performance_metrics 
            WHERE recorded_at >= datetime('now', '-7 days')
            ORDER BY recorded_at DESC
            LIMIT 100
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return {'mae_scores': {'rushing_mae': 3.6, 'receiving_mae': 4.1}, 'prediction_count': 0}
            
            # Parse latest metrics
            latest = df.iloc[0] if len(df) > 0 else None
            mae_scores = {}
            prediction_count = 0
            
            if latest is not None:
                rushing_maes = df[df['metric_type'] == 'rushing_mae']['metric_value']
                receiving_maes = df[df['metric_type'] == 'receiving_mae']['metric_value']
                pred_counts = df[df['metric_type'] == 'prediction_count']['metric_value']
                
                mae_scores['rushing_mae'] = rushing_maes.iloc[0] if not rushing_maes.empty else 3.6
                mae_scores['receiving_mae'] = receiving_maes.iloc[0] if not receiving_maes.empty else 4.1
                prediction_count = pred_counts.iloc[0] if not pred_counts.empty else 0
            else:
                mae_scores = {'rushing_mae': 3.6, 'receiving_mae': 4.1}
                prediction_count = 0
            
            return {'mae_scores': mae_scores, 'prediction_count': prediction_count}
            
        except Exception as e:
            logger.error(f"Error getting model metrics: {e}")
            return {'mae_scores': {'rushing_mae': 3.6, 'receiving_mae': 4.1}, 'prediction_count': 0}
    
    def _get_feed_update_status(self) -> Dict[str, datetime]:
        """Get last update times for all data feeds."""
        try:
            conn = sqlite3.connect(config.database.path)
            
            query = """
            SELECT feed, MAX(as_of) as last_update
            FROM feed_freshness
            WHERE recorded_at >= datetime('now', '-2 days')
            GROUP BY feed
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            feed_updates = {}
            for _, row in df.iterrows():
                feed_updates[row['feed']] = row['last_update']
            
            return feed_updates
            
        except Exception as e:
            logger.error(f"Error getting feed status: {e}")
            return {}
    
    def _get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        try:
            from scripts.simple_cache import simple_cached_client
            return simple_cached_client.get_cache_stats()
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    def _get_betting_metrics(self) -> Dict[str, Any]:
        """Get value betting performance metrics."""
        try:
            conn = sqlite3.connect(config.database.path)
            
            # Get current active bets
            active_query = """
            SELECT COUNT(*) as active_bets,
                   AVG(edge_percentage) as average_edge,
                   AVG(expected_roi) as average_roi
            FROM enhanced_value_bets 
            WHERE date_identified >= datetime('now', '-1 day')
            """
            
            df = pd.read_sql_query(active_query, conn)
            conn.close()
            
            if df.empty:
                return {
                    'active_bets': 0,
                    'average_edge': 0,
                    'daily_roi': 0
                }
            
            row = df.iloc[0]
            return {
                'active_bets': int(row['active_bets']),
                'average_edge': float(row['average_edge'] or 0),
                'daily_roi': (float(row['average_roi'] or 0)) * 100
            }
            
        except Exception as e:
            logger.error(f"Error getting betting metrics: {e}")
            return {'active_bets': 0, 'average_edge': 0, 'daily_roi': 0}
    
    def should_send_alert(self, report: Dict[str, Any]) -> bool:
        """Determine if alert should be sent based on report content."""
        return (
            report['system_status'] in ['CRITICAL', 'WARNING'] or
            len([alert for component in report['components'].values() 
                 for alert in component.get('alerts', [])]) > 0
        )
    
    def format_alert_message(self, report: Dict[str, Any]) -> str:
        """Format alert message for notification."""
        status_emoji = {'HEALTHY': '✅', 'WARNING': '⚠️', 'CRITICAL': '🚨', 'ERROR': '❌'}
        
        message = f"{status_emoji.get(report['system_status'], '❓')} NFL Algorithm Alert\n"
        message += f"Status: {report['system_status']}\n"
        message += f"Time: {report['generated_at']}\n\n"
        
        for component_name, component_data in report['components'].items():
            if component_data.get('alerts'):
                message += f"{component_name.upper()}:\n"
                for alert in component_data['alerts'][:3]:  # Limit to 3 alerts per component
                    message += f"  • {alert}\n"
                message += "\n"
        
        message += f"Full details available in monitoring dashboard."
        return message
    
    def save_monitoring_report(self, report: Dict[str, Any]) -> str:
        """Save monitoring report to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = config.logs_dir / f"monitoring_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Monitoring report saved to {report_path}")
        return str(report_path)


class DailyReporter:
    """Generates daily performance and monitoring reports."""
    
    def __init__(self, monitor: SystemMonitor):
        self.monitor = monitor
    
    def generate_daily_report(self) -> str:
        """Generate comprehensive daily performance report."""
        report_data = self.monitor.generate_comprehensive_report()
        
        markdown = f"# NFL Algorithm Daily Report\n\n"
        markdown += f"**Generated:** {report_data['generated_at']}\n"
        markdown += f"**System Status:** {report_data['system_status']}\n\n"
        
        # Component summaries
        for component_name, component_data in report_data['components'].items():
            status = component_data['status']
            emoji = {'HEALTHY': '✅', 'WARNING': '⚠️', 'CRITICAL': '🚨'}.get(status, '❓')
            
            markdown += f"## {component_name.replace('_', ' ').title()} {emoji}\n"
            
            # Add key metrics
            if 'metrics' in component_data:
                metrics = component_data['metrics']
                for metric_name, metric_value in metrics.items():
                    if metric_value is not None:
                        if isinstance(metric_value, float):
                            markdown += f"- **{metric_name.replace('_', ' ').title()}:** {metric_value:.2f}\n"
                        else:
                            markdown += f"- **{metric_name.replace('_', ' ').title()}:** {metric_value}\n"
            
            # Add alerts if any
            if component_data.get('alerts'):
                markdown += "\n**Alerts:**\n"
                for alert in component_data['alerts']:
                    markdown += f"- {alert}\n"
            
            markdown += "\n"
        
        # System overview
        markdown += "## System Overview\n"
        markdown += f"- Overall Status: {report_data['system_status']}\n"
        markdown += f"- Components with Issues: {sum(1 for c in report_data['components'].values() if c['status'] != 'HEALTHY')}\n"
        markdown += f"- Total Components: {len(report_data['components'])}\n"
        
        return markdown
    
    def save_daily_report(self) -> str:
        """Save daily markdown report."""
        report_content = self.generate_daily_report()
        
        timestamp = datetime.now().strftime('%Y%m%d')
        report_path = config.reports_dir / f"daily_report_{timestamp}.md"
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Daily report saved to {report_path}")
        return str(report_path)


# Global instances
system_monitor = SystemMonitor()
daily_reporter = DailyReporter(system_monitor)
