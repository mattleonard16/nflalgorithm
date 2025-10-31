#!/bin/bash
# Archive ALL old logs and historical files
# Run with: bash archive_all.sh

set -e

mkdir -p archive/{docs,logs,reports,migrations,old_logs}

echo "📁 Archiving all logs and historical files..."

# Archive all log files
if [ -f "prop_update.log" ]; then
    mv prop_update.log archive/old_logs/ && echo "✅ Archived prop_update.log"
fi

if [ -f "dashboard.log" ]; then
    mv dashboard.log archive/old_logs/ && echo "✅ Archived dashboard.log"
fi

if [ -f "logs/pipeline.log" ]; then
    mv logs/pipeline.log archive/old_logs/ && echo "✅ Archived logs/pipeline.log"
fi

if [ -f "logs/value_betting.log" ]; then
    mv logs/value_betting.log archive/old_logs/ && echo "✅ Archived logs/value_betting.log"
fi

# Archive all monitoring reports
if ls logs/monitoring_report_*.json 1> /dev/null 2>&1; then
    mv logs/monitoring_report_*.json archive/logs/ && echo "✅ Archived monitoring reports"
fi

# Archive old validation results (keep leaderboard)
if [ -f "logs/validation_results.csv" ]; then
    mv logs/validation_results.csv archive/logs/ && echo "✅ Archived validation_results.csv"
fi

# Archive old reports (keep current week)
if [ -f "reports/TRANSFORMATION_COMPLETE.md" ]; then
    mv reports/TRANSFORMATION_COMPLETE.md archive/docs/
fi

echo ""
echo "📊 Archive summary:"
du -sh archive/* 2>/dev/null | awk '{print "  " $2 ": " $1}'

echo ""
echo "✅ All logs archived!"
