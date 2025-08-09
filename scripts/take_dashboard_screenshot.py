#!/usr/bin/env python3
"""
Dashboard Screenshot Utility
============================
Takes a screenshot of the running dashboard for documentation
"""

import subprocess
import time
from pathlib import Path

def take_screenshot():
    """Take a screenshot of the dashboard using screencapture on macOS"""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    screenshot_path = reports_dir / "dashboard_screenshot.png"
    
    print("📸 Taking dashboard screenshot...")
    print("   Opening dashboard in browser...")
    
    # Open dashboard in default browser
    subprocess.run(["open", "http://localhost:8501"], check=True)
    
    # Wait for page to load
    time.sleep(5)
    
    print("   📷 Please use Cmd+Shift+4 to take a screenshot of the dashboard")
    print("   💾 Save it as 'dashboard_screenshot.png' in the reports/ folder")
    print("   ✅ Dashboard is running at: http://localhost:8501")
    
    return True

if __name__ == "__main__":
    take_screenshot()