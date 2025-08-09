# NFL Prop Line Scraper Setup Guide

## Overview
This system scrapes NFL player prop lines from sportsbooks and integrates them with your existing breakout prediction model to find value betting opportunities.

## Files Created
- `prop_line_scraper.py` - Main scraper for prop lines
- `prop_integration.py` - Integrates props with your predictions
- `run_prop_update.py` - Weekly update script
- `requirements_scraper.txt` - Python dependencies

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements_scraper.txt
```

### 2. Get Odds API Key (Optional but Recommended)
1. Sign up at https://the-odds-api.com/
2. Get your free API key (1,000 requests/month)
3. Set environment variable:
   ```bash
   export ODDS_API_KEY="your_api_key_here"
   ```
   Or add to your `.env` file:
   ```
   ODDS_API_KEY=your_api_key_here
   ```

### 3. Test the Scraper
```bash
python prop_line_scraper.py
```

This will:
- Create `nfl_prop_lines.db` SQLite database
- Generate sample prop lines (or real ones if API key is set)
- Save data to `current_prop_lines.csv`
- Flag suspicious lines

### 4. Run Weekly Updates
```bash
python run_prop_update.py
```

This will:
- Scrape current prop lines
- Integrate with your existing projections
- Generate value opportunities
- Create `weekly_value_report.txt`

## Integration with Existing System

### With `real_time_value_finder.py`
The prop scraper creates a `prop_opportunities` table in your existing database that your value finder can use:

```python
# In your existing code, you can now pull real prop lines:
def get_current_odds(self):
    conn = sqlite3.connect(self.db_path)
    df = pd.read_sql_query('''
        SELECT player as player_name, stat as prop_type, book as sportsbook,
               line, over_odds, under_odds
        FROM prop_opportunities
        WHERE value_rating != 'NO_VALUE'
    ''', conn)
    conn.close()
    return df
```

### With `season_2025_predictor.py`
The integration automatically compares your projections with current lines:

```python
# Your existing projections are automatically matched with prop lines
# Edge calculations are done automatically
```

## Data Structure

### Prop Lines Table
```sql
CREATE TABLE prop_lines (
    player TEXT,
    team TEXT,
    position TEXT,
    book TEXT,
    stat TEXT,  -- 'rushing_yards', 'receiving_yards', 'passing_yards'
    line REAL,
    over_odds INTEGER,
    under_odds INTEGER,
    last_updated TEXT,
    season TEXT
);
```

### Output CSV Format
```csv
player,team,position,book,stat,line,over_odds,under_odds,last_updated,season
Christian McCaffrey,SF,RB,DraftKings,rushing_yards,1250.5,-115,-105,2025-01-11T10:30:00,2025-2026
```

## Suspicious Line Detection

The system automatically flags lines that seem too low:
- RB rushing yards < 800
- WR receiving yards < 800  
- TE receiving yards < 600
- QB passing yards < 3500

## Automation Options

### Cron Job (Linux/Mac)
```bash
# Run every Sunday at 10 AM
0 10 * * 0 cd /path/to/nflalgorithm && python run_prop_update.py
```

### Task Scheduler (Windows)
- Create task to run `run_prop_update.py` weekly
- Set working directory to your project folder

## Troubleshooting

### No API Key
- System will use sample data for testing
- Sample data includes realistic prop lines for major players

### Database Issues
- Delete `nfl_prop_lines.db` and run again to recreate
- Check file permissions in project directory

### Missing Projections
- Ensure `2024_nfl_projections.csv` exists
- Check that your `nfl_data.db` has player stats

## Extending the System

### Adding New Stats
Edit `prop_line_scraper.py` and add to the `markets` list:
```python
markets = [
    'player_pass_yds',
    'player_rush_yds', 
    'player_rec_yds',
    'player_pass_tds',    # Add touchdown props
    'player_rush_tds',
    'player_rec_tds'
]
```

### Adding New Sportsbooks
The Odds API supports many sportsbooks. Real data will include:
- DraftKings, FanDuel, BetMGM, Caesars, etc.
- International books if available

### Custom Thresholds
Edit suspicious line thresholds in `prop_integration.py`:
```python
thresholds = {
    'rushing_yards': {'RB': 800, 'QB': 400},
    'receiving_yards': {'WR': 800, 'TE': 600, 'RB': 300},
    'passing_yards': {'QB': 3500}
}
```

## Sample Output

```
NFL PROP LINE VALUE OPPORTUNITIES
==================================================
Last Updated: 2025-01-11 10:30:00
Total Opportunities: 12

🔥 HIGH VALUE OPPORTUNITIES (3):
  Christian McCaffrey (RB, SF) - rushing_yards
    Line: 1250.5 | Model: 1337.0
    Edge: +86.5 yards (+6.9%)
    Book: DraftKings | Recommendation: OVER

📈 MEDIUM VALUE OPPORTUNITIES (4):
  Tyreek Hill - receiving_yards: 1350.5 (+4.2% edge)
```

This system provides real, actionable prop line data that integrates seamlessly with your existing NFL breakout prediction model.