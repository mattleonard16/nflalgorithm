import pandas as pd
import sys
import os

# Check if file exists
file_path = 'nov21.csv'

if not os.path.exists(file_path):
    print(f"File {file_path} not found.")
    sys.exit(1)

try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error reading CSV: {e}")
    sys.exit(1)

print(f"--- Analysis of {file_path} ---")
print(f"Total Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

# 1. Exact Duplicates
exact_dupes = df[df.duplicated()]
print(f"\n1. Exact Duplicates (completely identical rows): {len(exact_dupes)}")
if not exact_dupes.empty:
    print(exact_dupes.head().to_string())

# 2. Key Duplicates (Player + Prop + Sportsbook)
# We expect unique combinations of Player + Prop + Sportsbook (usually)
key_cols = ['player_name', 'prop_type', 'sportsbook']
if set(key_cols).issubset(df.columns):
    key_dupes = df[df.duplicated(subset=key_cols, keep=False)]
    print(f"\n2. Key Duplicates (Same Player, Prop, Sportsbook): {len(key_dupes)}")
    if not key_dupes.empty:
        print("Sample of Key Duplicates:")
        print(key_dupes.sort_values(by=key_cols).head(10)[key_cols + ['line', 'price', 'team_display']].to_string())
else:
    print(f"\nSkipping Key Duplicate check (missing columns from {key_cols})")

# 3. QB Rushing Yards Analysis
print("\n3. QB Rushing Yards Specific Analysis")
if 'position' in df.columns and 'prop_type' in df.columns:
    qb_rush = df[(df['position'] == 'QB') & (df['prop_type'] == 'Rushing Yards')]
    print(f"Total QB Rushing Yards Rows: {len(qb_rush)}")
    
    if not qb_rush.empty:
        # Group by player
        player_counts = qb_rush['player_name'].value_counts()
        print("\nRows per QB (Top 10):")
        print(player_counts.head(10))
        
        # Check for specific examples
        top_player = player_counts.index[0]
        print(f"\nDetail for top QB ({top_player}):")
        print(qb_rush[qb_rush['player_name'] == top_player][['sportsbook', 'line', 'price', 'team_display', 'opponent']].to_string())
else:
    print("Skipping QB Analysis (missing position or prop_type columns)")

# 4. SimBook Analysis
if 'sportsbook' in df.columns:
    simbook = df[df['sportsbook'] == 'SimBook']
    print(f"\n4. SimBook Analysis")
    print(f"Total SimBook rows: {len(simbook)}")
    if not simbook.empty:
        print("Sample SimBook rows:")
        cols_to_show = [c for c in ['player_name', 'team_display', 'opponent', 'prop_type'] if c in df.columns]
        print(simbook[cols_to_show].head(10).to_string())

# 5. Team Display Anomalies (Mismatch Debugging)
print("\n5. Team Display Anomalies (Metadata mismatches)")
if 'team_display' in df.columns:
    anomalies = df[df['team_display'].str.contains(r'\[', na=False)]
    print(f"Rows with bracketed team info (potential mismatches): {len(anomalies)}")
    if not anomalies.empty:
        print(anomalies[['player_name', 'team_display', 'opponent', 'prop_type']].head(10).to_string())
