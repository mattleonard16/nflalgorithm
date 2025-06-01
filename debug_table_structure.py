import requests
from bs4 import BeautifulSoup

url = "https://www.pro-football-reference.com/years/2023/rushing.htm"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

response = requests.get(url, headers=headers, timeout=10)
soup = BeautifulSoup(response.content, 'html.parser')

# Find the rushing table
table = soup.find('table', {'id': 'rushing'})
if table:
    # Get all rows
    all_rows = table.find_all('tr')
    
    print("Analyzing table structure...")
    print(f"Total rows: {len(all_rows)}")
    
    # Look at first few data rows
    data_rows_found = 0
    for i, row in enumerate(all_rows):
        # Check if this is a data row
        player_cell = row.find('td', {'data-stat': 'player'})
        
        if player_cell and data_rows_found < 5:
            data_rows_found += 1
            print(f"\n--- Row {i} (Data Row {data_rows_found}) ---")
            
            # Show all td elements in this row
            all_tds = row.find_all('td')
            print(f"Number of td elements: {len(all_tds)}")
            
            # Show what data-stat attributes are available
            for td in all_tds[:10]:  # First 10 columns
                stat = td.get('data-stat', 'unknown')
                text = td.text.strip()[:20]  # First 20 chars
                print(f"  {stat}: {text}")
            
            # Check player link structure
            player_link = player_cell.find('a')
            if player_link:
                print(f"  Player link href: {player_link.get('href', 'None')}")
    
    # Check tbody specifically
    tbody = table.find('tbody')
    if tbody:
        tbody_rows = tbody.find_all('tr')
        print(f"\nRows in tbody: {len(tbody_rows)}")
    
    # Check for any special row classes
    print("\nChecking row classes...")
    for i, row in enumerate(all_rows[:10]):
        classes = row.get('class', [])
        if classes:
            print(f"Row {i} classes: {classes}") 