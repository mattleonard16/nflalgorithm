import requests
from bs4 import BeautifulSoup, Comment

url = "https://www.pro-football-reference.com/years/2023/rushing.htm"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

response = requests.get(url, headers=headers, timeout=10)
soup = BeautifulSoup(response.content, 'html.parser')

# Find all comments
comments = soup.find_all(string=lambda text: isinstance(text, Comment))
print(f"Found {len(comments)} comments")

# Check if any comments contain table data
for i, comment in enumerate(comments):
    if 'table' in comment and 'rushing' in comment:
        print(f"\nComment {i} contains a rushing table!")
        
        # Parse the comment as HTML
        comment_soup = BeautifulSoup(comment, 'html.parser')
        comment_table = comment_soup.find('table')
        
        if comment_table:
            print("Successfully parsed table from comment")
            rows = comment_table.find_all('tr')
            print(f"Table has {len(rows)} rows")
            
            # Try to find a data row
            for row in rows[:10]:
                player_cell = row.find('td', {'data-stat': 'player'})
                if player_cell:
                    print(f"Found player: {player_cell.text.strip()}")
                    break

# Also check the main table directly with tbody
print("\nChecking main table tbody...")
table = soup.find('table', {'id': 'rushing'})
if table:
    tbody = table.find('tbody')
    if tbody:
        first_data_row = tbody.find('tr')
        if first_data_row:
            print("First row in tbody:")
            tds = first_data_row.find_all('td')
            for td in tds[:5]:
                print(f"  {td.get('data-stat', 'unknown')}: {td.text.strip()[:20]}") 