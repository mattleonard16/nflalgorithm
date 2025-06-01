import requests
from bs4 import BeautifulSoup
import time

# Test scraping to debug issues
url = "https://www.pro-football-reference.com/years/2023/rushing.htm"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

print(f"Fetching: {url}")
response = requests.get(url, headers=headers, timeout=10)
print(f"Status Code: {response.status_code}")

if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Look for any table
    all_tables = soup.find_all('table')
    print(f"\nFound {len(all_tables)} tables on the page")
    
    for i, table in enumerate(all_tables):
        table_id = table.get('id', 'no-id')
        print(f"Table {i}: id='{table_id}'")
    
    # Check specifically for the rushing table
    rushing_table = soup.find('table', {'id': 'rushing'})
    if rushing_table:
        print("\nFound rushing table!")
        rows = rushing_table.find_all('tr')
        print(f"Table has {len(rows)} rows")
    else:
        print("\nNo table with id='rushing' found")
        
    # Check for comments (sometimes tables are in comments)
    comments = soup.find_all(string=lambda text: isinstance(text, str) and text.strip().startswith('<!--'))
    print(f"\nFound {len(comments)} HTML comments")
    
    # Save HTML for inspection
    with open('debug_page.html', 'w', encoding='utf-8') as f:
        f.write(response.text)
    print("\nSaved page HTML to debug_page.html for inspection") 