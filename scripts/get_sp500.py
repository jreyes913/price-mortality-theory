import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

def fetch_sp500_tickers():
    """
    Fetch S&P 500 ticker symbols from Wikipedia.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch Wikipedia page. Status code: {response.status_code}")
        
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    
    if table is None:
        # Fallback: try finding by class if ID failed
        table = soup.find('table', {'class': 'wikitable'})
        
    if table is None:
        raise Exception("Could not find S&P 500 constituents table on Wikipedia.")
    
    tickers = []
    for row in table.findAll('tr')[1:]:
        cells = row.findAll('td')
        if not cells:
            continue
        ticker = cells[0].text.strip()
        # Some tickers use '.' instead of '-' (e.g. BRK.B vs BRK-B)
        # yfinance prefers '-'
        ticker = ticker.replace('.', '-')
        tickers.append(ticker)
        
    return sorted(tickers)

if __name__ == "__main__":
    tickers = fetch_sp500_tickers()
    print(f"Fetched {len(tickers)} tickers.")
    print(tickers[:10])
    # Save to a text file for reference
    os.makedirs("data", exist_ok=True)
    with open("data/sp500_tickers.txt", "w") as f:
        for ticker in tickers:
            f.write(f"{ticker}\n")
