import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from tqdm import tqdm
from scripts.get_sp500 import fetch_sp500_tickers
from main import run_pipeline

def bulk_analysis(start_date="2019-01-01", end_date="2025-01-01"):
    """
    Fetch S&P 500 tickers and run PMT analysis for each.
    """
    tickers = fetch_sp500_tickers()
    os.makedirs("data", exist_ok=True)
    with open("data/sp500_tickers.txt", "w") as f:
        for t in tickers:
            f.write(f"{t}\n")
    print(f"Starting bulk analysis for {len(tickers)} S&P 500 tickers.")
    
    # Ranges for analysis
    w_range = range(20, 301, 20)
    h_range = range(5, 61, 5)
    
    summary_results = []
    
    for ticker in tqdm(tickers, desc="Analyzing S&P 500"):
        try:
            w_star, h_star, rho_star = run_pipeline(ticker, start_date, end_date, w_range, h_range, do_save=False)
            summary_results.append({
                'ticker': ticker,
                'W_star': w_star,
                'h_star': h_star,
                'rho_star': rho_star
            })
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            continue
            
    # Save a summary of all ticker results
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv("results/sp500_summary_results.csv", index=False)
    print("Bulk analysis complete. Summary saved to results/sp500_summary_results.csv")

if __name__ == "__main__":
    # Ensure stdout is flushed properly for background execution
    sys.stdout.reconfigure(line_buffering=True)
    bulk_analysis()
