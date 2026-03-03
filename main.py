import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data.loader import load_ticker_data
from evaluation.sweep import sweep_correlation, find_optimal_params

def run_pipeline(ticker, start_date, end_date, w_range, h_range, do_save=True):
    """
    Run full PMT pipeline: data fetch, sweep, results, and plotting.
    """
    if do_save:
        print(f"--- Running PMT Analysis for {ticker} ({start_date} to {end_date}) ---")
    
    # 1. Fetch data
    df = load_ticker_data(ticker, start_date=start_date, end_date=end_date)
    
    # 2. Run sweep
    corr_matrix = sweep_correlation(df, w_range, h_range)
    
    # 3. Find optimal
    w_star, h_star, rho_star = find_optimal_params(corr_matrix)
    
    if do_save:
        print(f"\n--- Optimal Parameters ---")
        print(f"Optimal Window (W*): {w_star} days")
        print(f"Optimal Horizon (h*): {h_star} days")
        print(f"Spearman Rho: {rho_star:.4f}")
        
        # 4. Save results
        os.makedirs("results", exist_ok=True)
        corr_matrix.to_csv(f"results/{ticker}_correlation_matrix.csv")
        
        # 5. Plot Heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, cmap="RdBu_r", center=0, annot=False)
        plt.title(f"PMT Correlation Surface - {ticker}")
        plt.xlabel("Window Size W (days)")
        plt.ylabel("Forward Horizon h (days)")
        plt.savefig(f"results/{ticker}_heatmap.png")
        plt.close() # Close to free memory
    
    return w_star, h_star, rho_star

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Price Mortality Theory (PMT) Pipeline")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument("--start", type=str, default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--w_min", type=int, default=10, help="Min window W (days)")
    parser.add_argument("--w_max", type=int, default=300, help="Max window W (days)")
    parser.add_argument("--h_min", type=int, default=5, help="Min horizon h (days)")
    parser.add_argument("--h_max", type=int, default=60, help="Max horizon h (days)")
    parser.add_argument("--step", type=int, default=5, help="Step size for sweep (days)")

    args = parser.parse_args()
    
    w_range = range(args.w_min, args.w_max + 1, args.step)
    h_range = range(args.h_min, args.h_max + 1, args.step)
    
    run_pipeline(args.ticker, args.start, args.end, w_range, h_range)
