import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
from scipy.stats import spearmanr
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.loader import load_ticker_data
from evaluation.volatility import compute_realized_volatility, get_forward_volatility

def garch_analysis(start_date="2019-01-01", end_date="2025-01-01", horizon=21):
    """
    Run AR(1)-GARCH(1,1) for S&P 500 tickers and compute Spearman correlation
    between forecasted volatility and forward realized volatility.
    """
    with open("data/sp500_tickers.txt", "r") as f:
        tickers = [line.strip() for line in f.readlines()]
    
    print(f"Starting GARCH comparison for {len(tickers)} tickers (h={horizon}).")
    
    garch_results = []
    
    for ticker in tqdm(tickers, desc="Analyzing GARCH"):
        try:
            df = load_ticker_data(ticker, start_date=start_date, end_date=end_date)
            returns = df['Log_Return'] * 100 # arch prefers scaled returns
            
            # 1. Fit AR(1)-GARCH(1,1)
            model = arch_model(returns, vol='Garch', p=1, q=1, mean='AR', lags=1)
            res = model.fit(disp='off')
            
            # 2. Get rolling volatility forecasts (conditional volatility)
            # res.conditional_volatility is the sigma_t estimate
            # For a fair comparison with the forward h-day realized volatility,
            # we should look at the correlation of sigma_t with forward vol.
            
            sigma = res.conditional_volatility / 100 # rescale back
            
            # 3. Target: Forward realized volatility
            vol = compute_realized_volatility(df['Log_Return'], horizon)
            forward_vol = get_forward_volatility(vol, horizon)
            
            # Align and compute rho
            aligned = pd.concat([sigma, forward_vol], axis=1).dropna()
            if len(aligned) > 50:
                rho, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                garch_results.append({
                    'ticker': ticker,
                    'rho_garch': rho
                })
            
        except Exception as e:
            # print(f"Error analyzing {ticker}: {e}")
            continue
            
    summary_df = pd.DataFrame(garch_results)
    summary_df.to_csv("results/garch_summary_results.csv", index=False)
    print("GARCH analysis complete. Results saved to results/garch_summary_results.csv")

if __name__ == "__main__":
    # Use h=31 as it was the mean optimal horizon found by PMT
    garch_analysis(horizon=31)
