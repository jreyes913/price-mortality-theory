import yfinance as yf
import pandas as pd
import numpy as np
import os

def load_ticker_data(ticker, start_date=None, end_date=None, interval='1d'):
    """
    Load historical price data for a given ticker.
    
    Args:
        ticker (str): Ticker symbol (e.g., 'AAPL').
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        interval (str): Data interval (default '1d').
        
    Returns:
        pd.DataFrame: DataFrame with Close prices and log returns.
    """
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    
    # Ensure MultiIndex columns from yfinance are handled if necessary
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df[['Close']].copy()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    return df.dropna()

if __name__ == "__main__":
    # Test loader
    data = load_ticker_data("AAPL", start_date="2020-01-01", end_date="2025-01-01")
    print(data.head())
