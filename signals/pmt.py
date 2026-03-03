import pandas as pd
import numpy as np

def compute_sma(price_series, window):
    """
    Compute Simple Moving Average.
    """
    return price_series.rolling(window=window).mean()

def compute_mu_surface(price_series, window_range):
    """
    Compute the mu surface for a range of window sizes.
    mu(W, t) = -(1/SMA) * [dSMA/dW + dSMA/dt]
    
    Args:
        price_series (pd.Series): Time series of prices.
        window_range (list or np.array): Range of window sizes (W) to compute for.
        
    Returns:
        pd.DataFrame: DataFrame where columns are window sizes and rows are time,
                     containing mu values.
    """
    mu_data = {}
    
    # Sort window range to ensure we can take derivatives across W
    window_range = sorted(window_range)
    
    # Pre-compute SMAs for required windows
    # We need W-1 and W+1 for central difference in W
    extended_windows = set()
    for w in window_range:
        extended_windows.add(w)
        extended_windows.add(w - 1)
        extended_windows.add(w + 1)
    
    smas = {w: compute_sma(price_series, w) for w in extended_windows if w > 0}
    
    for w in window_range:
        if w - 1 not in smas or w + 1 not in smas:
            continue
            
        sma_w = smas[w]
        sma_w_plus = smas[w + 1]
        sma_w_minus = smas[w - 1]
        
        # dSMA/dW central difference: (SMA(W+1, t) - SMA(W-1, t)) / 2
        d_sma_dw = (sma_w_plus - sma_w_minus) / 2.0
        
        # dSMA/dt central difference: (SMA(W, t+1) - SMA(W, t-1)) / 2
        # Using shift to get t+1 and t-1
        d_sma_dt = (sma_w.shift(-1) - sma_w.shift(1)) / 2.0
        
        # mu(W, t) = -(1/SMA) * [dSMA/dW + dSMA/dt]
        mu = -(1.0 / sma_w) * (d_sma_dw + d_sma_dt)
        mu_data[w] = mu
        
    return pd.DataFrame(mu_data)

if __name__ == "__main__":
    # Test PMT computation
    dates = pd.date_range("2020-01-01", periods=100)
    prices = pd.Series(np.cumsum(np.random.randn(100)) + 100, index=dates)
    windows = range(10, 20)
    mu_surface = compute_mu_surface(prices, windows)
    print(mu_surface.tail())
