import pandas as pd
import numpy as np

def compute_realized_volatility(log_returns, horizon):
    """
    Compute realized volatility over a rolling horizon.
    Realized volatility is the standard deviation of log returns.
    
    Args:
        log_returns (pd.Series): Time series of log returns.
        horizon (int): Number of days for the horizon (h).
        
    Returns:
        pd.Series: Series of realized volatility values.
    """
    # Annualize if desired, but for rank correlation it doesn't matter.
    # We'll just return the raw volatility over the horizon.
    return log_returns.rolling(window=horizon).std()

def get_forward_volatility(realized_vol, horizon):
    """
    Shift realized volatility to get forward-looking volatility.
    If realized_vol[t] is the vol over [t-h+1, t], 
    then forward_vol[t] should be vol over [t+1, t+h].
    
    Args:
        realized_vol (pd.Series): Realized volatility series.
        horizon (int): The horizon h.
        
    Returns:
        pd.Series: Forward-shifted volatility.
    """
    # realized_vol at time t+h is the vol from t+1 to t+h
    return realized_vol.shift(-horizon)
