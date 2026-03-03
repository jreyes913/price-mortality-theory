import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
from signals.pmt import compute_mu_surface
from evaluation.volatility import compute_realized_volatility, get_forward_volatility

def sweep_correlation(price_data, windows, horizons):
    """
    Perform a grid search over window sizes (W) and horizons (h).
    Returns a matrix of Spearman correlations.
    """
    # 1. Compute mu surface (all W)
    # mu_surface[W] is computed using central differences, so it looks ahead by 1 day.
    mu_surface = compute_mu_surface(price_data['Close'], windows)
    
    # 2. Shift mu surface to be causal (mu at time t-1 is known at time t and uses P_t)
    # Actually, mu(W, t) = f(SMA(W, t+1), SMA(W, t-1))
    # If we want mu(t) to use P_t as the latest, we use mu(t-1)
    mu_surface_causal = mu_surface.shift(1)
    
    log_returns = price_data['Log_Return']
    
    results = []
    
    # Grid search
    for h in tqdm(horizons, desc="Sweeping horizons"):
        # Realized volatility over interval of size h
        # vol[t] is std of [r_{t-h+1}, ..., r_t]
        vol = compute_realized_volatility(log_returns, h)
        # Forward vol starting from t+1: vol over [r_{t+1}, ..., r_{t+h}]
        forward_vol = get_forward_volatility(vol, h)
        
        row = {}
        for w in windows:
            if w not in mu_surface_causal.columns:
                continue
                
            # Align mu and forward vol
            feature = mu_surface_causal[w]
            target = forward_vol
            
            # Combine and drop NaNs
            aligned = pd.concat([feature, target], axis=1).dropna()
            
            if len(aligned) < 50: # Arbitrary minimum sample size
                row[w] = np.nan
                continue
                
            rho, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
            row[w] = rho
            
        results.append(pd.Series(row, name=h))
        
    return pd.DataFrame(results)

def find_optimal_params(corr_matrix):
    """
    Find W* and h* that maximize Spearman rho.
    """
    # Stack the matrix to find the max index
    stacked = corr_matrix.stack()
    if stacked.empty:
        return None, None, None
        
    idx_max = stacked.idxmax()
    h_star, w_star = idx_max
    rho_star = stacked[idx_max]
    
    return w_star, h_star, rho_star
