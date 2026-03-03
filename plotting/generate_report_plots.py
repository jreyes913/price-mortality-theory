import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.stats import spearmanr
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from signals.pmt import compute_mu_surface

def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10
    })

def generate_bulk_distribution():
    print("Generating bulk distribution plot...")
    if not os.path.exists("results/sp500_summary_results.csv"):
        print("Summary results not found. Skipping bulk plot.")
        return

    df = pd.read_csv("results/sp500_summary_results.csv")
    plt.figure(figsize=(6, 4))
    sns.histplot(df['rho_star'], kde=True, color='#2563eb', alpha=0.6)
    median_rho = df['rho_star'].median()
    plt.axvline(median_rho, color='#d97706', linestyle='--', label=f'Median: {median_rho:.3f}')
    plt.xlabel(r'Spearman $\rho$')
    plt.ylabel('Frequency')
    plt.title('Distribution of Max Correlation (S&P 500)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("report/images/rho_distribution.png", dpi=300)
    plt.close()

def generate_spy_plots():
    print("Generating SPY plots (Heatmap and 3D Surface)...")
    ticker = "SPY"
    data = yf.download(ticker, start="2019-01-01", end="2025-01-01", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    close = data['Close']
    log_returns = np.log(close / close.shift(1))
    
    windows = range(20, 301, 10)
    horizons = range(5, 65, 5)
    
    mu_surface_df = compute_mu_surface(close, windows).shift(1)
    
    # 1. Heatmap
    results = []
    for h in horizons:
        vol = log_returns.rolling(window=h).std()
        forward_vol = vol.shift(-h)
        row = {}
        for w in windows:
            aligned = pd.concat([mu_surface_df[w], forward_vol], axis=1).dropna()
            if len(aligned) > 50:
                rho, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                row[w] = rho
        results.append(pd.Series(row, name=h))
    
    corr_matrix = pd.DataFrame(results)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, cmap="RdYlGn", center=0)
    plt.xlabel("Window Size W (days)")
    plt.ylabel("Forward Horizon h (days)")
    plt.title(f"PMT Correlation Surface - {ticker}")
    plt.tight_layout()
    plt.savefig("report/images/spy_heatmap.png", dpi=300)
    plt.close()

    # 2. 3D Surface Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Downsample for better visibility
    W_ds = list(windows)[::2]
    t_ds = mu_surface_df.index[::5]
    mu_vals = mu_surface_df.loc[t_ds, W_ds].values
    
    W_grid, T_grid = np.meshgrid(W_ds, range(len(t_ds)))
    
    # Clip outliers for better scale
    mu_vals = np.clip(mu_vals, np.nanpercentile(mu_vals, 2), np.nanpercentile(mu_vals, 98))
    
    surf = ax.plot_surface(W_grid, T_grid, mu_vals, cmap='RdYlGn', alpha=0.8, linewidth=0)
    ax.set_xlabel('Window W')
    ax.set_ylabel('Time (sampled)')
    ax.set_zlabel(r'$\mu$')
    plt.title(f'Momentum Surface $\mu(W, t)$ - {ticker}')
    plt.tight_layout()
    plt.savefig("report/images/spy_mu_surface.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    set_style()
    os.makedirs("report/images", exist_ok=True)
    generate_bulk_distribution()
    generate_spy_plots()
    print("Plots updated in report/images/")
