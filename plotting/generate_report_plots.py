import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.stats import spearmanr
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from signals.pmt import compute_mu_surface


def set_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10,
        }
    )


def generate_bulk_distribution():
    print("Generating bulk distribution plot...")
    if not os.path.exists("results/sp500_summary_results.csv"):
        print("Summary results not found. Skipping bulk plot.")
        return

    df = pd.read_csv("results/sp500_summary_results.csv")
    plt.figure(figsize=(6, 4))
    rho_col = "rho_star" if "rho_star" in df.columns else "rho_train"
    sns.histplot(df[rho_col], kde=True, color="#2563eb", alpha=0.6)
    median_rho = df[rho_col].median()
    plt.axvline(median_rho, color="#d97706", linestyle="--", label=f"Median: {median_rho:.3f}")
    plt.xlabel(r"Spearman $\rho$")
    plt.ylabel("Frequency")
    plt.title("Distribution of Max Correlation (S&P 500)")
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

    close = data["Close"]
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
    ax = fig.add_subplot(111, projection="3d")

    W_ds = list(windows)[::2]
    t_ds = mu_surface_df.index[::5]
    mu_vals = mu_surface_df.loc[t_ds, W_ds].values

    W_grid, T_grid = np.meshgrid(W_ds, range(len(t_ds)))
    mu_vals = np.clip(mu_vals, np.nanpercentile(mu_vals, 2), np.nanpercentile(mu_vals, 98))

    ax.plot_surface(W_grid, T_grid, mu_vals, cmap="RdYlGn", alpha=0.8, linewidth=0)
    ax.set_xlabel("Window W")
    ax.set_ylabel("Time (sampled)")
    ax.set_zlabel(r"$\mu$")
    plt.title(f"Momentum Surface $\mu(W, t)$ - {ticker}")
    plt.tight_layout()
    plt.savefig("report/images/spy_mu_surface.png", dpi=300)
    plt.close()


def generate_model_comparison_plots():
    print("Generating model comparison plots...")
    pmt_path = "results/pmt_horizon_aggregate.csv"
    garch_path = "results/garch_horizon_aggregate.csv"
    combo_path = "results/combo_horizon_aggregate.csv"

    if not (os.path.exists(pmt_path) and os.path.exists(garch_path)):
        print("PMT/GARCH aggregate files missing. Skipping model comparison plots.")
        return

    pmt = pd.read_csv(pmt_path)
    garch = pd.read_csv(garch_path)
    merged = pmt.merge(garch, on="horizon", suffixes=("_pmt", "_garch"))

    if os.path.exists(combo_path):
        combo = pd.read_csv(combo_path)
        merged = merged.merge(combo, on="horizon", how="left")

    # Plot 1: test medians by horizon
    plt.figure(figsize=(8, 5))
    plt.plot(merged["horizon"], merged["rho_test_median_pmt"], marker="o", label="PMT")
    plt.plot(merged["horizon"], merged["rho_test_median_garch"], marker="o", label="GARCH")
    if "rho_combo_test_median" in merged.columns:
        plt.plot(merged["horizon"], merged["rho_combo_test_median"], marker="o", label="Combo")
    plt.axhline(0, color="gray", lw=1, linestyle="--")
    plt.xlabel("Horizon (days)")
    plt.ylabel("Median test Spearman rho")
    plt.title("Out-of-sample Median Rank Correlation by Horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig("report/images/horizon_median_comparison.png", dpi=300)
    plt.close()

    # Plot 2: test means by horizon
    plt.figure(figsize=(8, 5))
    plt.plot(merged["horizon"], merged["rho_test_mean_pmt"], marker="o", label="PMT")
    plt.plot(merged["horizon"], merged["rho_test_mean_garch"], marker="o", label="GARCH")
    if "rho_combo_test_mean" in merged.columns:
        plt.plot(merged["horizon"], merged["rho_combo_test_mean"], marker="o", label="Combo")
    plt.axhline(0, color="gray", lw=1, linestyle="--")
    plt.xlabel("Horizon (days)")
    plt.ylabel("Mean test Spearman rho")
    plt.title("Out-of-sample Mean Rank Correlation by Horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig("report/images/horizon_mean_comparison.png", dpi=300)
    plt.close()

    # Plot 3: PMT relative to GARCH
    diff = pd.DataFrame(
        {
            "horizon": merged["horizon"],
            "median_diff": merged["rho_test_median_pmt"] - merged["rho_test_median_garch"],
            "mean_diff": merged["rho_test_mean_pmt"] - merged["rho_test_mean_garch"],
        }
    )
    plt.figure(figsize=(8, 5))
    width = 3
    plt.bar(diff["horizon"] - width / 2, diff["median_diff"], width=width, label="PMT-GARCH median")
    plt.bar(diff["horizon"] + width / 2, diff["mean_diff"], width=width, label="PMT-GARCH mean")
    plt.axhline(0, color="gray", lw=1, linestyle="--")
    plt.xlabel("Horizon (days)")
    plt.ylabel("Difference in test Spearman rho")
    plt.title("PMT Relative Lift vs GARCH")
    plt.legend()
    plt.tight_layout()
    plt.savefig("report/images/horizon_pmt_vs_garch_delta.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    set_style()
    os.makedirs("report/images", exist_ok=True)
    generate_bulk_distribution()
    generate_spy_plots()
    generate_model_comparison_plots()
    print("Plots updated in report/images/")
