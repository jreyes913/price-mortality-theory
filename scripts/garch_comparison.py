import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import spearmanr
from tqdm import tqdm
import yfinance as yf

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.loader import load_ticker_data
from main import split_train_test
from evaluation.volatility import compute_realized_volatility, get_forward_volatility


def get_git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def parse_horizons(horizons_arg):
    return [int(x.strip()) for x in horizons_arg.split(",") if x.strip()]


def write_failures(run_id, failures):
    os.makedirs("results", exist_ok=True)
    path = f"results/{run_id}_garch_failures.csv"
    pd.DataFrame(failures).to_csv(path, index=False)
    return path


def fit_garch_sigma(returns_pct):
    model = arch_model(returns_pct, vol="Garch", p=1, q=1, mean="AR", lags=1)
    res = model.fit(disp="off")
    return res.conditional_volatility / 100


def garch_analysis(
    start_date="2019-01-01", end_date="2025-01-01", horizons=None, train_ratio=0.7
):
    """Run AR(1)-GARCH(1,1) and report train/test rho per ticker per horizon."""
    if horizons is None:
        horizons = [10, 21, 31, 42]

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    with open("data/sp500_tickers.txt", "r") as f:
        tickers = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Starting GARCH comparison for {len(tickers)} tickers, horizons={horizons}.")

    garch_results = []
    failures = []
    row_counts = []

    for ticker in tqdm(tickers, desc="Analyzing GARCH"):
        try:
            df = load_ticker_data(ticker, start_date=start_date, end_date=end_date)
            train_df, test_df = split_train_test(df, train_ratio=train_ratio)
            sigma_train = fit_garch_sigma(train_df["Log_Return"] * 100)
            sigma_test = fit_garch_sigma(test_df["Log_Return"] * 100)

            for horizon in horizons:
                vol_train = compute_realized_volatility(train_df["Log_Return"], horizon)
                forward_vol_train = get_forward_volatility(vol_train, horizon)
                aligned_train = pd.concat([sigma_train, forward_vol_train], axis=1).dropna()
                rho_train = np.nan
                if len(aligned_train) > 50:
                    rho_train, _ = spearmanr(aligned_train.iloc[:, 0], aligned_train.iloc[:, 1])

                vol_test = compute_realized_volatility(test_df["Log_Return"], horizon)
                forward_vol_test = get_forward_volatility(vol_test, horizon)
                aligned_test = pd.concat([sigma_test, forward_vol_test], axis=1).dropna()
                rho_test = np.nan
                if len(aligned_test) > 50:
                    rho_test, _ = spearmanr(aligned_test.iloc[:, 0], aligned_test.iloc[:, 1])

                garch_results.append(
                    {
                        "run_id": run_id,
                        "ticker": ticker,
                        "horizon": horizon,
                        "rho_train": rho_train,
                        "rho_test": rho_test,
                    }
                )
            row_counts.append(len(df))
        except Exception as e:
            failures.append(
                {
                    "run_id": run_id,
                    "ticker": ticker,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(limit=1).strip(),
                }
            )
            continue

    summary_df = pd.DataFrame(garch_results)
    os.makedirs("results", exist_ok=True)
    summary_path = "results/garch_summary_results.csv"
    summary_df.to_csv(summary_path, index=False)

    aggregate = (
        summary_df.groupby("horizon")[["rho_train", "rho_test"]]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    aggregate.columns = [
        "horizon",
        "rho_train_count",
        "rho_train_mean",
        "rho_train_median",
        "rho_test_count",
        "rho_test_mean",
        "rho_test_median",
    ]
    agg_path = "results/garch_horizon_aggregate.csv"
    aggregate.to_csv(agg_path, index=False)
    failures_path = write_failures(run_id, failures)

    row_stats = {
        "min_rows": int(min(row_counts)) if row_counts else None,
        "median_rows": float(pd.Series(row_counts).median()) if row_counts else None,
        "max_rows": int(max(row_counts)) if row_counts else None,
    }

    manifest = {
        "run_id": run_id,
        "script": "scripts/garch_comparison.py",
        "git_commit": get_git_commit_hash(),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "start_date": start_date,
        "end_date": end_date,
        "horizons": horizons,
        "train_ratio": train_ratio,
        "ticker_count": len(tickers),
        "tickers_attempted": len(tickers),
        "tickers_succeeded": int(summary_df["ticker"].nunique()) if not summary_df.empty else 0,
        "tickers_failed": len(failures),
        "summary_path": summary_path,
        "aggregate_path": agg_path,
        "failures_path": failures_path,
        "data_provenance": {
            "provider": "yfinance",
            "yfinance_version": getattr(yf, "__version__", "unknown"),
            "field": "Close",
            "return_field": "Log_Return",
            "interval": "1d",
            "adjustment": "yfinance default (auto_adjust unset)",
        },
        "row_count_stats": row_stats,
    }
    manifest_path = f"results/{run_id}_garch_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"GARCH analysis complete. Results saved to {summary_path}")
    print(f"Horizon aggregate saved to {agg_path}")
    print(
        f"Tickers attempted={len(tickers)} succeeded={manifest['tickers_succeeded']} failed={len(failures)}"
    )
    print(f"Run manifest saved to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GARCH comparison over a horizon grid")
    parser.add_argument("--start", default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--horizons",
        default="10,21,31,42",
        help="Comma-separated horizon list, e.g. '10,21,31,42'",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Chronological train split ratio")
    args = parser.parse_args()

    garch_analysis(
        start_date=args.start,
        end_date=args.end,
        horizons=parse_horizons(args.horizons),
        train_ratio=args.train_ratio,
    )
