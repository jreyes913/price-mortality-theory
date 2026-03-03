import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.loader import load_ticker_data
from evaluation.sweep import find_optimal_params, sweep_correlation
from main import run_pipeline, split_train_test
from scripts.get_sp500 import fetch_sp500_tickers


PMT_COMPARISON_HORIZONS = [10, 21, 31, 42]


def get_git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_tickers(use_frozen_universe=True, frozen_path="data/sp500_tickers.txt"):
    os.makedirs("data", exist_ok=True)
    if use_frozen_universe and os.path.exists(frozen_path):
        with open(frozen_path, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
        source = "frozen"
        return tickers, source

    tickers = fetch_sp500_tickers()
    with open(frozen_path, "w") as f:
        for t in tickers:
            f.write(f"{t}\n")
    source = "live"
    return tickers, source


def write_manifest(run_id, manifest):
    os.makedirs("results", exist_ok=True)
    manifest_path = f"results/{run_id}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def bulk_analysis(
    start_date="2019-01-01",
    end_date="2025-01-01",
    use_frozen_universe=True,
    train_ratio=0.7,
):
    """Run PMT analysis for S&P 500 tickers with reproducibility metadata."""
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    tickers, ticker_source = load_tickers(use_frozen_universe=use_frozen_universe)
    print(f"Starting bulk analysis for {len(tickers)} tickers (source={ticker_source}).")

    os.makedirs("results", exist_ok=True)
    with open(f"results/{run_id}_sp500_universe_used.txt", "w") as f:
        for ticker in tickers:
            f.write(f"{ticker}\n")

    w_range = list(range(20, 301, 20))
    h_range = list(range(5, 61, 5))

    summary_results = []
    pmt_horizon_results = []

    for ticker in tqdm(tickers, desc="Analyzing S&P 500"):
        try:
            result = run_pipeline(
                ticker,
                start_date,
                end_date,
                w_range,
                h_range,
                do_save=False,
                use_train_test_split=True,
                train_ratio=train_ratio,
            )
            summary_results.append(
                {
                    "run_id": run_id,
                    "ticker": ticker,
                    "W_star": result.get("W_star"),
                    "h_star": result.get("h_star"),
                    "rho_train": result.get("rho_train"),
                    "rho_test": result.get("rho_test"),
                }
            )

            # Side-by-side comparable PMT results at fixed horizons.
            df = load_ticker_data(ticker, start_date=start_date, end_date=end_date)
            train_df, test_df = split_train_test(df, train_ratio=train_ratio)
            for horizon in PMT_COMPARISON_HORIZONS:
                corr_train_h = sweep_correlation(train_df, w_range, [horizon])
                w_star_h, _, rho_train_h = find_optimal_params(corr_train_h)
                rho_test_h = None
                if w_star_h is not None:
                    corr_test_h = sweep_correlation(test_df, [w_star_h], [horizon])
                    try:
                        rho_test_h = float(corr_test_h.loc[horizon, w_star_h])
                    except Exception:
                        rho_test_h = None

                pmt_horizon_results.append(
                    {
                        "run_id": run_id,
                        "ticker": ticker,
                        "horizon": horizon,
                        "W_star": w_star_h,
                        "rho_train": rho_train_h,
                        "rho_test": rho_test_h,
                    }
                )
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            continue

    summary_df = pd.DataFrame(summary_results)
    summary_path = "results/sp500_summary_results.csv"
    summary_df.to_csv(summary_path, index=False)

    pmt_horizon_df = pd.DataFrame(pmt_horizon_results)
    pmt_horizon_path = "results/pmt_horizon_summary_results.csv"
    pmt_horizon_df.to_csv(pmt_horizon_path, index=False)

    pmt_agg = (
        pmt_horizon_df.groupby("horizon")[["rho_train", "rho_test"]]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    pmt_agg.columns = [
        "horizon",
        "rho_train_count",
        "rho_train_mean",
        "rho_train_median",
        "rho_test_count",
        "rho_test_mean",
        "rho_test_median",
    ]
    pmt_agg_path = "results/pmt_horizon_aggregate.csv"
    pmt_agg.to_csv(pmt_agg_path, index=False)

    manifest = {
        "run_id": run_id,
        "script": "scripts/bulk_sweep.py",
        "git_commit": get_git_commit_hash(),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "start_date": start_date,
        "end_date": end_date,
        "windows": w_range,
        "horizons": h_range,
        "comparison_horizons": PMT_COMPARISON_HORIZONS,
        "train_ratio": train_ratio,
        "ticker_count": len(tickers),
        "ticker_source": ticker_source,
        "summary_path": summary_path,
        "pmt_horizon_summary_path": pmt_horizon_path,
        "pmt_horizon_aggregate_path": pmt_agg_path,
    }
    manifest_path = write_manifest(run_id, manifest)
    print(f"Bulk analysis complete. Summary saved to {summary_path}")
    print(f"PMT horizon summary saved to {pmt_horizon_path}")
    print(f"PMT horizon aggregate saved to {pmt_agg_path}")
    print(f"Run manifest saved to {manifest_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Bulk PMT S&P 500 analysis")
    parser.add_argument("--start", default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--use_live_universe",
        action="store_true",
        help="Fetch current S&P 500 tickers from Wikipedia instead of using data/sp500_tickers.txt",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Chronological train split ratio")
    args = parser.parse_args()

    bulk_analysis(
        start_date=args.start,
        end_date=args.end,
        use_frozen_universe=not args.use_live_universe,
        train_ratio=args.train_ratio,
    )
