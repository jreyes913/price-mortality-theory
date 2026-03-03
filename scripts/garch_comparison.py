import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd
from arch import arch_model
from scipy.stats import spearmanr
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.loader import load_ticker_data
from evaluation.volatility import compute_realized_volatility, get_forward_volatility


def get_git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def parse_horizons(horizons_arg):
    return [int(x.strip()) for x in horizons_arg.split(",") if x.strip()]


def garch_analysis(start_date="2019-01-01", end_date="2025-01-01", horizons=None):
    """Run AR(1)-GARCH(1,1) and report rho per ticker per horizon."""
    if horizons is None:
        horizons = [10, 21, 31, 42]

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    with open("data/sp500_tickers.txt", "r") as f:
        tickers = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Starting GARCH comparison for {len(tickers)} tickers, horizons={horizons}.")

    garch_results = []

    for ticker in tqdm(tickers, desc="Analyzing GARCH"):
        try:
            df = load_ticker_data(ticker, start_date=start_date, end_date=end_date)
            returns = df["Log_Return"] * 100

            model = arch_model(returns, vol="Garch", p=1, q=1, mean="AR", lags=1)
            res = model.fit(disp="off")
            sigma = res.conditional_volatility / 100

            for horizon in horizons:
                vol = compute_realized_volatility(df["Log_Return"], horizon)
                forward_vol = get_forward_volatility(vol, horizon)

                aligned = pd.concat([sigma, forward_vol], axis=1).dropna()
                if len(aligned) > 50:
                    rho, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                    garch_results.append(
                        {
                            "run_id": run_id,
                            "ticker": ticker,
                            "horizon": horizon,
                            "rho_garch": rho,
                        }
                    )
        except Exception:
            continue

    summary_df = pd.DataFrame(garch_results)
    os.makedirs("results", exist_ok=True)
    summary_path = "results/garch_summary_results.csv"
    summary_df.to_csv(summary_path, index=False)

    aggregate = (
        summary_df.groupby("horizon")["rho_garch"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .rename(columns={"count": "n_tickers"})
    )
    agg_path = "results/garch_horizon_aggregate.csv"
    aggregate.to_csv(agg_path, index=False)

    manifest = {
        "run_id": run_id,
        "script": "scripts/garch_comparison.py",
        "git_commit": get_git_commit_hash(),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "start_date": start_date,
        "end_date": end_date,
        "horizons": horizons,
        "ticker_count": len(tickers),
        "summary_path": summary_path,
        "aggregate_path": agg_path,
    }
    manifest_path = f"results/{run_id}_garch_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"GARCH analysis complete. Results saved to {summary_path}")
    print(f"Horizon aggregate saved to {agg_path}")
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
    args = parser.parse_args()

    garch_analysis(
        start_date=args.start,
        end_date=args.end,
        horizons=parse_horizons(args.horizons),
    )
