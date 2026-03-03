import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from data.loader import load_ticker_data
from evaluation.sweep import sweep_correlation, find_optimal_params


def split_train_test(df, train_ratio=0.7):
    """Chronological split preserving temporal order."""
    split_idx = int(len(df) * train_ratio)
    split_idx = max(1, min(split_idx, len(df) - 1))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def run_pipeline(
    ticker,
    start_date,
    end_date,
    w_range,
    h_range,
    do_save=True,
    use_train_test_split=False,
    train_ratio=0.7,
):
    """
    Run PMT pipeline: data fetch, sweep, and optional plotting/saving.

    Returns:
        dict: Contains W*/h* and rho metrics.
    """
    if do_save:
        print(f"--- Running PMT Analysis for {ticker} ({start_date} to {end_date}) ---")

    df = load_ticker_data(ticker, start_date=start_date, end_date=end_date)

    if use_train_test_split:
        train_df, test_df = split_train_test(df, train_ratio=train_ratio)
        corr_matrix_train = sweep_correlation(train_df, w_range, h_range)
        w_star, h_star, rho_train = find_optimal_params(corr_matrix_train)

        rho_test = None
        if w_star is not None and h_star is not None:
            corr_matrix_test = sweep_correlation(test_df, [w_star], [h_star])
            try:
                rho_test = float(corr_matrix_test.loc[h_star, w_star])
            except Exception:
                rho_test = None

        if do_save:
            print("\n--- Optimal Parameters (Train Split) ---")
            print(f"Optimal Window (W*): {w_star} days")
            print(f"Optimal Horizon (h*): {h_star} days")
            print(f"Spearman Rho (train): {rho_train:.4f}" if rho_train is not None else "Spearman Rho (train): None")
            print(f"Spearman Rho (test): {rho_test:.4f}" if rho_test is not None else "Spearman Rho (test): None")

            os.makedirs("results", exist_ok=True)
            corr_matrix_train.to_csv(f"results/{ticker}_correlation_matrix_train.csv")

            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_matrix_train, cmap="RdBu_r", center=0, annot=False)
            plt.title(f"PMT Correlation Surface (Train) - {ticker}")
            plt.xlabel("Window Size W (days)")
            plt.ylabel("Forward Horizon h (days)")
            plt.savefig(f"results/{ticker}_heatmap_train.png")
            plt.close()

        return {
            "W_star": w_star,
            "h_star": h_star,
            "rho_train": rho_train,
            "rho_test": rho_test,
        }

    corr_matrix = sweep_correlation(df, w_range, h_range)
    w_star, h_star, rho_star = find_optimal_params(corr_matrix)

    if do_save:
        print("\n--- Optimal Parameters ---")
        print(f"Optimal Window (W*): {w_star} days")
        print(f"Optimal Horizon (h*): {h_star} days")
        print(f"Spearman Rho: {rho_star:.4f}")

        os.makedirs("results", exist_ok=True)
        corr_matrix.to_csv(f"results/{ticker}_correlation_matrix.csv")

        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, cmap="RdBu_r", center=0, annot=False)
        plt.title(f"PMT Correlation Surface - {ticker}")
        plt.xlabel("Window Size W (days)")
        plt.ylabel("Forward Horizon h (days)")
        plt.savefig(f"results/{ticker}_heatmap.png")
        plt.close()

    return {"W_star": w_star, "h_star": h_star, "rho_star": rho_star}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Price Mortality Theory (PMT) Pipeline")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument("--start", type=str, default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--w_min", type=int, default=10, help="Min window W (days)")
    parser.add_argument("--w_max", type=int, default=300, help="Max window W (days)")
    parser.add_argument("--h_min", type=int, default=5, help="Min horizon h (days)")
    parser.add_argument("--h_max", type=int, default=60, help="Max horizon h (days)")
    parser.add_argument("--step", type=int, default=5, help="Step size for sweep (days)")
    parser.add_argument("--train_test_split", action="store_true", help="Enable 70/30 chronological train/test evaluation")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train ratio when --train_test_split is enabled")

    args = parser.parse_args()

    w_range = range(args.w_min, args.w_max + 1, args.step)
    h_range = range(args.h_min, args.h_max + 1, args.step)

    run_pipeline(
        args.ticker,
        args.start,
        args.end,
        w_range,
        h_range,
        use_train_test_split=args.train_test_split,
        train_ratio=args.train_ratio,
    )
