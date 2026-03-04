# Price Mortality Theory (PMT) Signal Analysis

A quantitative finance framework for volatility forecasting built entirely from price action. This repository implements **Price Mortality Theory (PMT)** by analyzing the momentum surface $\mu(W, t)$, then benchmarking it against AR(1)-GARCH(1,1) and a PMT+GARCH combination model.

## Latest Findings (All Checked-In Runs, S&P 500, 2019--2025)
These findings integrate every current artifact in `results/`:

- **Universe and split:** frozen 503-name S&P 500 snapshot, chronological 70/30 train/test.
- **Run completion:** PMT 497/503, GARCH 499/503, combo 500/503.
- **PMT full-grid (train-selected best params):**
  - Train: mean `rho_train=0.508`, median `0.531`
  - Test: mean `rho_test=-0.018`, median `0.010`
  - Parameter concentration: `W*` median `300`, `h*` median `60`
- **Fixed-horizon PMT vs GARCH (test):**
  - PMT has stronger **medians** at 10/21/31 days.
  - GARCH has stronger **means** at all horizons.
  - At 42 days, PMT turns ~flat/negative while GARCH remains positive.
- **Combo run (PMT + GARCH):**
  - Very strong in-sample fit, but test performance underdelivers.
  - Test means by horizon (10/21/31/42): `0.053`, `0.029`, `-0.012`, `-0.040`.
  - Current combo implementation is **not** a test-time upgrade over either standalone model.

### Out-of-Sample Horizon Aggregates (Spearman `rho_test`)
| Horizon | PMT Mean | PMT Median | GARCH Mean | GARCH Median | Combo Mean | Combo Median |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.074 | 0.087 | 0.087 | 0.051 | 0.053 | 0.059 |
| 21 | 0.069 | 0.063 | 0.079 | 0.033 | 0.029 | 0.036 |
| 31 | 0.033 | 0.048 | 0.066 | 0.016 | -0.012 | -0.012 |
| 42 | -0.003 | -0.002 | 0.059 | 0.010 | -0.040 | -0.039 |

---

## Repository Structure
- `data/`: yfinance loaders and ticker management.
- `signals/`: Core $\mu(W, t)$ surface construction and differentiation logic.
- `evaluation/`: Realized volatility and Spearman rank correlation sweeps.
- `scripts/`: Bulk S&P 500 sweeps and benchmark model comparisons.
- `plotting/`: Visualization scripts for the research paper.
- `report/`: LaTeX source and compiled PDF for the final findings.
- `results/`: CSV summaries, manifests, failure logs, and visualization output.

---

## Getting Started

### 1. Environment Setup
Clone the repository and install required dependencies (Python 3.10+ recommended):
```bash
pip install -r requirements.txt
```

### 2. Run a Single Ticker Analysis
Generate a correlation surface heatmap and optimal $(W, h)$ pair for one asset:
```bash
python main.py --ticker AAPL --start 2020-01-01 --end 2025-01-01
```

### 3. Run the Bulk PMT Sweep (S&P 500)
```bash
python scripts/bulk_sweep.py
# optional: fetch fresh universe instead of frozen file
# python scripts/bulk_sweep.py --use_live_universe
```

### 4. Run the GARCH Comparison
```bash
python scripts/garch_comparison.py
# optional horizon grid
# python scripts/garch_comparison.py --horizons 10,21,31,42
```

### 5. Run the PMT + GARCH Combination Benchmark
```bash
python scripts/combo_comparison.py
# optional horizon grid
# python scripts/combo_comparison.py --horizons 10,21,31,42
```

### 6. Generate Report Plots and Compile LaTeX
```bash
python plotting/generate_report_plots.py
cd report
pdflatex report.tex
```

---

## Methodology
The core signal $\mu(W, t)$ is derived from the Simple Moving Average (SMA) field:
$$\mu(W, t) = -\frac{1}{SMA} \left[ \frac{\partial SMA}{\partial W} + \frac{\partial SMA}{\partial t} \right]$$

This captures both temporal momentum and structural sensitivity to look-back depth, giving a scale-invariant measure of trend ``mortality.''

---

## Reproducibility & Evaluation Notes
- **Frozen universe by default:** `scripts/bulk_sweep.py` uses `data/sp500_tickers.txt` unless `--use_live_universe` is passed.
- **Train/test separation:** PMT, GARCH, and combo outputs are all reported with explicit train and test metrics.
- **Run manifests:** JSON manifests in `results/` include run IDs, parameters, data provenance, and completion counts.
- **Failure accounting:** each run emits failure CSVs for skipped tickers and exception reasons.
- **Shared benchmark horizons:** PMT/GARCH/combo comparisons use `10,21,31,42` by default.

## Contact
**Jose Reyes**  
jstunner55@gmail.com  
March 3, 2026

*Note: Data is fetched via yfinance during fresh runs. Checked-in artifacts are reproducible snapshots of prior executions.*
