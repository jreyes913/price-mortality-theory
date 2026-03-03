# Price Mortality Theory (PMT) Signal Analysis

A quantitative finance framework for volatility forecasting built entirely from price action. This repository implements the **Price Mortality Theory (PMT)**, identifying structural lead indicators for market volatility by analyzing the momentum surface $\mu(W, t)$.

## Findings (S&P 500, 2019--2025)
The PMT signal demonstrates a robust predictive lead over traditional realized volatility measures:
- **Median Spearman $\rho$**: 0.531 (Robust across 500+ tickers).
- **Mean Optimal Window ($W^*$)**: 242 days (Long-term structural look-back).
- **Mean Optimal Horizon ($h^*$)**: 43 days (Mid-term volatility lead).
- **Benchmark Comparison**: Outperforms AR(1)-GARCH(1,1) (Mean $\rho$: 0.531 vs 0.363).

---

## Repository Structure
- `data/`: yfinance loaders and ticker management.
- `signals/`: Core $\mu(W, t)$ surface construction and differentiation logic.
- `evaluation/`: Realized volatility and Spearman rank correlation sweeps.
- `scripts/`: Bulk S&P 500 sweeps and GARCH comparison models.
- `plotting/`: Visualization scripts for the research paper.
- `report/`: LaTeX source and compiled PDF for the final findings.
- `results/`: CSV summaries and visualization output.

---

## Getting Started

### 1. Environment Setup
Clone the repository and install the required dependencies (Python 3.10+ recommended):
```bash
pip install -r requirements.txt
```

### 2. Run a Single Ticker Analysis
To generate a correlation surface heatmap and optimal $(W, h)$ pair for a specific asset:
```bash
python main.py --ticker AAPL --start 2020-01-01 --end 2025-01-01
```

### 3. Run the Bulk S&P 500 Sweep
Execute the full PMT analysis across all S&P 500 constituents. This generates the aggregate statistical findings used in the report:
```bash
python scripts/bulk_sweep.py
# optional: fetch fresh universe instead of frozen file
# python scripts/bulk_sweep.py --use_live_universe
```

### 4. Run the GARCH Comparison
To validate the PMT signal against the industry-standard AR(1)-GARCH(1,1) model:
```bash
python scripts/garch_comparison.py
# optional horizon grid
# python scripts/garch_comparison.py --horizons 10,21,31,42
```

### 5. Generate the Research Report
To regenerate the 3D surface plots and distribution figures, then compile the LaTeX report:
```bash
python plotting/generate_report_plots.py
cd report
pdflatex report.tex
```

---

## Methodology
The core signal $\mu(W, t)$ is derived from the Simple Moving Average (SMA) field:
$$\mu(W, t) = -\frac{1}{SMA} \left[ \frac{\partial SMA}{\partial W} + \frac{\partial SMA}{\partial t} \right]$$

This captures both temporal momentum and structural sensitivity to the look-back window, providing a scale-invariant measure of trend "mortality." High mortality precedes regime shifts and spikes in realized volatility.

---


## Reproducibility & Evaluation Notes
- **Frozen universe by default**: `scripts/bulk_sweep.py` now uses `data/sp500_tickers.txt` unless `--use_live_universe` is set.
- **Train/test PMT reporting**: bulk sweep selects `(W*, h*)` on train data and reports both `rho_train` and `rho_test`.
- **Run manifests**: bulk and GARCH scripts now emit JSON manifests in `results/` with run parameters and commit hash.
- **GARCH horizon grid**: GARCH results are now emitted for a shared horizon grid (`10,21,31,42` by default).

## Contact
**Jose Reyes**  
jstunner55@gmail.com  
March 3, 2026

*Note: All data is fetched dynamically via the yfinance API. Ensure internet connectivity during execution.*
