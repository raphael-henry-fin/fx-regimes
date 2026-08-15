# fx-regimes
USD/EUR Exchange Rate Analysis Using FRED Data: Comparison of Risk Across Two Market Regimes, Regression Against the S&amp;P 500, and Automated Generation of an Excel Workbook and HTML Report.

Isolated two opposite market regimes in the USD/EUR exchange rate and showed that volatility fails to separate them — Sharpe ratios of −2.18 and +0.20 sit on virtually identical volatility (6.59% vs 6.20%), across 1,840 trading sessions of FRED data analysed in Python.

## Key results

- **Volatility does not discriminate:** 6.59% (bearish) vs 6.20% (bullish) — a 0.39 pt gap for returns of −12.36% and +3.27%
- **Regime asymmetry:** Sharpe −2.1789 vs +0.2044, max drawdown −21.39% vs −8.72%
- **Weak equity linkage:** the S&P 500 explains 3.2% of the pair's variation (R² = 0.0315), beta = 0.0788 significant at p < 0.001
- **Non-normal residuals:** Shapiro-Wilk W = 0.9010, p < 0.001 — fat tails invalidate the regression's confidence intervals
- **Full sample:** −0.42% annualised return, 6.21% volatility, −21.44% max drawdown over 2021-01-11 → 2026-01-28

## Repository structure

- `financial_analysis.py` — full pipeline: loading, metrics, regression, charts, reports
- `financial_report.html` — self-contained report, charts embedded as base64
- `financial_report.xlsx` — four-sheet workbook (summary, regimes, raw data, regression)
- `period_justification.txt` — rationale for the two regime boundaries
- `benchmark_justification.txt` — rationale for the S&P 500 benchmark
- `ai_critique.txt` — critique of AI-generated Sharpe ratio code
- `market_data_DEXUSEU.json` — daily DEXUSEU series
- `market_data_context_DEXUSEU.csv` — context series including SP500 and NASDAQCOM

## Data sources

Both source files are included, so the script runs as is. They come from:

- [U.S. Dollars to Euro Spot Exchange Rate (DEXUSEU) — FRED](https://fred.stlouisfed.org/series/DEXUSEU)
- [S&P 500 (SP500) — FRED](https://fred.stlouisfed.org/series/SP500)
- [NASDAQ Composite (NASDAQCOM) — FRED](https://fred.stlouisfed.org/series/NASDAQCOM)

Regime boundaries were read off the FRED chart of the series: `2021-06-01 → 2022-09-26` (dollar strengthening, euro below parity) and `2024-09-01 → 2025-12-31` (euro rebound).

## Methods

Annualised return and volatility over 252 sessions, Sharpe ratio net of a 2% risk-free rate, maximum drawdown from running peak. OLS regression of daily returns against the benchmark, Shapiro-Wilk test on the residuals. Object-oriented design: a base class holds the calculations, a subclass applies them independently to each regime.

## Requirements

Python with `pandas`, `numpy`, `matplotlib`, `scipy`, `statsmodels`, `xlsxwriter`, `jinja2`.

```
pip install pandas numpy matplotlib scipy statsmodels xlsxwriter jinja2
```

## Running it

The script resolves its paths relative to itself, so no path needs editing:

```
python financial_analysis.py
```

It reads the two data files sitting next to it and writes every output to the same folder. Roughly ten seconds.
