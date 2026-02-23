# Uber Stock (UBER) — Finance-Style Returns & Risk Analysis

This project analyzes **Uber (UBER) stock** using a **finance-style workflow** built in Python with a modular structure.  
Instead of relying on raw price-level analysis (which can be misleading for statistical inference), the project pivots to **daily returns and log-returns**, which are the standard units for risk and time-series analysis in finance.

The repository is structured for **portfolio presentation**, reproducibility, and clean execution in **VS Code**.

---

## Project Objective

The goal of this project is to:

- Build a **modular, reusable financial analytics pipeline** for stock data
- Analyze **return behavior** rather than only raw prices
- Quantify risk using practical metrics such as:
  - **Annualized return**
  - **Annualized volatility**
  - **Sharpe ratio**
  - **Value at Risk (VaR)**
  - **Conditional Value at Risk (CVaR)**
  - **Maximum drawdown**
- Run key return diagnostics (normality, stationarity, volatility clustering)
- Create a simple **baseline model** for next-day return prediction

---

## Dataset

- Source: **Yahoo Finance historical Uber stock data**
- File: `data/raw/uber_stock_data.csv`

### Data columns used
- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Adj Close` (preferred for return calculations)
- `Volume`

---

## Project Structure

```text
uber-stock-prices/
│
├── data/
│   ├── raw/
│   │   └── uber_stock_data.csv
│   └── processed/
│
├── notebooks/
│   └── report.ipynb
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── tables/
│
├── scripts/
│   └── run_pipeline.py
│
├── src/
│   └── uber_stock/
│       ├── __init__.py
│       ├── config.py
│       ├── data_loader.py
│       ├── features.py
│       ├── diagnostics.py
│       ├── plots.py
│       ├── models.py
│       └── pipeline.py
│
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md