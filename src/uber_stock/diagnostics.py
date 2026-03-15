import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller


def run_return_diagnostics(df: pd.DataFrame) -> dict:
    """
    Statistical tests on the log-return series.

    All tests operate on DECIMAL log-returns (the 'log_return' column).
    Scale does not affect test statistics for ADF, Ljung-Box, or ARCH LM.
    Jarque-Bera skewness and kurtosis are scale-invariant.
    """
    r = df["log_return"].dropna()

    if len(r) < 30:
        raise ValueError(
            f"Only {len(r)} non-NaN log-return observations. "
            "Need at least 30 for meaningful diagnostic tests."
        )

    # ── Scale sanity check ────────────────────────────────────────────────
    # Decimal daily log-returns for any liquid equity should have
    # std in the range 0.005 – 0.08 (i.e. 0.5% – 8% per day).
    # Values > 0.10 almost certainly mean pct-scaled data was passed in,
    # which would make annualized_volatility and VaR wildly wrong.
    r_std = float(r.std())
    if r_std > 0.10:
        raise ValueError(
            f"log_return std = {r_std:.6f}. "
            "This is too large for DECIMAL daily log-returns "
            "(expected range 0.005–0.08 for a liquid equity). "
            "Likely cause: pre-IPO rows in the CSV are producing a single "
            "enormous return on the first real trading day. "
            "Fix: ensure data_loader.py drops all rows before "
            "the UBER IPO date (2019-05-10), or the 'log_return' column "
            "is accidentally receiving percentage-scaled values."
        )
    if r_std < 0.001:
        raise ValueError(
            f"log_return std = {r_std:.8f}. "
            "This is too small for DECIMAL daily log-returns "
            "(expected range 0.005–0.08). "
            "Check that features.py sets "
            "log_return = ln(price / price.shift(1)) without dividing further."
        )

    results = {}

    # ── Normality: Jarque-Bera ─────────────────────────────────────────────
    # H0: skewness=0 and excess-kurtosis=0 (Normal distribution).
    # Expected result for equity returns: strongly reject (p ≈ 0).
    jb_stat, jb_p = jarque_bera(r)
    results["jarque_bera"] = {
        "stat":     float(jb_stat),
        "p_value":  float(jb_p),
        "skewness": float(r.skew()),
        "kurtosis": float(r.kurt()),   # excess kurtosis (Normal = 0)
    }

    # ── Stationarity: Augmented Dickey-Fuller ─────────────────────────────
    # H0: unit root present (non-stationary).
    # Expected result: strongly reject — log-returns are stationary.
    adf_stat, adf_p, *_ = adfuller(r)
    results["adf"] = {
        "stat":    float(adf_stat),
        "p_value": float(adf_p),
    }

    # ── Autocorrelation in returns: Ljung-Box ─────────────────────────────
    # H0: no autocorrelation up to lag k.
    lb = acorr_ljungbox(r, lags=[10, 20], return_df=True)
    results["ljung_box"] = {
        str(k): v for k, v in lb.to_dict(orient="index").items()
    }

    # ── Volatility clustering: ARCH LM test ───────────────────────────────
    # H0: no ARCH effects (constant conditional variance).
    # Expected result: strongly reject — justifies GARCH modelling.
    arch_results = het_arch(r)
    results["arch_test"] = {
        "stat":    float(arch_results[0]),
        "p_value": float(arch_results[1]),
    }

    return results


def compute_risk_metrics(
    df: pd.DataFrame,
    risk_free_rate_annual: float = 0.0,
) -> dict:
    """
    Compute standard risk/performance metrics.

    CAGR
    ----
    Uses actual calendar time between the first and last index date:

        years = (last_date - first_date).days / 365.25
        CAGR  = (P_final / P_initial) ^ (1 / years) - 1

    Annualized Volatility
    ---------------------
    std(log_return) × √252, where log_return is in DECIMAL form.
    Typical range for UBER: 40–65%.

    VaR / CVaR
    ----------
    Computed on DECIMAL simple_return (not percentage).
    VaR 95%  = 5th percentile of the daily simple-return distribution.
    CVaR 95% = mean of all returns at or below the VaR threshold.
    Both are negative numbers (losses).
    Typical range for UBER: VaR ≈ −3% to −5%, CVaR ≈ −5% to −8%.
    """
    required_cols = {"simple_return", "log_return", "drawdown", "price"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"compute_risk_metrics requires columns: {required_cols}. "
            f"Missing: {missing}. Did you call add_finance_features() first?"
        )

    r_simple = df["simple_return"].dropna()
    r_log    = df["log_return"].dropna()

    # ── Scale sanity check (matches run_return_diagnostics guard) ─────────
    r_log_std = float(r_log.std())
    if r_log_std > 0.10:
        raise ValueError(
            f"log_return std = {r_log_std:.6f} in compute_risk_metrics. "
            "This is too large for DECIMAL daily log-returns "
            "(expected 0.005–0.08). "
            "The DataFrame likely contains pre-IPO rows — "
            "ensure data_loader.py drops all rows before 2019-05-10."
        )

    # ── Annualized return — CAGR using actual calendar days ───────────────
    p_initial  = float(df["price"].dropna().iloc[0])
    p_final    = float(df["price"].dropna().iloc[-1])
    first_date = df.index[0]
    last_date  = df.index[-1]
    years      = (last_date - first_date).days / 365.25

    ann_ret = (
        (p_final / p_initial) ** (1.0 / years) - 1.0
        if years > 0 and p_initial > 0
        else float("nan")
    )

    # ── Annualized volatility from DECIMAL log-returns ────────────────────
    # std is in decimal/day (e.g. 0.028 for a typical UBER trading day).
    # Multiply by √252 to scale to annual (e.g. 0.028 × 15.87 ≈ 0.44 = 44%).
    ann_vol = float(r_log_std * np.sqrt(252))

    # ── Sharpe ratio ───────────────────────────────────────────────────────
    rf_daily     = (1 + risk_free_rate_annual) ** (1.0 / 252) - 1
    excess_daily = r_simple - rf_daily
    sharpe = (
        float((excess_daily.mean() / excess_daily.std()) * np.sqrt(252))
        if excess_daily.std() > 0
        else float("nan")
    )

    # ── Tail risk ──────────────────────────────────────────────────────────
    var_95  = float(np.quantile(r_simple, 0.05))
    cvar_95 = float(r_simple[r_simple <= var_95].mean())
    max_dd  = float(df["drawdown"].min())

    return {
        "annualized_return":     float(ann_ret),
        "annualized_volatility": float(ann_vol),
        "sharpe_ratio":          float(sharpe),
        "VaR_95_daily":          var_95,
        "CVaR_95_daily":         cvar_95,
        "max_drawdown":          max_dd,
    }