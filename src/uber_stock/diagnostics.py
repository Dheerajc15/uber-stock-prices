import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller


def run_return_diagnostics(df: pd.DataFrame) -> dict:
    r = df["log_return"].dropna()

    results = {}

    # Normality (returns are often non-normal)
    jb_stat, jb_p = jarque_bera(r)
    results["jarque_bera"] = {"stat": float(jb_stat), "p_value": float(jb_p)}

    # Stationarity
    adf_stat, adf_p, *_ = adfuller(r)
    results["adf"] = {"stat": float(adf_stat), "p_value": float(adf_p)}

    # Autocorrelation in returns
    lb = acorr_ljungbox(r, lags=[10, 20], return_df=True)
    results["ljung_box"] = lb.to_dict(orient="index")

    # ARCH effects (volatility clustering)
    arch_stat, arch_p, _, _ = het_arch(r)
    results["arch_test"] = {"stat": float(arch_stat), "p_value": float(arch_p)}

    return results


def compute_risk_metrics(df: pd.DataFrame, risk_free_rate_annual: float = 0.0) -> dict:
    r = df["simple_return"].dropna()

    ann_ret = (1 + r.mean()) ** 252 - 1
    ann_vol = r.std() * np.sqrt(252)

    rf_daily = (1 + risk_free_rate_annual) ** (1 / 252) - 1
    excess_daily = r - rf_daily
    sharpe = (excess_daily.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else np.nan

    var_95 = np.quantile(r, 0.05)
    cvar_95 = r[r <= var_95].mean()

    max_dd = df["drawdown"].min()

    return {
        "annualized_return": float(ann_ret),
        "annualized_volatility": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "VaR_95_daily": float(var_95),
        "CVaR_95_daily": float(cvar_95),
        "max_drawdown": float(max_dd),
    }