from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred):  # type: ignore[misc]
        return float(mean_squared_error(y_true, y_pred) ** 0.5)


# ── GARCH model ────────────────────────────────────────────────────────────

def run_garch_model(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    forecast_horizon: int = 5,
    winsor_sigma: float = 4.0,
) -> tuple[object, dict, pd.DataFrame, pd.DataFrame]:
    
    # ── Input validation ──────────────────────────────────────────────────
    if "log_return_pct" not in df.columns:
        raise ValueError(
            "'log_return_pct' column not found. "
            "Call add_finance_features() before run_garch_model(). "
            "This column should equal log_return × 100."
        )

    returns_pct = df["log_return_pct"].dropna()

    # Scale sanity check — std of pct returns must be in 0.5–20 range.
    ret_std = float(returns_pct.std())
    if ret_std < 0.1:
        raise ValueError(
            f"'log_return_pct' std = {ret_std:.6f}. "
            "Looks like decimal returns (expected ~0.5–5.0 for pct). "
            "Ensure features.py sets log_return_pct = log_return × 100."
        )
    if ret_std > 20.0:
        raise ValueError(
            f"'log_return_pct' std = {ret_std:.4f}. "
            "Unusually large — check for outliers or wrong units."
        )
    if len(returns_pct) < 50:
        raise ValueError(
            f"Only {len(returns_pct)} non-NaN return observations — need ≥ 50."
        )

    # ── Chronological train / test split ──────────────────────────────────
    split_idx = int(len(returns_pct) * train_ratio)
    train_ret = returns_pct.iloc[:split_idx]
    test_ret  = returns_pct.iloc[split_idx:]

    print(f"  GARCH split  →  train: {len(train_ret)} days  |  test: {len(test_ret)} days")
    print(f"  Train period :  {train_ret.index[0].date()}  →  {train_ret.index[-1].date()}")
    print(f"  Test  period :  {test_ret.index[0].date()}   →  {test_ret.index[-1].date()}")
    print(f"  Return std (pct/day): {ret_std:.4f}  ← should be 0.5–5.0")

    # ── Winsorise training returns at ±winsor_sigma × std ─────────────────
    # The COVID-19 crash (Mar–Apr 2020) produced UBER daily returns of −23%
    # in pct form.  With ~1,150 training obs, 4–6 such extreme points push
    # the GARCH optimizer to persistence = 1 and ν < 3 (IGARCH + infinite
    # variance), making ALL model outputs meaningless.
    # Winsorising at ±4σ clips ~0–5 observations and keeps the optimizer
    # in the well-identified region of the parameter space.
    train_mean   = float(train_ret.mean())
    train_std    = float(train_ret.std())
    lower_clip   = train_mean - winsor_sigma * train_std
    upper_clip   = train_mean + winsor_sigma * train_std
    n_clipped    = int(((train_ret < lower_clip) | (train_ret > upper_clip)).sum())

    if n_clipped > 0:
        warnings.warn(
            f"Winsorising {n_clipped} extreme training return(s) at "
            f"±{winsor_sigma}σ (bounds: [{lower_clip:.2f}%, {upper_clip:.2f}%]). "
            "This prevents COVID-crash outliers from causing IGARCH convergence.",
            UserWarning,
            stacklevel=2,
        )
    train_ret_w = train_ret.clip(lower=lower_clip, upper=upper_clip)

    # ── Fit GARCH(1,1)-AR(1) on winsorised training set ───────────────────
    am_train = arch_model(
        train_ret_w,
        mean="AR",   lags=1,    # AR(1) mean: captures mild return autocorrelation
        vol="Garch", p=1, q=1,  # GARCH(1,1): one ARCH lag + one GARCH lag
        dist="t",               # Student-t: accommodates fat tails
        rescale=False,          # Do NOT let arch rescale — we've already scaled
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        garch_result = am_train.fit(
            disp="off",
            options={"maxiter": 500},
        )

    # ── Extract parameters ─────────────────────────────────────────────────
    params      = garch_result.params
    omega       = float(params.get("omega",    params.get("Const", 0.0)))
    alpha_1     = float(params.get("alpha[1]", 0.0))
    beta_1      = float(params.get("beta[1]",  0.0))
    nu          = float(params.get("nu", float("nan")))
    persistence = alpha_1 + beta_1

    # ── Convergence warnings ───────────────────────────────────────────────
    if persistence >= 1.0:
        warnings.warn(
            f"GARCH persistence α+β = {persistence:.6f} ≥ 1.0 (IGARCH). "
            "Even after winsorisation the model is on the IGARCH boundary. "
            "Consider using a larger winsor_sigma or inspecting the data "
            "for additional structural breaks.",
            UserWarning, stacklevel=2,
        )

    if not np.isnan(nu) and nu < 3.0:
        warnings.warn(
            f"Student-t ν = {nu:.4f} < 3 (implies infinite variance). "
            "This usually signals extreme outliers in the training data "
            "or a scale error in log_return_pct.",
            UserWarning, stacklevel=2,
        )

    # ── Unit conversions: pct/day  →  decimal annualised ──────────────────
    # arch.conditional_volatility is in pct/day (same units as input).
    # Divide by 100 to get decimal/day, then multiply by √252 to annualise.
    cond_vol_pct = garch_result.conditional_volatility      # pct/day Series

    # Use the MEAN in-sample conditional variance as the OOS starting state,
    # NOT the terminal value.  The terminal training value may be at the
    # COVID peak (very high) and would contaminate the first many OOS steps.
    mean_sigma2_train = float((cond_vol_pct ** 2).mean())   # pct²/day

    last_cond_vol_ann = float(
        (cond_vol_pct.iloc[-1] / 100.0) * np.sqrt(252)
    )

    # Unconditional (long-run) variance in pct²/day
    if persistence < 1.0:
        uncond_var_pct2  = omega / (1.0 - persistence)
        uncond_vol_daily = float(np.sqrt(uncond_var_pct2)) / 100.0  # pct → decimal
        uncond_vol_ann   = uncond_vol_daily * np.sqrt(252)
    else:
        uncond_vol_ann = float("nan")

    # ── Out-of-sample rolling 1-step-ahead forecast ────────────────────────
    sigma2_prev = mean_sigma2_train                          # pct²/day
    resid_prev  = float(train_ret_w.mean())                  # pct/day (use mean, not last)

    oos_predicted_vol_ann: list[float] = []
    realized_vol_proxy:    list[float] = []

    for ret_val_pct in test_ret.values:
        # One-step-ahead variance forecast in pct²/day
        sigma2_forecast  = omega + alpha_1 * (resid_prev ** 2) + beta_1 * sigma2_prev
        # Convert pct/day → decimal annualised
        vol_ann_forecast = (np.sqrt(sigma2_forecast) / 100.0) * np.sqrt(252)
        oos_predicted_vol_ann.append(float(vol_ann_forecast))
        # Realised proxy: |pct return| converted to decimal
        realized_vol_proxy.append(abs(float(ret_val_pct)) / 100.0)
        # Roll forward state
        resid_prev  = ret_val_pct           # actual (unwinsorised) OOS return
        sigma2_prev = sigma2_forecast

    oos_df = pd.DataFrame(
        {
            "predicted_vol_ann":  oos_predicted_vol_ann,
            "realized_vol_proxy": realized_vol_proxy,
        },
        index=test_ret.index,
    )

    # ── h-step-ahead forecast on full dataset ─────────────────────────────
    # Re-fit on ALL available data (winsorised) so the forecast uses the
    # most recent market state.
    full_mean   = float(returns_pct.mean())
    full_std    = float(returns_pct.std())
    full_lower  = full_mean - winsor_sigma * full_std
    full_upper  = full_mean + winsor_sigma * full_std
    returns_pct_w = returns_pct.clip(lower=full_lower, upper=full_upper)

    am_full = arch_model(
        returns_pct_w,
        mean="AR", lags=1,
        vol="Garch", p=1, q=1,
        dist="t",
        rescale=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        garch_full = am_full.fit(disp="off", options={"maxiter": 500})

    forecast_obj      = garch_full.forecast(horizon=forecast_horizon, reindex=False)
    forecast_var_pct2 = forecast_obj.variance.iloc[-1].values  # pct²/day, shape (h,)
    forecast_vol_ann  = (np.sqrt(forecast_var_pct2) / 100.0) * np.sqrt(252)

    last_date    = returns_pct.index[-1]
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_horizon,
    )
    forecast_df = pd.DataFrame(
        {"annualized_vol_forecast": forecast_vol_ann.tolist()},
        index=future_dates,
    )
    forecast_df.index.name = "Date"

    # ── OOS evaluation metrics ─────────────────────────────────────────────
    oos_mae  = float(
        mean_absolute_error(oos_df["realized_vol_proxy"], oos_df["predicted_vol_ann"])
    )
    oos_corr = float(
        oos_df["predicted_vol_ann"].corr(oos_df["realized_vol_proxy"])
    )

    # ── Save training-set model summary text ──────────────────────────────
    from uber_stock.config import OUTPUT_METRICS
    with open(OUTPUT_METRICS / "garch_model_summary.txt", "w") as fh:
        fh.write(garch_result.summary().as_text())

    # ── Bundle all metrics ─────────────────────────────────────────────────
    garch_metrics: dict = {
        "omega":                 omega,
        "alpha_1":               alpha_1,
        "beta_1":                beta_1,
        "persistence":           persistence,
        "nu":                    float(nu),
        "unconditional_vol_ann": uncond_vol_ann,
        "last_cond_vol_ann":     last_cond_vol_ann,
        "aic":                   float(garch_result.aic),
        "bic":                   float(garch_result.bic),
        "log_likelihood":        float(garch_result.loglikelihood),
        "n_train":               int(len(train_ret)),
        "n_test":                int(len(test_ret)),
        "n_clipped_train":       n_clipped,
        "winsor_sigma":          winsor_sigma,
        "oos_mae_vol":           oos_mae,
        "oos_corr_vol":          oos_corr,
    }

    return garch_result, garch_metrics, oos_df, forecast_df

