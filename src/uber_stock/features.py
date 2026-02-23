import numpy as np
import pandas as pd


def add_finance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Returns
    df["simple_return"] = df["price"].pct_change()
    df["log_return"] = np.log(df["price"]).diff()

    # Volume transforms
    df["log_volume"] = np.log(df["Volume"].clip(lower=1))
    df["volume_log_change"] = df["log_volume"].diff()

    # Time features
    df["weekday"] = df["Date"].dt.day_name()
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter

    # Rolling volatility (annualized)
    trading_days = 252
    df["vol_21d"] = df["log_return"].rolling(21).std() * np.sqrt(trading_days)
    df["vol_63d"] = df["log_return"].rolling(63).std() * np.sqrt(trading_days)

    # Cumulative performance + drawdown
    df["cum_return"] = (1 + df["simple_return"].fillna(0)).cumprod()
    df["rolling_peak"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_return"] / df["rolling_peak"] - 1

    # Lagged features for regression 
    df["ret_lag1"] = df["log_return"].shift(1)
    df["ret_lag2"] = df["log_return"].shift(2)
    df["ret_lag5"] = df["log_return"].shift(5)

    return df