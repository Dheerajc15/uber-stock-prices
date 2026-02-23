import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def run_return_regression(df: pd.DataFrame):
    model_df = df.copy()

    # Predict next day's return (shift target backward)
    model_df["target_next_ret"] = model_df["log_return"].shift(-1)

    cols = ["ret_lag1", "ret_lag2", "ret_lag5", "volume_log_change", "target_next_ret"]
    model_df = model_df[cols].dropna()

    # Time-based split (better than random split for time series)
    split_idx = int(len(model_df) * 0.8)
    train = model_df.iloc[:split_idx]
    test = model_df.iloc[split_idx:]

    X_train = sm.add_constant(train[["ret_lag1", "ret_lag2", "ret_lag5", "volume_log_change"]])
    y_train = train["target_next_ret"]

    X_test = sm.add_constant(test[["ret_lag1", "ret_lag2", "ret_lag5", "volume_log_change"]])
    y_test = test["target_next_ret"]

    model = sm.OLS(y_train, X_train).fit()

    y_pred = model.predict(X_test)

    metrics = {
        "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }

    return model, metrics, test.assign(predicted=y_pred)