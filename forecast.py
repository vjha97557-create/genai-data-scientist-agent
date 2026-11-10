import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def monthly_series(df, date_col, value_col):
    return df.dropna(subset=[date_col]).set_index(date_col)[value_col].resample("MS").sum()


def run_forecast(series, horizon=3):
    n = len(series)
    if n < 12:
        raise ValueError("At least 12 months of data are required to build a forecast.")
    test_size = max(3, int(n * 0.2))
    X = np.arange(n).reshape(-1, 1)
    y = series.values

    # Chronological split: older data for training, newer data for testing (no shuffling)
    model = LinearRegression().fit(X[:-test_size], y[:-test_size])
    pred = model.predict(X[-test_size:])
    baseline = np.repeat(y[-test_size - 1], test_size)  # naive baseline: repeat the previous period

    def metrics(true, p):
        return {"MAE": float(mean_absolute_error(true, p)),
                "RMSE": float(np.sqrt(mean_squared_error(true, p)))}

    final = LinearRegression().fit(X, y)
    future_x = np.arange(n, n + horizon).reshape(-1, 1)
    future_idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    return {"future": pd.Series(final.predict(future_x), index=future_idx),
            "model_metrics": metrics(y[-test_size:], pred),
            "baseline_metrics": metrics(y[-test_size:], baseline),
            "test_size": test_size,
            "note": "This is a linear trend model. It does not capture seasonality or sudden changes."}
