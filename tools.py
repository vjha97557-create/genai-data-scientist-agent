"""The LLM may only call these approved functions."""
import inspect
import pandas as pd

ALLOWED_AGG = ["sum", "mean", "count", "max", "min"]


def get_data_profile(df):
    return {"rows": int(df.shape[0]), "columns": int(df.shape[1]),
            "missing_values": df.isna().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum())}


def top_n(df, group_col, value_col, limit=5, agg="sum"):
    return df.groupby(group_col)[value_col].agg(agg).sort_values(ascending=False).head(limit).reset_index()


def bottom_n(df, group_col, value_col, limit=5, agg="sum"):
    return df.groupby(group_col)[value_col].agg(agg).sort_values().head(limit).reset_index()


def column_summary(df, column):
    return df[column].describe().reset_index()


def monthly_trend(df, date_col, value_col):
    s = df.dropna(subset=[date_col]).set_index(date_col)[value_col].resample("MS").sum().reset_index()
    s[date_col] = s[date_col].dt.strftime("%Y-%m")
    return s


# tool -> (function, {argument: expected column type})
TOOLS = {
    "get_data_profile": (get_data_profile, {}),
    "top_n": (top_n, {"group_col": "any", "value_col": "num"}),
    "bottom_n": (bottom_n, {"group_col": "any", "value_col": "num"}),
    "column_summary": (column_summary, {"column": "num"}),
    "monthly_trend": (monthly_trend, {"date_col": "date", "value_col": "num"}),
}


def validate_and_run(df, tool, args):
    """Validate the LLM arguments before running a function."""
    if tool not in TOOLS:
        raise ValueError(f"Tool '{tool}' is not allowed.")
    func, spec = TOOLS[tool]
    args = args or {}
    clean = {}
    for arg, kind in spec.items():
        col = args.get(arg)
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in the dataset.")
        if kind == "num" and not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"'{col}' is not a numeric column.")
        if kind == "date" and not pd.api.types.is_datetime64_any_dtype(df[col]):
            raise ValueError(f"'{col}' is not a date column.")
        clean[arg] = col
    params = inspect.signature(func).parameters
    if "limit" in params:
        try:
            clean["limit"] = max(1, min(int(args.get("limit", 5)), 50))
        except (TypeError, ValueError):
            clean["limit"] = 5
    if "agg" in params:
        agg = args.get("agg", "sum")
        clean["agg"] = agg if agg in ALLOWED_AGG else "sum"
    return func(df, **clean)
