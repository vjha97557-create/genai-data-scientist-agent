"""Shared helpers: Streamlit version compatibility, file reading, type cleaning, widget keys."""
import importlib
import inspect
import io
import re
import warnings

import pandas as pd
import streamlit as st


# ---------------- Streamlit version compatibility ----------------
def _uses_width_api(fn) -> bool:
    """True on Streamlit versions where `width="stretch"` replaced `use_container_width`."""
    try:
        return inspect.signature(fn).parameters["width"].default == "stretch"
    except (KeyError, ValueError, TypeError):
        return False


_NEW_PLOTLY = _uses_width_api(st.plotly_chart)
_NEW_DF = _uses_width_api(st.dataframe)


def show_chart(fig, key=None):
    """Render a Plotly figure full-width on any Streamlit version. Pass a unique `key` when
    the same page can contain two very similar charts."""
    if fig is None:
        return
    if _NEW_PLOTLY:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Streamlit serialises tables with Arrow, which fails on object columns holding mixed
    types (e.g. numbers and text in one column). Convert only those columns to text."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            kinds = {type(v) for v in out[c].dropna().head(5000)}
            if len(kinds) > 1:
                out[c] = out[c].map(lambda v: v if pd.isna(v) else str(v))
    return out


def show_df(data, **kw):
    if isinstance(data, pd.DataFrame):
        data = arrow_safe(data)
    if _NEW_DF:
        st.dataframe(data, width="stretch", **kw)
    else:
        st.dataframe(data, use_container_width=True, **kw)


# ---------------- widget keys / dataset identity ----------------
def wkey(name: str) -> str:
    """Widget key tied to the current dataset. Column-dependent widgets (filters, column
    mapping, chart builder) get a fresh state whenever the dataset changes, so selections
    from the previous dataset can never leak into the new one."""
    return f"{name}::{st.session_state.get('_ds_sig', '')}"


def optional_import(name: str):
    """Import a project module lazily. A missing module/dependency shows an error in the tab
    that needs it instead of crashing the whole app."""
    try:
        return importlib.import_module(name)
    except ImportError as e:
        st.error(f"Module `{name}` could not be loaded: {e}. "
                 "Install the packages from requirements.txt and restart the app.")
        return None


# ---------------- file reading ----------------
@st.cache_data(show_spinner=False)
def read_table(data: bytes, name: str, sheet=None):
    if name.lower().endswith(".csv"):
        last_err = None
        for enc in ("utf-8-sig", "latin-1"):          # utf-8-sig also strips an Excel BOM
            try:
                df = pd.read_csv(io.BytesIO(data), encoding=enc)
                if df.shape[1] == 1:                  # maybe ';' or tab separated
                    alt = pd.read_csv(io.BytesIO(data), encoding=enc, sep=None, engine="python")
                    if alt.shape[1] > 1:
                        df = alt
                return df
            except UnicodeDecodeError as e:
                last_err = e
        raise last_err
    return pd.read_excel(io.BytesIO(data), sheet_name=sheet if sheet is not None else 0)


def sheet_names(data: bytes):
    return pd.ExcelFile(io.BytesIO(data)).sheet_names


# ---------------- type cleaning ----------------
def clean_columns(cols):
    """Trim names, replace spaces with '_', fill blanks and make every name unique
    (duplicate names make df[col] return a DataFrame and break everything downstream)."""
    seen, out = {}, []
    for i, c in enumerate(cols):
        name = re.sub(r"\s+", "_", str(c).strip()) or f"column_{i + 1}"
        n = seen.get(name, 0)
        seen[name] = n + 1
        out.append(name if n == 0 else f"{name}_{n + 1}")
    return out


_DATE_HINT = re.compile(
    r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"                  # 2024-01-31, 31/01/2024
    r"|\d{1,2}[\s-][A-Za-z]{3,9}[\s,-]+\d{2,4}"          # 31 Jan 2024, 31-Jan-24
    r"|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{2,4}")          # Jan 31, 2024


def parse_dates(s: pd.Series, dayfirst=True, min_ratio=0.8):
    """Return `s` parsed as datetimes if at least `min_ratio` of its non-empty values are
    dates, otherwise None. Content-based (not name-based), so a text column such as
    'Candidate' is never turned into dates and a date column may have any name."""
    nonnull = s.dropna()
    if nonnull.empty:
        return None
    sample = nonnull.astype(str).head(300)
    if sample.str.contains(_DATE_HINT).mean() < min_ratio:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
        if parsed.notna().sum() / len(nonnull) < min_ratio:
            try:                                          # mixed date formats in one column
                parsed = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, format="mixed")
            except (ValueError, TypeError):
                pass
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        return None
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed if parsed.notna().sum() / len(nonnull) >= min_ratio else None


_NUM_NOISE = re.compile(r"[,\s₹$€£%]")


def coerce_numeric_text(s: pd.Series, min_ratio=0.95):
    """Turn text such as '1,200' or '₹2,500.50' into numbers when (almost) every value is one.
    Plain digit strings are left alone so codes like '00123' keep their leading zeros."""
    nonnull = s.dropna().astype(str).str.strip()
    nonnull = nonnull[nonnull != ""]
    if nonnull.empty or not nonnull.str.contains(r"[,₹$€£%]").any():
        return None
    cleaned = nonnull.str.replace(_NUM_NOISE, "", regex=True)
    nums = pd.to_numeric(cleaned, errors="coerce")
    if nums.notna().mean() < min_ratio:
        return None
    return pd.to_numeric(s.astype(str).str.replace(_NUM_NOISE, "", regex=True), errors="coerce")


def to_markdown_table(df: pd.DataFrame) -> str:
    """Small dependency-free markdown table (DataFrame.to_markdown needs `tabulate`)."""
    d = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    rows = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
            for r in d.itertuples(index=False)]
    return "\n".join([head, sep] + rows)
