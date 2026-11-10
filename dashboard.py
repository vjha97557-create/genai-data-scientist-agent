"""Power BI-style dashboard: data-quality cards, KPI cards, slicers, charts and auto insights.

Nothing here depends on specific column names. `detect_columns` guesses a role for each
column from its name *and* its data (type, cardinality); the sidebar mapping lets the user
correct a wrong guess. Every chart is skipped cleanly when the data for it does not exist.
"""
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import show_chart, show_df, wkey

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
BLUE, ORANGE = "#118DFF", "#E66C37"
PALETTE = [BLUE, "#1AAB40", "#12239E", ORANGE, "#6B007B", "#E044A7", "#744EC2", "#D9B300"]

_CSS = """<style>
[data-testid="stMetricValue"] {font-size: 1.7rem; font-weight: 600;}
[data-testid="stMetricLabel"] p {opacity: .75; font-size: .85rem;}
</style>"""


# ---------------- column detection ----------------
def _norm(c):
    return str(c).lower().replace("_", " ").replace("-", " ")


_NON_MEASURE_TOKENS = {"id", "uuid", "sku", "index", "zip", "pincode", "postal", "phone",
                       "year", "month", "day", "lat", "latitude", "lon", "lng", "longitude"}


def _name_is_id(c):
    name = str(c)
    return bool(set(_norm(name).split()) & {"id", "uuid", "sku"}) or bool(re.search(r"[a-z]ID$", name))


def _numeric_cols(df):
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def _is_non_measure(df, c):
    """Identifier / code / calendar columns that should not be summed as a business measure."""
    if set(_norm(c).split()) & _NON_MEASURE_TOKENS or _name_is_id(c):
        return True
    s = df[c].dropna()
    if len(s) > 20 and s.is_unique and s.is_monotonic_increasing:   # running row number
        return bool(np.all(np.mod(s.to_numpy(dtype=float), 1) == 0))
    return False


def _measure_cols(df):
    nums = _numeric_cols(df)
    return [c for c in nums if not _is_non_measure(df, c)] or nums


def detect_columns(df):
    """Guess which column plays which role. Returns {role: column name or None}."""
    cols = list(df.columns)
    nums = _numeric_cols(df)
    dates = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    texts = [c for c in cols if c not in nums and c not in dates]
    texts_noid = [c for c in texts if not _name_is_id(c)]
    measures = [c for c in nums if not _is_non_measure(df, c)]

    def find(pool, any_of, none_of=()):
        for key in any_of:                       # keyword priority first, then column order
            for c in pool:
                n = _norm(c)
                if key in n and not any(k in n for k in none_of):
                    return c
        return None

    amount = find(measures, ["amount", "sales", "revenue", "total", "gmv", "turnover", "income"],
                  ["profit", "qty", "quantity", "discount", "tax"])
    quantity = find([c for c in measures if c != amount],
                    ["quantity", "qty", "units", "volume", "items", "count"])
    profit = find([c for c in measures if c not in (amount, quantity)], ["profit"])
    if amount is None:      # no obvious name: use the numeric column with the largest magnitude
        rest = [c for c in measures if c not in (quantity, profit)]
        if rest:
            amount = max(rest, key=lambda c: float(np.nan_to_num(df[c].abs().sum())))

    subcategory = find(texts_noid, ["sub category", "subcategory", "sub cat"])
    category = find(texts_noid, ["category", "segment", "department", "genre", "class",
                                 "brand", "product", "group", "type"], ["sub", "payment"])
    payment = find(texts_noid, ["payment", "pay mode", "paymode", "method"])
    state = find(texts_noid, ["state", "region", "province", "country", "city", "zone",
                              "location", "area"])
    customer = find(texts_noid, ["customer", "client", "buyer", "user", "member"])
    order_id = find([c for c in cols if c not in dates],
                    ["order id", "orderid", "order no", "order number", "invoice",
                     "transaction", "txn", "bill"])

    if category is None or state is None:   # fall back to low-cardinality text columns
        used = {subcategory, category, payment, state, customer}
        low = [c for c in texts_noid if c not in used and 2 <= df[c].nunique() <= 30]
        if category is None and low:
            category = low.pop(0)
        if state is None and low:
            state = low.pop(0)

    date = find(dates, ["date", "time", "day"]) or (dates[0] if dates else None)
    return {"amount": amount, "quantity": quantity, "profit": profit, "category": category,
            "subcategory": subcategory, "payment": payment, "state": state,
            "customer": customer, "order_id": order_id, "date": date}


def mapping_ui(df):
    """Sidebar mapping so the user can correct a wrong guess. Only columns that make sense
    for a role are offered (numbers for Amount/Quantity/Profit, dates for Date), and the
    widget keys are tied to the dataset so a new file never inherits the old selection."""
    auto = detect_columns(df)
    labels = {"amount": "Amount / Sales", "quantity": "Quantity", "profit": "Profit",
              "category": "Category", "subcategory": "Sub-Category", "payment": "Payment Mode",
              "state": "State / Region", "customer": "Customer", "order_id": "Order ID",
              "date": "Date"}
    nums = _numeric_cols(df)
    dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    pools = {"amount": nums, "quantity": nums, "profit": nums, "date": dates}
    others = [c for c in df.columns if c not in dates]
    m = {}
    with st.sidebar.expander("🧭 Column mapping (for the dashboard)"):
        for k, label in labels.items():
            opts = ["(none)"] + list(pools.get(k, others))
            idx = opts.index(auto[k]) if auto[k] in opts else 0
            pick = st.selectbox(label, opts, index=idx, key=wkey(f"map_{k}"))
            m[k] = None if pick == "(none)" else pick
    return m


# ---------------- reusable aggregations ----------------
def agg_by(df, cat, val, n=5, asc=False, how="sum"):
    g = df.groupby(cat, observed=True)[val].agg(how).dropna()
    return g.sort_values(ascending=asc).head(n).reset_index()


def month_profit(df, m, val=None):
    """`val` summed per calendar month in chronological order ('Jan 2024', 'Feb 2024', ...).
    Months of different years are kept apart instead of being merged."""
    val = val or m["profit"]
    d = df[[m["date"], val]].dropna(subset=[m["date"]])
    g = d.groupby(d[m["date"]].dt.to_period("M"))[val].sum().sort_index()
    return pd.DataFrame({"Month": g.index.strftime("%b %Y"), val: g.to_numpy()})


def _avg_order_value(df, m):
    A = m.get("amount")
    if not A:
        return None
    orders = df[m["order_id"]].nunique() if m.get("order_id") else len(df)
    return df[A].sum() / orders if orders else 0.0


def kpis(df, m):
    out = {}
    A, Q, P = m.get("amount"), m.get("quantity"), m.get("profit")
    if A:
        out[f"Sum of {A}"] = df[A].sum()
    if Q:
        out[f"Sum of {Q}"] = df[Q].sum()
    if P:
        out[f"Sum of {P}"] = df[P].sum()
    if A:
        label = "Avg Order Value" if m.get("order_id") else f"Avg {A} per Record"
        out[label] = _avg_order_value(df, m)
    if not out:             # nothing mapped: still show something useful
        for c in _measure_cols(df)[:4]:
            out[f"Sum of {c}"] = df[c].sum()
    return out


def human(v):
    if v is None or pd.isna(v):
        return "–"
    a = abs(v)
    if a >= 1e7:
        return f"{v / 1e7:.2f}Cr"
    if a >= 1e5:
        return f"{v / 1e5:.2f}L"
    if a >= 1e3:
        return f"{v / 1e3:.1f}K"
    return f"{v:,.0f}"


# ---------------- chart styling ----------------
def _style(fig, h=300, legend=False):
    # No fixed template: st.plotly_chart applies the active Streamlit theme (light or dark),
    # so labels stay readable in both.
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      title_x=0.5, title_font_size=15, showlegend=legend)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def _labels(g, cat):
    g = g.copy()
    g[cat] = g[cat].astype(str)      # numeric/date categories must be treated as labels
    return g


def _ok(df, cat, val):
    return bool(cat) and bool(val) and cat != val and cat in df.columns and val in df.columns


def _rank_bar(g, cat, val, title, color=BLUE, multicolor=False):
    """Horizontal bars for an already sorted frame (largest first -> drawn at the top)."""
    if g.empty:
        return None
    g = _labels(g, cat).iloc[::-1]
    fig = px.bar(g, x=val, y=cat, orientation="h", title=title, text_auto=".2s",
                 color=cat if multicolor else None,
                 color_discrete_sequence=PALETTE if multicolor else [color])
    return _style(fig)


def _hbar(df, cat, val, title, n=5, color=BLUE, multicolor=False, how="sum"):
    if not _ok(df, cat, val):
        return None
    return _rank_bar(agg_by(df, cat, val, n, how=how), cat, val, title, color, multicolor)


def _donut(df, cat, val, title):
    if not _ok(df, cat, val):
        return None
    g = agg_by(df, cat, val, 8)
    g = g[g[val] > 0]                # a pie cannot show zero/negative slices
    if g.empty:
        return None
    fig = px.pie(_labels(g, cat), names=cat, values=val, hole=0.6, title=title,
                 color_discrete_sequence=PALETTE)
    fig.update_traces(textinfo="label+percent")
    return _style(fig)


def _vbar_top(df, cat, val, title, n=5):
    if not _ok(df, cat, val):
        return None
    g = _labels(agg_by(df, cat, val, n), cat)
    if g.empty:
        return None
    cmap = {v: (ORANGE if i == 0 else BLUE) for i, v in enumerate(g[cat])}   # highlight top bar
    fig = px.bar(g, x=cat, y=val, title=title, color=cat, color_discrete_map=cmap, text_auto=".2s")
    return _style(fig)


def _profit_month(df, m):
    if not (m.get("date") and m.get("profit")):
        return None
    g = month_profit(df, m)
    if g.empty:
        return None
    P = m["profit"]
    g["Sign"] = np.where(g[P] >= 0, "Profit", "Loss")
    fig = px.bar(g, x="Month", y=P, color="Sign", title=f"{P} by Month", text_auto=".2s",
                 color_discrete_map={"Profit": BLUE, "Loss": ORANGE})
    return _style(fig, 340)


def _trend(df, date, val, how, title):
    d = df[[date, val]].dropna()
    if d.empty:
        return None
    span = (d[date].max() - d[date].min()).days
    g = d.set_index(date)[val].resample("D" if span <= 90 else "MS").agg(how).reset_index()
    g.columns = ["Period", val]
    return _style(px.line(g, x="Period", y=val, markers=True, title=title,
                          color_discrete_sequence=[BLUE]), 320)


def _hist(df, col, title):
    s = df[col].dropna()
    if s.empty:
        return None
    counts, edges = np.histogram(s.to_numpy(dtype=float), bins=30)   # light even for millions of rows
    mid = (edges[:-1] + edges[1:]) / 2
    fig = px.bar(x=mid, y=counts, title=title, labels={"x": col, "y": "Count"},
                 color_discrete_sequence=[BLUE])
    fig.update_traces(marker_line_width=0)
    fig.update_layout(bargap=0.05)
    return _style(fig, 320)


def _corr(df, cols):
    cols = [c for c in cols if df[c].nunique() > 1][:12]
    if len(cols) < 2:
        return None
    fig = px.imshow(df[cols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto", title="Correlation Heatmap")
    return _style(fig, 380)


def _group_table(df, dim, val, how):
    g = df.groupby(dim, observed=True)[val].agg(["count", "sum", "mean"])
    total = g["sum"].sum()
    g["share_%"] = (g["sum"] / total * 100).round(1) if total else np.nan
    g = g.sort_values("sum" if how in ("sum", "count") else how, ascending=False)
    return g.round(2).head(50).reset_index()


def _grid(items, ncols=2):
    """items: list of (key, figure). Skips empty figures and lays the rest out in a grid."""
    items = [(k, f) for k, f in items if f is not None]
    for i in range(0, len(items), ncols):
        for col, (k, fig) in zip(st.columns(ncols), items[i:i + ncols]):
            with col.container(border=True):
                show_chart(fig, key=wkey(f"dash_{k}"))


# ---------------- dashboard page ----------------
def render_dashboard(df, m, raw_df=None):
    st.markdown(_CSS, unsafe_allow_html=True)
    date, cat = m.get("date"), m.get("category")
    A, Q, P = m.get("amount"), m.get("quantity"), m.get("profit")
    full = df

    # ---- slicers (Power BI-style quarter buttons and category dropdown)
    c1, c2 = st.columns([2, 1])
    if date:
        q = c1.radio("Quarter", ["All", "Qtr 1", "Qtr 2", "Qtr 3", "Qtr 4"], horizontal=True,
                     key=wkey("dash_quarter"))
        if q != "All":
            df = df[df[date].dt.quarter == int(q[-1])]
    if cat:
        cat_opts = ["All"] + sorted(full[cat].dropna().astype(str).unique())
        pick = c2.selectbox(cat, cat_opts, key=wkey("dash_category"))
        if pick != "All":
            df = df[df[cat].astype(str) == pick]
    if df.empty:
        st.warning("No data is available for this selection.")
        return df

    # ---- data overview cards
    dup = int((raw_df if raw_df is not None else df).duplicated().sum())
    o1, o2, o3, o4 = st.columns(4)
    for col, (label, val, tip) in zip((o1, o2, o3, o4), [
            ("Total Records", f"{len(df):,}", "Rows in the current selection"),
            ("Total Columns", f"{len(df.columns):,}", None),
            ("Missing Values", f"{int(df.isna().sum().sum()):,}", "Empty cells in the current selection"),
            ("Duplicate Records", f"{dup:,}", "Duplicate rows in the original file")]):
        with col.container(border=True):
            st.metric(label, val, help=tip)

    # ---- KPI cards
    k = kpis(df, m)
    if k:
        for col, (label, val) in zip(st.columns(len(k)), k.items()):
            with col.container(border=True):
                st.metric(label, human(val))
    else:
        st.info("No numeric columns were found, so KPI cards cannot be shown.")

    # ---- business overview (uses the column mapping)
    overview = []
    if m.get("state") and A:
        overview.append(("state", _hbar(df, m["state"], A, f"Top 5 {m['state']} by {A}")))
    if cat and (Q or A):
        overview.append(("cat", _donut(df, cat, Q or A, f"{Q or A} by {cat}")))
    if m.get("customer") and A:
        overview.append(("cust", _vbar_top(df, m["customer"], A, f"Top 5 {m['customer']} by {A}")))
    if m.get("payment") and (Q or A):
        overview.append(("pay", _donut(df, m["payment"], Q or A, f"{Q or A} by {m['payment']}")))
    if date and P:
        overview.append(("profit_month", _profit_month(df, m)))
    if m.get("subcategory") and P:
        overview.append(("subcat", _hbar(df, m["subcategory"], P, f"{P} by {m['subcategory']}",
                                         multicolor=True)))
    if any(f is not None for _, f in overview):
        st.subheader("Business overview")
        _grid(overview)

    # ---- explore: measure x dimension x aggregation, trend, distribution, top/bottom 5
    measures = _measure_cols(df)
    dims = [c for c in df.columns
            if c not in _numeric_cols(df) and not pd.api.types.is_datetime64_any_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c]) and df[c].nunique() >= 2]
    if measures:
        st.subheader("Explore your data")
        s1, s2, s3 = st.columns(3)
        measure = s1.selectbox("Measure", measures, key=wkey("dash_measure"),
                               index=measures.index(A) if A in measures else 0)
        pref = [m.get(r) for r in ("state", "category", "subcategory", "customer", "payment")]
        dim_default = next((c for c in pref if c in dims), dims[0] if dims else None)
        dim = s2.selectbox("Group by", dims, key=wkey("dash_dim"),
                           index=dims.index(dim_default) if dim_default else 0) if dims else None
        how = s3.selectbox("Aggregation", ["sum", "mean", "count", "max", "min"],
                           key=wkey("dash_how"))

        row = []
        if date:
            row.append(("trend", _trend(df, date, measure, how, f"{how.title()} of {measure} over time")))
        row.append(("hist", _hist(df, measure, f"Distribution of {measure}")))
        _grid(row)

        if dim and dim != measure:
            _grid([("top5", _hbar(df, dim, measure, f"Top 5 {dim} by {how} of {measure}", how=how)),
                   ("bottom5", _rank_bar(agg_by(df, dim, measure, 5, asc=True, how=how), dim, measure,
                                         f"Bottom 5 {dim} by {how} of {measure}", color=ORANGE))])
            st.markdown(f"**Category-wise analysis: {measure} by {dim}**")
            show_df(_group_table(df, dim, measure, how))
        elif not dims:
            st.info("No categorical columns were found for Top 5 / Bottom 5 and category analysis.")

        # ---- correlation heatmap and numeric summary
        nums = [c for c in _numeric_cols(df)]
        corr = _corr(df, _measure_cols(df))
        if corr is not None or nums:
            st.subheader("Correlation & numeric summary")
            if corr is not None:
                with st.container(border=True):
                    show_chart(corr, key=wkey("dash_corr"))
            else:
                st.info("At least two varying numeric columns are needed for a correlation heatmap.")
            show_df(df[nums].describe().T)
    else:
        st.info("No numeric columns were found. Charts that need numbers are not available "
                "for this dataset; the Data Profile and Chart Builder tabs still work.")
    return df


# ---------------- insights ----------------
def _share(g):
    return g.iloc[0] / g.sum() * 100 if g.sum() else 0


def generate_insights(df, m):
    out = []
    if df.empty:
        return out
    A, Q, P = m.get("amount"), m.get("quantity"), m.get("profit")
    if A:
        out.append(f"Total {A} **{df[A].sum():,.0f}**, average per order/record "
                   f"**{_avg_order_value(df, m):,.0f}**.")
    if A and P and df[A].sum():
        out.append(f"Overall profit margin **{df[P].sum() / df[A].sum() * 100:.1f}%** "
                   f"(total profit {df[P].sum():,.0f}).")
    if P:
        out.append(f"**{(df[P] < 0).mean() * 100:.1f}%** of records are loss-making (negative profit).")
    if m.get("state") and A:
        g = df.groupby(m["state"], observed=True)[A].sum().sort_values(ascending=False)
        if len(g) > 1:
            out.append(f"Top {m['state']}: **{g.index[0]}** ({g.iloc[0]:,.0f}, {_share(g):.0f}% of total {A}). "
                       f"Lowest: **{g.index[-1]}** ({g.iloc[-1]:,.0f}).")
    if m.get("category") and (Q or A):
        v = Q or A
        g = df.groupby(m["category"], observed=True)[v].sum().sort_values(ascending=False)
        if len(g) > 1:
            out.append(f"**{g.index[0]}** is the largest {m['category']} ({_share(g):.0f}% of {v}).")
    if m.get("payment"):
        pay = m["payment"]
        g = (df.groupby(pay, observed=True)[Q].sum().sort_values(ascending=False) if Q
             else df[pay].value_counts())
        if len(g) > 1:
            out.append(f"Most-used {pay}: **{g.index[0]}** ({_share(g):.0f}%).")
    if m.get("customer") and A:
        g = df.groupby(m["customer"], observed=True)[A].sum().sort_values(ascending=False)
        if len(g) > 1:
            out.append(f"Top {m['customer']}: **{g.index[0]}** ({g.iloc[0]:,.0f}).")
    if m.get("date") and P:
        g = month_profit(df, m).set_index("Month")[P]
        if len(g) > 1:
            out.append(f"Highest {P} in **{g.idxmax()}** ({g.max():,.0f}); "
                       f"lowest in **{g.idxmin()}** ({g.min():,.0f}).")
            loss = [mo for mo, v in g.items() if v < 0]
            if loss:
                out.append("Loss-making months: **" + ", ".join(loss) + "**.")
        gq = df.groupby(df[m["date"]].dt.quarter)[P].sum()
        if len(gq) > 1:
            out.append(f"Best quarter: **Qtr {int(gq.idxmax())}** ({P} {gq.max():,.0f}).")
    if m.get("subcategory") and P:
        g = df.groupby(m["subcategory"], observed=True)[P].sum().sort_values(ascending=False)
        if len(g) > 1:
            out.append(f"Most profitable {m['subcategory']}: **{g.index[0]}** ({g.iloc[0]:,.0f}).")
            bad = g[g < 0]
            if len(bad):
                out.append(f"Loss-making {m['subcategory']}: **" + ", ".join(map(str, bad.index)) + "**.")
    return out


def generate_recommendations(df, m):
    rec = []
    if df.empty:
        return rec
    A, Q, P = m.get("amount"), m.get("quantity"), m.get("profit")
    if m.get("subcategory") and P:
        g = df.groupby(m["subcategory"], observed=True)[P].sum()
        bad = list(g[g < 0].index)
        if bad:
            rec.append("Review pricing, discounts and costs for the loss-making sub-categories (" + ", ".join(map(str, bad)) + ").")
    if m.get("date") and P:
        g = month_profit(df, m).set_index("Month")[P]
        bad = [mo for mo, v in g.items() if v < 0]
        if bad:
            rec.append("Investigate discounts and returns in the loss-making months (" + ", ".join(bad) + ").")
    if m.get("payment"):
        pay = m["payment"]
        g = df.groupby(pay, observed=True)[Q].sum() if Q else df[pay].value_counts()
        cod = [i for i in g.index if str(i).strip().upper() == "COD"]
        if cod and g.sum() > 0 and g[cod[0]] / g.sum() > 0.4:
            rec.append(f"Cash on delivery (COD) accounts for {g[cod[0]] / g.sum() * 100:.0f}% of orders. Prepaid/UPI incentives can reduce returns and cash-handling costs.")
    if m.get("state") and A:
        g = df.groupby(m["state"], observed=True)[A].sum()
        if len(g) > 1 and g.sum() > 0 and g.max() / g.sum() > 0.4:
            rec.append(f"Sales depend heavily on {g.idxmax()} ({g.max() / g.sum() * 100:.0f}%). Consider expanding marketing in other {m['state']} values.")
    if not rec:
        rec.append("No major risk signals were found. Map additional columns for a more detailed analysis.")
    return rec
