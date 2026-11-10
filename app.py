import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
import markdown
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
try:
    os.environ["GROQ_API_KEY"]=st.secrets["GROQ_API_KEY"]
    os.environ["GROQ_MODEL"]=st.secrets.get("GROQ_MODEL","openai/gpt-oss-120b")
except (KeyError, FileNotFoundError):
    pass

from dashboard import (detect_columns, generate_insights, generate_recommendations,
                       kpis, human, mapping_ui, render_dashboard)
from utils import read_table, sheet_names, show_chart, show_df

st.set_page_config(page_title="Data Scientist Agent", page_icon="📊", layout="wide")
st.title("📊 GenAI Data Scientist Agent")
st.caption("Upload an Excel/CSV file → clean the data → explore a Power BI-style dashboard → get insights → ask AI questions → export a report")

# ======================= 1. FILE UPLOAD =======================
use_sample = st.sidebar.checkbox("Use demo data (data/sample_sales.csv)")
uploaded = None
if use_sample:
    raw_df, file_name = pd.read_csv("data/sample_sales.csv"), "sample_sales.csv"
else:
    uploaded = st.file_uploader("Choose a dataset", type=["csv", "xlsx"])
    if not uploaded:
        st.info("Upload a file to get started, or tick 'Use demo data' in the sidebar.")
        st.stop()
    file_name, data = uploaded.name, uploaded.getvalue()
    try:
        sheet = None
        if file_name.lower().endswith(".xlsx"):
            sheet = st.sidebar.selectbox("Select sheet", sheet_names(data))
        raw_df = read_table(data, file_name, sheet)
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        st.stop()

if raw_df.empty:
    st.error("The file is empty.")
    st.stop()

# ======================= 2. CLEANING =======================
st.sidebar.header("🧹 Cleaning")
df = raw_df.copy()
df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

# trim leading/trailing spaces in text columns
for c in df.columns:
    if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c]):
        df[c] = df[c].astype("object").map(lambda v: v.strip() if isinstance(v, str) else v)

if st.sidebar.checkbox("Remove duplicate rows", True):
    df = df.drop_duplicates()

dayfirst = st.sidebar.checkbox("Dates are in DD-MM-YYYY format", True)
for c in df.columns:
    if any(k in c.lower() for k in ["date", "time"]) and not pd.api.types.is_numeric_dtype(df[c]) \
            and not pd.api.types.is_datetime64_any_dtype(df[c]):
        df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=dayfirst)

opt = st.sidebar.selectbox("How should missing values be handled?",
                           ["Do nothing", "Delete rows", "Numeric = Median, Text = 'Unknown'"])
if opt == "Delete rows":
    df = df.dropna()
elif opt.startswith("Numeric"):
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        elif not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].fillna("Unknown")

num_cols = df.select_dtypes(include=np.number).columns.tolist()
date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
cat_cols = [c for c in df.columns if c not in num_cols + date_cols and not pd.api.types.is_bool_dtype(df[c])]

# ======================= 3. FILTERS (slicers) =======================
st.sidebar.header("🎛️ Filters")
filter_cols = [c for c in cat_cols if df[c].nunique() <= 50][:4]
for c in filter_cols:
    chosen = st.sidebar.multiselect(c, sorted(df[c].dropna().astype(str).unique()))
    if chosen:
        df = df[df[c].astype(str).isin(chosen)]

if date_cols and df[date_cols[0]].notna().any():
    dc = date_cols[0]
    dmin, dmax = df[dc].min().date(), df[dc].max().date()
    if dmin < dmax:
        rng = st.sidebar.date_input(f"{dc} range", (dmin, dmax), min_value=dmin, max_value=dmax)
        if isinstance(rng, (tuple, list)) and len(rng) == 2:
            df = df[df[dc].between(pd.Timestamp(rng[0]), pd.Timestamp(rng[1]) + pd.Timedelta(days=1))]

if df.empty:
    st.warning("No data remains after applying the filters.")
    st.stop()

mapping = mapping_ui(df)

# ======================= 4. TABS =======================
tabs = st.tabs(["📊 Dashboard", "📋 Data Profile", "📈 Chart Builder", "💡 Insights",
                "📄 Report", "💬 Ask AI", "🗄️ SQL", "📚 PDF Q&A + 🔮 Forecast"])
tab_dash, tab_prof, tab_chart, tab_ins, tab_rep, tab_ai, tab_sql, tab_doc = tabs

# ----- DASHBOARD -----
with tab_dash:
    dash_df = render_dashboard(df, mapping)

# ----- PROFILE -----
with tab_prof:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Duplicates (original file)", int(raw_df.duplicated().sum()))
    c4.metric("Missing cells", int(df.isna().sum().sum()))
    st.subheader("Data preview")
    show_df(df.head(50))
    st.subheader("Missing values")
    miss = df.isna().sum().reset_index()
    miss.columns = ["Column", "Missing"]
    miss["Missing_%"] = (miss["Missing"] / len(df) * 100).round(2)
    show_df(miss)
    st.subheader("Numeric summary")
    if num_cols:
        show_df(df[num_cols].describe().T)
    st.download_button("⬇️ Cleaned data (CSV)", df.to_csv(index=False).encode(), "cleaned_data.csv", "text/csv")

# ----- CHART BUILDER -----
with tab_chart:
    st.subheader("Build your own chart")
    ctype = st.selectbox("Chart type", ["Bar", "Line", "Pie", "Histogram", "Scatter", "Box", "Heatmap (correlation)"])
    if ctype in ["Bar", "Line", "Pie"] and (cat_cols or date_cols) and num_cols:
        x = st.selectbox("Category (X)", cat_cols + date_cols)
        y = st.selectbox("Value (Y)", num_cols)
        agg = st.selectbox("Aggregation", ["sum", "mean", "count", "max", "min"])
        g = df.groupby(x)[y].agg(agg).reset_index()
        if ctype == "Bar":
            fig = px.bar(g.sort_values(y, ascending=False).head(20), x=x, y=y, title=f"{agg} of {y} by {x}")
        elif ctype == "Line":
            fig = px.line(g.sort_values(x), x=x, y=y, markers=True, title=f"{agg} of {y} by {x}")
        else:
            fig = px.pie(g.sort_values(y, ascending=False).head(10), names=x, values=y, title=f"{y} share by {x}")
        show_chart(fig)
    elif ctype == "Histogram" and num_cols:
        y = st.selectbox("Column", num_cols)
        show_chart(px.histogram(df, x=y, nbins=30, title=f"Distribution of {y}"))
    elif ctype == "Scatter" and len(num_cols) >= 2:
        x = st.selectbox("X", num_cols)
        y = st.selectbox("Y", num_cols, index=1)
        color = st.selectbox("Color by", [None] + cat_cols)
        show_chart(px.scatter(df, x=x, y=y, color=color, title=f"{y} vs {x}"))
    elif ctype == "Box" and num_cols:
        y = st.selectbox("Value", num_cols)
        x = st.selectbox("Group by", [None] + cat_cols)
        show_chart(px.box(df, x=x, y=y, title=f"{y} spread"))
    elif ctype.startswith("Heatmap") and len(num_cols) >= 2:
        show_chart(px.imshow(df[num_cols].corr(), text_auto=".2f", title="Correlation"))
    else:
        st.warning("No suitable columns were found for this chart type.")

# ----- INSIGHTS -----
insights = generate_insights(df, mapping)
recs = generate_recommendations(df, mapping)
if not insights and num_cols:   # generic insights when column mapping is unavailable
    for c in num_cols[:5]:
        insights.append(f"**{c}**: total {df[c].sum():,.2f}, average {df[c].mean():,.2f}, "
                        f"max {df[c].max():,.2f}, min {df[c].min():,.2f}")
with tab_ins:
    st.subheader("💡 Auto insights")
    for i in insights:
        st.markdown("- " + i)
    st.subheader("✅ Recommendations")
    for r in recs:
        st.markdown("- " + r)
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs().unstack().sort_values(ascending=False)
        corr = corr[corr < 0.9999].iloc[::2]
        if len(corr):
            (a, b), v = corr.index[0], corr.iloc[0]
            st.markdown(f"- Strongest numeric relationship: **{a}** and **{b}** (correlation {v:.2f}). "
                        "Correlation does not imply causation.")
    for c in num_cols[:5]:
        q1, q3 = df[c].quantile([0.25, 0.75])
        n_out = int(((df[c] < q1 - 1.5 * (q3 - q1)) | (df[c] > q3 + 1.5 * (q3 - q1))).sum())
        if n_out:
            st.markdown(f"- **{c}** contains {n_out} outliers (IQR method).")

# ----- REPORT -----
with tab_rep:
    from report_pdf import build_pdf, make_pdf_charts
    st.subheader("Business Report")
    k = kpis(df, mapping)
    kpi_text = {label: human(v) for label, v in k.items()}
    summary = [f"The dataset contains {len(df):,} rows and {len(df.columns)} columns after cleaning.",
               f"The original file contained {int(raw_df.duplicated().sum())} duplicate rows. "
               f"Remaining missing cells: {int(df.isna().sum().sum())}."]
    md = ["# Data Analysis Report", f"*Generated: {datetime.now():%d %b %Y, %H:%M}*",
          f"**File:** {file_name}", "", "## Executive Summary"] + [f"- {s}" for s in summary]
    if kpi_text:
        md += ["", "## KPIs"] + [f"- **{a}:** {b}" for a, b in kpi_text.items()]
    md += ["", "## Key Insights"] + [f"- {i}" for i in insights]
    md += ["", "## Recommendations"] + [f"- {r}" for r in recs]
    if num_cols:
        md += ["", "## Numeric Summary", df[num_cols].describe().T.round(2).to_markdown()]
    if st.session_state.get("forecast_text"):
        md += ["", "## Forecast", st.session_state["forecast_text"]]
    md += ["", "## Limitations", "- Insights are based solely on the uploaded data.",
           "- Correlation does not imply causation."]
    report_md = "\n".join(md)
    st.markdown(report_md)

    html = ("<html><head><meta charset='utf-8'><style>body{font-family:Arial;max-width:900px;margin:auto}"
            "table{border-collapse:collapse}td,th{border:1px solid #999;padding:4px}</style></head><body>"
            + markdown.markdown(report_md, extensions=["tables"]) + "</body></html>")
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Markdown", report_md, "report.md", "text/markdown")
    d2.download_button("⬇️ HTML", html, "report.html", "text/html")
    try:
        pdf_bytes = build_pdf("Data Analysis Report", file_name, summary, kpi_text,
                              [i for i in insights], recs, make_pdf_charts(df, mapping),
                              {"Numeric Summary": df[num_cols].describe().T.reset_index()} if num_cols else None,
                              st.session_state.get("forecast_text"))
        d3.download_button("⬇️ PDF (with charts)", pdf_bytes, "report.pdf", "application/pdf")
    except Exception as e:
        d3.error(f"Could not generate the PDF: {e}")

# ----- ASK AI -----
has_key = bool(os.getenv("GROQ_API_KEY"))
with tab_ai:
    from llm_agent import answer_question
    st.subheader("💬 Ask questions about your data")
    if not has_key:
        st.warning("GROQ_API_KEY is not set. Add your key to the .env file and restart the app.")
    q = st.text_input("Question (e.g. Top 5 states by Amount)", key="ask_q")
    if st.button("Ask") and q:
        with st.spinner("Analyzing..."):
            try:
                ans, res = answer_question(df, q)
                st.write(ans)
                if res is not None:
                    show_df(res)
                    if res.shape[1] == 2:
                        show_chart(px.bar(res, x=res.columns[0], y=res.columns[1]))
            except Exception as e:
                st.error(f"LLM error: {e}. Your uploaded data is safe.")

# ----- SQL -----
with tab_sql:
    from sql_tools import question_to_sql, run_sql, save_to_db
    st.subheader("🗄️ SQL Assistant (SELECT queries only)")
    if not has_key:
        st.warning("Text-to-SQL requires GROQ_API_KEY. You can also write SQL directly below.")
    try:
        save_to_db(df)
        st.caption("Table name: `sales`")
    except Exception as e:
        st.error(f"Database error: {e}")
    sq = st.text_input("Ask a question about your data", key="sql_q")
    if st.button("Generate and run SQL") and sq:
        try:
            sql = question_to_sql(sq)
            st.code(sql, language="sql")
            show_df(run_sql(sql))
        except Exception as e:
            st.error(f"Query rejected or failed: {e}")
    with st.expander("Or write SQL manually"):
        manual = st.text_area("SQL", 'SELECT State, SUM(Amount) AS total FROM sales GROUP BY State ORDER BY total DESC')
        if st.button("Run query", key="run_manual"):
            try:
                show_df(run_sql(manual))
            except Exception as e:
                st.error(f"Query rejected or failed: {e}")

# ----- PDF Q&A + FORECAST -----
with tab_doc:
    from forecast import monthly_series, run_forecast
    st.subheader("📚 Ask questions about a PDF")
    pdf = st.file_uploader("Upload a company PDF", type=["pdf"], key="pdf")
    if pdf:
        from rag import SimpleRAG, load_pdf_chunks
        try:
            if st.session_state.get("pdf_name") != pdf.name:
                st.session_state.rag = SimpleRAG(load_pdf_chunks(pdf, pdf.name))
                st.session_state.pdf_name = pdf.name
            dq = st.text_input("What would you like to know from the PDF?", key="doc_q")
            if st.button("Search the PDF") and dq:
                ans, sources = st.session_state.rag.answer(dq)
                st.write(ans)
                if sources:
                    st.caption("Sources: " + ", ".join(f"{d} (page {p})" for d, p in sources))
        except Exception as e:
            st.error(f"PDF error: {e}")

    st.divider()
    st.subheader("🔮 Forecast")
    if date_cols and num_cols:
        default_val = num_cols.index(mapping["amount"]) if mapping["amount"] in num_cols else 0
        fd = st.selectbox("Date column", date_cols)
        fv = st.selectbox("Value column", num_cols, index=default_val, key="fv")
        hz = st.slider("Months to forecast", 1, 12, 3)
        if st.button("Run forecast"):
            try:
                series = monthly_series(df, fd, fv)
                r = run_forecast(series, hz)
                hist = series.reset_index(); hist.columns = ["Month", "Value"]; hist["Type"] = "Actual"
                fut = r["future"].reset_index(); fut.columns = ["Month", "Value"]; fut["Type"] = "Forecast"
                show_chart(px.line(pd.concat([hist, fut]), x="Month", y="Value", color="Type", markers=True))
                st.write("**Test-period metrics (model):**", r["model_metrics"])
                st.write("**Baseline (previous-month value):**", r["baseline_metrics"])
                st.caption(r["note"])
                st.session_state["forecast_text"] = (
                    f"Next {hz} months forecast ({fv}): "
                    + ", ".join(f"{d:%b %Y}: {v:,.0f}" for d, v in r["future"].items())
                    + f". Model MAE={r['model_metrics']['MAE']:,.0f}, baseline MAE={r['baseline_metrics']['MAE']:,.0f}. "
                    + r["note"])
                st.success("The forecast has also been added to the Report tab.")
            except Exception as e:
                st.error(str(e))
    else:
        st.info("A forecast requires one date column and one numeric column.")
