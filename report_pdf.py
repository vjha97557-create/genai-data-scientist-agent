from io import BytesIO
from xml.sax.saxutils import escape
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from dashboard import agg_by, month_profit

BLUE, ORANGE = "#118DFF", "#E66C37"


def _png(fig):
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return buf.getvalue()


def make_pdf_charts(df, m):
    """Render the main dashboard charts as images for the PDF report."""
    A, Q, P = m["amount"], m["quantity"], m["profit"]
    charts = []
    if m["state"] and A:
        g = agg_by(df, m["state"], A, 6).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, 3)); ax.barh(g[m["state"]].astype(str), g[A], color=BLUE)
        ax.set_title(f"{A} by {m['state']}"); charts.append((f"{A} by {m['state']}", _png(fig)))
    if m["date"] and P:
        g = month_profit(df, m)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(g["Month"].str[:3], g[P], color=[BLUE if v >= 0 else ORANGE for v in g[P]])
        ax.axhline(0, color="grey", lw=0.8); ax.set_title("Profit by Month")
        charts.append(("Profit by Month", _png(fig)))
    if m["subcategory"] and P:
        g = agg_by(df, m["subcategory"], P, 6).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(g[m["subcategory"]].astype(str), g[P], color=[BLUE if v >= 0 else ORANGE for v in g[P]])
        ax.set_title("Profit by Sub-Category"); charts.append(("Profit by Sub-Category", _png(fig)))
    for key, title in (("category", "Category share"), ("payment", "Payment mode share")):
        if m[key] and (Q or A):
            g = agg_by(df, m[key], Q or A, 8)
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            ax.pie(g[Q or A], labels=g[m[key]].astype(str), autopct="%1.0f%%", wedgeprops=dict(width=0.45))
            ax.set_title(f"{title} ({Q or A})"); charts.append((title, _png(fig)))
    return charts


def _p(text, style):
    return Paragraph(escape(str(text)).replace("**", ""), style)


def build_pdf(title, filename, summary_lines, kpi_dict, insights, recommendations,
              chart_images=None, tables=None, forecast_text=None):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm)
    s = getSampleStyleSheet()
    story = [_p(title, s["Title"]), _p(f"File: {filename}", s["Normal"]), Spacer(1, 12),
             _p("Executive Summary", s["Heading2"])]
    story += [_p(x, s["Normal"]) for x in summary_lines]

    if kpi_dict:
        story += [Spacer(1, 10), _p("KPIs", s["Heading2"])]
        t = Table([list(kpi_dict.keys()), list(kpi_dict.values())])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#118DFF")),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                               ("GRID", (0, 0), (-1, -1), 0.4, colors.grey)]))
        story.append(t)

    story += [Spacer(1, 10), _p("Key Insights", s["Heading2"])]
    story += [_p("• " + i, s["Normal"]) for i in insights]

    if chart_images:
        story += [Spacer(1, 10), _p("Charts", s["Heading2"])]
        for ctitle, png in chart_images:
            story += [Image(BytesIO(png), width=13 * cm, height=6.6 * cm), Spacer(1, 6)]

    for name, tdf in (tables or {}).items():
        data = [list(map(str, tdf.columns))] + tdf.round(2).astype(str).values.tolist()
        t = Table(data[:16], repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                               ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                               ("FONTSIZE", (0, 0), (-1, -1), 7)]))
        story += [Spacer(1, 10), _p(name, s["Heading2"]), t]

    if forecast_text:
        story += [Spacer(1, 10), _p("Forecast", s["Heading2"]), _p(forecast_text, s["Normal"])]
    story += [Spacer(1, 10), _p("Recommendations", s["Heading2"])]
    story += [_p("• " + r, s["Normal"]) for r in recommendations]
    story += [Spacer(1, 10), _p("Limitations", s["Heading2"]),
              _p("Insights are based solely on the uploaded data. Correlation does not imply causation.", s["Normal"])]
    doc.build(story)
    return buf.getvalue()
