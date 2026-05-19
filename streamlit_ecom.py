"""
streamlit_ecom.py  —  KPI Anomaly Root Cause Dashboard
Run with:  streamlit run streamlit_ecom.py
"""

import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# App config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KPI Anomaly Root Cause Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE       = Path(__file__).parent
REPORT_DIR = BASE / "reports_project"

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* App background */
[data-testid="stAppViewContainer"] > .main {
    background-color: #F0F4F8;
}
/* Sidebar dark gradient */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] p {
    color: #CBD5E1 !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #F1F5F9 !important;
}
/* Sidebar radio pills */
[data-testid="stSidebar"] .stRadio > div {
    gap: 4px;
}
[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 8px 12px !important;
    transition: background 0.2s;
    font-size: 13px !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.12) !important;
}
/* Force all radio label text to be visible */
[data-testid="stSidebar"] .stRadio label p,
[data-testid="stSidebar"] .stRadio label span,
[data-testid="stSidebar"] .stRadio label div {
    color: #CBD5E1 !important;
}
/* Selected radio item — highlighted with blue tint */
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: rgba(59,130,246,0.22) !important;
    border-left: 3px solid #3B82F6;
}
[data-testid="stSidebar"] .stRadio label:has(input:checked) p,
[data-testid="stSidebar"] .stRadio label:has(input:checked) span,
[data-testid="stSidebar"] .stRadio label:has(input:checked) div {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}
/* Metric cards */
[data-testid="metric-container"],
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 20px 22px 32px 22px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06) !important;
    overflow: visible !important;
}
/* Metric value — broad selectors to override theme */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div,
[data-testid="stMetricValue"] div,
[data-testid="stMetricValue"] p,
[data-testid="stMetricValue"] span {
    color: #000000 !important;
    font-size: 36px !important;
    font-weight: 800 !important;
    line-height: 1.1 !important;
}
/* Metric label */
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] span {
    color: #334155 !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
/* Metric delta */
[data-testid="stMetricDelta"] svg { display: none; }
[data-testid="stMetricDelta"] > div {
    font-size: 14px !important;
    font-weight: 700 !important;
}
/* Dividers */
hr {
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 18px 0;
}
/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden;
    border: 1px solid #E2E8F0 !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}
/* Caption */
[data-testid="stCaptionContainer"] p {
    color: #64748B !important;
    font-size: 12.5px !important;
}
/* Block container */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2.5rem !important;
}
/* Selectbox container */
[data-testid="stSelectbox"] > div > div {
    border-radius: 8px !important;
    border-color: #CBD5E1 !important;
    background: white !important;
}
/* All selectbox dropdown value text → black (box has white bg) */
[data-testid="stSelectbox"] div[data-baseweb="select"] span,
[data-testid="stSelectbox"] div[data-baseweb="select"] div,
[data-testid="stSelectbox"] [role="combobox"] span,
[data-testid="stSelectbox"] [role="combobox"] div {
    color: #000000 !important;
}
/* Main canvas selectbox label → black */
.main [data-testid="stSelectbox"] > label,
.main [data-testid="stSelectbox"] > label p,
.main [data-testid="stSelectbox"] > label span {
    color: #000000 !important;
    font-weight: 600 !important;
}
/* Sidebar selectbox label → light (readable on dark background) */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > label,
[data-testid="stSidebar"] [data-testid="stSelectbox"] > label p,
[data-testid="stSidebar"] [data-testid="stSelectbox"] > label span {
    color: #CBD5E1 !important;
    font-weight: 600 !important;
}
/* Confidence Guide — transparent markdown container */
.main [data-testid="stMarkdownContainer"] {
    background: transparent !important;
}
/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants & design tokens
# ─────────────────────────────────────────────────────────────────────────────
KPI_LABEL = {
    "purchase_rate":       "Purchase Rate",
    "avg_session_revenue": "Avg Session Revenue",
    "avg_discount":        "Avg Discount",
}

# Plain-English labels for business-facing pages
KPI_LABEL_PLAIN = {
    "purchase_rate":       "Purchase Conversion",
    "avg_session_revenue": "Revenue per Session",
    "avg_discount":        "Average Discount",
}

STMT_COL = {
    "Purchase_Rate_3_Month_Window_Statement":       "purchase_rate",
    "Avg_Session_Revenue_3_Month_Window_Statement": "avg_session_revenue",
    "Avg_Discount_3_Month_Window_Statement":        "avg_discount",
}

GLM_KPIS = {"purchase_rate", "avg_session_revenue", "avg_discount"}

# Unified Plotly chart theme (no xaxis/yaxis — set per chart to avoid kwarg conflicts)
_CHART = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Arial, sans-serif", color="#000000", size=12),
)


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────
def fmt_kpi(kpi: str, val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    v = float(val)
    if kpi == "avg_session_revenue":
        return f"${v:,.2f}"
    elif kpi == "purchase_rate":
        return f"{v * 100:.2f}%"
    else:
        return f"{v:.2f}%"


def fmt_delta(kpi: str, val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    v = float(val)
    if kpi == "avg_session_revenue":
        return f"${v:+,.2f}"
    elif kpi == "purchase_rate":
        return f"{v * 100:+.2f} pp"
    else:
        return f"{v:+.2f} pp"


def confidence_label(pval) -> str:
    if pval is None or (isinstance(pval, float) and np.isnan(pval)):
        return "N/A"
    p = float(pval)
    if p < 0.05:
        return "High Confidence"
    elif p < 0.10:
        return "Moderate"
    else:
        return "Indicative"


def delta_col_for(kpi: str) -> str:
    return "glm_delta" if kpi in GLM_KPIS else "kpi_delta"


_SEG_TYPE_ORDER = ["DEPT", "DEPT_LINE", "DEPT_SUB", "CHANNEL"]


def _seg_order(seg: str) -> tuple:
    if seg.startswith("DEPT_LINE"):
        return (1, seg)
    elif seg.startswith("DEPT_SUB"):
        return (2, seg)
    elif seg.startswith("DEPT "):
        return (0, seg)
    elif seg.startswith("CHANNEL"):
        return (3, seg)
    else:
        return (4, seg)


def severity_label(pct, direction: str) -> str:
    """Return a plain-English severity label for the heatmap (no raw numbers)."""
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return "—"
    a = abs(float(pct))
    icon = "▼" if direction == "Decreased" else "▲"
    if a >= 25:
        word = "Large Drop" if direction == "Decreased" else "Large Rise"
    elif a >= 10:
        word = "Moderate Drop" if direction == "Decreased" else "Moderate Rise"
    else:
        word = "Small Drop" if direction == "Decreased" else "Small Rise"
    return f"{icon} {word}"


# ─────────────────────────────────────────────────────────────────────────────
# UI component helpers
# ─────────────────────────────────────────────────────────────────────────────
def page_header(title: str, subtitle: str, tag: str = ""):
    tag_html = (
        f"<div style='display:inline-block; background:rgba(255,255,255,0.18); "
        f"border-radius:20px; padding:3px 12px; font-size:10px; font-weight:700; "
        f"letter-spacing:2px; color:#BFDBFE; text-transform:uppercase; margin-bottom:10px;'>"
        f"{tag}</div>"
    ) if tag else ""
    st.markdown(f"""
<div style='background: linear-gradient(135deg, #1E3A5F 0%, #1D4ED8 100%);
     padding: 26px 32px; border-radius: 16px; margin-bottom: 24px;
     box-shadow: 0 6px 24px rgba(29,78,216,0.22);'>
  {tag_html}
  <div style='font-size:27px; font-weight:700; color:white; line-height:1.25;
              font-family:Arial,sans-serif; letter-spacing:-0.3px;'>{title}</div>
  <div style='font-size:13px; color:#BFDBFE; margin-top:7px;
              font-family:Arial,sans-serif; font-weight:400;'>{subtitle}</div>
</div>
""", unsafe_allow_html=True)


def section_title(text: str):
    st.markdown(f"""
<div style='margin: 24px 0 12px; padding-bottom: 10px;
            border-bottom: 2.5px solid #3B82F6;'>
  <span style='font-size:20px; font-weight:700; color:#1E3A5F;
               font-family:Arial,sans-serif; letter-spacing:-0.3px;'>{text}</span>
</div>
""", unsafe_allow_html=True)


def sidebar_section(text: str):
    st.sidebar.markdown(
        f"<div style='font-size:11px; font-weight:700; color:#64748B; "
        f"text-transform:uppercase; letter-spacing:1.2px; margin:14px 0 8px;'>{text}</div>",
        unsafe_allow_html=True,
    )


def info_box(text: str):
    st.markdown(
        f"<div style='background:#EFF6FF; border-left:4px solid #3B82F6; "
        f"border-radius:8px; padding:12px 16px; margin-bottom:16px; "
        f"font-size:13px; color:#1E3A5F; font-family:Arial,sans-serif;'>{text}</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    contrib = pd.read_csv(REPORT_DIR / "rca_contributions_ecom.csv")
    summary = pd.read_csv(REPORT_DIR / "rca_summary_ecom.csv")
    anomaly = pd.read_csv(BASE / "ecom_anomalies.csv")
    parent = (
        contrib.groupby(["segment", "anomalous_kpi"])
        .agg(
            parent_cur=("parent_kpi_current", "first"),
            parent_pri=("parent_kpi_prior",   "first"),
        )
        .reset_index()
    )
    parent["parent_pct"] = (
        (parent["parent_cur"] - parent["parent_pri"]) / parent["parent_pri"].abs() * 100
    )
    return contrib, summary, anomaly, parent


contrib_df, summary_df, anomaly_df, parent_df = load_data()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='margin: 16px 0 6px;'>
  <div style='font-size:20px; font-weight:800; color:#F1F5F9;
              font-family:Arial,sans-serif; line-height:1.3; letter-spacing:-0.3px;'>
    KPI Anomaly Dashboard
  </div>
  <div style='font-size:11px; color:#64748B; margin-top:5px; letter-spacing:0.2px; line-height:1.9;'>
    6-month analysis<br>
    <span style='color:#94A3B8;'>Prior period:</span> Jun–Aug 2024<br>
    <span style='color:#94A3B8;'>Current period:</span> Sep–Nov 2024
  </div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown(
    "<hr style='border:none; border-top:1px solid #334155; margin:12px 0;'>",
    unsafe_allow_html=True,
)

sidebar_section("Navigate")
page = st.sidebar.radio(
    "",
    ["Anomaly Overview", "Root Cause Analysis", "Technical Detail"],
    label_visibility="collapsed",
)

st.sidebar.markdown(
    "<hr style='border:none; border-top:1px solid #334155; margin:12px 0;'>",
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Anomaly Overview
# ═════════════════════════════════════════════════════════════════════════════
if page == "Anomaly Overview":

    page_header(
        "What Happened?",
        "A quick overview of which business areas improved and which declined.",
        tag="Overview",
    )

    info_box(
        "<div style='display:flex; gap:40px; flex-wrap:wrap; align-items:center;'>"
        "<span>��&nbsp;&nbsp;<b>Red</b> &mdash; a KPI declined</span>"
        "<span>��&nbsp;&nbsp;<b>Green</b> &mdash; a KPI improved</span>"
        "<span>&mdash;&nbsp;&nbsp;No significant change detected</span>"
        "</div>"
    )

    sidebar_section("Filters")
    _avail_types  = set(s.split()[0] for s in summary_df["segment"].unique())
    seg_type_opts = ["All"] + [t for t in _SEG_TYPE_ORDER if t in _avail_types] + \
                   sorted(t for t in _avail_types if t not in _SEG_TYPE_ORDER)
    seg_type      = st.sidebar.selectbox("Segment type", seg_type_opts)
    dir_opts      = ["All", "Decreased", "Increased"]
    dir_sel       = st.sidebar.selectbox("Direction", dir_opts)

    view = summary_df.copy()
    if seg_type != "All":
        view = view[view["segment"].str.startswith(seg_type + " ")]
    if dir_sel != "All":
        view = view[view["direction"] == dir_sel]

    view = (
        view.sort_values("dimension", ascending=True)
            .drop_duplicates(subset=["segment", "anomalous_kpi"], keep="last")
    )
    view = view.merge(
        parent_df[["segment", "anomalous_kpi", "parent_pct"]],
        on=["segment", "anomalous_kpi"], how="left",
    )

    all_segs = sorted(view["segment"].unique(), key=_seg_order)
    all_kpis = list(KPI_LABEL.keys())

    cell_lookup = {
        (r["segment"], r["anomalous_kpi"]): r
        for _, r in view.iterrows()
    }

    section_title("Business Health Overview")
    st.caption(
        "Each cell shows whether that metric improved or declined for that business area.  "
        "The smaller text names the biggest driver of the change."
    )

    header_cells = (
        "<th style='padding:10px 16px; text-align:left; color:#F1F5F9; "
        "font-size:12px; font-weight:600; letter-spacing:0.4px;'>Business Area</th>"
        + "".join(
            f"<th style='padding:10px 16px; text-align:center; color:#F1F5F9; "
            f"font-size:12px; font-weight:600; letter-spacing:0.4px;'>"
            f"{KPI_LABEL_PLAIN.get(k, KPI_LABEL[k])}</th>"
            for k in all_kpis
        )
    )

    rows_html = []
    for i, seg in enumerate(all_segs):
        row_bg = "rgba(255,255,255,0.03)" if i % 2 == 0 else "transparent"
        cells = [
            f"<td style='padding:9px 16px; font-weight:600; white-space:nowrap; "
            f"color:#ffffff; font-size:12px; background:{row_bg};'>{seg}</td>"
        ]
        for kpi in all_kpis:
            data = cell_lookup.get((seg, kpi))
            if data is not None:
                pct  = data.get("parent_pct")
                dirn = data.get("direction", "")
                top  = data.get("top_contributor", "")
                bg   = "#DC2626" if dirn == "Decreased" else "#16A34A"
                sev  = severity_label(pct, dirn)
                cells.append(
                    f"<td style='background:{bg}; padding:9px 16px; text-align:center; "
                    f"vertical-align:middle; border-radius:0;'>"
                    f"<div style='font-weight:700; color:#ffffff; font-size:13px;'>{sev}</div>"
                    f"<div style='font-size:10px; color:rgba(255,255,255,0.82); margin-top:2px;'>"
                    f"Key driver: {top}</div>"
                    f"</td>"
                )
            else:
                cells.append(
                    "<td style='background:#EEF2F7; padding:9px 16px; text-align:center; "
                    "color:#94A3B8; font-size:13px;'>—</td>"
                )
        rows_html.append(
            f"<tr style='border-bottom:1px solid rgba(255,255,255,0.06);'>"
            + "".join(cells) + "</tr>"
        )

    table_html = f"""
<div style='overflow-x:auto; border-radius:14px; box-shadow:0 4px 16px rgba(0,0,0,0.10);
            border:1px solid #334155; margin-bottom:6px;'>
  <table style='border-collapse:collapse; font-size:13px; width:100%;
                font-family:Arial,sans-serif; background:#1E293B;'>
    <thead>
      <tr style='background:linear-gradient(90deg,#0F172A,#1E3A5F);
                 border-bottom:2px solid #334155;'>{header_cells}</tr>
    </thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>
</div>
"""

    col_hm, col_stmts = st.columns([5, 2])

    with col_hm:
        st.markdown(table_html, unsafe_allow_html=True)

        # Severity legend
        st.markdown("""
<div style='margin-top:10px; font-size:11px; color:#64748B; font-family:Arial,sans-serif;'>
  <b>Severity guide:</b>&nbsp;
  <span style='color:#DC2626; font-weight:700;'>▼ Large Drop</span> = &gt;25% decline &nbsp;·&nbsp;
  <span style='color:#DC2626;'>▼ Moderate Drop</span> = 10–25% decline &nbsp;·&nbsp;
  <span style='color:#DC2626;'>▼ Small Drop</span> = &lt;10% decline &nbsp;·&nbsp;
  <span style='color:#16A34A; font-weight:700;'>▲ Large Rise</span> = &gt;25% increase
</div>
""", unsafe_allow_html=True)

    with col_stmts:
        section_title("What the Data Says")
        seg_opts     = sorted(anomaly_df["Segment"].unique(), key=_seg_order)
        sel_stmt_seg = st.selectbox("Select a business area", seg_opts, key="stmt_seg")

        seg_row   = anomaly_df[anomaly_df["Segment"] == sel_stmt_seg].iloc[0]
        found_any = False
        for col, kpi in STMT_COL.items():
            val = str(seg_row.get(col, "")).strip()
            if val.lower() not in {"nan", "none", "null", ""}:
                found_any = True
                dirn  = "Decreased" if any(w in val.lower() for w in ("decreased", "fell", "dropped")) else "Increased"
                color = "error" if dirn == "Decreased" else "success"
                getattr(st, color)(f"**{KPI_LABEL_PLAIN.get(kpi, KPI_LABEL.get(kpi, kpi))}:** {val}")
        if not found_any:
            st.info("No significant changes detected for this area.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Root Cause Analysis
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Root Cause Analysis":

    page_header(
        "Why Did It Happen?",
        "Trace which product lines and marketing channels drove each change.",
        tag="Root Cause",
    )

    # ── Sidebar controls ──────────────────────────────────────────────────────
    dept_segs = sorted(
        s for s in summary_df["segment"].unique()
        if s.startswith("DEPT ") and not s.startswith("DEPT_")
    )

    if not dept_segs:
        st.info("No department-level segments found in the data.")
        st.stop()

    sidebar_section("Business Area")
    sel_seg = st.sidebar.selectbox("", dept_segs, label_visibility="collapsed", key="why_seg")

    # All anomalous KPIs for this DEPT
    seg_kpis_df = (
        summary_df[summary_df["segment"] == sel_seg]
        .sort_values("dimension")
        .drop_duplicates(subset=["anomalous_kpi"], keep="last")
        .merge(
            parent_df[["segment", "anomalous_kpi", "parent_pct"]],
            on=["segment", "anomalous_kpi"], how="left",
        )
    )
    seg_kpis_df = seg_kpis_df[seg_kpis_df["anomalous_kpi"].isin(KPI_LABEL)]

    available_kpis = seg_kpis_df["anomalous_kpi"].tolist()
    kpi_options = [k for k in ["purchase_rate", "avg_session_revenue", "avg_discount"] if k in available_kpis]
    if not kpi_options:
        kpi_options = available_kpis

    sidebar_section("KPI to Investigate")
    sel_kpi_why = st.sidebar.selectbox(
        "",
        kpi_options,
        format_func=lambda k: KPI_LABEL_PLAIN.get(k, KPI_LABEL.get(k, k)),
        label_visibility="collapsed",
        key="why_kpi",
    )

    # ── Helper functions ──────────────────────────────────────────────────────
    def _nid(*parts):
        return re.sub(r"[^a-zA-Z0-9]", "_", "__".join(str(p) for p in parts))

    def _esc(s):
        return str(s).replace("\\", "\\\\").replace('"', '\\"')

    def _top1(seg, kpi, dim):
        rows = (
            contrib_df[
                (contrib_df["segment"]       == seg) &
                (contrib_df["anomalous_kpi"] == kpi) &
                (contrib_df["dimension"]     == dim)
            ]
            .dropna(subset=["glm_delta"])
            .sort_values("glm_delta", key=abs, ascending=False)
        )
        return str(rows.iloc[0]["sub_segment"]) if not rows.empty else None

    # ── Causal Chain (Knowledge Graph) ───────────────────────────────────────
    section_title("Causal Chain")
    info_box(
        "Follow the arrows from left to right: the current period triggered changes in "
        f"<b>{sel_seg}</b>, which affected the KPIs shown. Each KPI's top contributing "
        "product line and marketing channel are shown at the far right."
    )

    if seg_kpis_df.empty:
        st.info(f"No significant changes detected for {sel_seg}.")
        st.stop()

    seg_id = _nid("seg", sel_seg)

    dot = [
        "digraph G {",
        "  rankdir=LR;",
        "  graph [nodesep=0.7, ranksep=1.7];",
        '  node [shape=box, style="rounded,filled", fontname="Arial",'
        '        fontsize=11, margin="0.2,0.14"];',
        '  edge [color="#94A3B8", arrowsize=0.8, penwidth=1.3];',
        "",
        '  n_prior [label="Prior Window\\nJun-Aug 2024",'
        '           fillcolor="#334155", fontcolor="white", color="#334155"];',
        '  n_cur   [label="Current Window\\nSep-Nov 2024",'
        '           fillcolor="#1E3A5F", fontcolor="white", color="#1E3A5F"];',
        '  n_prior -> n_cur [label="3-month rolling shift", fontsize=9,'
        '             fontcolor="#64748B", style=dashed, color="#94A3B8"];',
        "",
        f'  {seg_id} [label="{_esc(sel_seg)}",'
        f'             fillcolor="#1D4ED8", fontcolor="white", color="#1E3A8A"];',
        f"  n_cur -> {seg_id};",
        "",
    ]

    for _, row in seg_kpis_df.iterrows():
        kpi      = row["anomalous_kpi"]
        dirn     = str(row.get("direction", ""))
        pct      = row.get("parent_pct", None)
        kpi_text = KPI_LABEL_PLAIN.get(kpi, KPI_LABEL.get(kpi, kpi))

        chg_fill = (
            "#DC2626" if dirn == "Decreased" else
            "#16A34A" if dirn == "Increased" else
            "#64748B"
        )
        pct_str = f"{pct:+.1f}%" if (pct is not None and pd.notna(pct)) else "N/A"
        chg_id  = _nid("chg", sel_seg, kpi)

        dot.append(
            f'  {chg_id} [label="{_esc(kpi_text)}: {pct_str}",'
            f'             fillcolor="{chg_fill}", fontcolor="white", color="{chg_fill}"];'
        )
        dot.append(f"  {seg_id} -> {chg_id};")

        # Top 1 Product Line contributor
        dl    = _top1(sel_seg, kpi, "DEPT_LINE")
        dl_id = _nid("dl", sel_seg, kpi)
        if dl:
            dot.append(
                f'  {dl_id} [label="Product Line:\\n{_esc(dl)}",'
                f'            fillcolor="#EFF6FF", fontcolor="#1E293B", color="#93C5FD"];'
            )
        else:
            dot.append(
                f'  {dl_id} [label="Product Line:\\nno data",'
                f'            fillcolor="#F8FAFC", fontcolor="#94A3B8", color="#E2E8F0"];'
            )
        dot.append(f"  {chg_id} -> {dl_id};")

        # Top 1 Marketing Channel contributor
        ch    = _top1(sel_seg, kpi, "CHANNEL")
        ch_id = _nid("ch", sel_seg, kpi)
        if ch:
            dot.append(
                f'  {ch_id} [label="Channel:\\n{_esc(ch)}",'
                f'            fillcolor="#F0FDF4", fontcolor="#1E293B", color="#86EFAC"];'
            )
        else:
            dot.append(
                f'  {ch_id} [label="Channel:\\nno data",'
                f'            fillcolor="#F8FAFC", fontcolor="#94A3B8", color="#E2E8F0"];'
            )
        dot.append(f"  {chg_id} -> {ch_id};")
        dot.append("")

    dot.append("}")
    st.graphviz_chart("\n".join(dot), use_container_width=True)

    st.markdown("---")

    # ── Ranked impact charts ──────────────────────────────────────────────────
    section_title(
        f"Ranked Impact on {KPI_LABEL_PLAIN.get(sel_kpi_why, sel_kpi_why)}"
    )
    st.markdown(
        f"<p style='color:#64748B; font-size:13px; margin-bottom:16px;'>"
        f"Which product lines and marketing channels contributed most to the change in "
        f"<b>{KPI_LABEL_PLAIN.get(sel_kpi_why, sel_kpi_why).lower()}</b>? "
        f"Bars pointing left = negative impact (decline), bars pointing right = positive impact (growth).</p>",
        unsafe_allow_html=True,
    )

    dl_data = (
        contrib_df[
            (contrib_df["segment"]       == sel_seg) &
            (contrib_df["anomalous_kpi"] == sel_kpi_why) &
            (contrib_df["dimension"]     == "DEPT_LINE")
        ]
        .dropna(subset=["glm_delta"])
        .sort_values("glm_delta", key=abs, ascending=True)
    )

    if not dl_data.empty:
        dl_data  = dl_data.tail(8)
        dl_vals  = dl_data["glm_delta"].tolist()
        dl_lbls  = [fmt_delta(sel_kpi_why, v) for v in dl_vals]
        dl_clrs  = ["#EF4444" if v < 0 else "#22C55E" for v in dl_vals]

        fig_dl = go.Figure(go.Bar(
            x=dl_vals,
            y=dl_data["sub_segment"],
            orientation="h",
            marker_color=dl_clrs,
            marker_line_width=0,
            text=dl_lbls,
            textposition="outside",
            textfont=dict(color="#1E293B", size=12, family="Arial, sans-serif"),
            hovertemplate="<b>%{y}</b><br>Contribution: %{x:.5f}<extra></extra>",
        ))
        fig_dl.update_layout(
            **_CHART,
            title=dict(
                text="Product Lines",
                font=dict(size=14, color="#1E3A5F", family="Arial, sans-serif"),
                x=0,
            ),
            xaxis_title="Contribution to Change",
            yaxis_title="",
            height=min(160 + len(dl_data) * 56, 540),
            margin=dict(l=10, r=220, t=40, b=50),
            xaxis=dict(
                zeroline=True, zerolinecolor="#94A3B8", zerolinewidth=1.5,
                gridcolor="#F1F5F9", linecolor="#E2E8F0",
                tickfont=dict(color="#000000", size=11),
                title_font=dict(color="#000000", size=12),
            ),
            yaxis=dict(
                gridcolor="#F1F5F9", linecolor="#E2E8F0",
                tickfont=dict(color="#000000", size=13, family="Arial, sans-serif"),
                title_font=dict(color="#000000", size=12),
            ),
        )
        st.plotly_chart(fig_dl, use_container_width=True)
    else:
        st.info("No product line data for this selection.")

    ch_data = (
        contrib_df[
            (contrib_df["segment"]       == sel_seg) &
            (contrib_df["anomalous_kpi"] == sel_kpi_why) &
            (contrib_df["dimension"]     == "CHANNEL")
        ]
        .dropna(subset=["glm_delta"])
        .sort_values("glm_delta", key=abs, ascending=True)
    )

    if not ch_data.empty:
        ch_data  = ch_data.tail(8)
        ch_vals  = ch_data["glm_delta"].tolist()
        ch_lbls  = [fmt_delta(sel_kpi_why, v) for v in ch_vals]
        ch_clrs  = ["#EF4444" if v < 0 else "#22C55E" for v in ch_vals]

        fig_ch = go.Figure(go.Bar(
            x=ch_vals,
            y=ch_data["sub_segment"],
            orientation="h",
            marker_color=ch_clrs,
            marker_line_width=0,
            text=ch_lbls,
            textposition="outside",
            textfont=dict(color="#1E293B", size=12, family="Arial, sans-serif"),
            hovertemplate="<b>%{y}</b><br>Contribution: %{x:.5f}<extra></extra>",
        ))
        fig_ch.update_layout(
            **_CHART,
            title=dict(
                text="Marketing Channels",
                font=dict(size=14, color="#1E3A5F", family="Arial, sans-serif"),
                x=0,
            ),
            xaxis_title="Contribution to Change",
            yaxis_title="",
            height=min(160 + len(ch_data) * 56, 540),
            margin=dict(l=10, r=220, t=40, b=50),
            xaxis=dict(
                zeroline=True, zerolinecolor="#94A3B8", zerolinewidth=1.5,
                gridcolor="#F1F5F9", linecolor="#E2E8F0",
                tickfont=dict(color="#000000", size=11),
                title_font=dict(color="#000000", size=12),
            ),
            yaxis=dict(
                gridcolor="#F1F5F9", linecolor="#E2E8F0",
                tickfont=dict(color="#000000", size=13, family="Arial, sans-serif"),
                title_font=dict(color="#000000", size=12),
            ),
        )
        st.plotly_chart(fig_ch, use_container_width=True)
    else:
        st.info("No channel data for this selection.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TECHNICAL DETAIL
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Technical Detail":

    page_header(
        "Technical Detail",
        "Statistical drill-down for academic review — GLM attribution and confidence levels.",
        tag="Statistical",
    )

    sidebar_section("Selection")
    all_segs = sorted(summary_df["segment"].unique(), key=_seg_order)
    sel_seg  = st.sidebar.selectbox("Segment", all_segs, key="td_seg")

    seg_kpis = sorted(
        k for k in summary_df[summary_df["segment"] == sel_seg]["anomalous_kpi"].unique()
        if k not in ("revenue_share", "traffic_share")
    )
    sel_kpi = st.sidebar.selectbox(
        "KPI", seg_kpis, format_func=lambda k: KPI_LABEL.get(k, k), key="td_kpi"
    )
    sidebar_section("Breakdown Dimension")
    dim_choice = st.sidebar.radio("", ["DEPT_LINE", "CHANNEL"], label_visibility="collapsed", key="td_dim")

    seg_data = contrib_df[
        (contrib_df["segment"]       == sel_seg) &
        (contrib_df["anomalous_kpi"] == sel_kpi) &
        (contrib_df["dimension"]     == dim_choice)
    ].copy()

    if not seg_data.empty:
        p_cur   = seg_data.iloc[0]["parent_kpi_current"]
        p_pri   = seg_data.iloc[0]["parent_kpi_prior"]
        p_delta = seg_data.iloc[0]["parent_kpi_delta"]
        p_pct   = (p_delta / abs(p_pri) * 100) if (p_pri and not np.isnan(p_pri)) else 0

        c1, c2, c3 = st.columns(3)
        _d_color = "#EF4444" if p_delta < 0 else "#22C55E"
        _d_arrow = "▼" if p_delta < 0 else "▲"
        _d_str   = fmt_delta(sel_kpi, p_delta)
        _pct_color = "#EF4444" if p_pct < 0 else "#22C55E"

        _card = (
            "<div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px;"
            "padding:20px 22px; box-shadow:0 2px 10px rgba(0,0,0,0.06); min-height:110px;'>"
            "<div style='font-size:12px; font-weight:700; color:#334155;"
            "text-transform:uppercase; letter-spacing:0.6px; margin-bottom:8px;{extra_label}'>"
            "{label}{delta_inline}</div>"
            "<div style='font-size:36px; font-weight:800; color:{val_color}; line-height:1.1;"
            "font-family:Arial,sans-serif;'>{value}</div></div>"
        )

        c1.markdown(
            _card.format(
                label="Prior Period", delta_inline="", extra_label="",
                value=fmt_kpi(sel_kpi, p_pri), val_color="#000000",
            ),
            unsafe_allow_html=True,
        )
        c2.markdown(
            _card.format(
                label="Current Period",
                delta_inline=(
                    f"&nbsp;&nbsp;<span style='font-size:13px; font-weight:700;"
                    f"color:{_d_color};'>{_d_arrow} {_d_str}</span>"
                ),
                extra_label="",
                value=fmt_kpi(sel_kpi, p_cur), val_color="#000000",
            ),
            unsafe_allow_html=True,
        )
        c3.markdown(
            _card.format(
                label="% Change", delta_inline="", extra_label="",
                value=f"{p_pct:+.1f}%", val_color=_pct_color,
            ),
            unsafe_allow_html=True,
        )
    else:
        st.warning("No data found for this segment / KPI / dimension combination.")
        st.stop()

    st.markdown("---")

    d_col    = delta_col_for(sel_kpi)
    use_glm  = d_col == "glm_delta" and seg_data["glm_delta"].notna().any()
    rank_col = "glm_delta" if use_glm else "kpi_delta"

    plot_df = (
        seg_data[["sub_segment", "kpi_current", "kpi_prior",
                  "kpi_delta", "glm_delta", "sessions_current",
                  "sessions_prior", "glm_pvalue"]]
        .dropna(subset=[rank_col])
        .sort_values(rank_col, key=abs, ascending=True)
    )

    bar_vals   = plot_df[rank_col].tolist()
    bar_labels = [fmt_delta(sel_kpi, v) for v in bar_vals]
    bar_colors = ["#EF4444" if v < 0 else "#22C55E" for v in bar_vals]

    section_title(
        f"Top Contributors — {KPI_LABEL.get(sel_kpi)}  ·  {sel_seg}  ·  by {dim_choice}"
    )

    fig = go.Figure(go.Bar(
        x=bar_vals,
        y=plot_df["sub_segment"],
        orientation="h",
        marker_color=bar_colors,
        marker_line_width=0,
        text=bar_labels,
        textposition="outside",
        textfont=dict(color="#1E293B", size=12, family="Arial, sans-serif"),
        hovertemplate="<b>%{y}</b><br>Change: %{x:.5f}<extra></extra>",
    ))
    fig.update_layout(
        **_CHART,
        xaxis_title="GLM Period Delta" if use_glm else "Raw Delta",
        yaxis_title="",
        height=min(150 + len(plot_df) * 52, 500),
        margin=dict(l=10, r=220, t=20, b=40),
        xaxis=dict(
            zeroline=True, zerolinecolor="#94A3B8", zerolinewidth=1.5,
            gridcolor="#F1F5F9", linecolor="#E2E8F0",
            tickfont=dict(color="#000000", size=11),
            title_font=dict(color="#000000", size=12),
        ),
        yaxis=dict(
            gridcolor="#F1F5F9", linecolor="#E2E8F0",
            tickfont=dict(color="#000000", size=12, family="Arial, sans-serif"),
            title_font=dict(color="#000000", size=12),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    section_title("Sub-Segment Detail Table")

    detail = seg_data[["sub_segment", "kpi_prior", "kpi_current",
                        rank_col, "kpi_pct_change",
                        "sessions_prior", "sessions_current",
                        "glm_pvalue"]].copy()

    detail["Prior"]            = detail["kpi_prior"].apply(lambda v: fmt_kpi(sel_kpi, v))
    detail["Current"]          = detail["kpi_current"].apply(lambda v: fmt_kpi(sel_kpi, v))
    detail["Change"]           = detail[rank_col].apply(lambda v: fmt_delta(sel_kpi, v))
    detail["% Change"]         = detail["kpi_pct_change"].apply(
                                     lambda v: f"{v:+.1f}%" if pd.notna(v) else "—")
    detail["Sessions (Prior)"] = detail["sessions_prior"].apply(
                                     lambda v: f"{int(v):,}" if pd.notna(v) else "—")
    detail["Sessions (Now)"]   = detail["sessions_current"].apply(
                                     lambda v: f"{int(v):,}" if pd.notna(v) else "—")
    detail["Confidence"]       = detail["glm_pvalue"].apply(confidence_label)

    display_cols = ["sub_segment", "Prior", "Current", "Change", "% Change",
                    "Sessions (Prior)", "Sessions (Now)", "Confidence"]
    detail_out = detail[display_cols].rename(columns={"sub_segment": "Sub-Segment"})

    _CONF_COLORS = {
        "High Confidence": "#16A34A",
        "Moderate":        "#D97706",
        "Indicative":      "#EF4444",
        "N/A":             "#3B82F6",
    }

    def _style_conf(s):
        return [
            f"color: {_CONF_COLORS.get(v, '#1E293B')}; font-weight: 700;"
            for v in s
        ]

    styled_detail = (
        detail_out.reset_index(drop=True)
        .style.apply(_style_conf, subset=["Confidence"])
    )
    st.dataframe(styled_detail, use_container_width=True, hide_index=True)

    st.markdown("""
<div style='padding:12px 0 4px; margin-top:10px;
            border-top:2px solid #E2E8F0;'>
  <div style='font-size:11px; font-weight:700; color:#1E3A5F; margin-bottom:8px;
              text-transform:uppercase; letter-spacing:0.8px;'>Confidence Guide</div>
  <table style='border-collapse:collapse; font-size:11px; width:100%;'>
  <tbody>
  <tr>
    <td style='padding:4px 10px; color:#16A34A; font-weight:700; width:140px;'>High Confidence</td>
    <td style='padding:4px 10px; color:#334155; width:80px;'>p &lt; 0.05</td>
    <td style='padding:4px 10px; color:#334155;'>Statistically confirmed shift</td>
  </tr>
  <tr>
    <td style='padding:4px 10px; color:#D97706; font-weight:700;'>Moderate</td>
    <td style='padding:4px 10px; color:#334155;'>p &lt; 0.10</td>
    <td style='padding:4px 10px; color:#334155;'>Likely real but needs more data</td>
  </tr>
  <tr>
    <td style='padding:4px 10px; color:#EF4444; font-weight:700;'>Indicative</td>
    <td style='padding:4px 10px; color:#334155;'>p ≥ 0.10</td>
    <td style='padding:4px 10px; color:#334155;'>Directional lead; small sample</td>
  </tr>
  <tr>
    <td style='padding:4px 10px; color:#64748B; font-weight:700;'>N/A</td>
    <td style='padding:4px 10px; color:#334155;'>—</td>
    <td style='padding:4px 10px; color:#334155;'>Mix metric, no GLM applied</td>
  </tr>
  </tbody>
  </table>
</div>
""", unsafe_allow_html=True)
