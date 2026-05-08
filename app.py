"""
Chemelex Demand Forecast Cockpit - UI-aligned Streamlit mockup
----------------------------------------------------------------
Designed to match the Chemelex screenshots shared by the user:
1) Material Forecast page
2) Forecast Analysis page

Input expected from Synthefy / Excel:
- Year
- Month
- Supply_Chain_Zone
- Plant Name
- Product 1 Category/Division Name
- Sum of Order Value in USD from LC
- Sum of Order quantity BU

The app also works without upload by generating realistic demo data.
"""

from __future__ import annotations

import io
import math
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Streamlit page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Chemelex | Demand Forecast Cockpit",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
MONTH_SHORT = {m: m[:3] for m in MONTH_ORDER}
MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTH_ORDER)}
TODAY_LABEL = "18/11/2023 14:23:51"
DEFAULT_SELECTED_MONTH = "January, 2025"

REQUIRED_COLUMNS = [
    "Year",
    "Month",
    "Supply_Chain_Zone",
    "Plant Name",
    "Product 1 Category/Division Name",
    "Sum of Order Value in USD from LC",
    "Sum of Order quantity BU",
]

CHEMELEX_BLUE = "#0076ad"
CHEMELEX_GREEN = "#2eb35a"
CHEMELEX_ORANGE = "#f28c00"
CHEMELEX_RED = "#ff625b"
GRID = "rgba(0,0,0,.075)"

# -----------------------------------------------------------------------------
# CSS to mimic the screenshot style
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {
    padding-top: 0rem !important;
    padding-left: 0.25rem !important;
    padding-right: 0.25rem !important;
    max-width: 100% !important;
}
html, body, [data-testid="stAppViewContainer"] {
    background: #fafafa;
    color: #4b5563;
    font-family: Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
}
[data-testid="stVerticalBlock"] { gap: 0.65rem !important; }

.chem-topbar {
    height: 42px;
    background: #fff;
    border-bottom: 1px solid #ddd;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 28px 0 32px;
}
.chem-logo-small {
    font-size: 12px;
    color: #6b7280;
    font-style: italic;
    margin-right: 5px;
}
.chem-logo {
    font-size: 30px;
    color: #3a7890;
    font-weight: 900;
    font-style: italic;
    letter-spacing: -1.2px;
}
.chem-icons {
    color: #333;
    font-size: 20px;
    display: flex;
    gap: 14px;
    align-items: center;
}
.notif-dot {
    background: #e3342f;
    color: white;
    border-radius: 999px;
    font-size: 10px;
    padding: 0 5px;
    position: relative;
    left: -10px;
    top: -8px;
}
.chem-tabs {
    height: 40px;
    background: #fff;
    border-bottom: 1px solid #e5e7eb;
    display: flex;
    align-items: center;
    gap: 10px;
    padding-left: 22px;
}
.chem-tab {
    border-radius: 999px;
    border: 1px solid transparent;
    padding: 7px 17px;
    font-size: 13px;
    color: #333;
    cursor: default;
}
.chem-tab.active {
    border-color: #0087c7;
    color: #0a5f88;
    background: #f0f9ff;
}
.chem-strip {
    height: 24px;
    background: #fff4e4;
    border-bottom: 1px solid #f5dfbf;
    box-shadow: 0 2px 6px rgba(0,0,0,.08);
    font-size: 12px;
    color: #333;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 13px 0 7px;
    border-left: 4px solid #f28c00;
}
.strip-icons { color: #eb8500; font-weight: 800; }
.page-wrap { padding: 22px 22px 0 22px; }
.panel {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    box-shadow: 0 1px 5px rgba(0,0,0,.04);
    padding: 14px;
}
.page-titlebar {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    box-shadow: 0 1px 6px rgba(0,0,0,.05);
    padding: 13px 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 18px;
}
.page-title {
    font-size: 19px;
    font-weight: 600;
    color: #555;
}
.filters-line {
    display: flex;
    gap: 14px;
    align-items: center;
    flex-wrap: wrap;
}
.fake-select {
    min-width: 155px;
    background: #fff;
    border: 1px solid #e5e7eb;
    color: #555;
    border-radius: 999px;
    padding: 7px 12px;
    font-size: 12px;
}
.month-nav {
    display:flex;
    align-items:center;
    gap:8px;
    background:#f8fbfd;
    border: 1px solid #e5e7eb;
    border-radius:999px;
    padding: 3px 8px;
    font-size:12px;
    color:#555;
}
.nav-btn {
    background:#0087c7;
    color:white;
    width:30px;
    height:30px;
    display:inline-grid;
    place-items:center;
    border-radius:999px;
    font-weight:900;
    box-shadow: 0 2px 5px rgba(0,118,173,.2);
}
.download-btn {
    background:#0087c7;
    color:white;
    width:34px;
    height:34px;
    display:inline-grid;
    place-items:center;
    border-radius:999px;
    font-weight:900;
    box-shadow: 0 2px 6px rgba(0,118,173,.35);
}
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 18px;
    margin-bottom: 18px;
}
.kpi-card {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 3px;
    min-height: 44px;
    padding: 8px 10px;
}
.kpi-label { font-size: 12px; color:#777; margin-bottom:2px; }
.kpi-value { font-size: 18px; color:#4b5563; font-weight: 600; line-height: 1.05; }
.kpi-sub { font-size: 12px; color:#16a34a; float:right; margin-top:-16px; }
.card-title { font-size: 13px; color:#666; font-weight: 700; margin-bottom: 8px; }
.small-muted { font-size: 11px; color:#9ca3af; }
.var-badge {
    background: #ffd6d6;
    color:#7c5252;
    border-radius: 5px;
    padding: 7px 14px;
    font-size: 17px;
    display:inline-block;
    float:right;
}
.driver-card {
    border:1px solid #ddd;
    border-radius:15px;
    overflow:hidden;
    background:white;
    min-height: 390px;
}
.driver-head {
    display:grid;
    grid-template-columns: 55% 45%;
    background:#f4f4f4;
    border-bottom:1px solid #ddd;
}
.driver-head div { padding: 12px 14px; font-size: 12px; color:#555; }
.driver-row {
    display:grid;
    grid-template-columns: 32px 1fr 85px 55px;
    border-bottom:1px solid #ececec;
    min-height: 26px;
    align-items:center;
    font-size:12px;
}
.driver-row div { padding: 4px 6px; }
.driver-check {
    width: 13px; height: 13px; border:1px solid #bbb; border-radius:3px; display:inline-block; margin-left:6px;
}
.driver-check.on { background:#0087c7; border-color:#0087c7; position:relative; }
.driver-check.on:after { content:'✓'; color:white; font-size:10px; position:absolute; left:2px; top:-2px; }
.driver-val { background:#eef8f1; font-size:16px; color:#555; height:100%; display:flex; align-items:center; }
.driver-impact.positive { color:#15803d; }
.driver-impact.negative { color:#555; }
.driver-impact.warn { background:#fbe8e8; }
.driver-impact.neutral { background:#fff8e8; }
.table-wrap { overflow-x:auto; }
table.chem-table {
    width:100%; border-collapse:collapse; font-size:12px; color:#606060;
}
table.chem-table th {
    background:#e9e9e9; color:#666; text-align:left; padding:11px 10px; border:1px solid #ddd; font-weight:700;
}
table.chem-table td {
    padding:8px 10px; border:1px solid #e7e7e7; background:white;
}
table.chem-table tr:nth-child(even) td { background:#fcfcfc; }
.reco-box {
    background:#f8fbff;
    border:1px solid #dbeafe;
    border-left:4px solid #0087c7;
    border-radius:10px;
    padding:12px 14px;
    color:#365166;
    font-size:13px;
}
.ai-box {
    background: linear-gradient(135deg,#f0f9ff,#fff);
    border:1px solid #cdeafe;
    border-radius:16px;
    padding:14px;
}
.prompt-chip {
    display:inline-block;
    padding:7px 11px;
    border:1px solid #e5e7eb;
    background:#fff;
    border-radius:999px;
    font-size:12px;
    margin:4px;
    color:#555;
}
@media (max-width: 1000px){ .kpi-grid{grid-template-columns:repeat(2,1fr);} }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def clean_money(x):
    if pd.isna(x):
        return 0.0
    if isinstance(x, str):
        return float(x.replace(",", "").replace("$", "").strip() or 0)
    return float(x)


def clean_year(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.replace(",", "").strip()
    try:
        return int(float(x))
    except Exception:
        return np.nan


def fmt_money(v):
    try:
        v = float(v)
    except Exception:
        return "$0"
    if abs(v) >= 1_000_000:
        return f"${v/1_000_000:,.1f}M"
    if abs(v) >= 1_000:
        return f"${v/1_000:,.0f}K"
    return f"${v:,.0f}"


def fmt_int(v):
    return f"{int(round(float(v))):,}"


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    lower_map = {c.lower().strip(): c for c in df.columns}

    aliases = {
        "year": "Year",
        "ear": "Year",
        "month": "Month",
        "supply_chain_zone": "Supply_Chain_Zone",
        "supply chain zone": "Supply_Chain_Zone",
        "plant name": "Plant Name",
        "plant_name": "Plant Name",
        "product 1 category/division name": "Product 1 Category/Division Name",
        "product_1_category/division_name": "Product 1 Category/Division Name",
        "sum of order value in usd from lc": "Sum of Order Value in USD from LC",
        "sum_of_order_value_in_usd_from_lc": "Sum of Order Value in USD from LC",
        "sum of order quantity bu": "Sum of Order quantity BU",
        "sum_of_order_quantity_bu": "Sum of Order quantity BU",
    }
    for raw, target in aliases.items():
        if raw in lower_map:
            rename_map[lower_map[raw]] = target
    df = df.rename(columns=rename_map)

    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            if c == "Year":
                df[c] = 2025
            elif c == "Month":
                df[c] = "January"
            elif c == "Supply_Chain_Zone":
                df[c] = "Push / MPS"
            elif c == "Plant Name":
                df[c] = "Chemelex - Trenton"
            elif c == "Product 1 Category/Division Name":
                df[c] = "PD / Heat Tracing Components"
            elif c == "Sum of Order Value in USD from LC":
                df[c] = 0.0
            elif c == "Sum of Order quantity BU":
                df[c] = 0.0

    df = df[REQUIRED_COLUMNS].copy()
    df["Year"] = df["Year"].apply(clean_year).fillna(2025).astype(int)
    df["Month"] = df["Month"].astype(str).str.strip().str.title()
    df["Month"] = df["Month"].replace({"Sept": "September", "Sep": "September"})
    df["Month_Num"] = df["Month"].map(MONTH_NUM).fillna(1).astype(int)
    df["Supply_Chain_Zone"] = df["Supply_Chain_Zone"].astype(str).str.strip()
    df["Plant Name"] = df["Plant Name"].astype(str).str.strip()
    df["Product 1 Category/Division Name"] = df["Product 1 Category/Division Name"].astype(str).str.strip()
    df["Order_Value"] = df["Sum of Order Value in USD from LC"].apply(clean_money)
    df["Order_Qty"] = pd.to_numeric(df["Sum of Order quantity BU"], errors="coerce").fillna(0.0)

    # Synthetic planning fields for demo until Synthefy provides forecast / actual / AOP fields.
    # Uses deterministic formulas from existing order value/quantity so upload data still drives visuals.
    rng_base = (df["Year"] * 13 + df["Month_Num"] * 17 + df.index * 19) % 100
    df["Actual"] = df["Order_Value"]
    df["Forecast"] = df["Actual"] * (0.88 + rng_base / 500.0)
    df["Previous_Year"] = df["Actual"] * (0.72 + ((rng_base + 31) % 100) / 420.0)
    df["Demand_Plan"] = df["Actual"] * (0.95 + ((rng_base + 11) % 100) / 650.0)
    df["AOP_FOP"] = df["Actual"] * (1.02 + ((rng_base + 29) % 100) / 900.0)
    df["Variance"] = df["Actual"] - df["Forecast"]
    df["Variance_pct"] = np.where(df["Forecast"].abs() > 0, df["Variance"] / df["Forecast"] * 100, 0)

    df["Category"] = df["Product 1 Category/Division Name"].str.replace("PD / ", "", regex=False).str.strip()
    df["SKU"] = [f"{2000000000 + i:010d}" for i in range(len(df))]
    df["Sub_Category"] = df["Supply_Chain_Zone"].str.replace("/", " / ", regex=False)
    df["Product_Description"] = df["Category"].str.slice(0, 4).str.upper() + " " + df["SKU"].str[-4:] + " DEMO"
    return df


@st.cache_data(show_spinner=False)
def make_demo_data() -> pd.DataFrame:
    rows = [
        [2021, "April", "Push / RM", "Chemelex - RWC/UCDC", "PD / Heat Tracing Components", 436.01, 59],
        [2021, "April", "Push / MPS", "Chemelex - Trenton", "PD / Fire and Performance Wiring", 2140.5, 100],
        [2021, "April", "BUFFER (at CODP)", "Chemelex - Trenton", "PD / MI Heat Tracing", 8920, 25],
        [2021, "April", "Pull / Kanban", "Chemelex - RWC/UCDC", "PD / Floor Heating", 180.5, 12],
        [2021, "April", "Push / MPS", "Chemelex - Pharr, T", "PD / Heat Tracing Components", 950, 50],
    ]
    base = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    extra = []
    plants = ["Chemelex - RWC/UCDC", "Chemelex - Trenton", "Chemelex - Pharr, T", "Chemelex - Redwood City"]
    zones = ["Push / RM", "Push / MPS", "BUFFER (at CODP)", "Pull / Kanban"]
    cats = [
        "PD / Heat Tracing Components",
        "PD / Floor Heating",
        "PD / MI Heat Tracing",
        "PD / Fire and Performance Wiring",
        "IHS.Standard Poly Heaters",
        "BIS Commercial EHT",
        "BIS Residential EHT",
    ]
    rng = np.random.default_rng(42)
    for year in [2022, 2023, 2024, 2025]:
        for month in MONTH_ORDER:
            for cat in cats:
                value = rng.lognormal(mean=10.1, sigma=.55)
                qty = rng.integers(10, 500)
                if "Floor" in cat and month in ["October", "November", "December"]:
                    value *= 1.25
                if "Commercial" in cat and month in ["January", "February"]:
                    value *= 1.15
                extra.append([year, month, rng.choice(zones), rng.choice(plants), cat, round(value, 2), int(qty)])
    demo = pd.concat([base, pd.DataFrame(extra, columns=REQUIRED_COLUMNS)], ignore_index=True)
    return normalize_columns(demo)


def load_data() -> pd.DataFrame:
    st.sidebar.markdown("### Upload Synthefy / Excel output")
    f = st.sidebar.file_uploader("Upload CSV / XLSX", type=["csv", "xlsx", "xls"])
    if f is None:
        st.sidebar.info("Using demo data shaped like your sample rows.")
        return make_demo_data()
    try:
        if f.name.lower().endswith(".csv"):
            raw = pd.read_csv(f)
        else:
            raw = pd.read_excel(f)
        return normalize_columns(raw)
    except Exception as e:
        st.sidebar.error(f"Could not read uploaded file: {e}")
        return make_demo_data()


def topbar(active_page: str):
    active_mat = "active" if active_page == "Material Forecast" else ""
    active_analysis = "active" if active_page == "Forecast Analysis" else ""
    st.markdown(
        f"""
<div class="chem-topbar">
  <div><span class="chem-logo-small">algo</span><span class="chem-logo">chemelex</span></div>
  <div class="chem-icons"><span>🔔<span class="notif-dot">14</span></span><span>⚙️</span><span>●</span><span>⌄</span></div>
</div>
<div class="chem-tabs">
  <span class="chem-tab {active_mat}">Material Forecast</span>
  <span class="chem-tab {active_analysis}">Forecast Analysis</span>
</div>
<div class="chem-strip">
  <div>{TODAY_LABEL}&nbsp;&nbsp;&nbsp; Lorem Ipsum dolor sit amet&nbsp; 🔗</div>
  <div><span class="strip-icons">◀</span>&nbsp; 1 of 14 &nbsp;<span class="strip-icons">▶</span>&nbsp;&nbsp;<span class="strip-icons">×</span></div>
</div>
""",
        unsafe_allow_html=True,
    )


def plotly_layout(fig, height=315):
    fig.update_layout(
        height=height,
        margin=dict(l=42, r=18, t=20, b=38),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, Segoe UI, Arial", size=11, color="#666"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, linecolor="#e5e7eb", tickfont=dict(size=10))
    fig.update_yaxes(gridcolor="#eeeeee", zeroline=False, tickfont=dict(size=10))
    return fig


def grouped_category_chart(cat_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_bar(x=cat_df["Category"], y=cat_df["Forecast"], name="Forecast", marker_color=CHEMELEX_BLUE)
    fig.add_bar(x=cat_df["Category"], y=cat_df["Actual"], name="Actual", marker_color=CHEMELEX_GREEN)
    fig.add_bar(x=cat_df["Category"], y=cat_df["Previous_Year"], name="Previous Year", marker_color=CHEMELEX_ORANGE)
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    return plotly_layout(fig, 305)


def variance_chart(cat_df: pd.DataFrame):
    y = (cat_df["Actual"] - cat_df["Forecast"]).abs()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cat_df["Category"], y=y, mode="lines", line=dict(color=CHEMELEX_RED, width=2), name="Variance"))
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    return plotly_layout(fig, 305)


def trend_chart(monthly: pd.DataFrame, show_overlay=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["Month_Label"], y=monthly["Actual"], mode="lines", name="Actual Demand", line=dict(color=CHEMELEX_BLUE, width=2)))
    fig.add_trace(go.Scatter(x=monthly["Month_Label"], y=monthly["Forecast"], mode="lines", name="Forecasted Demand", line=dict(color=CHEMELEX_ORANGE, width=2)))
    if show_overlay:
        fig.add_trace(go.Scatter(x=monthly["Month_Label"], y=monthly["Demand_Plan"], mode="lines", name="Demand Plan", line=dict(color=CHEMELEX_GREEN, width=1.8, dash="dot"), fill="tozeroy", fillcolor="rgba(46,179,90,.08)"))
    # today vertical line around latest actual
    if len(monthly):
        pos = max(0, len(monthly) - 4)
        fig.add_vline(x=monthly.iloc[pos]["Month_Label"], line_width=1, line_dash="dash", line_color="#aaa")
    fig.update_yaxes(tickformat="~s")
    return plotly_layout(fig, 405)


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Category", as_index=False).agg(
        Forecast=("Forecast", "sum"),
        Actual=("Actual", "sum"),
        Previous_Year=("Previous_Year", "sum"),
        Demand_Plan=("Demand_Plan", "sum"),
        AOP_FOP=("AOP_FOP", "sum"),
        Order_Qty=("Order_Qty", "sum"),
    )


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    m = df.groupby(["Year", "Month_Num", "Month"], as_index=False).agg(
        Actual=("Actual", "sum"),
        Forecast=("Forecast", "sum"),
        Demand_Plan=("Demand_Plan", "sum"),
        AOP_FOP=("AOP_FOP", "sum"),
    ).sort_values(["Year", "Month_Num"])
    m["Month_Label"] = m["Month"].map(MONTH_SHORT).fillna(m["Month"].str[:3]) + " " + m["Year"].astype(str).str[-2:]
    return m


def page_shell(title, right_html=""):
    st.markdown(
        f"""
<div class="page-wrap">
  <div class="page-titlebar">
    <div class="page-title">{title}</div>
    <div class="filters-line">{right_html}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def material_forecast_page(df: pd.DataFrame):
    topbar("Material Forecast")
    plants = ["All"] + sorted(df["Plant Name"].dropna().unique().tolist())
    zones = ["All"] + sorted(df["Supply_Chain_Zone"].dropna().unique().tolist())

    with st.container():
        c1, c2, c3 = st.columns([4, 1.25, 1.25])
        with c1:
            st.markdown('<div class="page-wrap"><div class="page-titlebar"><div class="page-title">Material Forecast</div>', unsafe_allow_html=True)
        with c2:
            plant = st.selectbox("Plant", plants, label_visibility="collapsed", key="mat_plant")
        with c3:
            zone = st.selectbox("Region", zones, label_visibility="collapsed", key="mat_zone")
        st.markdown('</div></div>', unsafe_allow_html=True)

    filtered = df.copy()
    if plant != "All":
        filtered = filtered[filtered["Plant Name"] == plant]
    if zone != "All":
        filtered = filtered[filtered["Supply_Chain_Zone"] == zone]

    latest_year = int(filtered["Year"].max()) if len(filtered) else 2025
    latest_month = int(filtered[filtered["Year"] == latest_year]["Month_Num"].max()) if len(filtered) else 1
    current = filtered[(filtered["Year"] == latest_year) & (filtered["Month_Num"] == latest_month)]
    if current.empty:
        current = filtered

    total_forecast = current["Forecast"].sum()
    total_actual = current["Actual"].sum()
    variance_pct = (total_forecast - total_actual) / total_forecast * 100 if total_forecast else 0

    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="kpi-grid">
  <div class="kpi-card"><div class="kpi-label">Active SKUs</div><div class="kpi-value">{fmt_int(current['SKU'].nunique())}</div></div>
  <div class="kpi-card"><div class="kpi-label">Total Forecast</div><div class="kpi-value">{fmt_money(total_forecast)}</div></div>
  <div class="kpi-card"><div class="kpi-label">Total Actual</div><div class="kpi-value">{fmt_money(total_actual)}</div></div>
  <div class="kpi-card"><div class="kpi-label">Variance %</div><div class="kpi-value">{variance_pct:.1f}%</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

    cat = category_summary(current).sort_values("Actual", ascending=False).head(7)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="panel"><div class="card-title">Categories</div>', unsafe_allow_html=True)
        st.plotly_chart(grouped_category_chart(cat), use_container_width=True, key="mat_category_grouped")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="panel"><div class="card-title">Variance Analysis <span class="var-badge">Variance: {variance_pct:.1f}%</span></div>', unsafe_allow_html=True)
        st.plotly_chart(variance_chart(cat), use_container_width=True, key="mat_variance_line")
        st.markdown('</div>', unsafe_allow_html=True)

    table = current[["SKU", "Category", "Sub_Category", "Product_Description", "Forecast", "Actual", "Previous_Year", "Order_Qty"]].copy()
    table = table.sort_values("Actual", ascending=False).head(100)
    show = table.copy()
    for c in ["Forecast", "Actual", "Previous_Year"]:
        show[c] = show[c].map(fmt_money)
    show["Order_Qty"] = show["Order_Qty"].map(fmt_int)
    st.markdown(f'<div class="panel" style="margin-top:18px"><div class="card-title">SKU Wise Material Forecast</div><div class="small-muted">{len(table):,} items • Updated 3 minutes ago</div>', unsafe_allow_html=True)
    st.dataframe(show, use_container_width=True, hide_index=True, height=270)
    st.download_button("⬇️ Download SKU material forecast", data=to_csv_bytes(table), file_name="chemelex_material_forecast.csv", mime="text/csv")
    st.markdown('</div></div>', unsafe_allow_html=True)


def forecast_analysis_page(df: pd.DataFrame):
    topbar("Forecast Analysis")
    plants = ["All"] + sorted(df["Plant Name"].dropna().unique().tolist())
    zones = ["All"] + sorted(df["Supply_Chain_Zone"].dropna().unique().tolist())
    categories = ["All"] + sorted(df["Category"].dropna().unique().tolist())

    with st.container():
        c0, c1, c2, c3, c4 = st.columns([2.2, 1.2, 1.2, 1.2, 1.2])
        with c0:
            st.markdown('<div class="page-wrap"><div class="page-titlebar"><div class="page-title">Forecast Analysis</div>', unsafe_allow_html=True)
        with c1:
            plant = st.selectbox("Plant", plants, label_visibility="collapsed", key="ana_plant")
        with c2:
            zone = st.selectbox("Region", zones, label_visibility="collapsed", key="ana_zone")
        with c3:
            category = st.selectbox("SOP Family", categories, label_visibility="collapsed", key="ana_cat")
        with c4:
            period = st.selectbox("Period", ["01/10/2024 - 07/11/2024", "Apr 30, 2024 - May 04, 2024", "FY2025 YTD"], label_visibility="collapsed", key="ana_period")
        st.markdown('</div></div>', unsafe_allow_html=True)

    f = df.copy()
    if plant != "All":
        f = f[f["Plant Name"] == plant]
    if zone != "All":
        f = f[f["Supply_Chain_Zone"] == zone]
    if category != "All":
        f = f[f["Category"] == category]
    if f.empty:
        f = df.copy()

    baseline = f["Actual"].sum()
    final_forecast = f["Forecast"].sum()
    total_impact = final_forecast - baseline
    variance = (final_forecast - baseline) / baseline * 100 if baseline else 0

    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="kpi-grid" style="grid-template-columns: repeat(3, minmax(0,1fr));">
  <div class="kpi-card"><div class="kpi-label">Baseline</div><div class="kpi-value">{fmt_int(baseline)} FT</div></div>
  <div class="kpi-card"><div class="kpi-label">Total Impact</div><div class="kpi-value">{fmt_int(total_impact)} FT</div></div>
  <div class="kpi-card"><div class="kpi-label">Final Forecast</div><div class="kpi-value">{fmt_int(final_forecast)} <span style="font-size:12px;color:#16a34a;float:right;">{variance:.1f}% variance</span></div></div>
</div>
""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 3.2])
    with left:
        drivers = [
            ("Recent Demand Trend", "18,900 FT", "", "on", "positive"),
            ("Demand Volatility", "0.47", "+0.14", "", "neutral"),
            ("Commercial Mix", "58%", "+6%", "", "positive"),
            ("Intercompany Mix", "42%", "-6%", "", "warn"),
            ("Korea Share", "36%", "+8%", "on-green", "warn"),
            ("China Share", "17%", "+8%", "", "positive"),
            ("Top Order Contribution", "34%", "-6%", "", "neutral"),
            ("Top Customer Share", "32%", "-4%", "", ""),
            ("Confirmed Backlog", "2800 FT", "", "", "positive"),
            ("Pipeline Impact", "+1200 FT", "", "", "positive"),
            ("Manual Override Impact", "+600 FT", "", "", "positive"),
        ]
        html = ['<div class="driver-card"><div class="driver-head"><div>Fluctuation Drivers</div><div>Apr 30, 2024-May 04, 2024</div></div>']
        for label, val, imp, on, tone in drivers:
            check_cls = "driver-check on" if on else "driver-check"
            if on == "on-green":
                check_cls = "driver-check on" 
            html.append(f'<div class="driver-row"><div><span class="{check_cls}"></span></div><div>{label}</div><div class="driver-val {tone}">{val}</div><div class="driver-impact {tone}">{imp}</div></div>')
        html.append('</div>')
        st.markdown("".join(html), unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel" style="padding:10px 14px 0 14px;"><div style="text-align:right;font-size:13px;color:#555;margin-bottom:3px;"><span style="color:#f28c00;font-size:16px;">■</span> Forecasted Demand &nbsp;&nbsp; ⊕ Today</div>', unsafe_allow_html=True)
        st.plotly_chart(trend_chart(monthly_summary(f), show_overlay=True), use_container_width=True, key="analysis_demand_trend")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    w = category_summary(f)
    w["Gap_vs_Demand_Plan"] = w["Forecast"] - w["Demand_Plan"]
    w["Gap_vs_AOP_FOP"] = w["Forecast"] - w["AOP_FOP"]
    w["Gap_%"] = np.where(w["Demand_Plan"].abs() > 0, w["Gap_vs_Demand_Plan"] / w["Demand_Plan"] * 100, 0)
    w["Priority"] = np.where(w["Gap_%"].abs() > 12, "High", np.where(w["Gap_%"].abs() > 5, "Medium", "Low"))
    w["Planner Recommendation"] = np.where(
        w["Gap_vs_Demand_Plan"] > 0,
        "Review demand plan uplift; validate backlog/pipeline before supply commit",
        "Risk of over-planning; validate with sales before reducing forecast",
    )
    show_w = w[["Category", "Forecast", "Demand_Plan", "AOP_FOP", "Gap_vs_Demand_Plan", "Gap_%", "Priority", "Planner Recommendation"]].copy()
    for c in ["Forecast", "Demand_Plan", "AOP_FOP", "Gap_vs_Demand_Plan"]:
        show_w[c] = show_w[c].map(fmt_money)
    show_w["Gap_%"] = show_w["Gap_%"].map(lambda x: f"{x:.1f}%")

    st.markdown('<div class="panel"><div class="card-title">Gap Watchlist & Planner Actions</div>', unsafe_allow_html=True)
    st.dataframe(show_w, use_container_width=True, hide_index=True, height=235)
    st.download_button("⬇️ Download visible watchlist", data=to_csv_bytes(w), file_name="chemelex_gap_watchlist.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

    col_ai, col_reco = st.columns([1.35, 1])
    with col_ai:
        st.markdown(
            """
<div class="ai-box">
  <div class="card-title">AI Planner Query Concept</div>
  <div style="font-size:18px;color:#333;font-weight:600;margin-bottom:8px;">“Do you think my demand is going to change in the next 3 months?”</div>
  <div class="small-muted">The cockpit answers using recent demand trend, volatility, commercial/intercompany mix, backlog, pipeline, manual override and SOP-family history.</div>
  <div style="margin-top:12px;"><span class="prompt-chip">Which SOP families are over-planned?</span><span class="prompt-chip">Where is actual below FOP?</span><span class="prompt-chip">What should I produce next month?</span></div>
</div>
""",
            unsafe_allow_html=True,
        )
    with col_reco:
        st.markdown(
            f"""
<div class="reco-box">
  <b>Recommended demo story:</b><br/>
  Start with Material Forecast → show category variance → drill into Forecast Analysis → explain drivers → show watchlist → ask AI planner query for next 3 months.<br/><br/>
  <b>Current signal:</b> Forecast is {variance:.1f}% vs baseline for selected view. Prioritize high-gap SOP families first.
</div>
""",
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# App entry
# -----------------------------------------------------------------------------
df = load_data()

# This radio is hidden visually in sidebar, but gives real interactivity.
page = st.sidebar.radio("Page", ["Material Forecast", "Forecast Analysis"], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("**Expected columns**")
st.sidebar.code("\n".join(REQUIRED_COLUMNS), language="text")
st.sidebar.download_button("Download demo-shaped data", data=to_csv_bytes(df[REQUIRED_COLUMNS]), file_name="chemelex_demo_input.csv", mime="text/csv")

if page == "Material Forecast":
    material_forecast_page(df)
else:
    forecast_analysis_page(df)
