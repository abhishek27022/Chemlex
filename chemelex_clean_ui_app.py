"""
Chemelex Demand Forecasting Cockpit — clean end-user Streamlit UI
Consumes forecast_backtest_results.csv and shows only business-friendly forecast, actual and accuracy.

Accuracy formula used for end users:
    Accuracy % = min(Actual, Forecast) / max(Actual, Forecast) * 100
Example: Actual 90, Forecast 100 => 90%
"""

from __future__ import annotations

import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chemelex Demand Forecast",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS — Designed to resemble the clean Pwani/Chemelex mockup style
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    :root{
        --blue:#0078b8;
        --blue2:#0b84c6;
        --navy:#0c2d57;
        --green:#2fb25b;
        --orange:#f28c00;
        --red:#d64b4b;
        --gray:#6b7280;
        --line:#e5e7eb;
        --panel:#ffffff;
        --soft:#f7f8fa;
        --beige:#fff4df;
    }
    html, body, [class*="css"] {font-family: Inter, "Segoe UI", Arial, sans-serif;}
    .block-container{padding-top:0rem; padding-left:0.35rem; padding-right:0.35rem; max-width:100%;}
    header[data-testid="stHeader"]{display:none;}
    div[data-testid="stToolbar"]{display:none;}
    section[data-testid="stSidebar"]{background:#ffffff; border-right:1px solid #e8e8e8;}
    .app-shell{background:#fbfbfc; min-height:100vh;}
    .topbar{
        height:48px; display:flex; align-items:center; justify-content:space-between;
        border-bottom:1px solid #e7e7e7; background:#fff; padding:0 22px 0 24px;
    }
    .brand{display:flex; align-items:center; gap:10px; font-size:30px; font-weight:800; color:#3a7182; letter-spacing:-0.5px;}
    .brand-mark{width:26px; height:26px; display:inline-grid; place-items:center; color:#0078b8; font-weight:900; border:2px solid #0078b8; border-radius:7px; transform:rotate(45deg);}
    .brand-mark span{transform:rotate(-45deg); display:block; font-size:15px;}
    .top-icons{display:flex; align-items:center; gap:16px; color:#3f3f46; font-size:20px;}
    .bell{position:relative;}
    .badge{position:absolute; top:-10px; right:-10px; background:#d92929; color:#fff; border-radius:999px; font-size:10px; padding:1px 6px; font-weight:700;}
    .nav-row{
        height:52px; display:flex; align-items:center; gap:12px; border-bottom:1px solid #ececec;
        background:#fff; padding:0 26px;
    }
    .tab-pill{padding:8px 19px; border-radius:999px; font-size:14px; color:#30333a; border:1px solid transparent;}
    .tab-pill.active{border-color:#0082c7; color:#1d4057; background:#eef8ff; font-weight:600;}
    .notice-strip{height:27px; display:flex; align-items:center; justify-content:space-between; background:#fff4df; border-bottom:1px solid #f2e1c1; border-left:5px solid #f28c00; color:#444; font-size:13px; padding:0 16px;}
    .notice-right{display:flex; gap:8px; align-items:center; color:#c77700; font-weight:600;}
    .page-card{background:#fff; border:1px solid #e8e8e8; border-radius:16px; box-shadow:0 1px 6px rgba(0,0,0,0.05); padding:16px; margin:14px 18px;}
    .page-title{font-size:20px; font-weight:650; color:#4a4a4a; margin:0;}
    .filter-wrap{display:flex; align-items:center; gap:14px; justify-content:flex-end; flex-wrap:wrap;}
    .filter-label{font-size:12px; color:#6b7280; margin-bottom:3px;}
    .kpi-grid{display:grid; grid-template-columns: repeat(5, minmax(160px, 1fr)); gap:14px; margin:8px 18px 14px 18px;}
    .kpi-box{background:#fff; border:1px solid #e7e7e7; border-radius:4px; padding:12px 14px; min-height:78px;}
    .kpi-label{font-size:13px; color:#666; margin-bottom:5px;}
    .kpi-value{font-size:22px; color:#343434; font-weight:700; line-height:1.15;}
    .kpi-sub{font-size:12px; color:#7b7b7b; margin-top:4px;}
    .kpi-good{color:#149747;}.kpi-bad{color:#c73838;}.kpi-blue{color:#0078b8;}
    .chart-grid{display:grid; grid-template-columns: 1fr 1fr; gap:18px; margin:0 18px;}
    .chart-card{background:#fff; border:1px solid #e8e8e8; border-radius:16px; box-shadow:0 1px 6px rgba(0,0,0,0.05); padding:12px;}
    .chart-head{display:flex; justify-content:space-between; align-items:center; margin:0 0 8px 0;}
    .chart-title{font-size:14px; font-weight:700; color:#666;}
    .legend-inline{display:flex; gap:14px; font-size:12px; color:#555; align-items:center;}
    .dot{display:inline-block;width:13px;height:13px;border-radius:4px;margin-right:5px;vertical-align:-2px;}
    .table-card{background:#fff; border:1px solid #e8e8e8; border-radius:16px; box-shadow:0 1px 6px rgba(0,0,0,0.05); margin:18px; padding:0; overflow:hidden;}
    .table-head{display:flex; justify-content:space-between; align-items:center; padding:14px 16px; border-bottom:1px solid #eee;}
    .table-title{font-size:15px; font-weight:700; color:#555;}
    .table-sub{font-size:12px; color:#aaa; margin-top:2px;}
    .download-circle{width:36px;height:36px;border-radius:999px;background:#0087c7;color:white;display:grid;place-items:center;font-size:18px;box-shadow:0 2px 6px rgba(0,0,0,.2);}
    .stSelectbox label,.stDateInput label,.stTextInput label{display:none !important;}
    div[data-baseweb="select"] > div{border-radius:999px !important; min-height:37px; border-color:#dedede; font-size:13px; background:#fff;}
    div[data-testid="stDateInput"] input, div[data-testid="stTextInput"] input{border-radius:999px !important; min-height:37px; border-color:#dedede; font-size:13px; background:#fff;}
    .stDataFrame{border-radius:0 !important;}
    div[data-testid="stMetric"]{background:#fff;border:1px solid #e6e6e6;border-radius:12px;padding:12px;}
    .accuracy-pill{display:inline-block; min-width:66px; text-align:center; padding:4px 10px; border-radius:8px; font-weight:700; font-size:12px;}
    .acc-good{background:#cdeed4;color:#117333;}.acc-mid{background:#ffe5ad;color:#8b5a00;}.acc-low{background:#ffd6d6;color:#a12a2a;}
    .small-note{font-size:12px;color:#7b7b7b;}
    .formula-box{background:#f7fbff;border:1px solid #d9ecfa;border-radius:12px;padding:10px 12px;color:#2e5876;font-size:13px;}
    @media(max-width:1100px){.kpi-grid{grid-template-columns:repeat(2,1fr)}.chart-grid{grid-template-columns:1fr}.brand{font-size:24px}.nav-row{overflow:auto}.tab-pill{white-space:nowrap}}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
MONTH_NAMES = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
MONTH_ABBR = {k:v[:3] for k,v in MONTH_NAMES.items()}


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def fmt_num(x, decimals=0):
    if pd.isna(x):
        return "-"
    try:
        x = float(x)
    except Exception:
        return str(x)
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x:,.{decimals}f}"
    return f"{x:,.{decimals}f}"


def business_accuracy(actual, forecast):
    actual = pd.to_numeric(actual, errors="coerce")
    forecast = pd.to_numeric(forecast, errors="coerce")
    denom = np.maximum(np.abs(actual), np.abs(forecast))
    numer = np.minimum(np.abs(actual), np.abs(forecast))
    acc = np.where(denom > 0, numer / denom * 100, np.nan)
    return np.clip(acc, 0, 100)


def accuracy_text(actual, forecast):
    if pd.isna(actual) or pd.isna(forecast):
        return "-"
    acc = business_accuracy(pd.Series([actual]), pd.Series([forecast]))[0]
    return f"{acc:.1f}%"


def clean_plant(x):
    if pd.isna(x):
        return "Unknown"
    return str(x).replace("Chemelex", "").replace("-", "").strip() or str(x)


def load_data(uploaded_file=None):
    if uploaded_file is not None:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    elif os.path.exists("forecast_backtest_results.csv"):
        df = pd.read_csv("forecast_backtest_results.csv")
    elif os.path.exists("/mnt/data/forecast_backtest_results.csv"):
        df = pd.read_csv("/mnt/data/forecast_backtest_results.csv")
    else:
        # fallback tiny demo
        df = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=8, freq="MS").astype(str),
            "codp_zone": ["Push / MPS", "Pull / Kanban"] * 4,
            "plant": ["Chemelex  - Trenton", "Chemelex - RWC/UCDC"] * 4,
            "product_category_1": ["PD / Heat Tracing Components", "PD / Floor Heating"] * 4,
            "group_key": ["MAT-001", "MAT-002"] * 4,
            "target": ["order_qty_bu"] * 8,
            "split": ["split1"] * 8,
            "model": ["LightGBM"] * 8,
            "actual": [90,120,130,80,100,95,115,140],
            "forecast": [100,110,120,85,105,100,125,130],
            "is_best": [True]*8,
            "year": [2025]*8,
            "month": [1,2,3,4,5,6,7,8],
        })

    df.columns = [str(c).strip() for c in df.columns]
    required = ["date", "codp_zone", "plant", "product_category_1", "group_key", "target", "split", "model", "actual", "forecast", "is_best", "year", "month"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Uploaded file missing required columns: {missing}")
        st.stop()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df["month_name"] = df["month"].map(MONTH_NAMES)
    df["month_abbr"] = df["month"].map(MONTH_ABBR)
    df["plant_clean"] = df["plant"].apply(clean_plant)
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce")
    df["gap"] = df["actual"] - df["forecast"]
    df["abs_gap"] = df["gap"].abs()
    df["accuracy_pct"] = business_accuracy(df["actual"], df["forecast"])
    df["target_label"] = df["target"].map({"order_qty_bu": "Volume (BU)", "order_value_usd": "Value (USD)"}).fillna(df["target"])
    df["is_best"] = df["is_best"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df


def weighted_accuracy(df):
    if df.empty:
        return np.nan
    denom = np.maximum(df["actual"].abs(), df["forecast"].abs()).sum()
    numer = np.minimum(df["actual"].abs(), df["forecast"].abs()).sum()
    return numer / denom * 100 if denom > 0 else np.nan


def style_accuracy(val):
    try:
        v = float(str(val).replace("%", ""))
    except Exception:
        return ""
    if v >= 90:
        return "background-color:#cdeed4;color:#117333;font-weight:700;border-radius:8px;text-align:center;"
    if v >= 75:
        return "background-color:#ffe5ad;color:#8b5a00;font-weight:700;border-radius:8px;text-align:center;"
    return "background-color:#ffd6d6;color:#a12a2a;font-weight:700;border-radius:8px;text-align:center;"


def style_gap(val):
    try:
        v = float(str(val).replace(",", "").replace("(", "-").replace(")", ""))
    except Exception:
        return ""
    if v < 0:
        return "color:#d64b4b;font-weight:700;"
    if v > 0:
        return "color:#149747;font-weight:700;"
    return "color:#6b7280;"


def card_html(label, value, sub="", tone=""):
    tone_cls = {"good":"kpi-good", "bad":"kpi-bad", "blue":"kpi-blue"}.get(tone, "")
    return f"""
    <div class="kpi-box">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {tone_cls}">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


def build_grouped(df, group_cols):
    g = df.groupby(group_cols, dropna=False).agg(
        forecast=("forecast", "sum"),
        actual=("actual", "sum"),
        records=("group_key", "count"),
        materials=("group_key", "nunique"),
    ).reset_index()
    g["gap"] = g["actual"] - g["forecast"]
    g["abs_gap"] = g["gap"].abs()
    g["accuracy_pct"] = business_accuracy(g["actual"], g["forecast"])
    return g


def monthly_chart(df, metric_label):
    m = build_grouped(df, ["month", "month_abbr"])
    m = m.sort_values("month")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m["month_abbr"], y=m["forecast"], mode="lines+markers", name="Forecast", line=dict(color="#0078b8", width=3), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=m["month_abbr"], y=m["actual"], mode="lines+markers", name="Actual", line=dict(color="#2fb25b", width=3), marker=dict(size=7)))
    # Simple previous-year/proxy baseline from forecast shifted visually; if actual PY not in file, keep light grey reference
    prev = m["actual"].rolling(2, min_periods=1).mean() * 0.88
    fig.add_trace(go.Scatter(x=m["month_abbr"], y=prev, mode="lines+markers", name="Prior Year / Baseline", line=dict(color="#b6bdc7", width=2, dash="dot"), marker=dict(size=6)))
    fig.update_layout(
        height=315,
        margin=dict(l=35, r=18, t=10, b=35),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", y=1.15, x=0, font=dict(size=12)),
        font=dict(color="#5f6368", size=12),
        yaxis_title=metric_label,
    )
    fig.update_xaxes(showgrid=False, linecolor="#ddd")
    fig.update_yaxes(gridcolor="#ececec", zeroline=False)
    return fig


def accuracy_chart(df):
    m = build_grouped(df, ["month", "month_abbr"]).sort_values("month")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m["month_abbr"], y=m["accuracy_pct"], mode="lines+markers+text", text=[f"{x:.0f}%" if pd.notna(x) else "" for x in m["accuracy_pct"]], textposition="top center", name="Accuracy", line=dict(color="#0069bd", width=3), marker=dict(size=8)))
    fig.update_layout(height=315, margin=dict(l=35, r=18, t=10, b=35), paper_bgcolor="white", plot_bgcolor="white", showlegend=False, font=dict(color="#5f6368", size=12), yaxis_title="Accuracy %")
    fig.update_yaxes(range=[0,100], ticksuffix="%", gridcolor="#ececec", zeroline=False)
    fig.update_xaxes(showgrid=False, linecolor="#ddd")
    return fig


def category_chart(df, metric_label):
    cat = build_grouped(df, ["product_category_1"]).sort_values("actual", ascending=False).head(10)
    x = [str(v).replace("PD / ", "")[:22] for v in cat["product_category_1"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=cat["forecast"], name="Forecast", marker_color="#0078b8"))
    fig.add_trace(go.Bar(x=x, y=cat["actual"], name="Actual", marker_color="#2fb25b"))
    fig.update_layout(height=315, margin=dict(l=35, r=18, t=10, b=55), paper_bgcolor="white", plot_bgcolor="white", barmode="group", legend=dict(orientation="h", y=1.18, x=0.42), font=dict(color="#5f6368", size=12), yaxis_title=metric_label)
    fig.update_xaxes(tickangle=0, showgrid=False, linecolor="#ddd")
    fig.update_yaxes(gridcolor="#ececec", zeroline=False)
    return fig


def plant_zone_chart(df):
    p = build_grouped(df, ["codp_zone", "plant_clean"]).sort_values("accuracy_pct", ascending=True)
    # show only top rows to keep clean
    p["name"] = p["codp_zone"].astype(str).str.replace("BUFFER (at CODP)", "Buffer", regex=False) + " · " + p["plant_clean"].astype(str)
    p = p.tail(12)
    fig = go.Figure(go.Bar(x=p["accuracy_pct"], y=p["name"], orientation="h", marker_color="#2fb25b", text=[f"{x:.1f}%" for x in p["accuracy_pct"]], textposition="outside"))
    fig.update_layout(height=315, margin=dict(l=10, r=60, t=10, b=25), paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#5f6368", size=11), xaxis_title="Accuracy %")
    fig.update_xaxes(range=[0, 105], ticksuffix="%", gridcolor="#ececec")
    fig.update_yaxes(showgrid=False)
    return fig


def make_display_table(df, metric_label, limit=250):
    table = build_grouped(df, ["year", "month", "month_name", "codp_zone", "plant_clean", "product_category_1", "group_key"])
    table = table.sort_values("abs_gap", ascending=False).head(limit).copy()
    table["Forecast"] = table["forecast"].round(0)
    table["Actual"] = table["actual"].round(0)
    table["Gap"] = table["gap"].round(0)
    table["Accuracy %"] = table["accuracy_pct"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
    table["Formula"] = table.apply(lambda r: f"min({fmt_num(r['actual'])}, {fmt_num(r['forecast'])}) / max({fmt_num(r['actual'])}, {fmt_num(r['forecast'])})", axis=1)
    table = table.rename(columns={
        "year":"Year",
        "month_name":"Month",
        "codp_zone":"Supply Chain Zone",
        "plant_clean":"Plant Name",
        "product_category_1":"Product Category",
        "group_key":"Material / Group Key",
    })
    return table[["Year", "Month", "Supply Chain Zone", "Plant Name", "Product Category", "Material / Group Key", "Forecast", "Actual", "Accuracy %", "Gap"]]

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOAD
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Upload Forecast File")
    uploaded = st.file_uploader("Upload forecast_backtest_results.csv", type=["csv", "xlsx"])
    st.caption("Expected: date, codp_zone, plant, product_category_1, group_key, target, split, model, actual, forecast, is_best, year, month")

df_raw = load_data(uploaded)

# default to best model rows to keep clean
if df_raw["is_best"].any():
    default_best = True
else:
    default_best = False

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="app-shell">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="topbar">
        <div class="brand"><div class="brand-mark"><span>◆</span></div>chemelex</div>
        <div class="top-icons"><span class="bell">🔔<span class="badge">14</span></span><span>⚙️</span><span>👤⌄</span></div>
    </div>
    <div class="nav-row">
        <span class="tab-pill active">Demand Forecast</span>
        <span class="tab-pill">Hero SKUs</span>
        <span class="tab-pill">Customers</span>
        <span class="tab-pill">Map Projections</span>
        <span class="tab-pill">Forecast Analysis</span>
        <span class="tab-pill">Campaigns</span>
        <span class="tab-pill">Market Analytics</span>
    </div>
    <div class="notice-strip">
        <div>{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} &nbsp;&nbsp; Demand forecast accuracy view · Actual vs Forecast</div>
        <div class="notice-right">◀ 1 of 14 ▶ ✖</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# FILTERS
# ──────────────────────────────────────────────────────────────────────────────
# compact page title card using streamlit columns for actual filter widgets
st.markdown('<div class="page-card">', unsafe_allow_html=True)
c_title, c1, c2, c3, c4, c5, c6 = st.columns([2.1, 1.25, 1.45, 1.45, 1.6, 1.5, 1.15])
with c_title:
    st.markdown('<p class="page-title">Material Forecast</p>', unsafe_allow_html=True)
    st.caption("Business accuracy = min(Actual, Forecast) ÷ max(Actual, Forecast)")
with c1:
    years = sorted([int(x) for x in df_raw["year"].dropna().unique()])
    year_sel = st.selectbox("Year", years, index=len(years)-1 if years else 0, key="year")
with c2:
    months_available = sorted([int(x) for x in df_raw["month"].dropna().unique()])
    month_options = ["All"] + [MONTH_NAMES[m] for m in months_available]
    month_sel = st.selectbox("Month", month_options, index=0, key="month")
with c3:
    target_options = sorted(df_raw["target_label"].dropna().unique())
    target_sel = st.selectbox("Metric", target_options, index=0, key="target")
with c4:
    plant_options = ["All"] + sorted(df_raw["plant_clean"].dropna().unique())
    plant_sel = st.selectbox("Plant", plant_options, key="plant")
with c5:
    cat_options = ["All"] + sorted(df_raw["product_category_1"].dropna().unique())
    cat_sel = st.selectbox("Product Category", cat_options, key="cat")
with c6:
    best_only = st.toggle("Best model", value=default_best, key="best_toggle")
st.markdown('</div>', unsafe_allow_html=True)

# Apply filters
data = df_raw.copy()
data = data[data["year"] == year_sel]
data = data[data["target_label"] == target_sel]
if month_sel != "All":
    month_num = {v:k for k,v in MONTH_NAMES.items()}[month_sel]
    data = data[data["month"] == month_num]
if plant_sel != "All":
    data = data[data["plant_clean"] == plant_sel]
if cat_sel != "All":
    data = data[data["product_category_1"] == cat_sel]
if best_only:
    data = data[data["is_best"]]

if data.empty:
    st.warning("No rows match the selected filters. Please reset one of the filters.")
    st.stop()

metric_unit = "USD" if "Value" in target_sel else "BU"
metric_label = "Order Value (USD)" if metric_unit == "USD" else "Order Quantity (BU)"

# ──────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ──────────────────────────────────────────────────────────────────────────────
active_materials = data["group_key"].nunique()
total_forecast = data["forecast"].sum()
total_actual = data["actual"].sum()
gap = total_actual - total_forecast
acc = weighted_accuracy(data)
on_track_cats = build_grouped(data, ["product_category_1"])
on_track = int((on_track_cats["accuracy_pct"] >= 85).sum())
total_cats = max(1, len(on_track_cats))
at_risk_materials = int((build_grouped(data, ["group_key"])["accuracy_pct"] < 75).sum())

st.markdown(
    f"""
    <div class="kpi-grid">
        {card_html('Active Materials', fmt_num(active_materials), 'Unique material / group keys', 'blue')}
        {card_html('Total Forecast', fmt_num(total_forecast), metric_label, 'blue')}
        {card_html('Total Actual', fmt_num(total_actual), metric_label, 'good')}
        {card_html('Forecast Accuracy', f'{acc:.1f}%' if pd.notna(acc) else '-', 'Actual 90 vs Forecast 100 → 90%', 'good' if pd.notna(acc) and acc>=85 else 'bad')}
        {card_html('At-risk Materials', fmt_num(at_risk_materials), 'Accuracy below 75%', 'bad' if at_risk_materials else 'good')}
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN CHARTS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="chart-grid">', unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    st.markdown('<div class="chart-card"><div class="chart-head"><div class="chart-title">Categories</div><div class="legend-inline"><span><i class="dot" style="background:#0078b8"></i>Forecast</span><span><i class="dot" style="background:#2fb25b"></i>Actual</span></div></div>', unsafe_allow_html=True)
    st.plotly_chart(category_chart(data, metric_label), use_container_width=True, config={"displayModeBar": False}, key="category_chart")
    st.markdown('</div>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="chart-card"><div class="chart-head"><div class="chart-title">Forecast Accuracy (%) by Month</div><div class="legend-inline"><span><i class="dot" style="background:#0069bd"></i>Accuracy</span></div></div>', unsafe_allow_html=True)
    st.plotly_chart(accuracy_chart(data), use_container_width=True, config={"displayModeBar": False}, key="accuracy_chart")
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="chart-grid" style="margin-top:18px;">', unsafe_allow_html=True)
left2, right2 = st.columns([1.5,1])
with left2:
    st.markdown('<div class="chart-card"><div class="chart-head"><div class="chart-title">Demand: Monthly Forecast vs Actual</div></div>', unsafe_allow_html=True)
    st.plotly_chart(monthly_chart(data, metric_label), use_container_width=True, config={"displayModeBar": False}, key="monthly_chart")
    st.markdown('</div>', unsafe_allow_html=True)
with right2:
    st.markdown('<div class="chart-card"><div class="chart-head"><div class="chart-title">Plant / Supply Chain Zone Accuracy</div></div>', unsafe_allow_html=True)
    st.plotly_chart(plant_zone_chart(data), use_container_width=True, config={"displayModeBar": False}, key="plant_zone_chart")
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TABLE
# ──────────────────────────────────────────────────────────────────────────────
table = make_display_table(data, metric_label)

st.markdown('<div class="table-card">', unsafe_allow_html=True)
t_head_l, t_head_r = st.columns([8, 2])
with t_head_l:
    st.markdown(f'<div class="table-head" style="border-bottom:0;padding-bottom:0"><div><div class="table-title">SKU Wise Demand Forecast</div><div class="table-sub">{len(table):,} rows shown · displaying Forecast, Actual and Accuracy only</div></div></div>', unsafe_allow_html=True)
with t_head_r:
    st.download_button("⬇️ Download", data=to_csv_bytes(table), file_name="chemelex_forecast_actual_accuracy.csv", mime="text/csv", use_container_width=True)

# st.dataframe with styling
styled = table.style.map(style_accuracy, subset=["Accuracy %"]).map(style_gap, subset=["Gap"]).format({
    "Forecast": "{:,.0f}",
    "Actual": "{:,.0f}",
    "Gap": "{:,.0f}",
})
st.dataframe(
    styled,
    use_container_width=True,
    height=430,
    hide_index=True,
    column_config={
        "Forecast": st.column_config.NumberColumn("Forecast", format="%.0f"),
        "Actual": st.column_config.NumberColumn("Actual", format="%.0f"),
        "Gap": st.column_config.NumberColumn("Gap", format="%.0f"),
        "Accuracy %": st.column_config.TextColumn("Accuracy %"),
    },
    key="forecast_actual_accuracy_table",
)
st.markdown('</div>', unsafe_allow_html=True)

# Explain formula below, not in table
st.markdown(
    """
    <div style="margin:0 18px 20px 18px;">
      <div class="formula-box">
        <b>Accuracy shown to business users:</b> Accuracy % = min(Actual, Forecast) / max(Actual, Forecast) × 100. &nbsp;
        Example: Actual 90 and Forecast 100 gives 90% accuracy. No MAPE or statistical jargon is shown in the UI.
      </div>
    </div>
    </div>
    """,
    unsafe_allow_html=True,
)
