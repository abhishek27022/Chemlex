import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Chemelex Demand Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_LEVELLED_FILE = "dashboard_forecast_accuracy_levelled.csv"
DEFAULT_BACKTEST_FILE = "forecast_backtest_results.csv"

# -----------------------------------------------------------------------------
# CSS: clean Chemelex/Pwani-like UI
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
    .stApp { background:#f6f7f9; color:#25324b; }
    [data-testid="stHeader"] { display:none; }
    [data-testid="stToolbar"] { display:none; }
    [data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #e7e9ee; }
    [data-testid="stSidebar"] .block-container { padding:1.1rem 1rem 2rem 1rem !important; }
    .block-container { padding:0 !important; max-width:100% !important; }

    .topbar {
        height:58px; background:#fff; border-bottom:1px solid #e6e8ec;
        display:flex; align-items:center; justify-content:space-between; padding:0 24px;
        position:sticky; top:0; z-index:50;
    }
    .brand { display:flex; align-items:center; gap:10px; font-weight:800; color:#006fb9; font-size:25px; }
    .brand-mark { width:30px; height:30px; border-radius:9px; background:linear-gradient(135deg,#0077bf,#0bafe8); display:grid; place-items:center; color:#fff; font-weight:900; }
    .searchbar { width:430px; height:36px; border:1px solid #dcdfe5; border-radius:20px; display:flex; align-items:center; padding:0 14px; color:#9aa3ad; font-size:14px; background:#fff; }
    .top-icons { display:flex; align-items:center; gap:15px; color:#2f3542; font-size:19px; }

    .navrow { height:52px; background:#fff; border-bottom:1px solid #e6e8ec; display:flex; align-items:center; gap:10px; padding:0 24px; }
    .nav-pill { padding:9px 18px; border-radius:22px; color:#3f4654; font-size:14px; border:1px solid transparent; font-weight:600; }
    .nav-pill.active { border-color:#0077bf; color:#1f4f73; background:#eaf6ff; }

    .alert-strip { min-height:30px; background:#fff4e4; border-left:5px solid #f28c00; display:flex; align-items:center; justify-content:space-between; padding:6px 18px 6px 10px; color:#424a57; font-size:13px; box-shadow:0 3px 9px rgba(0,0,0,.04); }
    .alert-actions { display:flex; align-items:center; gap:10px; color:#ef8a00; font-weight:800; }

    .page { padding:24px 28px 34px 28px; }
    .page-title { display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; gap:18px; }
    .title-text h1 { margin:0; font-size:28px; color:#102b5c; font-weight:850; }
    .title-text p { margin:4px 0 0 0; color:#6b7280; font-size:13px; }
    .asof { color:#4b5563; font-size:12px; text-align:right; }

    .filter-panel { background:#fff; border:1px solid #e3e6eb; border-radius:18px; padding:14px 16px; box-shadow:0 2px 8px rgba(20,32,56,.04); margin-bottom:16px; }
    div[data-testid="stSelectbox"] label, div[data-testid="stTextInput"] label, div[data-testid="stDateInput"] label, div[data-testid="stMultiselect"] label, div[data-testid="stCheckbox"] label { font-size:11px !important; color:#7a8596 !important; font-weight:800 !important; text-transform:uppercase !important; letter-spacing:.04em !important; }
    div[data-baseweb="select"] > div, input { border-radius:14px !important; border-color:#dfe3e9 !important; min-height:38px !important; }

    .kpi-card { background:#fff; border:1px solid #e3e6eb; border-radius:14px; padding:15px 16px; box-shadow:0 2px 8px rgba(20,32,56,.04); height:100%; }
    .kpi-label { font-size:12px; color:#6b7280; margin-bottom:6px; }
    .kpi-value { font-size:26px; line-height:1.1; color:#12305d; font-weight:850; }
    .kpi-sub { margin-top:5px; color:#7b8492; font-size:12px; }
    .kpi-green { color:#0f9d58 !important; }
    .kpi-red { color:#e35645 !important; }
    .kpi-orange { color:#e67700 !important; }

    .panel { background:#fff; border:1px solid #e3e6eb; border-radius:18px; padding:16px; box-shadow:0 2px 8px rgba(20,32,56,.04); margin-bottom:16px; }
    .panel-title { display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
    .panel-title h3 { margin:0; font-size:15px; color:#384357; font-weight:800; }
    .muted { color:#8a93a3; font-size:12px; }
    .formula-box { background:#eff8ff; border:1px solid #cfeaff; color:#145374; border-radius:14px; padding:10px 14px; font-size:13px; }
    .formula-box b { color:#0b5c9a; }
    .level-pill { display:inline-flex; align-items:center; padding:4px 10px; background:#eaf6ff; color:#0b6fae; border-radius:999px; font-size:12px; font-weight:800; border:1px solid #bfe5ff; }
    .status-ok { color:#0f9d58; font-weight:800; }
    .status-bad { color:#d43f32; font-weight:800; }
    .status-warn { color:#e67700; font-weight:800; }
    div[data-testid="stDataFrame"] { border-radius:16px !important; overflow:hidden !important; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# DATA HELPERS
# -----------------------------------------------------------------------------
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None:
        df = pd.DataFrame()
    return df.to_csv(index=False).encode("utf-8")


def month_label(m):
    try:
        return pd.Timestamp(year=2025, month=int(m), day=1).strftime("%b")
    except Exception:
        return str(m)


def read_uploaded_or_default():
    uploaded = st.sidebar.file_uploader(
        "Upload forecast file",
        type=["csv", "xlsx", "xls"],
        help="Preferred file: dashboard_forecast_accuracy_levelled.csv",
    )

    if uploaded is not None:
        raw = uploaded.getvalue()
        if uploaded.name.lower().endswith(".csv"):
            return pd.read_csv(io.BytesIO(raw)), uploaded.name
        return pd.read_excel(io.BytesIO(raw)), uploaded.name

    candidates = [
        Path(DEFAULT_LEVELLED_FILE),
        Path(DEFAULT_BACKTEST_FILE),
        Path("/mnt/data/dashboard_forecast_accuracy_levelled.csv"),
        Path("/mnt/data/forecast_backtest_results.csv"),
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p), p.name

    return make_demo_data(), "generated_demo_data"


def make_demo_data():
    rng = np.random.default_rng(42)
    rows = []
    zones = ["Push / MPS", "BUFFER (at CODP)", "Pull / Kanban", "Push / RM"]
    plants = ["Chemelex - Trenton", "Chemelex - RWC/UCDC", "Chemelex - Pharr, T"]
    cats = ["PD / Heat Tracing Components", "PD / Fire and Performance Wiring", "PD / Floor Heating"]
    for m in range(1, 13):
        for z in zones:
            for p in plants:
                for c in cats:
                    f = float(rng.integers(1500, 20000))
                    a = f * float(rng.uniform(0.72, 1.22))
                    rows.append({
                        "level": "Group",
                        "dimension_value": f"{z} | {p} | {c}",
                        "codp_zone": z,
                        "plant": p,
                        "product_category_1": c,
                        "date": f"2025-{m:02d}-01",
                        "year": 2025,
                        "month": m,
                        "target": "order_qty_bu",
                        "model": "Best Model",
                        "split": "split2",
                        "is_best": True,
                        "actual": round(a, 2),
                        "forecast": round(f, 2),
                        "data_quality_flags": "OK",
                    })
    return pd.DataFrame(rows)


def business_accuracy(actual, forecast):
    actual = pd.to_numeric(actual, errors="coerce")
    forecast = pd.to_numeric(forecast, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(forecast == 0, np.nan, (actual / forecast) * 100)


def standardize_input(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower().strip(): c for c in df.columns}

    # Preferred Synthefy levelled output
    if "level" in lower and "actual" in lower and "forecast" in lower:
        rename = {
            lower.get("level"): "forecast_level",
            lower.get("dimension_value"): "line_item",
            lower.get("codp_zone"): "codp_zone",
            lower.get("plant"): "plant",
            lower.get("product_category_1"): "product_category_1",
            lower.get("date"): "date",
            lower.get("year"): "year",
            lower.get("month"): "month",
            lower.get("target"): "target_metric",
            lower.get("model"): "model",
            lower.get("split"): "split",
            lower.get("is_best"): "is_best",
            lower.get("actual"): "actual",
            lower.get("forecast"): "forecast",
            lower.get("data_quality_flags"): "data_quality_flags",
        }
        rename = {k: v for k, v in rename.items() if k is not None}
        out = df.rename(columns=rename)
    else:
        # Older backtest file fallback
        required = ["actual", "forecast"]
        missing = [c for c in required if c not in lower]
        if missing:
            raise ValueError("File must contain actual and forecast columns. Preferred file: dashboard_forecast_accuracy_levelled.csv")
        rename = {
            lower.get("codp_zone"): "codp_zone",
            lower.get("plant"): "plant",
            lower.get("product_category_1"): "product_category_1",
            lower.get("group_key"): "line_item",
            lower.get("date"): "date",
            lower.get("year"): "year",
            lower.get("month"): "month",
            lower.get("target"): "target_metric",
            lower.get("model"): "model",
            lower.get("split"): "split",
            lower.get("is_best"): "is_best",
            lower.get("actual"): "actual",
            lower.get("forecast"): "forecast",
        }
        out = df.rename(columns={k: v for k, v in rename.items() if k is not None})
        out["forecast_level"] = "Group"
        if "line_item" not in out.columns:
            out["line_item"] = out[[c for c in ["codp_zone", "plant", "product_category_1"] if c in out.columns]].astype(str).agg(" | ".join, axis=1)
        out["data_quality_flags"] = "OK"

    # Ensure expected columns
    defaults = {
        "forecast_level": "Group", "line_item": "All", "codp_zone": "All", "plant": "All", "product_category_1": "All",
        "date": None, "year": None, "month": None, "target_metric": "order_qty_bu", "model": "Unknown", "split": "Unknown", "is_best": True,
        "data_quality_flags": "OK",
    }
    for c, default in defaults.items():
        if c not in out.columns:
            out[c] = default

    out["actual"] = pd.to_numeric(out["actual"], errors="coerce").fillna(0)
    out["forecast"] = pd.to_numeric(out["forecast"], errors="coerce").fillna(0)
    out["gap"] = out["actual"] - out["forecast"]
    out["abs_gap"] = out["gap"].abs()
    out["accuracy_pct"] = business_accuracy(out["actual"], out["forecast"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").fillna(out["date"].dt.year).astype("Int64")
    out["month"] = pd.to_numeric(out["month"], errors="coerce").fillna(out["date"].dt.month).astype("Int64")

    out["forecast_level"] = out["forecast_level"].fillna("Unknown").astype(str)
    out["line_item"] = out["line_item"].fillna("All").astype(str)
    out["codp_zone"] = out["codp_zone"].fillna("All").astype(str)
    out["plant"] = out["plant"].fillna("All").astype(str)
    out["product_category_1"] = out["product_category_1"].fillna("All").astype(str)
    out["target_metric"] = out["target_metric"].fillna("Unknown").astype(str)
    out["model"] = out["model"].fillna("Unknown").astype(str)
    out["split"] = out["split"].fillna("Unknown").astype(str)
    out["data_quality_flags"] = out["data_quality_flags"].fillna("OK").replace("nan", "OK").astype(str)

    return out


def fmt_num(v):
    try:
        v = float(v)
        if abs(v) >= 1_000_000:
            return f"{v/1_000_000:.1f}M"
        if abs(v) >= 1_000:
            return f"{v/1_000:.1f}K"
        return f"{v:,.0f}"
    except Exception:
        return "-"


def fmt_pct(v):
    if pd.isna(v) or not np.isfinite(v):
        return "-"
    return f"{float(v):.1f}%"


def accuracy_status(v):
    if pd.isna(v) or not np.isfinite(v):
        return "No forecast"
    v = float(v)
    if 90 <= v <= 110:
        return "On track"
    if 75 <= v < 90 or 110 < v <= 125:
        return "Review"
    return "At risk"


def clean_data_status(flag, actual, forecast, acc):
    text = str(flag).strip()
    if not text or text.lower() in ["nan", "none", "ok"]:
        if forecast == 0 and actual > 0:
            return "No forecast with demand"
        if forecast == 0 and actual == 0:
            return "No demand / no forecast"
        if forecast < 0:
            return "Invalid negative forecast"
        return accuracy_status(acc)
    return text.replace("_", " ").title()


def aggregate_current_view(df):
    # Data is already levelled. Aggregate only to avoid duplicate rows from same dimensions/model after filtering.
    keys = ["forecast_level", "line_item", "codp_zone", "plant", "product_category_1", "year", "month"]
    grouped = df.groupby(keys, dropna=False, as_index=False).agg(
        forecast=("forecast", "sum"),
        actual=("actual", "sum"),
        data_quality_flags=("data_quality_flags", lambda x: "OK" if (x.astype(str).str.lower().isin(["ok", "nan", "none", ""]).all()) else "; ".join(sorted(set([i for i in x.astype(str) if i.lower() not in ["ok", "nan", "none", ""]]))[:3])),
    )
    grouped["gap"] = grouped["actual"] - grouped["forecast"]
    grouped["abs_gap"] = grouped["gap"].abs()
    grouped["accuracy_pct"] = business_accuracy(grouped["actual"], grouped["forecast"])
    grouped["data_status"] = [clean_data_status(f, a, fc, ac) for f, a, fc, ac in zip(grouped["data_quality_flags"], grouped["actual"], grouped["forecast"], grouped["accuracy_pct"])]
    return grouped


def plotly_layout(fig, height=340):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=45, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#384357"),
        legend=dict(orientation="h", y=1.1, x=1, xanchor="right"),
    )
    fig.update_xaxes(gridcolor="#edf0f4")
    fig.update_yaxes(gridcolor="#edf0f4")
    return fig


def kpi_card(label, value, sub="", cls=""):
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-value {cls}'>{value}</div>
        <div class='kpi-sub'>{sub}</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------
st.sidebar.markdown("### Data Upload")
st.sidebar.caption("Preferred file: dashboard_forecast_accuracy_levelled.csv")
raw_df, source_name = read_uploaded_or_default()
try:
    data = standardize_input(raw_df)
except Exception as exc:
    st.error(f"Could not read forecast file: {exc}")
    st.stop()

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown("""
<div class='topbar'>
  <div class='brand'><div class='brand-mark'>C</div><span>Chemelex</span></div>
  <div class='searchbar'>🔍 Search or filter demand forecast line items</div>
  <div class='top-icons'><span>🔔</span><span>⚙️</span><span>👤</span></div>
</div>
<div class='navrow'>
  <span class='nav-pill active'>Material Forecast</span>
  <span class='nav-pill'>Forecast Analysis</span>
  <span class='nav-pill'>Forecast Level Clarity</span>
</div>
<div class='alert-strip'>
  <div>⚠️ Review at-risk forecast line items before demand review. Business accuracy shown as Actual ÷ Forecast × 100.</div>
  <div class='alert-actions'><span>Formula</span><span>Accuracy %</span></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='page'>", unsafe_allow_html=True)
st.markdown(f"""
<div class='page-title'>
  <div class='title-text'>
    <h1>Demand Forecasting</h1>
    <p>Clean business view for Forecast vs Actual, hierarchy filters, and forecast accuracy.</p>
  </div>
  <div class='asof'>Source: <b>{source_name}</b><br/>Rows loaded: {len(data):,}</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FILTERS
# -----------------------------------------------------------------------------
st.markdown("<div class='filter-panel'>", unsafe_allow_html=True)
filter_cols = st.columns([1, 1, 1.2, 1.2, 1.2, 1.5])

level_options = sorted(data["forecast_level"].dropna().unique().tolist())
default_level = "Group" if "Group" in level_options else (level_options[0] if level_options else "All")
with filter_cols[0]:
    selected_level = st.selectbox("Forecast Level", level_options, index=level_options.index(default_level) if default_level in level_options else 0)

with filter_cols[1]:
    years = sorted([int(x) for x in data["year"].dropna().unique().tolist()])
    selected_years = st.multiselect("Year", years, default=years[-1:] if years else [])

with filter_cols[2]:
    month_options = sorted([int(x) for x in data["month"].dropna().unique().tolist()])
    selected_months = st.multiselect("Month", month_options, default=month_options)

with filter_cols[3]:
    plants = ["All"] + sorted([x for x in data["plant"].dropna().unique() if x != "All"])
    selected_plant = st.selectbox("Plant", plants)

with filter_cols[4]:
    cats = ["All"] + sorted([x for x in data["product_category_1"].dropna().unique() if x != "All"])
    selected_cat = st.selectbox("Product Category", cats)

with filter_cols[5]:
    search_text = st.text_input("Search Line Item", placeholder="Search CODP / plant / category")

filter_cols2 = st.columns([1.2, 1.2, 1, 1, 1.6])
with filter_cols2[0]:
    zones = ["All"] + sorted([x for x in data["codp_zone"].dropna().unique() if x != "All"])
    selected_zone = st.selectbox("CODP / Supply Chain Zone", zones)
with filter_cols2[1]:
    targets = sorted(data["target_metric"].dropna().unique().tolist())
    default_target = "order_qty_bu" if "order_qty_bu" in targets else (targets[0] if targets else "")
    selected_target = st.selectbox("Metric", targets, index=targets.index(default_target) if default_target in targets else 0)
with filter_cols2[2]:
    splits = sorted(data["split"].dropna().unique().tolist())
    default_split = "split2" if "split2" in splits else (splits[-1] if splits else "")
    selected_split = st.selectbox("Split", splits, index=splits.index(default_split) if default_split in splits else 0)
with filter_cols2[3]:
    best_only = st.checkbox("Best model only", value=True)
with filter_cols2[4]:
    model_df = data[data["is_best"].astype(str).str.lower().isin(["true", "1", "yes"])].copy() if best_only else data
    models = ["All"] + sorted(model_df["model"].dropna().unique().tolist())
    selected_model = st.selectbox("Model", models)
st.markdown("</div>", unsafe_allow_html=True)

filtered = data.copy()
filtered = filtered[filtered["forecast_level"] == selected_level]
if selected_years:
    filtered = filtered[filtered["year"].astype("Int64").isin(selected_years)]
if selected_months:
    filtered = filtered[filtered["month"].astype("Int64").isin(selected_months)]
if selected_plant != "All":
    filtered = filtered[filtered["plant"] == selected_plant]
if selected_cat != "All":
    filtered = filtered[filtered["product_category_1"] == selected_cat]
if selected_zone != "All":
    filtered = filtered[filtered["codp_zone"] == selected_zone]
if selected_target:
    filtered = filtered[filtered["target_metric"] == selected_target]
if selected_split:
    filtered = filtered[filtered["split"] == selected_split]
if best_only:
    filtered = filtered[filtered["is_best"].astype(str).str.lower().isin(["true", "1", "yes"])]
if selected_model != "All":
    filtered = filtered[filtered["model"] == selected_model]
if search_text.strip():
    s = search_text.strip().lower()
    searchable = (filtered["line_item"].astype(str) + " " + filtered["codp_zone"].astype(str) + " " + filtered["plant"].astype(str) + " " + filtered["product_category_1"].astype(str)).str.lower()
    filtered = filtered[searchable.str.contains(s, na=False)]

view = aggregate_current_view(filtered)

# -----------------------------------------------------------------------------
# KPIS
# -----------------------------------------------------------------------------
total_forecast = view["forecast"].sum()
total_actual = view["actual"].sum()
overall_accuracy = business_accuracy(total_actual, total_forecast)
overall_accuracy_value = float(overall_accuracy[0]) if len(overall_accuracy) else np.nan
gap = total_actual - total_forecast
at_risk = view[~view["accuracy_pct"].between(90, 110, inclusive="both")].shape[0]

k1, k2, k3, k4, k5 = st.columns(5)
with k1: kpi_card("Active Line Items", f"{len(view):,}", f"Level: {selected_level}")
with k2: kpi_card("Total Forecast", fmt_num(total_forecast), selected_target)
with k3: kpi_card("Total Actual", fmt_num(total_actual), "Actual demand")
with k4: kpi_card("Accuracy %", fmt_pct(overall_accuracy_value), "Actual ÷ Forecast × 100", "kpi-green" if 90 <= overall_accuracy_value <= 110 else "kpi-orange")
with k5: kpi_card("At-risk Items", f"{at_risk:,}", "Outside 90–110%", "kpi-red" if at_risk else "kpi-green")

st.markdown("<br/>", unsafe_allow_html=True)
st.markdown("""
<div class='formula-box'>
<b>Business Accuracy Formula:</b> Accuracy % = Actual ÷ Forecast × 100. Example: Forecast 100 and Actual 120 → 120%; Forecast 100 and Actual 80 → 80%.
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Material Forecast", "Forecast Analysis", "Forecast Level Clarity"])

with tab1:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("<div class='panel'><div class='panel-title'><h3>Monthly Forecast vs Actual</h3><span class='muted'>Filtered view</span></div>", unsafe_allow_html=True)
        month_chart = view.groupby(["year", "month"], as_index=False).agg(forecast=("forecast", "sum"), actual=("actual", "sum"))
        month_chart["period"] = month_chart["year"].astype(str) + "-" + month_chart["month"].astype(int).astype(str).str.zfill(2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=month_chart["period"], y=month_chart["forecast"], mode="lines+markers", name="Forecast", line=dict(color="#1f77b4", width=3)))
        fig.add_trace(go.Scatter(x=month_chart["period"], y=month_chart["actual"], mode="lines+markers", name="Actual", line=dict(color="#2ca02c", width=3)))
        st.plotly_chart(plotly_layout(fig, 335), use_container_width=True, key="monthly_forecast_actual")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='panel'><div class='panel-title'><h3>Accuracy by Month</h3><span class='muted'>Actual ÷ Forecast</span></div>", unsafe_allow_html=True)
        month_chart["accuracy_pct"] = business_accuracy(month_chart["actual"], month_chart["forecast"])
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=month_chart["period"], y=month_chart["accuracy_pct"], marker_color="#0b79bf", name="Accuracy %"))
        fig2.add_hline(y=100, line_dash="dash", line_color="#f28c00", annotation_text="100% aligned")
        st.plotly_chart(plotly_layout(fig2, 335), use_container_width=True, key="accuracy_by_month")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'><div class='panel-title'><h3>Forecast Line Items</h3><span class='level-pill'>One row per selected forecast level</span></div>", unsafe_allow_html=True)
    table = view.sort_values("abs_gap", ascending=False).copy()
    table["Period"] = table["year"].astype(str) + "-" + table["month"].astype(int).astype(str).str.zfill(2)
    table["Forecast Level"] = table["forecast_level"]
    table["Line Item"] = table["line_item"]
    table["CODP / Supply Chain Zone"] = table["codp_zone"]
    table["Plant"] = table["plant"]
    table["Product Category"] = table["product_category_1"]
    table["Forecast"] = table["forecast"].round(2)
    table["Actual"] = table["actual"].round(2)
    table["Accuracy %"] = table["accuracy_pct"].round(1)
    table["Gap"] = table["gap"].round(2)
    table["Data Status"] = table["data_status"]
    display_cols = ["Period", "Forecast Level", "Line Item", "CODP / Supply Chain Zone", "Plant", "Product Category", "Forecast", "Actual", "Accuracy %", "Gap", "Data Status"]
    st.dataframe(
        table[display_cols],
        use_container_width=True,
        hide_index=True,
        height=520,
        column_config={
            "Forecast": st.column_config.NumberColumn("Forecast", format="%.2f"),
            "Actual": st.column_config.NumberColumn("Actual", format="%.2f"),
            "Accuracy %": st.column_config.NumberColumn("Accuracy %", format="%.1f%%"),
            "Gap": st.column_config.NumberColumn("Gap", format="%.2f"),
        },
    )
    st.download_button("⬇️ Download visible forecast table", data=to_csv_bytes(table[display_cols]), file_name="chemelex_visible_forecast_table.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    left, right = st.columns([1, 1])
    with left:
        st.markdown("<div class='panel'><div class='panel-title'><h3>Forecast vs Actual by Product Category</h3><span class='muted'>Top 12 by forecast</span></div>", unsafe_allow_html=True)
        cat = view.groupby("product_category_1", as_index=False).agg(forecast=("forecast", "sum"), actual=("actual", "sum"))
        cat = cat.sort_values("forecast", ascending=False).head(12)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=cat["product_category_1"], y=cat["forecast"], name="Forecast", marker_color="#0b79bf"))
        fig3.add_trace(go.Bar(x=cat["product_category_1"], y=cat["actual"], name="Actual", marker_color="#2ca02c"))
        fig3.update_layout(barmode="group")
        st.plotly_chart(plotly_layout(fig3, 430), use_container_width=True, key="category_forecast_actual")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='panel'><div class='panel-title'><h3>At-risk Forecast Items</h3><span class='muted'>Outside 90–110%</span></div>", unsafe_allow_html=True)
        risk = table[~table["Accuracy %"].between(90, 110, inclusive="both")].copy().head(20)
        st.dataframe(risk[["Period", "Line Item", "Forecast", "Actual", "Accuracy %", "Gap", "Data Status"]], use_container_width=True, hide_index=True, height=430)
        st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='panel'><div class='panel-title'><h3>What level are we forecasting at?</h3><span class='muted'>Data governance note</span></div>", unsafe_allow_html=True)
    level_summary = data.groupby("forecast_level", as_index=False).agg(rows=("forecast", "size"), total_forecast=("forecast", "sum"), total_actual=("actual", "sum"))
    level_summary["accuracy_pct"] = business_accuracy(level_summary["total_actual"], level_summary["total_forecast"])
    level_summary["What this means"] = level_summary["forecast_level"].map({
        "Total": "One total business forecast line.",
        "CODP Zone": "One line per CODP / supply chain zone.",
        "Plant": "One line per plant.",
        "Product Category": "One line per product category.",
        "Group": "Most detailed current level: CODP Zone + Plant + Product Category.",
    }).fillna("Forecast level available in the uploaded file.")
    st.dataframe(level_summary.rename(columns={"forecast_level": "Forecast Level", "rows": "Rows", "total_forecast": "Forecast", "total_actual": "Actual", "accuracy_pct": "Accuracy %"}), use_container_width=True, hide_index=True)
    st.markdown("""
    <div class='formula-box' style='margin-top:12px;'>
    <b>Important:</b> The current Synthefy file does not include material code, SKU, or material description. So the most detailed reliable view is <b>Group = CODP Zone + Plant + Product Category</b>. Do not call this SKU-level unless Synthefy provides material-level fields.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
