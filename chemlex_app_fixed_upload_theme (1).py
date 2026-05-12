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
# SIDEBAR: ALWAYS VISIBLE DATA UPLOAD + THEME
# -----------------------------------------------------------------------------
st.sidebar.title("Chemelex Forecast")
st.sidebar.caption("Business forecast accuracy dashboard")

st.sidebar.markdown("---")
st.sidebar.markdown("### 1) Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload forecast CSV / Excel",
    type=["csv", "xlsx", "xls"],
    help="Preferred file: dashboard_forecast_accuracy_levelled.csv. Fallback supported: forecast_backtest_results.csv",
    key="forecast_data_upload",
)
st.sidebar.caption("If no file is uploaded, the app will use the CSV committed in the GitHub repo.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 2) Display")
theme_mode = st.sidebar.radio(
    "Theme",
    ["Light", "Dark"],
    index=0,
    horizontal=True,
    key="theme_mode_radio",
)
DARK_MODE = theme_mode == "Dark"

# -----------------------------------------------------------------------------
# THEME CSS
# -----------------------------------------------------------------------------
if DARK_MODE:
    COLORS = {
        "app_bg": "#0f172a",
        "card_bg": "#111827",
        "card_bg_2": "#1f2937",
        "text": "#f8fafc",
        "muted": "#cbd5e1",
        "border": "#334155",
        "primary": "#38bdf8",
        "accent_bg": "#0b2536",
        "alert_bg": "#2d1b08",
        "alert_border": "#f59e0b",
        "input_bg": "#111827",
    }
else:
    COLORS = {
        "app_bg": "#f6f7f9",
        "card_bg": "#ffffff",
        "card_bg_2": "#f8fafc",
        "text": "#102b5c",
        "muted": "#6b7280",
        "border": "#e3e6eb",
        "primary": "#0077bf",
        "accent_bg": "#eaf6ff",
        "alert_bg": "#fff4e4",
        "alert_border": "#f28c00",
        "input_bg": "#ffffff",
    }

st.markdown(
    f"""
<style>
    .stApp {{ background:{COLORS['app_bg']}; color:{COLORS['text']}; }}
    [data-testid="stHeader"] {{ display:none; }}
    [data-testid="stToolbar"] {{ display:none; }}
    [data-testid="stSidebar"] {{ background:{COLORS['card_bg']}; border-right:1px solid {COLORS['border']}; }}
    [data-testid="stSidebar"] * {{ color:{COLORS['text']}; }}
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] p {{ color:{COLORS['muted']}; }}
    .block-container {{ padding:0 !important; max-width:100% !important; }}

    .topbar {{
        height:58px; background:{COLORS['card_bg']}; border-bottom:1px solid {COLORS['border']};
        display:flex; align-items:center; justify-content:space-between; padding:0 24px;
        position:sticky; top:0; z-index:50;
    }}
    .brand {{ display:flex; align-items:center; gap:10px; font-weight:850; color:{COLORS['primary']}; font-size:25px; }}
    .brand-mark {{ width:30px; height:30px; border-radius:9px; background:linear-gradient(135deg,#0077bf,#0bafe8); display:grid; place-items:center; color:#fff; font-weight:900; }}
    .searchbar {{ width:430px; height:36px; border:1px solid {COLORS['border']}; border-radius:20px; display:flex; align-items:center; padding:0 14px; color:{COLORS['muted']}; font-size:14px; background:{COLORS['input_bg']}; }}
    .top-icons {{ display:flex; align-items:center; gap:15px; color:{COLORS['text']}; font-size:19px; }}

    .navrow {{ height:52px; background:{COLORS['card_bg']}; border-bottom:1px solid {COLORS['border']}; display:flex; align-items:center; gap:10px; padding:0 24px; }}
    .nav-pill {{ padding:9px 18px; border-radius:22px; color:{COLORS['muted']}; font-size:14px; border:1px solid transparent; font-weight:650; }}
    .nav-pill.active {{ border-color:{COLORS['primary']}; color:{COLORS['primary']}; background:{COLORS['accent_bg']}; }}

    .alert-strip {{ min-height:34px; background:{COLORS['alert_bg']}; border-left:5px solid {COLORS['alert_border']}; display:flex; align-items:center; justify-content:space-between; padding:7px 18px 7px 10px; color:{COLORS['text']}; font-size:13px; box-shadow:0 3px 9px rgba(0,0,0,.06); }}
    .alert-actions {{ display:flex; align-items:center; gap:10px; color:{COLORS['alert_border']}; font-weight:850; }}
    .page {{ padding:24px 28px 34px 28px; }}
    .page-title {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; gap:18px; }}
    .title-text h1 {{ margin:0; font-size:28px; color:{COLORS['text']}; font-weight:850; }}
    .title-text p {{ margin:4px 0 0 0; color:{COLORS['muted']}; font-size:13px; }}
    .asof {{ color:{COLORS['muted']}; font-size:12px; text-align:right; }}

    .filter-panel, .kpi-card, .panel {{ background:{COLORS['card_bg']}; border:1px solid {COLORS['border']}; box-shadow:0 2px 8px rgba(20,32,56,.08); }}
    .filter-panel {{ border-radius:18px; padding:14px 16px; margin-bottom:16px; }}
    .kpi-card {{ border-radius:14px; padding:15px 16px; height:100%; }}
    .kpi-label {{ font-size:12px; color:{COLORS['muted']}; margin-bottom:6px; }}
    .kpi-value {{ font-size:26px; line-height:1.1; color:{COLORS['text']}; font-weight:850; }}
    .kpi-sub {{ margin-top:5px; color:{COLORS['muted']}; font-size:12px; }}
    .kpi-green {{ color:#0f9d58 !important; }}
    .kpi-red {{ color:#e35645 !important; }}
    .kpi-orange {{ color:#e67700 !important; }}
    .panel {{ border-radius:18px; padding:16px; margin-bottom:16px; }}
    .panel-title {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }}
    .panel-title h3 {{ margin:0; font-size:15px; color:{COLORS['text']}; font-weight:800; }}
    .muted {{ color:{COLORS['muted']}; font-size:12px; }}
    .formula-box {{ background:{COLORS['accent_bg']}; border:1px solid {COLORS['border']}; color:{COLORS['text']}; border-radius:14px; padding:10px 14px; font-size:13px; }}
    .level-pill {{ display:inline-flex; align-items:center; padding:4px 10px; background:{COLORS['accent_bg']}; color:{COLORS['primary']}; border-radius:999px; font-size:12px; font-weight:800; border:1px solid {COLORS['border']}; }}
    .status-ok {{ color:#0f9d58; font-weight:800; }}
    .status-bad {{ color:#d43f32; font-weight:800; }}
    .status-warn {{ color:#e67700; font-weight:800; }}

    div[data-testid="stSelectbox"] label, div[data-testid="stTextInput"] label, div[data-testid="stDateInput"] label, div[data-testid="stMultiselect"] label, div[data-testid="stCheckbox"] label {{
        font-size:11px !important; color:{COLORS['muted']} !important; font-weight:850 !important; text-transform:uppercase !important; letter-spacing:.04em !important;
    }}
    div[data-baseweb="select"] > div, input {{ border-radius:14px !important; border-color:{COLORS['border']} !important; min-height:38px !important; background:{COLORS['input_bg']} !important; color:{COLORS['text']} !important; }}
    div[data-testid="stDataFrame"] {{ border-radius:16px !important; overflow:hidden !important; }}
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


def read_uploaded_or_default(uploaded):
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
    levels = ["Group"]
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
    actual_s = pd.to_numeric(actual, errors="coerce")
    forecast_s = pd.to_numeric(forecast, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(forecast_s == 0, np.nan, (actual_s / forecast_s) * 100)


def business_accuracy_scalar(actual, forecast):
    try:
        actual = float(actual)
        forecast = float(forecast)
        if forecast == 0 or not np.isfinite(forecast):
            return np.nan
        return (actual / forecast) * 100
    except Exception:
        return np.nan


def standardize_input(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower().strip(): c for c in df.columns}

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
        out = df.rename(columns={k: v for k, v in rename.items() if k is not None})
    else:
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
            cols = [c for c in ["codp_zone", "plant", "product_category_1"] if c in out.columns]
            out["line_item"] = out[cols].astype(str).agg(" | ".join, axis=1) if cols else "All"
        out["data_quality_flags"] = "OK"

    defaults = {
        "forecast_level": "Group",
        "line_item": "All",
        "codp_zone": "All",
        "plant": "All",
        "product_category_1": "All",
        "date": None,
        "year": None,
        "month": None,
        "target_metric": "order_qty_bu",
        "model": "Unknown",
        "split": "Unknown",
        "is_best": True,
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

    for col, default in [
        ("forecast_level", "Unknown"),
        ("line_item", "All"),
        ("codp_zone", "All"),
        ("plant", "All"),
        ("product_category_1", "All"),
        ("target_metric", "Unknown"),
        ("model", "Unknown"),
        ("split", "Unknown"),
        ("data_quality_flags", "OK"),
    ]:
        out[col] = out[col].fillna(default).astype(str)

    out["data_quality_flags"] = out["data_quality_flags"].replace("nan", "OK")
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
    keys = ["forecast_level", "line_item", "codp_zone", "plant", "product_category_1", "year", "month"]
    if df.empty:
        empty = pd.DataFrame(columns=keys + ["forecast", "actual", "data_quality_flags", "gap", "abs_gap", "accuracy_pct", "data_status"])
        return empty
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
    font_color = COLORS["text"]
    grid_color = "#263244" if DARK_MODE else "#edf0f4"
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=45, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color=font_color),
        legend=dict(orientation="h", y=1.1, x=1, xanchor="right"),
    )
    fig.update_xaxes(gridcolor=grid_color, zerolinecolor=grid_color)
    fig.update_yaxes(gridcolor=grid_color, zerolinecolor=grid_color)
    return fig


def kpi_card(label, value, sub="", cls=""):
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value {cls}'>{value}</div>
            <div class='kpi-sub'>{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------
raw_df, source_name = read_uploaded_or_default(uploaded_file)
try:
    data = standardize_input(raw_df)
except Exception as exc:
    st.error(f"Could not read forecast file: {exc}")
    st.stop()

if uploaded_file is not None:
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")
else:
    st.sidebar.info(f"Using repo/default file: {source_name}")

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
<div class='topbar'>
  <div class='brand'><div class='brand-mark'>C</div><span>Chemelex</span></div>
  <div class='searchbar'>🔍 Search or filter demand forecast line items</div>
  <div class='top-icons'><span>🔔</span><span>⚙️</span><span>👤</span></div>
</div>
<div class='navrow'>
  <div class='nav-pill active'>Material Forecast</div>
  <div class='nav-pill'>Forecast Analysis</div>
  <div class='nav-pill'>Forecast Level Clarity</div>
</div>
<div class='alert-strip'>
  <div>⚠️ Forecast accuracy uses business formula: <b>Actual ÷ Forecast × 100</b>. No MAPE or technical error metrics shown in this business view.</div>
  <div class='alert-actions'><span>Formula</span><span>Download</span></div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<div class='page'>", unsafe_allow_html=True)
st.markdown(
    f"""
<div class='page-title'>
  <div class='title-text'>
    <h1>Demand Forecasting Dashboard</h1>
    <p>Forecast vs Actual review by Month, CODP Zone, Plant, Product Category and Forecast Level</p>
  </div>
  <div class='asof'>Source: <b>{source_name}</b><br/>Theme: <b>{theme_mode}</b></div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# FILTERS
# -----------------------------------------------------------------------------
st.markdown("<div class='filter-panel'>", unsafe_allow_html=True)
f1, f2, f3, f4 = st.columns([1, 1, 1.2, 1.2])
with f1:
    year_options = sorted([int(y) for y in data["year"].dropna().unique().tolist()])
    selected_years = st.multiselect("Year", year_options, default=year_options, key="year_filter")
with f2:
    month_options = sorted([int(m) for m in data["month"].dropna().unique().tolist()])
    selected_months = st.multiselect("Month", month_options, default=month_options, format_func=lambda x: f"{x:02d}", key="month_filter")
with f3:
    level_options = sorted(data["forecast_level"].dropna().unique().tolist())
    default_level = "Group" if "Group" in level_options else level_options[0]
    selected_level = st.selectbox("Forecast Level", level_options, index=level_options.index(default_level), key="level_filter")
with f4:
    target_options = sorted(data["target_metric"].dropna().unique().tolist())
    default_target = "order_qty_bu" if "order_qty_bu" in target_options else target_options[0]
    selected_target = st.selectbox("Metric", target_options, index=target_options.index(default_target), key="metric_filter")

# Filter available options by level and target first
base_option_df = data[(data["forecast_level"] == selected_level) & (data["target_metric"] == selected_target)].copy()

f5, f6, f7, f8 = st.columns([1.4, 1.4, 1.4, 1.2])
with f5:
    zone_options = sorted([z for z in base_option_df["codp_zone"].dropna().unique().tolist() if str(z).strip()])
    selected_zones = st.multiselect("CODP / Supply Chain Zone", zone_options, default=zone_options, key="codp_filter")
with f6:
    plant_options = sorted([p for p in base_option_df["plant"].dropna().unique().tolist() if str(p).strip()])
    selected_plants = st.multiselect("Plant", plant_options, default=plant_options, key="plant_filter")
with f7:
    cat_options = sorted([c for c in base_option_df["product_category_1"].dropna().unique().tolist() if str(c).strip()])
    selected_cats = st.multiselect("Product Category", cat_options, default=cat_options, key="category_filter")
with f8:
    split_options = sorted(base_option_df["split"].dropna().unique().tolist())
    default_split = "split2" if "split2" in split_options else split_options[0]
    selected_split = st.selectbox("Split", split_options, index=split_options.index(default_split), key="split_filter")

f9, f10, f11 = st.columns([1.1, 1.6, 2])
with f9:
    best_only = st.checkbox("Best model only", value=True, key="best_model_filter")
with f10:
    model_df = base_option_df[base_option_df["split"] == selected_split]
    if best_only:
        model_df = model_df[model_df["is_best"].astype(str).str.lower().isin(["true", "1", "yes"])]
    models = ["All"] + sorted(model_df["model"].dropna().unique().tolist())
    selected_model = st.selectbox("Model", models, key="model_filter")
with f11:
    search_text = st.text_input("Search line item", placeholder="Search by CODP, plant, category or line item", key="search_filter")
st.markdown("</div>", unsafe_allow_html=True)

filtered = data.copy()
filtered = filtered[filtered["forecast_level"] == selected_level]
filtered = filtered[filtered["target_metric"] == selected_target]
if selected_years:
    filtered = filtered[filtered["year"].astype("Int64").isin(selected_years)]
if selected_months:
    filtered = filtered[filtered["month"].astype("Int64").isin(selected_months)]
if selected_zones:
    filtered = filtered[filtered["codp_zone"].isin(selected_zones)]
else:
    filtered = filtered.iloc[0:0]
if selected_plants:
    filtered = filtered[filtered["plant"].isin(selected_plants)]
else:
    filtered = filtered.iloc[0:0]
if selected_cats:
    filtered = filtered[filtered["product_category_1"].isin(selected_cats)]
else:
    filtered = filtered.iloc[0:0]
if selected_split:
    filtered = filtered[filtered["split"] == selected_split]
if best_only:
    filtered = filtered[filtered["is_best"].astype(str).str.lower().isin(["true", "1", "yes"])]
if selected_model != "All":
    filtered = filtered[filtered["model"] == selected_model]
if search_text.strip():
    s = search_text.strip().lower()
    searchable = (
        filtered["line_item"].astype(str) + " " +
        filtered["codp_zone"].astype(str) + " " +
        filtered["plant"].astype(str) + " " +
        filtered["product_category_1"].astype(str)
    ).str.lower()
    filtered = filtered[searchable.str.contains(s, na=False)]

view = aggregate_current_view(filtered)

# -----------------------------------------------------------------------------
# KPIS
# -----------------------------------------------------------------------------
total_forecast = view["forecast"].sum() if not view.empty else 0.0
total_actual = view["actual"].sum() if not view.empty else 0.0
overall_accuracy_value = business_accuracy_scalar(total_actual, total_forecast)
gap = total_actual - total_forecast
at_risk = view[~view["accuracy_pct"].between(90, 110, inclusive="both")].shape[0] if not view.empty else 0

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    kpi_card("Active Line Items", f"{len(view):,}", f"Level: {selected_level}")
with k2:
    kpi_card("Total Forecast", fmt_num(total_forecast), selected_target)
with k3:
    kpi_card("Total Actual", fmt_num(total_actual), "Actual demand")
with k4:
    kpi_tone = "kpi-green" if pd.notna(overall_accuracy_value) and 90 <= overall_accuracy_value <= 110 else "kpi-orange"
    kpi_card("Accuracy %", fmt_pct(overall_accuracy_value), "Actual ÷ Forecast × 100", kpi_tone)
with k5:
    kpi_card("At-risk Items", f"{at_risk:,}", "Outside 90–110%", "kpi-red" if at_risk else "kpi-green")

st.markdown("<br/>", unsafe_allow_html=True)
st.markdown(
    """
<div class='formula-box'>
<b>Business Accuracy Formula:</b> Accuracy % = Actual ÷ Forecast × 100. Example: Forecast 100 and Actual 120 → 120%; Forecast 100 and Actual 80 → 80%.
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Material Forecast", "Forecast Analysis", "Forecast Level Clarity"])

with tab1:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("<div class='panel'><div class='panel-title'><h3>Monthly Forecast vs Actual</h3><span class='muted'>Filtered view</span></div>", unsafe_allow_html=True)
        if view.empty:
            st.info("No rows match the selected filters.")
        else:
            month_chart = view.groupby(["year", "month"], as_index=False).agg(forecast=("forecast", "sum"), actual=("actual", "sum"))
            month_chart["period"] = month_chart["year"].astype(str) + "-" + month_chart["month"].astype(str).str.zfill(2)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=month_chart["period"], y=month_chart["forecast"], mode="lines+markers", name="Forecast", line=dict(color="#0077bf", width=3)))
            fig.add_trace(go.Scatter(x=month_chart["period"], y=month_chart["actual"], mode="lines+markers", name="Actual", line=dict(color="#f28c00", width=3)))
            st.plotly_chart(plotly_layout(fig, 330), use_container_width=True, key="monthly_forecast_actual")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='panel'><div class='panel-title'><h3>Accuracy by Month</h3><span class='muted'>Actual ÷ Forecast × 100</span></div>", unsafe_allow_html=True)
        if view.empty:
            st.info("No rows match the selected filters.")
        else:
            acc = view.groupby(["year", "month"], as_index=False).agg(forecast=("forecast", "sum"), actual=("actual", "sum"))
            acc["accuracy_pct"] = business_accuracy(acc["actual"], acc["forecast"])
            acc["period"] = acc["year"].astype(str) + "-" + acc["month"].astype(str).str.zfill(2)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=acc["period"], y=acc["accuracy_pct"], name="Accuracy %", marker_color="#0f9d58"))
            fig.add_hline(y=100, line_dash="dash", line_color="#6b7280")
            st.plotly_chart(plotly_layout(fig, 330), use_container_width=True, key="accuracy_by_month")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'><div class='panel-title'><h3>Forecast Table</h3><span class='muted'>One clean row per selected forecast level</span></div>", unsafe_allow_html=True)
    table = view.sort_values("abs_gap", ascending=False).copy()
    table["Period"] = table["year"].astype(str) + "-" + table["month"].astype(str).str.zfill(2)
    display = table[["Period", "forecast_level", "line_item", "codp_zone", "plant", "product_category_1", "forecast", "actual", "accuracy_pct", "gap", "data_status"]].rename(columns={
        "forecast_level": "Forecast Level",
        "line_item": "Line Item",
        "codp_zone": "CODP / Supply Chain Zone",
        "plant": "Plant",
        "product_category_1": "Product Category",
        "forecast": "Forecast",
        "actual": "Actual",
        "accuracy_pct": "Accuracy %",
        "gap": "Gap",
        "data_status": "Data Status",
    })
    st.dataframe(
        display,
        use_container_width=True,
        height=430,
        hide_index=True,
        column_config={
            "Forecast": st.column_config.NumberColumn(format="%.2f"),
            "Actual": st.column_config.NumberColumn(format="%.2f"),
            "Accuracy %": st.column_config.NumberColumn(format="%.1f%%"),
            "Gap": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.download_button("⬇️ Download visible forecast table", data=to_csv_bytes(display), file_name="chemelex_visible_forecast_table.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("<div class='panel'><div class='panel-title'><h3>Forecast vs Actual by Product Category</h3><span class='muted'>Current filters</span></div>", unsafe_allow_html=True)
        if view.empty:
            st.info("No rows match the selected filters.")
        else:
            cat = view.groupby("product_category_1", as_index=False).agg(forecast=("forecast", "sum"), actual=("actual", "sum")).sort_values("forecast", ascending=False).head(12)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=cat["product_category_1"], y=cat["forecast"], name="Forecast", marker_color="#0077bf"))
            fig.add_trace(go.Bar(x=cat["product_category_1"], y=cat["actual"], name="Actual", marker_color="#f28c00"))
            fig.update_layout(barmode="group")
            st.plotly_chart(plotly_layout(fig, 390), use_container_width=True, key="category_forecast_actual")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='panel'><div class='panel-title'><h3>Plant Accuracy</h3><span class='muted'>Actual ÷ Forecast × 100</span></div>", unsafe_allow_html=True)
        if view.empty:
            st.info("No rows match the selected filters.")
        else:
            plant = view.groupby("plant", as_index=False).agg(forecast=("forecast", "sum"), actual=("actual", "sum"))
            plant["accuracy_pct"] = business_accuracy(plant["actual"], plant["forecast"])
            plant = plant.sort_values("accuracy_pct", ascending=False).head(15)
            fig = go.Figure(go.Bar(x=plant["accuracy_pct"], y=plant["plant"], orientation="h", marker_color="#0f9d58"))
            fig.add_vline(x=100, line_dash="dash", line_color="#6b7280")
            st.plotly_chart(plotly_layout(fig, 390), use_container_width=True, key="plant_accuracy")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'><div class='panel-title'><h3>At-risk Forecast Lines</h3><span class='muted'>Outside 90–110%</span></div>", unsafe_allow_html=True)
    risk = view[~view["accuracy_pct"].between(90, 110, inclusive="both")].sort_values("abs_gap", ascending=False).head(100).copy()
    risk["Period"] = risk["year"].astype(str) + "-" + risk["month"].astype(str).str.zfill(2)
    risk_display = risk[["Period", "forecast_level", "line_item", "codp_zone", "plant", "product_category_1", "forecast", "actual", "accuracy_pct", "gap", "data_status"]].rename(columns={
        "forecast_level": "Forecast Level",
        "line_item": "Line Item",
        "codp_zone": "CODP / Supply Chain Zone",
        "plant": "Plant",
        "product_category_1": "Product Category",
        "forecast": "Forecast",
        "actual": "Actual",
        "accuracy_pct": "Accuracy %",
        "gap": "Gap",
        "data_status": "Data Status",
    })
    st.dataframe(risk_display, use_container_width=True, height=370, hide_index=True)
    st.download_button("⬇️ Download at-risk rows", data=to_csv_bytes(risk_display), file_name="chemelex_at_risk_forecast_rows.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("### Forecast Level Clarity")
    st.write("This page clarifies what one row means at each forecast level. The dashboard does not claim SKU/material-level forecasting unless material fields exist in the uploaded file.")
    level_summary = data.groupby("forecast_level", as_index=False).agg(
        rows=("forecast", "size"),
        total_forecast=("forecast", "sum"),
        total_actual=("actual", "sum"),
        unique_line_items=("line_item", "nunique"),
    )
    level_summary["accuracy_pct"] = business_accuracy(level_summary["total_actual"], level_summary["total_forecast"])
    level_summary = level_summary.rename(columns={
        "forecast_level": "Forecast Level",
        "rows": "Rows Available",
        "total_forecast": "Total Forecast",
        "total_actual": "Total Actual",
        "unique_line_items": "Unique Line Items",
        "accuracy_pct": "Accuracy %",
    })
    st.dataframe(level_summary, use_container_width=True, hide_index=True, column_config={"Accuracy %": st.column_config.NumberColumn(format="%.1f%%")})
    st.markdown("""
<div class='formula-box'>
<b>Important:</b> Group level means the most detailed available grain in the current file, typically CODP Zone + Plant + Product Category. It is not SKU/material level unless the file contains material_code or SKU columns.
</div>
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
