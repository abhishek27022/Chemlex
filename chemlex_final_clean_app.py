import io
from pathlib import Path
from datetime import datetime

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
    initial_sidebar_state="collapsed",
)

APP_TITLE = "Chemelex Demand Forecasting Cockpit"
DEFAULT_FILE = "forecast_backtest_results.csv"

# -----------------------------------------------------------------------------
# CSS: clean Chemelex / Pwani-like UI
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
    .stApp { background:#f7f8fa; color:#25324b; }
    [data-testid="stHeader"] { display:none; }
    [data-testid="stToolbar"] { display:none; }
    [data-testid="stSidebar"] { display:none; }
    .block-container { padding:0 0 2rem 0 !important; max-width:100% !important; }

    .topbar {
        height:58px; background:#fff; border-bottom:1px solid #e6e8ec;
        display:flex; align-items:center; justify-content:space-between; padding:0 24px;
        position:sticky; top:0; z-index:50;
    }
    .brand { display:flex; align-items:center; gap:10px; font-weight:800; color:#006fb9; font-size:25px; letter-spacing:.2px; }
    .brand-mark { width:29px; height:29px; border-radius:8px; background:linear-gradient(135deg,#0077bf,#0bafe8); display:grid; place-items:center; color:#fff; font-weight:900; }
    .searchbar { width:430px; height:36px; border:1px solid #dcdfe5; border-radius:20px; display:flex; align-items:center; padding:0 14px; color:#9aa3ad; font-size:14px; background:#fff; }
    .top-icons { display:flex; align-items:center; gap:16px; color:#2f3542; font-size:20px; }
    .badge { margin-left:-12px; margin-top:-16px; background:#e53e3e; color:#fff; font-size:10px; border-radius:999px; padding:1px 6px; }

    .navrow { height:52px; background:#fff; border-bottom:1px solid #e6e8ec; display:flex; align-items:center; gap:10px; padding:0 24px; }
    .nav-pill { padding:9px 18px; border-radius:22px; color:#3f4654; font-size:14px; border:1px solid transparent; font-weight:500; }
    .nav-pill.active { border-color:#0077bf; color:#1f4f73; background:#eaf6ff; }

    .alert-strip { height:28px; background:#fff4e4; border-left:5px solid #f28c00; display:flex; align-items:center; justify-content:space-between; padding:0 18px 0 8px; color:#424a57; font-size:13px; box-shadow:0 3px 9px rgba(0,0,0,.05); }
    .alert-left { display:flex; align-items:center; gap:18px; }
    .alert-actions { display:flex; align-items:center; gap:10px; color:#ef8a00; font-weight:700; }

    .page { padding:24px 28px 34px 28px; }
    .page-title { display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; }
    .title-text h1 { margin:0; font-size:28px; color:#102b5c; font-weight:800; }
    .title-text p { margin:4px 0 0 0; color:#6b7280; font-size:13px; }
    .asof { color:#4b5563; font-size:12px; text-align:right; }

    .filter-panel { background:#fff; border:1px solid #e3e6eb; border-radius:18px; padding:14px 16px; box-shadow:0 2px 8px rgba(20,32,56,.04); margin-bottom:16px; }
    .filter-label { font-size:11px; color:#7a8596; font-weight:700; text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px; }
    div[data-testid="stSelectbox"] label, div[data-testid="stTextInput"] label, div[data-testid="stDateInput"] label, div[data-testid="stCheckbox"] label { font-size:11px !important; color:#7a8596 !important; font-weight:700 !important; text-transform:uppercase !important; letter-spacing:.04em !important; }
    div[data-baseweb="select"] > div, input { border-radius:14px !important; border-color:#dfe3e9 !important; min-height:38px !important; }

    .kpi-card { background:#fff; border:1px solid #e3e6eb; border-radius:14px; padding:15px 16px; box-shadow:0 2px 8px rgba(20,32,56,.04); height:100%; }
    .kpi-label { font-size:12px; color:#6b7280; margin-bottom:6px; }
    .kpi-value { font-size:26px; line-height:1.1; color:#12305d; font-weight:800; }
    .kpi-sub { margin-top:5px; color:#7b8492; font-size:12px; }
    .kpi-green { color:#0f9d58 !important; }
    .kpi-red { color:#e35645 !important; }
    .kpi-orange { color:#e67700 !important; }

    .panel { background:#fff; border:1px solid #e3e6eb; border-radius:18px; padding:16px; box-shadow:0 2px 8px rgba(20,32,56,.04); }
    .panel-title { display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
    .panel-title h3 { margin:0; font-size:15px; color:#384357; font-weight:700; }
    .muted { color:#8a93a3; font-size:12px; }

    .formula-box { background:#eff8ff; border:1px solid #cfeaff; color:#145374; border-radius:14px; padding:10px 14px; font-size:13px; }
    .formula-box b { color:#0b5c9a; }
    .download-float { float:right; }

    .table-caption { color:#8a93a3; font-size:13px; margin-top:-8px; margin-bottom:8px; }
    div[data-testid="stDataFrame"] { border-radius:16px !important; overflow:hidden !important; }

    .insight-card { display:flex; gap:12px; align-items:flex-start; padding:12px 4px; border-bottom:1px solid #edf0f4; }
    .insight-card:last-child { border-bottom:none; }
    .insight-icon { width:40px; height:40px; border-radius:50%; display:grid; place-items:center; font-size:18px; }
    .green-bg { background:#e8f7ee; color:#0f9d58; }
    .red-bg { background:#fff0ef; color:#d43f32; }
    .orange-bg { background:#fff3e2; color:#e67700; }
    .blue-bg { background:#eaf6ff; color:#0077bf; }

    .framework-row { display:flex; align-items:center; justify-content:space-between; padding:9px 10px; margin-bottom:7px; border-radius:10px; background:#f8fafc; border:1px solid #ecf0f4; font-size:12px; }
    .framework-row strong { color:#25324b; }

    .level-pill { display:inline-flex; align-items:center; padding:4px 10px; background:#eaf6ff; color:#0b6fae; border-radius:999px; font-size:12px; font-weight:700; border:1px solid #bfe5ff; }
    .negative { color:#d43f32; font-weight:700; }
    .positive { color:#0f9d58; font-weight:700; }

    .stTabs [data-baseweb="tab-list"] { gap:10px; background:#fff; border-bottom:1px solid #e6e8ec; padding-left:20px; }
    .stTabs [data-baseweb="tab"] { border-radius:20px 20px 0 0; padding:10px 18px; font-weight:600; color:#4b5563; }
    .stTabs [aria-selected="true"] { background:#eaf6ff !important; color:#0b6fae !important; border:1px solid #8ed0ff; border-bottom:0; }

    @media (max-width: 900px) { .searchbar{display:none;} .topbar{padding:0 14px;} .brand{font-size:20px;} .page{padding:18px 14px;} }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# DATA HELPERS
# -----------------------------------------------------------------------------
def month_name_to_num(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer, float, np.floating)):
        return int(value)
    text = str(value).strip()
    try:
        return int(text)
    except Exception:
        pass
    month_map = {m.lower(): i for i, m in enumerate([
        "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"
    ], start=1)}
    return month_map.get(text.lower()[:3], month_map.get(text.lower(), np.nan))


def clean_name(x):
    return str(x).replace("  ", " ").strip() if pd.notna(x) else "Unknown"


def business_accuracy(actual, forecast):
    """Business-facing accuracy: Actual ÷ Forecast × 100.
    Example: actual=120, forecast=100 -> 120%; actual=80, forecast=100 -> 80%.
    """
    actual = pd.to_numeric(actual, errors="coerce")
    forecast = pd.to_numeric(forecast, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(forecast == 0, np.nan, (actual / forecast) * 100)
    return acc


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False)
def load_csv_or_excel(file_bytes, filename):
    name = filename.lower()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes))
    raise ValueError("Upload a CSV or Excel file.")


def load_default_file():
    candidates = [
        Path(DEFAULT_FILE),
        Path("/mnt/data/forecast_backtest_results.csv"),
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p), p.name
    return make_demo_data(), "generated_demo_data"


def make_demo_data():
    rows = []
    zones = ["Push / MPS", "BUFFER (at CODP)", "Pull / Kanban", "Push / RM"]
    plants = ["Chemelex - Trenton", "Chemelex - RWC/UCDC", "Chemelex - Pharr, T"]
    cats = [
        "PD / Heat Tracing Components",
        "PD / Fire and Performance Wiring",
        "PD / Floor Heating",
        "PD / MI Heat Tracing",
        "PD / Control, Monitoring & Power Distribution",
    ]
    models = ["Best Statistical Model", "Moving Average", "Seasonal Naive"]
    rng = np.random.default_rng(42)
    for y in [2024, 2025]:
        for m in range(1, 13):
            for z in zones:
                for p in plants:
                    for c in cats:
                        base = rng.integers(2500, 45000)
                        forecast = base * rng.uniform(0.85, 1.20)
                        actual = forecast * rng.uniform(0.75, 1.25)
                        rows.append({
                            "date": f"{y}-{m:02d}-01",
                            "codp_zone": z,
                            "plant": p,
                            "product_category_1": c,
                            "group_key": f"{z} | {p} | {c}",
                            "target": "Sum of Order quantity BU",
                            "split": "test" if y == 2025 else "train",
                            "model": "Best Statistical Model",
                            "actual": round(actual, 2),
                            "forecast": round(forecast, 2),
                            "is_best": True,
                            "year": y,
                            "month": m,
                        })
                        for model in models[1:]:
                            f2 = forecast * rng.uniform(0.9, 1.1)
                            rows.append({
                                "date": f"{y}-{m:02d}-01", "codp_zone": z, "plant": p, "product_category_1": c,
                                "group_key": f"{z} | {p} | {c}", "target": "Sum of Order quantity BU", "split": "test" if y == 2025 else "train",
                                "model": model, "actual": round(actual, 2), "forecast": round(f2, 2), "is_best": False,
                                "year": y, "month": m,
                            })
    return pd.DataFrame(rows)


def standardize(df):
    """Normalize either forecast_backtest_results.csv or Synthefy aggregate output."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Map aggregate/Synthefy style columns if present.
    colmap = {
        "Supply_Chain_Zone": "codp_zone",
        "Plant Name": "plant",
        "Product 1 Category/Division Name": "product_category_1",
        "Sum of Order quantity BU": "actual",
        "Sum of Order Value in USD from LC": "actual_value",
        "Year": "year",
        "Month": "month",
    }
    for src, dst in colmap.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    required_base = ["codp_zone", "plant", "product_category_1", "actual"]
    missing = [c for c in required_base if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    if "forecast" not in df.columns:
        # If raw actual-only data comes from Synthefy, create a simple rolling baseline forecast.
        sort_cols = [c for c in ["year", "month"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)
        df["forecast"] = (
            df.groupby(["codp_zone", "plant", "product_category_1"])["actual"]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
            .fillna(df["actual"])
        )

    if "year" not in df.columns:
        if "date" in df.columns:
            dt = pd.to_datetime(df["date"], errors="coerce")
            df["year"] = dt.dt.year
        else:
            df["year"] = datetime.now().year

    if "month" not in df.columns:
        if "date" in df.columns:
            dt = pd.to_datetime(df["date"], errors="coerce")
            df["month"] = dt.dt.month
        else:
            df["month"] = 1

    df["month"] = df["month"].apply(month_name_to_num).astype("Int64")
    df["month_name"] = pd.to_datetime(df["month"].astype(str), format="%m", errors="coerce").dt.month_name()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01", errors="coerce")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for c in ["codp_zone", "plant", "product_category_1"]:
        df[c] = df[c].apply(clean_name)

    if "group_key" not in df.columns:
        df["group_key"] = df["codp_zone"] + " | " + df["plant"] + " | " + df["product_category_1"]

    if "model" not in df.columns:
        df["model"] = "Forecast Baseline"
    if "target" not in df.columns:
        df["target"] = "Volume"
    if "split" not in df.columns:
        df["split"] = "actual"
    if "is_best" not in df.columns:
        df["is_best"] = True

    df["actual"] = pd.to_numeric(df["actual"], errors="coerce").fillna(0)
    df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce").fillna(0)
    df["gap"] = df["actual"] - df["forecast"]
    df["abs_gap"] = df["gap"].abs()
    df["accuracy_pct"] = business_accuracy(df["actual"], df["forecast"])
    df["accuracy_pct"] = pd.Series(df["accuracy_pct"]).replace([np.inf, -np.inf], np.nan)
    df["risk"] = np.select(
        [df["accuracy_pct"].between(90, 110, inclusive="both"), df["accuracy_pct"].between(75, 125, inclusive="both")],
        ["On Track", "Watch"],
        default="At Risk",
    )
    return df


def aggregate_for_level(df, level_cols):
    group_cols = ["year", "month", "month_name"] + level_cols
    out = df.groupby(group_cols, dropna=False, as_index=False).agg(
        forecast=("forecast", "sum"),
        actual=("actual", "sum"),
        abs_gap=("abs_gap", "sum"),
        records=("group_key", "count"),
    )
    out["gap"] = out["actual"] - out["forecast"]
    out["accuracy_pct"] = business_accuracy(out["actual"], out["forecast"])
    out["risk"] = np.select(
        [out["accuracy_pct"].between(90, 110, inclusive="both"), out["accuracy_pct"].between(75, 125, inclusive="both")],
        ["On Track", "Watch"],
        default="At Risk",
    )
    out["line_item"] = out[level_cols].astype(str).agg(" | ".join, axis=1)
    return out


def weighted_accuracy(df):
    total_forecast = df["forecast"].sum()
    if total_forecast == 0:
        return np.nan
    return (df["actual"].sum() / total_forecast) * 100


def format_num(v):
    if pd.isna(v):
        return "—"
    return f"{v:,.0f}"


def format_pct(v):
    if pd.isna(v):
        return "—"
    return f"{v:,.1f}%"


def kpi_card(label, value, sub="", tone=""):
    tone_cls = ""
    if tone == "green": tone_cls = "kpi-green"
    if tone == "red": tone_cls = "kpi-red"
    if tone == "orange": tone_cls = "kpi-orange"
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-value {tone_cls}'>{value}</div>
        <div class='kpi-sub'>{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def fig_style(fig, height=340):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=35, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=12, color="#4b5563"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#edf0f4", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#edf0f4", zeroline=False)
    return fig


def make_group_bar(df, category_col):
    d = df.groupby(category_col, as_index=False).agg(forecast=("forecast", "sum"), actual=("actual", "sum"))
    d = d.sort_values("forecast", ascending=False).head(10)
    fig = go.Figure()
    fig.add_bar(x=d[category_col], y=d["forecast"], name="Forecast", marker_color="#0077bf")
    fig.add_bar(x=d[category_col], y=d["actual"], name="Actual", marker_color="#2fb35a")
    fig.update_layout(barmode="group", title="Forecast vs Actual by Category")
    return fig_style(fig, 320)


def make_monthly_chart(df):
    d = df.groupby(["year", "month", "month_name"], as_index=False).agg(forecast=("forecast", "sum"), actual=("actual", "sum"))
    d = d.sort_values(["year", "month"])
    d["period"] = d["month_name"].str.slice(0, 3) + " " + d["year"].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["period"], y=d["forecast"], mode="lines+markers", name="Forecast", line=dict(color="#0077bf", width=3)))
    fig.add_trace(go.Scatter(x=d["period"], y=d["actual"], mode="lines+markers", name="Actual", line=dict(color="#16883a", width=3)))
    fig.update_layout(title="Monthly Forecast vs Actual")
    return fig_style(fig, 360)


def make_accuracy_chart(df):
    d = df.groupby(["year", "month", "month_name"], as_index=False).agg(forecast=("forecast", "sum"), actual=("actual", "sum"))
    d["accuracy_pct"] = business_accuracy(d["actual"], d["forecast"])
    d = d.sort_values(["year", "month"])
    d["period"] = d["month_name"].str.slice(0, 3) + " " + d["year"].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["period"], y=d["accuracy_pct"], mode="lines+markers+text", text=[format_pct(x) for x in d["accuracy_pct"]], textposition="top center", name="Accuracy %", line=dict(color="#0067c0", width=3)))
    fig.add_hrect(y0=90, y1=110, fillcolor="#e8f7ee", opacity=0.45, line_width=0)
    fig.add_hline(y=100, line_dash="dot", line_color="#94a3b8")
    fig.update_layout(title="Forecast Accuracy (%) by Month", yaxis_title="Actual ÷ Forecast × 100")
    fig.update_yaxes(ticksuffix="%")
    return fig_style(fig, 320)


def make_zone_plant_chart(df):
    d = df.groupby(["codp_zone", "plant"], as_index=False).agg(forecast=("forecast", "sum"), actual=("actual", "sum"))
    d["accuracy_pct"] = business_accuracy(d["actual"], d["forecast"])
    d = d.sort_values("accuracy_pct")
    fig = go.Figure()
    colors = ["#e35645" if x < 75 or x > 125 else "#f59e0b" if x < 90 or x > 110 else "#2fb35a" for x in d["accuracy_pct"]]
    fig.add_bar(y=d["codp_zone"] + " | " + d["plant"], x=d["accuracy_pct"], orientation="h", marker_color=colors, text=[format_pct(x) for x in d["accuracy_pct"]], textposition="outside")
    fig.add_vrect(x0=90, x1=110, fillcolor="#e8f7ee", opacity=0.35, line_width=0)
    fig.add_vline(x=100, line_dash="dot", line_color="#94a3b8")
    fig.update_layout(title="CODP Zone / Plant Accuracy", xaxis_title="Accuracy %")
    fig.update_xaxes(ticksuffix="%")
    return fig_style(fig, 360)

# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------
with st.sidebar:
    uploaded = st.file_uploader("Upload forecast backtest CSV/XLSX", type=["csv", "xlsx", "xls"])

try:
    if uploaded is not None:
        raw = load_csv_or_excel(uploaded.getvalue(), uploaded.name)
        source_name = uploaded.name
    else:
        raw, source_name = load_default_file()
    df = standardize(raw)
except Exception as e:
    st.error(f"Unable to load data: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class='topbar'>
  <div class='brand'><div class='brand-mark'>◆</div><span>chemelex</span></div>
  <div class='searchbar'>🔍 &nbsp; Search material, plant, category...</div>
  <div class='top-icons'><span>🔔</span><span class='badge'>14</span><span>⚙️</span><span style='background:#0b67ad;color:#fff;border-radius:999px;padding:6px 9px;font-size:13px;'>AM</span></div>
</div>
<div class='navrow'>
  <span class='nav-pill active'>Demand Forecast</span>
  <span class='nav-pill'>Forecast Accuracy</span>
  <span class='nav-pill'>Demand Plan</span>
  <span class='nav-pill'>Material View</span>
  <span class='nav-pill'>Supply Chain</span>
  <span class='nav-pill'>Insights</span>
</div>
<div class='alert-strip'>
  <div class='alert-left'><span>{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</span><span>Demand forecast accuracy view · Source: {source_name}</span><span style='color:#ef8a00'>↗</span></div>
  <div class='alert-actions'><span>‹</span><span>1 of 14</span><span>›</span><span>×</span></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='page'>", unsafe_allow_html=True)
st.markdown("""
<div class='page-title'>
  <div class='title-text'>
    <h1>Chemelex Demand Forecasting Cockpit</h1>
    <p>Clean end-user view of forecast, actual, and business accuracy at CODP / Plant / Product Category / Material level.</p>
  </div>
  <div class='asof'>Data as of<br><b>Latest loaded file</b></div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FILTERS
# -----------------------------------------------------------------------------
st.markdown("<div class='filter-panel'>", unsafe_allow_html=True)
filter_cols = st.columns([1.1, 1.1, 1.5, 1.8, 2.2, 1.5, 2.2, 1.1])

years = sorted([int(x) for x in df["year"].dropna().unique()])
months_present = sorted([int(x) for x in df["month"].dropna().unique()])
month_labels = {i: datetime(2000, i, 1).strftime("%B") for i in range(1, 13)}

with filter_cols[0]:
    year_sel = st.selectbox("Year", ["All"] + years, index=len(years) if years else 0)
with filter_cols[1]:
    month_options = ["All"] + [month_labels[m] for m in months_present]
    month_sel = st.selectbox("Month", month_options, index=0)
with filter_cols[2]:
    zone_sel = st.selectbox("CODP / Supply Chain Zone", ["All"] + sorted(df["codp_zone"].dropna().unique().tolist()))
with filter_cols[3]:
    plant_sel = st.selectbox("Plant", ["All"] + sorted(df["plant"].dropna().unique().tolist()))
with filter_cols[4]:
    cat_sel = st.selectbox("Product Category", ["All"] + sorted(df["product_category_1"].dropna().unique().tolist()))
with filter_cols[5]:
    target_sel = st.selectbox("Metric", ["All"] + sorted(df["target"].dropna().astype(str).unique().tolist()))
with filter_cols[6]:
    search = st.text_input("Search", placeholder="Search material / group key")
with filter_cols[7]:
    best_only = st.checkbox("Best only", value=True)
st.markdown("</div>", unsafe_allow_html=True)

fdf = df.copy()
if year_sel != "All":
    fdf = fdf[fdf["year"] == int(year_sel)]
if month_sel != "All":
    month_num = {v: k for k, v in month_labels.items()}[month_sel]
    fdf = fdf[fdf["month"] == month_num]
if zone_sel != "All":
    fdf = fdf[fdf["codp_zone"] == zone_sel]
if plant_sel != "All":
    fdf = fdf[fdf["plant"] == plant_sel]
if cat_sel != "All":
    fdf = fdf[fdf["product_category_1"] == cat_sel]
if target_sel != "All":
    fdf = fdf[fdf["target"].astype(str) == target_sel]
if best_only and "is_best" in fdf.columns:
    fdf = fdf[fdf["is_best"].astype(bool)]
if search.strip():
    s = search.strip().lower()
    fdf = fdf[fdf["group_key"].astype(str).str.lower().str.contains(s, na=False)]

if fdf.empty:
    st.warning("No data available for the selected filters. Please clear filters or upload another file.")
    st.stop()

# -----------------------------------------------------------------------------
# KPI ROW
# -----------------------------------------------------------------------------
active_materials = fdf["group_key"].nunique()
total_forecast = fdf["forecast"].sum()
total_actual = fdf["actual"].sum()
gap = total_actual - total_forecast
acc = weighted_accuracy(fdf)
at_risk = fdf.groupby("group_key", as_index=False).agg(forecast=("forecast", "sum"), actual=("actual", "sum"))
at_risk["accuracy_pct"] = business_accuracy(at_risk["actual"], at_risk["forecast"])
at_risk_count = ((at_risk["accuracy_pct"] < 75) | (at_risk["accuracy_pct"] > 125)).sum()
on_track_count = at_risk["accuracy_pct"].between(90, 110, inclusive="both").sum()

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1: kpi_card("Active Materials", f"{active_materials:,}", "unique forecast groups")
with k2: kpi_card("Total Forecast", f"{format_num(total_forecast)}", "selected period")
with k3: kpi_card("Total Actual", f"{format_num(total_actual)}", "selected period")
with k4: kpi_card("Forecast Accuracy", format_pct(acc), "Actual ÷ Forecast × 100", "green" if 90 <= acc <= 110 else "orange")
with k5: kpi_card("Forecast Gap", f"{format_num(gap)}", "Actual - Forecast", "red" if gap < 0 else "green")
with k6: kpi_card("At-risk Materials", f"{int(at_risk_count):,}", "accuracy <75% or >125%", "red" if at_risk_count else "green")

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Material Forecast", "Forecast Analysis", "Accuracy Workbench"])

with tab1:
    r1c1, r1c2 = st.columns([1.15, 1])
    with r1c1:
        st.markdown("<div class='panel'><div class='panel-title'><h3>Categories</h3><span class='muted'>Forecast vs Actual</span></div>", unsafe_allow_html=True)
        st.plotly_chart(make_group_bar(fdf, "product_category_1"), use_container_width=True, key="category_bar")
        st.markdown("</div>", unsafe_allow_html=True)
    with r1c2:
        st.markdown("<div class='panel'><div class='panel-title'><h3>Variance Analysis</h3><span class='level-pill'>Accuracy: " + format_pct(acc) + "</span></div>", unsafe_allow_html=True)
        st.plotly_chart(make_accuracy_chart(fdf), use_container_width=True, key="accuracy_month_chart")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    h1, h2 = st.columns([4, 1])
    with h1:
        st.markdown("### SKU / Material-wise Forecast Accuracy")
        st.markdown(f"<div class='table-caption'>{active_materials:,} groups · One row is one forecast line item at the selected grain</div>", unsafe_allow_html=True)
    with h2:
        level_choice = st.selectbox(
            "Forecast Level",
            [
                "CODP Zone",
                "Plant",
                "Product Category",
                "CODP + Plant",
                "CODP + Plant + Category",
                "Material / Group Key",
            ],
            index=4,
            key="level_choice_main",
        )

    level_map = {
        "CODP Zone": ["codp_zone"],
        "Plant": ["plant"],
        "Product Category": ["product_category_1"],
        "CODP + Plant": ["codp_zone", "plant"],
        "CODP + Plant + Category": ["codp_zone", "plant", "product_category_1"],
        "Material / Group Key": ["codp_zone", "plant", "product_category_1", "group_key"],
    }
    level_cols = level_map[level_choice]
    material = aggregate_for_level(fdf, level_cols)
    material = material.sort_values("abs_gap", ascending=False).head(300).copy()

    display = material.copy()
    display["Forecast Level"] = level_choice
    display["Forecast Line Item"] = display["line_item"]
    display["Forecast"] = display["forecast"].round(0)
    display["Actual"] = display["actual"].round(0)
    display["Accuracy %"] = display["accuracy_pct"].round(1)
    display["Gap"] = display["gap"].round(0)
    display["Formula"] = "Actual ÷ Forecast × 100"
    visible_cols = ["Forecast Level", "Forecast Line Item", "Forecast", "Actual", "Accuracy %", "Gap", "Formula", "risk"]
    st.dataframe(
        display[visible_cols],
        use_container_width=True,
        height=460,
        hide_index=True,
        column_config={
            "Forecast": st.column_config.NumberColumn("Forecast", format="%.0f"),
            "Actual": st.column_config.NumberColumn("Actual", format="%.0f"),
            "Accuracy %": st.column_config.ProgressColumn("Accuracy %", min_value=0, max_value=150, format="%.1f%%"),
            "Gap": st.column_config.NumberColumn("Gap", format="%.0f"),
            "risk": st.column_config.TextColumn("Status"),
        },
        key="material_table",
    )
    st.download_button("⬇️ Download visible table", data=to_csv_bytes(display[visible_cols]), file_name="chemelex_forecast_accuracy_table.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    a, b = st.columns([2.2, 1])
    with a:
        st.markdown("<div class='panel'><div class='panel-title'><h3>Demand: Monthly Forecast vs Actual</h3><span class='muted'>simple business view</span></div>", unsafe_allow_html=True)
        st.plotly_chart(make_monthly_chart(fdf), use_container_width=True, key="monthly_forecast_actual")
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='panel'><div class='panel-title'><h3>Key Insights</h3><span class='muted'>auto-generated</span></div>", unsafe_allow_html=True)
        best_plant = fdf.groupby("plant", as_index=False).agg(forecast=("forecast", "sum"), actual=("actual", "sum"))
        best_plant["accuracy_pct"] = business_accuracy(best_plant["actual"], best_plant["forecast"])
        best_plant["distance"] = (best_plant["accuracy_pct"] - 100).abs()
        bp = best_plant.sort_values("distance").head(1).iloc[0]
        weakest = best_plant.sort_values("distance", ascending=False).head(1).iloc[0]
        top_gap = material.sort_values("abs_gap", ascending=False).head(1).iloc[0]
        st.markdown(f"""
        <div class='insight-card'><div class='insight-icon green-bg'>🏆</div><div><b class='positive'>Best Performing Plant</b><br>{bp['plant']}<br><span class='muted'>{format_pct(bp['accuracy_pct'])} accuracy</span></div></div>
        <div class='insight-card'><div class='insight-icon red-bg'>⚠️</div><div><b class='negative'>Weakest Plant</b><br>{weakest['plant']}<br><span class='muted'>{format_pct(weakest['accuracy_pct'])} accuracy</span></div></div>
        <div class='insight-card'><div class='insight-icon orange-bg'>📈</div><div><b style='color:#e67700'>Top Gap Line Item</b><br>{top_gap['line_item']}<br><span class='muted'>{format_num(top_gap['gap'])} gap · {format_pct(top_gap['accuracy_pct'])} accuracy</span></div></div>
        <div class='insight-card'><div class='insight-icon blue-bg'>🎯</div><div><b style='color:#0b6fae'>Recommended Action</b><br>Review top gap line items with sales and supply planning. Confirm if uplift/drop is real demand or forecast bias.</div></div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("<div class='panel'><div class='panel-title'><h3>CODP Zone / Plant Analysis</h3><span class='muted'>accuracy by planning level</span></div>", unsafe_allow_html=True)
        st.plotly_chart(make_zone_plant_chart(fdf), use_container_width=True, key="zone_plant_chart")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='panel'><div class='panel-title'><h3>Demand Forecast Framework</h3><span class='muted'>what this view means</span></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='framework-row'><strong>Forecast</strong><span>{format_num(total_forecast)}</span></div>
        <div class='framework-row'><strong>Actual</strong><span>{format_num(total_actual)}</span></div>
        <div class='framework-row'><strong>Accuracy</strong><span>{format_pct(acc)}</span></div>
        <div class='framework-row'><strong>Gap</strong><span class='{ 'negative' if gap < 0 else 'positive' }'>{format_num(gap)}</span></div>
        <div class='formula-box'><b>Formula:</b> Accuracy = Actual ÷ Forecast × 100.<br>Example: Actual 120 and Forecast 100 gives 120%; Actual 80 and Forecast 100 gives 80%.</div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("### Forecast Level Clarity")
    st.markdown("This section makes it clear what level the forecast is being reviewed at. Select a grain and the table becomes one row per forecast line item at that exact level.")
    lvl = st.radio(
        "Choose forecast review level",
        ["CODP Zone", "Plant", "Product Category", "CODP + Plant", "CODP + Plant + Category", "Material / Group Key"],
        horizontal=True,
        key="level_choice_workbench",
        index=4,
    )
    lvl_cols = level_map[lvl]
    lvl_df = aggregate_for_level(fdf, lvl_cols).sort_values("abs_gap", ascending=False).copy()
    lvl_df["Forecast Level"] = lvl
    lvl_df["Line Item"] = lvl_df["line_item"]
    lvl_df["Forecast"] = lvl_df["forecast"].round(0)
    lvl_df["Actual"] = lvl_df["actual"].round(0)
    lvl_df["Accuracy %"] = lvl_df["accuracy_pct"].round(1)
    lvl_df["Gap"] = lvl_df["gap"].round(0)
    wb_cols = ["Forecast Level", "Line Item", "Forecast", "Actual", "Accuracy %", "Gap", "risk", "records"]
    st.dataframe(
        lvl_df[wb_cols].head(500),
        use_container_width=True,
        height=520,
        hide_index=True,
        column_config={
            "Accuracy %": st.column_config.ProgressColumn("Accuracy %", min_value=0, max_value=150, format="%.1f%%"),
            "risk": st.column_config.TextColumn("Status"),
            "records": st.column_config.NumberColumn("Underlying Rows"),
        },
        key="level_workbench_table",
    )
    st.download_button("⬇️ Download level workbench", data=to_csv_bytes(lvl_df[wb_cols]), file_name="chemelex_forecast_level_workbench.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
