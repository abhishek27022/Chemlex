import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# Chemelex Demand Forecasting Dashboard
# Clean business UI + corrected Synthefy outputs
# Primary file: corrected_dashboard_forecast_accuracy_levelled.csv
# Optional chart/backtest file: corrected_forecast_backtest.csv
# =============================================================================

st.set_page_config(
    page_title="Chemelex Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ZONE_ORDER = [
    "BUFFER (at CODP)",
    "Pull / FG",
    "Pull / Kanban",
    "Push / MPS",
    "Push / RM",
]

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

MAIN_FILES = [
    "corrected_dashboard_forecast_accuracy_levelled.csv",
    "dashboard_forecast_accuracy_levelled.csv",
    "forecast_backtest_results.csv",
]
BACKTEST_FILES = [
    "corrected_forecast_backtest.csv",
    "forecast_backtest_results.csv",
    "corrected_dashboard_forecast_accuracy_levelled.csv",
]

# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def find_existing(candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if Path(name).exists():
            return name
    return None


def read_file(uploaded, candidates: list[str], file_label: str) -> Tuple[pd.DataFrame, str]:
    if uploaded is not None:
        if uploaded.name.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded), f"Uploaded: {uploaded.name}"
        return pd.read_csv(uploaded), f"Uploaded: {uploaded.name}"

    found = find_existing(candidates)
    if found is None:
        raise FileNotFoundError(
            f"{file_label} missing. Upload a CSV/XLSX or add one of these files to repo root: {', '.join(candidates)}"
        )
    if found.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(found), f"Repo file: {found}"
    return pd.read_csv(found), f"Repo file: {found}"


def clean_str(s: pd.Series, fallback="All") -> pd.Series:
    return s.fillna(fallback).astype(str).str.strip().replace({"": fallback, "nan": fallback, "None": fallback})


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize both corrected levelled output and older backtest output."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {
        "Supply_Chain_Zone": "codp_zone",
        "Plant Name": "plant",
        "Product 1 Category/Division Name": "product_category_1",
        "Accuracy %": "accuracy_pct",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Older backtest file: make a Group-level line item.
    if "level" not in df.columns:
        df["level"] = "Group"
    if "dimension_value" not in df.columns:
        if {"codp_zone", "plant", "product_category_1"}.issubset(df.columns):
            df["dimension_value"] = (
                df["codp_zone"].astype(str) + " | " +
                df["plant"].astype(str) + " | " +
                df["product_category_1"].astype(str)
            )
        else:
            df["dimension_value"] = "Total"

    for c in ["level", "dimension_value", "codp_zone", "plant", "product_category_1", "target", "model", "split"]:
        if c not in df.columns:
            df[c] = "All" if c in ["dimension_value", "codp_zone", "plant", "product_category_1"] else "Unknown"
        df[c] = clean_str(df[c], "All")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT

    if "year" not in df.columns or df["year"].isna().all():
        df["year"] = df["date"].dt.year
    if "month" not in df.columns or df["month"].isna().all():
        df["month"] = df["date"].dt.month

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    if df["date"].isna().all() and {"year", "month"}.issubset(df.columns):
        df["date"] = pd.to_datetime(
            dict(year=df["year"].fillna(1900).astype(int), month=df["month"].fillna(1).astype(int), day=1),
            errors="coerce",
        )

    for c in ["actual", "forecast", "abs_error", "error", "accuracy_pct"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df["abs_error"].isna().all():
        df["abs_error"] = (df["forecast"] - df["actual"]).abs()
    if df["error"].isna().all():
        df["error"] = df["actual"] - df["forecast"]

    # User-required business accuracy / forecast alignment: Actual ÷ Forecast × 100.
    df["business_accuracy_pct"] = np.where(
        df["forecast"].replace(0, np.nan).notna(),
        df["actual"] / df["forecast"].replace(0, np.nan) * 100,
        np.nan,
    )

    if "is_best" not in df.columns:
        df["is_best"] = 1
    df["is_best"] = df["is_best"].replace({True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0})
    df["is_best"] = pd.to_numeric(df["is_best"], errors="coerce").fillna(0).astype(int)

    if "data_quality_flags" not in df.columns:
        df["data_quality_flags"] = "OK"
    df["data_quality_flags"] = clean_str(df["data_quality_flags"], "OK")
    df.loc[df["data_quality_flags"].isin(["<NA>", "nan", "None", ""]), "data_quality_flags"] = "OK"

    # Exclude Unknown/non-inventory. User explicitly asked not to show Unknown.
    for c in ["codp_zone", "plant", "product_category_1", "dimension_value"]:
        bad = df[c].str.lower().isin(["unknown", "non-inventory", "non inventory"])
        df = df[~bad]

    # Supply Chain Zone must show exactly the official five zones in dropdown; allow All for Total/rolled rows.
    df = df[df["codp_zone"].isin(ZONE_ORDER) | df["codp_zone"].eq("All")]
    return df.reset_index(drop=True)


def fmt_value(v, target="order_value_usd") -> str:
    if v is None or pd.isna(v) or np.isinf(v):
        return "—"
    v = float(v)
    sign = "-" if v < 0 else ""
    x = abs(v)
    if target == "order_value_usd":
        if x >= 1_000_000_000:
            return f"{sign}${x/1_000_000_000:.2f}B"
        if x >= 1_000_000:
            return f"{sign}${x/1_000_000:.1f}M"
        if x >= 1_000:
            return f"{sign}${x/1_000:.1f}K"
        return f"{sign}${x:,.0f}"
    if x >= 1_000_000:
        return f"{sign}{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{sign}{x/1_000:.1f}K"
    return f"{sign}{x:,.0f}"


def fmt_pct(v) -> str:
    if v is None or pd.isna(v) or np.isinf(v):
        return "—"
    return f"{float(v):,.1f}%"


def calculate_kpis(data: pd.DataFrame):
    actual = data["actual"].sum(min_count=1)
    forecast = data["forecast"].sum(min_count=1)
    abs_error = data["abs_error"].sum(min_count=1)
    gap = actual - forecast if pd.notna(actual) and pd.notna(forecast) else np.nan
    wape = abs_error / actual * 100 if pd.notna(actual) and abs(actual) > 1e-12 else np.nan
    alignment = actual / forecast * 100 if pd.notna(forecast) and abs(forecast) > 1e-12 else np.nan
    bias = (forecast - actual) / actual * 100 if pd.notna(actual) and abs(actual) > 1e-12 else np.nan
    at_risk = int(((data["data_quality_flags"] != "OK") | (data["business_accuracy_pct"] < 80) | (data["business_accuracy_pct"] > 120)).sum())
    return actual, forecast, abs_error, gap, wape, alignment, bias, at_risk


def month_name(m):
    try:
        return MONTH_NAMES.get(int(m), str(m))
    except Exception:
        return str(m)


def period_label(row):
    try:
        return f"{MONTH_NAMES.get(int(row['month']), int(row['month']))} {int(row['year'])}"
    except Exception:
        return str(row.get("date", ""))


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# -----------------------------------------------------------------------------
# Theme and CSS
# -----------------------------------------------------------------------------

if "theme_choice" not in st.session_state:
    st.session_state.theme_choice = "Light"

def apply_css(theme: str):
    dark = theme == "Dark"
    if dark:
        colors = {
            "bg": "#0b1220", "panel": "#111827", "card": "#111827", "card2": "#172033", "text": "#f8fafc",
            "muted": "#a8b3c7", "border": "#29364f", "soft": "#0f1b2d", "input": "#0f172a"
        }
    else:
        colors = {
            "bg": "#f6f8fc", "panel": "#ffffff", "card": "#ffffff", "card2": "#ffffff", "text": "#102a43",
            "muted": "#63738a", "border": "#e0e7f0", "soft": "#eef6ff", "input": "#ffffff"
        }
    st.markdown(f"""
    <style>
      .stApp {{ background: {colors['bg']}; color: {colors['text']}; }}
      .block-container {{ max-width: 1480px; padding-top: 0.7rem; padding-bottom: 2rem; }}
      section[data-testid="stSidebar"] {{ background: {colors['panel']}; border-right: 1px solid {colors['border']}; }}
      .chex-shell {{ color: {colors['text']}; }}
      .chex-header {{ display:flex; align-items:center; justify-content:space-between; gap:18px; padding: 14px 18px 12px 18px; border-radius: 18px; background: linear-gradient(135deg, #ffffff 0%, #eff6ff 50%, #ecfeff 100%); border:1px solid {colors['border']}; box-shadow: 0 12px 30px rgba(15,23,42,.07); }}
      .dark-header {{ background: linear-gradient(135deg, #0f172a 0%, #16314e 55%, #134e4a 100%); }}
      .brand {{ display:flex; align-items:center; gap:12px; }}
      .brand-logo {{ width:42px; height:42px; border-radius:13px; display:grid; place-items:center; background:#0b75bd; color:white; font-weight:900; font-size:20px; box-shadow: inset 0 -10px 18px rgba(0,0,0,.12); }}
      .brand-title {{ font-size: 28px; line-height: 1.05; color:{colors['text']}; font-weight: 900; letter-spacing:-.03em; margin:0; }}
      .brand-subtitle {{ color:{colors['muted']}; font-size:13px; margin-top:4px; }}
      .header-right {{ text-align:right; color:{colors['muted']}; font-size:12px; }}
      .nav-row {{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin:12px 0 8px 0; }}
      .nav-pills {{ display:flex; gap:10px; flex-wrap:wrap; }}
      .nav-pill {{ padding:9px 16px; border-radius:999px; border:1px solid #0b75bd; color:#0b75bd; background:#eff6ff; font-weight:700; font-size:14px; }}
      .nav-pill-muted {{ border-color:{colors['border']}; color:{colors['muted']}; background:{colors['panel']}; }}
      .alert-strip {{ background:#fff7ed; color:#9a3412; border:1px solid #fed7aa; border-left:5px solid #f59e0b; border-radius:13px; padding:10px 14px; margin: 10px 0 14px 0; font-size:13px; font-weight:650; }}
      .upload-band {{ background:{colors['panel']}; border:1px dashed #0b75bd; border-radius:16px; padding: 12px 14px; margin:12px 0; }}
      .filter-card {{ background: {colors['panel']}; border:1px solid {colors['border']}; border-radius:18px; padding:16px; margin: 14px 0; box-shadow: 0 8px 26px rgba(15,23,42,.055); }}
      .section-title {{ color:{colors['text']}; font-size:18px; font-weight:850; margin: 2px 0 12px 0; }}
      .kpi-grid {{ display:grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap:14px; margin: 14px 0 16px 0; }}
      .kpi-card {{ position:relative; overflow:hidden; min-height:132px; background:{colors['card']}; border:1px solid {colors['border']}; border-radius:18px; padding:16px; box-shadow:0 11px 26px rgba(15,23,42,.08); }}
      .kpi-card:before {{ content:""; position:absolute; right:-36px; top:-36px; width:94px; height:94px; border-radius:999px; background:var(--accent-soft); }}
      .kpi-label {{ color:{colors['muted']}; font-size:11px; font-weight:850; text-transform:uppercase; letter-spacing:.06em; }}
      .kpi-value {{ color:{colors['text']}; font-size:26px; font-weight:900; margin-top:9px; letter-spacing:-.03em; }}
      .kpi-sub {{ color:{colors['muted']}; font-size:12px; margin-top:4px; }}
      .kpi-icon {{ position:absolute; right:15px; top:18px; width:42px; height:42px; border-radius:14px; display:grid; place-items:center; background:var(--accent-soft); color:var(--accent); font-size:21px; }}
      .insight-grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:14px; margin: 12px 0 16px; }}
      .insight-card {{ background:{colors['card']}; border:1px solid {colors['border']}; border-radius:18px; padding:16px; box-shadow:0 9px 24px rgba(15,23,42,.06); }}
      .insight-title {{ font-weight:850; color:{colors['text']}; font-size:16px; }}
      .insight-text {{ color:{colors['muted']}; font-size:13px; margin-top:4px; line-height:1.45; }}
      .viz-card {{ background:{colors['card']}; border:1px solid {colors['border']}; border-radius:18px; padding:14px; box-shadow:0 9px 24px rgba(15,23,42,.055); margin-bottom:14px; }}
      div[data-testid="stDataFrame"] {{ border-radius: 16px; overflow:hidden; border:1px solid {colors['border']}; }}
      div[data-baseweb="select"] > div, input, textarea {{ background-color:{colors['input']}!important; border-color:{colors['border']}!important; color:{colors['text']}!important; }}
      label, .stMarkdown, .stTextInput label, .stSelectbox label, .stMultiSelect label {{ color:{colors['text']}!important; }}
      .stTabs [data-baseweb="tab-list"] {{ gap: 9px; background:transparent; }}
      .stTabs [data-baseweb="tab"] {{ background:{colors['panel']}; border:1px solid {colors['border']}; border-radius: 999px; padding: 9px 18px; font-weight:750; }}
      .stTabs [aria-selected="true"] {{ background:#0b75bd!important; color:white!important; border-color:#0b75bd!important; }}
      @media (max-width: 1180px) {{ .kpi-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} }}
      @media (max-width: 740px) {{ .kpi-grid {{ grid-template-columns: 1fr; }} .insight-grid {{ grid-template-columns: 1fr; }} .chex-header {{ flex-direction:column; align-items:flex-start; }} }}
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar + visible uploader controls
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    theme = st.radio("Theme", ["Light", "Dark"], horizontal=True, key="theme_toggle")
    st.markdown("### 📤 Upload Data")
    main_upload_sidebar = st.file_uploader(
        "Main file: corrected_dashboard_forecast_accuracy_levelled.csv",
        type=["csv", "xlsx", "xls"],
        key="main_upload_sidebar",
    )
    backtest_upload_sidebar = st.file_uploader(
        "Optional chart file: corrected_forecast_backtest.csv",
        type=["csv", "xlsx", "xls"],
        key="backtest_upload_sidebar",
    )
    st.caption("The app defaults to repo files when nothing is uploaded.")

apply_css(theme)

st.markdown(f"""
<div class='chex-shell'>
  <div class='chex-header {'dark-header' if theme == 'Dark' else ''}'>
    <div class='brand'>
      <div class='brand-logo'>⌁</div>
      <div>
        <h1 class='brand-title'>chemelex Demand Forecasting Cockpit</h1>
        <div class='brand-subtitle'>Executive view of forecast performance, demand alignment, and planning-level risk.</div>
      </div>
    </div>
    <div class='header-right'>
      <div><b>Data as of</b></div>
      <div>May 10, 2026 · 09:30 AM</div>
      <div style='margin-top:6px;'>🔔 14 · ⚙️ · AM ▾</div>
    </div>
  </div>
  <div class='nav-row'>
    <div class='nav-pills'>
      <div class='nav-pill'>Demand Forecast</div>
      <div class='nav-pill nav-pill-muted'>Forecast Accuracy</div>
      <div class='nav-pill nav-pill-muted'>Demand Plan</div>
      <div class='nav-pill nav-pill-muted'>Material View</div>
      <div class='nav-pill nav-pill-muted'>Insights</div>
    </div>
  </div>
  <div class='alert-strip'>18/11/2023 14:23:51 · Corrected Synthefy model loaded. Old Chemlex SIOP forecast is not directly comparable because it is category-level and COGS-based, while this model is zone × plant × category and revenue-based.</div>
</div>
""", unsafe_allow_html=True)

# Prominent uploader on main page, always visible.
st.markdown("<div class='upload-band'><b>Upload / replace data files</b><br><span style='font-size:12px;color:#64748b;'>Use corrected_dashboard_forecast_accuracy_levelled.csv as main dashboard file. Optional: corrected_forecast_backtest.csv for backtest charts.</span></div>", unsafe_allow_html=True)
u1, u2 = st.columns(2)
with u1:
    main_upload_top = st.file_uploader("Upload main dashboard CSV/XLSX", type=["csv", "xlsx", "xls"], key="main_upload_top")
with u2:
    backtest_upload_top = st.file_uploader("Upload optional backtest/chart CSV/XLSX", type=["csv", "xlsx", "xls"], key="backtest_upload_top")

main_uploaded = main_upload_top if main_upload_top is not None else main_upload_sidebar
backtest_uploaded = backtest_upload_top if backtest_upload_top is not None else backtest_upload_sidebar

# -----------------------------------------------------------------------------
# Data load
# -----------------------------------------------------------------------------

try:
    raw_main, main_source = read_file(main_uploaded, MAIN_FILES, "Main dashboard file")
    df = normalize_df(raw_main)
except Exception as exc:
    st.error(f"Could not load the main dashboard file: {exc}")
    st.stop()

try:
    raw_backtest, backtest_source = read_file(backtest_uploaded, BACKTEST_FILES, "Backtest/chart file")
    chart_df = normalize_df(raw_backtest)
except Exception:
    chart_df = df.copy()
    backtest_source = "Using main file for charts"

st.caption(f"Main source: {main_source} · Chart source: {backtest_source}")

# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------

year_options = sorted([int(x) for x in df["year"].dropna().unique().tolist()])
default_years = [2025] if 2025 in year_options else (year_options[-1:] if year_options else [])
month_options = sorted([int(x) for x in df["month"].dropna().unique().tolist()])
level_options = [x for x in ["Total", "CODP Zone", "Plant", "Product Category", "Group"] if x in set(df["level"].unique())]
if not level_options:
    level_options = sorted(df["level"].dropna().unique().tolist())
default_level = "Group" if "Group" in level_options else level_options[0]

target_options = sorted(df["target"].dropna().unique().tolist())
default_target = "order_value_usd" if "order_value_usd" in target_options else target_options[0]
split_options = sorted(df["split"].dropna().unique().tolist())
default_split = "split2" if "split2" in split_options else split_options[0]

st.markdown("<div class='filter-card'><div class='section-title'>Filters</div>", unsafe_allow_html=True)
f1, f2, f3, f4 = st.columns([1, 1.2, 1.5, 2.2])
with f1:
    years = st.multiselect("Year", year_options, default=default_years, key="year_filter")
with f2:
    months = st.multiselect("Month", month_options, default=month_options, format_func=month_name, key="month_filter")
with f3:
    selected_level = st.selectbox("Forecast Level", level_options, index=level_options.index(default_level), key="level_filter")
with f4:
    selected_zones = st.multiselect("Supply Chain Zone", ZONE_ORDER, default=ZONE_ORDER, key="zone_filter")

# Dynamic plant/product options after basic filtering.
pre = df.copy()
if years:
    pre = pre[pre["year"].isin(years)]
if months:
    pre = pre[pre["month"].isin(months)]
if selected_zones:
    pre = pre[pre["codp_zone"].isin(selected_zones) | pre["codp_zone"].eq("All")]

plant_options = sorted([x for x in pre["plant"].dropna().unique().tolist() if x not in ["All", "Unknown"]])
category_options = sorted([x for x in pre["product_category_1"].dropna().unique().tolist() if x not in ["All", "Unknown"]])

f5, f6, f7 = st.columns([2.0, 2.4, 1.6])
with f5:
    selected_plants = st.multiselect("Plant", plant_options, default=plant_options, key="plant_filter")
with f6:
    selected_categories = st.multiselect("Product Category", category_options, default=category_options, key="category_filter")
with f7:
    search = st.text_input("Search", placeholder="Search line item...", key="line_search")

with st.expander("Advanced Settings", expanded=False):
    a1, a2, a3, a4 = st.columns([1.4, 1.0, 1.0, 1.6])
    with a1:
        target = st.selectbox("Target", target_options, index=target_options.index(default_target), key="target_filter")
    with a2:
        split = st.selectbox("Split", split_options, index=split_options.index(default_split), key="split_filter")
    with a3:
        best_only = st.toggle("Best model only", value=True, key="best_model_toggle")
    with a4:
        model_options = sorted(df["model"].dropna().unique().tolist())
        selected_models = st.multiselect("Model", model_options, default=model_options, key="model_filter")
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Apply filters
# -----------------------------------------------------------------------------

def apply_common_filters(data: pd.DataFrame, force_level: str | None = None) -> pd.DataFrame:
    out = data.copy()
    if force_level is not None and "level" in out.columns:
        out = out[out["level"].eq(force_level)]
    out = out[out["target"].eq(target)]
    out = out[out["split"].eq(split)]
    if best_only:
        out = out[out["is_best"].eq(1)]
    if selected_models:
        out = out[out["model"].isin(selected_models)]
    if years:
        out = out[out["year"].isin(years)]
    if months:
        out = out[out["month"].isin(months)]
    if selected_zones:
        out = out[out["codp_zone"].isin(selected_zones) | out["codp_zone"].eq("All")]
    if selected_plants:
        out = out[out["plant"].isin(selected_plants) | out["plant"].eq("All")]
    if selected_categories:
        out = out[out["product_category_1"].isin(selected_categories) | out["product_category_1"].eq("All")]
    if search.strip():
        s = search.strip().lower()
        mask = (
            out["dimension_value"].str.lower().str.contains(s, na=False) |
            out["codp_zone"].str.lower().str.contains(s, na=False) |
            out["plant"].str.lower().str.contains(s, na=False) |
            out["product_category_1"].str.lower().str.contains(s, na=False)
        )
        out = out[mask]
    return out

filtered = apply_common_filters(df, selected_level)
chart_filtered = apply_common_filters(chart_df, selected_level if selected_level in set(chart_df["level"].unique()) else None)

actual, forecast, abs_error, gap, wape, alignment, bias, at_risk = calculate_kpis(filtered)
active_items = filtered["dimension_value"].nunique()

# -----------------------------------------------------------------------------
# KPI grid: pure HTML grid so it never stacks into ugly full-width cards.
# -----------------------------------------------------------------------------

kpi_items = [
    ("Active Line Items", f"{active_items:,}", f"Selected level: {selected_level}", "#2563eb", "#dbeafe", "▦"),
    ("Total Forecast", fmt_value(forecast, target), target, "#0b75bd", "#e0f2fe", "↗"),
    ("Total Actual", fmt_value(actual, target), target, "#16a34a", "#dcfce7", "✓"),
    ("WAPE", fmt_pct(wape), "sum(abs_error) / sum(actual)", "#f97316", "#ffedd5", "≈"),
    ("Forecast Alignment", fmt_pct(alignment), "Actual ÷ Forecast × 100", "#059669", "#d1fae5", "%"),
    ("Bias", fmt_pct(bias), "(Forecast - Actual) ÷ Actual", "#dc2626", "#fee2e2", "Δ"),
]

html = "<div class='kpi-grid'>"
for label, value, sub, accent, soft, icon in kpi_items:
    html += f"""
    <div class='kpi-card' style='--accent:{accent};--accent-soft:{soft};'>
      <div class='kpi-icon'>{icon}</div>
      <div class='kpi-label'>{label}</div>
      <div class='kpi-value'>{value}</div>
      <div class='kpi-sub'>{sub}</div>
    </div>
    """
html += "</div>"
st.markdown(html, unsafe_allow_html=True)

st.markdown("""
<div class='insight-grid'>
  <div class='insight-card'>
    <div class='insight-title'>✅ Pull / Kanban Fixed</div>
    <div class='insight-text'>$22.3M restored · 9.0% WAPE · 94.0% business accuracy. This zone is now included as an official inventory CODP stream.</div>
  </div>
  <div class='insight-card'>
    <div class='insight-title'>✅ Pull / FG Restored</div>
    <div class='insight-text'>$474M restored · 47.6% of demand · largest CODP zone. This is now visible in the official Supply Chain Zone filter.</div>
  </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Charts + table
# -----------------------------------------------------------------------------

def style_plot(fig, height=360):
    template = "plotly_dark" if theme == "Dark" else "plotly_white"
    fig.update_layout(
        template=template,
        height=height,
        margin=dict(l=20, r=20, t=54, b=38),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=12),
    )
    return fig

tab1, tab2, tab3 = st.tabs(["Material Forecast", "Forecast Analysis", "Data Quality"])

with tab1:
    c1, c2 = st.columns([1.15, 1])
    monthly = (
        chart_filtered.groupby(["year", "month"], dropna=False)
        .agg(actual=("actual", "sum"), forecast=("forecast", "sum"), abs_error=("abs_error", "sum"))
        .reset_index()
        .dropna(subset=["year", "month"])
        .sort_values(["year", "month"])
    )
    if not monthly.empty:
        monthly["period"] = monthly.apply(period_label, axis=1)
        monthly["alignment"] = np.where(monthly["forecast"].replace(0, np.nan).notna(), monthly["actual"] / monthly["forecast"].replace(0, np.nan) * 100, np.nan)
        monthly["wape"] = np.where(monthly["actual"].replace(0, np.nan).notna(), monthly["abs_error"] / monthly["actual"].replace(0, np.nan) * 100, np.nan)
        with c1:
            st.markdown("<div class='viz-card'>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly["period"], y=monthly["forecast"], mode="lines+markers", name="Forecast", line=dict(color="#0b75bd", width=3)))
            fig.add_trace(go.Scatter(x=monthly["period"], y=monthly["actual"], mode="lines+markers", name="Actual", line=dict(color="#16a34a", width=3)))
            fig.update_layout(title="Demand: Monthly Forecast vs Actual")
            st.plotly_chart(style_plot(fig, 360), use_container_width=True, key="monthly_forecast_actual")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='viz-card'>", unsafe_allow_html=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=monthly["period"], y=monthly["alignment"], mode="lines+markers", name="Forecast Alignment %", line=dict(color="#2563eb", width=3)))
            fig2.add_hline(y=100, line_dash="dash", line_color="#94a3b8")
            fig2.update_layout(title="Forecast Alignment (%) by Month", yaxis_title="Actual ÷ Forecast × 100")
            st.plotly_chart(style_plot(fig2, 360), use_container_width=True, key="alignment_month")
            st.markdown("</div>", unsafe_allow_html=True)

    b1, b2 = st.columns([1, 1])
    with b1:
        category = (
            filtered.groupby("product_category_1", dropna=False)
            .agg(actual=("actual", "sum"), forecast=("forecast", "sum"), abs_error=("abs_error", "sum"))
            .reset_index()
        )
        category = category[category["product_category_1"].ne("All")].sort_values("actual", ascending=False).head(10)
        if not category.empty:
            st.markdown("<div class='viz-card'>", unsafe_allow_html=True)
            cat_long = category.melt(id_vars="product_category_1", value_vars=["forecast", "actual"], var_name="Metric", value_name="Value")
            fig3 = px.bar(cat_long, x="product_category_1", y="Value", color="Metric", barmode="group", title="Category Performance", color_discrete_map={"forecast": "#0b75bd", "actual": "#16a34a"})
            fig3.update_xaxes(tickangle=-20)
            st.plotly_chart(style_plot(fig3, 340), use_container_width=True, key="category_perf")
            st.markdown("</div>", unsafe_allow_html=True)
    with b2:
        zone_perf = (
            filtered.groupby("codp_zone", dropna=False)
            .agg(actual=("actual", "sum"), forecast=("forecast", "sum"), abs_error=("abs_error", "sum"))
            .reset_index()
        )
        zone_perf = zone_perf[zone_perf["codp_zone"].isin(ZONE_ORDER)]
        if not zone_perf.empty:
            zone_perf["alignment"] = zone_perf["actual"] / zone_perf["forecast"].replace(0, np.nan) * 100
            st.markdown("<div class='viz-card'>", unsafe_allow_html=True)
            fig4 = px.bar(zone_perf, y="codp_zone", x="alignment", orientation="h", title="Supply Chain Zone Forecast Alignment", color="alignment", color_continuous_scale="Blues")
            fig4.add_vline(x=100, line_dash="dash", line_color="#94a3b8")
            st.plotly_chart(style_plot(fig4, 340), use_container_width=True, key="zone_alignment")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Forecast vs Actual Detail</div>", unsafe_allow_html=True)
    group_cols = ["year", "month", "level", "dimension_value", "codp_zone", "plant", "product_category_1", "data_quality_flags"]
    table = (
        filtered.groupby(group_cols, dropna=False)
        .agg(actual=("actual", "sum"), forecast=("forecast", "sum"), abs_error=("abs_error", "sum"))
        .reset_index()
    )
    table["accuracy_pct"] = table["actual"] / table["forecast"].replace(0, np.nan) * 100
    table["gap"] = table["actual"] - table["forecast"]
    table["period"] = table.apply(period_label, axis=1)
    table["Forecast"] = table["forecast"].apply(lambda x: fmt_value(x, target))
    table["Actual"] = table["actual"].apply(lambda x: fmt_value(x, target))
    table["Accuracy %"] = table["accuracy_pct"].apply(fmt_pct)
    table["Gap"] = table["gap"].apply(lambda x: fmt_value(x, target))
    display = table.rename(columns={
        "period": "Period",
        "level": "Forecast Level",
        "dimension_value": "Line Item",
        "codp_zone": "Supply Chain Zone",
        "plant": "Plant",
        "product_category_1": "Product Category",
        "data_quality_flags": "Data Status",
    })[["Period", "Forecast Level", "Line Item", "Supply Chain Zone", "Plant", "Product Category", "Forecast", "Actual", "Accuracy %", "Gap", "Data Status"]]
    # Sort by absolute displayed gap using raw table.
    display = display.iloc[table["gap"].abs().sort_values(ascending=False).index].head(500)
    st.dataframe(display, use_container_width=True, hide_index=True, height=430)
    st.download_button("⬇️ Download visible detail", data=csv_bytes(display), file_name="chemelex_forecast_visible_detail.csv", mime="text/csv")

with tab2:
    st.markdown("<div class='section-title'>Forecast Analysis</div>", unsafe_allow_html=True)
    a, b, c = st.columns([1, 1, 1])
    with a:
        st.markdown(f"<div class='insight-card'><div class='insight-title'>Baseline Actual</div><div class='kpi-value'>{fmt_value(actual, target)}</div><div class='insight-text'>Selected filters and selected forecast level.</div></div>", unsafe_allow_html=True)
    with b:
        st.markdown(f"<div class='insight-card'><div class='insight-title'>Total Gap</div><div class='kpi-value'>{fmt_value(gap, target)}</div><div class='insight-text'>Actual - Forecast.</div></div>", unsafe_allow_html=True)
    with c:
        st.markdown(f"<div class='insight-card'><div class='insight-title'>Forecast Alignment</div><div class='kpi-value'>{fmt_pct(alignment)}</div><div class='insight-text'>Actual ÷ Forecast × 100.</div></div>", unsafe_allow_html=True)

    if not monthly.empty:
        st.markdown("<div class='viz-card'>", unsafe_allow_html=True)
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(x=monthly["period"], y=monthly["forecast"], name="Forecast", marker_color="#0b75bd"))
        fig5.add_trace(go.Bar(x=monthly["period"], y=monthly["actual"], name="Actual", marker_color="#16a34a"))
        fig5.update_layout(title="Monthly Forecast Analysis", barmode="group")
        st.plotly_chart(style_plot(fig5, 390), use_container_width=True, key="forecast_analysis_monthly")
        st.markdown("</div>", unsafe_allow_html=True)

    # Diagnostic section for Pull / Kanban if visible.
    pk = df[(df["codp_zone"].eq("Pull / Kanban")) & (df["target"].eq(target)) & (df["split"].eq(split))]
    if best_only:
        pk = pk[pk["is_best"].eq(1)]
    if not pk.empty:
        pk_actual, pk_forecast, pk_abs, pk_gap, pk_wape, pk_acc, pk_bias, _ = calculate_kpis(pk)
        st.markdown(f"""
        <div class='insight-card'>
          <div class='insight-title'>Pull / Kanban Diagnostic</div>
          <div class='insight-text'>Visible in official CODP filter. Forecast alignment: <b>{fmt_pct(pk_acc)}</b>, WAPE: <b>{fmt_pct(pk_wape)}</b>, Actual: <b>{fmt_value(pk_actual, target)}</b>.</div>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='section-title'>Data Quality & Scope Notes</div>", unsafe_allow_html=True)
    dq = filtered["data_quality_flags"].value_counts(dropna=False).reset_index()
    dq.columns = ["Data Status", "Rows"]
    st.dataframe(dq, use_container_width=True, hide_index=True)
    st.info("Current most detailed validated forecast level is Group = Supply Chain Zone × Plant × Product Category. Do not call this SKU/material level unless Synthefy provides SKU/material code and material description columns.")
    st.warning("Old Chemlex SIOP forecast is not directly comparable because it is category-level and COGS-based, while this corrected model is zone × plant × category and revenue-based.")
