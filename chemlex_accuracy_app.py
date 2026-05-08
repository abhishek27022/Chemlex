"""
Chemelex Demand Forecasting Cockpit
-----------------------------------
Purpose-built for the client demo:
- Consume forecast_backtest_results.csv from Synthefy/backtesting output.
- Show end-user friendly forecast accuracy, not MAPE.
- Accuracy formula used everywhere:
      Accuracy % = min(Actual, Forecast) / max(Actual, Forecast) * 100
  Example: Actual = 90 and Forecast = 100 => 90% accuracy.
- Provide Chemelex-style Material Forecast + Forecast Analysis pages.

Expected primary input columns:
    date, codp_zone, plant, product_category_1, group_key, target, split,
    model, actual, forecast, is_best, year, month

Also supports raw order history structure and builds a simple rolling baseline:
    Year, Month, Supply_Chain_Zone, Plant Name,
    Product 1 Category/Division Name,
    Sum of Order Value in USD from LC,
    Sum of Order quantity BU
"""

import io
import os
import re
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chemelex Demand Forecasting Cockpit",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS / CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_FILE_CANDIDATES = [
    "forecast_backtest_results.csv",
    "Forecast_Backtest_Results.csv",
    "data.csv",
]

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
MONTH_TO_NUM = {m.lower(): i + 1 for i, m in enumerate(MONTH_ORDER)}
MONTH_TO_NUM.update({m[:3].lower(): i + 1 for i, m in enumerate(MONTH_ORDER)})

TARGET_LABELS = {
    "order_value_usd": "Order Value ($)",
    "order_qty_bu": "Order Quantity (BU)",
}

TARGET_SHORT = {
    "order_value_usd": "$",
    "order_qty_bu": "BU",
}

CHEMELEX_BLUE = "#0077A8"
CHEMELEX_DARK_BLUE = "#2D6F88"
CHEMELEX_GREEN = "#34A853"
CHEMELEX_ORANGE = "#F28C00"
CHEMELEX_RED = "#F0645A"
CHEMELEX_BG = "#F8F8F7"


# ─────────────────────────────────────────────────────────────────────────────
# CSS — CHEMELEX-LIKE LIGHT UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .stApp {
        background: #f7f7f6;
        color: #333333;
    }
    .block-container {
        max-width: 1500px;
        padding-top: 0.35rem;
        padding-bottom: 2rem;
    }
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e8e8e8;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e4e4e4;
        border-radius: 8px;
        padding: 11px 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.035);
    }
    div[data-testid="stMetric"] label {
        color: #6b6b6b !important;
        font-size: 0.82rem !important;
    }
    div[data-testid="stMetricValue"] {
        color: #2e2e2e;
        font-size: 1.35rem !important;
        font-weight: 600 !important;
    }
    .chex-header {
        margin: -0.2rem -0.5rem 0.1rem -0.5rem;
        background: #ffffff;
        border-bottom: 1px solid #dfdfdf;
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 14px 0 18px;
    }
    .chex-logo-row {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .algo-text {
        font-size: 0.72rem;
        color: #9aa3a8;
        font-weight: 600;
    }
    .chex-logo {
        font-size: 1.75rem;
        line-height: 1;
        font-weight: 800;
        color: #356f82;
        font-style: italic;
        letter-spacing: -0.035em;
    }
    .chex-icons {
        color: #3c3c3c;
        font-size: 1.1rem;
        letter-spacing: 0.35rem;
    }
    .time-strip {
        background: #fff4e2;
        border-left: 5px solid #f28c00;
        border-top: 1px solid #f7dfbd;
        border-bottom: 1px solid #f7dfbd;
        color: #444;
        height: 25px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 12px 0 8px;
        margin: 0 -0.5rem 1rem -0.5rem;
        font-size: 0.78rem;
    }
    .pill-tab-row {
        display: flex;
        align-items: center;
        gap: 12px;
        background: #ffffff;
        border-bottom: 1px solid #e8e8e8;
        padding: 7px 20px 5px 20px;
        margin: 0 -0.5rem 0 -0.5rem;
    }
    .pill-tab {
        border: 1px solid #0b84bd;
        color: #4b4b4b;
        background: #f4fbff;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .pill-tab-muted {
        color: #444;
        font-size: 0.82rem;
        padding: 6px 10px;
    }
    .page-card {
        background: #ffffff;
        border: 1px solid #e4e4e4;
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.035);
        margin-bottom: 14px;
    }
    .section-title {
        font-size: 1.15rem;
        color: #555;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    .subtle-title {
        font-size: 0.88rem;
        color: #666;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .soft-panel {
        border: 1px solid #e3e3e3;
        border-radius: 14px;
        background: #ffffff;
        padding: 13px;
    }
    .accuracy-chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        padding: 6px 12px;
        color: #7a3232;
        background: #ffd7d7;
        font-weight: 600;
    }
    .good-chip { background:#dcfce7; color:#166534; }
    .warn-chip { background:#fef3c7; color:#92400e; }
    .bad-chip { background:#fee2e2; color:#991b1b; }
    .blue-chip { background:#e0f2fe; color:#075985; }
    .driver-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
    }
    .driver-table th {
        background:#f1f1f1;
        color:#555;
        padding: 10px 8px;
        border: 1px solid #ddd;
    }
    .driver-table td {
        padding: 7px 8px;
        border: 1px solid #e6e6e6;
    }
    .driver-green { background:#e7f4eb; }
    .driver-red { background:#fae5e2; }
    .driver-amber { background:#fff1dc; }
    .ai-box {
        background: linear-gradient(135deg, #f4fbff 0%, #ffffff 70%);
        border: 1px solid #cbe7f7;
        border-radius: 14px;
        padding: 15px;
    }
    .formula-box {
        background: #fff8ec;
        border: 1px solid #f7d8a7;
        border-radius: 14px;
        padding: 14px;
        color: #6b4a12;
    }
    .small-muted { color:#858585; font-size:0.78rem; }
    .download-dot {
        background:#0088c2; color:white; border-radius:50%; width:32px; height:32px;
        display:inline-flex; align-items:center; justify-content:center; font-weight:700;
        box-shadow: 0 2px 5px rgba(0,0,0,.18);
    }
    /* Streamlit tabs look too heavy; make them subtle */
    button[data-baseweb="tab"] {
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# GENERIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None:
        df = pd.DataFrame()
    return df.to_csv(index=False).encode("utf-8")


def clean_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def fmt_currency(v: float) -> str:
    try:
        v = float(v)
    except Exception:
        return "$0"
    if abs(v) >= 1_000_000:
        return f"${v/1_000_000:,.1f}M"
    if abs(v) >= 1_000:
        return f"${v/1_000:,.1f}K"
    return f"${v:,.0f}"


def fmt_qty(v: float) -> str:
    try:
        v = float(v)
    except Exception:
        return "0 BU"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:,.1f}M BU"
    if abs(v) >= 1_000:
        return f"{v/1_000:,.1f}K BU"
    return f"{v:,.0f} BU"


def fmt_value(v: float, target: str) -> str:
    return fmt_currency(v) if target == "order_value_usd" else fmt_qty(v)


def simple_accuracy(actual: float, forecast: float) -> float:
    """
    End-user friendly accuracy.
    Example: Actual 90, Forecast 100 -> 90%.
    Symmetric for over/under forecast: Actual 100, Forecast 90 -> 90%.
    """
    try:
        a = max(0.0, float(actual))
        f = max(0.0, float(forecast))
    except Exception:
        return np.nan
    denom = max(a, f)
    if denom == 0:
        return 100.0
    return max(0.0, min(a, f) / denom * 100.0)


def aggregate_accuracy(actual_series: Iterable[float], forecast_series: Iterable[float]) -> float:
    a = float(np.nansum(list(actual_series)))
    f = float(np.nansum(list(forecast_series)))
    return simple_accuracy(a, f)


def accuracy_band(acc: float) -> str:
    if pd.isna(acc):
        return "No Signal"
    if acc >= 90:
        return "Excellent"
    if acc >= 80:
        return "Good"
    if acc >= 65:
        return "Watch"
    return "Action Needed"


def priority_from_accuracy(acc: float, abs_gap: float, p85_gap: float) -> str:
    if pd.isna(acc):
        return "No Signal"
    if acc < 65 or abs_gap >= p85_gap:
        return "High"
    if acc < 80:
        return "Medium"
    return "Low"


def safe_str(v, default="All"):
    if pd.isna(v):
        return default
    return str(v)


def add_chex_shell():
    st.markdown(
        """
<div class="chex-header">
  <div class="chex-logo-row">
    <div class="algo-text">algo8</div>
    <div class="chex-logo">chemelex</div>
  </div>
  <div class="chex-icons">🔔 ⚙️ 👤⌄</div>
</div>
<div class="pill-tab-row">
  <span class="pill-tab">Material Forecast</span>
  <span class="pill-tab-muted">Forecast Analysis</span>
</div>
<div class="time-strip">
  <span>18/11/2023 14:23:51 &nbsp;&nbsp; Demand Forecast Cockpit · Actual vs Forecast Accuracy</span>
  <span>◀ &nbsp; 1 of 14 &nbsp; ▶ &nbsp; ✖</span>
</div>
""",
        unsafe_allow_html=True,
    )


def style_plot(fig, height=360):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, Arial", size=12, color="#5c5c5c"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#e5e5e5",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(gridcolor="#eeeeee", zerolinecolor="#e5e5e5")
    fig.update_yaxes(gridcolor="#eeeeee", zerolinecolor="#e5e5e5")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING / STANDARDIZATION
# ─────────────────────────────────────────────────────────────────────────────
def read_any_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Please upload a CSV or Excel file.")


def find_default_file() -> Optional[str]:
    for f in DEFAULT_FILE_CANDIDATES:
        if os.path.exists(f):
            return f
    return None


def load_default_or_demo() -> Tuple[pd.DataFrame, str]:
    f = find_default_file()
    if f:
        return pd.read_csv(f), f"Loaded default file: {f}"
    return build_demo_backtest_data(), "Using generated demo data because no default file was found."


def build_demo_backtest_data() -> pd.DataFrame:
    np.random.seed(42)
    categories = [
        "PD / Heat Tracing Components",
        "PD / Fire and Performance Wiring",
        "PD / MI Heat Tracing",
        "PD / Floor Heating",
        "PD / Control, Monitoring & Power Distribution",
    ]
    plants = ["Chemelex - Trenton", "Chemelex - RWC/UCDC", "Chemelex - Pharr, T"]
    zones = ["Push / MPS", "Push / RM", "Pull / Kanban", "BUFFER (at CODP)"]
    rows = []
    for target in ["order_value_usd", "order_qty_bu"]:
        for split in ["split1", "split2"]:
            for plant in plants:
                for zone in zones:
                    for cat in categories:
                        base = np.random.uniform(8000, 65000) if target == "order_value_usd" else np.random.uniform(30, 800)
                        for m in range(1, 13):
                            season = 1 + 0.18 * np.sin((m - 2) / 12 * 2 * np.pi)
                            actual = max(0, base * season * np.random.normal(1, 0.18))
                            forecast = max(0, actual * np.random.normal(1.03, 0.17))
                            rows.append({
                                "date": f"2025-{m:02d}-01",
                                "codp_zone": zone,
                                "plant": plant,
                                "product_category_1": cat,
                                "group_key": f"{zone} | {plant} | {cat}",
                                "target": target,
                                "split": split,
                                "model": "Best Statistical Model",
                                "actual": actual,
                                "forecast": forecast,
                                "is_best": True,
                                "year": 2025,
                                "month": m,
                            })
    return pd.DataFrame(rows)


def standardize_input(raw: pd.DataFrame) -> pd.DataFrame:
    """Return normalized backtest-style dataframe."""
    df = raw.copy()
    original_cols = list(df.columns)
    c_map = {c: clean_col(c) for c in df.columns}
    df = df.rename(columns=c_map)

    # Case 1: already forecast backtest output
    expected = {"date", "codp_zone", "plant", "product_category_1", "actual", "forecast"}
    if expected.issubset(set(df.columns)):
        out = df.copy()
        if "group_key" not in out.columns:
            out["group_key"] = (
                out["codp_zone"].astype(str) + " | " +
                out["plant"].astype(str) + " | " +
                out["product_category_1"].astype(str)
            )
        if "target" not in out.columns:
            out["target"] = "order_value_usd"
        if "model" not in out.columns:
            out["model"] = "Uploaded Forecast"
        if "split" not in out.columns:
            out["split"] = "current"
        if "is_best" not in out.columns:
            out["is_best"] = True
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        if "year" not in out.columns:
            out["year"] = out["date"].dt.year
        if "month" not in out.columns:
            out["month"] = out["date"].dt.month
        out["actual"] = pd.to_numeric(out["actual"], errors="coerce").fillna(0)
        out["forecast"] = pd.to_numeric(out["forecast"], errors="coerce").fillna(0)
        out["actual_clean"] = out["actual"].clip(lower=0)
        out["forecast_clean"] = out["forecast"].clip(lower=0)
        out["source_shape"] = "forecast_backtest_results"
        return enrich_accuracy(out)

    # Case 2: raw order history file matching sample rows from user
    # Column names after cleaning:
    # year, month, supply_chain_zone, plant_name, product_1_category_division_name,
    # sum_of_order_value_in_usd_from_lc, sum_of_order_quantity_bu
    aliases = {
        "year": ["year", "ear"],
        "month": ["month"],
        "codp_zone": ["supply_chain_zone", "codp_zone", "supply_chain"],
        "plant": ["plant_name", "plant"],
        "product_category_1": ["product_1_category_division_name", "product_category_1", "category"],
        "value": ["sum_of_order_value_in_usd_from_lc", "order_value_usd", "value_usd"],
        "qty": ["sum_of_order_quantity_bu", "sum_of_order_quantity_bu_", "order_qty_bu", "quantity_bu"],
    }

    def pick(keys):
        for k in keys:
            if k in df.columns:
                return k
        return None

    year_col = pick(aliases["year"])
    month_col = pick(aliases["month"])
    zone_col = pick(aliases["codp_zone"])
    plant_col = pick(aliases["plant"])
    cat_col = pick(aliases["product_category_1"])
    value_col = pick(aliases["value"])
    qty_col = pick(aliases["qty"])

    required = [year_col, month_col, zone_col, plant_col, cat_col]
    if any(c is None for c in required) or (value_col is None and qty_col is None):
        raise ValueError(
            "Input format not recognized. Expected either forecast_backtest_results columns "
            "or raw Chemelex order columns. Original columns: " + ", ".join(original_cols)
        )

    # Build a baseline forecast from raw order history for demo purposes.
    id_cols = [zone_col, plant_col, cat_col]
    records = []
    for target_name, source_col in [("order_value_usd", value_col), ("order_qty_bu", qty_col)]:
        if not source_col:
            continue
        tmp = df[[year_col, month_col, zone_col, plant_col, cat_col, source_col]].copy()
        tmp = tmp.rename(columns={
            year_col: "year", month_col: "month_raw", zone_col: "codp_zone",
            plant_col: "plant", cat_col: "product_category_1", source_col: "actual",
        })
        tmp["month"] = tmp["month_raw"].apply(lambda x: MONTH_TO_NUM.get(str(x).strip().lower(), x))
        tmp["month"] = pd.to_numeric(tmp["month"], errors="coerce")
        tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce")
        tmp["actual"] = pd.to_numeric(tmp["actual"], errors="coerce").fillna(0)
        tmp = tmp.dropna(subset=["year", "month"])
        tmp["year"] = tmp["year"].astype(int)
        tmp["month"] = tmp["month"].astype(int)
        tmp["date"] = pd.to_datetime(dict(year=tmp["year"], month=tmp["month"], day=1), errors="coerce")
        tmp["group_key"] = tmp["codp_zone"].astype(str) + " | " + tmp["plant"].astype(str) + " | " + tmp["product_category_1"].astype(str)
        tmp = tmp.sort_values(["group_key", "date"])
        tmp["forecast"] = (
            tmp.groupby("group_key")["actual"]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        )
        overall_mean = tmp["actual"].mean()
        tmp["forecast"] = tmp["forecast"].fillna(overall_mean).fillna(0)
        tmp["target"] = target_name
        tmp["split"] = "rolling_backtest"
        tmp["model"] = "MA3 Baseline"
        tmp["is_best"] = True
        records.append(tmp)

    out = pd.concat(records, ignore_index=True)
    out["actual_clean"] = out["actual"].clip(lower=0)
    out["forecast_clean"] = out["forecast"].clip(lower=0)
    out["source_shape"] = "raw_order_history_generated_baseline"
    return enrich_accuracy(out)


def enrich_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["month"] = pd.to_numeric(out["month"], errors="coerce").fillna(out["date"].dt.month).astype(int)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").fillna(out["date"].dt.year).astype(int)
    out["month_name"] = out["month"].apply(lambda m: MONTH_ORDER[int(m) - 1] if 1 <= int(m) <= 12 else str(m))
    out["actual_clean"] = pd.to_numeric(out.get("actual_clean", out["actual"]), errors="coerce").fillna(0).clip(lower=0)
    out["forecast_clean"] = pd.to_numeric(out.get("forecast_clean", out["forecast"]), errors="coerce").fillna(0).clip(lower=0)
    out["gap"] = out["forecast_clean"] - out["actual_clean"]
    out["abs_gap"] = out["gap"].abs()
    out["accuracy_pct"] = [simple_accuracy(a, f) for a, f in zip(out["actual_clean"], out["forecast_clean"])]
    out["accuracy_band"] = out["accuracy_pct"].apply(accuracy_band)
    out["target"] = out["target"].astype(str)
    out["target_label"] = out["target"].map(TARGET_LABELS).fillna(out["target"])
    out["codp_zone"] = out["codp_zone"].fillna("Unknown")
    out["plant"] = out["plant"].fillna("Unknown")
    out["product_category_1"] = out["product_category_1"].fillna("Unknown")
    out["group_key"] = out["group_key"].fillna(out["codp_zone"].astype(str) + " | " + out["plant"].astype(str) + " | " + out["product_category_1"].astype(str))
    out["model"] = out["model"].fillna("Unknown")
    out["split"] = out["split"].fillna("Unknown")
    # Convert is_best strings safely.
    if out["is_best"].dtype == object:
        out["is_best"] = out["is_best"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# DATA TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_view(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["actual", "forecast", "gap", "abs_gap", "accuracy_pct"])
    agg = df.groupby(group_cols, dropna=False).agg(
        actual=("actual_clean", "sum"),
        forecast=("forecast_clean", "sum"),
        rows=("actual_clean", "size"),
        models=("model", lambda x: ", ".join(sorted(set(map(str, x)))[:3])),
    ).reset_index()
    agg["gap"] = agg["forecast"] - agg["actual"]
    agg["abs_gap"] = agg["gap"].abs()
    agg["accuracy_pct"] = [simple_accuracy(a, f) for a, f in zip(agg["actual"], agg["forecast"])]
    p85 = agg["abs_gap"].quantile(0.85) if len(agg) else 0
    agg["priority"] = [priority_from_accuracy(a, g, p85) for a, g in zip(agg["accuracy_pct"], agg["abs_gap"])]
    agg["accuracy_band"] = agg["accuracy_pct"].apply(accuracy_band)
    agg["recommendation"] = agg.apply(make_recommendation, axis=1)
    return agg


def make_recommendation(row: pd.Series) -> str:
    acc = row.get("accuracy_pct", np.nan)
    gap = row.get("gap", 0)
    if pd.isna(acc):
        return "Check data quality before review."
    if acc >= 90:
        return "Forecast is close to actuals. Keep current plan and monitor."
    if gap > 0:
        return "Forecast is higher than actuals. Validate demand signal before increasing plan."
    if gap < 0:
        return "Actuals are higher than forecast. Review supply readiness and possible upside."
    return "No material gap. Monitor only."


def build_driver_table(df: pd.DataFrame) -> pd.DataFrame:
    monthly = aggregate_view(df, ["date"])
    if monthly.empty:
        return pd.DataFrame()
    monthly = monthly.sort_values("date")
    recent = monthly.tail(3)
    previous = monthly.iloc[-6:-3] if len(monthly) >= 6 else monthly.head(3)
    recent_avg = recent["actual"].mean() if len(recent) else 0
    prev_avg = previous["actual"].mean() if len(previous) else 0
    trend_delta = ((recent_avg - prev_avg) / prev_avg * 100) if prev_avg else 0
    vol = (monthly["actual"].std() / monthly["actual"].mean()) if monthly["actual"].mean() else 0
    acc = aggregate_accuracy(df["actual_clean"], df["forecast_clean"])
    gap = df["forecast_clean"].sum() - df["actual_clean"].sum()
    over_forecast = (gap > 0)
    zero_ratio = (df["actual_clean"].eq(0).mean() * 100) if len(df) else 0
    model_count = df["model"].nunique()
    family_count = df["group_key"].nunique()
    high_gap = (aggregate_view(df, ["group_key"])["priority"].eq("High").sum()) if len(df) else 0
    rows = [
        ["Recent Demand Trend", f"{recent_avg:,.0f}", f"{trend_delta:+.1f}%", "green" if trend_delta >= 0 else "red"],
        ["Demand Volatility", f"{vol:.2f}", "CV", "amber" if vol > 0.6 else "green"],
        ["Simple Accuracy", f"{acc:.1f}%", "min/max", "green" if acc >= 80 else "red"],
        ["Forecast Bias", "Over" if over_forecast else "Under", f"{gap:,.0f}", "red" if abs(gap) > 0 else "green"],
        ["Zero Demand Rows", f"{zero_ratio:.1f}%", "actual=0", "amber" if zero_ratio > 20 else "green"],
        ["Models Compared", f"{model_count}", "backtest", "green"],
        ["Families / Materials", f"{family_count}", "visible", "green"],
        ["High Gap Families", f"{high_gap}", "priority", "red" if high_gap else "green"],
    ]
    return pd.DataFrame(rows, columns=["Fluctuation Drivers", "Value", "Signal", "tone"])


def driver_table_html(drivers: pd.DataFrame) -> str:
    if drivers.empty:
        return "<p class='small-muted'>No drivers available.</p>"
    trs = []
    for i, r in drivers.iterrows():
        tone_class = {
            "green": "driver-green",
            "red": "driver-red",
            "amber": "driver-amber",
        }.get(r["tone"], "")
        check = "☑" if i in [0, 2] else "☐"
        trs.append(
            f"<tr><td>{check}</td><td>{r['Fluctuation Drivers']}</td>"
            f"<td class='{tone_class}'>{r['Value']}</td><td>{r['Signal']}</td></tr>"
        )
    return (
        "<table class='driver-table'><thead><tr><th></th><th>Fluctuation Drivers</th>"
        "<th>Value</th><th>Signal</th></tr></thead><tbody>" + "".join(trs) + "</tbody></table>"
    )


def model_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    lb = aggregate_view(df, ["model"])
    if lb.empty:
        return lb
    return lb.sort_values("accuracy_pct", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def category_forecast_chart(cat: pd.DataFrame, target: str):
    top = cat.sort_values("actual", ascending=False).head(8).copy()
    top_long = top.melt(
        id_vars=["product_category_1"],
        value_vars=["forecast", "actual"],
        var_name="Measure",
        value_name="Value",
    )
    top_long["Measure"] = top_long["Measure"].map({"forecast": "Forecast", "actual": "Actual"})
    fig = px.bar(
        top_long,
        x="product_category_1",
        y="Value",
        color="Measure",
        barmode="group",
        title="Categories",
        color_discrete_map={"Forecast": CHEMELEX_BLUE, "Actual": CHEMELEX_GREEN},
    )
    fig.update_layout(xaxis_title="", yaxis_title=TARGET_LABELS.get(target, target))
    fig.update_xaxes(tickangle=0)
    return style_plot(fig, 340)


def category_accuracy_chart(cat: pd.DataFrame):
    top = cat.sort_values("abs_gap", ascending=False).head(8).copy()
    fig = px.line(
        top,
        x="product_category_1",
        y="accuracy_pct",
        markers=True,
        title="Variance / Accuracy Analysis",
    )
    fig.update_traces(line_color=CHEMELEX_RED, marker=dict(size=8, color=CHEMELEX_RED))
    fig.update_layout(yaxis_title="Simple Accuracy %", xaxis_title="")
    fig.add_hline(y=90, line_dash="dot", line_color="#34A853", annotation_text="90% target")
    fig.add_hline(y=80, line_dash="dot", line_color="#F28C00", annotation_text="80% watch")
    return style_plot(fig, 340)


def monthly_trend_chart(monthly: pd.DataFrame, target: str):
    if monthly.empty:
        return go.Figure()
    monthly = monthly.sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["actual"], mode="lines+markers",
        name="Actual", line=dict(color=CHEMELEX_GREEN, width=3)
    ))
    fig.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["forecast"], mode="lines+markers",
        name="Forecast", line=dict(color=CHEMELEX_ORANGE, width=3)
    ))
    fig.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["gap"], mode="lines",
        name="Forecast - Actual Gap", line=dict(color=CHEMELEX_RED, width=2, dash="dot"), yaxis="y2"
    ))
    fig.update_layout(
        title="Forecasted Demand vs Actuals",
        xaxis_title="Month",
        yaxis_title=TARGET_LABELS.get(target, target),
        yaxis2=dict(title="Gap", overlaying="y", side="right", showgrid=False),
    )
    return style_plot(fig, 420)


def model_chart(lb: pd.DataFrame):
    if lb.empty:
        return go.Figure()
    top = lb.head(10).sort_values("accuracy_pct")
    fig = px.bar(
        top,
        x="accuracy_pct",
        y="model",
        orientation="h",
        text=top["accuracy_pct"].map(lambda x: f"{x:.1f}%"),
        title="Model Accuracy Leaderboard",
        color="accuracy_pct",
        color_continuous_scale=[[0, "#F0645A"], [0.5, "#F28C00"], [1, "#34A853"]],
    )
    fig.update_layout(xaxis_title="Simple Accuracy %", yaxis_title="")
    return style_plot(fig, 380)


def heatmap_chart(df: pd.DataFrame):
    h = aggregate_view(df, ["codp_zone", "plant"])
    if h.empty:
        return go.Figure()
    pivot = h.pivot_table(index="codp_zone", columns="plant", values="accuracy_pct", aggfunc="mean")
    fig = px.imshow(
        pivot,
        text_auto=".0f",
        aspect="auto",
        color_continuous_scale=[[0, "#F0645A"], [0.5, "#F28C00"], [1, "#34A853"]],
        title="Accuracy Heatmap by CODP Zone and Plant",
    )
    return style_plot(fig, 380)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Forecast Backtest Input")
    st.caption("Upload Synthefy/Excel output or use the default repo file.")
    uploaded = st.file_uploader(
        "Upload CSV / Excel",
        type=["csv", "xlsx", "xls"],
        help="Preferred: forecast_backtest_results.csv",
    )
    st.markdown("---")
    st.markdown("### Accuracy formula")
    st.markdown(
        """
<div class='formula-box'>
<b>Simple Accuracy %</b><br>
Accuracy = min(Actual, Forecast) ÷ max(Actual, Forecast) × 100<br><br>
Example: Actual = 90, Forecast = 100 → <b>90%</b>
</div>
""",
        unsafe_allow_html=True,
    )
    st.caption("No MAPE shown to end users.")

try:
    if uploaded is not None:
        raw_df = read_any_file(uploaded)
        source_note = f"Uploaded: {uploaded.name}"
    else:
        raw_df, source_note = load_default_or_demo()
    data = standardize_input(raw_df)
except Exception as e:
    st.error(f"Unable to read the file: {e}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
add_chex_shell()


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL FILTERS
# ─────────────────────────────────────────────────────────────────────────────
with st.container():
    st.markdown("<div class='page-card'>", unsafe_allow_html=True)
    filter_cols = st.columns([2.2, 1.7, 1.7, 1.7, 1.6, 1.4, 1.2])
    with filter_cols[0]:
        st.markdown("<div class='section-title'>Material Forecast</div>", unsafe_allow_html=True)
    targets = sorted(data["target"].dropna().unique().tolist())
    with filter_cols[1]:
        target = st.selectbox("Target", targets, format_func=lambda x: TARGET_LABELS.get(x, x), label_visibility="collapsed")
    base = data[data["target"] == target].copy()

    with filter_cols[2]:
        best_only = st.selectbox("Model rows", ["Best model only", "All models"], label_visibility="collapsed")
    if best_only == "Best model only" and "is_best" in base.columns:
        base = base[base["is_best"] == True].copy()

    plants = ["All"] + sorted(base["plant"].dropna().astype(str).unique().tolist())
    zones = ["All"] + sorted(base["codp_zone"].dropna().astype(str).unique().tolist())
    models = ["All"] + sorted(base["model"].dropna().astype(str).unique().tolist())
    splits = ["All"] + sorted(base["split"].dropna().astype(str).unique().tolist())

    with filter_cols[3]:
        plant_filter = st.selectbox("Plant", plants, label_visibility="collapsed")
    with filter_cols[4]:
        zone_filter = st.selectbox("CODP Zone", zones, label_visibility="collapsed")
    with filter_cols[5]:
        model_filter = st.selectbox("Model", models, label_visibility="collapsed")
    with filter_cols[6]:
        split_filter = st.selectbox("Split", splits, label_visibility="collapsed")

    st.markdown("</div>", unsafe_allow_html=True)

filtered = base.copy()
if plant_filter != "All":
    filtered = filtered[filtered["plant"] == plant_filter]
if zone_filter != "All":
    filtered = filtered[filtered["codp_zone"] == zone_filter]
if model_filter != "All":
    filtered = filtered[filtered["model"] == model_filter]
if split_filter != "All":
    filtered = filtered[filtered["split"] == split_filter]

if filtered.empty:
    st.warning("No rows match the selected filters.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# PREPARE AGGREGATES
# ─────────────────────────────────────────────────────────────────────────────
total_actual = filtered["actual_clean"].sum()
total_forecast = filtered["forecast_clean"].sum()
total_gap = total_forecast - total_actual
total_accuracy = simple_accuracy(total_actual, total_forecast)
active_groups = filtered["group_key"].nunique()
cat = aggregate_view(filtered, ["product_category_1"])
material = aggregate_view(filtered, ["group_key", "codp_zone", "plant", "product_category_1", "model", "split"])
monthly = aggregate_view(filtered, ["date"])
monthly["date"] = pd.to_datetime(monthly["date"], errors="coerce")
watch = material.sort_values(["priority", "abs_gap"], ascending=[True, False]).copy()
watch_display = watch.copy()


# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Active Materials", f"{active_groups:,}")
k2.metric("Total Forecast", fmt_value(total_forecast, target))
k3.metric("Total Actual", fmt_value(total_actual, target))
k4.metric("Gap", fmt_value(total_gap, target), delta=f"{((total_gap/total_actual)*100 if total_actual else 0):+.1f}%")
k5.metric("Accuracy %", f"{total_accuracy:.1f}%", delta=accuracy_band(total_accuracy))

st.caption(f"{source_note} · {len(filtered):,} visible rows · Formula: min(Actual, Forecast) / max(Actual, Forecast).")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Material Forecast", "Forecast Analysis", "Accuracy Workbench"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — MATERIAL FORECAST
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.markdown("<div class='soft-panel'>", unsafe_allow_html=True)
        st.plotly_chart(category_forecast_chart(cat, target), use_container_width=True, key="category_forecast_chart")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='soft-panel'>", unsafe_allow_html=True)
        chip_class = "good-chip" if total_accuracy >= 90 else "warn-chip" if total_accuracy >= 80 else "bad-chip"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<div class='subtle-title'>Variance Analysis</div>"
            f"<div class='accuracy-chip {chip_class}'>Accuracy: {total_accuracy:.1f}%</div></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(category_accuracy_chart(cat), use_container_width=True, key="category_accuracy_chart")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='page-card'>", unsafe_allow_html=True)
    table_cols = [
        "group_key", "codp_zone", "plant", "product_category_1", "model", "split",
        "actual", "forecast", "gap", "accuracy_pct", "priority", "recommendation",
    ]
    table = material[table_cols].sort_values("abs_gap", ascending=False).head(100).copy()
    table = table.rename(columns={
        "group_key": "Material / Forecast Group",
        "codp_zone": "CODP Zone",
        "plant": "Plant",
        "product_category_1": "Category",
        "model": "Model",
        "split": "Split",
        "actual": "Actual",
        "forecast": "Forecast",
        "gap": "Gap",
        "accuracy_pct": "Accuracy %",
        "priority": "Priority",
        "recommendation": "Planner Recommendation",
    })
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
        f"<div><b>SKU / Material-wise Forecast Accuracy</b><br><span class='small-muted'>{len(material):,} groups · Showing top 100 by absolute gap</span></div>"
        f"<span class='download-dot'>↓</span></div>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        table,
        use_container_width=True,
        height=420,
        hide_index=True,
        column_config={
            "Actual": st.column_config.NumberColumn(format="%.2f"),
            "Forecast": st.column_config.NumberColumn(format="%.2f"),
            "Gap": st.column_config.NumberColumn(format="%.2f"),
            "Accuracy %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
        },
    )
    st.download_button(
        "⬇️ Download Material Forecast Accuracy CSV",
        data=to_csv_bytes(table),
        file_name="chemelex_material_forecast_accuracy.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — FORECAST ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    fa1, fa2, fa3 = st.columns(3)
    fa1.metric("Baseline Actual", fmt_value(total_actual, target))
    fa2.metric("Total Gap / Impact", fmt_value(abs(total_gap), target), delta="Forecast - Actual")
    fa3.metric("Final Forecast", fmt_value(total_forecast, target), delta=f"{total_accuracy:.1f}% accuracy")

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([0.95, 2.75])
    with left:
        st.markdown("<div class='soft-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='subtle-title'>Fluctuation Drivers</div>", unsafe_allow_html=True)
        st.markdown(driver_table_html(build_driver_table(filtered)), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='soft-panel'>", unsafe_allow_html=True)
        st.plotly_chart(monthly_trend_chart(monthly, target), use_container_width=True, key="monthly_trend_chart")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='page-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>AI Planner Query Concept</div>", unsafe_allow_html=True)
    query_cols = st.columns([1.4, 1, 1])
    with query_cols[0]:
        selected_group = st.selectbox("Ask about Material / Forecast Group", ["Portfolio Total"] + material["group_key"].head(100).tolist())
    if selected_group == "Portfolio Total":
        q_actual, q_forecast, q_acc, q_gap = total_actual, total_forecast, total_accuracy, total_gap
    else:
        row = material[material["group_key"] == selected_group].iloc[0]
        q_actual, q_forecast, q_acc, q_gap = row["actual"], row["forecast"], row["accuracy_pct"], row["gap"]
    with query_cols[1]:
        st.metric("Selected Accuracy", f"{q_acc:.1f}%")
    with query_cols[2]:
        st.metric("Selected Gap", fmt_value(q_gap, target))
    st.markdown(
        f"""
<div class='ai-box'>
<b>Planner question:</b> Do you think my demand is going to change in the next three months?<br><br>
<b>AI-style answer:</b> Current forecast accuracy is <b>{q_acc:.1f}%</b> using the simple business formula.
Forecast is {'above' if q_gap > 0 else 'below' if q_gap < 0 else 'aligned with'} actual demand by <b>{fmt_value(abs(q_gap), target)}</b>.
Recommended action: <b>{make_recommendation(pd.Series({'accuracy_pct': q_acc, 'gap': q_gap}))}</b><br><br>
<span class='small-muted'>Data looked at: actuals, forecast, CODP zone, plant, product category, model, split and month-wise backtest history.</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ACCURACY WORKBENCH
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='formula-box'>", unsafe_allow_html=True)
    st.markdown(
        """
### End-user Accuracy Logic
We are deliberately showing a plain business accuracy number, not MAPE.

**Formula:** `Accuracy % = min(Actual, Forecast) / max(Actual, Forecast) × 100`

Example: if Actual = 90 meters and Forecast = 100 meters, Accuracy = 90 / 100 = **90%**.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    m1, m2 = st.columns([1, 1])
    with m1:
        st.plotly_chart(model_chart(model_leaderboard(filtered)), use_container_width=True, key="model_leaderboard_chart")
    with m2:
        st.plotly_chart(heatmap_chart(filtered), use_container_width=True, key="accuracy_heatmap_chart")

    st.markdown("<div class='page-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Accuracy Watchlist</div>", unsafe_allow_html=True)
    watch_cols = [
        "group_key", "codp_zone", "plant", "product_category_1", "actual", "forecast",
        "gap", "accuracy_pct", "accuracy_band", "priority", "recommendation",
    ]
    watch_table = watch[watch_cols].sort_values(["priority", "accuracy_pct", "abs_gap"], ascending=[True, True, False]).copy()
    watch_table = watch_table.rename(columns={
        "group_key": "Material / Forecast Group",
        "codp_zone": "CODP Zone",
        "plant": "Plant",
        "product_category_1": "Category",
        "actual": "Actual",
        "forecast": "Forecast",
        "gap": "Gap",
        "accuracy_pct": "Accuracy %",
        "accuracy_band": "Accuracy Band",
        "priority": "Priority",
        "recommendation": "Recommended Action",
    })
    st.dataframe(
        watch_table,
        use_container_width=True,
        height=440,
        hide_index=True,
        column_config={
            "Actual": st.column_config.NumberColumn(format="%.2f"),
            "Forecast": st.column_config.NumberColumn(format="%.2f"),
            "Gap": st.column_config.NumberColumn(format="%.2f"),
            "Accuracy %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
        },
    )
    st.download_button(
        "⬇️ Download Accuracy Watchlist",
        data=to_csv_bytes(watch_table),
        file_name="chemelex_accuracy_watchlist.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.caption(
    "Chemelex Demand Forecasting Cockpit · Built for forecast backtest demo · "
    "Accuracy shown as min(actual, forecast) / max(actual, forecast), not MAPE."
)
