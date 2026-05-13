"""
Chemelex Demand Forecasting Cockpit
====================================
Executive view of forecast performance, demand-plan alignment,
and IBP-vs-ML head-to-head comparison.

Primary data file (auto-loaded if present in working directory):
    corrected_dashboard_forecast_accuracy_levelled.csv

Optional companion files (auto-loaded if present):
    ibp_vs_model_comparison_2025.csv  - Synthefy's pre-computed IBP-vs-ML
                                         category-month comparison
    ibp_reconciliation.csv            - Reconciliation audit trail
    ibp_pipeline.md                   - IBP pipeline documentation

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Chemelex Demand Forecasting Cockpit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRIMARY_FILE = "corrected_dashboard_forecast_accuracy_levelled.csv"
COMPARISON_FILE = "ibp_vs_model_comparison_2025.csv"
RECONCILIATION_FILE = "ibp_reconciliation.csv"
PIPELINE_DOC = "ibp_pipeline.md"

OFFICIAL_CODP_ZONES = [
    "BUFFER (at CODP)",
    "Pull / FG",
    "Pull / Kanban",
    "Push / MPS",
    "Push / RM",
]

FORECAST_LEVELS = ["Total", "CODP Zone", "Plant", "Product Category", "Group"]

# Defaults
DEFAULT_LEVEL = "Group"
DEFAULT_TARGET = "order_value_usd"
DEFAULT_SPLIT = "split2"
DEFAULT_YEAR = 2025

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# IBP-relevant constants (from ibp_pipeline.md)
IBP_OBSERVED_MONTHS = pd.date_range("2025-01-01", "2025-05-01", freq="MS")
IBP_COVERED_CATEGORIES = [
    "PD / Heat Tracing Components",
    "PD / Floor Heating",
    "PD / Control, Monitoring & Power Distribution",
    "PD / Polymer Pipe Heat Tracing - BIS",
    "PD / Polymer Pipe Heat Tracing - IND",
    "PD / Snow Melting & De-Icing",
]

# ---------------------------------------------------------------------------
# Theme system
# ---------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Light"


def get_theme() -> dict:
    """Color tokens for the active theme."""
    if st.session_state.theme == "Dark":
        return {
            "bg":           "#0b1220",
            "surface":      "#111a2e",
            "surface_2":    "#18233d",
            "border":       "#1f2d4d",
            "text":         "#e5edff",
            "text_muted":   "#8aa0c8",
            "text_subtle":  "#5f7494",
            "primary":      "#4f8bff",
            "accent":       "#7ba6ff",
            "success":      "#2dd4bf",
            "warning":      "#fbbf24",
            "danger":       "#f87171",
            "info":         "#60a5fa",
            "ibp":          "#f59e0b",     # IBP forecast (orange)
            "model":        "#4f8bff",     # Model forecast (blue)
            "actual":       "#10b981",     # Actuals (green)
            "header_grad":  "linear-gradient(135deg, #1e3a8a 0%, #3b5bdb 100%)",
            "warn_bg":      "#3a2e15",
            "warn_text":    "#fbbf24",
            "plotly_tmpl":  "plotly_dark",
        }
    return {
        "bg":           "#f5f7fb",
        "surface":      "#ffffff",
        "surface_2":    "#f1f5f9",
        "border":       "#e2e8f0",
        "text":         "#0f172a",
        "text_muted":   "#475569",
        "text_subtle":  "#64748b",
        "primary":      "#2563eb",
        "accent":       "#3b82f6",
        "success":      "#059669",
        "warning":      "#d97706",
        "danger":       "#dc2626",
        "info":         "#0284c7",
        "ibp":          "#ea580c",
        "model":        "#2563eb",
        "actual":       "#059669",
        "header_grad":  "linear-gradient(135deg, #1e3a8a 0%, #3b5bdb 100%)",
        "warn_bg":      "#fef3c7",
        "warn_text":    "#92400e",
        "plotly_tmpl":  "plotly_white",
    }


def inject_css() -> None:
    """One block of CSS, theme-aware, no raw-HTML leakage risk."""
    c = get_theme()
    css = f"""
    <style>
      /* ---- App canvas ---- */
      .stApp {{
        background: {c['bg']};
        color: {c['text']};
      }}
      .block-container {{
        padding-top: 1.0rem;
        padding-bottom: 2.0rem;
        max-width: 1480px;
      }}
      /* ---- Header band ---- */
      .cx-header {{
        background: {c['header_grad']};
        padding: 18px 24px;
        border-radius: 14px;
        margin: 6px 0 14px 0;
        color: white;
        box-shadow: 0 6px 22px rgba(15, 23, 42, 0.18);
      }}
      .cx-header h1 {{
        color: white;
        margin: 0;
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.01em;
      }}
      .cx-header p {{
        color: rgba(255,255,255,0.86);
        margin: 4px 0 0 0;
        font-size: 13.5px;
      }}
      /* ---- Section title ---- */
      .cx-section {{
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {c['text_muted']};
        margin: 18px 0 10px 0;
      }}
      /* ---- KPI cards ---- */
      .cx-kpi {{
        background: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 10px;
        padding: 14px 16px;
        height: 100%;
      }}
      .cx-kpi-title {{
        color: {c['text_muted']};
        font-size: 11.5px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
        font-weight: 600;
      }}
      .cx-kpi-value {{
        color: {c['text']};
        font-size: 24px;
        font-weight: 700;
        line-height: 1.2;
      }}
      .cx-kpi-sub {{
        font-size: 11.5px;
        margin-top: 6px;
        font-weight: 500;
      }}
      .cx-sub-pos {{ color: {c['success']}; }}
      .cx-sub-neg {{ color: {c['danger']}; }}
      .cx-sub-warn {{ color: {c['warning']}; }}
      .cx-sub-info {{ color: {c['text_muted']}; }}
      /* ---- Insight cards ---- */
      .cx-insight {{
        border-radius: 12px;
        padding: 16px 18px;
        color: white;
        height: 100%;
        position: relative;
      }}
      .cx-ins-green {{
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
      }}
      .cx-ins-blue {{
        background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%);
      }}
      .cx-ins-amber {{
        background: linear-gradient(135deg, #b45309 0%, #f59e0b 100%);
      }}
      .cx-ins-title {{
        color: white;
        font-size: 14.5px;
        font-weight: 700;
        margin: 0 0 8px 0;
      }}
      .cx-ins-line {{
        color: rgba(255,255,255,0.92);
        font-size: 13px;
        margin: 2px 0;
        line-height: 1.5;
      }}
      .cx-ins-strong {{ font-weight: 700; }}
      /* ---- Warning banner ---- */
      .cx-warn {{
        background: {c['warn_bg']};
        border-left: 4px solid {c['warning']};
        color: {c['warn_text']};
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 13px;
        margin: 8px 0 14px 0;
        line-height: 1.55;
      }}
      /* ---- Filter card ---- */
      .cx-filter-label {{
        font-size: 13px;
        font-weight: 700;
        color: {c['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
      }}
      /* ---- Tab styling ---- */
      .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
      }}
      .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
        font-size: 14px;
        font-weight: 600;
      }}
      .stTabs [data-baseweb="tab"] {{
        padding: 8px 16px;
        border-radius: 8px 8px 0 0;
      }}
      /* ---- Hide Streamlit chrome ---- */
      footer {{ visibility: hidden; }}
      #MainMenu {{ visibility: hidden; }}
      header[data-testid="stHeader"] {{ background: transparent; }}
      /* ---- DataFrames ---- */
      [data-testid="stDataFrame"] {{
        border: 1px solid {c['border']};
        border-radius: 8px;
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading & normalization
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _read_csv_bytes(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content), low_memory=False)


@st.cache_data(show_spinner=False)
def _read_excel_bytes(content: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(content))


@st.cache_data(show_spinner=False)
def _read_csv_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def normalize_main(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the main forecast accuracy dataframe."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # is_best handles 1/0, True/False, "1"/"0", "True"/"False"
    if "is_best" in df.columns:
        df["is_best"] = (
            df["is_best"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0,
                  "nan": 0})
            .fillna(0)
            .astype(int)
        )

    # Numeric coercion
    for col in [
        "actual", "forecast", "accuracy_pct", "error", "abs_error", "bias_pct",
        "wape_component_num", "wape_component_den", "year", "month",
        "ibp_forecast_cogs", "ibp_actual_cogs", "ibp_accuracy_pct",
        "is_forward",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "year" not in df.columns or df["year"].isna().all():
            df["year"] = df["date"].dt.year
        if "month" not in df.columns or df["month"].isna().all():
            df["month"] = df["date"].dt.month

    # Strip strings
    for col in ["level", "dimension_value", "codp_zone", "plant",
                "product_category_1", "target", "model", "split",
                "data_quality_flags", "model_vs_ibp_winner"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(["nan", "NaN", "None", ""]), col] = np.nan

    # year/month to int
    if "year" in df.columns:
        df["year"] = df["year"].fillna(-1).astype(int)
    if "month" in df.columns:
        df["month"] = df["month"].fillna(-1).astype(int)

    return df


def try_autoload() -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], list[str]]:
    """Auto-load main + comparison file from working dir."""
    msgs = []
    main = None
    comp = None

    if Path(PRIMARY_FILE).exists():
        try:
            main = normalize_main(_read_csv_path(PRIMARY_FILE))
            msgs.append(f"📂 Auto-loaded `{PRIMARY_FILE}` ({len(main):,} rows)")
        except Exception as e:
            msgs.append(f"⚠️ Could not auto-load `{PRIMARY_FILE}`: {e}")

    if Path(COMPARISON_FILE).exists():
        try:
            comp = _read_csv_path(COMPARISON_FILE)
            msgs.append(f"📂 Auto-loaded `{COMPARISON_FILE}` ({len(comp):,} rows)")
        except Exception as e:
            msgs.append(f"⚠️ Could not auto-load `{COMPARISON_FILE}`: {e}")

    return main, comp, msgs


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_money(x: float, compact: bool = True) -> str:
    if pd.isna(x):
        return "—"
    a = abs(x)
    sign = "-" if x < 0 else ""
    if compact:
        if a >= 1_000_000_000:
            return f"{sign}${a/1_000_000_000:.2f}B"
        if a >= 1_000_000:
            return f"{sign}${a/1_000_000:.2f}M"
        if a >= 1_000:
            return f"{sign}${a/1_000:.1f}K"
    return f"{sign}${a:,.0f}"


def fmt_int(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{int(x):,}"


def fmt_pct(x: float, decimals: int = 1) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}%"


# ---------------------------------------------------------------------------
# UI primitives
# ---------------------------------------------------------------------------
def header() -> None:
    st.markdown(
        """
        <div class="cx-header">
          <h1>📊 Chemelex Demand Forecasting Cockpit</h1>
          <p>Executive view of forecast performance, demand-plan alignment, and IBP-vs-ML head-to-head</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(title: str, value: str, sub: str = "", sub_class: str = "cx-sub-info") -> None:
    sub_html = f'<div class="cx-kpi-sub {sub_class}">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="cx-kpi">
          <div class="cx-kpi-title">{title}</div>
          <div class="cx-kpi-value">{value}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    st.markdown(f'<div class="cx-section">{text}</div>', unsafe_allow_html=True)


def empty_fig(message: str) -> go.Figure:
    c = get_theme()
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=14, color=c["text_muted"]),
    )
    fig.update_layout(
        template=c["plotly_tmpl"], height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def apply_filters(
    df: pd.DataFrame, *,
    year: Optional[int],
    months: list[int],
    codp_zones: list[str],
    plants: list[str],
    categories: list[str],
    level: str,
    target: Optional[str],
    split: Optional[str],
    best_only: bool,
    model: Optional[str],
    search: str = "",
    include_aggregate_rows: bool = True,
) -> pd.DataFrame:
    """Apply business filters. `include_aggregate_rows` lets 'All' pass for
    zone/plant/category when looking at non-Group levels."""
    out = df.copy()

    if year is not None and "year" in out.columns:
        out = out[out["year"] == year]
    if months and "month" in out.columns:
        out = out[out["month"].isin(months)]
    if level and "level" in out.columns:
        out = out[out["level"] == level]
    if target and "target" in out.columns:
        out = out[out["target"] == target]
    if split and "split" in out.columns:
        out = out[out["split"] == split]
    if best_only and "is_best" in out.columns:
        out = out[out["is_best"] == 1]
    if model and "model" in out.columns:
        out = out[out["model"] == model]

    # Dimension filters — keep 'All' rows when filtering by non-Group level
    extra = ["All"] if include_aggregate_rows else []
    if codp_zones and "codp_zone" in out.columns:
        out = out[out["codp_zone"].isin(codp_zones + extra)]
    if plants and "plant" in out.columns:
        out = out[out["plant"].isin(plants + extra)]
    if categories and "product_category_1" in out.columns:
        out = out[out["product_category_1"].isin(categories + extra)]

    if search and "dimension_value" in out.columns:
        s = search.strip().lower()
        out = out[out["dimension_value"].astype(str).str.lower().str.contains(s, na=False)]

    return out


# ---------------------------------------------------------------------------
# KPI calculations
# ---------------------------------------------------------------------------
def compute_kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {k: np.nan for k in
                ["actual", "forecast", "wape", "alignment", "bias", "gap",
                 "active_items", "at_risk_items"]}

    total_actual = df["actual"].sum() if "actual" in df.columns else np.nan
    total_forecast = df["forecast"].sum() if "forecast" in df.columns else np.nan

    if "abs_error" in df.columns and total_actual and total_actual != 0:
        wape = df["abs_error"].sum() / total_actual * 100
    else:
        wape = np.nan

    alignment = (total_actual / total_forecast * 100) if total_forecast else np.nan
    bias = ((total_forecast - total_actual) / total_actual * 100) if total_actual else np.nan
    gap = total_actual - total_forecast if (not pd.isna(total_actual) and not pd.isna(total_forecast)) else np.nan

    active_items = df["dimension_value"].nunique() if "dimension_value" in df.columns else len(df)
    # At-risk: alignment outside 80-120% band
    if "dimension_value" in df.columns and {"actual", "forecast"}.issubset(df.columns):
        per_item = df.groupby("dimension_value", dropna=False).agg(
            a=("actual", "sum"), f=("forecast", "sum")
        )
        per_item["align"] = np.where(per_item["f"] != 0,
                                      per_item["a"] / per_item["f"] * 100,
                                      np.nan)
        at_risk = int(((per_item["align"] < 80) | (per_item["align"] > 120)).sum())
    else:
        at_risk = 0

    return dict(actual=total_actual, forecast=total_forecast, wape=wape,
                alignment=alignment, bias=bias, gap=gap,
                active_items=active_items, at_risk_items=at_risk)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def chart_monthly_fa(df: pd.DataFrame) -> go.Figure:
    c = get_theme()
    if df.empty or not {"date", "forecast", "actual"}.issubset(df.columns):
        return empty_fig("No time-series data")
    g = (df.dropna(subset=["date"])
           .groupby("date", as_index=False)[["forecast", "actual"]]
           .sum().sort_values("date"))
    if g.empty:
        return empty_fig("No data after filtering")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=g["date"], y=g["forecast"], mode="lines+markers",
        name="Forecast", line=dict(color=c["model"], width=3),
        marker=dict(size=7),
    ))
    fig.add_trace(go.Scatter(
        x=g["date"], y=g["actual"], mode="lines+markers",
        name="Actual", line=dict(color=c["actual"], width=3),
        marker=dict(size=7),
    ))
    fig.update_layout(
        template=c["plotly_tmpl"], height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    return fig


def chart_alignment_by_month(df: pd.DataFrame) -> go.Figure:
    c = get_theme()
    if df.empty or not {"date", "actual", "forecast"}.issubset(df.columns):
        return empty_fig("No data")
    g = (df.dropna(subset=["date"])
           .groupby("date", as_index=False).agg(a=("actual", "sum"), f=("forecast", "sum"))
           .sort_values("date"))
    g["alignment"] = np.where(g["f"] != 0, g["a"] / g["f"] * 100, np.nan)
    if g.empty:
        return empty_fig("No data")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=g["date"], y=g["alignment"], mode="lines+markers",
        line=dict(color=c["primary"], width=3),
        marker=dict(size=8, color=c["primary"]),
        fill="tozeroy",
        fillcolor=f"rgba(37, 99, 235, 0.08)" if st.session_state.theme == "Light"
                 else f"rgba(79, 139, 255, 0.10)",
        name="Alignment %",
    ))
    fig.add_hline(y=100, line_dash="dash", line_color=c["text_muted"],
                  annotation_text="Perfect (100%)",
                  annotation_position="top right",
                  annotation_font_color=c["text_muted"])
    fig.update_layout(
        template=c["plotly_tmpl"], height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(ticksuffix="%"),
        showlegend=False, hovermode="x unified",
    )
    return fig


def chart_category(df: pd.DataFrame) -> go.Figure:
    c = get_theme()
    if df.empty or "product_category_1" not in df.columns:
        return empty_fig("No category data")
    g = (df.groupby("product_category_1")
           .agg(forecast=("forecast", "sum"), actual=("actual", "sum"))
           .reset_index()
           .sort_values("actual", ascending=False)
           .head(10))
    if g.empty:
        return empty_fig("No data")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Forecast", x=g["product_category_1"], y=g["forecast"],
                        marker_color=c["model"]))
    fig.add_trace(go.Bar(name="Actual", x=g["product_category_1"], y=g["actual"],
                        marker_color=c["actual"]))
    fig.update_layout(
        template=c["plotly_tmpl"], barmode="group",
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    fig.update_xaxes(tickangle=-25)
    return fig


def chart_wape_by_zone(df: pd.DataFrame) -> go.Figure:
    c = get_theme()
    if df.empty or not {"codp_zone", "abs_error", "actual"}.issubset(df.columns):
        return empty_fig("No zone data")
    g = (df.groupby("codp_zone")
           .agg(ae=("abs_error", "sum"), a=("actual", "sum"))
           .reset_index())
    g = g[g["codp_zone"].isin(OFFICIAL_CODP_ZONES)]
    g["wape"] = np.where(g["a"] != 0, g["ae"] / g["a"] * 100, np.nan)
    g = g.dropna(subset=["wape"]).sort_values("wape")
    if g.empty:
        return empty_fig("No zone data")

    colors_bars = [c["success"] if v < 15 else (c["warning"] if v < 30 else c["danger"])
                   for v in g["wape"]]
    fig = go.Figure(go.Bar(
        x=g["wape"], y=g["codp_zone"], orientation="h",
        marker_color=colors_bars,
        text=[f"{v:.1f}%" for v in g["wape"]],
        textposition="outside",
    ))
    fig.update_layout(
        template=c["plotly_tmpl"], height=340,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(ticksuffix="%"),
    )
    return fig


# ---------------------------------------------------------------------------
# Business table builder
# ---------------------------------------------------------------------------
def build_business_table(df: pd.DataFrame, level: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    # Group keys per level
    if level == "Total":
        group_cols = []
    elif level == "CODP Zone":
        group_cols = ["codp_zone"]
    elif level == "Plant":
        group_cols = ["plant"]
    elif level == "Product Category":
        group_cols = ["product_category_1"]
    else:  # Group
        group_cols = ["codp_zone", "plant", "product_category_1"]

    group_cols = [c for c in group_cols if c in df.columns]

    if group_cols:
        g = df.groupby(group_cols, dropna=False).agg(
            Forecast=("forecast", "sum"),
            Actual=("actual", "sum"),
        ).reset_index()
    else:
        g = pd.DataFrame({
            "Forecast": [df["forecast"].sum() if "forecast" in df.columns else np.nan],
            "Actual": [df["actual"].sum() if "actual" in df.columns else np.nan],
        })

    g["Forecast Alignment %"] = np.where(
        g["Forecast"].abs() > 1e-6, g["Actual"] / g["Forecast"] * 100, np.nan
    )
    g["Gap"] = g["Actual"] - g["Forecast"]

    # Line Item label
    if level == "Group" and {"codp_zone", "plant", "product_category_1"}.issubset(g.columns):
        g["Line Item"] = (g["codp_zone"].astype(str) + " | " +
                          g["plant"].astype(str) + " | " +
                          g["product_category_1"].astype(str))
    elif level == "CODP Zone" and "codp_zone" in g.columns:
        g["Line Item"] = g["codp_zone"].astype(str)
    elif level == "Plant" and "plant" in g.columns:
        g["Line Item"] = g["plant"].astype(str)
    elif level == "Product Category" and "product_category_1" in g.columns:
        g["Line Item"] = g["product_category_1"].astype(str)
    else:
        g["Line Item"] = "Total"

    # Data status
    def _status(row):
        a = row.get("Forecast Alignment %", np.nan)
        if pd.isna(a):
            return "⚪ N/A"
        if 90 <= a <= 110:
            return "🟢 On Track"
        if 80 <= a < 90 or 110 < a <= 120:
            return "🟡 Watch"
        return "🔴 Off Track"

    g["Data Status"] = g.apply(_status, axis=1)
    g["Forecast Level"] = level

    # Rename for display
    rename = {"codp_zone": "CODP / Supply Chain Zone",
              "plant": "Plant",
              "product_category_1": "Product Category"}
    g = g.rename(columns=rename)

    # Column order — business-facing only
    cols = ["Forecast Level", "Line Item"]
    for opt in ["CODP / Supply Chain Zone", "Plant", "Product Category"]:
        if opt in g.columns:
            cols.append(opt)
    cols += ["Forecast", "Actual", "Forecast Alignment %", "Gap", "Data Status"]
    g = g[[c for c in cols if c in g.columns]]
    g = g.sort_values("Actual", ascending=False, na_position="last").reset_index(drop=True)
    return g


def format_business_table(df: pd.DataFrame, currency_compact: bool = False) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "Forecast" in out:
        out["Forecast"] = out["Forecast"].apply(lambda v: fmt_money(v, compact=currency_compact))
    if "Actual" in out:
        out["Actual"] = out["Actual"].apply(lambda v: fmt_money(v, compact=currency_compact))
    if "Gap" in out:
        out["Gap"] = out["Gap"].apply(lambda v: fmt_money(v, compact=currency_compact))
    if "Forecast Alignment %" in out:
        out["Forecast Alignment %"] = out["Forecast Alignment %"].apply(fmt_pct)
    return out


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------
def main():
    inject_css()
    header()

    # ---- Top controls: upload + theme ---------------------------------------
    tc1, tc2, tc3 = st.columns([5, 2, 1.5])
    with tc1:
        upload = st.file_uploader(
            "📁 Upload / Replace Forecast File  (csv, xlsx)",
            type=["csv", "xlsx", "xls"],
            label_visibility="visible",
            key="uploader",
        )
    with tc2:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        theme = st.radio(
            "Theme",
            options=["Light", "Dark"],
            index=0 if st.session_state.theme == "Light" else 1,
            horizontal=True,
            key="theme_radio",
        )
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.rerun()
    with tc3:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.caption(f"Default: `{PRIMARY_FILE}`")

    # ---- Load data ---------------------------------------------------------
    df: Optional[pd.DataFrame] = None
    comp_df: Optional[pd.DataFrame] = None
    autoload_messages: list[str] = []

    if upload is not None:
        try:
            content = upload.getvalue()
            if upload.name.lower().endswith((".xlsx", ".xls")):
                raw = _read_excel_bytes(content)
            else:
                raw = _read_csv_bytes(content)
            df = normalize_main(raw)
            st.success(f"✅ Loaded `{upload.name}` ({len(df):,} rows)")
        except Exception as e:
            st.error(f"Could not read uploaded file: {e}")
    else:
        df, comp_df, autoload_messages = try_autoload()

    if df is None or df.empty:
        st.info(
            f"👋 **Welcome.** Place `{PRIMARY_FILE}` next to `app.py`, "
            "or upload a forecast file above to begin."
        )
        for m in autoload_messages:
            st.caption(m)
        st.stop()

    # If uploaded, try to also auto-load companion comparison file from disk
    if comp_df is None and Path(COMPARISON_FILE).exists():
        try:
            comp_df = _read_csv_path(COMPARISON_FILE)
        except Exception:
            pass

    for msg in autoload_messages:
        st.caption(msg)

    # SIOP banner
    st.markdown(
        """
        <div class="cx-warn">
          <b>📢 Note.</b> Chemelex SIOP old forecast is not directly comparable —
          it is category-level and COGS-based, while this corrected model is
          zone × plant × category and revenue-based. IBP covers ~10% of US revenue
          (Jan–May 2025 observed; 2024 and Jun 2025–Jun 2026 use a unit-proxy).
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =======================================================================
    # GLOBAL FILTERS
    # =======================================================================
    with st.container(border=True):
        section_title("🔎 Filters")

        # Build options
        years = sorted([int(y) for y in df["year"].dropna().unique() if y > 0])
        zone_opts = [z for z in OFFICIAL_CODP_ZONES
                     if z in set(df["codp_zone"].dropna().unique())] or OFFICIAL_CODP_ZONES
        plant_opts = sorted([p for p in df["plant"].dropna().unique()
                             if p not in ("All",)])
        cat_opts = sorted([c for c in df["product_category_1"].dropna().unique()
                           if c not in ("All",)])

        r1 = st.columns([1, 2.2, 1.4])
        with r1[0]:
            year_sel = st.selectbox(
                "Year",
                options=years,
                index=years.index(DEFAULT_YEAR) if DEFAULT_YEAR in years
                      else (len(years) - 1 if years else 0),
            ) if years else None
        with r1[1]:
            month_names_sel = st.multiselect(
                "Month",
                options=MONTH_NAMES,
                default=MONTH_NAMES,
                help="Leave all selected for full year",
            )
            months_sel = [i + 1 for i, n in enumerate(MONTH_NAMES) if n in month_names_sel]
        with r1[2]:
            level_sel = st.selectbox(
                "Forecast Level",
                options=FORECAST_LEVELS,
                index=FORECAST_LEVELS.index(DEFAULT_LEVEL),
            )

        r2 = st.columns([2, 2, 2])
        with r2[0]:
            zone_sel = st.multiselect("Supply Chain Zone / CODP",
                                       options=zone_opts, default=zone_opts)
        with r2[1]:
            plant_sel = st.multiselect("Plant", options=plant_opts, default=plant_opts)
        with r2[2]:
            cat_sel = st.multiselect("Product Category", options=cat_opts, default=cat_opts)

        r3 = st.columns([5, 1])
        with r3[0]:
            search_text = st.text_input("Search line items", value="",
                                         placeholder="Type to filter by dimension value…")

        # Advanced
        with st.expander("⚙️ Advanced Settings (target / split / model)"):
            ar = st.columns(4)
            with ar[0]:
                if "target" in df.columns:
                    targets = sorted(df["target"].dropna().unique().tolist())
                    target_sel = st.selectbox(
                        "Target / Metric",
                        options=targets,
                        index=targets.index(DEFAULT_TARGET) if DEFAULT_TARGET in targets else 0,
                    )
                else:
                    target_sel = None
                    st.caption("No `target` column")
            with ar[1]:
                if "split" in df.columns:
                    splits = sorted(df["split"].dropna().unique().tolist())
                    split_sel = st.selectbox(
                        "Split",
                        options=splits,
                        index=splits.index(DEFAULT_SPLIT) if DEFAULT_SPLIT in splits else 0,
                    )
                else:
                    split_sel = None
                    st.caption("No `split` column")
            with ar[2]:
                best_only = st.checkbox("Best model only (is_best = 1)", value=True)
            with ar[3]:
                if "model" in df.columns:
                    model_opts = ["(any)"] + sorted(
                        [m for m in df["model"].dropna().unique() if m != "IBP_Manual"]
                    )
                    m_pick = st.selectbox("Model", options=model_opts, index=0)
                    model_sel = None if m_pick == "(any)" else m_pick
                else:
                    model_sel = None
                    st.caption("No `model` column")

    # Apply filters globally — exclude IBP_Manual rows from main views; IBP tab handles it
    filtered = apply_filters(
        df[df.get("model", pd.Series(dtype=object)) != "IBP_Manual"],
        year=year_sel, months=months_sel, codp_zones=zone_sel,
        plants=plant_sel, categories=cat_sel, level=level_sel,
        target=target_sel, split=split_sel, best_only=best_only,
        model=model_sel, search=search_text,
    )

    # =======================================================================
    # INSIGHTS + KPIs
    # =======================================================================
    section_title("Highlights")
    insight_cols = st.columns(3)
    with insight_cols[0]:
        st.markdown(
            """
            <div class="cx-insight cx-ins-green">
              <div class="cx-ins-title">✅ Pull / Kanban Fixed</div>
              <div class="cx-ins-line">
                <span class="cx-ins-strong">$22.3M</span> restored •
                <span class="cx-ins-strong">9.0%</span> WAPE •
                <span class="cx-ins-strong">94.0%</span> business accuracy
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with insight_cols[1]:
        st.markdown(
            """
            <div class="cx-insight cx-ins-blue">
              <div class="cx-ins-title">📈 Pull / FG Restored</div>
              <div class="cx-ins-line">
                <span class="cx-ins-strong">$474M</span> restored •
                <span class="cx-ins-strong">47.6%</span> of demand •
                largest CODP zone
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with insight_cols[2]:
        # Pull live numbers from comparison file if available
        ibp_wape_txt = "—"
        model_wape_txt = "—"
        winner_txt = "—"
        if comp_df is not None and not comp_df.empty:
            try:
                m_w = (comp_df["model_abs_error"].sum()
                       / comp_df["model_actual_usd"].sum() * 100)
                i_w = (comp_df["ibp_abs_error_cogs"].sum()
                       / comp_df["ibp_actual_cogs"].sum() * 100)
                ibp_wape_txt = f"{i_w:.1f}%"
                model_wape_txt = f"{m_w:.1f}%"
                winner_txt = "Model" if m_w < i_w else "IBP"
            except Exception:
                pass
        st.markdown(
            f"""
            <div class="cx-insight cx-ins-amber">
              <div class="cx-ins-title">🏁 IBP vs Model (Jan–May 2025)</div>
              <div class="cx-ins-line">
                Model WAPE: <span class="cx-ins-strong">{model_wape_txt}</span> •
                IBP WAPE: <span class="cx-ins-strong">{ibp_wape_txt}</span>
              </div>
              <div class="cx-ins-line">Winner: <span class="cx-ins-strong">{winner_txt}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_title("KPIs")
    kpis = compute_kpis(filtered)
    k1 = st.columns(4)
    with k1[0]:
        kpi_card("Total Forecast", fmt_money(kpis["forecast"]))
    with k1[1]:
        gap_class = "cx-sub-pos" if (not pd.isna(kpis["gap"]) and kpis["gap"] >= 0) else "cx-sub-neg"
        kpi_card("Total Actual", fmt_money(kpis["actual"]),
                 f"Gap: {fmt_money(kpis['gap'])}", gap_class)
    with k1[2]:
        align_class = ("cx-sub-pos"
                       if (not pd.isna(kpis["alignment"]) and 90 <= kpis["alignment"] <= 110)
                       else "cx-sub-neg")
        kpi_card("Forecast Alignment", fmt_pct(kpis["alignment"]),
                 "Actual ÷ Forecast", align_class)
    with k1[3]:
        wape_class = ("cx-sub-pos"
                      if (not pd.isna(kpis["wape"]) and kpis["wape"] < 20)
                      else "cx-sub-neg")
        kpi_card("WAPE", fmt_pct(kpis["wape"]), "weighted error", wape_class)

    k2 = st.columns(4)
    with k2[0]:
        bias_class = ("cx-sub-pos"
                      if (not pd.isna(kpis["bias"]) and abs(kpis["bias"]) < 10)
                      else "cx-sub-warn")
        kpi_card("Bias", fmt_pct(kpis["bias"]), "(F − A) ÷ A", bias_class)
    with k2[1]:
        kpi_card("Gap", fmt_money(kpis["gap"]))
    with k2[2]:
        kpi_card("Active Line Items", fmt_int(kpis["active_items"]))
    with k2[3]:
        risk_class = "cx-sub-pos" if kpis["at_risk_items"] == 0 else "cx-sub-warn"
        kpi_card("At-risk Line Items", fmt_int(kpis["at_risk_items"]),
                 "alignment outside 80–120%", risk_class)

    # =======================================================================
    # TABS
    # =======================================================================
    tabs = st.tabs([
        "📈 Overview",
        "🎯 Forecast Accuracy",
        "🏁 IBP vs Model",
        "🔍 Level Drilldown",
        "🧮 Model View",
        "🧹 Data Quality",
    ])

    # ---- OVERVIEW ----------------------------------------------------------
    with tabs[0]:
        oc1, oc2 = st.columns(2)
        with oc1:
            section_title("Monthly Forecast vs Actual")
            st.plotly_chart(chart_monthly_fa(filtered), use_container_width=True)
        with oc2:
            section_title("Forecast Alignment by Month")
            st.plotly_chart(chart_alignment_by_month(filtered), use_container_width=True)

        oc3, oc4 = st.columns(2)
        with oc3:
            section_title("Forecast vs Actual by Category")
            st.plotly_chart(chart_category(filtered), use_container_width=True)
        with oc4:
            section_title("WAPE by Supply Chain Zone")
            st.plotly_chart(chart_wape_by_zone(filtered), use_container_width=True)

    # ---- FORECAST ACCURACY -------------------------------------------------
    with tabs[1]:
        section_title("Business Table")
        st.caption(
            f"Forecast Level: **{level_sel}**  •  "
            f"Forecast Alignment = Actual ÷ Forecast × 100"
        )
        table_df = build_business_table(filtered, level_sel)
        if table_df.empty:
            st.info("No data after filtering.")
        else:
            st.dataframe(
                format_business_table(table_df, currency_compact=False),
                use_container_width=True, hide_index=True, height=520,
            )
            csv_bytes = table_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download as CSV",
                data=csv_bytes,
                file_name=f"chemelex_forecast_{level_sel.replace(' ', '_').lower()}.csv",
                mime="text/csv",
            )

    # ---- IBP vs MODEL ------------------------------------------------------
    with tabs[2]:
        render_ibp_tab(df, comp_df,
                       year_sel=year_sel,
                       months_sel=months_sel,
                       zone_sel=zone_sel,
                       plant_sel=plant_sel,
                       cat_sel=cat_sel,
                       level_sel=level_sel,
                       target_sel=target_sel,
                       split_sel=split_sel)

    # ---- LEVEL DRILLDOWN ---------------------------------------------------
    with tabs[3]:
        section_title("Drill by Dimension")
        drill_choices = [c for c in ["codp_zone", "plant", "product_category_1"]
                         if c in filtered.columns]
        if not drill_choices:
            st.info("No drill dimensions available.")
        else:
            d_label = {"codp_zone": "CODP Zone", "plant": "Plant",
                       "product_category_1": "Product Category"}
            dc1, dc2 = st.columns([1.5, 4])
            with dc1:
                drill_dim = st.selectbox(
                    "Drill by",
                    options=drill_choices,
                    format_func=lambda x: d_label.get(x, x),
                )
                top_n = st.slider("Top N", min_value=5, max_value=30, value=10)

            if not filtered.empty and drill_dim:
                agg_dict = dict(Forecast=("forecast", "sum"),
                                 Actual=("actual", "sum"))
                if "abs_error" in filtered.columns:
                    agg_dict["AbsError"] = ("abs_error", "sum")
                agg = (filtered.groupby(drill_dim)
                               .agg(**agg_dict)
                               .reset_index())
                agg["Alignment %"] = np.where(
                    agg["Forecast"].abs() > 1e-6, agg["Actual"] / agg["Forecast"] * 100, np.nan
                )
                if "AbsError" in agg.columns:
                    agg["WAPE %"] = np.where(agg["Actual"].abs() > 1e-6,
                                              agg["AbsError"] / agg["Actual"] * 100, np.nan)
                agg = agg.sort_values("Actual", ascending=False).head(top_n)

                c_t = get_theme()
                fig = go.Figure()
                fig.add_trace(go.Bar(name="Forecast", x=agg[drill_dim], y=agg["Forecast"],
                                    marker_color=c_t["model"]))
                fig.add_trace(go.Bar(name="Actual", x=agg[drill_dim], y=agg["Actual"],
                                    marker_color=c_t["actual"]))
                fig.update_layout(
                    template=c_t["plotly_tmpl"], barmode="group",
                    height=380, margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
                )
                fig.update_yaxes(tickprefix="$", separatethousands=True)
                fig.update_xaxes(tickangle=-25)
                with dc2:
                    st.plotly_chart(fig, use_container_width=True)

                display = agg.copy()
                display["Forecast"] = display["Forecast"].apply(fmt_money)
                display["Actual"] = display["Actual"].apply(fmt_money)
                display["Alignment %"] = display["Alignment %"].apply(fmt_pct)
                if "WAPE %" in display.columns:
                    display["WAPE %"] = display["WAPE %"].apply(fmt_pct)
                    display = display.drop(columns=["AbsError"], errors="ignore")
                display = display.rename(columns={drill_dim: d_label.get(drill_dim, drill_dim)})
                st.dataframe(display, use_container_width=True, hide_index=True)

    # ---- MODEL VIEW --------------------------------------------------------
    with tabs[4]:
        section_title("Model Leaderboard")
        if "model" in df.columns and {"actual", "forecast"}.issubset(df.columns):
            mdf = df[
                (df["model"] != "IBP_Manual")
                & (df.get("year", -1) == year_sel)
                & (df.get("split", DEFAULT_SPLIT) == (split_sel or DEFAULT_SPLIT))
                & (df.get("target", DEFAULT_TARGET) == (target_sel or DEFAULT_TARGET))
                & (df.get("level", DEFAULT_LEVEL) == level_sel)
            ].copy()
            if mdf.empty:
                st.info("No model rows for current filters.")
            else:
                if "abs_error" not in mdf.columns:
                    mdf["abs_error"] = (mdf["actual"] - mdf["forecast"]).abs()
                leaderboard = (mdf.groupby("model")
                                 .apply(lambda g: pd.Series({
                                     "Forecast": g["forecast"].sum(),
                                     "Actual": g["actual"].sum(),
                                     "WAPE %": (g["abs_error"].sum() / g["actual"].sum() * 100)
                                                if g["actual"].sum() else np.nan,
                                     "Bias %": ((g["forecast"].sum() - g["actual"].sum())
                                                / g["actual"].sum() * 100)
                                                if g["actual"].sum() else np.nan,
                                     "Rows": len(g),
                                 }))
                                 .reset_index()
                                 .sort_values("WAPE %"))
                # Pretty
                lb = leaderboard.copy()
                lb["Forecast"] = lb["Forecast"].apply(fmt_money)
                lb["Actual"] = lb["Actual"].apply(fmt_money)
                lb["WAPE %"] = lb["WAPE %"].apply(fmt_pct)
                lb["Bias %"] = lb["Bias %"].apply(fmt_pct)
                lb["Rows"] = lb["Rows"].apply(fmt_int)
                st.dataframe(lb, use_container_width=True, hide_index=True)

                section_title("WAPE by Model")
                c_t = get_theme()
                fig = go.Figure(go.Bar(
                    x=leaderboard["model"],
                    y=leaderboard["WAPE %"],
                    marker_color=c_t["primary"],
                    text=[f"{v:.1f}%" for v in leaderboard["WAPE %"]],
                    textposition="outside",
                ))
                fig.update_layout(
                    template=c_t["plotly_tmpl"], height=360,
                    margin=dict(l=10, r=10, t=10, b=10),
                    yaxis=dict(ticksuffix="%"),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model data available.")

    # ---- DATA QUALITY ------------------------------------------------------
    with tabs[5]:
        section_title("Data Quality Overview")
        dq1, dq2, dq3, dq4 = st.columns(4)
        with dq1:
            kpi_card("Rows in view", fmt_int(len(filtered)))
        with dq2:
            ma = int(filtered["actual"].isna().sum()) if "actual" in filtered else 0
            kpi_card("Missing actuals", fmt_int(ma),
                     "rows with no actual",
                     "cx-sub-pos" if ma == 0 else "cx-sub-warn")
        with dq3:
            mf = int(filtered["forecast"].isna().sum()) if "forecast" in filtered else 0
            kpi_card("Missing forecasts", fmt_int(mf),
                     "rows with no forecast",
                     "cx-sub-pos" if mf == 0 else "cx-sub-warn")
        with dq4:
            fwd = int(filtered["is_forward"].sum()) if "is_forward" in filtered else 0
            kpi_card("Forward-looking rows", fmt_int(fwd),
                     "no actual yet (by design)", "cx-sub-info")

        if "data_quality_flags" in filtered.columns:
            section_title("Quality Flags")
            flag_counts = (filtered["data_quality_flags"].fillna("(clean)")
                           .value_counts().reset_index())
            flag_counts.columns = ["Flag", "Count"]
            cq1, cq2 = st.columns([2, 3])
            with cq1:
                st.dataframe(flag_counts, use_container_width=True, hide_index=True)
            with cq2:
                c_t = get_theme()
                fig = go.Figure(go.Bar(
                    x=flag_counts["Flag"], y=flag_counts["Count"],
                    marker_color=c_t["primary"],
                ))
                fig.update_layout(template=c_t["plotly_tmpl"], height=300,
                                   margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                """
                **Flag meanings** *(from `ibp_pipeline.md`)*
                - `(clean)` — no known data-quality issue
                - `ibp_unit_proxy_cogs` — IBP value derived from raw units × average COGS factor
                  ($3.366/unit); not a direct COGS read
                - `forward_looking_ibp` — Month is after May 2025; `ibp_actual_cogs` is NaN by design
                """
            )

        with st.expander("🔬 Show raw filtered rows (for QA)"):
            st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)
            st.caption(f"Showing first 500 of {len(filtered):,} filtered rows.")

    # ---- FOOTER ------------------------------------------------------------
    c_t = get_theme()
    st.markdown(
        f"""
        <div style="margin-top:30px; padding-top:14px; border-top:1px solid {c_t['border']};
                    color:{c_t['text_muted']}; font-size:12px; text-align:center;">
            Chemelex Demand Forecasting Cockpit  •  
            Forecast Alignment = Actual ÷ Forecast × 100  •  
            Theme: {st.session_state.theme}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# IBP vs MODEL TAB — separated for clarity
# ===========================================================================
def render_ibp_tab(df: pd.DataFrame, comp_df: Optional[pd.DataFrame], *,
                   year_sel, months_sel, zone_sel, plant_sel, cat_sel,
                   level_sel, target_sel, split_sel):
    """The IBP-vs-Model comparison tab. Uses both the levelled file
    (for IBP scope panel + history) and the Synthefy comparison CSV
    (for the authoritative head-to-head)."""

    section_title("IBP Manual Forecast vs ML Model")

    # Show pipeline provenance / known caveats
    with st.expander("📖 Pipeline & data provenance — read me", expanded=False):
        st.markdown(
            """
            **What this comparison is** — Chemelex's IBP (Integrated Business Planning)
            consensus forecast vs the ML model forecasts, scoped to the slice IBP plans.

            **Key facts** *(from `ibp_pipeline.md`)*

            - **IBP is COGS-based**, models are **revenue-based**. Direct dollar comparison
              is not valid; only **accuracy percentages** are comparable.
            - **IBP covers 6 of 13 product categories** (Heat Tracing Components, Floor Heating,
              Control/Monitoring/Power Distribution, Polymer Pipe Heat Tracing BIS & IND,
              Snow Melting & De-Icing). The other 7 have no IBP forecast.
            - **Jan–May 2025** is the only window with both IBP forecast and IBP actual in
              COGS USD (sourced from the FC Error sheet). Backtest accuracy is reliable here.
            - **2024** IBP values are a directional proxy (units × $3.366/unit average factor);
              no IBP actuals exist for 2024.
            - **Jun 2025 – Jun 2026** are forward-looking; IBP forecast uses the same unit
              proxy and there are no actuals yet (`is_forward = 1`).
            - **NAM, PRD scale, Final Demand Consensus** is the IBP filter scope.
            """
        )

    # Preconditions
    needed = {"model", "ibp_forecast_cogs", "ibp_actual_cogs"}
    if not needed.issubset(df.columns):
        st.warning("This view requires IBP columns in the data file.")
        return
    if "IBP_Manual" not in set(df["model"].dropna().unique()):
        st.warning("`IBP_Manual` rows not found in the loaded file.")
        return

    # ----- Local controls ---------------------------------------------------
    ic1, ic2, ic3 = st.columns([1, 2, 1.5])
    with ic1:
        ibp_years = sorted([int(y) for y in df["year"].dropna().unique() if y > 0])
        ibp_year_default = 2025 if 2025 in ibp_years else (year_sel or ibp_years[-1])
        ibp_year = st.selectbox(
            "Year",
            options=ibp_years,
            index=ibp_years.index(ibp_year_default),
            key="ibp_year",
        )
    with ic2:
        ml_choices = sorted([m for m in df["model"].dropna().unique() if m != "IBP_Manual"])
        default_ml = "XGBoost" if "XGBoost" in ml_choices else ml_choices[0]
        ml_model = st.selectbox(
            "ML model to compare",
            options=ml_choices,
            index=ml_choices.index(default_ml),
            key="ibp_ml_model",
        )
    with ic3:
        ibp_target = st.selectbox(
            "Target / Metric",
            options=sorted(df["target"].dropna().unique()),
            index=0,
            key="ibp_target",
        )

    # ----- Build IBP scope panel & ML rescaled panel -------------------------
    base = df[
        (df["year"] == ibp_year)
        & (df.get("split", DEFAULT_SPLIT) == "split2")
        & (df["target"] == ibp_target)
        & (df["level"] == level_sel)
    ].copy()
    # Honour main filters at the same level
    if "codp_zone" in base.columns and zone_sel:
        base = base[base["codp_zone"].isin(zone_sel + ["All"])]
    if "plant" in base.columns and plant_sel:
        base = base[base["plant"].isin(plant_sel + ["All"])]
    if "product_category_1" in base.columns and cat_sel:
        base = base[base["product_category_1"].isin(cat_sel + ["All"])]

    ibp_rows = base[base["model"] == "IBP_Manual"].copy()
    ml_rows = base[base["model"] == ml_model].copy()

    if ibp_rows.empty:
        st.info(f"No IBP rows for {ibp_year} at {level_sel} level with current filters.")
        return

    # --- Full-year IBP series (for the IBP-scope panel) ---
    ibp_monthly = (ibp_rows.groupby("date", as_index=False)
                   .agg(ibp_forecast=("ibp_forecast_cogs", "sum"),
                        ibp_actual=("ibp_actual_cogs", "sum"))
                   .sort_values("date"))

    all_dates = pd.to_datetime([f"{ibp_year}-{m:02d}-01" for m in range(1, 13)])
    ibp_monthly = (
        ibp_monthly.set_index(pd.to_datetime(ibp_monthly["date"]))
        .reindex(all_dates)
        .drop(columns=["date"], errors="ignore")
        .reset_index().rename(columns={"index": "date"})
    )

    # --- Full-year ML series (full scope, full year) ---
    ml_monthly = (ml_rows.groupby("date", as_index=False)
                  .agg(ml_forecast=("forecast", "sum"),
                       ml_actual=("actual", "sum"))
                  .sort_values("date"))
    ml_monthly = (
        ml_monthly.set_index(pd.to_datetime(ml_monthly["date"]))
        .reindex(all_dates)
        .drop(columns=["date"], errors="ignore")
        .reset_index().rename(columns={"index": "date"})
    )

    # --- ML rescored on IBP scope (CORRECT: use IBP_Manual rows as authoritative
    #     IBP-scope actuals, merge onto ML rows by date+dimension, then rescale).
    #     Note: include rows where ibp_actual_cogs is non-null (incl. negatives
    #     from returns) so the hierarchy reconciles exactly. ---
    ibp_auth = (
        ibp_rows[ibp_rows["ibp_actual_cogs"].notna()]
        .set_index(["date", "dimension_value"])
        [["ibp_actual_cogs"]]
        .rename(columns={"ibp_actual_cogs": "ibp_actual_auth"})
    )
    if not ibp_auth.empty and not ml_rows.empty:
        ml_joined = (
            ml_rows.set_index(["date", "dimension_value"])
            .join(ibp_auth, how="left")
            .reset_index()
        )
        ml_on_ibp = ml_joined[ml_joined["ibp_actual_auth"].notna()].copy()
        if not ml_on_ibp.empty:
            # Rescale: ml_forecast_in_ibp_scope = ml_forecast × (ibp_actual / ml_full_actual)
            # Cap ratio to avoid extreme distortion when ml actual is near 0
            ratio_raw = np.where(
                ml_on_ibp["actual"].abs() > 1e-6,
                ml_on_ibp["ibp_actual_auth"] / ml_on_ibp["actual"],
                np.nan,
            )
            # Cap at 2.0 to prevent absurd extrapolations on near-zero denominators
            ratio = np.clip(ratio_raw, 0, 2.0)
            ml_on_ibp["ml_forecast_ibpscope"] = ml_on_ibp["forecast"] * ratio
            ml_scope_f = float(ml_on_ibp["ml_forecast_ibpscope"].sum())
            ml_scope_a = float(ml_on_ibp["ibp_actual_auth"].sum())
            ml_scope_ae = float(
                (ml_on_ibp["ibp_actual_auth"] - ml_on_ibp["ml_forecast_ibpscope"]).abs().sum()
            )
        else:
            ml_scope_f = ml_scope_a = ml_scope_ae = float("nan")
    else:
        ml_on_ibp = pd.DataFrame()
        ml_scope_f = ml_scope_a = ml_scope_ae = float("nan")

    ibp_has = ibp_monthly["ibp_actual"].notna()
    sum_ibp_f = float(ibp_monthly.loc[ibp_has, "ibp_forecast"].sum())
    sum_ibp_a = float(ibp_monthly.loc[ibp_has, "ibp_actual"].sum())
    sum_ibp_ae = float(
        (ibp_monthly.loc[ibp_has, "ibp_actual"]
         - ibp_monthly.loc[ibp_has, "ibp_forecast"]).abs().sum()
    )

    ibp_wape = (sum_ibp_ae / sum_ibp_a * 100) if sum_ibp_a else np.nan
    ibp_align = (sum_ibp_a / sum_ibp_f * 100) if sum_ibp_f else np.nan
    ml_wape = (ml_scope_ae / ml_scope_a * 100) if ml_scope_a else np.nan

    # ----- KPI row ---------------------------------------------------------
    section_title("Head-to-Head KPIs (IBP scope)")
    kp = st.columns(4)
    with kp[0]:
        kpi_card("IBP WAPE", fmt_pct(ibp_wape),
                 "IBP scope • observed months",
                 "cx-sub-pos" if (not pd.isna(ibp_wape) and ibp_wape < 20) else "cx-sub-warn")
    with kp[1]:
        better = (not pd.isna(ml_wape) and not pd.isna(ibp_wape) and ml_wape < ibp_wape)
        kpi_card(f"{ml_model} WAPE", fmt_pct(ml_wape),
                 "rescaled to IBP scope",
                 "cx-sub-pos" if better else "cx-sub-warn")
    with kp[2]:
        kpi_card("IBP Alignment", fmt_pct(ibp_align),
                 "Actual ÷ Forecast",
                 "cx-sub-pos" if (not pd.isna(ibp_align) and 90 <= ibp_align <= 110)
                  else "cx-sub-warn")
    with kp[3]:
        # Use Synthefy's authoritative comparison CSV if available
        if comp_df is not None and not comp_df.empty:
            try:
                m_w_global = (comp_df["model_abs_error"].sum()
                              / comp_df["model_actual_usd"].sum() * 100)
                i_w_global = (comp_df["ibp_abs_error_cogs"].sum()
                              / comp_df["ibp_actual_cogs"].sum() * 100)
                wins_model = int(comp_df["model_more_accurate"].sum())
                total_compared = len(comp_df)
                wins_ibp = total_compared - wins_model
                kpi_card("Category-Month Wins",
                         f"ML {wins_model} – {wins_ibp} IBP",
                         f"{wins_model/total_compared*100:.0f}% ML wins (Synthefy)",
                         "cx-sub-pos" if wins_model > wins_ibp else "cx-sub-warn")
            except Exception:
                kpi_card("Category-Month Wins", "—", "Synthefy CSV unavailable",
                         "cx-sub-info")
        else:
            kpi_card("Category-Month Wins", "—",
                     "Load ibp_vs_model_comparison_2025.csv",
                     "cx-sub-info")

    # ----- Two scoped panels ------------------------------------------------
    section_title("Forecast vs Actual — by scope")
    p1, p2 = st.columns(2)
    c_t = get_theme()

    # ---- Panel 1: IBP scope, full year with greyed band ----
    with p1:
        st.markdown(
            f"**IBP scope** (~10% of US revenue) — IBP COGS forecast vs IBP-covered actual"
        )
        fig = go.Figure()
        # Grey vrect for months without IBP actual
        missing = ibp_monthly[ibp_monthly["ibp_actual"].isna() |
                              (ibp_monthly["ibp_actual"] <= 0)]
        if len(missing):
            fig.add_vrect(
                x0=missing["date"].min(), x1=missing["date"].max(),
                fillcolor=c_t["text_muted"], opacity=0.10,
                layer="below", line_width=0,
                annotation_text="No IBP actual / unit proxy",
                annotation_position="top left",
                annotation_font_color=c_t["text_muted"],
            )
        fig.add_trace(go.Bar(
            x=ibp_monthly["date"], y=ibp_monthly["ibp_forecast"],
            name="IBP Forecast (COGS)", marker_color=c_t["ibp"], opacity=0.95,
        ))
        fig.add_trace(go.Bar(
            x=ibp_monthly["date"], y=ibp_monthly["ibp_actual"],
            name="IBP-scope Actual", marker_color=c_t["actual"], opacity=0.95,
        ))
        fig.update_layout(
            barmode="group", template=c_t["plotly_tmpl"], height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        )
        fig.update_xaxes(tickformat="%b %Y")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Panel 2: Full US scope, full year ----
    with p2:
        st.markdown(
            f"**Full US scope** — {ml_model} revenue forecast vs total US actual"
        )
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=ml_monthly["date"], y=ml_monthly["ml_forecast"],
            name=f"{ml_model} Forecast", marker_color=c_t["model"], opacity=0.95,
        ))
        fig2.add_trace(go.Bar(
            x=ml_monthly["date"], y=ml_monthly["ml_actual"],
            name="Full-scope Actual", marker_color=c_t["actual"], opacity=0.95,
        ))
        fig2.update_layout(
            barmode="group", template=c_t["plotly_tmpl"], height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        )
        fig2.update_xaxes(tickformat="%b %Y")
        st.plotly_chart(fig2, use_container_width=True)

    # ----- Apples-to-apples on IBP scope ---------------------------------
    section_title("Apples-to-Apples — Forecast lines on IBP scope")
    apples = ibp_monthly[["date", "ibp_forecast", "ibp_actual"]].copy()
    if not ml_on_ibp.empty and "ml_forecast_ibpscope" in ml_on_ibp.columns:
        ml_scope = (ml_on_ibp.groupby("date", as_index=False)["ml_forecast_ibpscope"].sum())
        ml_scope["date"] = pd.to_datetime(ml_scope["date"])
        apples = apples.merge(ml_scope, on="date", how="left")
    else:
        apples["ml_forecast_ibpscope"] = np.nan

    fig3 = go.Figure()
    missing2 = apples[apples["ibp_actual"].isna() | (apples["ibp_actual"] <= 0)]
    if len(missing2):
        fig3.add_vrect(
            x0=missing2["date"].min(), x1=missing2["date"].max(),
            fillcolor=c_t["text_muted"], opacity=0.10,
            layer="below", line_width=0,
            annotation_text="No IBP actual",
            annotation_position="top left",
            annotation_font_color=c_t["text_muted"],
        )
    fig3.add_trace(go.Scatter(
        x=apples["date"], y=apples["ibp_actual"],
        mode="lines+markers", name="Actual (IBP scope)",
        line=dict(color=c_t["actual"], width=3),
    ))
    fig3.add_trace(go.Scatter(
        x=apples["date"], y=apples["ibp_forecast"],
        mode="lines+markers", name="IBP Forecast",
        line=dict(color=c_t["ibp"], width=3),
    ))
    fig3.add_trace(go.Scatter(
        x=apples["date"], y=apples["ml_forecast_ibpscope"],
        mode="lines+markers", name=f"{ml_model} Forecast (rescaled)",
        line=dict(color=c_t["model"], width=3, dash="dash"),
    ))
    fig3.update_layout(
        template=c_t["plotly_tmpl"], height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig3.update_xaxes(tickformat="%b %Y")
    st.plotly_chart(fig3, use_container_width=True)

    # ----- Synthefy authoritative comparison ----------------------------
    if comp_df is not None and not comp_df.empty:
        section_title("Synthefy Authoritative Comparison — Jan–May 2025 by Category × Month")
        st.caption(
            "Source: `ibp_vs_model_comparison_2025.csv`. "
            "This is Synthefy's pre-computed head-to-head at category × month, "
            "where both forecasts are scored against the IBP-COGS actual."
        )

        # By category summary
        cat_summary = []
        for cat in comp_df["product_category_1"].unique():
            sub = comp_df[comp_df["product_category_1"] == cat]
            m_w = sub["model_abs_error"].sum() / sub["model_actual_usd"].sum() * 100
            i_w = sub["ibp_abs_error_cogs"].sum() / sub["ibp_actual_cogs"].sum() * 100
            wins_m = int(sub["model_more_accurate"].sum())
            wins_i = len(sub) - wins_m
            cat_summary.append(dict(
                Category=cat,
                **{"Model WAPE %": m_w, "IBP WAPE %": i_w,
                   "Model wins": wins_m, "IBP wins": wins_i,
                   "Winner": ("Model 🏆" if m_w < i_w else "IBP 🏆")}
            ))
        cat_summary_df = pd.DataFrame(cat_summary).sort_values("Model WAPE %")

        # Chart: side-by-side WAPE bars per category
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=cat_summary_df["Category"], y=cat_summary_df["Model WAPE %"],
            name=f"{ml_model} WAPE",
            marker_color=c_t["model"],
            text=[f"{v:.1f}%" for v in cat_summary_df["Model WAPE %"]],
            textposition="outside",
        ))
        fig4.add_trace(go.Bar(
            x=cat_summary_df["Category"], y=cat_summary_df["IBP WAPE %"],
            name="IBP WAPE",
            marker_color=c_t["ibp"],
            text=[f"{v:.1f}%" for v in cat_summary_df["IBP WAPE %"]],
            textposition="outside",
        ))
        fig4.update_layout(
            template=c_t["plotly_tmpl"], barmode="group",
            height=400, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
            yaxis=dict(ticksuffix="%"),
        )
        fig4.update_xaxes(tickangle=-25)
        st.plotly_chart(fig4, use_container_width=True)

        # Pretty table
        cs = cat_summary_df.copy()
        cs["Model WAPE %"] = cs["Model WAPE %"].apply(fmt_pct)
        cs["IBP WAPE %"] = cs["IBP WAPE %"].apply(fmt_pct)
        st.dataframe(cs, use_container_width=True, hide_index=True)

        # By month
        section_title("Head-to-Head by Month")
        month_summary = []
        for m in sorted(comp_df["month"].unique()):
            sub = comp_df[comp_df["month"] == m]
            m_w = sub["model_abs_error"].sum() / sub["model_actual_usd"].sum() * 100
            i_w = sub["ibp_abs_error_cogs"].sum() / sub["ibp_actual_cogs"].sum() * 100
            wins_m = int(sub["model_more_accurate"].sum())
            wins_i = len(sub) - wins_m
            month_summary.append(dict(
                Month=MONTH_NAMES[int(m) - 1] + " 2025",
                **{"Model WAPE %": fmt_pct(m_w), "IBP WAPE %": fmt_pct(i_w),
                   "Model wins": wins_m, "IBP wins": wins_i,
                   "Winner": ("Model 🏆" if m_w < i_w else "IBP 🏆")}
            ))
        st.dataframe(pd.DataFrame(month_summary), use_container_width=True, hide_index=True)
    else:
        section_title("Synthefy Comparison File Not Loaded")
        st.info(
            f"Place `{COMPARISON_FILE}` next to `app.py` to see the authoritative "
            "category-month head-to-head from Synthefy's pipeline."
        )

    # ----- Reading guide ----------------------------------------------------
    st.markdown(
        """
        <div class="cx-warn">
          <b>Reading this comparison.</b> IBP forecasts are in COGS USD; ML forecasts
          are in revenue USD. Dollar values cannot be compared directly across the two —
          only accuracy percentages (WAPE, Alignment) are valid for head-to-head.
          The Synthefy comparison file is the authoritative reference because both
          sides are scored against the same IBP-COGS actuals.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
