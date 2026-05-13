"""
Chemelex Demand Forecasting Cockpit  —  Executive view.

Plain-English dashboard for senior leadership: forecast accuracy, where we
are off-track, and how AI compares to the manual demand plan.

Data files (auto-loaded if present in the working directory):
    corrected_dashboard_forecast_accuracy_levelled.csv   (required)
    ibp_vs_model_comparison_2025.csv                     (optional)
    ibp_reconciliation.csv                               (optional)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(
    page_title="Chemelex Demand Forecasting Cockpit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PRIMARY_FILE     = "corrected_dashboard_forecast_accuracy_levelled.csv"
COMPARISON_FILE  = "ibp_vs_model_comparison_2025.csv"
RECON_FILE       = "ibp_reconciliation.csv"

OFFICIAL_ZONES = [
    "BUFFER (at CODP)",
    "Pull / FG",
    "Pull / Kanban",
    "Push / MPS",
    "Push / RM",
]

LEVELS       = ["Total", "CODP Zone", "Plant", "Product Category", "Group"]
MONTH_NAMES  = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

TARGET_LABELS = {
    "order_value_usd": "Revenue ($)",
    "order_qty_bu":    "Volume (units)",
}

# =============================================================================
# THEME
# =============================================================================
if "theme" not in st.session_state:
    st.session_state.theme = "Light"


def get_theme() -> dict:
    if st.session_state.theme == "Dark":
        return {
            "bg":            "#0f1419",
            "surface":       "#1a2027",
            "surface_2":     "#222831",
            "border":        "#2d3748",
            "text":          "#e5e7eb",
            "text_muted":    "#94a3b8",
            "primary":       "#3b82f6",
            "accent":        "#60a5fa",
            "success":       "#10b981",
            "warning":       "#f59e0b",
            "danger":        "#ef4444",
            "actual":        "#3b82f6",
            "ai":            "#10b981",
            "manual":        "#f59e0b",
            "plotly":        "plotly_dark",
            "warn_bg":       "#3a2e15",
        }
    return {
        "bg":            "#f8fafc",
        "surface":        "#ffffff",
        "surface_2":      "#f1f5f9",
        "border":         "#e2e8f0",
        "text":           "#0f172a",
        "text_muted":     "#64748b",
        "primary":        "#2563eb",
        "accent":         "#3b82f6",
        "success":        "#059669",
        "warning":        "#d97706",
        "danger":         "#dc2626",
        "actual":         "#2563eb",
        "ai":             "#059669",
        "manual":         "#f59e0b",
        "plotly":         "plotly_white",
        "warn_bg":        "#fef3c7",
    }


def inject_css() -> None:
    c = get_theme()
    css = f"""
    <style>
      .stApp {{ background-color: {c['bg']}; color: {c['text']}; }}
      .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px; }}
      .cx-header {{
        background: linear-gradient(135deg, {c['primary']} 0%, {c['accent']} 100%);
        padding: 18px 24px; border-radius: 12px; margin-bottom: 14px;
        color: white; box-shadow: 0 4px 16px rgba(0,0,0,0.08);
      }}
      .cx-header h1 {{ color: white; margin: 0; font-size: 22px; font-weight: 700; }}
      .cx-header p  {{ color: rgba(255,255,255,0.88); margin: 4px 0 0 0; font-size: 13px; }}
      .cx-card {{
        background: {c['surface']}; border: 1px solid {c['border']};
        border-radius: 10px; padding: 14px 16px; height: 100%;
      }}
      .cx-card-title {{
        color: {c['text_muted']}; font-size: 11px; text-transform: uppercase;
        letter-spacing: 0.04em; margin-bottom: 6px; font-weight: 600;
      }}
      .cx-card-value {{ color: {c['text']}; font-size: 24px; font-weight: 700; }}
      .cx-card-delta {{ font-size: 12px; margin-top: 4px; }}
      .cx-delta-pos     {{ color: {c['success']}; }}
      .cx-delta-neg     {{ color: {c['danger']};  }}
      .cx-delta-neutral {{ color: {c['text_muted']}; }}
      .cx-section {{
        color: {c['text']}; font-size: 14px; font-weight: 700;
        margin: 18px 0 8px 0; padding-bottom: 4px;
        border-bottom: 2px solid {c['border']};
      }}
      .cx-insight {{
        border-radius: 10px; padding: 14px 18px; color: white; height: 100%;
      }}
      .cx-insight-blue   {{ background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%); }}
      .cx-insight-green  {{ background: linear-gradient(135deg, #10b981 0%, #059669 100%); }}
      .cx-insight-orange {{ background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); }}
      .cx-insight h3 {{ color: white; margin: 0 0 6px 0; font-size: 14px; font-weight: 700; }}
      .cx-insight .big {{ font-size: 22px; font-weight: 700; margin: 4px 0; }}
      .cx-insight p {{ color: rgba(255,255,255,0.92); margin: 1px 0; font-size: 12.5px; }}
      .cx-info {{
        background: {c['surface_2']}; border-left: 4px solid {c['primary']};
        color: {c['text']}; padding: 10px 14px; border-radius: 6px;
        font-size: 13px; margin: 8px 0 14px 0;
      }}
      .cx-chart-title {{
        color: {c['text']}; font-size: 14px; font-weight: 600;
        margin: 8px 0 4px 0;
      }}
      .cx-chart-sub {{
        color: {c['text_muted']}; font-size: 12px; margin-bottom: 4px;
      }}
      .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
        font-size: 14px; font-weight: 600;
      }}
      footer  {{ visibility: hidden; }}
      #MainMenu {{ visibility: hidden; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data(show_spinner=False)
def _read_bytes(content: bytes, filename: str) -> pd.DataFrame:
    bio = io.BytesIO(content)
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(bio)
    return pd.read_csv(bio, low_memory=False)


@st.cache_data(show_spinner=False)
def _read_path(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p)
    return pd.read_csv(p, low_memory=False)


def normalize_main(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for c in ("is_best", "is_forward"):
        if c in df.columns:
            df[c] = (df[c].astype(str).str.strip().str.lower()
                     .map({"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
                     .fillna(0).astype(int))

    numeric = ["actual", "forecast", "accuracy_pct", "error", "abs_error",
               "bias_pct", "wape_component_num", "wape_component_den",
               "year", "month", "ibp_forecast_cogs", "ibp_actual_cogs",
               "ibp_accuracy_pct"]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "year" not in df.columns or df["year"].isna().all():
            df["year"] = df["date"].dt.year
        if "month" not in df.columns or df["month"].isna().all():
            df["month"] = df["date"].dt.month

    for c in ("level", "dimension_value", "codp_zone", "plant",
              "product_category_1", "target", "model", "split",
              "data_quality_flags"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].isin(["nan", "NaN", "None", ""]), c] = np.nan
    return df


def try_autoload():
    main_df = comp_df = recon_df = None
    msgs = []
    if Path(PRIMARY_FILE).exists():
        try:
            main_df = normalize_main(_read_path(PRIMARY_FILE))
            msgs.append(f"Loaded `{PRIMARY_FILE}` ({len(main_df):,} rows)")
        except Exception as e:
            msgs.append(f"Could not load `{PRIMARY_FILE}`: {e}")
    if Path(COMPARISON_FILE).exists():
        try:
            comp_df = _read_path(COMPARISON_FILE)
            comp_df.columns = [c.strip() for c in comp_df.columns]
            msgs.append(f"Loaded `{COMPARISON_FILE}` ({len(comp_df):,} rows)")
        except Exception as e:
            msgs.append(f"Could not load `{COMPARISON_FILE}`: {e}")
    if Path(RECON_FILE).exists():
        try:
            recon_df = _read_path(RECON_FILE)
            recon_df.columns = [c.strip() for c in recon_df.columns]
            msgs.append(f"Loaded `{RECON_FILE}` ({len(recon_df):,} rows)")
        except Exception as e:
            msgs.append(f"Could not load `{RECON_FILE}`: {e}")
    return main_df, comp_df, recon_df, msgs


# =============================================================================
# FORMATTING
# =============================================================================
def fmt_money(x, compact: bool = True) -> str:
    if pd.isna(x): return "—"
    a, sign = abs(x), "-" if x < 0 else ""
    if compact:
        if a >= 1e9: return f"{sign}${a/1e9:.2f}B"
        if a >= 1e6: return f"{sign}${a/1e6:.2f}M"
        if a >= 1e3: return f"{sign}${a/1e3:.1f}K"
    return f"{sign}${a:,.0f}"


def fmt_int(x) -> str:
    if pd.isna(x): return "—"
    return f"{int(x):,}"


def fmt_pct(x, decimals: int = 1) -> str:
    if pd.isna(x): return "—"
    return f"{x:.{decimals}f}%"


def fmt_pp(x, decimals: int = 1) -> str:
    if pd.isna(x): return "—"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.{decimals}f} pts"


# =============================================================================
# ACCURACY HELPERS
# =============================================================================
def compute_accuracy(actual_sum: float, abs_err_sum: float) -> float:
    if actual_sum is None or not actual_sum or actual_sum <= 0:
        return np.nan
    err_pct = abs_err_sum / actual_sum * 100
    return max(0.0, 100.0 - err_pct)


def compute_alignment(actual_sum: float, forecast_sum: float) -> float:
    if not forecast_sum or pd.isna(forecast_sum) or forecast_sum == 0:
        return np.nan
    return actual_sum / forecast_sum * 100


# =============================================================================
# KPI CARD
# =============================================================================
def kpi_card(title: str, value: str, sub: str = "",
             sub_class: str = "cx-delta-neutral") -> None:
    sub_html = f'<div class="cx-card-delta {sub_class}">{sub}</div>' if sub else ""
    st.markdown(
        f"""<div class="cx-card">
              <div class="cx-card-title">{title}</div>
              <div class="cx-card-value">{value}</div>
              {sub_html}
            </div>""",
        unsafe_allow_html=True,
    )


# =============================================================================
# CHART HELPERS
# =============================================================================
def _empty_fig(msg: str, height: int = 320) -> go.Figure:
    c = get_theme()
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=14, color=c["text_muted"]))
    fig.update_layout(template=c["plotly"], height=height,
                      margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def _base_layout(fig: go.Figure, height: int = 360, title: str = "") -> go.Figure:
    c = get_theme()
    fig.update_layout(
        template=c["plotly"], height=height,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=title if title else None,
    )
    return fig


def _add_no_data_shading(fig, df, value_col, c):
    if value_col not in df.columns:
        return
    missing = df[df[value_col].isna()]
    if len(missing):
        fig.add_vrect(
            x0=missing["date"].min(), x1=missing["date"].max(),
            fillcolor=c["text_muted"], opacity=0.10,
            layer="below", line_width=0,
            annotation_text="no data yet",
            annotation_position="top left",
            annotation=dict(font=dict(color=c["text_muted"], size=11)))


# =============================================================================
# HEADER & TOP CONTROLS
# =============================================================================
def render_header():
    st.markdown(
        """<div class="cx-header">
             <h1>📊 Chemelex Demand Forecasting Cockpit</h1>
             <p>Forecast accuracy, AI vs manual plan, and where we're off track — US region</p>
           </div>""",
        unsafe_allow_html=True,
    )


def render_top_controls():
    c1, c2 = st.columns([5, 2])
    with c1:
        upload = st.file_uploader(
            "📁 Upload / Replace Forecast File  (csv, xlsx)",
            type=["csv", "xlsx", "xls"],
            label_visibility="visible",
            key="uploader",
        )
    with c2:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        choice = st.radio(
            "Theme",
            options=["Light", "Dark"],
            index=0 if st.session_state.theme == "Light" else 1,
            horizontal=True,
            key="theme_radio",
        )
        if choice != st.session_state.theme:
            st.session_state.theme = choice
            st.rerun()
    if upload is not None:
        return upload.getvalue(), upload.name
    return None, None


# =============================================================================
# FILTERS
# =============================================================================
def apply_filters(df, *, year, months, zones, plants, cats,
                  level, target, split, best_only, model, search_text):
    out = df.copy()
    if year is not None and "year" in out.columns:
        out = out[out["year"] == year]
    if months and "month" in out.columns:
        out = out[out["month"].isin(months)]
    if zones and "codp_zone" in out.columns:
        out = out[out["codp_zone"].isin(zones + ["All"])]
    if plants and "plant" in out.columns:
        out = out[out["plant"].isin(plants + ["All"])]
    if cats and "product_category_1" in out.columns:
        out = out[out["product_category_1"].isin(cats + ["All"])]
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
    if search_text and "dimension_value" in out.columns:
        s = search_text.strip().lower()
        out = out[out["dimension_value"].astype(str).str.lower()
                  .str.contains(s, na=False)]
    return out


# =============================================================================
# CEO METRICS & BEST MODEL
# =============================================================================
def ceo_metrics(df):
    if df.empty:
        return {k: np.nan for k in
                ["actual", "forecast", "accuracy", "alignment", "gap",
                 "active_items", "off_track_items"]}

    total_actual   = df["actual"].sum()   if "actual"   in df.columns else np.nan
    total_forecast = df["forecast"].sum() if "forecast" in df.columns else np.nan
    abs_err = df["abs_error"].sum() if "abs_error" in df.columns else np.nan

    accuracy  = compute_accuracy(total_actual, abs_err) if not pd.isna(abs_err) else np.nan
    alignment = compute_alignment(total_actual, total_forecast)
    gap = (total_actual - total_forecast) if pd.notna(total_actual) and pd.notna(total_forecast) else np.nan

    active = df["dimension_value"].nunique() if "dimension_value" in df.columns else len(df)
    if {"actual", "forecast", "dimension_value"}.issubset(df.columns):
        per = df.groupby("dimension_value", dropna=False).agg(
            a=("actual", "sum"), f=("forecast", "sum"))
        per["align"] = per["a"] / per["f"] * 100
        off = int(((per["align"] < 80) | (per["align"] > 120)).sum())
    else:
        off = np.nan

    return {"actual": total_actual, "forecast": total_forecast,
            "accuracy": accuracy, "alignment": alignment, "gap": gap,
            "active_items": active, "off_track_items": off}


def best_ai_model(df, *, year=2025, target="order_value_usd",
                  level="Total", on_ibp_scope=True):
    pool = df[(df["year"] == year)
              & (df["split"] == "split2")
              & (df["target"] == target)
              & (df["level"] == level)
              & (df["model"] != "IBP_Manual")
              & (df["is_forward"] == 0)]
    if on_ibp_scope:
        pool = pool[pool["ibp_actual_cogs"].notna()]
    if pool.empty:
        return ("—", np.nan)

    results = []
    for m, sub in pool.groupby("model"):
        if on_ibp_scope:
            ratio = np.where(sub["actual"].abs() > 1e-6,
                             sub["ibp_actual_cogs"] / sub["actual"], np.nan)
            forecast_scaled = sub["forecast"] * ratio
            actual_used = sub["ibp_actual_cogs"]
            abs_err = (actual_used - forecast_scaled).abs().sum()
        else:
            actual_used = sub["actual"]
            abs_err = (sub["actual"] - sub["forecast"]).abs().sum()
        acc = compute_accuracy(float(actual_used.sum()), float(abs_err))
        results.append((m, acc))

    results = [(m, a) for m, a in results if not pd.isna(a)]
    if not results:
        return ("—", np.nan)
    results.sort(key=lambda t: t[1], reverse=True)
    return results[0]


def _best_in_filtered(fdf):
    if fdf.empty or "model" not in fdf.columns:
        return ("—", np.nan)
    g = (fdf.groupby("model")
         .agg(a=("actual", "sum"), ae=("abs_error", "sum"))
         .reset_index())
    g = g[g["a"] > 0]
    if g.empty:
        return ("—", np.nan)
    g["acc"] = 100 - (g["ae"] / g["a"] * 100)
    g["acc"] = g["acc"].clip(lower=0, upper=100)
    g = g.sort_values("acc", ascending=False)
    return (g.iloc[0]["model"], g.iloc[0]["acc"])


# =============================================================================
# MAIN
# =============================================================================
def main():
    inject_css()
    render_header()

    upload_bytes, upload_name = render_top_controls()

    main_df = comp_df = recon_df = None
    if upload_bytes is not None:
        try:
            raw = _read_bytes(upload_bytes, upload_name)
            main_df = normalize_main(raw)
            st.success(f"Loaded `{upload_name}` ({len(main_df):,} rows)")
        except Exception as e:
            st.error(f"Could not read uploaded file: {e}")

    auto_main, comp_df, recon_df, msgs = try_autoload()
    if main_df is None:
        main_df = auto_main

    if main_df is None or main_df.empty:
        st.info(f"👋 **Welcome.** Place `{PRIMARY_FILE}` next to `app.py` "
                "or upload a forecast file above to begin.")
        for m in msgs:
            st.caption(m)
        st.stop()

    # ---- Filters ----
    with st.container(border=True):
        st.markdown("**🔎 Filters**")

        years = sorted([int(y) for y in main_df["year"].dropna().unique()])
        default_year = 2025 if 2025 in years else (years[-1] if years else None)

        zone_opts = [z for z in OFFICIAL_ZONES
                     if z in set(main_df["codp_zone"].dropna().unique())] or OFFICIAL_ZONES
        plant_opts = sorted([p for p in main_df["plant"].dropna().unique() if p != "All"])
        cat_opts   = sorted([c for c in main_df["product_category_1"].dropna().unique()
                             if c != "All"])

        r1 = st.columns([1.0, 2.2, 1.4, 1.6])
        with r1[0]:
            year_sel = st.selectbox(
                "Year", options=years,
                index=years.index(default_year) if default_year in years else 0)
        with r1[1]:
            mo_names = st.multiselect("Month", options=MONTH_NAMES,
                                      default=MONTH_NAMES,
                                      help="Leave all selected for full year")
            month_sel = [i + 1 for i, n in enumerate(MONTH_NAMES) if n in mo_names]
        with r1[2]:
            level_sel = st.selectbox("Forecast Level", options=LEVELS,
                                     index=LEVELS.index("Group"))
        with r1[3]:
            search_text = st.text_input("Search line items", value="",
                                        placeholder="Filter by name…")

        r2 = st.columns([2, 2, 2])
        with r2[0]:
            zone_sel = st.multiselect("Supply Chain Zone",
                                      options=zone_opts, default=zone_opts)
        with r2[1]:
            plant_sel = st.multiselect("Plant", options=plant_opts, default=plant_opts)
        with r2[2]:
            cat_sel = st.multiselect("Product Category",
                                     options=cat_opts, default=cat_opts)

        target_sel = "order_value_usd" if "order_value_usd" in (
            main_df.get("target", pd.Series([])).dropna().unique().tolist()) else None
        split_sel = "split2" if "split2" in (
            main_df.get("split", pd.Series([])).dropna().unique().tolist()) else None
        best_only = True
        model_sel = None

        with st.expander("⚙️ Advanced view (for analysts)"):
            adv = st.columns(3)
            with adv[0]:
                if "target" in main_df.columns:
                    targets = sorted(main_df["target"].dropna().unique().tolist())
                    target_sel = st.selectbox(
                        "Measure",
                        options=targets,
                        index=targets.index("order_value_usd")
                              if "order_value_usd" in targets else 0,
                        format_func=lambda t: TARGET_LABELS.get(t, t))
            with adv[1]:
                if "model" in main_df.columns:
                    model_opts = ["(all AI models)"] + sorted(
                        m for m in main_df["model"].dropna().unique() if m != "IBP_Manual")
                    pick = st.selectbox("Filter to one AI model",
                                        options=model_opts, index=0)
                    model_sel = None if pick == "(all AI models)" else pick
            with adv[2]:
                best_only = st.checkbox("Best model per line only", value=True)

    filtered = apply_filters(
        main_df[main_df.get("model", pd.Series(["x"] * len(main_df))) != "IBP_Manual"]
        if "model" in main_df.columns else main_df,
        year=year_sel, months=month_sel, zones=zone_sel, plants=plant_sel,
        cats=cat_sel, level=level_sel, target=target_sel, split=split_sel,
        best_only=best_only, model=model_sel, search_text=search_text,
    )

    tabs = st.tabs([
        "📈 Executive Overview",
        "🆚 AI vs Manual Plan",
        "🎯 Accuracy by Line",
        "🔍 Drill Down",
        "📦 Forward Plan",
    ])

    with tabs[0]:
        render_overview(filtered, year_sel)
    with tabs[1]:
        render_ai_vs_manual(main_df, comp_df, zone_sel, plant_sel, cat_sel, year_sel)
    with tabs[2]:
        render_accuracy_table(filtered, level_sel, year_sel)
    with tabs[3]:
        render_drilldown(filtered)
    with tabs[4]:
        render_forward_plan(main_df, year_sel, zone_sel, plant_sel, cat_sel)


# =============================================================================
# TAB 1 — EXECUTIVE OVERVIEW
# =============================================================================
def render_overview(fdf, year_sel):
    c = get_theme()

    top_contributor = "—"
    top_contributor_pct = np.nan
    if "product_category_1" in fdf.columns:
        cat_sum = (fdf[fdf["product_category_1"] != "All"]
                   .groupby("product_category_1")["actual"].sum()
                   .sort_values(ascending=False))
        if len(cat_sum) and cat_sum.sum() > 0:
            top_contributor = cat_sum.index[0]
            top_contributor_pct = cat_sum.iloc[0] / cat_sum.sum() * 100

    best_zone = "—"
    best_zone_acc = np.nan
    if {"codp_zone", "abs_error", "actual"}.issubset(fdf.columns):
        z = (fdf[fdf["codp_zone"].isin(OFFICIAL_ZONES)]
             .groupby("codp_zone").agg(ae=("abs_error", "sum"), a=("actual", "sum"))
             .reset_index())
        z = z[z["a"] > 0]
        if len(z):
            z["acc"] = (100 - z["ae"] / z["a"] * 100).clip(lower=0)
            z = z.sort_values("acc", ascending=False)
            best_zone = z.iloc[0]["codp_zone"]
            best_zone_acc = z.iloc[0]["acc"]

    ins1, ins2 = st.columns(2)
    with ins1:
        st.markdown(
            f"""<div class="cx-insight cx-insight-blue">
                <h3>📈 Top revenue contributor</h3>
                <div class="big">{top_contributor}</div>
                <p>{fmt_pct(top_contributor_pct)} of business in current view</p>
                </div>""", unsafe_allow_html=True)
    with ins2:
        st.markdown(
            f"""<div class="cx-insight cx-insight-green">
                <h3>✅ Most accurate zone</h3>
                <div class="big">{best_zone}</div>
                <p>{fmt_pct(best_zone_acc)} forecast accuracy</p>
                </div>""", unsafe_allow_html=True)

    m = ceo_metrics(fdf)

    k = st.columns(4)
    with k[0]:
        kpi_card("Total Forecast", fmt_money(m["forecast"]),
                 "What we planned for")
    with k[1]:
        gap = m["gap"]
        cls = "cx-delta-pos" if (not pd.isna(gap) and gap >= 0) else "cx-delta-neg"
        sub = (f"Sold {fmt_money(abs(gap))} more than planned" if pd.notna(gap) and gap > 0
               else f"Sold {fmt_money(abs(gap))} less than planned"
               if pd.notna(gap) else "")
        kpi_card("Total Actual", fmt_money(m["actual"]), sub, cls)
    with k[2]:
        acc = m["accuracy"]
        cls = ("cx-delta-pos" if (not pd.isna(acc) and acc >= 85)
               else "cx-delta-neg")
        kpi_card("Forecast Accuracy", fmt_pct(acc),
                 "Closer to 100% is better", cls)
    with k[3]:
        align = m["alignment"]
        if pd.isna(align):
            sub = ""
            cls = "cx-delta-neutral"
        elif 95 <= align <= 105:
            sub = "On target"
            cls = "cx-delta-pos"
        elif align > 105:
            sub = "Demand stronger than planned"
            cls = "cx-delta-pos"
        else:
            sub = "Demand weaker than planned"
            cls = "cx-delta-neg"
        kpi_card("Plan vs Reality", fmt_pct(align), sub, cls)

    k2 = st.columns(4)
    with k2[0]:
        kpi_card("Active Forecast Lines", fmt_int(m["active_items"]),
                 "Distinct product / plant lines")
    with k2[1]:
        off = m["off_track_items"]
        cls = "cx-delta-pos" if off == 0 else "cx-delta-neg"
        kpi_card("Off-track Lines", fmt_int(off),
                 "More than 20% off plan", cls)
    with k2[2]:
        if "date" in fdf.columns and not fdf.empty:
            mn = pd.to_datetime(fdf["date"]).min()
            mx = pd.to_datetime(fdf["date"]).max()
            period = f"{mn:%b %Y} – {mx:%b %Y}"
        else:
            period = "—"
        kpi_card("Period", period, f"Year: {year_sel}")
    with k2[3]:
        bm, ba = _best_in_filtered(fdf)
        kpi_card("Top AI Model", bm if bm else "—",
                 (f"{fmt_pct(ba)} accuracy" if pd.notna(ba) else "—"),
                 "cx-delta-pos" if (pd.notna(ba) and ba >= 85) else "cx-delta-neutral")

    st.markdown('<div class="cx-section">Forecast vs Reality — by Month</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_monthly_fa(fdf), use_container_width=True)
    with c2:
        st.plotly_chart(chart_accuracy_by_month(fdf), use_container_width=True)

    st.markdown('<div class="cx-section">Where the Business Is</div>',
                unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(chart_category_bars(fdf), use_container_width=True)
    with c4:
        st.plotly_chart(chart_accuracy_by_zone(fdf), use_container_width=True)


def chart_monthly_fa(fdf):
    c = get_theme()
    if fdf.empty or not {"date", "forecast", "actual"}.issubset(fdf.columns):
        return _empty_fig("No data")
    g = (fdf.groupby("date", as_index=False)[["forecast", "actual"]]
         .sum().sort_values("date"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["date"], y=g["forecast"], mode="lines+markers",
                             name="Forecast",
                             line=dict(color=c["manual"], width=3)))
    fig.add_trace(go.Scatter(x=g["date"], y=g["actual"], mode="lines+markers",
                             name="Actual",
                             line=dict(color=c["actual"], width=3)))
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    return _base_layout(fig, height=340)


def chart_accuracy_by_month(fdf):
    c = get_theme()
    if fdf.empty or not {"date", "actual", "forecast"}.issubset(fdf.columns):
        return _empty_fig("No data")
    g = (fdf.groupby("date", as_index=False)
         .agg(a=("actual", "sum"), ae=("abs_error", "sum"))
         .sort_values("date"))
    g = g[g["a"] > 0]
    if g.empty:
        return _empty_fig("No data")
    g["acc"] = (100 - g["ae"] / g["a"] * 100).clip(lower=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=g["date"], y=g["acc"], mode="lines+markers",
        line=dict(color=c["primary"], width=3),
        marker=dict(size=9),
        fill="tozeroy", fillcolor=f"rgba(37, 99, 235, 0.08)",
        name="Accuracy %",
    ))
    fig.add_hline(y=100, line_dash="dash", line_color=c["success"],
                  annotation_text="100% (perfect)",
                  annotation_position="top right")
    fig.update_yaxes(ticksuffix="%", range=[0, 110])
    fig.update_layout(showlegend=False, title="Forecast Accuracy by Month")
    return _base_layout(fig, height=340, title="")


def chart_category_bars(fdf):
    c = get_theme()
    if fdf.empty or "product_category_1" not in fdf.columns:
        return _empty_fig("No category data")
    g = (fdf[fdf["product_category_1"] != "All"]
         .groupby("product_category_1")
         .agg(forecast=("forecast", "sum"), actual=("actual", "sum"))
         .reset_index()
         .sort_values("actual", ascending=False)
         .head(10))
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Forecast", x=g["product_category_1"],
                         y=g["forecast"], marker_color=c["manual"]))
    fig.add_trace(go.Bar(name="Actual", x=g["product_category_1"],
                         y=g["actual"], marker_color=c["actual"]))
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    fig.update_xaxes(tickangle=-25)
    fig.update_layout(barmode="group",
                      title="Forecast vs Actual — by Product Category")
    return _base_layout(fig, height=380, title="")


def chart_accuracy_by_zone(fdf):
    c = get_theme()
    if fdf.empty or not {"codp_zone", "abs_error", "actual"}.issubset(fdf.columns):
        return _empty_fig("No zone data")
    g = (fdf[fdf["codp_zone"].isin(OFFICIAL_ZONES)]
         .groupby("codp_zone").agg(ae=("abs_error", "sum"), a=("actual", "sum"))
         .reset_index())
    g = g[g["a"] > 0]
    if g.empty:
        return _empty_fig("No zone data")
    g["acc"] = (100 - g["ae"] / g["a"] * 100).clip(lower=0)
    g = g.sort_values("acc", ascending=False)

    cols = [c["success"] if v >= 85 else (c["warning"] if v >= 70 else c["danger"])
            for v in g["acc"]]
    fig = go.Figure(go.Bar(
        x=g["acc"], y=g["codp_zone"], orientation="h",
        marker_color=cols,
        text=[f"{v:.1f}%" for v in g["acc"]],
        textposition="outside"))
    fig.update_xaxes(ticksuffix="%", range=[0, 110])
    fig.update_layout(title="Accuracy by Supply Chain Zone")
    return _base_layout(fig, height=380, title="")


# =============================================================================
# TAB 2 — AI vs MANUAL PLAN
# =============================================================================
def render_ai_vs_manual(main_df, comp_df, zone_sel, plant_sel, cat_sel, year_sel):
    c = get_theme()

    needed = {"model", "ibp_forecast_cogs", "ibp_actual_cogs",
              "is_forward", "data_quality_flags"}
    if not needed.issubset(main_df.columns):
        st.warning("This view needs a comparison-ready data file. "
                   "Please replace the file or contact the analytics team.")
        return
    if "IBP_Manual" not in set(main_df["model"].dropna().unique()):
        st.warning("Manual plan rows not found in the data.")
        return

    st.markdown(
        """<div class="cx-info">
             <b>How to read this page.</b>
             We compare the <b>manual demand plan</b> against our <b>AI forecast</b>
             on the same months and products. Accuracy is shown as
             "how close to 100%", so higher is better. Bigger AI improvement =
             stronger case for AI-led planning.
           </div>""", unsafe_allow_html=True)

    ml_models = sorted(m for m in main_df["model"].dropna().unique() if m != "IBP_Manual")

    best_name, best_acc = best_ai_model(
        main_df, year=year_sel, target="order_value_usd",
        level="Total", on_ibp_scope=True)

    control_options = ["🏆 Best AI Model (auto)"] + ml_models

    ctrl = st.columns([1.0, 1.6, 1.4])
    with ctrl[0]:
        years = sorted([int(y) for y in main_df["year"].dropna().unique()])
        ibp_year = st.selectbox(
            "Year", options=years,
            index=years.index(2025) if 2025 in years else 0,
            key="aim_year")
    with ctrl[1]:
        ai_pick = st.selectbox(
            "AI Model", options=control_options, index=0,
            key="aim_model",
            help=("Auto-pick chooses the AI model with the best accuracy "
                  "on the same months and products as the manual plan."))
    with ctrl[2]:
        targets = sorted(main_df["target"].dropna().unique())
        target = st.selectbox(
            "Measure", options=targets,
            index=targets.index("order_value_usd") if "order_value_usd" in targets else 0,
            format_func=lambda t: TARGET_LABELS.get(t, t),
            key="aim_target")

    # Recompute best for the selected year (best_acc was for default year)
    if ai_pick.startswith("🏆"):
        best_for_year, _ = best_ai_model(
            main_df, year=ibp_year, target=target,
            level="Total", on_ibp_scope=True)
        ai_model = best_for_year if best_for_year and best_for_year != "—" else (ml_models[0] if ml_models else None)
        ai_label = f"AI ({ai_model}) — auto-selected"
    else:
        ai_model = ai_pick
        ai_label = f"AI ({ai_model})"

    if ai_model is None:
        st.info("No AI models available in this data.")
        return

    base = main_df[
        (main_df["year"] == ibp_year)
        & (main_df["split"] == "split2")
        & (main_df["target"] == target)
        & (main_df["level"] == "Total")
    ].copy()

    if zone_sel and "codp_zone" in base.columns:
        base = base[base["codp_zone"].isin(zone_sel + ["All"])]
    if plant_sel and "plant" in base.columns:
        base = base[base["plant"].isin(plant_sel + ["All"])]
    if cat_sel and "product_category_1" in base.columns:
        base = base[base["product_category_1"].isin(cat_sel + ["All"])]

    if base.empty:
        st.info("No data for this selection.")
        return

    ibp_rows = base[base["model"] == "IBP_Manual"].copy()
    ai_rows  = base[base["model"] == ai_model].copy()

    ibp_monthly = (ibp_rows.groupby("date", as_index=False)
                   .agg(ibp_forecast=("ibp_forecast_cogs", "sum"),
                        ibp_actual=("ibp_actual_cogs", "sum"),
                        is_forward=("is_forward", "max"))
                   .sort_values("date"))
    ibp_monthly.loc[ibp_monthly["is_forward"] == 1, "ibp_actual"] = np.nan
    ibp_monthly.loc[ibp_monthly["ibp_actual"] == 0, "ibp_actual"] = np.nan

    all_dates = pd.to_datetime([f"{ibp_year}-{m:02d}-01" for m in range(1, 13)])
    ibp_full = (ibp_monthly.set_index(pd.to_datetime(ibp_monthly["date"]))
                .reindex(all_dates).drop(columns=["date"]).reset_index()
                .rename(columns={"index": "date"}))

    ai_on_ibp = ai_rows[(ai_rows["ibp_actual_cogs"].notna())
                        & (ai_rows["is_forward"] == 0)].copy()
    if not ai_on_ibp.empty and (ai_on_ibp["actual"].abs() > 1e-6).any():
        ratio = np.where(ai_on_ibp["actual"].abs() > 1e-6,
                         ai_on_ibp["ibp_actual_cogs"] / ai_on_ibp["actual"], np.nan)
        ai_on_ibp["ai_forecast_scoped"] = ai_on_ibp["forecast"] * ratio

    if not ai_on_ibp.empty:
        ai_sum_a  = float(ai_on_ibp["ibp_actual_cogs"].sum())
        ai_sum_ae = float((ai_on_ibp["ibp_actual_cogs"]
                          - ai_on_ibp["ai_forecast_scoped"]).abs().sum())
        ai_acc    = compute_accuracy(ai_sum_a, ai_sum_ae)
        ai_err    = ai_sum_ae
    else:
        ai_acc = np.nan
        ai_err = np.nan

    mask = ibp_full["ibp_forecast"].notna() & ibp_full["ibp_actual"].notna()
    ibp_sum_f = float(ibp_full.loc[mask, "ibp_forecast"].sum())
    ibp_sum_a = float(ibp_full.loc[mask, "ibp_actual"].sum())
    ibp_sum_ae = float((ibp_full.loc[mask, "ibp_actual"]
                        - ibp_full.loc[mask, "ibp_forecast"]).abs().sum())
    ibp_acc   = compute_accuracy(ibp_sum_a, ibp_sum_ae)
    ibp_err   = ibp_sum_ae
    months_observed = int(mask.sum())

    improvement = ai_acc - ibp_acc if (pd.notna(ai_acc) and pd.notna(ibp_acc)) else np.nan
    error_reduction = ibp_err - ai_err if (pd.notna(ibp_err) and pd.notna(ai_err)) else np.nan

    st.markdown('<div class="cx-section">Headline — AI vs Manual</div>',
                unsafe_allow_html=True)

    kc = st.columns(4)
    with kc[0]:
        kpi_card("Manual Plan Accuracy",
                 fmt_pct(ibp_acc),
                 f"How close to 100% · {months_observed} months observed",
                 "cx-delta-pos" if (pd.notna(ibp_acc) and ibp_acc >= 85) else "cx-delta-neg")
    with kc[1]:
        kpi_card("AI Forecast Accuracy",
                 fmt_pct(ai_acc),
                 f"Using {ai_model}",
                 "cx-delta-pos" if (pd.notna(ai_acc) and ai_acc >= 85) else "cx-delta-neg")
    with kc[2]:
        cls = "cx-delta-pos" if (pd.notna(improvement) and improvement > 0) else "cx-delta-neg"
        verdict = ("AI is more accurate" if pd.notna(improvement) and improvement > 0
                   else ("Manual is more accurate" if pd.notna(improvement) else "—"))
        kpi_card("AI Improvement",
                 fmt_pp(improvement),
                 verdict, cls)
    with kc[3]:
        cls = "cx-delta-pos" if (pd.notna(error_reduction) and error_reduction > 0) else "cx-delta-neg"
        kpi_card("Error Reduction",
                 fmt_money(error_reduction),
                 "Less forecast error", cls)

    st.markdown('<div class="cx-section">Side-by-side — Actuals, AI Forecast, Manual Plan</div>',
                unsafe_allow_html=True)

    all_vals = []
    all_vals.extend(ibp_full["ibp_forecast"].dropna().tolist())
    all_vals.extend(ibp_full["ibp_actual"].dropna().tolist())
    if not ai_on_ibp.empty:
        all_vals.extend(ai_on_ibp["ai_forecast_scoped"].dropna().tolist())
    y_max = max(all_vals) * 1.12 if all_vals else None

    if not ai_on_ibp.empty:
        ai_by_month = (ai_on_ibp.groupby("date", as_index=False)["ai_forecast_scoped"]
                       .sum())
        ai_by_month["date"] = pd.to_datetime(ai_by_month["date"])
    else:
        ai_by_month = pd.DataFrame(columns=["date", "ai_forecast_scoped"])

    if not ai_by_month.empty:
        ai_full = (ai_by_month.set_index(pd.to_datetime(ai_by_month["date"]))
                   .reindex(all_dates).reset_index()
                   .rename(columns={"index": "date"}))
    else:
        ai_full = pd.DataFrame({"date": all_dates,
                                "ai_forecast_scoped": [np.nan] * len(all_dates)})

    g1, g2, g3 = st.columns(3)

    with g1:
        st.markdown(
            f"""<div class="cx-chart-title" style="color:{c['actual']};">
                  📊 What Actually Happened
                </div>
                <div class="cx-chart-sub">Real sales — the truth we're trying to predict</div>""",
            unsafe_allow_html=True)
        fig = go.Figure()
        _add_no_data_shading(fig, ibp_full, "ibp_actual", c)
        fig.add_trace(go.Bar(
            x=ibp_full["date"], y=ibp_full["ibp_actual"],
            name="Actual", marker_color=c["actual"]))
        fig.update_xaxes(tickformat="%b")
        fig.update_yaxes(tickprefix="$", separatethousands=True,
                         range=[0, y_max] if y_max else None)
        fig.update_layout(showlegend=False)
        st.plotly_chart(_base_layout(fig, height=340), use_container_width=True)
        st.caption(f"Total: {fmt_money(ibp_sum_a)} over {months_observed} months observed")

    with g2:
        st.markdown(
            f"""<div class="cx-chart-title" style="color:{c['ai']};">
                  🤖 AI Forecast — {fmt_pct(ai_acc)} accurate
                </div>
                <div class="cx-chart-sub">{ai_label}</div>""",
            unsafe_allow_html=True)
        fig = go.Figure()
        _add_no_data_shading(fig, ai_full, "ai_forecast_scoped", c)
        fig.add_trace(go.Bar(
            x=ai_full["date"], y=ai_full["ai_forecast_scoped"],
            name="AI Forecast", marker_color=c["ai"]))
        fig.update_xaxes(tickformat="%b")
        fig.update_yaxes(tickprefix="$", separatethousands=True,
                         range=[0, y_max] if y_max else None)
        fig.update_layout(showlegend=False)
        st.plotly_chart(_base_layout(fig, height=340), use_container_width=True)
        st.caption(f"Total error: {fmt_money(ai_err)}")

    with g3:
        st.markdown(
            f"""<div class="cx-chart-title" style="color:{c['manual']};">
                  ✍️ Manual Plan — {fmt_pct(ibp_acc)} accurate
                </div>
                <div class="cx-chart-sub">Today's demand consensus</div>""",
            unsafe_allow_html=True)
        fig = go.Figure()
        _add_no_data_shading(fig, ibp_full, "ibp_forecast", c)
        fig.add_trace(go.Bar(
            x=ibp_full["date"], y=ibp_full["ibp_forecast"],
            name="Manual Plan", marker_color=c["manual"]))
        fig.update_xaxes(tickformat="%b")
        fig.update_yaxes(tickprefix="$", separatethousands=True,
                         range=[0, y_max] if y_max else None)
        fig.update_layout(showlegend=False)
        st.plotly_chart(_base_layout(fig, height=340), use_container_width=True)
        st.caption(f"Total error: {fmt_money(ibp_err)}")

    if pd.notna(improvement) and pd.notna(error_reduction):
        if improvement > 0:
            verdict_line = (
                f"For the months we can verify, the AI forecast was "
                f"<b style='color:{c['success']};'>{abs(improvement):.1f} percentage points "
                f"more accurate</b> than the manual plan — that's "
                f"<b style='color:{c['success']};'>{fmt_money(error_reduction)} less error</b> "
                f"on the same products."
            )
        else:
            verdict_line = (
                f"For the months we can verify, the manual plan was "
                f"<b style='color:{c['warning']};'>{abs(improvement):.1f} percentage points "
                f"more accurate</b> than this AI model — "
                f"<b style='color:{c['warning']};'>{fmt_money(abs(error_reduction))} more error</b> "
                f"in AI predictions."
            )
        st.markdown(
            f"""<div class="cx-info" style="font-size:14px; padding:14px 18px;">
                  {verdict_line}
                </div>""", unsafe_allow_html=True)

    if comp_df is not None and not comp_df.empty:
        st.markdown('<div class="cx-section">Detail by Product Category × Month</div>',
                    unsafe_allow_html=True)
        cmp_filt = comp_df.copy()
        if cat_sel:
            cmp_filt = cmp_filt[cmp_filt["product_category_1"].isin(cat_sel)]

        if cmp_filt.empty:
            st.caption("No comparison rows for selected categories.")
        else:
            disp = cmp_filt.copy()
            disp["Month"] = disp["month"].apply(
                lambda m: MONTH_NAMES[int(m) - 1] if 1 <= int(m) <= 12 else str(m))
            disp["AI Accuracy"]     = (100 - disp["model_wape"]).clip(lower=0)
            disp["Manual Accuracy"] = (100 - disp["ibp_wape"]).clip(lower=0)
            disp["Winner"] = np.where(
                disp["AI Accuracy"] > disp["Manual Accuracy"],
                "🤖 AI", "✍️ Manual")
            disp["AI Accuracy"]     = disp["AI Accuracy"].apply(fmt_pct)
            disp["Manual Accuracy"] = disp["Manual Accuracy"].apply(fmt_pct)
            for col in ("model_actual_usd", "model_forecast_usd",
                        "ibp_actual_cogs", "ibp_fc_cogs"):
                if col in disp.columns:
                    disp[col] = disp[col].apply(fmt_money)

            rename = {"product_category_1": "Product Category",
                      "ibp_actual_cogs": "Manual Actual",
                      "ibp_fc_cogs": "Manual Forecast",
                      "model_actual_usd": "AI Actual",
                      "model_forecast_usd": "AI Forecast"}
            disp = disp.rename(columns=rename)
            keep_cols = ["Product Category", "Month",
                         "Manual Forecast", "Manual Actual", "Manual Accuracy",
                         "AI Forecast", "AI Actual", "AI Accuracy", "Winner"]
            keep_cols = [c for c in keep_cols if c in disp.columns]
            st.dataframe(disp[keep_cols], use_container_width=True,
                         hide_index=True, height=420)

            ai_wins  = (disp["Winner"] == "🤖 AI").sum()
            man_wins = (disp["Winner"] == "✍️ Manual").sum()
            st.caption(f"AI won in **{ai_wins}** of {len(disp)} category × month "
                       f"comparisons; manual plan won in **{man_wins}**.")

    with st.expander("📘 About this comparison (data scope & method)"):
        st.markdown(
            """
- **Region.** All numbers are US (NAM) only.
- **Manual plan coverage.** The manual plan covers about a tenth of total
  US revenue — the planned-production product set. The AI forecast covers
  the full US revenue surface, so for a fair head-to-head we rescale the AI
  forecast to the same products the manual plan covers.
- **Observed window.** Manual plan actuals are only available for the first
  five months of 2025. Later months in the charts show forecast only, with
  greyed-out actual bars.
- **Categories covered by manual plan (6):** Heat Tracing Components,
  Floor Heating, Control/Monitoring/Power Distribution,
  Polymer Pipe Heat Tracing – BIS, Polymer Pipe Heat Tracing – IND,
  Snow Melting & De-Icing.
            """
        )


# =============================================================================
# TAB 3 — ACCURACY BY LINE
# =============================================================================
def render_accuracy_table(fdf, level_sel, year_sel):
    st.markdown(
        f'<div class="cx-section">Accuracy by Line — {level_sel} · {year_sel}</div>',
        unsafe_allow_html=True)
    st.caption("Accuracy = how close the forecast was to actual sales. "
               "Higher is better. Status: ✅ on track · 🟡 watch · 🔴 off track.")

    if fdf.empty:
        st.info("No rows for the current filter selection.")
        return

    group_cols = {
        "Total": [],
        "CODP Zone": ["codp_zone"],
        "Plant": ["plant"],
        "Product Category": ["product_category_1"],
        "Group": ["codp_zone", "plant", "product_category_1"],
    }.get(level_sel, [])
    group_cols = [c for c in group_cols if c in fdf.columns]

    if group_cols:
        agg = (fdf.groupby(group_cols, dropna=False)
               .agg(Forecast=("forecast", "sum"),
                    Actual=("actual", "sum"),
                    AbsErr=("abs_error", "sum"))
               .reset_index())
    else:
        agg = pd.DataFrame({
            "Forecast": [fdf["forecast"].sum()],
            "Actual":   [fdf["actual"].sum()],
            "AbsErr":   [fdf["abs_error"].sum() if "abs_error" in fdf.columns
                         else (fdf["actual"] - fdf["forecast"]).abs().sum()]})

    agg["Accuracy"] = (100 - agg["AbsErr"] / agg["Actual"] * 100).clip(lower=0)
    agg["Alignment"] = np.where(agg["Forecast"] > 0,
                                agg["Actual"] / agg["Forecast"] * 100, np.nan)
    agg["Gap"] = agg["Actual"] - agg["Forecast"]

    def _status(a):
        if pd.isna(a):           return "⚠️ Missing"
        if a >= 90:               return "✅ On Track"
        if a >= 75:               return "🟡 Watch"
        return "🔴 Off Track"

    agg["Status"] = agg["Accuracy"].apply(_status)
    if group_cols:
        agg["Line Item"] = agg[group_cols].astype(str).agg(" | ".join, axis=1)
    else:
        agg["Line Item"] = "Total"

    rename = {"codp_zone": "Supply Chain Zone", "plant": "Plant",
              "product_category_1": "Product Category"}
    agg = agg.rename(columns=rename)
    disp = pd.DataFrame()
    disp["Forecast Level"] = [level_sel] * len(agg)
    disp["Line Item"] = agg["Line Item"].values
    for opt in ("Supply Chain Zone", "Plant", "Product Category"):
        if opt in agg.columns:
            disp[opt] = agg[opt].values
    disp["Forecast"]  = agg["Forecast"].apply(lambda v: fmt_money(v, compact=False))
    disp["Actual"]    = agg["Actual"].apply(lambda v: fmt_money(v, compact=False))
    disp["Accuracy"]  = agg["Accuracy"].apply(fmt_pct)
    disp["Gap"]       = agg["Gap"].apply(lambda v: fmt_money(v, compact=False))
    disp["Status"]    = agg["Status"].values

    disp = disp.sort_values("Line Item").reset_index(drop=True)
    st.dataframe(disp, use_container_width=True, hide_index=True, height=520)

    csv = disp.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download as CSV", data=csv,
        file_name=f"accuracy_{level_sel.lower().replace(' ', '_')}.csv",
        mime="text/csv")


# =============================================================================
# TAB 4 — DRILL DOWN
# =============================================================================
def render_drilldown(fdf):
    c = get_theme()
    st.markdown('<div class="cx-section">Drill Down by Dimension</div>',
                unsafe_allow_html=True)
    if fdf.empty:
        st.info("No rows for current filters.")
        return

    dim_opts = [d for d in ("codp_zone", "plant", "product_category_1")
                if d in fdf.columns]
    if not dim_opts:
        st.info("No drilldown dimensions available.")
        return

    nice = {"codp_zone": "Supply Chain Zone",
            "plant": "Plant",
            "product_category_1": "Product Category"}

    col1, col2 = st.columns([1.2, 4])
    with col1:
        dim = st.selectbox("View by", options=dim_opts,
                           format_func=lambda d: nice[d])

    if "abs_error" in fdf.columns:
        agg = (fdf[fdf[dim] != "All"].groupby(dim)
               .agg(Forecast=("forecast", "sum"),
                    Actual=("actual", "sum"),
                    AbsErr=("abs_error", "sum"))
               .reset_index())
    else:
        agg = (fdf[fdf[dim] != "All"].groupby(dim)
               .agg(Forecast=("forecast", "sum"),
                    Actual=("actual", "sum"))
               .reset_index())
        agg["AbsErr"] = (agg["Actual"] - agg["Forecast"]).abs()

    agg = agg[agg["Actual"] > 0]
    agg["Accuracy"] = (100 - agg["AbsErr"] / agg["Actual"] * 100).clip(lower=0)
    agg = agg.sort_values("Actual", ascending=False).head(15)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Forecast", x=agg[dim], y=agg["Forecast"],
                         marker_color=c["manual"]))
    fig.add_trace(go.Bar(name="Actual", x=agg[dim], y=agg["Actual"],
                         marker_color=c["actual"]))
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    fig.update_xaxes(tickangle=-25)
    fig.update_layout(barmode="group",
                      title=f"Forecast vs Actual by {nice[dim]}")
    with col2:
        st.plotly_chart(_base_layout(fig, height=380, title=""),
                        use_container_width=True)

    disp = agg.copy()
    disp["Forecast"] = disp["Forecast"].apply(fmt_money)
    disp["Actual"]   = disp["Actual"].apply(fmt_money)
    disp["Accuracy"] = disp["Accuracy"].apply(fmt_pct)
    disp = disp.drop(columns=["AbsErr"])
    disp = disp.rename(columns={dim: nice[dim]})
    st.dataframe(disp, use_container_width=True, hide_index=True)


# =============================================================================
# TAB 5 — FORWARD PLAN
# =============================================================================
def render_forward_plan(main_df, year_sel, zone_sel, plant_sel, cat_sel):
    c = get_theme()
    st.markdown('<div class="cx-section">Forward Plan — Looking Ahead</div>',
                unsafe_allow_html=True)
    st.caption("Forward months (no actuals yet). Shows what the AI forecast "
               "and the manual plan expect in the coming periods.")

    if "is_forward" not in main_df.columns:
        st.info("This file does not include forward-looking rows.")
        return

    best_name, _ = best_ai_model(main_df, year=year_sel,
                                 target="order_value_usd", level="Total",
                                 on_ibp_scope=True)

    fwd = main_df[(main_df["is_forward"] == 1)
                  & (main_df["target"] == "order_value_usd")
                  & (main_df["level"] == "Total")
                  & (main_df["split"] == "split2")]

    if zone_sel and "codp_zone" in fwd.columns:
        fwd = fwd[fwd["codp_zone"].isin(zone_sel + ["All"])]
    if plant_sel and "plant" in fwd.columns:
        fwd = fwd[fwd["plant"].isin(plant_sel + ["All"])]
    if cat_sel and "product_category_1" in fwd.columns:
        fwd = fwd[fwd["product_category_1"].isin(cat_sel + ["All"])]

    if fwd.empty:
        st.info("No forward-looking rows in current selection.")
        return

    ai_fwd  = (fwd[fwd["model"] == best_name].groupby("date", as_index=False)
               ["forecast"].sum().sort_values("date")) if best_name and best_name != "—" else pd.DataFrame()
    ibp_fwd = (fwd[fwd["model"] == "IBP_Manual"].groupby("date", as_index=False)
               ["ibp_forecast_cogs"].sum().sort_values("date"))

    fig = go.Figure()
    if not ai_fwd.empty:
        fig.add_trace(go.Scatter(
            x=ai_fwd["date"], y=ai_fwd["forecast"], mode="lines+markers",
            name=f"AI Forecast ({best_name})",
            line=dict(color=c["ai"], width=3)))
    if not ibp_fwd.empty:
        fig.add_trace(go.Scatter(
            x=ibp_fwd["date"], y=ibp_fwd["ibp_forecast_cogs"],
            mode="lines+markers", name="Manual Plan",
            line=dict(color=c["manual"], width=3)))
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    fig.update_xaxes(tickformat="%b %Y")
    st.plotly_chart(_base_layout(fig, height=400, title="Forward forecasts"),
                    use_container_width=True)

    if not ai_fwd.empty or not ibp_fwd.empty:
        df_merge = pd.merge(
            ai_fwd.rename(columns={"forecast": "AI Forecast"}) if not ai_fwd.empty
                else pd.DataFrame(columns=["date", "AI Forecast"]),
            ibp_fwd.rename(columns={"ibp_forecast_cogs": "Manual Plan"})
                if not ibp_fwd.empty else pd.DataFrame(columns=["date", "Manual Plan"]),
            on="date", how="outer").sort_values("date")
        df_merge["Month"] = pd.to_datetime(df_merge["date"]).dt.strftime("%b %Y")
        for col in ("AI Forecast", "Manual Plan"):
            if col in df_merge.columns:
                df_merge[col] = df_merge[col].apply(fmt_money)
        cols = ["Month"] + [c for c in ("AI Forecast", "Manual Plan")
                            if c in df_merge.columns]
        st.dataframe(df_merge[cols], use_container_width=True, hide_index=True)


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    main()
    c = get_theme()
    st.markdown(
        f"""<div style="margin-top:24px;padding-top:12px;
            border-top:1px solid {c['border']};
            color:{c['text_muted']};font-size:12px;text-align:center;">
            Chemelex Demand Forecasting Cockpit • Theme: {st.session_state.theme}
            </div>""",
        unsafe_allow_html=True,
    )
