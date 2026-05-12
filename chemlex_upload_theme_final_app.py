import os
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# Chemelex Demand Forecasting Dashboard
# Final clean business version
# - Visible upload on main page (not hidden in sidebar/expander)
# - Visible Light/Dark theme toggle on main page
# - Product Category / Plant / CODP multi-select filters
# - No weather, no MAPE, no target/change columns in business table
# - Business Accuracy = Actual / Forecast * 100
# =============================================================================

st.set_page_config(
    page_title="Chemelex Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MONTH_NAME = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}

# ----------------------------
# Helpers
# ----------------------------
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None:
        df = pd.DataFrame()
    return df.to_csv(index=False).encode("utf-8")


def fmt_num(x, decimals=0):
    try:
        if pd.isna(x):
            return "-"
        x = float(x)
    except Exception:
        return "-"
    sign = "-" if x < 0 else ""
    x_abs = abs(x)
    if x_abs >= 1_000_000_000:
        return f"{sign}{x_abs/1_000_000_000:,.{decimals}f}B"
    if x_abs >= 1_000_000:
        return f"{sign}{x_abs/1_000_000:,.{decimals}f}M"
    if x_abs >= 1_000:
        return f"{sign}{x_abs/1_000:,.{decimals}f}K"
    return f"{x:,.{decimals}f}"


def business_accuracy_series(actual, forecast):
    actual = pd.to_numeric(actual, errors="coerce")
    forecast = pd.to_numeric(forecast, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (actual / forecast) * 100
    return out.replace([np.inf, -np.inf], np.nan)


def business_accuracy_scalar(actual, forecast):
    try:
        actual = float(actual)
        forecast = float(forecast)
    except Exception:
        return np.nan
    if forecast == 0:
        return 100.0 if actual == 0 else np.nan
    return actual / forecast * 100


def data_status(row):
    flags = str(row.get("data_quality_flags", "")).strip()
    if flags.lower() in ["", "nan", "none", "null"]:
        flags = ""
    try:
        forecast = float(row.get("forecast", np.nan))
        actual = float(row.get("actual", np.nan))
    except Exception:
        return flags or "Check values"
    if forecast < 0:
        return "Invalid negative forecast"
    if actual < 0:
        return "Invalid negative actual"
    if forecast == 0 and actual > 0:
        return "No forecast with demand"
    if forecast == 0 and actual == 0:
        return "No demand / no forecast"
    if flags:
        return flags.replace("_", " ").title()
    return "OK"


def load_csv_or_excel(file_obj):
    name = getattr(file_obj, "name", "").lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_obj)
    return pd.read_csv(file_obj)


def load_default_data():
    candidates = [
        "corrected_dashboard_forecast_accuracy_levelled.csv",
        "dashboard_forecast_accuracy_levelled.csv",
        "forecast_backtest_results.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path), path

    # Demo fallback so the app never opens blank.
    demo = pd.DataFrame({
        "level": ["Group", "Group", "Group", "Plant", "Product Category"],
        "dimension_value": [
            "Push / MPS | Chemelex - Trenton | PD / Heat Tracing Components",
            "Pull / Kanban | Chemelex - RWC/UCDC | PD / Floor Heating",
            "BUFFER (at CODP) | Chemelex - Trenton | PD / MI Heat Tracing",
            "Chemelex - Trenton",
            "PD / Heat Tracing Components",
        ],
        "codp_zone": ["Push / MPS", "Pull / Kanban", "BUFFER (at CODP)", "All", "All"],
        "plant": ["Chemelex - Trenton", "Chemelex - RWC/UCDC", "Chemelex - Trenton", "Chemelex - Trenton", "All"],
        "product_category_1": ["PD / Heat Tracing Components", "PD / Floor Heating", "PD / MI Heat Tracing", "All", "PD / Heat Tracing Components"],
        "date": ["2025-01-01"] * 5,
        "year": [2025] * 5,
        "month": [1] * 5,
        "target": ["order_qty_bu"] * 5,
        "model": ["Best Model"] * 5,
        "split": ["split2"] * 5,
        "is_best": [1] * 5,
        "actual": [120, 80, 160, 280, 120],
        "forecast": [100, 100, 200, 300, 100],
        "data_quality_flags": [""] * 5,
    })
    return demo, "generated demo data"


def normalize_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {
        "Supply_Chain_Zone": "codp_zone",
        "Supply Chain Zone": "codp_zone",
        "CODP Zone": "codp_zone",
        "Plant Name": "plant",
        "Product 1 Category/Division Name": "product_category_1",
        "group": "dimension_value",
        "group_key": "dimension_value",
        "accuracy_percent": "accuracy_pct",
        "target_metric": "target",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    defaults = {
        "level": "Group",
        "dimension_value": "",
        "codp_zone": "All",
        "plant": "All",
        "product_category_1": "All",
        "target": "order_qty_bu",
        "model": "Best Model",
        "split": "split2",
        "is_best": 1,
        "data_quality_flags": "",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

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

    # If date was missing, rebuild from year/month.
    missing_date = df["date"].isna()
    if missing_date.any():
        rebuilt = pd.to_datetime(
            df.loc[missing_date, "year"].astype(str) + "-" + df.loc[missing_date, "month"].astype(str) + "-01",
            errors="coerce",
        )
        df.loc[missing_date, "date"] = rebuilt

    for col in ["actual", "forecast"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in ["level", "dimension_value", "codp_zone", "plant", "product_category_1", "target", "model", "split"]:
        df[col] = df[col].fillna("All").astype(str).str.strip().replace({"": "All", "nan": "All", "None": "All"})

    # Build line item if missing.
    missing_dim = df["dimension_value"].isin(["", "All", "nan", "None"])
    df.loc[missing_dim, "dimension_value"] = (
        df.loc[missing_dim, "codp_zone"].astype(str)
        + " | " + df.loc[missing_dim, "plant"].astype(str)
        + " | " + df.loc[missing_dim, "product_category_1"].astype(str)
    )

    df["accuracy_pct"] = business_accuracy_series(df["actual"], df["forecast"])
    df["gap"] = df["actual"] - df["forecast"]
    df["abs_gap"] = df["gap"].abs()
    df["data_status"] = df.apply(data_status, axis=1)
    df["period"] = df["year"].astype(str) + " - " + df["month"].map(lambda m: MONTH_NAME.get(int(m), str(m)) if pd.notna(m) else "Unknown")
    return df


def unique_sorted(series, exclude_all=False) -> List[str]:
    vals = [str(v).strip() for v in series.dropna().unique().tolist() if str(v).strip() not in ["", "nan", "None"]]
    if exclude_all:
        vals = [v for v in vals if v != "All"]
    return sorted(vals)


# ----------------------------
# Theme and CSS
# ----------------------------
def inject_css(theme: str):
    dark = theme == "Dark"
    bg = "#0f172a" if dark else "#f7f8fb"
    panel = "#111827" if dark else "#ffffff"
    panel2 = "#1e293b" if dark else "#f3f6fb"
    text = "#f8fafc" if dark else "#111827"
    muted = "#94a3b8" if dark else "#64748b"
    border = "#334155" if dark else "#e8edf3"
    alert_bg = "#2a1e10" if dark else "#fff7e6"
    alert_border = "#7c2d12" if dark else "#ffd591"
    alert_text = "#fed7aa" if dark else "#7c4a03"

    st.markdown(
        f"""
        <style>
        .stApp {{ background:{bg}; color:{text}; }}
        .block-container {{ padding-top:1.2rem; max-width:1450px; }}
        [data-testid="stSidebar"] {{ background:{panel}; }}
        .topbar,.upload-card,.filter-card,.metric-card,.chart-card,.table-card,.info-card {{
            background:{panel}; border:1px solid {border}; color:{text}; border-radius:20px;
            box-shadow:0 10px 26px rgba(15,23,42,.05);
        }}
        .topbar {{ padding:18px 22px; margin-bottom:14px; }}
        .upload-card {{ padding:16px; margin-bottom:16px; border:2px dashed #1677ff; }}
        .filter-card {{ padding:16px; margin-bottom:16px; }}
        .metric-card {{ padding:16px; }}
        .chart-card,.table-card,.info-card {{ padding:16px; margin-bottom:16px; }}
        .brand-row {{ display:flex; align-items:center; justify-content:space-between; gap:16px; }}
        .brand-left {{ display:flex; align-items:center; gap:14px; }}
        .logo-box {{ width:44px; height:44px; border-radius:14px; background:#1677ff; display:flex; align-items:center; justify-content:center; color:white; font-weight:900; font-size:20px; }}
        .page-title {{ font-size:24px; font-weight:900; line-height:1.1; margin:0; color:{text}; }}
        .page-subtitle,.muted,.small-note,.metric-help {{ color:{muted}; }}
        .tab-row {{ display:flex; gap:10px; margin:8px 0 14px 0; flex-wrap:wrap; }}
        .tab-pill {{ border-radius:999px; padding:9px 15px; font-size:13px; font-weight:800; background:{panel2}; color:{muted}; border:1px solid {border}; }}
        .tab-pill.active {{ background:#1677ff; color:white; border-color:#1677ff; }}
        .alert-strip {{ background:{alert_bg}; border:1px solid {alert_border}; color:{alert_text}; border-radius:14px; padding:10px 14px; margin:12px 0 16px 0; font-size:13px; font-weight:700; }}
        .section-title {{ font-size:16px; font-weight:900; margin-bottom:4px; color:{text}; }}
        .metric-label {{ font-size:12px; text-transform:uppercase; letter-spacing:.04em; font-weight:900; color:{muted}; }}
        .metric-value {{ font-size:29px; font-weight:950; margin-top:6px; color:{text}; }}
        .metric-grid {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:14px; margin-bottom:16px; }}
        .status-ok {{ color:#16a34a; font-weight:800; }}
        .status-bad {{ color:#dc2626; font-weight:800; }}
        .theme-label {{ font-size:12px; text-transform:uppercase; font-weight:900; color:{muted}; margin-bottom:4px; }}
        @media (max-width:1100px) {{ .metric-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_plot(fig, theme):
    dark = theme == "Dark"
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb" if dark else "#111827"),
        margin=dict(l=20, r=20, t=45, b=25),
        height=360,
        legend=dict(orientation="h", y=1.15, x=0),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.18)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.18)")
    return fig


# ----------------------------
# Always-visible upload and theme controls
# ----------------------------
control_col1, control_col2 = st.columns([3.2, 1])
with control_col2:
    theme_choice = st.radio("Theme", ["Light", "Dark"], horizontal=True, key="theme_visible")

inject_css(theme_choice)

st.markdown(
    """
    <div class="topbar">
      <div class="brand-row">
        <div class="brand-left">
          <div class="logo-box">C</div>
          <div>
            <div class="page-title">Chemelex Demand Forecasting</div>
            <div class="page-subtitle">Forecast vs Actual Accuracy · Clean business review</div>
          </div>
        </div>
        <div class="muted" style="font-size:12px;">Accuracy % = Actual ÷ Forecast × 100</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="upload-card">
      <div class="section-title">📤 Upload Forecast Data</div>
      <div class="small-note">Upload <b>corrected_dashboard_forecast_accuracy_levelled.csv</b>. If no file is uploaded, the app uses the CSV stored in the GitHub repo.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
uploaded_file = st.file_uploader(
    "Upload / replace forecast file (.csv, .xlsx, .xls)",
    type=["csv", "xlsx", "xls"],
    key="main_visible_uploader",
    label_visibility="visible",
)

# Keep a duplicate sidebar uploader as backup, but main uploader above is primary.
with st.sidebar:
    st.markdown("### Backup Upload")
    sidebar_upload = st.file_uploader(
        "Upload forecast file",
        type=["csv", "xlsx", "xls"],
        key="sidebar_backup_uploader",
    )

try:
    final_upload = uploaded_file if uploaded_file is not None else sidebar_upload
    if final_upload is not None:
        raw_df = load_csv_or_excel(final_upload)
        source_name = final_upload.name
    else:
        raw_df, source_name = load_default_data()
    df = normalize_data(raw_df)
except Exception as exc:
    st.error(f"Unable to load forecast file: {exc}")
    st.stop()

st.markdown(
    f"""
    <div class="tab-row">
      <span class="tab-pill active">Material Forecast</span>
      <span class="tab-pill">Forecast Analysis</span>
      <span class="tab-pill">Forecast Level Clarity</span>
    </div>
    <div class="alert-strip">
      Loaded source: {source_name}. Select filters below to review one clean forecast line item per selected level.
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Filters
# ----------------------------
levels = unique_sorted(df["level"])
default_level = "Group" if "Group" in levels else (levels[0] if levels else "Group")
targets = unique_sorted(df["target"])
default_target = "order_qty_bu" if "order_qty_bu" in targets else (targets[0] if targets else "order_qty_bu")
splits = unique_sorted(df["split"])
default_split = "split2" if "split2" in splits else (splits[0] if splits else "split2")
years = sorted([int(y) for y in df["year"].dropna().unique().tolist()])
months = sorted([int(m) for m in df["month"].dropna().unique().tolist() if 1 <= int(m) <= 12])

st.markdown('<div class="filter-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Filters</div><div class="small-note">Product Category, Plant and CODP are multi-select. Leave blank to include all.</div>', unsafe_allow_html=True)

r1c1, r1c2, r1c3, r1c4 = st.columns([1.1, 1.1, 1.1, 1.3])
with r1c1:
    selected_level = st.selectbox("Forecast Level", levels, index=levels.index(default_level) if default_level in levels else 0)
with r1c2:
    selected_target = st.selectbox("Metric", targets, index=targets.index(default_target) if default_target in targets else 0)
with r1c3:
    selected_split = st.selectbox("Split", splits, index=splits.index(default_split) if default_split in splits else 0)
with r1c4:
    best_only = st.toggle("Best model only", value=True)

base = df[(df["level"] == selected_level) & (df["target"] == selected_target) & (df["split"] == selected_split)].copy()
if best_only:
    base = base[base["is_best"].astype(str).isin(["1", "True", "true", "YES", "Yes"])]

models = unique_sorted(base["model"])
plants = unique_sorted(base["plant"], exclude_all=True)
codps = unique_sorted(base["codp_zone"], exclude_all=True)
categories = unique_sorted(base["product_category_1"], exclude_all=True)

r2c1, r2c2, r2c3, r2c4 = st.columns([1, 1, 1.25, 1.25])
with r2c1:
    selected_years = st.multiselect("Year", years, default=years[-1:] if years else [])
with r2c2:
    month_labels = [MONTH_NAME.get(m, str(m)) for m in months]
    month_map = {MONTH_NAME.get(m, str(m)): m for m in months}
    selected_month_labels = st.multiselect("Month", month_labels, default=month_labels)
    selected_months = [month_map[m] for m in selected_month_labels]
with r2c3:
    selected_plants = st.multiselect("Plant", plants, default=[])
with r2c4:
    selected_categories = st.multiselect("Product Category", categories, default=[])

r3c1, r3c2, r3c3 = st.columns([1.25, 1.25, 1.5])
with r3c1:
    selected_codps = st.multiselect("CODP / Supply Chain Zone", codps, default=[])
with r3c2:
    selected_models = st.multiselect("Model", models, default=models[:1] if models else [])
with r3c3:
    search_text = st.text_input("Search line item", placeholder="Search CODP / plant / product category...")

st.markdown('</div>', unsafe_allow_html=True)

filtered = base.copy()
if selected_years:
    filtered = filtered[filtered["year"].isin(selected_years)]
if selected_months:
    filtered = filtered[filtered["month"].isin(selected_months)]
if selected_plants:
    filtered = filtered[filtered["plant"].isin(selected_plants)]
if selected_categories:
    filtered = filtered[filtered["product_category_1"].isin(selected_categories)]
if selected_codps:
    filtered = filtered[filtered["codp_zone"].isin(selected_codps)]
if selected_models:
    filtered = filtered[filtered["model"].isin(selected_models)]
if search_text.strip():
    s = search_text.strip().lower()
    mask = (
        filtered["dimension_value"].str.lower().str.contains(s, na=False)
        | filtered["codp_zone"].str.lower().str.contains(s, na=False)
        | filtered["plant"].str.lower().str.contains(s, na=False)
        | filtered["product_category_1"].str.lower().str.contains(s, na=False)
    )
    filtered = filtered[mask]

# ----------------------------
# KPI Cards
# ----------------------------
total_forecast = filtered["forecast"].sum()
total_actual = filtered["actual"].sum()
overall_acc = business_accuracy_scalar(total_actual, total_forecast)
total_gap = total_actual - total_forecast
at_risk = int((filtered["data_status"] != "OK").sum())

st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
for label, value, help_text in [
    ("Active Line Items", f"{len(filtered):,}", f"Selected level: {selected_level}"),
    ("Total Forecast", fmt_num(total_forecast, 1), selected_target),
    ("Total Actual", fmt_num(total_actual, 1), selected_target),
    ("Accuracy %", "-" if pd.isna(overall_acc) else f"{overall_acc:,.1f}%", "Actual ÷ Forecast × 100"),
    ("Gap", fmt_num(total_gap, 1), "Actual - Forecast"),
]:
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-help">{help_text}</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if filtered.empty:
    st.warning("No rows match the selected filters. Change filters or upload another file.")
    st.stop()

# ----------------------------
# Charts
# ----------------------------
tab1, tab2, tab3 = st.tabs(["Material Forecast", "Forecast Analysis", "Forecast Level Clarity"])

with tab1:
    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.markdown('<div class="chart-card"><div class="section-title">Forecast vs Actual by Month</div><div class="small-note">Simple business view. No MAPE shown.</div>', unsafe_allow_html=True)
        monthly = filtered.groupby(["year", "month"], as_index=False)[["actual", "forecast"]].sum()
        monthly["period_label"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
        fig = go.Figure()
        fig.add_bar(x=monthly["period_label"], y=monthly["forecast"], name="Forecast", marker_color="#1677ff")
        fig.add_bar(x=monthly["period_label"], y=monthly["actual"], name="Actual", marker_color="#12b76a")
        fig.update_layout(barmode="group", title="Forecast vs Actual")
        st.plotly_chart(style_plot(fig, theme_choice), use_container_width=True, key="monthly_forecast_actual")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="chart-card"><div class="section-title">Accuracy by Month</div><div class="small-note">Accuracy % = Actual ÷ Forecast × 100</div>', unsafe_allow_html=True)
        monthly["accuracy_pct"] = business_accuracy_series(monthly["actual"], monthly["forecast"])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=monthly["period_label"], y=monthly["accuracy_pct"], mode="lines+markers", name="Accuracy %", line=dict(color="#f79009", width=3)))
        fig2.add_hline(y=100, line_dash="dash", line_color="#667085", annotation_text="100% aligned")
        fig2.update_layout(title="Monthly Accuracy %")
        st.plotly_chart(style_plot(fig2, theme_choice), use_container_width=True, key="monthly_accuracy")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="table-card"><div class="section-title">Forecast Accuracy Table</div><div class="small-note">Business table only. Target/change/MAPE are intentionally not shown.</div>', unsafe_allow_html=True)
    display_cols = [
        "period", "level", "dimension_value", "codp_zone", "plant", "product_category_1",
        "forecast", "actual", "accuracy_pct", "gap", "data_status",
    ]
    table = filtered.sort_values("abs_gap", ascending=False).head(500)[display_cols].copy()
    table = table.rename(columns={
        "period": "Period",
        "level": "Forecast Level",
        "dimension_value": "Line Item",
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
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Forecast": st.column_config.NumberColumn(format="%.2f"),
            "Actual": st.column_config.NumberColumn(format="%.2f"),
            "Accuracy %": st.column_config.NumberColumn(format="%.1f%%"),
            "Gap": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.download_button("⬇️ Download visible table", data=to_csv_bytes(table), file_name="chemelex_visible_forecast_table.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="chart-card"><div class="section-title">Product Category: Forecast vs Actual</div>', unsafe_allow_html=True)
        cat = filtered.groupby("product_category_1", as_index=False)[["actual", "forecast"]].sum().sort_values("forecast", ascending=False).head(15)
        fig3 = go.Figure()
        fig3.add_bar(y=cat["product_category_1"], x=cat["forecast"], orientation="h", name="Forecast", marker_color="#1677ff")
        fig3.add_bar(y=cat["product_category_1"], x=cat["actual"], orientation="h", name="Actual", marker_color="#12b76a")
        fig3.update_layout(barmode="group", title="Top Categories")
        st.plotly_chart(style_plot(fig3, theme_choice), use_container_width=True, key="category_chart")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="chart-card"><div class="section-title">Plant Accuracy</div>', unsafe_allow_html=True)
        plant = filtered[filtered["plant"] != "All"].groupby("plant", as_index=False)[["actual", "forecast"]].sum()
        plant["accuracy_pct"] = business_accuracy_series(plant["actual"], plant["forecast"])
        plant = plant.sort_values("accuracy_pct", ascending=True).head(15)
        fig4 = go.Figure()
        fig4.add_bar(y=plant["plant"], x=plant["accuracy_pct"], orientation="h", name="Accuracy %", marker_color="#7c3aed")
        fig4.add_vline(x=100, line_dash="dash", line_color="#667085")
        fig4.update_layout(title="Plant Accuracy %")
        st.plotly_chart(style_plot(fig4, theme_choice), use_container_width=True, key="plant_accuracy")
        st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="info-card"><div class="section-title">Forecast Level Clarity</div>', unsafe_allow_html=True)
    st.write(
        "This dashboard can show one clean line item per selected forecast level. "
        "For the current Synthefy file, the most detailed usable level is usually **Group**, which represents **CODP Zone + Plant + Product Category**. "
        "Do not call it SKU/material-level unless the file contains material_code / SKU / material_description columns."
    )
    level_summary = df.groupby("level", as_index=False).agg(
        rows=("level", "size"),
        total_forecast=("forecast", "sum"),
        total_actual=("actual", "sum"),
    )
    level_summary["accuracy_pct"] = business_accuracy_series(level_summary["total_actual"], level_summary["total_forecast"])
    level_summary = level_summary.rename(columns={
        "level": "Forecast Level",
        "rows": "Rows Available",
        "total_forecast": "Total Forecast",
        "total_actual": "Total Actual",
        "accuracy_pct": "Accuracy %",
    })
    st.dataframe(level_summary, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
