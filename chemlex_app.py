
"""
Chemelex Demand Forecast Cockpit
--------------------------------
A Streamlit dashboard designed to match the Chemelex-style mockup shared in the screenshots.

Input supported:
- CSV / XLSX export from Synthefy / Power BI / Excel
- Expected or auto-detected columns:
  Year
  Month
  Supply_Chain_Zone
  Plant Name
  Product 1 Category/Division Name
  Sum of Order Value in USD from LC
  Sum of Order quantity BU

Run locally:
    streamlit run app.py

requirements.txt:
    streamlit>=1.35.0
    pandas>=2.0.0
    numpy>=1.24.0
    plotly>=5.20.0
    openpyxl>=3.1.0
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chemelex Demand Forecast Cockpit",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────────────────
# CSS — Chemelex / PowerBI-like mockup style
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.block-container {
    padding-top: 0.2rem;
    padding-left: 1.1rem;
    padding-right: 1.1rem;
    max-width: 100%;
}

[data-testid="stSidebar"] {
    background: #fbfbfb;
    border-right: 1px solid #e8e8e8;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.chex-topbar {
    height: 42px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #e5e5e5;
    background: #ffffff;
    margin: -0.2rem -1.1rem 0 -1.1rem;
    padding: 0 26px;
}

.chex-logo {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #3f7182;
    font-weight: 800;
    font-size: 29px;
    letter-spacing: -1.5px;
    font-style: italic;
}

.algo-mark {
    color:#8b9aa7;
    font-size: 13px;
    font-style: normal;
    font-weight: 600;
    margin-right: -3px;
}

.chex-icons {
    display:flex;
    align-items:center;
    gap: 12px;
    color: #383838;
    font-size: 20px;
}

.chex-badge {
    position: relative;
    display:inline-block;
}
.chex-badge::after {
    content: "14";
    position:absolute;
    top:-8px;
    right:-9px;
    background:#d92f2f;
    color:white;
    border-radius:999px;
    font-size:9px;
    padding:1px 5px;
    font-weight:800;
}

.chex-tabs {
    display:flex;
    gap: 18px;
    align-items:center;
    height: 38px;
    border-bottom: 1px solid #e5e5e5;
    background:white;
    margin: 0 -1.1rem;
    padding: 0 23px;
}

.chex-tab {
    border: none;
    background: transparent;
    color: #2c2c2c;
    padding: 7px 16px;
    border-radius: 18px;
    font-size: 13px;
}

.chex-tab.active {
    border: 1.5px solid #0089c4;
    color: #2c4e60;
    background: #eef9ff;
}

.note-strip {
    height: 24px;
    background: #fff5e6;
    color: #333;
    display:flex;
    justify-content: space-between;
    align-items:center;
    border-left: 4px solid #ef8a00;
    box-shadow: 0 2px 8px rgba(0,0,0,.08);
    margin: 0 -1.1rem 18px -1.1rem;
    padding: 0 12px 0 4px;
    font-size: 12px;
}

.main-panel {
    background: #fbfbfb;
    border: 1px solid #eeeeee;
    border-radius: 0;
    padding: 14px 8px 18px 8px;
    min-height: 100vh;
}

.page-card {
    background: white;
    border: 1px solid #e7e7e7;
    border-radius: 16px;
    box-shadow: 0 1px 7px rgba(0,0,0,.045);
    padding: 14px;
    margin-bottom: 18px;
}

.page-header {
    display:flex;
    justify-content:space-between;
    align-items:center;
    border-radius: 14px;
    background: white;
    border: 1px solid #e9e9e9;
    box-shadow: 0 2px 8px rgba(0,0,0,.05);
    padding: 14px 14px;
    margin-bottom: 18px;
}

.page-title {
    font-size: 18px;
    font-weight: 650;
    color: #4c4c4c;
}

.filter-row {
    display:flex;
    gap: 14px;
    align-items:center;
    flex-wrap:wrap;
}

.filter-pill {
    border: 1px solid #e0e0e0;
    border-radius: 18px;
    height: 32px;
    padding: 0 14px;
    background:white;
    min-width: 160px;
    color: #555;
    font-size: 12px;
}

.month-pill {
    background:#f8f8f8;
    border:1px solid #d9d9d9;
    color:#50606a;
    height:34px;
    min-width:150px;
    border-radius:18px;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    font-size:12px;
    font-weight:600;
}

.circle-btn {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    background:#0085bd;
    color:white;
    font-weight:800;
    box-shadow: 0 2px 6px rgba(0,133,189,.35);
}

.kpi-grid {
    display:grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 18px;
    margin-bottom: 18px;
}

.kpi-card {
    background:white;
    border: 1px solid #e7e7e7;
    box-shadow: 0 1px 4px rgba(0,0,0,.03);
    min-height: 55px;
    padding: 10px 12px;
}

.kpi-label {
    color:#6c6c6c;
    font-size:12px;
    font-weight:500;
}
.kpi-value {
    color:#3a3a3a;
    font-size:20px;
    font-weight:600;
    margin-top:3px;
}
.kpi-sub-green {
    float:right;
    color:#16a34a;
    font-size:11px;
    margin-top: 6px;
}

.chart-title {
    color:#666;
    font-size:13px;
    font-weight:650;
    margin-bottom:8px;
}

.soft-table {
    width:100%;
    border-collapse: collapse;
    font-size:12px;
    color:#535353;
}
.soft-table thead th {
    background:#eeeeee;
    color:#555;
    font-weight:650;
    padding:10px 12px;
    border-right: 1px solid #e1e1e1;
    text-align:left;
}
.soft-table tbody td {
    padding:8px 12px;
    border-top: 1px solid #ececec;
    border-right: 1px solid #f0f0f0;
}
.soft-table tbody tr:hover td {
    background:#f8fbfd;
}

.driver-table {
    border:1px solid #ddd;
    border-radius: 16px;
    overflow:hidden;
    background:white;
}
.driver-row {
    display:grid;
    grid-template-columns: 24px 1.25fr .78fr .48fr;
    align-items:center;
    border-bottom:1px solid #e8e8e8;
    min-height:28px;
    font-size:12px;
}
.driver-row.header {
    background:#efefef;
    font-weight:600;
    min-height:48px;
}
.driver-cell {
    padding:6px 8px;
}
.driver-value {
    font-size: 17px;
    color:#555;
}
.driver-delta.pos { color:#15803d; }
.driver-delta.neg { color:#9f1239; }
.driver-delta.flat { color:#64748b; }

.check-box {
    width:13px;
    height:13px;
    border:1px solid #c7c7c7;
    border-radius:3px;
    margin:auto;
    display:grid;
    place-items:center;
    font-size:9px;
}
.check-box.on {
    background:#0089c4;
    color:white;
    border-color:#0089c4;
}

.alert-card {
    background: #fff8ed;
    border-left: 4px solid #ef8a00;
    border-radius: 10px;
    padding: 12px;
    color: #6b4c00;
    font-size: 12px;
}

.reco-card {
    background:#f6fbff;
    border:1px solid #d7ebf8;
    border-radius:14px;
    padding:12px;
}

.small-muted { color:#8b8b8b; font-size:12px; }
.caption-light { color:#9a9a9a; font-size:11px; }

hr {
    border:0;
    border-top:1px solid #ececec;
    margin:14px 0;
}

@media (max-width: 900px) {
    .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .page-header { align-items:flex-start; flex-direction:column; gap:12px; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────
MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

MONTH_SHORT = {m: m[:3] for m in MONTH_ORDER}

SAMPLE_ROWS = [
    {
        "Year": 2021,
        "Month": "April",
        "Supply_Chain_Zone": "Push / RM",
        "Plant Name": "Chemelex - RWC/UCDC",
        "Product 1 Category/Division Name": "PD / Heat Tracing Components",
        "Sum of Order Value in USD from LC": 436.01,
        "Sum of Order quantity BU": 59,
    },
    {
        "Year": 2021,
        "Month": "April",
        "Supply_Chain_Zone": "Push / MPS",
        "Plant Name": "Chemelex - Trenton",
        "Product 1 Category/Division Name": "PD / Fire and Performance Wiring",
        "Sum of Order Value in USD from LC": 2140.5,
        "Sum of Order quantity BU": 100,
    },
    {
        "Year": 2021,
        "Month": "April",
        "Supply_Chain_Zone": "BUFFER (at CODP)",
        "Plant Name": "Chemelex - Trenton",
        "Product 1 Category/Division Name": "PD / MI Heat Tracing",
        "Sum of Order Value in USD from LC": 8920,
        "Sum of Order quantity BU": 25,
    },
    {
        "Year": 2021,
        "Month": "April",
        "Supply_Chain_Zone": "Pull / Kanban",
        "Plant Name": "Chemelex - RWC/UCDC",
        "Product 1 Category/Division Name": "PD / Floor Heating",
        "Sum of Order Value in USD from LC": 180.5,
        "Sum of Order quantity BU": 12,
    },
    {
        "Year": 2021,
        "Month": "April",
        "Supply_Chain_Zone": "Push / MPS",
        "Plant Name": "Chemelex - Pharr, T",
        "Product 1 Category/Division Name": "PD / Heat Tracing Components",
        "Sum of Order Value in USD from LC": 950,
        "Sum of Order quantity BU": 50,
    },
]


COLUMN_ALIASES = {
    "year": [
        "Year", "year", "Calendar Year", "Fiscal Year", "0CALYEAR", "YEAR",
    ],
    "month": [
        "Month", "month", "Calendar Month", "Fiscal Month", "0CALMONTH", "MONTH",
    ],
    "zone": [
        "Supply_Chain_Zone", "Supply Chain Zone", "Supply_Chain", "CODP",
        "Supply Chain", "Zone", "supply_chain_zone",
    ],
    "plant": [
        "Plant Name", "Plant", "PlantName", "0PLANT", "Plant_Name", "Location",
    ],
    "category": [
        "Product 1 Category/Division Name", "Product Category", "Category",
        "Division", "SOP Family", "Product Division", "Product_1_Category_Division_Name",
    ],
    "value": [
        "Sum of Order Value in USD from LC", "Order Value", "Order Value USD",
        "Actual Value", "Sales Value", "Revenue", "Value", "sum_order_value_usd",
    ],
    "qty": [
        "Sum of Order quantity BU", "Order Quantity", "Order Qty", "Quantity",
        "Actual Qty", "BU Quantity", "Volume", "Qty", "sum_order_quantity_bu",
    ],
    "sku": ["SKU", "Material", "Material ID", "Product ID", "0MATERIAL", "Item", "Material Number"],
    "description": ["Product Description", "Description", "Material Description", "SKU Description"],
}


@dataclass
class ColumnMap:
    year: str
    month: str
    zone: str
    plant: str
    category: str
    value: str
    qty: str
    sku: Optional[str] = None
    description: Optional[str] = None


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None:
        df = pd.DataFrame()
    return df.to_csv(index=False).encode("utf-8")


def detect_col(df: pd.DataFrame, aliases: Iterable[str], required: bool = True) -> Optional[str]:
    normalized = {re.sub(r"[^a-z0-9]+", "", c.lower()): c for c in df.columns}
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]+", "", alias.lower())
        if key in normalized:
            return normalized[key]
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]+", "", alias.lower())
        for norm_col, original in normalized.items():
            if key in norm_col or norm_col in key:
                return original
    if required:
        raise ValueError(f"Could not detect required column. Tried: {aliases}")
    return None


def build_column_map(df: pd.DataFrame) -> ColumnMap:
    return ColumnMap(
        year=detect_col(df, COLUMN_ALIASES["year"]),
        month=detect_col(df, COLUMN_ALIASES["month"]),
        zone=detect_col(df, COLUMN_ALIASES["zone"]),
        plant=detect_col(df, COLUMN_ALIASES["plant"]),
        category=detect_col(df, COLUMN_ALIASES["category"]),
        value=detect_col(df, COLUMN_ALIASES["value"]),
        qty=detect_col(df, COLUMN_ALIASES["qty"]),
        sku=detect_col(df, COLUMN_ALIASES["sku"], required=False),
        description=detect_col(df, COLUMN_ALIASES["description"], required=False),
    )


def clean_year(x) -> int:
    if pd.isna(x):
        return 0
    s = str(x).replace(",", "").strip()
    try:
        return int(float(s))
    except Exception:
        m = re.search(r"(20\d{2})", s)
        return int(m.group(1)) if m else 0


def clean_month(x) -> str:
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip()
    if s.isdigit():
        idx = int(s)
        if 1 <= idx <= 12:
            return MONTH_ORDER[idx - 1]
    s_norm = s.lower()[:3]
    for m in MONTH_ORDER:
        if m.lower().startswith(s_norm):
            return m
    return s


def category_short(cat: str) -> str:
    s = str(cat).replace("Product", "").strip()
    s = re.sub(r"^PD\s*/\s*", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s)
    return s if len(s) <= 28 else s[:27] + "…"


def money(v: float) -> str:
    try:
        v = float(v)
    except Exception:
        return "$0"
    if abs(v) >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"${v/1_000:.0f}K"
    return f"${v:,.0f}"


def qty_fmt(v: float) -> str:
    try:
        v = float(v)
    except Exception:
        return "0"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"{v/1_000:.1f}K"
    return f"{v:,.0f}"


def percent(v: float) -> str:
    try:
        return f"{float(v):+.1%}"
    except Exception:
        return "0.0%"


def load_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame(SAMPLE_ROWS)

    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Please upload a CSV or Excel file.")


def standardize(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, ColumnMap]:
    df = df_raw.copy()
    cmap = build_column_map(df)

    out = pd.DataFrame()
    out["Year"] = df[cmap.year].apply(clean_year)
    out["Month"] = df[cmap.month].apply(clean_month)
    out["Supply_Chain_Zone"] = df[cmap.zone].astype(str).fillna("Unknown")
    out["Plant Name"] = df[cmap.plant].astype(str).fillna("Unknown")
    out["Product Category"] = df[cmap.category].astype(str).fillna("Unknown")
    out["Actual Value"] = pd.to_numeric(df[cmap.value], errors="coerce").fillna(0.0)
    out["Actual Qty"] = pd.to_numeric(df[cmap.qty], errors="coerce").fillna(0.0)

    if cmap.sku:
        out["SKU"] = df[cmap.sku].astype(str).fillna("")
    else:
        out["SKU"] = [
            f"MAT-{abs(hash((r['Product Category'], r['Plant Name'], r['Supply_Chain_Zone']))) % 900000:06d}"
            for _, r in out.iterrows()
        ]

    if cmap.description:
        out["Product Description"] = df[cmap.description].astype(str).fillna("")
    else:
        out["Product Description"] = out["Product Category"].apply(category_short)

    out = out[out["Year"] > 0].copy()
    if out.empty:
        raise ValueError("No usable rows after cleaning Year/Month columns.")

    out["Month_Num"] = out["Month"].map({m: i + 1 for i, m in enumerate(MONTH_ORDER)}).fillna(99).astype(int)
    out["Period"] = pd.to_datetime(
        out["Year"].astype(str) + "-" + out["Month_Num"].astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )
    out = out.dropna(subset=["Period"]).copy()

    # Deterministic demo scenario fields:
    # Forecast / Previous Year are generated from actuals when the uploaded extract only has order actuals.
    # If Synthefy later gives dedicated Forecast/Plan/AOP columns, this can be extended by mapping them.
    key_cols = ["Product Category", "Supply_Chain_Zone", "Plant Name", "Month_Num"]
    prev = (
        out[["Product Category", "Supply_Chain_Zone", "Plant Name", "Year", "Month_Num", "Actual Value", "Actual Qty"]]
        .copy()
    )
    prev["Year"] += 1
    prev = prev.rename(columns={"Actual Value": "Previous Year Value", "Actual Qty": "Previous Year Qty"})
    out = out.merge(
        prev,
        on=["Product Category", "Supply_Chain_Zone", "Plant Name", "Year", "Month_Num"],
        how="left",
    )

    # Stable pseudo-forecast by category/zone to give repeatable business mockups.
    seed = (
        out["Product Category"].astype(str) + "|" +
        out["Supply_Chain_Zone"].astype(str) + "|" +
        out["Plant Name"].astype(str)
    ).apply(lambda x: (abs(hash(x)) % 23 - 10) / 100.0)

    out["Forecast Value"] = out["Actual Value"] * (1 + seed)
    out["Forecast Qty"] = out["Actual Qty"] * (1 + seed)

    out["Previous Year Value"] = out["Previous Year Value"].fillna(out["Actual Value"] * (0.72 + seed.abs()))
    out["Previous Year Qty"] = out["Previous Year Qty"].fillna(out["Actual Qty"] * (0.72 + seed.abs()))

    # Demand plan and AOP/FOP simulation
    out["Demand Plan Value"] = out["Forecast Value"] * (1 + 0.04 * np.sign(seed))
    out["AOP Value"] = out["Previous Year Value"] * 1.12

    out["Variance Value"] = out["Actual Value"] - out["Forecast Value"]
    out["Variance %"] = np.where(out["Forecast Value"] != 0, out["Variance Value"] / out["Forecast Value"], 0)

    out["Confidence"] = (0.86 - out["Variance %"].abs().clip(0, 0.35) * 0.8).clip(0.55, 0.94)
    out["Direction"] = np.select(
        [
            out["Forecast Value"] > out["Actual Value"] * 1.08,
            out["Forecast Value"] < out["Actual Value"] * 0.92,
        ],
        ["Increase expected", "Softening expected"],
        default="Stable / watch",
    )
    out["Priority"] = np.select(
        [
            out["Variance %"].abs() > 0.25,
            out["Variance %"].abs() > 0.12,
        ],
        ["High", "Medium"],
        default="Low",
    )

    return out, cmap


def aggregate_current(df: pd.DataFrame, plant: str, zone: str, selected_year: int, selected_month: str) -> pd.DataFrame:
    dff = df.copy()
    if plant != "All":
        dff = dff[dff["Plant Name"] == plant]
    if zone != "All":
        dff = dff[dff["Supply_Chain_Zone"] == zone]
    dff = dff[(dff["Year"] == selected_year) & (dff["Month"] == selected_month)].copy()
    if dff.empty:
        dff = df.copy()
        if plant != "All":
            dff = dff[dff["Plant Name"] == plant]
        if zone != "All":
            dff = dff[dff["Supply_Chain_Zone"] == zone]
        # fallback latest period
        latest = dff["Period"].max()
        dff = dff[dff["Period"] == latest].copy()
    return dff


def category_summary(dff: pd.DataFrame) -> pd.DataFrame:
    if dff.empty:
        return pd.DataFrame(columns=["Product Category", "Forecast Value", "Actual Value", "Previous Year Value", "Variance Value", "Variance %"])
    g = (
        dff.groupby("Product Category", as_index=False)
        .agg(
            Forecast_Value=("Forecast Value", "sum"),
            Actual_Value=("Actual Value", "sum"),
            Previous_Year_Value=("Previous Year Value", "sum"),
            Actual_Qty=("Actual Qty", "sum"),
            Forecast_Qty=("Forecast Qty", "sum"),
            SKU_Count=("SKU", "nunique"),
        )
    )
    g["Variance_Value"] = g["Actual_Value"] - g["Forecast_Value"]
    g["Variance_%"] = np.where(g["Forecast_Value"] != 0, g["Variance_Value"] / g["Forecast_Value"], 0)
    g["Category Short"] = g["Product Category"].apply(category_short)
    return g.sort_values("Actual_Value", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def plot_category_bars(g: pd.DataFrame) -> go.Figure:
    x = g["Category Short"].tolist()
    fig = go.Figure()
    fig.add_bar(x=x, y=g["Forecast_Value"], name="Forecast", marker_color="#0486bd")
    fig.add_bar(x=x, y=g["Actual_Value"], name="Actual", marker_color="#32ad52")
    fig.add_bar(x=x, y=g["Previous_Year_Value"], name="Previous Year", marker_color="#ef8a00")
    fig.update_layout(
        barmode="group",
        height=300,
        margin=dict(l=10, r=10, t=22, b=22),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", x=0.52, y=1.18, xanchor="center"),
        font=dict(size=11, color="#5b5b5b"),
    )
    fig.update_yaxes(tickprefix="$", tickformat="~s", gridcolor="#eeeeee", zeroline=False)
    fig.update_xaxes(tickangle=0)
    return fig


def plot_variance_line(g: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=g["Category Short"],
        y=g["Variance_Value"].abs(),
        mode="lines",
        name="Variance",
        line=dict(color="#ff6b66", width=2),
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=22, b=22),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        font=dict(size=11, color="#5b5b5b"),
    )
    fig.update_yaxes(tickprefix="$", tickformat="~s", gridcolor="#eeeeee", zeroline=False)
    return fig


def plot_forecast_analysis(df: pd.DataFrame, category: str, zone: str, plant: str) -> go.Figure:
    dff = df.copy()
    if category != "All":
        dff = dff[dff["Product Category"] == category]
    if zone != "All":
        dff = dff[dff["Supply_Chain_Zone"] == zone]
    if plant != "All":
        dff = dff[dff["Plant Name"] == plant]

    monthly = (
        dff.groupby("Period", as_index=False)
        .agg(
            Actual=("Actual Qty", "sum"),
            Forecast=("Forecast Qty", "sum"),
            Demand_Plan=("Demand Plan Value", "sum"),
            AOP=("AOP Value", "sum"),
        )
        .sort_values("Period")
    )
    if monthly.empty:
        monthly = pd.DataFrame({"Period": pd.date_range("2024-01-01", periods=8, freq="MS"), "Actual": [0]*8, "Forecast": [0]*8, "Demand_Plan": [0]*8, "AOP": [0]*8})

    fig = go.Figure()
    fig.add_scatter(x=monthly["Period"], y=monthly["Actual"], mode="lines", name="Actual Demand", line=dict(color="#0078c2", width=2))
    fig.add_scatter(x=monthly["Period"], y=monthly["Forecast"], mode="lines", name="Forecasted Demand", line=dict(color="#ef8a00", width=2))
    fig.add_scatter(x=monthly["Period"], y=monthly["AOP"], mode="lines", name="AOP/FOP", line=dict(color="#32ad52", width=2, dash="dot"), fill="tozeroy", fillcolor="rgba(50,173,82,.09)")
    if len(monthly):
        today_x = monthly["Period"].iloc[int(len(monthly) * 0.78)] if len(monthly) > 1 else monthly["Period"].iloc[0]
        fig.add_vline(x=today_x, line_dash="dot", line_color="#c9c9c9")
        fig.add_annotation(x=today_x, y=monthly[["Actual", "Forecast"]].max().max(), text="Today", showarrow=False, yshift=12, font=dict(color="#999", size=10))
    fig.update_layout(
        height=390,
        margin=dict(l=12, r=12, t=28, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", y=1.1, x=0.72),
        font=dict(size=11, color="#5b5b5b"),
    )
    fig.update_xaxes(gridcolor="#eeeeee", tickformat="%d/%m/%Y", tickangle=-90)
    fig.update_yaxes(gridcolor="#eeeeee", zeroline=False)
    return fig


def plot_mini_trend(df: pd.DataFrame, category: str) -> go.Figure:
    dff = df[df["Product Category"] == category].copy() if category != "All" else df.copy()
    monthly = dff.groupby("Period", as_index=False).agg(Actual=("Actual Value", "sum"), Forecast=("Forecast Value", "sum")).sort_values("Period")
    fig = go.Figure()
    fig.add_scatter(x=monthly["Period"], y=monthly["Actual"], mode="lines+markers", name="Actual", line=dict(color="#32ad52", width=2))
    fig.add_scatter(x=monthly["Period"], y=monthly["Forecast"], mode="lines+markers", name="Forecast", line=dict(color="#0486bd", width=2))
    fig.update_layout(height=210, margin=dict(l=8, r=8, t=22, b=12), paper_bgcolor="white", plot_bgcolor="white", legend=dict(orientation="h"))
    fig.update_yaxes(tickprefix="$", tickformat="~s", gridcolor="#eeeeee", zeroline=False)
    fig.update_xaxes(gridcolor="#f3f3f3")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def render_chrome(active: str):
    st.markdown(
        """
<div class="chex-topbar">
  <div class="chex-logo"><span class="algo-mark">algo8</span>chemelex</div>
  <div class="chex-icons"><span class="chex-badge">🔔</span><span>⚙️</span><span>👤⌄</span></div>
</div>
""",
        unsafe_allow_html=True,
    )
    material_active = "active" if active == "Material Forecast" else ""
    analysis_active = "active" if active == "Forecast Analysis" else ""
    st.markdown(
        f"""
<div class="chex-tabs">
  <span class="chex-tab {material_active}">Material Forecast</span>
  <span class="chex-tab {analysis_active}">Forecast Analysis</span>
</div>
<div class="note-strip">
  <div>18/11/2023 14:23:51&nbsp;&nbsp;&nbsp; Lorem Ipsum dolor sit amet&nbsp; 🔗</div>
  <div>◀ &nbsp;1 of 14&nbsp; ▶ &nbsp;✖</div>
</div>
""",
        unsafe_allow_html=True,
    )


def page_header(title: str, filters_html: str):
    st.markdown(
        f"""
<div class="page-header">
  <div class="page-title">{title}</div>
  <div class="filter-row">{filters_html}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: Optional[str] = None) -> str:
    sub_html = f'<span class="kpi-sub-green">{sub}</span>' if sub else ""
    return f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{sub_html}</div>'


def render_kpis(cards: List[Tuple[str, str, Optional[str]]]):
    st.markdown(
        '<div class="kpi-grid">' + "".join(kpi_card(a, b, c) for a, b, c in cards) + "</div>",
        unsafe_allow_html=True,
    )


def select_value(label: str, options: List[str], key: str) -> str:
    return st.selectbox(label, options, index=0, key=key, label_visibility="collapsed")


def driver_panel(metrics: Dict[str, Tuple[str, str, str]]):
    rows = []
    for i, (name, (value, delta, tone)) in enumerate(metrics.items()):
        on = "on" if i in (0, 4) else ""
        delta_cls = "pos" if "+" in delta else "neg" if "-" in delta else "flat"
        color = "#e6f4ea" if tone == "green" else "#fde8e8" if tone == "red" else "#fff7e6" if tone == "amber" else "#f7f7f7"
        rows.append(
            f"""
<div class="driver-row">
  <div class="driver-cell"><div class="check-box {on}">{'✓' if on else ''}</div></div>
  <div class="driver-cell">{name}</div>
  <div class="driver-cell driver-value" style="background:{color};">{value}</div>
  <div class="driver-cell driver-delta {delta_cls}">{delta}</div>
</div>
"""
        )
    st.markdown(
        f"""
<div class="driver-table">
  <div class="driver-row header">
    <div></div><div class="driver-cell">Fluctuation Drivers</div>
    <div class="driver-cell">Apr 30, 2026-<br>May 07, 2026</div><div></div>
  </div>
  {''.join(rows)}
</div>
""",
        unsafe_allow_html=True,
    )


def render_table(df: pd.DataFrame, max_rows: int = 12):
    show = df.head(max_rows).copy()
    html = show.to_html(index=False, classes="soft-table", escape=False)
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("📥 Upload Synthefy / Excel Output")
    uploaded = st.file_uploader(
        "Upload CSV or XLSX with Year, Month, Supply Chain Zone, Plant, Category, Order Value and Quantity",
        type=["csv", "xlsx", "xls"],
        key="input_file",
    )
    st.caption("If no file is uploaded, the app uses the sample structure you shared.")
    st.divider()
    st.markdown("**Expected sample columns:**")
    st.code(
        "Year, Month, Supply_Chain_Zone, Plant Name,\n"
        "Product 1 Category/Division Name,\n"
        "Sum of Order Value in USD from LC,\n"
        "Sum of Order quantity BU",
        language="text",
    )

try:
    raw = load_file(uploaded)
    data, cmap = standardize(raw)
except Exception as e:
    st.error(f"Could not load/standardize file: {e}")
    st.stop()

# Global selectors based on available data
years = sorted([int(y) for y in data["Year"].dropna().unique() if y])
selected_year_default = years[-1] if years else 2021
months_for_year = data.loc[data["Year"] == selected_year_default, "Month"].dropna().unique().tolist()
month_options = [m for m in MONTH_ORDER if m in months_for_year] or sorted(data["Month"].dropna().unique().tolist())

# Main nav
view = st.radio(
    "View",
    ["Material Forecast", "Forecast Analysis"],
    horizontal=True,
    label_visibility="collapsed",
)

render_chrome(view)

plants = ["All"] + sorted(data["Plant Name"].dropna().unique().tolist())
zones = ["All"] + sorted(data["Supply_Chain_Zone"].dropna().unique().tolist())
categories = ["All"] + sorted(data["Product Category"].dropna().unique().tolist())

if view == "Material Forecast":
    # Header controls
    c0, c1, c2, c3, c4 = st.columns([4.8, 1.5, 1.5, .4, .9])
    with c0:
        st.markdown('<div class="page-title" style="margin: 17px 0 6px 12px;">Material Forecast</div>', unsafe_allow_html=True)
    with c1:
        plant = st.selectbox("Plant", plants, key="mf_plant", label_visibility="collapsed")
    with c2:
        zone = st.selectbox("Region", zones, key="mf_zone", label_visibility="collapsed")
    with c3:
        st.markdown('<div class="circle-btn">‹</div>', unsafe_allow_html=True)
    with c4:
        selected_month = st.selectbox("Month", month_options, key="mf_month", label_visibility="collapsed")
    dff = aggregate_current(data, plant, zone, selected_year_default, selected_month)
    g = category_summary(dff)

    active_skus = int(dff["SKU"].nunique())
    total_forecast = dff["Forecast Value"].sum()
    total_actual = dff["Actual Value"].sum()
    variance_pct = (total_actual - total_forecast) / total_forecast if total_forecast else 0

    render_kpis([
        ("Active SKUs", f"{active_skus:,.0f}", None),
        ("Total Forecast", money(total_forecast), None),
        ("Total Actual", money(total_actual), None),
        ("Variance %", f"{variance_pct:.1%}", None),
    ])

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="page-card"><div class="chart-title">Categories</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_category_bars(g), use_container_width=True, key="mf_category_bars")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown(
            f'<div class="page-card"><div style="display:flex; justify-content:space-between;"><div class="chart-title">Variance Analysis</div><div style="background:#ffd9d9; padding:7px 15px; border-radius:5px; color:#8d5c5c;">Variance: {variance_pct:.1%}</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(plot_variance_line(g), use_container_width=True, key="mf_variance_line")
        st.markdown("</div>", unsafe_allow_html=True)

    sku_table = dff.copy()
    sku_table["Category"] = sku_table["Product Category"].apply(category_short)
    sku_table["Sub-Category"] = sku_table["Supply_Chain_Zone"]
    sku_table["Forecast"] = sku_table["Forecast Value"].apply(lambda x: f"${x:,.0f}")
    sku_table["Actual"] = sku_table["Actual Value"].apply(lambda x: f"${x:,.0f}")
    sku_table["Previous Year"] = sku_table["Previous Year Value"].apply(lambda x: f"${x:,.0f}")
    sku_table = sku_table[["SKU", "Category", "Sub-Category", "Product Description", "Forecast", "Actual", "Previous Year"]]

    st.markdown(
        f"""
<div class="page-card">
  <div style="display:flex; justify-content:space-between;">
    <div><div class="chart-title">SKU Wise Material Forecast</div>
    <div class="caption-light">{len(sku_table):,.0f} items • Updated 3 minutes ago</div></div>
    <div class="circle-btn">↓</div>
  </div>
""",
        unsafe_allow_html=True,
    )
    render_table(sku_table, max_rows=18)
    st.download_button(
        "⬇️ Download SKU material forecast",
        data=to_csv_bytes(sku_table),
        file_name="chemelex_material_forecast.csv",
        mime="text/csv",
        key="mf_download",
    )
    st.markdown("</div>", unsafe_allow_html=True)

else:
    # Forecast Analysis
    c0, c1, c2, c3, c4, c5 = st.columns([2.3, 1.25, 1.25, 1.25, 1.3, .35])
    with c0:
        st.markdown('<div class="page-title" style="margin: 17px 0 6px 12px;">Forecast Analysis</div>', unsafe_allow_html=True)
    with c1:
        plant = st.selectbox("Plant", plants, key="fa_plant", label_visibility="collapsed")
    with c2:
        zone = st.selectbox("Region", zones, key="fa_zone", label_visibility="collapsed")
    with c3:
        category = st.selectbox("SKU", categories, key="fa_category", label_visibility="collapsed")
    with c4:
        date_range = st.selectbox("Date range", ["01/10/2024 - 07/11/2024", "Apr 30, 2026 - May 07, 2026", "January, 2025"], key="fa_range", label_visibility="collapsed")
    with c5:
        st.markdown('<div class="circle-btn">↓</div>', unsafe_allow_html=True)

    dff = data.copy()
    if plant != "All":
        dff = dff[dff["Plant Name"] == plant]
    if zone != "All":
        dff = dff[dff["Supply_Chain_Zone"] == zone]
    if category != "All":
        dff = dff[dff["Product Category"] == category]

    baseline_qty = dff["Previous Year Qty"].sum()
    final_forecast = dff["Forecast Qty"].sum()
    total_impact = final_forecast - baseline_qty
    variance = total_impact / baseline_qty if baseline_qty else 0

    render_kpis([
        ("Baseline", f"{qty_fmt(baseline_qty)} FT", None),
        ("Total Impact", f"{qty_fmt(total_impact)} FT", None),
        ("Final Forecast", f"{qty_fmt(final_forecast)} FT", f"{variance:.1%} variance"),
        ("Visible Materials", f"{dff['SKU'].nunique():,.0f}", None),
    ])

    driver_metrics = {
        "Recent Demand Trend": (f"{qty_fmt(dff['Actual Qty'].tail(12).mean() if len(dff) else 0)} FT", "+6%", "green"),
        "Demand Volatility": (f"{(dff['Actual Qty'].std() / dff['Actual Qty'].mean()) if dff['Actual Qty'].mean() else 0:.2f}", "+0.14", "amber"),
        "Commercial Mix": (f"{(dff['Supply_Chain_Zone'].str.contains('Push', case=False).mean() if len(dff) else 0):.0%}", "+6%", "green"),
        "Intercompany Mix": (f"{(dff['Supply_Chain_Zone'].str.contains('BUFFER|CODP', case=False, regex=True).mean() if len(dff) else 0):.0%}", "-6%", "red"),
        "Korea Share": ("36%", "+8%", "red"),
        "China Share": ("17%", "+8%", "green"),
        "Top Order Contribution": (f"{(dff.groupby('Product Category')['Actual Value'].sum().max() / dff['Actual Value'].sum()) if dff['Actual Value'].sum() else 0:.0%}", "-6%", "amber"),
        "Top Customer Share": ("32%", "-4%", "neutral"),
        "Confirmed Backlog": (f"{qty_fmt(dff['Actual Qty'].sum() * .18)} FT", "", "green"),
        "Pipeline Impact": (f"+{qty_fmt(abs(total_impact) * .35)} FT", "", "green"),
        "Manual Override Impact": (f"+{qty_fmt(abs(total_impact) * .18)} FT", "", "green"),
    }

    left, right = st.columns([1.05, 3.2])
    with left:
        driver_panel(driver_metrics)
        st.markdown(
            """
<div class="alert-card" style="margin-top:14px;">
  <b>Planner note:</b> Use this view to explain why the forecast changed — not only that it changed.
  The final version should connect these drivers to backlog, intercompany demand, Korea/China mix,
  confirmed pipeline and manual overrides.
</div>
""",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
<div class="page-card">
  <div style="display:flex; justify-content:flex-end; gap:18px; align-items:center;">
    <span style="display:flex;gap:6px;align-items:center;font-size:12px;color:#555;"><span style="height:13px;width:13px;background:#ef8a00;border-radius:3px;display:inline-block;"></span>Forecasted Demand</span>
    <span style="font-size:12px;color:#aaa;">◻ Today</span>
  </div>
""",
            unsafe_allow_html=True,
        )
        st.plotly_chart(plot_forecast_analysis(data, category, zone, plant), use_container_width=True, key="fa_analysis_chart")
        st.markdown("</div>", unsafe_allow_html=True)

    # Recommended action + watchlist
    cat_watch = category_summary(dff if not dff.empty else data)
    if not cat_watch.empty:
        cat_watch["Abs Gap"] = cat_watch["Variance_Value"].abs()
        cat_watch["Priority"] = np.select([cat_watch["Variance_%"].abs() > .25, cat_watch["Variance_%"].abs() > .12], ["High", "Medium"], default="Low")
        cat_watch["Recommendation"] = np.where(
            cat_watch["Variance_Value"] < 0,
            "Demand plan may be above actual run-rate — validate with Sales before committing supply.",
            "Actuals are running above forecast — check backlog/pipeline and adjust short-term plan."
        )
        watch = cat_watch.sort_values("Abs Gap", ascending=False).head(10).copy()
        watch_display = pd.DataFrame({
            "SOP / Product Family": watch["Product Category"].apply(category_short),
            "Actual": watch["Actual_Value"].apply(money),
            "AI Forecast": watch["Forecast_Value"].apply(money),
            "Demand Plan": watch["Actual_Value"].mul(1.05).apply(money),
            "AOP / FOP": watch["Previous_Year_Value"].mul(1.12).apply(money),
            "Gap %": watch["Variance_%"].apply(lambda x: f"{x:.1%}"),
            "Priority": watch["Priority"],
            "Recommendation": watch["Recommendation"],
        })
    else:
        watch_display = pd.DataFrame()

    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown('<div class="page-card"><div class="chart-title">Gap Watchlist — What should planner review first?</div>', unsafe_allow_html=True)
        render_table(watch_display, max_rows=10)
        st.download_button(
            "⬇️ Download visible watchlist",
            data=to_csv_bytes(watch_display),
            file_name="chemelex_gap_watchlist.csv",
            mime="text/csv",
            key="fa_watch_download",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        top_cat = watch_display.iloc[0]["SOP / Product Family"] if not watch_display.empty else "selected family"
        st.markdown(
            f"""
<div class="page-card">
  <div class="chart-title">AI Planner Query Concept</div>
  <div class="reco-card">
    <div class="small-muted">Planner asks</div>
    <h3 style="margin:5px 0 8px 0;">Do you think my demand is going to change in the next 3 months?</h3>
    <p style="font-size:13px; color:#4d5b65;">
      For <b>{top_cat}</b>, the cockpit compares current actual run-rate, AI forecast,
      demand plan, and AOP/FOP. The planner should focus on high-gap families first,
      then confirm whether the variance is driven by trend, backlog, intercompany demand,
      Korea/China mix, or manual override.
    </p>
  </div>
  <hr>
  <div class="chart-title">Data used by the mock</div>
  <ul style="font-size:12px;color:#555;line-height:1.8;">
    <li>Historical order value and BU quantity</li>
    <li>Plant and supply-chain zone hierarchy</li>
    <li>Product category / SOP family hierarchy</li>
    <li>Generated forecast, demand plan and AOP/FOP scenario layer</li>
  </ul>
</div>
""",
            unsafe_allow_html=True,
        )

# Footer data contract
with st.expander("Data contract / column mapping used by this mockup", expanded=False):
    st.write("Detected columns:")
    st.json({
        "Year": cmap.year,
        "Month": cmap.month,
        "Supply Chain Zone": cmap.zone,
        "Plant": cmap.plant,
        "Product Category / Division": cmap.category,
        "Order Value": cmap.value,
        "Order Quantity": cmap.qty,
        "SKU": cmap.sku or "generated from category + plant + zone",
        "Product Description": cmap.description or "generated from category",
    })
    st.dataframe(data.head(20), use_container_width=True, hide_index=True)
