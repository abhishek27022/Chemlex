"""
Chemelex Demand Forecast Cockpit · Streamlit App
================================================
Purpose
-------
A polished, user-centric demo cockpit for the Chemelex demand forecasting call.
It turns the mock forecast cockpit data into an executive + planner workflow:
  1) Demand reconciliation overview
  2) Actual vs Forecast vs Demand Plan vs FOP comparison
  3) SOP family / material drilldown
  4) Top gap watchlist
  5) Next 3-month outlook
  6) AI planner query concept
  7) Data lineage + confidence + validation questions

How to run
----------
streamlit run app.py

Recommended requirements.txt
----------------------------
streamlit>=1.35.0
pandas>=2.0.0
plotly>=5.20.0
numpy>=1.24.0

Data behavior
-------------
- If Chemelex_Forecast_Cockpit_Mock.html exists in repo root, the app extracts const DATA from it.
- Otherwise, it uses a small embedded fallback dataset so the app never fails during demo.
- The sidebar also supports uploading the HTML mock file directly.

Notes
-----
The mock file explicitly says it uses synthetic mock data. The app keeps that visible.
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Chemelex · Demand Forecast Cockpit",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# DESIGN SYSTEM
# =============================================================================
st.markdown(
    """
<style>
:root {
  --bg: #070B14;
  --panel: #0F172A;
  --panel-2: #111827;
  --panel-3: #1E293B;
  --border: rgba(148, 163, 184, 0.18);
  --text: #F8FAFC;
  --muted: #94A3B8;
  --blue: #38BDF8;
  --green: #22C55E;
  --amber: #F59E0B;
  --red: #EF4444;
  --violet: #8B5CF6;
}

html, body, [class*="css"] {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.block-container {
  max-width: 1540px;
  padding-top: 1rem;
  padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0F172A 0%, #07111F 55%, #050816 100%);
  border-right: 1px solid rgba(148, 163, 184, 0.12);
}

section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span {
  color: #D1D5DB !important;
}

[data-testid="stMetric"] {
  background: linear-gradient(180deg, rgba(255,255,255,0.075), rgba(255,255,255,0.025));
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 20px;
  padding: 16px 18px;
  min-height: 122px;
  box-shadow: 0 18px 60px rgba(0,0,0,0.18);
}

[data-testid="stMetric"] label { color: #CBD5E1 !important; }
[data-testid="stMetricValue"] { font-size: 1.85rem; font-weight: 850; }
[data-testid="stMetricDelta"] { font-weight: 750; }

.hero {
  position: relative;
  overflow: hidden;
  border-radius: 28px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  padding: 26px 28px;
  background:
    radial-gradient(circle at 10% 10%, rgba(56,189,248,0.22), transparent 30%),
    radial-gradient(circle at 80% 20%, rgba(139,92,246,0.20), transparent 30%),
    radial-gradient(circle at 70% 95%, rgba(34,197,94,0.13), transparent 28%),
    linear-gradient(135deg, #0F172A 0%, #111827 45%, #07111F 100%);
  box-shadow: 0 22px 70px rgba(0,0,0,0.24);
  margin-bottom: 18px;
}
.hero h1 {
  color: #F8FAFC;
  font-size: 2.2rem;
  line-height: 1.1;
  margin: 0 0 8px 0;
  letter-spacing: -0.04em;
  font-weight: 900;
}
.hero p {
  color: #CBD5E1;
  max-width: 940px;
  font-size: 0.98rem;
  line-height: 1.55;
}
.hero-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 20px;
}
.hero-tile {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(148,163,184,0.18);
  border-radius: 18px;
  padding: 14px 16px;
}
.hero-tile .label { color:#94A3B8; font-size:0.75rem; text-transform:uppercase; letter-spacing:.08em; }
.hero-tile .value { color:#F8FAFC; font-size:1.25rem; font-weight:850; margin-top:4px; }
.hero-tile .hint { color:#CBD5E1; font-size:0.78rem; margin-top:3px; }

.section-card {
  background: rgba(15, 23, 42, 0.66);
  border: 1px solid rgba(148, 163, 184, 0.14);
  border-radius: 24px;
  padding: 18px 18px;
  margin-bottom: 18px;
  box-shadow: 0 16px 60px rgba(0,0,0,0.12);
}
.section-card h3 {
  color: #F8FAFC;
  margin: 0 0 6px 0;
  font-weight: 850;
  letter-spacing: -0.02em;
}
.section-card p {
  color: #94A3B8;
  margin: 0;
}

.pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: 999px;
  padding: 5px 11px;
  font-size: 0.74rem;
  font-weight: 800;
  border: 1px solid rgba(255,255,255,0.12);
}
.pill-red { background:#7F1D1D; color:white; }
.pill-amber { background:#92400E; color:white; }
.pill-green { background:#065F46; color:white; }
.pill-blue { background:#075985; color:white; }
.pill-violet { background:#581C87; color:white; }
.pill-slate { background:#334155; color:white; }

.callout {
  border-radius: 20px;
  padding: 16px 18px;
  border: 1px solid rgba(148,163,184,0.16);
  background: rgba(255,255,255,0.045);
  color: #E5E7EB;
}
.callout strong { color:#F8FAFC; }
.callout small { color:#94A3B8; }

.ai-box {
  border-radius: 24px;
  background:
    radial-gradient(circle at top left, rgba(56,189,248,0.18), transparent 35%),
    linear-gradient(135deg, rgba(15,23,42,0.92), rgba(30,41,59,0.66));
  border: 1px solid rgba(56,189,248,0.28);
  padding: 20px;
}
.ai-answer {
  border-left: 4px solid #38BDF8;
  background: rgba(56,189,248,0.06);
  border-radius: 16px;
  padding: 14px 16px;
  color:#E5E7EB;
}

hr { border-color: rgba(148,163,184,0.12) !important; }

@media (max-width: 1000px) {
  .hero-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# DATA LOADING
# =============================================================================
FALLBACK_DATA: Dict[str, Any] = {
    "meta": {"as_of": "FY26 P3 Close (fallback sample)", "currency": "USD", "company": "Chemelex"},
    "totals": {
        "ly_actual": 156_869_590.35,
        "ai_forecast": 171_936_662.30,
        "demand_plan": 182_224_672.13,
        "fop": 170_028_092.56,
        "fy26_actual": 169_970_523.20,
    },
    "streams": [
        {"name": "1. Product (SDM)", "ly": 102_242_286.91, "ai": 116_355_233.43, "dp": 120_208_518.93, "fop": 114_121_775.92, "actual": 114_987_096.28},
        {"name": "2. Project MRO (PJM)", "ly": 30_930_523.96, "ai": 32_409_509.07, "dp": 34_225_469.68, "fop": 30_162_445.61, "actual": 30_503_747.24},
        {"name": "3. Project Major (PJM)", "ly": 21_162_665.78, "ai": 20_262_182.61, "dp": 24_770_718.07, "fop": 22_771_437.71, "actual": 21_642_251.76},
        {"name": "4. Project Mega (PJM)", "ly": 2_534_113.68, "ai": 2_909_737.16, "dp": 3_019_965.43, "fop": 2_972_433.30, "actual": 2_837_427.92},
    ],
    "sop_families": [],
}


def extract_data_from_html_text(html_text: str) -> Dict[str, Any]:
    marker = "const DATA = "
    if marker not in html_text:
        raise ValueError("Could not find `const DATA =` in the uploaded HTML mock.")

    start = html_text.index(marker) + len(marker)
    level = 0
    in_string = False
    escape = False
    end = None

    for idx, ch in enumerate(html_text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            level += 1
        elif ch == "}":
            level -= 1
            if level == 0:
                end = idx + 1
                break

    if end is None:
        raise ValueError("Could not parse DATA JSON from HTML mock.")

    return json.loads(html_text[start:end])


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


@st.cache_data(show_spinner=False)
def load_data_from_html_bytes(file_bytes: bytes) -> Dict[str, Any]:
    html_text = file_bytes.decode("utf-8", errors="ignore")
    return extract_data_from_html_text(html_text)


@st.cache_data(show_spinner=False)
def load_data_from_repo() -> Tuple[Dict[str, Any], str]:
    candidates = [
        "Chemelex_Forecast_Cockpit_Mock.html",
        "chemelex_forecast_cockpit_mock.html",
        "forecast_cockpit_mock.html",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            html_text = read_text_file(candidate)
            return extract_data_from_html_text(html_text), candidate
    return FALLBACK_DATA, "embedded fallback sample"


# =============================================================================
# TRANSFORMATION HELPERS
# =============================================================================
def fmt_money(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    v = float(v)
    sign = "-" if v < 0 else ""
    v_abs = abs(v)
    if v_abs >= 1_000_000_000:
        return f"{sign}${v_abs / 1_000_000_000:.1f}B"
    if v_abs >= 1_000_000:
        return f"{sign}${v_abs / 1_000_000:.1f}M"
    if v_abs >= 1_000:
        return f"{sign}${v_abs / 1_000:.1f}K"
    return f"{sign}${v_abs:,.0f}"


def fmt_pct(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v * 100:+.1f}%"


def raw_pct(v: float) -> float:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 0.0
    return float(v) * 100


def make_stream_df(data: Dict[str, Any]) -> pd.DataFrame:
    streams = pd.DataFrame(data.get("streams", []))
    if streams.empty:
        return pd.DataFrame(columns=["stream", "ly", "ai", "dp", "fop", "actual"])
    streams = streams.rename(columns={"name": "stream"})
    streams["ai_vs_fop_gap"] = streams["ai"] - streams["fop"]
    streams["actual_vs_fop_gap"] = streams["actual"] - streams["fop"]
    streams["actual_vs_dp_gap"] = streams["actual"] - streams["dp"]
    streams["forecast_accuracy"] = 1 - (streams["actual"] - streams["ai"]).abs() / streams["actual"].replace(0, np.nan)
    streams["forecast_accuracy"] = streams["forecast_accuracy"].clip(lower=0, upper=1)
    return streams


def make_family_df(data: Dict[str, Any]) -> pd.DataFrame:
    fam = pd.DataFrame(data.get("sop_families", []))
    if fam.empty:
        # fallback family layer built from streams
        streams = make_stream_df(data)
        fam = streams.rename(columns={"stream": "name"}).copy()
        fam["stream"] = fam["name"]
        fam["confidence"] = 80
        fam["lines"] = 0
        fam["regions"] = 0
        fam["drivers"] = [{"seasonality": 0.45, "trend": 0.35, "recent_momentum": 0.20}] * len(fam)
        fam["flags"] = [[]] * len(fam)
        fam["reco_type"] = "review"
        fam["reco_text"] = "Review with sales and validate hierarchy mapping."
        fam["monthly_history"] = [[]] * len(fam)
        fam["monthly_forward"] = [[]] * len(fam)

    for col in ["ly", "ai", "dp", "fop", "actual", "confidence"]:
        if col not in fam.columns:
            fam[col] = np.nan
        fam[col] = pd.to_numeric(fam[col], errors="coerce")

    fam["ai_vs_fop_gap"] = fam["ai"] - fam["fop"]
    fam["ai_vs_dp_gap"] = fam["ai"] - fam["dp"]
    fam["actual_vs_fop_gap"] = fam["actual"] - fam["fop"]
    fam["actual_vs_dp_gap"] = fam["actual"] - fam["dp"]
    fam["abs_gap"] = fam["actual_vs_fop_gap"].abs()
    fam["gap_pct"] = fam["actual_vs_fop_gap"] / fam["fop"].replace(0, np.nan)
    fam["accuracy"] = 1 - (fam["actual"] - fam["ai"]).abs() / fam["actual"].replace(0, np.nan)
    fam["accuracy"] = fam["accuracy"].clip(lower=0, upper=1)
    fam["flag_count"] = fam.get("flags", pd.Series([[]] * len(fam))).apply(lambda x: len(x) if isinstance(x, list) else 0)
    fam["priority"] = fam.apply(assign_priority, axis=1)
    fam["direction"] = fam.apply(assign_direction, axis=1)
    return fam


def assign_priority(row: pd.Series) -> str:
    volume = float(row.get("actual", 0) or 0)
    gap_pct = abs(float(row.get("gap_pct", 0) or 0))
    flag_count = int(row.get("flag_count", 0) or 0)
    confidence = float(row.get("confidence", 0) or 0)
    if (volume > 2_000_000 and gap_pct > 0.10) or flag_count >= 2 or confidence < 60:
        return "High"
    if gap_pct > 0.07 or flag_count == 1 or confidence < 75:
        return "Medium"
    return "Low"


def assign_direction(row: pd.Series) -> str:
    ai = float(row.get("ai", 0) or 0)
    fop = float(row.get("fop", 0) or 0)
    if fop == 0:
        return "Stable"
    delta = (ai - fop) / abs(fop)
    if delta > 0.05:
        return "Increasing"
    if delta < -0.05:
        return "Decreasing"
    return "Stable"


def score_health(fam_df: pd.DataFrame) -> int:
    if fam_df.empty:
        return 0
    high = (fam_df["priority"] == "High").mean()
    med = (fam_df["priority"] == "Medium").mean()
    avg_conf = fam_df["confidence"].fillna(0).mean() / 100
    score = 100 - 45 * high - 18 * med + 12 * avg_conf
    return int(max(0, min(100, score)))


def confidence_bucket(v: float) -> str:
    if pd.isna(v):
        return "Unknown"
    if v >= 85:
        return "High"
    if v >= 70:
        return "Medium"
    return "Low"


def apply_filters(fam_df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    out = fam_df.copy()
    if filters.get("stream") and filters["stream"] != "All":
        out = out[out["stream"] == filters["stream"]]
    if filters.get("priority") and filters["priority"] != "All":
        out = out[out["priority"] == filters["priority"]]
    if filters.get("direction") and filters["direction"] != "All":
        out = out[out["direction"] == filters["direction"]]
    if filters.get("confidence") and filters["confidence"] != "All":
        out = out[out["confidence"].apply(confidence_bucket) == filters["confidence"]]
    if filters.get("search"):
        s = filters["search"].strip().lower()
        if s:
            out = out[out["name"].str.lower().str.contains(s, na=False) | out["stream"].str.lower().str.contains(s, na=False)]
    return out


def selected_family_record(fam_df: pd.DataFrame, name: str) -> Dict[str, Any]:
    match = fam_df[fam_df["name"] == name]
    if match.empty:
        return {}
    return match.iloc[0].to_dict()


def build_monthly_df(family: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for r in family.get("monthly_history", []) or []:
        period = r.get("period")
        series = r.get("series", "Actual")
        actual = r.get("actual")
        rows.append({"period": period, "series": series, "value": actual})

    for r in family.get("monthly_forward", []) or []:
        period = r.get("period")
        rows.append({"period": period, "series": "AI Forecast", "value": r.get("ai_forecast")})
        rows.append({"period": period, "series": "Demand Plan", "value": r.get("demand_plan")})
        rows.append({"period": period, "series": "FOP", "value": r.get("fop")})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["period", "series", "value"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def next_three_months(family: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for r in (family.get("monthly_forward", []) or [])[:3]:
        ai = float(r.get("ai_forecast", 0) or 0)
        dp = float(r.get("demand_plan", 0) or 0)
        fop = float(r.get("fop", 0) or 0)
        gap_vs_plan = ai - dp
        rows.append({
            "Period": r.get("period"),
            "AI Forecast": ai,
            "Demand Plan": dp,
            "FOP": fop,
            "Gap vs Demand Plan": gap_vs_plan,
            "Recommended Action": recommendation_from_gap(gap_vs_plan, dp),
        })
    return pd.DataFrame(rows)


def recommendation_from_gap(gap: float, base: float) -> str:
    if base == 0:
        return "Validate with sales"
    pct = gap / abs(base)
    if pct > 0.08:
        return "Review upside with sales; check supply feasibility"
    if pct < -0.08:
        return "Challenge demand plan; avoid over-build risk"
    return "No immediate action; monitor movement"


# =============================================================================
# PLOT HELPERS
# =============================================================================
def plot_theme(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=20, r=20, t=52, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.15)", zerolinecolor="rgba(148,163,184,0.18)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.15)", zerolinecolor="rgba(148,163,184,0.18)")
    return fig


def chart_reconciliation(stream_df: pd.DataFrame) -> go.Figure:
    df = stream_df.melt(
        id_vars=["stream"],
        value_vars=["ly", "actual", "ai", "dp", "fop"],
        var_name="series",
        value_name="value",
    )
    labels = {"ly": "LY Actual", "actual": "FY26 Actual", "ai": "AI Forecast", "dp": "Demand Plan", "fop": "FOP"}
    colors = {"LY Actual": "#94A3B8", "FY26 Actual": "#22C55E", "AI Forecast": "#38BDF8", "Demand Plan": "#F59E0B", "FOP": "#8B5CF6"}
    df["series"] = df["series"].map(labels)
    fig = px.bar(
        df,
        x="stream",
        y="value",
        color="series",
        barmode="group",
        title="Demand Reconciliation · Actual vs Forecast vs Demand Plan vs FOP",
        color_discrete_map=colors,
        custom_data=["series"],
    )
    fig.update_traces(hovertemplate="%{customdata[0]}<br>%{x}<br>%{y:$,.0f}<extra></extra>")
    return plot_theme(fig, 440)


def chart_gap_waterfall(totals: Dict[str, float]) -> go.Figure:
    actual = float(totals.get("fy26_actual", 0) or 0)
    ai = float(totals.get("ai_forecast", 0) or 0)
    dp = float(totals.get("demand_plan", 0) or 0)
    fop = float(totals.get("fop", 0) or 0)
    fig = go.Figure(
        go.Waterfall(
            name="Plan Bridge",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative"],
            x=["FY26 Actual", "Gap to AI", "Gap to FOP", "Gap to Demand Plan"],
            y=[actual, ai - actual, fop - ai, dp - fop],
            connector={"line": {"color": "rgba(148,163,184,0.45)"}},
            increasing={"marker": {"color": "#22C55E"}},
            decreasing={"marker": {"color": "#EF4444"}},
            totals={"marker": {"color": "#38BDF8"}},
        )
    )
    fig.update_layout(title="Executive Bridge · Actual → AI Forecast → FOP → Demand Plan")
    return plot_theme(fig, 360)


def chart_status_mix(fam_df: pd.DataFrame) -> go.Figure:
    d = fam_df["priority"].value_counts().rename_axis("Priority").reset_index(name="Count")
    colors = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#22C55E"}
    fig = px.pie(d, names="Priority", values="Count", hole=0.62, color="Priority", color_discrete_map=colors, title="Watchlist Risk Mix")
    fig.update_traces(textinfo="percent+label", marker=dict(line=dict(color="#0F172A", width=2)))
    return plot_theme(fig, 340)


def chart_family_gap(fam_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    d = fam_df.sort_values("abs_gap", ascending=False).head(top_n).sort_values("abs_gap")
    fig = px.bar(
        d,
        y="name",
        x="actual_vs_fop_gap",
        orientation="h",
        color="priority",
        color_discrete_map={"High": "#EF4444", "Medium": "#F59E0B", "Low": "#22C55E"},
        title=f"Top {top_n} SOP Families by Actual vs FOP Gap",
        hover_data={"actual": ":$,.0f", "fop": ":$,.0f", "gap_pct": ":.1%", "confidence": ":.0f"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.35)")
    return plot_theme(fig, 440)


def chart_monthly(family: Dict[str, Any]) -> go.Figure:
    df = build_monthly_df(family)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="12-Month View · No monthly detail available")
        return plot_theme(fig, 360)
    color_map = {"LY Actual": "#94A3B8", "FY Actual (YTD)": "#22C55E", "AI Forecast": "#38BDF8", "Demand Plan": "#F59E0B", "FOP": "#8B5CF6"}
    fig = px.line(
        df,
        x="period",
        y="value",
        color="series",
        markers=True,
        title=f"12-Month View · {family.get('name', 'Selected Family')}",
        color_discrete_map=color_map,
    )
    fig.update_traces(hovertemplate="%{fullData.name}<br>%{x}<br>%{y:$,.0f}<extra></extra>")
    return plot_theme(fig, 420)


def chart_drivers(family: Dict[str, Any]) -> go.Figure:
    drivers = family.get("drivers") or {}
    if not drivers:
        drivers = {"seasonality": 0.45, "trend": 0.35, "recent_momentum": 0.20}
    d = pd.DataFrame([{"Driver": k.replace("_", " ").title(), "Weight": v} for k, v in drivers.items()])
    fig = px.bar(d, x="Weight", y="Driver", orientation="h", title="What's Driving the AI Forecast", color="Weight", color_continuous_scale=["#0EA5E9", "#8B5CF6"])
    fig.update_layout(coloraxis_showscale=False)
    return plot_theme(fig, 300)


def chart_treemap(fam_df: pd.DataFrame) -> go.Figure:
    d = fam_df.copy()
    d["size"] = d["actual"].fillna(0).clip(lower=0)
    d["gap_pct_abs"] = d["gap_pct"].abs().fillna(0)
    fig = px.treemap(
        d,
        path=["stream", "name"],
        values="size",
        color="gap_pct_abs",
        color_continuous_scale=["#22C55E", "#F59E0B", "#EF4444"],
        title="Hierarchy View · Stream → SOP Family · Color = Gap Intensity",
        hover_data={"actual": ":$,.0f", "ai": ":$,.0f", "fop": ":$,.0f", "confidence": ":.0f"},
    )
    return plot_theme(fig, 520)


def chart_next_three(family: Dict[str, Any]) -> go.Figure:
    d = next_three_months(family)
    if d.empty:
        fig = go.Figure()
        fig.update_layout(title="Next 3-Month Outlook · No forward view available")
        return plot_theme(fig, 320)
    melted = d.melt(id_vars=["Period"], value_vars=["AI Forecast", "Demand Plan", "FOP"], var_name="Series", value_name="Value")
    fig = px.bar(melted, x="Period", y="Value", color="Series", barmode="group", title="Next 3-Month Outlook", color_discrete_map={"AI Forecast": "#38BDF8", "Demand Plan": "#F59E0B", "FOP": "#8B5CF6"})
    fig.update_traces(hovertemplate="%{fullData.name}<br>%{x}<br>%{y:$,.0f}<extra></extra>")
    return plot_theme(fig, 330)


# =============================================================================
# TABLE DISPLAY HELPERS
# =============================================================================
def style_priority(val: str) -> str:
    if val == "High":
        return "background-color:#7F1D1D;color:#FFFFFF;font-weight:800"
    if val == "Medium":
        return "background-color:#92400E;color:#FFFFFF;font-weight:800"
    if val == "Low":
        return "background-color:#064E3B;color:#FFFFFF;font-weight:700"
    return ""


def style_direction(val: str) -> str:
    if val == "Increasing":
        return "background-color:#075985;color:#FFFFFF;font-weight:800"
    if val == "Decreasing":
        return "background-color:#581C87;color:#FFFFFF;font-weight:800"
    if val == "Stable":
        return "background-color:#334155;color:#FFFFFF;font-weight:700"
    return ""


def display_watchlist(df: pd.DataFrame, height: int = 420, key: str = "watchlist"):
    cols = ["name", "stream", "actual", "ai", "dp", "fop", "actual_vs_fop_gap", "gap_pct", "confidence", "direction", "priority", "reco_text"]
    show = df[cols].copy()
    show = show.rename(columns={
        "name": "SOP Family / Material",
        "stream": "Demand Stream",
        "actual": "FY26 Actual",
        "ai": "AI Forecast",
        "dp": "Demand Plan",
        "fop": "FOP",
        "actual_vs_fop_gap": "Actual vs FOP Gap",
        "gap_pct": "Gap %",
        "confidence": "Confidence",
        "direction": "3M Direction",
        "priority": "Priority",
        "reco_text": "Recommendation",
    })
    styler = show.style
    if hasattr(styler, "map"):
        styler = styler.map(style_priority, subset=["Priority"]).map(style_direction, subset=["3M Direction"])
    else:
        styler = styler.applymap(style_priority, subset=["Priority"]).applymap(style_direction, subset=["3M Direction"])
    st.dataframe(
        styler,
        use_container_width=True,
        hide_index=True,
        height=height,
        key=key,
        column_config={
            "FY26 Actual": st.column_config.NumberColumn(format="$ %.0f"),
            "AI Forecast": st.column_config.NumberColumn(format="$ %.0f"),
            "Demand Plan": st.column_config.NumberColumn(format="$ %.0f"),
            "FOP": st.column_config.NumberColumn(format="$ %.0f"),
            "Actual vs FOP Gap": st.column_config.NumberColumn(format="$ %.0f"),
            "Gap %": st.column_config.NumberColumn(format="%.1f%%"),
            "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
            "Recommendation": st.column_config.TextColumn(width="large"),
        },
    )


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### 📈 Chemelex Forecast Cockpit")
    st.caption("Demand review · forecast reconciliation · planner workflow")
    st.divider()

    uploaded_html = st.file_uploader("Upload cockpit HTML mock", type=["html", "htm"], help="Optional. If not uploaded, app reads Chemelex_Forecast_Cockpit_Mock.html from the repo.")

    if uploaded_html is not None:
        data = load_data_from_html_bytes(uploaded_html.getvalue())
        data_source = uploaded_html.name
    else:
        data, data_source = load_data_from_repo()

    stream_df = make_stream_df(data)
    family_df = make_family_df(data)

    st.caption(f"Source: {data_source}")
    st.caption(f"As-of: {data.get('meta', {}).get('as_of', '—')}")
    st.divider()

    st.markdown("#### Global Filters")
    stream_options = ["All"] + sorted([x for x in family_df["stream"].dropna().unique()])
    priority_options = ["All", "High", "Medium", "Low"]
    direction_options = ["All", "Increasing", "Stable", "Decreasing"]
    confidence_options = ["All", "High", "Medium", "Low", "Unknown"]

    filters = {
        "stream": st.selectbox("Demand Stream", stream_options, key="filter_stream"),
        "priority": st.selectbox("Priority", priority_options, key="filter_priority"),
        "direction": st.selectbox("3M Direction", direction_options, key="filter_direction"),
        "confidence": st.selectbox("Confidence", confidence_options, key="filter_confidence"),
        "search": st.text_input("Search SOP family / material", placeholder="e.g., MONO, NGC, Heat trace...", key="filter_search"),
    }

    filtered_family_df = apply_filters(family_df, filters)

    st.divider()
    st.markdown("#### Demo Scope")
    st.markdown(
        """
        <span class="pill pill-blue">P0</span> Reconciliation overview<br>
        <span class="pill pill-blue">P0</span> SKU/SOP family drilldown<br>
        <span class="pill pill-amber">P1</span> AI planner query concept<br>
        <span class="pill pill-amber">P1</span> Data lineage & confidence
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# HEADER
# =============================================================================
totals = data.get("totals", {})
actual = float(totals.get("fy26_actual", 0) or 0)
ai = float(totals.get("ai_forecast", 0) or 0)
dp = float(totals.get("demand_plan", 0) or 0)
fop = float(totals.get("fop", 0) or 0)
ly = float(totals.get("ly_actual", 0) or 0)
health = score_health(filtered_family_df)
forecast_accuracy = 1 - abs(actual - ai) / actual if actual else 0
forecast_accuracy = max(0, min(1, forecast_accuracy))

st.markdown(
    f"""
<div class="hero">
  <h1>Chemelex Demand Forecast Cockpit</h1>
  <p>
    A planner-first cockpit for the 6 PM demand forecasting discussion: reconcile <b>Actual</b>, <b>AI Forecast</b>,
    <b>Demand Plan</b> and <b>FOP/AOP</b>; then drill into the SOP family/materials that need commercial review.
    The goal is not to show a generic report — it is to show where the plan is misaligned and what action the planner should take next.
  </p>
  <div class="hero-grid">
    <div class="hero-tile"><div class="label">Forecast Health</div><div class="value">{health}/100</div><div class="hint">Weighted by risk mix + confidence</div></div>
    <div class="hero-tile"><div class="label">FY26 Actual</div><div class="value">{fmt_money(actual)}</div><div class="hint">YTD / current-year actual in mock</div></div>
    <div class="hero-tile"><div class="label">Actual vs FOP</div><div class="value">{fmt_money(actual - fop)}</div><div class="hint">{fmt_pct((actual - fop) / fop if fop else 0)}</div></div>
    <div class="hero-tile"><div class="label">Accuracy Signal</div><div class="value">{forecast_accuracy * 100:.1f}%</div><div class="hint">Actual vs AI forecast fit</div></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# KPI row
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("LY Actual", fmt_money(ly))
k2.metric("FY26 Actual", fmt_money(actual), delta=fmt_money(actual - ly))
k3.metric("AI Forecast", fmt_money(ai), delta=fmt_money(ai - actual))
k4.metric("Demand Plan", fmt_money(dp), delta=fmt_money(dp - actual), delta_color="inverse")
k5.metric("FOP / AOP", fmt_money(fop), delta=fmt_money(fop - actual), delta_color="inverse")
k6.metric("High Priority Families", int((filtered_family_df["priority"] == "High").sum()), delta=f"of {len(filtered_family_df)} visible", delta_color="inverse")


# =============================================================================
# TABS
# =============================================================================
tabs = st.tabs([
    "🏠 Command Center",
    "🔍 SOP Family Drilldown",
    "🚨 Gap Watchlist",
    "🤖 AI Planner Query",
    "🧭 Hierarchy & Data Readiness",
])


# ----------------------------------------------------------------------------
# TAB 1 COMMAND CENTER
# ----------------------------------------------------------------------------
with tabs[0]:
    st.markdown('<div class="section-card"><h3>What we should show in the call</h3><p>Start with a simple executive reconciliation, then drill into material/SOP family level only where the gap is meaningful. This keeps the demo at the right aggregation level while still proving SKU-level actionability.</p></div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.65, 1])
    with c1:
        st.plotly_chart(chart_reconciliation(stream_df), use_container_width=True, key="cmd_reconciliation")
    with c2:
        st.plotly_chart(chart_gap_waterfall(totals), use_container_width=True, key="cmd_gap_waterfall")

    c3, c4 = st.columns([1.15, 1])
    with c3:
        st.plotly_chart(chart_family_gap(filtered_family_df, top_n=12), use_container_width=True, key="cmd_top_gaps")
    with c4:
        st.plotly_chart(chart_status_mix(filtered_family_df), use_container_width=True, key="cmd_risk_mix")

    st.markdown("### Top planner actions")
    action_pool = filtered_family_df.sort_values(["priority", "abs_gap"], ascending=[True, False]).copy()
    priority_rank = {"High": 0, "Medium": 1, "Low": 2}
    action_pool["rank"] = action_pool["priority"].map(priority_rank).fillna(9)
    action_pool = action_pool.sort_values(["rank", "abs_gap"], ascending=[True, False]).head(8)
    display_watchlist(action_pool, height=360, key="cmd_action_table")


# ----------------------------------------------------------------------------
# TAB 2 DRILLDOWN
# ----------------------------------------------------------------------------
with tabs[1]:
    st.markdown('<div class="section-card"><h3>SOP Family / Material Drilldown</h3><p>Select one family/material and show the exact story: what happened last year, what happened YTD, what the model expects, how it compares to demand plan/FOP, and what action the planner should take.</p></div>', unsafe_allow_html=True)

    family_options = filtered_family_df.sort_values("abs_gap", ascending=False)["name"].tolist()
    if not family_options:
        family_options = family_df["name"].tolist()
    selected_name = st.selectbox("Select SOP family / material", family_options, index=0 if family_options else None, key="selected_family")
    fam = selected_family_record(family_df, selected_name) if selected_name else {}

    if fam:
        d1, d2, d3, d4, d5, d6 = st.columns(6)
        d1.metric("Family", fam.get("name", "—"))
        d2.metric("Stream", str(fam.get("stream", "—")).replace("Project ", "Proj. "))
        d3.metric("Actual", fmt_money(fam.get("actual", 0)))
        d4.metric("AI Forecast", fmt_money(fam.get("ai", 0)), delta=fmt_money(fam.get("ai", 0) - fam.get("actual", 0)))
        d5.metric("Confidence", f"{float(fam.get('confidence', 0) or 0):.0f}%")
        d6.metric("Priority", fam.get("priority", "—"))

        left, right = st.columns([1.55, 1])
        with left:
            st.plotly_chart(chart_monthly(fam), use_container_width=True, key="drill_monthly")
        with right:
            st.markdown("#### Recommendation")
            tag_class = "pill-green" if fam.get("priority") == "Low" else "pill-amber" if fam.get("priority") == "Medium" else "pill-red"
            st.markdown(
                f"""
<div class="callout">
  <span class="pill {tag_class}">{fam.get('priority', 'Review')}</span><br><br>
  <strong>{fam.get('reco_text', 'Review with demand planning and sales.')}</strong><br><br>
  <small>Direction: {fam.get('direction', 'Stable')} · Gap vs FOP: {fmt_money(fam.get('actual_vs_fop_gap', 0))} · Gap%: {fmt_pct(fam.get('gap_pct', 0))}</small>
</div>
""",
                unsafe_allow_html=True,
            )
            st.plotly_chart(chart_drivers(fam), use_container_width=True, key="drill_drivers")

        out1, out2 = st.columns([1.05, 1])
        with out1:
            st.plotly_chart(chart_next_three(fam), use_container_width=True, key="drill_next_three")
        with out2:
            st.markdown("#### Next 3-month planner view")
            next3 = next_three_months(fam)
            if next3.empty:
                st.info("No forward monthly view available for this family.")
            else:
                st.dataframe(
                    next3,
                    use_container_width=True,
                    hide_index=True,
                    key="drill_next3_table",
                    column_config={
                        "AI Forecast": st.column_config.NumberColumn(format="$ %.0f"),
                        "Demand Plan": st.column_config.NumberColumn(format="$ %.0f"),
                        "FOP": st.column_config.NumberColumn(format="$ %.0f"),
                        "Gap vs Demand Plan": st.column_config.NumberColumn(format="$ %.0f"),
                        "Recommended Action": st.column_config.TextColumn(width="large"),
                    },
                )

        st.markdown("#### Data lineage & confidence")
        lineage_cols = st.columns(4)
        lineage_cols[0].markdown(f"<div class='callout'><strong>Data used</strong><br><small>Historical actuals, current actuals, demand plan, FOP/AOP, stream/SOP family mapping.</small></div>", unsafe_allow_html=True)
        lineage_cols[1].markdown(f"<div class='callout'><strong>Confidence</strong><br><small>{float(fam.get('confidence', 0) or 0):.0f}% based on stability, gap intensity and driver mix.</small></div>", unsafe_allow_html=True)
        lineage_cols[2].markdown(f"<div class='callout'><strong>Flags</strong><br><small>{len(fam.get('flags') or [])} flags found. {(fam.get('flags') or ['No major risk flags'])[0]}</small></div>", unsafe_allow_html=True)
        lineage_cols[3].markdown(f"<div class='callout'><strong>Business lens</strong><br><small>Use Product/MRO for statistical forecast; Major/Mega should include funnel/backlog judgment.</small></div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# TAB 3 WATCHLIST
# ----------------------------------------------------------------------------
with tabs[2]:
    st.markdown('<div class="section-card"><h3>Gap Watchlist</h3><p>This is the planner workbench: sort by actual vs FOP gap, confidence, and priority. High-volume + high-gap families should be reviewed first with sales and supply planning.</p></div>', unsafe_allow_html=True)

    w1, w2, w3 = st.columns(3)
    w1.metric("Visible families", len(filtered_family_df))
    w2.metric("High priority", int((filtered_family_df["priority"] == "High").sum()))
    w3.metric("Avg confidence", f"{filtered_family_df['confidence'].mean():.0f}%" if len(filtered_family_df) else "—")

    sort_by = st.selectbox("Sort watchlist by", ["abs_gap", "gap_pct", "confidence", "actual"], format_func={"abs_gap": "Absolute Gap", "gap_pct": "Gap %", "confidence": "Confidence", "actual": "Actual"}.get, key="watch_sort")
    ascending = False if sort_by != "confidence" else True
    watch = filtered_family_df.sort_values(sort_by, ascending=ascending).head(50)
    display_watchlist(watch, height=620, key="watchlist_full")

    st.download_button("⬇️ Download visible watchlist", data=to_csv_bytes(watch), file_name="chemelex_gap_watchlist.csv", mime="text/csv")


# ----------------------------------------------------------------------------
# TAB 4 AI PLANNER QUERY
# ----------------------------------------------------------------------------
with tabs[3]:
    st.markdown('<div class="section-card"><h3>AI Planner Query Concept</h3><p>Show this as a future-state capability, not as a fully validated model. It demonstrates how a planner could ask a natural-language question and get the chart, confidence, data used and recommended action back in one place.</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="ai-box">', unsafe_allow_html=True)
    q = st.text_input("Ask the planner assistant", value="Do you think demand for the selected SOP family will change in the next 3 months?", key="ai_query")
    family_for_ai = st.selectbox("Context family", family_df.sort_values("abs_gap", ascending=False)["name"].head(25).tolist(), key="ai_family_select")
    ai_fam = selected_family_record(family_df, family_for_ai)

    if ai_fam:
        direction = ai_fam.get("direction", "Stable")
        priority = ai_fam.get("priority", "Low")
        confidence = float(ai_fam.get("confidence", 0) or 0)
        answer = (
            f"Yes — demand looks **{direction.lower()}** over the next 3 months for **{family_for_ai}**. "
            f"The current signal is **{priority} priority** with **{confidence:.0f}% confidence**. "
            f"The main reason is the gap between AI forecast, FOP and demand plan, combined with seasonality/trend drivers."
        )
        st.markdown(f"<div class='ai-answer'>{answer}</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_next_three(ai_fam), use_container_width=True, key="ai_next_three")

        a1, a2, a3 = st.columns(3)
        a1.markdown(f"<div class='callout'><strong>Data used</strong><br><small>Monthly actuals, LY actuals, AI forecast, demand plan, FOP, driver weights.</small></div>", unsafe_allow_html=True)
        a2.markdown(f"<div class='callout'><strong>Confidence</strong><br><small>{confidence:.0f}% — {'High' if confidence >= 85 else 'Medium' if confidence >= 70 else 'Low'} confidence bucket.</small></div>", unsafe_allow_html=True)
        a3.markdown(f"<div class='callout'><strong>Suggested action</strong><br><small>{ai_fam.get('reco_text', 'Review with sales and planner.')}</small></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### Suggested demo prompts")
    st.markdown(
        """
        - Will demand for this SOP family change in the next 3 months?
        - Which families are most over-planned versus AI forecast?
        - Which demand streams should be reviewed with sales before supply review?
        - Is this variance likely seasonal, trend-driven, or project-driven?
        - Which materials should I not level-load blindly this month?
        """
    )


# ----------------------------------------------------------------------------
# TAB 5 HIERARCHY & READINESS
# ----------------------------------------------------------------------------
with tabs[4]:
    st.markdown('<div class="section-card"><h3>Hierarchy & Data Readiness</h3><p>The call transcript makes one thing clear: before modelling too deeply, we need the right hierarchy layer. Product hierarchy, sales hierarchy, region hierarchy, source/SOP family and demand scale should be treated as separate but connected views.</p></div>', unsafe_allow_html=True)

    st.plotly_chart(chart_treemap(filtered_family_df), use_container_width=True, key="hierarchy_treemap")

    h1, h2 = st.columns([1, 1])
    with h1:
        st.markdown("#### Hierarchy backbone")
        hierarchy_table = pd.DataFrame([
            {"View": "Product hierarchy", "Meaning": "What product/category/family it belongs to", "Used for": "Commercial demand review"},
            {"View": "Sales / region hierarchy", "Meaning": "Which region/sales team owns it", "Used for": "Sales accountability"},
            {"View": "Source / SOP family", "Meaning": "Where/how it is made or supplied", "Used for": "Supply planning"},
            {"View": "Demand scale", "Meaning": "Product / MRO / Major / Mega", "Used for": "Forecastability logic"},
            {"View": "BOM linkage", "Meaning": "FG → components / raw material impact", "Used for": "Supply impact after demand view"},
        ])
        st.dataframe(hierarchy_table, use_container_width=True, hide_index=True, key="hierarchy_table")

    with h2:
        st.markdown("#### Questions for Neeraj / validation")
        questions = pd.DataFrame([
            {"#": 1, "Question": "Which file/table has Actual, Forecast, Demand Plan and AOP/FOP?", "Why it matters": "Prevents mixing reporting vs planning metrics"},
            {"#": 2, "Question": "Is AOP available monthly or only annual?", "Why it matters": "Determines if AOP can be plotted as a monthly line"},
            {"#": 3, "Question": "Which field is the correct Product ID / Material ID?", "Why it matters": "Defines the drilldown key"},
            {"#": 4, "Question": "Is Scale available for MRO / PRD / Major / Mega?", "Why it matters": "Separates forecastable vs funnel/project demand"},
            {"#": 5, "Question": "What level should we demo: Product ID, Sales Manager 4, Product Family, Platform, or KSC?", "Why it matters": "Avoids too-low-level noise"},
            {"#": 6, "Question": "Which 5–10 SKUs should be deep-dived in the call?", "Why it matters": "Creates a clean client-facing story"},
        ])
        st.dataframe(questions, use_container_width=True, hide_index=True, key="questions_table")

    st.markdown("#### Demand bucket logic")
    bucket_logic = pd.DataFrame([
        {"Demand Bucket": "Product / PRD", "Recommended Handling": "Time-series forecast", "Dashboard Treatment": "Show in main forecast cockpit"},
        {"Demand Bucket": "MRO", "Recommended Handling": "Time-series + sales validation", "Dashboard Treatment": "Show with confidence and review trigger"},
        {"Demand Bucket": "Major", "Recommended Handling": "Backlog/funnel + judgment", "Dashboard Treatment": "Do not force pure statistical forecast"},
        {"Demand Bucket": "Mega", "Recommended Handling": "Project visibility / funnel", "Dashboard Treatment": "Flag as project-driven demand"},
    ])
    st.dataframe(bucket_logic, use_container_width=True, hide_index=True, key="bucket_logic")


# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.caption(
    "Chemelex Forecast Cockpit · built for demand forecasting demo · "
    "mock data should be validated against BW / SAP / Demand Consensus before production use. "
    f"Rendered at {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Return a DataFrame as UTF-8 CSV bytes for Streamlit downloads."""
    if df is None:
        df = pd.DataFrame()
    return df.to_csv(index=False).encode("utf-8")


