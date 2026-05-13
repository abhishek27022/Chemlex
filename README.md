# Chemelex Demand Forecasting Cockpit

Executive view of forecast performance, demand-plan alignment, and
IBP-vs-ML head-to-head comparison.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (typically http://localhost:8501).

## Data Files

Place these files **next to `app.py`** so they auto-load on startup:

| File | Required | Purpose |
|------|----------|---------|
| `corrected_dashboard_forecast_accuracy_levelled.csv` | **Yes** | Primary forecast accuracy data (11 ML models + IBP_Manual) |
| `ibp_vs_model_comparison_2025.csv` | Recommended | Synthefy's authoritative category-month head-to-head |
| `ibp_reconciliation.csv` | Optional | IBP unit-to-COGS audit trail |
| `ibp_pipeline.md` | Optional | IBP pipeline documentation |

If `corrected_dashboard_forecast_accuracy_levelled.csv` is missing, the
visible **Upload / Replace Forecast File** widget at the top of the page
accepts CSV or Excel.

## Default View

- **Year:** 2025
- **Forecast Level:** Group (CODP Zone × Plant × Product Category)
- **Target:** `order_value_usd`
- **Split:** `split2`
- **Best model only:** ON (`is_best = 1`)
- **All months, all 5 official CODP zones, all plants, all categories selected**

## Tabs

1. **📈 Overview** — Monthly F vs A, alignment by month, category bar, WAPE by zone
2. **🎯 Forecast Accuracy** — Business table at the chosen level, CSV download
3. **🏁 IBP vs Model** — Side-by-side scoped comparison, apples-to-apples chart,
   Synthefy's authoritative category-month comparison
4. **🔍 Level Drilldown** — Interactive dimension picker with top-N filter
5. **🧮 Model View** — Leaderboard across 11 ML models with WAPE/Bias
6. **🧹 Data Quality** — Missing actuals/forecasts, DQ flag counts, forward-looking row count

## Theme

Light/Dark toggle in the top-right of every page. Persists for the session.

## IBP Scope Caveat

IBP plans approximately **10% of US revenue**, scoped to 6 of 13 product
categories. IBP is **COGS-denominated**; ML models are **revenue-denominated**.
The dashboard:

- Shows the **IBP scope panel** alongside the **Full US scope** panel (never on
  the same y-axis).
- Greys out months where IBP has no actual (Jun–Dec 2025 forward; all of 2024).
- Surfaces the documented data-quality flags (`ibp_unit_proxy_cogs`,
  `forward_looking_ibp`).
- Uses Synthefy's pre-computed `ibp_vs_model_comparison_2025.csv` as the
  authoritative head-to-head for Jan–May 2025.

## Streamlit Cloud Deployment

1. Push this repo to GitHub.
2. On https://share.streamlit.io connect the repo and point at `app.py`.
3. `requirements.txt` is picked up automatically.

## Tested Against

- Python 3.12
- streamlit 1.35+
- pandas 2.0+
- plotly 5.20+
