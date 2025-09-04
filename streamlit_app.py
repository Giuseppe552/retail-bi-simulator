#!/usr/bin/env python3
import io, zipfile, hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# --- Local modules
from data.make_advanced_demo import build_demo_df   # advanced in-memory demo

from retail_bi import (
    load_transactions, monthly_agg, totals_series,
    forecast_with_ci, detect_anomalies, export_bi, write_exec_report
)

# ---------------------- Page setup ----------------------
st.set_page_config(
    page_title="Retail BI Simulator — Forecast & Anomaly Dashboard",
    page_icon="🛒",
    layout="wide"
)

# ---------------------- Sidebar controls ----------------
with st.sidebar:
    st.markdown("### Workflow\n1. Upload → 2. Configure → 3. Analyze → 4. Download")
    horizon = st.slider("Forecast horizon (months)", 1, 12, 3, step=1)
    ci_level = st.slider("Confidence band (%)", 50, 95, 80, step=5)
    z_thresh = st.slider("Anomaly Z-score threshold", 1.0, 4.0, 3.0, step=0.1)
    st.caption("Tip: 80% CI is common for BI; 95% is wider.")

# ---------------------- Header & help -------------------
st.markdown("## 🛒 Retail BI Simulator")
with st.expander("Input format (works with UCI/Kaggle *Online Retail* schemas)"):
    st.markdown(
        """
        **Required columns (case-insensitive aliases allowed):**
        - `InvoiceDate` **or** `Date`  
        - `Quantity`  
        - `UnitPrice` **or** `Price`  
        **Optional:** `Country`, `Description` / `StockCode` *(used to infer Category)*.
        """
    )
st.download_button(
    "Download CSV template",
    pd.DataFrame(columns=["InvoiceDate","Quantity","UnitPrice","Country","Description"]).to_csv(index=False),
    file_name="retail_template.csv",
    type="secondary"
)

# ---------------------- Data selection ------------------
uploaded = st.file_uploader("Step 1 — Upload CSV", type=["csv"])
use_demo_tiny = st.checkbox("Use tiny demo sample", value=False)
use_demo_advanced = st.checkbox("Use advanced demo (2 years, promos, anomalies)", value=True)

def tiny_demo_df():
    return pd.DataFrame({
        "InvoiceDate": pd.date_range("2024-01-01", periods=90, freq="D"),
        "Quantity":    np.random.poisson(2, 90),
        "UnitPrice":   np.random.normal(20, 3, 90).round(2),
        "Country":     np.random.choice(["United Kingdom","United States","France"], size=90),
        "Description": np.random.choice(["Notebook","Tea Mug","Desk Lamp"], size=90),
    })

# Decide source of raw data
if use_demo_advanced:
    raw_df = build_demo_df()                         # in-memory, fast
elif use_demo_tiny:
    raw_df = tiny_demo_df()
elif uploaded is not None:
    raw_df = pd.read_csv(uploaded)
else:
    st.info("Upload a CSV (or tick one of the demo options) to continue.")
    st.stop()

# ---------------------- Pipeline ------------------------
def run_pipeline(raw, horizon, ci_level, z_thresh):
    # 1) Clean & normalize
    cleaned = load_transactions(raw)  # accepts DataFrame or path
    # 2) Aggregations
    monthly = monthly_agg(cleaned)
    total   = totals_series(monthly)
    # 3) Forecast + CI
    fc = forecast_with_ci(total, horizon=horizon, ci=ci_level/100.0)
    # 4) Anomalies (on residuals to recent trend)
    anomalies = detect_anomalies(total, z_thresh=z_thresh)
    # 5) Splits
    by_country  = cleaned.groupby(["Country","Month"], as_index=False)["Revenue"].sum()
    by_category = cleaned.groupby(["Category","Month"], as_index=False)["Revenue"].sum()
    return cleaned, monthly, total, fc, anomalies, by_country, by_category

cleaned, monthly, total, fc, anomalies, by_country, by_category = run_pipeline(
    raw_df, horizon, ci_level, z_thresh
)

# ---------------------- Tabs ----------------------------
tab_dash, tab_fc, tab_anom, tab_export = st.tabs(["📊 Dashboard", "📈 Forecast", "⚠️ Anomalies", "📦 Exports"])

with tab_dash:
    # KPI cards
    latest_month = monthly["Month"].max()
    last_3m = monthly[monthly["Month"] >= (latest_month - pd.offsets.MonthBegin(2))]
    kpi_rev = last_3m["Revenue"].sum()
    top_country = (last_3m.groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(1))
    top_category = (last_3m.groupby("Category")["Revenue"].sum().sort_values(ascending=False).head(1))

    c1,c2,c3 = st.columns(3)
    c1.metric("Latest Month", str(latest_month.date())[:7])
    c2.metric("Revenue (last 3 mo)", f"£{kpi_rev:,.0f}")
    c3.metric("Top Country (L3M)", top_country.index[0] if len(top_country) else "—")

    # Charts
    cc1, cc2 = st.columns(2)
    with cc1:
        top_countries = last_3m.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
        fig = go.Figure(data=[go.Bar(x=top_countries.index, y=top_countries.values, name="Revenue")])
        fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        top_cats = last_3m.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
        fig = go.Figure(data=[go.Bar(x=top_cats.index, y=top_cats.values, name="Revenue")])
        fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(cleaned.sort_values("Month", ascending=False).head(200), use_container_width=True)

with tab_fc:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=total["Month"], y=total["Revenue"], mode="lines", name="Actual"))
    future_x = pd.date_range(total["Month"].max() + pd.offsets.MonthBegin(1), periods=len(fc), freq="MS")
    fig.add_trace(go.Scatter(x=future_x, y=fc["yhat"], mode="lines", name="Forecast"))
    fig.add_trace(go.Scatter(x=future_x.tolist()+future_x[::-1].tolist(),
                             y=fc["upper"].tolist()+fc["lower"][::-1].tolist(),
                             fill="toself", name=f"{ci_level}% CI", mode="lines"))
    fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(fc, use_container_width=True)

with tab_anom:
    if len(anomalies):
        st.dataframe(anomalies, use_container_width=True)
    else:
        st.success("No anomalies detected with the current threshold.")

with tab_export:
    # Build BI exports in-memory
    buf = export_bi(cleaned, monthly, total, fc, anomalies)
    st.download_button("Download cleaned transactions (CSV)", buf["transactions_csv"], "transactions_cleaned.csv")
    st.download_button("Download cleaned sales (CSV)", buf["sales_csv"], "cleaned_sales.csv")
    st.download_button("Download anomalies (CSV)", buf["anom_csv"], "anomalies.csv")
    st.download_button("Download forecast + CI (CSV)", buf["forecast_csv"], "forecast_ci.csv")
    # ZIP pack
    zip_bytes = io.BytesIO()
    with zipfile.ZipFile(zip_bytes, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("transactions_cleaned.csv", buf["transactions_csv"])
        z.writestr("cleaned_sales.csv", buf["sales_csv"])
        z.writestr("anomalies.csv", buf["anom_csv"])
        z.writestr("forecast_ci.csv", buf["forecast_csv"])
    st.download_button("Download full results pack (ZIP)", zip_bytes.getvalue(), "retail_bi_outputs.zip")

st.caption("Built by Giuseppe — BI-ready outputs for Power BI/Tableau.")
