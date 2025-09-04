#!/usr/bin/env python3
import io, zipfile, hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from retail_bi import (
    load_transactions, monthly_agg, totals_series,
    forecast_with_ci, detect_anomalies, export_bi, write_exec_report
)

# ---------- Page setup ----------
st.set_page_config(
    page_title="Retail BI Simulator — Forecast & Anomaly Dashboard",
    page_icon="🛒",
    layout="wide"
)

# ---------- Brand header ----------
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;">
  <div style="font-size:28px">🛒</div>
  <div>
    <h1 style="margin:0;">Retail BI Simulator</h1>
    <div style="opacity:.75;">
      From transactions → cleaned data → BI summaries → forecast with confidence bands → anomaly flags → BI-ready exports.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar wizard ----------
st.sidebar.header("Workflow")
st.sidebar.markdown("**1. Upload** → **2. Configure** → **3. Analyze** → **4. Download**")
st.sidebar.divider()

# Config controls
st.sidebar.subheader("Step 2 — Configure")
horizon   = st.sidebar.slider("Forecast horizon (months)", 3, 12, 3)
ci_level  = st.sidebar.slider("Confidence band (%)", 80, 95, 80, step=5, help="80% is tighter (BI-friendly); 95% is wider.")
z_thresh  = st.sidebar.slider("Anomaly Z-score threshold", 2.0, 4.0, 3.0, step=0.1, help="Higher = fewer anomalies flagged.")

# ---------- Input help + template ----------
with st.expander("Input format (works with UCI/Kaggle *Online Retail* schemas)", expanded=False):
    st.markdown("""
**Required columns (case-insensitive aliases allowed):**
- `InvoiceDate` or `Date`
- `Quantity`
- `UnitPrice` or `Price`

**Optional:** `Country`, `Description` / `StockCode` (used to infer Category).
""")

TEMPLATE = """InvoiceDate,Quantity,UnitPrice,Country,Description
2024-01-05,2,20.0,United Kingdom,Tea Mug
2024-02-16,5,9.0,France,Notebook
2024-03-04,1,900.0,United States,Laptop
"""
st.download_button("Download CSV template", TEMPLATE.encode("utf-8"), "transactions_template.csv")

# ---------- Upload / demo ----------
uploaded = st.file_uploader("Step 1 — Upload CSV", type=["csv"])
use_demo = st.checkbox("Or tick to use a tiny demo sample", value=False)

def demo_df():
    return pd.read_csv(io.StringIO(TEMPLATE))

def cache_key(df: pd.DataFrame, horizon: int, ci: int, z: float) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    h.update(str(horizon).encode()); h.update(str(ci).encode()); h.update(str(z).encode())
    return h.hexdigest()

@st.cache_data(show_spinner=False)
def run_pipeline(df_raw: pd.DataFrame, horizon: int, ci_level: int, z_thresh: float):
    # Use original file-based loader for identical cleaning
    tmp = Path(".streamlit_upload.csv")
    df_raw.to_csv(tmp, index=False)
    cleaned = load_transactions(tmp)
    try: tmp.unlink(missing_ok=True)
    except Exception: pass

    monthly = monthly_agg(cleaned)
    total   = totals_series(monthly)

    # Base forecast returns 80% CI; widen if user requests 85/90/95 (approx factor)
    fc, residuals = forecast_with_ci(total, steps=horizon)
    widen = {80:1.00, 85:1.15, 90:1.35, 95:1.80}.get(ci_level, 1.00)
    if widen != 1.00 and not fc.empty:
        base_mid = fc["yhat"]
        fc["lower"] = (base_mid - (base_mid - fc["lower"])*widen).clip(lower=0)
        fc["upper"] = base_mid + (fc["upper"] - base_mid)*widen

    anomalies = detect_anomalies(residuals, z_thresh=z_thresh)

    latest = monthly["Month"].max()
    last3  = monthly[monthly["Month"] >= (latest - pd.offsets.MonthBegin(2))]
    by_country  = last3.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
    by_category = last3.groupby("Category")["Revenue"].sum().sort_values(ascending=False)

    return cleaned, monthly, total, fc, anomalies, by_country, by_category

# ---------- Load data ----------
if use_demo:
    raw = demo_df()
elif uploaded:
    raw = pd.read_csv(uploaded)
else:
    st.info("Upload a CSV (or tick demo) to continue.")
    st.stop()

# ---------- Run ----------
with st.spinner("Analyzing…"):
    cleaned, monthly, total, fc, anomalies, by_country, by_category = run_pipeline(raw, horizon, ci_level, z_thresh)

# ---------- Tabs ----------
tab_dash, tab_forecast, tab_anom, tab_exports = st.tabs(["📊 Dashboard", "📈 Forecast", "⚠️ Anomalies", "⬇️ Exports"])

# === Dashboard ===
with tab_dash:
    # KPI row
    latest = monthly["Month"].max()
    last3  = monthly[monthly["Month"] >= (latest - pd.offsets.MonthBegin(2))]
    total_l3m = float(last3["Revenue"].sum()) if not last3.empty else 0.0
    top_ctry  = (last3.groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(1))
    top_cat   = (last3.groupby("Category")["Revenue"].sum().sort_values(ascending=False).head(1))
    next_sum  = float(fc["yhat"].sum()) if not fc.empty else 0.0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Latest Month", latest.strftime("%Y-%m"))
    c2.metric("Revenue (last 3 mo)", f"£{total_l3m:,.0f}")
    if not top_ctry.empty: c3.metric("Top Country (L3M)", top_ctry.index[0], f"£{float(top_ctry.iloc[0]):,.0f}")
    if not top_cat.empty:  c4.metric("Top Category (L3M)", top_cat.index[0], f"£{float(top_cat.iloc[0]):,.0f}")
    st.caption("KPIs auto-refresh when you adjust the configuration in the sidebar.")

    # Country & Category bars
    b1, b2 = st.columns(2)
    with b1:
        bc = go.Figure(go.Bar(x=by_country.index[:10], y=by_country.values[:10]))
        bc.update_layout(title="Top Countries (Last 3 Months)", xaxis_title="Country", yaxis_title="Revenue")
        st.plotly_chart(bc, use_container_width=True)
    with b2:
        bcat = go.Figure(go.Bar(x=by_category.index[:10], y=by_category.values[:10]))
        bcat.update_layout(title="Top Categories (Last 3 Months)", xaxis_title="Category", yaxis_title="Revenue")
        st.plotly_chart(bcat, use_container_width=True)

    st.markdown("#### Cleaned sales (last 3 months)")
    st.dataframe(last3, use_container_width=True, height=280)

# === Forecast ===
with tab_forecast:
    st.markdown("#### Monthly revenue with forecast and confidence band")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=total.index, y=total.values, name="Actual", mode="lines"))
    if not fc.empty:
        fig.add_trace(go.Scatter(x=fc.index, y=fc["yhat"], name="Forecast", mode="lines"))
        fig.add_trace(go.Scatter(
            x=list(fc.index)+list(fc.index[::-1]),
            y=list(fc["upper"])+list(fc["lower"][::-1]),
            fill="toself", name=f"{ci_level}% CI", mode="lines", opacity=0.2
        ))
    # mark anomalies on historical portion
    an_pts = anomalies[anomalies["Anomaly"]]
    if not an_pts.empty:
        y_vals = total.reindex(an_pts["Month"]).values
        fig.add_trace(go.Scatter(x=an_pts["Month"], y=y_vals, mode="markers",
                                 name="Anomalies", marker=dict(size=10, symbol="x")))
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10),
                      xaxis_title="Month", yaxis_title="Revenue")
    st.plotly_chart(fig, use_container_width=True)

    # Show forecast table
    if not fc.empty:
        st.markdown("##### Forecast table")
        st.dataframe(fc.reset_index().rename(columns={"index":"Month"}), use_container_width=True)

# === Anomalies ===
with tab_anom:
    st.markdown("#### Detected anomalies (z-score method)")
    if (anomalies["Anomaly"]==True).any():
        st.dataframe(anomalies[anomalies["Anomaly"]==True], use_container_width=True, height=320)
        st.caption("Investigate spikes/drops: promo effects, price changes, stockouts, data issues.")
    else:
        st.success("No anomalies detected with current threshold.")

# === Exports ===
with tab_exports:
    st.markdown("#### Download cleaned data and BI exports")
    cA, cB, cC, cD = st.columns(4)
    cA.download_button("Cleaned transactions (CSV)",
                       cleaned.to_csv(index=False).encode("utf-8"),
                       "cleaned_transactions.csv", "text/csv", use_container_width=True)
    cB.download_button("Cleaned sales (CSV)",
                       monthly.rename(columns={"Month":"Date"}).to_csv(index=False).encode("utf-8"),
                       "cleaned_sales.csv", "text/csv", use_container_width=True)
    cC.download_button("Anomalies (CSV)",
                       anomalies.to_csv(index=False).encode("utf-8"),
                       "anomalies.csv", "text/csv", use_container_width=True)
    if not fc.empty:
        fc_tbl = fc.reset_index().rename(columns={"index":"Month"})
        cD.download_button("Forecast + CI (CSV)",
                           fc_tbl.to_csv(index=False).encode("utf-8"),
                           "forecast_ci.csv", "text/csv", use_container_width=True)

    # Build BI exports + exec report + ZIP pack
    outdir = Path("outputs"); outdir.mkdir(parents=True, exist_ok=True)
    export_bi(monthly, totals_series(monthly), anomalies, outdir)
    write_exec_report(monthly, totals_series(monthly), fc, anomalies, outdir / "report.txt")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("cleaned_transactions.csv", cleaned.to_csv(index=False))
        z.writestr("cleaned_sales.csv", monthly.rename(columns={"Month":"Date"}).to_csv(index=False))
        z.writestr("anomalies.csv", anomalies.to_csv(index=False))
        if not fc.empty:
            z.writestr("forecast_ci.csv", fc_tbl.to_csv(index=False))
        # Include BI exports (CSV) + SQLite + report
        for p in (outdir/"bi_exports").glob("*.csv"):
            z.write(p, arcname=f"bi_exports/{p.name}")
        z.write(outdir/"retail_bi.sqlite", arcname="retail_bi.sqlite")
        z.write(outdir/"report.txt", arcname="report.txt")
    st.download_button("Download full results pack (ZIP)", buf.getvalue(), "retail_bi_results.zip", use_container_width=True)

# ---------- Footer ----------
st.markdown("<hr />", unsafe_allow_html=True)
st.caption("Built by Giuseppe • BI-ready outputs for Power BI/Tableau. Feedback welcome.")
