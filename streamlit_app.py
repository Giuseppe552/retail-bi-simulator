#!/usr/bin/env python3
import io, hashlib, zipfile
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from retail_bi import (
    load_transactions, monthly_agg, totals_series,
    forecast_with_ci, detect_anomalies, export_bi, write_exec_report
)

st.set_page_config(page_title="Retail BI Simulator", page_icon="🛒", layout="wide")

# ---------- Header ----------
col_logo, col_title = st.columns([1,8])
with col_logo:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("🛒")
with col_title:
    st.markdown("<h1 style='margin-bottom:0'>Retail BI Simulator</h1>", unsafe_allow_html=True)
    st.caption("Upload a retail transactions CSV → Clean data, BI summaries, **forecast with confidence bands**, anomaly flags, and BI-ready exports.")

# ---------- Sidebar: wizard ----------
st.sidebar.header("Workflow")
st.sidebar.markdown("**1. Upload** → **2. Configure** → **3. Analyze** → **4. Download**")
st.sidebar.divider()

with st.expander("Input format (works with UCI/Kaggle *Online Retail* columns)", expanded=False):
    st.markdown("""
    **Required columns** *(case-insensitive aliases allowed)*  
    - `InvoiceDate` or `Date`  
    - `Quantity`  
    - `UnitPrice` or `Price`  
    **Optional:** `Country`, `Description`/`StockCode` (used to infer Category)
    """)

# Template & tiny demo
TEMPLATE = """InvoiceDate,Quantity,UnitPrice,Country,Description
2024-01-05,2,20.0,United Kingdom,Tea Mug
2024-02-16,5,9.0,France,Notebook
2024-03-04,1,900.0,United States,Laptop
"""
st.download_button("Download CSV template", TEMPLATE.encode("utf-8"), "transactions_template.csv", use_container_width=True)

uploaded = st.file_uploader("Step 1 — Upload CSV", type=["csv"])
use_demo = st.checkbox("Or tick to use a tiny demo sample", value=False)

# ---------- Sidebar controls ----------
st.sidebar.subheader("Step 2 — Configure")
horizon = st.sidebar.slider("Forecast horizon (months)", 3, 12, 3)
ci_level = st.sidebar.slider("Confidence band", 80, 95, 80, step=5)
z_thresh = st.sidebar.slider("Anomaly Z-score threshold", 2.0, 4.0, 3.0, step=0.1)
st.sidebar.caption("Tip: 80% CI is common for BI; 95% is wider.")

# ---------- Helpers ----------
def _cache_key(df: pd.DataFrame, horizon: int, ci: int, z: float) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    h.update(str(horizon).encode()); h.update(str(ci).encode()); h.update(str(z).encode())
    return h.hexdigest()

@st.cache_data(show_spinner=False)
def run_pipeline(df: pd.DataFrame, horizon: int, ci: int, z: float):
    # Use original loader by writing to tmp (ensures identical cleaning)
    tmp = Path(".streamlit_upload.csv"); df.to_csv(tmp, index=False)
    cleaned = load_transactions(tmp)
    try: tmp.unlink(missing_ok=True)
    except Exception: pass

    monthly = monthly_agg(cleaned)
    total = totals_series(monthly)
    # Forecast with 80% CI in library; adjust band by re-scaling if user picks 90/95
    fc, residuals = forecast_with_ci(total, steps=horizon)  # returns 80% CI
    # basic rescale for other CI levels (approx): widen band by factor
    if ci != 80 and not fc.empty:
        widen = {85:1.15, 90:1.35, 95:1.8}.get(ci, 1.0)
        fc["lower"] = (total.iloc[-1] - (total.iloc[-1] - fc["lower"])*widen).clip(lower=0)
        fc["upper"] = fc["yhat"] + (fc["upper"] - fc["yhat"])*widen
    anomalies = detect_anomalies(residuals, z_thresh=z)

    # quick filters
    latest = monthly["Month"].max()
    last3 = monthly[monthly["Month"] >= (latest - pd.offsets.MonthBegin(2))]
    by_country = last3.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
    by_category = last3.groupby("Category")["Revenue"].sum().sort_values(ascending=False)

    return cleaned, monthly, total, fc, anomalies, by_country, by_category

def _df_download(df: pd.DataFrame, name: str):
    return st.download_button(f"Download {name} (CSV)", df.to_csv(index=False).encode("utf-8"),
                              file_name=f"{name}.csv", mime="text/csv", use_container_width=True)

# ---------- Load data ----------
if use_demo:
    df_raw = pd.read_csv(io.StringIO(TEMPLATE))
elif uploaded:
    df_raw = pd.read_csv(uploaded)
else:
    df_raw = None

if df_raw is None:
    st.info("Upload a CSV (or tick demo) to continue.")
    st.stop()

# ---------- Run analysis (cached) ----------
key = _cache_key(df_raw, horizon, ci_level, z_thresh)
with st.spinner("Crunching numbers…"):
    cleaned, monthly, total, fc, anomalies, by_country, by_category = run_pipeline(df_raw, horizon, ci_level, z_thresh)

# ---------- Step 3 — Analyze ----------
k1, k2, k3, k4 = st.columns(4)
latest = monthly["Month"].max()
k1.metric("Latest Month", latest.strftime("%Y-%m"))
if not by_country.empty: k2.metric("Top Country (L3M)", by_country.index[0], f"£{float(by_country.iloc[0]):,.0f}")
if not by_category.empty: k3.metric("Top Category (L3M)", by_category.index[0], f"£{float(by_category.iloc[0]):,.0f}")
anom_count = int((anomalies["Anomaly"]==True).sum())
k4.metric("Anomalies Detected", anom_count)

# Filters
st.markdown("### Filters")
fc1, fc2 = st.columns(2)
sel_country = fc1.multiselect("Countries", options=list(by_country.index), default=list(by_country.index[:5]))
sel_category = fc2.multiselect("Categories", options=list(by_category.index), default=list(by_category.index[:5]))
last3 = monthly[monthly["Month"] >= (latest - pd.offsets.MonthBegin(2))]
filtered = last3[last3["Country"].isin(sel_country) & last3["Category"].isin(sel_category)]

# Charts
st.markdown("### Forecast & Anomalies")
fig = go.Figure()
fig.add_trace(go.Scatter(x=total.index, y=total.values, name="Actual", mode="lines"))
if not fc.empty:
    fig.add_trace(go.Scatter(x=fc.index, y=fc["yhat"], name="Forecast", mode="lines"))
    fig.add_trace(go.Scatter(x=list(fc.index)+list(fc.index[::-1]),
                             y=list(fc["upper"])+list(fc["lower"][::-1]),
                             fill="toself", name=f"{ci_level}% CI", mode="lines", opacity=0.2))
anom_pts = anomalies[anomalies["Anomaly"]==True]
if not anom_pts.empty:
    y_vals = total.reindex(anom_pts["Month"]).values
    fig.add_trace(go.Scatter(x=anom_pts["Month"], y=y_vals, mode="markers",
                             name="Anomalies", marker=dict(size=10, symbol="x")))
fig.update_layout(margin=dict(l=10,r=10,t=40,b=10),
                  title="Monthly Revenue (Actual + Forecast + CI + Anomalies)",
                  xaxis_title="Month", yaxis_title="Revenue")
st.plotly_chart(fig, use_container_width=True)

t1, t2 = st.tabs(["🌍 Countries (L3M)", "🏷️ Categories (L3M)"])
with t1:
    st.dataframe(by_country.reset_index().rename(columns={"Revenue":"Revenue (L3M)"}), use_container_width=True)
with t2:
    st.dataframe(by_category.reset_index().rename(columns={"Revenue":"Revenue (L3M)"}), use_container_width=True)

st.markdown("### Filtered Breakdown (L3M)")
st.dataframe(filtered, use_container_width=True, height=260)

# ---------- Step 4 — Downloads ----------
st.markdown("## ⬇️ Downloads")
cA, cB, cC, cD = st.columns(4)
with cA: _df_download(cleaned, "cleaned_transactions")
with cB: _df_download(monthly.rename(columns={"Month":"Date"}), "cleaned_sales")
with cC: _df_download(anomalies, "anomalies")
if not fc.empty:
    with cD:
        fc_tbl = fc.reset_index().rename(columns={"index":"Month"})
        _df_download(fc_tbl, "forecast_ci")

# BI exports (CSV + SQLite) + Exec report + ZIP pack
outdir = Path("outputs"); outdir.mkdir(exist_ok=True, parents=True)
export_bi(monthly, total, anomalies, outdir)
write_exec_report(monthly, total, fc, anomalies, outdir / "report.txt")

# Build a single ZIP for convenience
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("cleaned_sales.csv", monthly.rename(columns={"Month":"Date"}).to_csv(index=False))
    z.writestr("anomalies.csv", anomalies.to_csv(index=False))
    if not fc.empty:
        z.writestr("forecast_ci.csv", fc_tbl.to_csv(index=False))
    # include BI exports
    for p in (outdir/"bi_exports").glob("*.csv"):
        z.write(p, arcname=f"bi_exports/{p.name}")
    # include sqlite + report
    z.write(outdir/"retail_bi.sqlite", arcname="retail_bi.sqlite")
    z.write(outdir/"report.txt", arcname="report.txt")
st.download_button("Download results pack (ZIP)", buf.getvalue(), "retail_bi_results.zip", use_container_width=True)

st.caption("Need help integrating with Power BI/Tableau? Message me on LinkedIn — happy to share a pre-wired PBIX.")
