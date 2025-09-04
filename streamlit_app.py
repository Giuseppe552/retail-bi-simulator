#!/usr/bin/env python3
import io, zipfile, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---- Import your pipeline helpers (keep these exactly as they exist in retail_bi.py) ----
from retail_bi import (
    load_transactions, monthly_agg, totals_series,
    forecast_with_ci, detect_anomalies, export_bi
)

# ====================== Page & sidebar ======================
st.set_page_config(page_title="Retail BI Simulator — Forecast & Anomaly Dashboard",
                   page_icon="🛒", layout="wide")

with st.sidebar:
    st.markdown("### Workflow\n1. Upload → 2. Configure → 3. Analyze → 4. Download")
    horizon = st.slider("Forecast horizon (months)", 1, 12, 3, step=1)
    ci_level = st.slider("Confidence band (%)", 50, 95, 80, step=5)
    z_thresh = st.slider("Anomaly Z-score threshold", 1.0, 4.0, 3.0, step=0.1)
    st.caption("Tip: 80% CI is common for BI; 95% is wider.")

# ====================== Header & help ======================
st.markdown("## 🛒 Retail BI Simulator")
with st.expander("Input format (works with UCI/Kaggle *Online Retail* schemas)"):
    st.markdown("""
**Required columns (case-insensitive aliases allowed):**
- `InvoiceDate` **or** `Date`
- `Quantity`
- `UnitPrice` **or** `Price`  
**Optional:** `Country`, `Description` / `StockCode` *(used to infer Category)*.
""")

st.download_button(
    "Download CSV template",
    pd.DataFrame(columns=["InvoiceDate","Quantity","UnitPrice","Country","Description"]).to_csv(index=False),
    file_name="retail_template.csv",
    type="secondary"
)

# ====================== Demo data (in-app, robust) ======================
def tiny_demo_df() -> pd.DataFrame:
    n = 90
    return pd.DataFrame({
        "InvoiceDate": pd.date_range("2024-01-01", periods=n, freq="D"),
        "Quantity":    np.random.poisson(2, n),
        "UnitPrice":   np.random.normal(20, 3, n).round(2),
        "Country":     np.random.choice(["United Kingdom","United States","France"], size=n),
        "Description": np.random.choice(["Notebook","Tea Mug","Desk Lamp"], size=n),
    })

def build_demo_df(n_rows: int = 20000, seed: int = 42) -> pd.DataFrame:
    """Fast, in-memory advanced demo (~20k rows)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2025-06-30", freq="D")
    countries = np.array(["United Kingdom","United States","Germany","France","Spain"])
    country_p = np.array([0.40, 0.25, 0.15, 0.12, 0.08])

    prods = {
        "Laptop":        (950, 120, 1.1),
        "Headphones":    ( 80,  20, 1.4),
        "Notebook":      ( 12,   3, 3.0),
        "Office Chair":  (180,  35, 1.2),
        "Tea Mug":       (  9,   2, 2.4),
        "Desk Lamp":     ( 35,   8, 1.6),
    }
    prod_names = np.array(list(prods.keys()))
    price_mu   = np.array([prods[p][0] for p in prod_names])
    price_sd   = np.array([prods[p][1] for p in prod_names])
    qty_mu     = np.array([prods[p][2] for p in prod_names])

    idx = rng.integers(0, len(dates), size=n_rows)
    dts = dates[idx]
    pidx = rng.integers(0, len(prod_names), size=n_rows)
    cidx = rng.choice(len(countries), size=n_rows, p=country_p)

    base_price = rng.normal(price_mu[pidx], price_sd[pidx]).clip(1)
    base_qty   = rng.poisson(lam=np.maximum(qty_mu[pidx], 0.5))

    # simple promo & seasonality
    month = pd.to_datetime(dts).month
    promo  = 1.0 - ((month == 11) | (month == 12)) * 0.10  # 10% off Q4
    seas   = 1.0 + ((month == 9)  | (month == 10)) * 0.15  # +15% back-to-school

    price = (base_price * promo).round(2)
    qty   = (base_qty * seas).clip(1)

    df = pd.DataFrame({
        "InvoiceDate": dts,
        "Quantity": qty,
        "UnitPrice": price,
        "Country": countries[cidx],
        "Description": prod_names[pidx],
    })
    return df

# ====================== Input selection ======================
uploaded = st.file_uploader("Step 1 — Upload CSV", type=["csv"])
use_demo_tiny = st.checkbox("Use tiny demo sample", value=False)
use_demo_adv  = st.checkbox("Use advanced demo (2 years, promos, anomalies)", value=True)

if use_demo_adv:
    raw_df = build_demo_df()
elif use_demo_tiny:
    raw_df = tiny_demo_df()
elif uploaded is not None:
    raw_df = pd.read_csv(uploaded)
else:
    st.info("Upload a CSV (or tick one of the demo options) to continue.")
    st.stop()

# ====================== Robust loader shim ======================
def safe_load_transactions(src_df: pd.DataFrame) -> pd.DataFrame:
    """Use your real loader; if it only accepts paths, write temp CSV as a fallback."""
    try:
        # If your load_transactions already accepts DataFrame, this will work:
        return load_transactions(src_df)
    except Exception:
        tmp = Path("._tmp_upload.csv")
        src_df.to_csv(tmp, index=False)
        try:
            df = load_transactions(tmp)
        finally:
            try: tmp.unlink(missing_ok=True)
            except Exception: pass
        return df

# ====================== Pipeline ======================
def to_monthly_series(total_df: pd.DataFrame) -> pd.Series:
    """Convert totals DataFrame (Month, Revenue) -> float Series with PeriodIndex."""
    s = pd.Series(total_df["Revenue"].values,
                  index=pd.PeriodIndex(pd.to_datetime(total_df["Month"]).to_period("M")))
    s = pd.to_numeric(s, errors="coerce").astype(float)
    return s.replace([np.inf, -np.inf], np.nan).dropna()

def run_pipeline(df_raw: pd.DataFrame, steps: int, ci_pct: int, z: float):
    cleaned = safe_load_transactions(df_raw)

    monthly = monthly_agg(cleaned)
    total   = totals_series(monthly)            # DataFrame with Month, Revenue
    s       = to_monthly_series(total)          # 1-D Series for modeling

    # NOTE: forecast_with_ci signature varies; we avoid the 'horizon' kw.
    # Use positional or 'steps=' which is common.
    try:
        fc = forecast_with_ci(s, steps=steps, ci=ci_pct/100.0)
    except TypeError:
        # Fallback to positional (series, steps, ci)
        fc = forecast_with_ci(s, steps, ci_pct/100.0)

    # anomalies: use the correct kwarg 'z', not 'z_thresh'
    anomalies = detect_anomalies(s, z=z)

    # split views for dashboard
    by_country  = cleaned.groupby(["Country","Month"], as_index=False)["Revenue"].sum()
    by_category = cleaned.groupby(["Category","Month"], as_index=False)["Revenue"].sum()

    return cleaned, monthly, total, fc, anomalies, by_country, by_category

with st.spinner("Analyzing…"):
    cleaned, monthly, total, fc, anomalies, by_country, by_category = run_pipeline(
        raw_df, steps=horizon, ci_pct=ci_level, z=z_thresh
    )

# ====================== Tabs ======================
tab_dash, tab_fc, tab_anom, tab_export = st.tabs(["📊 Dashboard", "📈 Forecast", "⚠️ Anomalies", "📦 Exports"])

with tab_dash:
    latest_month = monthly["Month"].max()
    last_3m = monthly[monthly["Month"] >= (latest_month - pd.offsets.MonthBegin(2))]
    kpi_rev = float(last_3m["Revenue"].sum()) if len(last_3m) else 0.0
    top_country = last_3m.groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(1)
    top_cat     = last_3m.groupby("Category")["Revenue"].sum().sort_values(ascending=False).head(1)

    c1,c2,c3 = st.columns(3)
    c1.metric("Latest Month", str(pd.to_datetime(latest_month).date())[:7] if pd.notna(latest_month) else "—")
    c2.metric("Revenue (last 3 mo)", f"£{kpi_rev:,.0f}")
    c3.metric("Top Country (L3M)", top_country.index[0] if len(top_country) else "—")

    cc1, cc2 = st.columns(2)
    with cc1:
        top_countries = last_3m.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
        fig = go.Figure([go.Bar(x=top_countries.index, y=top_countries.values, name="Revenue")])
        fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=300)
        st.plotly_chart(fig, width="stretch")
    with cc2:
        top_cats = last_3m.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
        fig = go.Figure([go.Bar(x=top_cats.index, y=top_cats.values, name="Revenue")])
        fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=300)
        st.plotly_chart(fig, width="stretch")

    st.dataframe(cleaned.sort_values("Month", ascending=False).head(200), width="stretch")

with tab_fc:
    # Plot actuals & forecast with CI band
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pd.to_datetime(total["Month"]), y=total["Revenue"], mode="lines", name="Actual"))
    # Create forecast x-axis from last month forward
    last_month = pd.Period(pd.to_datetime(total["Month"].max()), freq="M")
    future_x = pd.period_range(last_month+1, periods=len(fc), freq="M").to_timestamp()
    fig.add_trace(go.Scatter(x=future_x, y=fc["yhat"], mode="lines", name="Forecast"))
    # CI band
    fig.add_trace(go.Scatter(
        x=list(future_x)+list(future_x[::-1]),
        y=list(fc["upper"])+list(fc["lower"][::-1]),
        fill="toself", name=f"{ci_level}% CI", mode="lines"
    ))
    fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=420)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(fc, width="stretch")

with tab_anom:
    if len(anomalies):
        st.dataframe(anomalies, width="stretch")
    else:
        st.success("No anomalies detected with the current threshold.")

with tab_export:
    # Build BI exports in-memory
    out = export_bi(cleaned, monthly, total, fc, anomalies)
    st.download_button("Download cleaned transactions (CSV)", out["transactions_csv"], "transactions_cleaned.csv")
    st.download_button("Download cleaned sales (CSV)", out["sales_csv"], "cleaned_sales.csv")
    st.download_button("Download anomalies (CSV)", out["anom_csv"], "anomalies.csv")
    st.download_button("Download forecast + CI (CSV)", out["forecast_csv"], "forecast_ci.csv")

    # ZIP bundle
    zip_bytes = io.BytesIO()
    with zipfile.ZipFile(zip_bytes, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("transactions_cleaned.csv", out["transactions_csv"])
        z.writestr("cleaned_sales.csv", out["sales_csv"])
        z.writestr("anomalies.csv", out["anom_csv"])
        z.writestr("forecast_ci.csv", out["forecast_csv"])
    st.download_button("Download full results pack (ZIP)", zip_bytes.getvalue(), "retail_bi_outputs.zip")

st.caption("Built by Giuseppe — BI-ready outputs for Power BI/Tableau.")
