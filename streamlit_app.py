#!/usr/bin/env python3
import io
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Import the real functions from your pipeline
from retail_bi import (
    load_transactions, monthly_agg, totals_series,
    forecast_with_ci, detect_anomalies
)

st.set_page_config(page_title="Retail BI Simulator", page_icon="🛒", layout="wide")
st.title("🛒 Retail BI Simulator")
st.write("Upload a retail transactions CSV and get a cleaned dataset, BI summaries, a forecast with confidence bands, and anomaly flags.")

with st.expander("Input format (works with UCI/Kaggle *Online Retail* columns)", expanded=False):
    st.markdown("""
    **Required columns** (case-insensitive aliases allowed):
    - `InvoiceDate` or `Date`
    - `Quantity`
    - `UnitPrice` or `Price`
    - *Optional:* `Country`, `Description` (or `StockCode`)
    """)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
use_demo = st.checkbox("Use tiny demo sample", value=False)

def _demo_df():
    csv = io.StringIO("""InvoiceDate,Quantity,UnitPrice,Country,Description
2024-01-05,2,20.0,United Kingdom,Tea Mug
2024-01-12,1,120.0,Germany,Office Chair
2024-02-02,3,15.0,United Kingdom,Tea Spoon
2024-02-16,5,9.0,France,Notebook
2024-03-04,1,900.0,United States,Laptop
2024-03-20,2,35.0,United Kingdom,Water Bottle
2024-04-03,4,12.0,France,Notebook
2024-04-18,2,220.0,Germany,Desk Lamp
2024-05-01,3,25.0,United Kingdom,Tea Mug
2024-06-14,7,11.0,France,Notebook
2024-07-07,1,1000.0,United States,Laptop
2024-08-11,2,40.0,Germany,Headphones
""")
    return pd.read_csv(csv)

def _load_via_pipeline_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Your retail_bi.load_transactions expects a file path; to reuse it safely,
    write the uploaded DataFrame to a temp CSV and call the same loader.
    """
    tmp = Path(".streamlit_tmp.csv")
    df.to_csv(tmp, index=False)
    cleaned = load_transactions(tmp)
    try: tmp.unlink(missing_ok=True)
    except Exception: pass
    return cleaned

if uploaded or use_demo:
    try:
        raw = _demo_df() if use_demo else pd.read_csv(uploaded)
        df = _load_via_pipeline_from_df(raw)

        monthly = monthly_agg(df)
        total = totals_series(monthly)
        fc, residuals = forecast_with_ci(total, steps=3)
        anomalies = detect_anomalies(residuals)

        # KPIs
        c1, c2, c3 = st.columns(3)
        latest = monthly["Month"].max()
        last3 = monthly[monthly["Month"] >= (latest - pd.offsets.MonthBegin(2))]
        top_country = last3.groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(1)
        top_cat = last3.groupby("Category")["Revenue"].sum().sort_values(ascending=False).head(1)
        c1.metric("Latest month", latest.strftime("%Y-%m"))
        if not top_country.empty:
            c2.metric("Top country (L3M)", top_country.index[0], f"£{float(top_country.iloc[0]):,.0f}")
        if not top_cat.empty:
            c3.metric("Top category (L3M)", top_cat.index[0], f"£{float(top_cat.iloc[0]):,.0f}")

        # Forecast + CI + anomalies (Plotly)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=total.index, y=total.values, mode="lines", name="Actual"))
        if not fc.empty:
            fig.add_trace(go.Scatter(x=fc.index, y=fc["yhat"], mode="lines", name="Forecast"))
            fig.add_trace(go.Scatter(
                x=list(fc.index)+list(fc.index[::-1]),
                y=list(fc["upper"])+list(fc["lower"][::-1]),
                fill="toself", name="80% CI", mode="lines", opacity=0.2
            ))
        anom_pts = anomalies[anomalies["Anomaly"] == True]
        if not anom_pts.empty:
            y_vals = total.reindex(anom_pts["Month"]).values
            fig.add_trace(go.Scatter(x=anom_pts["Month"], y=y_vals, mode="markers",
                                     name="Anomalies", marker=dict(size=10, symbol="x")))
        fig.update_layout(title="Monthly Revenue (Actual + Forecast + CI + Anomalies)",
                          xaxis_title="Month", yaxis_title="Revenue")
        st.plotly_chart(fig, use_container_width=True)

        # Tables
        tab1, tab2, tab3 = st.tabs(["📈 Time series", "🌍 Countries (L3M)", "🏷️ Categories (L3M)"])
        with tab1:
            st.dataframe(total.reset_index().rename(columns={"index":"Month", 0:"Revenue"}), use_container_width=True)
        with tab2:
            st.dataframe(last3.groupby("Country")["Revenue"].sum().sort_values(ascending=False).reset_index(), use_container_width=True)
        with tab3:
            st.dataframe(last3.groupby("Category")["Revenue"].sum().sort_values(ascending=False).reset_index(), use_container_width=True)

        # Downloads
        st.subheader("⬇️ Downloads")
        st.download_button("Cleaned sales (CSV)",
                           monthly.rename(columns={"Month":"Date"}).to_csv(index=False).encode("utf-8"),
                           file_name="cleaned_sales.csv", mime="text/csv")
        st.download_button("Anomalies (CSV)",
                           anomalies.to_csv(index=False).encode("utf-8"),
                           file_name="anomalies.csv", mime="text/csv")
        if not fc.empty:
            fc_tbl = fc.reset_index().rename(columns={"index":"Month"})
            st.dataframe(fc_tbl)
            st.download_button("Forecast with CI (CSV)",
                               fc_tbl.to_csv(index=False).encode("utf-8"),
                               file_name="forecast_ci.csv", mime="text/csv")

        st.caption("Tip: load these CSVs into Power BI/Tableau for further slicing. The app handles cleaning, forecasting and anomaly flags.")
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
else:
    st.info("Upload a CSV (or tick the demo) to get started.")
