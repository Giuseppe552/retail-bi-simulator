import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from retail_bi import clean_and_forecast  # reuse your existing logic

st.set_page_config(page_title="Retail BI Simulator", layout="wide")

st.title("🛒 Retail BI Simulator")
st.write("Upload a retail transactions CSV and get BI summaries, forecasts, and anomaly detection.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    tmp_path = Path("data/upload.csv")
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())
    outputs = clean_and_forecast(tmp_path, out_dir="outputs")

    st.success("Processing complete ✅")

    # Forecast chart
    st.subheader("📈 Forecast")
    st.image("outputs/forecast.png")

    # Dashboard preview
    st.subheader("📊 Dashboard (static preview)")
    st.components.v1.html(Path("outputs/bi_dashboard.html").read_text(), height=600, scrolling=True)

    # Report
    st.subheader("📝 Executive Summary")
    st.text(Path("outputs/report.txt").read_text())
else:
    st.info("Upload a CSV file to get started.")
