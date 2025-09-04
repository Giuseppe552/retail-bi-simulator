# 🛒 Retail BI Simulator

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://retail-bi-simulator-ncsinurqpmtty7obwwcvnmw.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.x-yellow.svg)
![Plotly](https://img.shields.io/badge/Plotly-Graphing-lightblue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Status](https://img.shields.io/badge/Stage-Production%20Demo-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

Turn raw retail transactions → **clean data, BI summaries, a forecast with confidence bands, anomaly flags**, and **BI-ready exports** (CSV + SQLite) in minutes.  
Built to mirror a grad BI developer’s real workflow (clean → aggregate → forecast → QC → export).

---

## 🔎 Screenshots
<p align="center">
  <img src="docs/screenshots/dashboard.png" alt="Dashboard" width="80%"><br/><em>Executive KPIs + Top Countries/Categories</em>
</p>
<p align="center">
  <img src="docs/screenshots/forecast.png" alt="Forecast" width="80%"><br/><em>Monthly revenue with forecast & confidence band</em>
</p>
<p align="center">
  <img src="docs/screenshots/anomalies.png" alt="Anomalies" width="80%"><br/><em>Z-score anomaly detection on residuals</em>
</p>
<p align="center">
  <img src="docs/screenshots/exports.png" alt="Exports" width="80%"><br/><em>One-click downloads (cleaned data, anomalies, forecast, ZIP pack)</em>
</p>

> **Tip:** Tick **“Use advanced demo (2 years, promos, anomalies)”** in the app to see realistic seasonality, promo spikes and anomaly flags immediately.

---

## ✨ What it does
- **Cleans** messy transactions (handles column aliases, negatives, nulls).
- **Aggregates** to monthly revenue by **Country** and inferred **Category**.
- **Forecasts** next 3–12 months (point + CI band; width selectable).
- **Flags anomalies** in residuals (adjustable Z-score threshold).
- **Exports** clean tables for **Power BI/Tableau**: CSVs + a **SQLite** file (star schema + timeseries).
- **ZIP pack**: one click to download everything (cleaned data, forecast, anomalies, BI exports, executive report).

---

## 🧠 Workflow
1) **Upload** CSV (UCI/Kaggle *Online Retail* schema or similar).  
2) **Configure** forecast horizon, CI band, anomaly threshold.  
3) **Analyze**: KPIs, Forecast, Anomalies tabs.  
4) **Download** clean data & BI exports (CSV/SQLite/ZIP).

### Accepted columns (case-insensitive aliases)
- `InvoiceDate` or `Date`  
- `Quantity`  
- `UnitPrice` or `Price`  
- Optional: `Country`, `Description` / `StockCode` (used to infer Category)

---

## 🚀 Quick start (local)
```bash
git clone https://github.com/Giuseppe552/retail-bi-simulator.git
cd retail-bi-simulator
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
````

Or use the live app: **Streamlit badge above**.

---

## 📦 Outputs

* `cleaned_sales.csv` — monthly revenue by Country/Category
* `anomalies.csv` — flagged residuals with Z-scores
* `forecast_ci.csv` — forecast with lower/upper bands
* `bi_exports/` — `dim_date.csv`, `fact_sales.csv`, `total_timeseries.csv`
* `retail_bi.sqlite` — same model in a single file (easy import to BI tools)
* `report.txt` — executive summary (top country/category, forecast total)

---

## 🧰 Tech stack

**Python**, **Pandas**, **Statsmodels**, **NumPy**, **Plotly**, **Streamlit**
Packaging: CSV + SQLite exports for direct **Power BI/Tableau** use.

---

## 🔭 Roadmap

* Seasonal model toggle (SARIMA/Prophet).
* Category drill-down forecast.
* Simple promo calendar overlay.
* Cloud file connectors (GDrive/S3) for scheduled refresh.

---

## 🙋 About

Built by **Giuseppe** — BI-ready automation for retail analytics.
If you want this wired to your data warehouse or shipped as a custom BI tool, reach out.
contact.giuseppe00@gmail.com

```

