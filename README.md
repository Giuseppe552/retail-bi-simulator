# 🛒 Retail BI Simulator

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.x-yellow.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Status](https://img.shields.io/badge/Stage-Live%20Demo-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

> **Business context:**  
Retailers sit on mountains of messy sales transactions. BI teams need to turn these into **clean, actionable insights** for planning, forecasting, and anomaly detection.  
This project automates that journey — **from raw CSV → insights → BI-ready exports.**

---

## 🌐 Live Demo

👉 [**Try the Retail BI Simulator (Streamlit app)**](https://retail-bi-simulator-ncsinurqpmtty7obwwcwmw.streamlit.app)  

Upload your own CSV (UCI/Kaggle Online Retail schema) or use the demo data to explore.  

---

## 📊 Features

- **Data Cleaning** → raw transactions → cleaned sales by month, country, and category  
- **Executive Dashboard** → KPIs (latest revenue, top country/category)  
- **Forecasting** → 3–6 month ARIMA forecast with confidence intervals  
- **Anomaly Detection** → Z-score method for fraud/returns risk  
- **BI-Ready Exports** → CSVs + SQLite for Power BI / Tableau  

---

## 🖼️ Screenshots

### Dashboard KPIs
<p align="center"><img src="assets/dashboard.png" alt="Retail BI Dashboard" width="80%"></p>

### Forecast with Confidence Bands
<p align="center"><img src="assets/forecast.png" alt="Forecast with CI" width="80%"></p>

### Anomalies Tab
<p align="center"><img src="assets/anomalies.png" alt="Anomaly Detection" width="80%"></p>

### Export Tab
<p align="center"><img src="assets/exports.png" alt="BI-ready Exports" width="80%"></p>

---

## 🚀 How to Run

**Option 1 — Run online**  
Use the live demo: [Streamlit app](https://retail-bi-simulator-ncsinurqpmtty7obwwcwmw.streamlit.app)

**Option 2 — Run locally**
```bash
git clone https://github.com/Giuseppe552/retail-bi-simulator.git
cd retail-bi-simulator
pip install -r requirements.txt
streamlit run streamlit_app.py
````

Input format (case-insensitive column names):

* `InvoiceDate` or `Date`
* `Quantity`
* `UnitPrice` or `Price`
* Optional: `Country`, `Description` / `StockCode`

---

## 📈 Why This Matters

* **For retail teams** → plan stock, detect anomalies, and forecast cash flows.
* **For BI developers** → automate the messy cleaning + pipeline step.
* **For hiring managers** → demonstrates ability to design an end-to-end BI workflow, not just code snippets.

---

## 🔮 Next Steps

* Add support for **multiple forecast models** (Prophet, LSTM).
* Deploy BI exports straight to **Power BI Service / Tableau Online**.
* Build **multi-retailer benchmarking dashboard**.

---

👨‍💻 Built by [Giuseppe552](https://github.com/Giuseppe552) — feedback welcome!

```


