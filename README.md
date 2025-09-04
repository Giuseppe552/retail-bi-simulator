# 🛒 Retail BI Simulator

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.x-yellow.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-green.svg)
![Plotly](https://img.shields.io/badge/Plotly-Dashboards-lightblue.svg)
![Status](https://img.shields.io/badge/Stage-Prototype-orange.svg)

**Business context:** Retail companies sit on messy transaction logs. Turning them into **business-ready insights** is the core job of BI & Data Analytics.  

This project simulates a **BI workflow** end-to-end:
- Cleans raw retail transactions  
- Aggregates into **monthly sales by country & category**  
- Forecasts the next 3 months of revenue (ARIMA)  
- Builds a **dashboard (HTML)** and **executive summary report**  

---

## 📊 Example Outputs

**1. Forecast chart**
<p align="center"><img src="outputs/forecast.png" alt="Revenue Forecast" width="70%"></p>

**2. Executive report (excerpt)**
```

# Retail BI — Executive Summary

* Top country (last 3 months): United Kingdom — revenue £1,140
* Top category (last 3 months): Notebook — revenue £715
* Next 3 months forecast (point total): £3,350

Implications:

* Double down on top categories; monitor tail for stockouts/markdown risk.
* Use forecast to plan buys and working capital.

````

**3. Interactive dashboard**
👉 [bi_dashboard.html](./outputs/bi_dashboard.html) (open locally)

---

## 🚀 How to Run

1. Clone repo & install deps
   ```bash
   git clone https://github.com/Giuseppe552/retail-bi-simulator.git
   cd retail-bi-simulator
   pip install -r requirements.txt
   ````

2. Run with a CSV file (`InvoiceDate, Quantity, UnitPrice, Country, Description`)

   ```bash
   python3 retail_bi.py data/transactions.csv
   ```

3. Outputs in `outputs/`:

   * `cleaned_sales.csv` — cleaned + aggregated
   * `forecast.png` — forecast chart
   * `bi_dashboard.html` — interactive BI dashboard
   * `report.txt` — exec summary

---

## 💡 Why This Matters

* **For clients:** Automates the path from raw data → insight → decision.
* **For analysts:** Mirrors the role of BI Developers & Data Consultants.
* **For hiring managers:** Shows practical ability to clean, aggregate, visualize, and forecast real-world data.

---

## 🔮 Next Steps

* Connect to real retail datasets (e.g., Kaggle Online Retail).
* Add Power BI / Tableau export connectors.
* Extend forecast to include confidence bands & anomaly detection.

````

