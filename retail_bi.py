#!/usr/bin/env python3
import sys, re, sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go

# ---------- Loading & cleaning ----------

def load_transactions(path_or_df, low_memory=False):
    """
    Accepts a path, file-like object, or a pandas DataFrame.
    Returns a normalized transactions DataFrame with:
      InvoiceDate, Quantity, UnitPrice, Country, Description, Category, Revenue, Month
    """
    import pandas as pd
    # 1) Load
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = pd.read_csv(path_or_df, low_memory=low_memory, encoding="utf-8")

    # 2) Normalise column names (case-insensitive)
    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*options):
        for o in options:
            if o.lower() in cols: return cols[o.lower()]
        return None

    date_col   = pick("InvoiceDate","Date")
    qty_col    = pick("Quantity","Qty")
    price_col  = pick("UnitPrice","Price","Unit Price")
    country_c  = pick("Country","Market")
    desc_col   = pick("Description","StockCode","Item","Product","Category")

    rename = {}
    if date_col:   rename[date_col]  = "InvoiceDate"
    if qty_col:    rename[qty_col]   = "Quantity"
    if price_col:  rename[price_col] = "UnitPrice"
    if country_c:  rename[country_c] = "Country"
    if desc_col:   rename[desc_col]  = "Description"
    if rename: df = df.rename(columns=rename)

    # 3) Types & cleaning
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"]    = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"]   = pd.to_numeric(df["UnitPrice"], errors="coerce")
    if "Country" not in df:     df["Country"] = "Unknown"
    if "Description" not in df: df["Description"] = "Misc"

    df = df.dropna(subset=["InvoiceDate","Quantity","UnitPrice"])
    df = df[df["Quantity"]>0]
    df = df[df["UnitPrice"]>0]

    # 4) Derive category if missing
    if "Category" not in df.columns:
        cat = df["Description"].astype(str).str.extract(r"([A-Za-z ]+)")[0].str.strip()
        df["Category"] = cat.replace("", "Other")

    # 5) Derived fields
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["Month"]   = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

    return df
def monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    df["Month"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    grouped = (df.groupby(["Month", "Country", "Category"], as_index=False)["Revenue"]
                 .sum()
                 .sort_values(["Month", "Revenue"], ascending=[True, False]))
    return grouped

def totals_series(monthly: pd.DataFrame) -> pd.Series:
    return (monthly.groupby("Month")["Revenue"].sum()
            .sort_index()
            .asfreq("MS", fill_value=0.0))

# ---------- Forecast + CI + anomalies ----------

def forecast_with_ci(total: pd.Series, steps: int = 3):
    """
    Input: monthly revenue Series with a monthly DateTimeIndex (MS).
    Output: (forecast_df, residuals)
      - forecast_df: DataFrame with index as future months and columns yhat, lower, upper
      - residuals: in-sample residuals (Series) for anomaly detection
    """
    # clean to float Series
    s = pd.to_numeric(total, errors="coerce").astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()

    # Edge cases
    if s.empty:
        idx = pd.date_range(pd.Timestamp.today().to_period("M").to_timestamp(), periods=steps, freq="MS")
        fc = pd.DataFrame({"yhat":[0.0]*steps, "lower":[0.0]*steps, "upper":[0.0]*steps}, index=idx)
        residuals = s
        return fc, residuals

    if len(s) < 8:
        idx = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq="MS")
        base = float(s.iloc[-1])
        fc = pd.DataFrame({"yhat":[base]*steps, "lower":[base]*steps, "upper":[base]*steps}, index=idx)
        residuals = s - s.rolling(3, min_periods=1).mean()
        return fc, residuals

    # SARIMAX forecast + CI
    model = SARIMAX(s, order=(1,1,1), seasonal_order=(0,1,1,12),
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    residuals = res.resid

    fc_res = res.get_forecast(steps=steps)
    pred = fc_res.predicted_mean.clip(lower=0)
    ci   = fc_res.conf_int(alpha=0.20)  # 80% CI
    ci.columns = ["lower", "upper"]

    out = pd.DataFrame({
        "yhat":  pred,
        "lower": ci["lower"].clip(lower=0),
        "upper": ci["upper"].clip(lower=0),
    })
    return out, residuals



# ---------- Visuals ----------

def plot_forecast(total: pd.Series, fc: pd.DataFrame, outpath: Path):
    plt.figure(figsize=(10,6))
    plt.plot(total.index, total.values, label="Actual")
    if not fc.empty:
        # CI band
        plt.fill_between(fc.index, fc["lower"], fc["upper"], alpha=0.25, label="80% CI")
        plt.plot(fc.index, fc["yhat"], label="Forecast")
    plt.title("Monthly Revenue — 3-Month Forecast with 80% CI")
    plt.xlabel("Month"); plt.ylabel("Revenue")
    plt.legend(); plt.tight_layout()
    plt.savefig(outpath, dpi=180); plt.close()

def build_dashboard(monthly: pd.DataFrame,
                    total: pd.Series,
                    fc: pd.DataFrame,
                    anomalies: pd.DataFrame,
                    outpath: Path):
    # Time series + forecast + anomalies
    ts = go.Figure()
    ts.add_trace(go.Scatter(x=total.index, y=total.values, name="Actual", mode="lines"))
    if not fc.empty:
        ts.add_trace(go.Scatter(x=fc.index, y=fc["yhat"], name="Forecast", mode="lines"))
        ts.add_trace(go.Scatter(x=list(fc.index)+list(fc.index[::-1]),
                                y=list(fc["upper"])+list(fc["lower"][::-1]),
                                fill="toself", name="80% CI", mode="lines", opacity=0.2))
    # anomaly markers
    anom_pts = anomalies[anomalies["Anomaly"] == True]
    if not anom_pts.empty:
        ts.add_trace(go.Scatter(x=anom_pts["Month"], y=total.reindex(anom_pts["Month"]).values,
                                mode="markers", name="Anomalies", marker=dict(size=10, symbol="x")))
    ts.update_layout(title="Revenue (Actual + Forecast + CI + Anomalies)",
                     xaxis_title="Month", yaxis_title="Revenue")

    # Last 3 months breakdowns
    latest = monthly["Month"].max()
    last3  = monthly[monthly["Month"] >= (latest - pd.offsets.MonthBegin(2))]
    top_country = last3.groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(10)
    top_cat     = last3.groupby("Category")["Revenue"].sum().sort_values(ascending=False).head(10)

    bar_c = go.Figure(go.Bar(x=top_country.index, y=top_country.values, name="Countries"))
    bar_c.update_layout(title="Top Countries (Last 3 Months)", xaxis_title="Country", yaxis_title="Revenue")

    bar_cat = go.Figure(go.Bar(x=top_cat.index, y=top_cat.values, name="Categories"))
    bar_cat.update_layout(title="Top Categories (Last 3 Months)", xaxis_title="Category", yaxis_title="Revenue")

    html = "<h1>Retail BI Dashboard</h1>"
    html += ts.to_html(full_html=False, include_plotlyjs="cdn")
    html += bar_c.to_html(full_html=False, include_plotlyjs=False)
    html += bar_cat.to_html(full_html=False, include_plotlyjs=False)
    outpath.write_text(html, encoding="utf-8")

# ---------- BI exports (Power BI / Tableau friendly) ----------

def export_bi(monthly: pd.DataFrame, total: pd.Series, anomalies: pd.DataFrame, outdir: Path):
    bi_dir = outdir / "bi_exports"
    bi_dir.mkdir(parents=True, exist_ok=True)

    # Star-ish CSVs
    fact = monthly.rename(columns={"Month":"Date"}).copy()
    fact["Date"] = pd.to_datetime(fact["Date"])
    fact.to_csv(bi_dir / "fact_sales.csv", index=False)

    dim_date = pd.DataFrame({"Date": total.index})
    dim_date["Year"]  = dim_date["Date"].dt.year
    dim_date["Month"] = dim_date["Date"].dt.month
    dim_date["YearMonth"] = dim_date["Date"].dt.strftime("%Y-%m")
    dim_date.to_csv(bi_dir / "dim_date.csv", index=False)

    total.reset_index().rename(columns={"index":"Date","Revenue":"TotalRevenue"}).to_csv(bi_dir / "total_timeseries.csv", index=False)

    anomalies.to_csv(bi_dir / "anomalies.csv", index=False)

    # SQLite export (Power BI/Tableau can connect)
    db = outdir / "retail_bi.sqlite"
    with sqlite3.connect(db) as conn:
        fact.to_sql("fact_sales", conn, if_exists="replace", index=False)
        dim_date.to_sql("dim_date", conn, if_exists="replace", index=False)
        anomalies.to_sql("anomalies", conn, if_exists="replace", index=False)

# ---------- Exec summary ----------

def write_exec_report(monthly: pd.DataFrame, total: pd.Series, fc: pd.DataFrame, anomalies: pd.DataFrame, outpath: Path):
    latest = monthly["Month"].max()
    last3 = monthly[monthly["Month"] >= (latest - pd.offsets.MonthBegin(2))]
    top_country = last3.groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(1)
    top_cat = last3.groupby("Category")["Revenue"].sum().sort_values(ascending=False).head(1)

    lines = []
    lines.append("Retail BI — Executive Summary")
    lines.append("================================")
    if not top_country.empty:
        lines.append(f"- Top country (last 3 months): {top_country.index[0]} — £{float(top_country.iloc[0]):,.0f}")
    if not top_cat.empty:
        lines.append(f"- Top category (last 3 months): {top_cat.index[0]} — £{float(top_cat.iloc[0]):,.0f}")
    if not fc.empty:
        lines.append(f"- Next 3 months forecast: £{fc['yhat'].sum():,.0f} (point total), CI shown on chart")
    if (anomalies["Anomaly"] == True).any():
        n = int((anomalies['Anomaly'] == True).sum())
        lines.append(f"- Anomalies detected: {n} outliers in history (flagged by z-score ≥ 3)")
    lines.append("")
    lines.append("Implications:")
    lines.append("- Prioritise top markets/categories; investigate anomaly months for promo/stockout effects.")
    lines.append("- Use forecast & CI to set buy plans and cash buffers; alert if actuals breach CI band.")
    outpath.write_text("\n".join(lines), encoding="utf-8")

# ---------- Main ----------

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 retail_bi.py data/transactions.csv")
        sys.exit(1)
    src = Path(sys.argv[1])
    if not src.exists():
        print(f"File not found: {src}"); sys.exit(1)

    outdir = Path("outputs"); outdir.mkdir(parents=True, exist_ok=True)

    df = load_transactions(src)
    monthly = monthly_agg(df)
    monthly.to_csv(outdir / "cleaned_sales.csv", index=False)

    total = totals_series(monthly)
    fc, residuals = forecast_with_ci(total, steps=3)
    anomalies = detect_anomalies(residuals)

    plot_forecast(total, fc, outdir / "forecast.png")
    build_dashboard(monthly, total, fc, anomalies, outdir / "bi_dashboard.html")
    export_bi(monthly, total, anomalies, outdir)
    write_exec_report(monthly, total, fc, anomalies, outdir / "report.txt")

    print("Done.")
    print("Wrote:", outdir / "cleaned_sales.csv")
    print("Wrote:", outdir / "forecast.png")
    print("Wrote:", outdir / "bi_dashboard.html")
    print("Wrote:", outdir / "bi_exports/* and retail_bi.sqlite")
    print("Wrote:", outdir / "report.txt")

if __name__ == "__main__":
    main()

def export_bi_in_memory(cleaned: pd.DataFrame,
                        monthly: pd.DataFrame,
                        total: pd.Series,
                        fc: pd.DataFrame,
                        anomalies: pd.DataFrame) -> dict[str, bytes]:
    tx = cleaned.to_csv(index=False).encode("utf-8")
    sales = monthly.rename(columns={"Month":"Date"}).to_csv(index=False).encode("utf-8")
    anom = anomalies.to_csv(index=False).encode("utf-8")
    fcsv = fc.reset_index().rename(columns={"index":"Month"}).to_csv(index=False).encode("utf-8")
    return {
        "transactions_csv": tx,
        "sales_csv": sales,
        "anom_csv": anom,
        "forecast_csv": fcsv,
    }

