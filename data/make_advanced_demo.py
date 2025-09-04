#!/usr/bin/env python3
import numpy as np, pandas as pd
from pathlib import Path
rng = np.random.default_rng(42)

# ---- Config ----
months = pd.date_range("2023-01-01","2024-12-31", freq="D")
countries = ["United Kingdom","United States","Germany","France","Spain"]
categories = {
    "Laptop":         {"price_mu": 950, "price_sd": 120, "qty_mu": 1.1},
    "Headphones":     {"price_mu": 80,  "price_sd": 20,  "qty_mu": 1.4},
    "Notebook":       {"price_mu": 12,  "price_sd": 3,   "qty_mu": 3.0},
    "Office Chair":   {"price_mu": 180, "price_sd": 35,  "qty_mu": 1.2},
    "Tea Mug":        {"price_mu": 9,   "price_sd": 2,   "qty_mu": 2.4},
    "Desk Lamp":      {"price_mu": 35,  "price_sd": 8,   "qty_mu": 1.6},
}
country_weights = np.array([0.40, 0.25, 0.15, 0.12, 0.08])  # UK heavy
cat_names = list(categories.keys())
cat_weights = np.array([0.22, 0.18, 0.16, 0.14, 0.16, 0.14])

# Seasonality helpers
def year_seasonality(d):
    # strong Q4 uplift, small summer dip
    m = d.month
    return {
        1:0.95, 2:0.95, 3:1.00, 4:1.02, 5:1.00, 6:0.96,
        7:0.92, 8:0.96, 9:1.08, 10:1.15, 11:1.40, 12:1.60
    }[m]

def weekday_seasonality(d):
    # weekends lower, midweek higher (retail B2C online-ish)
    wd = d.weekday()
    return [0.9,1.05,1.10,1.10,1.05,0.95,0.85][wd]

promos = {
    # yyyy-mm-dd ranges with multipliers
    ("2023-11-24","2023-11-27"): 1.8,  # Black Friday wknd
    ("2024-11-29","2024-12-02"): 1.9,
    ("2023-12-26","2023-12-31"): 1.5,  # Boxing Week
    ("2024-12-26","2024-12-31"): 1.5,
    ("2023-08-20","2023-09-10"): 1.25, # Back-to-school
    ("2024-08-20","2024-09-10"): 1.25,
}
promo_dates = []
for (a,b),mult in promos.items():
    promo_dates.append((pd.to_datetime(a), pd.to_datetime(b), mult))

def promo_mult(d):
    for a,b,m in promo_dates:
        if a <= d <= b: return m
    return 1.0

# Intentionally inject a few anomalies (data glitches / extreme promos)
forced_anomalies = {
    pd.to_datetime("2023-03-15"): 0.4,  # sudden drop (stockout)
    pd.to_datetime("2024-03-20"): 2.2,  # unexplained spike
}

def anomaly_mult(d):
    return forced_anomalies.get(d, 1.0)

# ---- Synthesize transactions per day ----
rows = []
for d in months:
    base = 250  # baseline daily lines
    lam = base * year_seasonality(d) * weekday_seasonality(d) * promo_mult(d) * anomaly_mult(d)
    n_lines = max(1, rng.poisson(lam))
    # sample countries & categories in vectorized way
    ctry = rng.choice(countries, size=n_lines, p=country_weights/country_weights.sum())
    cats = rng.choice(cat_names, size=n_lines, p=cat_weights/cat_weights.sum())

    for country, cat in zip(ctry, cats):
        cfg = categories[cat]
        price = max(1.0, rng.normal(cfg["price_mu"], cfg["price_sd"]))
        # heavier quantity for low-priced goods
        qty = max(1, int(np.round(np.clip(rng.normal(cfg["qty_mu"], 0.8), 0.5, 6))))
        # noise on promos: some extra units sold
        if promo_mult(d) > 1.0 and rng.random() < 0.15:
            qty += rng.integers(1,3)
        # randomly add small returns (negative qty) 1% of lines
        if rng.random() < 0.01:
            qty = -qty

        rows.append({
            "InvoiceDate": d.strftime("%Y-%m-%d"),
            "Quantity": qty,
            "UnitPrice": round(price,2),
            "Country": country,
            "Description": cat
        })

df = pd.DataFrame(rows)
# sanity: remove zero revenue lines
df = df[~((df["Quantity"]==0) | (df["UnitPrice"]<=0))]
out = Path("advanced_transactions.csv")
df.to_csv(out, index=False)
print(f"Wrote {out} with {len(df):,} rows")
