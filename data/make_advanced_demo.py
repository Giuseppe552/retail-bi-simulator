#!/usr/bin/env python3
import numpy as np, pandas as pd

def build_demo_df(n_rows: int = 20000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2024-01-01","2025-06-30", freq="D")
    countries = ["United Kingdom","United States","Germany","France","Spain"]
    ctry_p = np.array([0.40, 0.25, 0.15, 0.12, 0.08])

    products = {
        "Laptop":       (950,120,1.1),
        "Headphones":   ( 80, 20,1.4),
        "Notebook":     ( 12,  3,3.0),
        "Office Chair": (180, 35,1.2),
        "Tea Mug":      (  9,  2,2.4),
        "Desk Lamp":    ( 35,  8,1.6),
    }
    prod_names = np.array(list(products.keys()))
    prod_p = np.array([0.30,0.18,0.14,0.12,0.14,0.12])

    idx = rng.choice(len(dates), size=n_rows, replace=True)
    inv_date = dates[idx]
    country = rng.choice(countries, p=ctry_p, size=n_rows)
    pidx = rng.choice(len(prod_names), p=prod_p, size=n_rows)
    prod = prod_names[pidx]

    price_mu = np.array([products[p][0] for p in prod])
    price_sd = np.array([products[p][1] for p in prod])
    qty_mu   = np.array([products[p][2] for p in prod])

    unit_price = np.clip(rng.normal(price_mu, price_sd), 1, None).round(2)
    quantity   = np.clip(rng.poisson(qty_mu).astype(float), 1, None)

    mm = pd.to_datetime(inv_date).month
    seas = np.where(mm.isin([11,12]), 1.25, np.where(mm==1, 0.9, 1.0))

    promo_mask = rng.random(n_rows) < 0.05
    unit_price = unit_price * np.where(promo_mask, 0.9, 1.0)

    spike_mask = (mm==12) & (rng.random(n_rows)<0.02)
    drop_mask  = (mm== 2) & (rng.random(n_rows)<0.02)
    quantity   = quantity * np.where(spike_mask, 4.0, 1.0)
    unit_price = unit_price * np.where(drop_mask, 0.7, 1.0)

    df = pd.DataFrame({
        "InvoiceDate": inv_date,
        "Quantity": quantity.astype(int),
        "UnitPrice": (unit_price * seas).astype(float),
        "Country": country,
        "Description": prod,
    })
    return df

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    out = "data/advanced_transactions.csv"
    df = build_demo_df()
    df.to_csv(out, index=False)
    print(f"Wrote {out} with {len(df):,} rows")
