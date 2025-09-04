#!/usr/bin/env python3
import numpy as np, pandas as pd
from pathlib import Path

def build_demo_df(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---- Config ----
    days = pd.date_range("2023-01-01","2024-12-31", freq="D")
    countries = ["United Kingdom","United States","Germany","France","Spain"]
    categories = {
        "Laptop":         {"price_mu": 950, "price_sd": 120, "qty_mu": 1.1},
        "Headphones":     {"price_mu": 80,  "price_sd": 20,  "qty_mu": 1.4},
        "Notebook":       {"price_mu": 12,  "price_sd": 3,   "qty_mu": 3.0},
        "Office Chair":   {"price_mu": 180, "price_sd": 35,  "qty_mu": 1.2},
        "Tea Mug":        {"price_mu": 9,   "price_sd": 2,   "qty_mu": 2.4},
        "Desk Lamp":      {"price_mu": 35,  "price_sd": 8,   "qty_mu": 1.6},
    }
    country_w = np.array([0.40, 0.25, 0.15, 0.12, 0.08])  # UK heavy
    country_w = country_w / country_w.sum()

    # monthly seasonality (Dec spike, Jan dip)
    seasonality = {
        1: 0.85, 2: 0.90, 3: 0.95, 4: 1.00, 5: 1.05, 6: 1.10,
        7: 1.05, 8: 1.00, 9: 1.05, 10: 1.10, 11: 1.20, 12: 1.40
    }

    rows = []
    invoice_id = 100000
    for d in days:
        # base daily order count with noise + monthly seasonal factor
        base_orders = rng.poisson(140)
        mult = seasonality[d.month]
        orders = int(max(20, base_orders * mult))

        # promo spike in 2023-11 and 2024-11 (Black Friday)
        if (d.month == 11) and (d.year in [2023, 2024]):
            orders = int(orders * 2.2)

        # stock-out dip in Feb 2024
        if (d.year == 2024 and d.month == 2):
            orders = int(orders * 0.55)

        for _ in range(orders):
            country = rng.choice(countries, p=country_w)
            cat = rng.choice(list(categories.keys()))
            cfg = categories[cat]
            qty = max(1, int(rng.normal(cfg["qty_mu"], 0.6)))
            price = max(1.0, rng.normal(cfg["price_mu"], cfg["price_sd"]))

            # mild trend upward over time
            months_since_start = (d.year-2023)*12 + (d.month-1)
            price *= (1.0 + 0.003 * months_since_start)

            desc = f"{cat} - {country}"
            rows.append({
                "InvoiceNo": invoice_id,
                "InvoiceDate": d.strftime("%Y-%m-%d %H:%M:%S"),
                "Quantity": qty,
                "UnitPrice": round(price, 2),
                "Country": country,
                "Description": desc,
            })
            invoice_id += 1

    df = pd.DataFrame(rows)
    # clean oddities just in case
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].reset_index(drop=True)
    return df

def main(output_path: Path | None = None, return_df: bool = False):
    df = build_demo_df()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Wrote {output_path} with {len(df):,} rows")
    if return_df:
        return df

if __name__ == "__main__":
    out = Path("data/advanced_transactions.csv")
    main(output_path=out, return_df=False)
