"""Generate a demo e-commerce dataset: data/sample_sales.csv"""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 1500
subcats = {"Clothing": ["Saree", "Kurti", "Shirt", "Trousers", "Skirt"],
           "Electronics": ["Printers", "Phones", "Accessories", "Electronic Games"],
           "Furniture": ["Bookcases", "Tables", "Chairs", "Furnishings"]}
cats = rng.choice(list(subcats), n, p=[0.63, 0.21, 0.16])
sub = [rng.choice(subcats[c]) for c in cats]
states = rng.choice(["Maharashtra", "Madhya Pradesh", "Uttar Pradesh", "Delhi", "Gujarat", "Punjab"],
                    n, p=[0.30, 0.22, 0.15, 0.10, 0.13, 0.10])
cust = rng.choice(["Harivansh", "Madhav", "Madan Mohan", "Shiva", "Aarav", "Diya", "Kabir", "Meera"], n)
dates = pd.to_datetime("2018-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
amount = rng.gamma(2.0, 150, n).round(0) + 20
profit = (amount * rng.normal(0.08, 0.25, n)).round(0)
df = pd.DataFrame({
    "Order ID": [f"B-{25000 + i}" for i in range(n)],
    "Order Date": dates.strftime("%d-%m-%Y"),
    "CustomerName": cust, "State": states,
    "City": rng.choice(["Mumbai", "Indore", "Lucknow", "Delhi", "Surat", "Amritsar"], n),
    "Category": cats, "Sub-Category": sub,
    "Amount": amount, "Profit": profit,
    "Quantity": rng.integers(1, 8, n),
    "PaymentMode": rng.choice(["COD", "UPI", "Debit Card", "Credit Card", "EMI"], n, p=[0.44, 0.21, 0.13, 0.12, 0.10]),
})
# add a little dirty data on purpose (to test cleaning)
df.loc[rng.choice(n, 25, replace=False), "Profit"] = np.nan
df.loc[rng.choice(n, 15, replace=False), "State"] = np.nan
df = pd.concat([df, df.sample(20, random_state=1)], ignore_index=True)
df.to_csv("data/sample_sales.csv", index=False)
print("Saved data/sample_sales.csv", df.shape)
