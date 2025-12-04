# analysis.py
# Power-of-Two inventory analysis (MIT ESD.273J case)
# This script creates the dataset, computes EOQ and power-of-two policy,
# and saves a comparison plot.

import math, os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

output_dir = "."
# Reconstructed dataset (from the MIT PDF). Coordinates are in hundreds of miles.
data = [
    (0, 4.7, 3.9, 1,   None),   # warehouse (demand will be set as sum of retailers later)
    (1, 0.2, 5.6, 10,  200),
    (2, 2.7, 0.9, 10,  700),
    (3, 6.1, 4.3, 5,   50),
    (4, 7.1, 7.9, 2,   20),
    (5, 8.4, 2.4, 10,  300),
    (6, 5.7, 6.1, 2,   20),
    (7, 8.7, 6.2, 2,   40),
    (8, 1.5, 4.6, 10,  500),
    (9, 9.9, 4.3, 3,   60),
    (10,7.9, 2.9, 10,  70),
    (11,9.7, 2.5, 3,   80),
    (12,6.5, 1.5, 2,   700),
    (13,3.3, 5.8, 2,   900),
    (14,3.6, 9.1, 10,  600),
    (15,6.9, 0.6, 14,  500),
    (16,5.0, 0.5, 5,   1200),
    (17,5.7, 6.8, 10,  600),
    (18,3.7, 8.0, 3,   300),
    (19,3.7, 5.6, 5,   400),
    (20,5.7, 4.7, 4,   500),
    (21,6.3, 3.2, 3,   50),
    (22,3.4, 4.4, 5,   100),
    (23,5.1, 3.8, 3,   70),
    (24,4.0, 6.0, 5,   150),
]

df = pd.DataFrame(data, columns=["idx","x_hundreds","y_hundreds","H","D"])
df["x_miles"] = df["x_hundreds"] * 100.0
df["y_miles"] = df["y_hundreds"] * 100.0

# Compute warehouse demand as sum of retailer demands (rows 1..24)
total_retail_demand = df.loc[df["idx"]!=0, "D"].sum()
df.loc[df["idx"]==0, "D"] = total_retail_demand

# Parameters from problem
alpha = 0.32  # $ per mile
C = 150.0     # fixed shipping component per retailer delivery ($)
S0 = 400.0    # warehouse ordering fixed cost ($)

# Compute distances between warehouse (idx 0) and each retailer
wx = df.loc[df["idx"]==0, "x_miles"].values[0]
wy = df.loc[df["idx"]==0, "y_miles"].values[0]
def euclid(x,y):
    return math.hypot(x-wx, y-wy)

df["distance_miles"] = df.apply(lambda r: euclid(r["x_miles"], r["y_miles"]), axis=1)
df.loc[df["idx"]==0, "distance_miles"] = 0.0

# Setup cost per retailer delivery
df["S_i"] = df["distance_miles"] * alpha + C
df.loc[df["idx"]==0, "S_i"] = np.nan  # warehouse has separate S0

# Continuous EOQ (optimal Q) and cycle time T (years)
df["Q_eoq"] = np.sqrt(2.0 * df["D"] * df["S_i"] / df["H"])
df["T_eoq_years"] = df["Q_eoq"] / df["D"]

# Base period = 1 week = 1/52 year
base = 1.0/52.0
k_values = list(range(-1, 8))
periods = sorted([base * (2**k) for k in k_values])

def nearest_power_of_two(T):
    periods_arr = np.array(periods)
    idx = np.abs(periods_arr - T).argmin()
    return periods_arr[idx]

df["T_pow2"] = df["T_eoq_years"].apply(lambda t: nearest_power_of_two(t))
df["Q_pow2"] = df["T_pow2"] * df["D"]

# Cost components
df["orders_per_year_eoq"] = df["D"] / df["Q_eoq"]
df["orders_per_year_pow2"] = df["D"] / df["Q_pow2"]

df["ordering_cost_eoq"] = df["orders_per_year_eoq"] * df["S_i"]
df["holding_cost_eoq"] = df["H"] * (df["Q_eoq"]/2.0)
df["total_cost_eoq"] = df["ordering_cost_eoq"] + df["holding_cost_eoq"]

df["ordering_cost_pow2"] = df["orders_per_year_pow2"] * df["S_i"]
df["holding_cost_pow2"] = df["H"] * (df["Q_pow2"]/2.0)
df["total_cost_pow2"] = df["ordering_cost_pow2"] + df["holding_cost_pow2"]

# Warehouse approximation
warehouse_orders_eoq = df.loc[df["idx"]!=0, "orders_per_year_eoq"].sum()
warehouse_orders_pow2 = df.loc[df["idx"]!=0, "orders_per_year_pow2"].sum()

avg_warehouse_inventory_eoq = (df.loc[df["idx"]!=0, "Q_eoq"].sum()) / 2.0
avg_warehouse_inventory_pow2 = (df.loc[df["idx"]!=0, "Q_pow2"].sum()) / 2.0

warehouse_h = df.loc[df["idx"]==0, "H"].values[0]

warehouse_ordering_cost_eoq = S0 * warehouse_orders_eoq
warehouse_ordering_cost_pow2 = S0 * warehouse_orders_pow2

warehouse_holding_cost_eoq = warehouse_h * avg_warehouse_inventory_eoq
warehouse_holding_cost_pow2 = warehouse_h * avg_warehouse_inventory_pow2

total_cost_eoq_all = df.loc[df["idx"]!=0, "total_cost_eoq"].sum() + warehouse_ordering_cost_eoq + warehouse_holding_cost_eoq
total_cost_pow2_all = df.loc[df["idx"]!=0, "total_cost_pow2"].sum() + warehouse_ordering_cost_pow2 + warehouse_holding_cost_pow2

percent_diff = (total_cost_pow2_all - total_cost_eoq_all) / total_cost_eoq_all * 100.0

# Save CSV and outputs
csv_path = os.path.join(output_dir, "mit_power_of_two_dataset.csv")
df.to_csv(csv_path, index=False)

print("=== Summary ===")
print("Total annual cost (continuous EOQ, retailers + warehouse): ${:,.2f}".format(total_cost_eoq_all))
print("Total annual cost (power-of-two rounding):           ${:,.2f}".format(total_cost_pow2_all))
print("Power-of-two is {:.2f}% {} continuous EOQ".format(abs(percent_diff), "higher than" if percent_diff>0 else "lower than"))

# Plot per-retailer comparison
rets = df.loc[df["idx"]!=0].copy().sort_values("idx")
x = rets["idx"]
plt.figure(figsize=(10,6))
plt.plot(x, rets["total_cost_eoq"], marker='o', label="EOQ cost (retailer)")
plt.plot(x, rets["total_cost_pow2"], marker='s', label="Power-of-two cost (retailer)")
plt.xlabel("Retailer idx")
plt.ylabel("Annual cost ($)")
plt.title("Retailer annual cost: EOQ vs Power-of-two")
plt.legend()
plt.grid(True)
plot_path = os.path.join(output_dir, "retailer_costs_compare.png")
plt.savefig(plot_path, dpi=150)
plt.close()

# Save summary json
summary = {
    "total_cost_eoq_all": total_cost_eoq_all,
    "total_cost_pow2_all": total_cost_pow2_all,
    "percent_diff": percent_diff,
    "csv_path": csv_path,
    "plot_path": plot_path
}
with open(os.path.join(output_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("Outputs saved:")
print(" - CSV:", csv_path)
print(" - Plot:", plot_path)
print(" - Summary JSON:", os.path.join(output_dir, "summary.json"))
