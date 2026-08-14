"""
Step 8 - Predicted vs Actual Power over time.

Loads the trained XGBoost model and the test set, runs predictions, and plots
actual vs predicted power on a time axis so you can see WHERE in the flight the
model agrees or diverges.

Handles ANY number of held-out flights. Each test flight gets its own figure
with its own time axis re-zeroed to that flight's start, so flights recorded on
different days do not get spread across a multi-week x axis.

Input  : ML/ml_data/test.csv
         ML/ml_data/feature_cols.txt
         ML/ml_data/xgboost_model.json
Output : ML/ml_data/plot_power_time_<FLIGHT>.png   (one per test flight)
         ML/ml_data/test_flight_metrics.csv        (per-flight R2 / MAE / RMSE)

Note: test.csv features are scaled, but `power` and `timestamp` are excluded
from scaling in 05_prepare_ml.py, so Watts and seconds stay interpretable.

Run: python 08_plot_power_time.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =====================================================
# PATHS
# =====================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
# Data folder. Override with the ML_DATA_DIR env var to run a variant dataset,
# e.g.  ML_DATA_DIR=ML/ml_data_inflight python 08_plot_power_time.py
INPUT_FOLDER = os.path.join(SCRIPT_DIR, os.environ.get("ML_DATA_DIR",
                                                       os.path.join("ML", "ml_data")))

# =====================================================
# LOAD
# =====================================================
test         = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
feature_cols = open(os.path.join(INPUT_FOLDER, "feature_cols.txt")).read().strip().split("\n")

model = XGBRegressor()
model.load_model(os.path.join(INPUT_FOLDER, "xgboost_model.json"))

test_flights = sorted(test["flight_id"].unique().tolist())
print(f"Test flights : {test_flights}  ({len(test)} rows total)\n")


# =====================================================
# PLOT ONE FLIGHT
# =====================================================
def plot_flight(df, flight_id, out_path):
    """Actual vs predicted power over time for a single flight, plus residuals."""

    # Sort by time so the line is drawn in chronological order
    df = df.sort_values("timestamp")

    X = df[feature_cols].values
    y = df["power"].values

    # Time axis re-zeroed to THIS flight's start
    t = df["timestamp"].values
    t = t - t[0]

    y_pred = model.predict(X)

    r2   = r2_score(y, y_pred)
    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    print(f"  {flight_id} : R2 = {r2:.4f}  |  MAE = {mae:.1f} W  |  "
          f"RMSE = {rmse:.1f} W  |  {len(df)} rows  |  {t[-1]:.0f} s")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"Predicted vs Actual Power over Time - {flight_id} (unseen flight)",
                 fontsize=13, fontweight="bold")

    # ---- Top panel: actual vs predicted ----
    ax = axes[0]
    ax.plot(t, y,      color="steelblue", linewidth=1.2, label="Actual power",    alpha=0.9)
    ax.plot(t, y_pred, color="tomato",    linewidth=1.2, label="Predicted power", alpha=0.9)
    ax.set_ylabel("Power (W)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"R² = {r2:.4f}  |  MAE = {mae:.1f} W  |  RMSE = {rmse:.1f} W",
                 fontsize=10)

    # ---- Bottom panel: residual (actual - predicted) ----
    residual = y - y_pred
    ax2 = axes[1]
    ax2.fill_between(t, residual, 0, where=(residual >= 0),
                     color="steelblue", alpha=0.4, label="Under-predicted")
    ax2.fill_between(t, residual, 0, where=(residual < 0),
                     color="tomato", alpha=0.4, label="Over-predicted")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Residual (W)", fontsize=11)
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Residual = Actual - Predicted", fontsize=10)

    axes[0].set_xlim(t[0], t[-1])
    axes[1].set_xlim(t[0], t[-1])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {"flight_id": flight_id, "n_rows": len(df),
            "duration_s": round(float(t[-1]), 1),
            "r2": round(float(r2), 4),
            "mae_w": round(float(mae), 1),
            "rmse_w": round(float(rmse), 1)}


# =====================================================
# RUN - one figure per held-out flight
# =====================================================
rows = []
for fid in test_flights:
    out_path = os.path.join(INPUT_FOLDER, f"plot_power_time_{fid}.png")
    rows.append(plot_flight(test[test["flight_id"] == fid], fid, out_path))
    print(f"     saved -> {out_path}")

# ---- Pooled metrics across the whole test set (comparable to 06/07) ----
y_all      = test["power"].values
y_all_pred = model.predict(test[feature_cols].values)
pooled = {"flight_id": "ALL (pooled)", "n_rows": len(test), "duration_s": np.nan,
          "r2":     round(float(r2_score(y_all, y_all_pred)), 4),
          "mae_w":  round(float(mean_absolute_error(y_all, y_all_pred)), 1),
          "rmse_w": round(float(np.sqrt(mean_squared_error(y_all, y_all_pred))), 1)}
rows.append(pooled)

summary = pd.DataFrame(rows)
csv_path = os.path.join(INPUT_FOLDER, "test_flight_metrics.csv")
summary.to_csv(csv_path, index=False)

print("\n" + "=" * 62)
print(summary.to_string(index=False))
print("=" * 62)
if len(test_flights) > 1:
    per = summary[summary.flight_id != "ALL (pooled)"]["r2"]
    print(f"Per-flight R² : mean = {per.mean():.4f}  std = {per.std():.4f}  "
          f"min = {per.min():.4f}  max = {per.max():.4f}")
print(f"\nMetrics saved -> {csv_path}")
