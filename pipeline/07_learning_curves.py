"""
Step 7 – Learning curves: how R², MAE and RMSE evolve as XGBoost adds more trees.

Uses the best hyperparameters found in step 6.
Trains on all 4 training flights, evaluates on:
  - Train set  (same 4 flights)
  - Test set   (F05 — unseen flight)

at every tree step → shows convergence and overfitting gap.

Input  : ML/ml_data/train.csv, ML/ml_data/test.csv,
         ML/ml_data/feature_cols.txt, ML/ml_data/xgboost_best_params.json
Output : ML/ml_data/plot_learning_curves.png

Run: python 07_learning_curves.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
# Data folder. Override with the ML_DATA_DIR env var to run a variant dataset,
# e.g.  ML_DATA_DIR=ML/ml_data_inflight python 07_learning_curves.py
INPUT_FOLDER = os.path.join(SCRIPT_DIR, os.environ.get("ML_DATA_DIR",
                                                       os.path.join("ML", "ml_data")))
RANDOM_SEED  = 42
N_ESTIMATORS = 500    # max trees to evaluate


# =====================================================
# LOAD
# =====================================================
train        = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
test         = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
feature_cols = open(os.path.join(INPUT_FOLDER, "feature_cols.txt")).read().strip().split("\n")
best_params  = json.load(open(os.path.join(INPUT_FOLDER, "xgboost_best_params.json")))

X_train = train[feature_cols].values
y_train = train["power"].values
X_test  = test[feature_cols].values
y_test  = test["power"].values

print(f"Train : {len(train)} rows — {sorted(train['flight_id'].unique().tolist())}")
print(f"Test  : {len(test)} rows  — {sorted(test['flight_id'].unique().tolist())}")


# =====================================================
# TRAIN STEP BY STEP — record metrics at each tree
# =====================================================
# Override n_estimators to the max we want to plot
params = {**best_params, "n_estimators": N_ESTIMATORS}

model = XGBRegressor(
    objective="reg:squarederror",
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbosity=0,
    eval_metric=["rmse", "mae"],   # set in constructor for XGBoost 3.x
    **params,
)

# eval_set lets XGBoost record RMSE on train and test at every tree step
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False,
)

evals = model.evals_result()
train_rmse = np.array(evals["validation_0"]["rmse"])
test_rmse  = np.array(evals["validation_1"]["rmse"])
train_mae  = np.array(evals["validation_0"]["mae"])
test_mae   = np.array(evals["validation_1"]["mae"])

# R² is not a built-in XGBoost metric — compute it at every 5th tree using staged predictions
# We use the booster's staged prediction (predict with ntree_limit)
import xgboost as xgb

print("Computing R² at each tree step (this may take a moment)...")
steps   = list(range(1, N_ESTIMATORS + 1, 5))   # every 5 trees
r2_train, r2_test = [], []

dmat_train = xgb.DMatrix(X_train)
dmat_test  = xgb.DMatrix(X_test)
booster    = model.get_booster()

for n in steps:
    ytr_pred = booster.predict(dmat_train, iteration_range=(0, n))
    yte_pred = booster.predict(dmat_test,  iteration_range=(0, n))
    r2_train.append(r2_score(y_train, ytr_pred))
    r2_test.append(r2_score(y_test,  yte_pred))

steps_all = list(range(1, N_ESTIMATORS + 1))   # for RMSE/MAE (recorded every tree)


# =====================================================
# PLOT — 3 subplots: R², MAE, RMSE
# =====================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
fig.suptitle("XGBoost Learning Curves — Metrics vs Number of Trees", fontsize=13, fontweight="bold")

# ---- R² ----
ax = axes[0]
ax.plot(steps, r2_train, color="red",  linewidth=1.2, label="Train (F01+F03+F04+F06)")
ax.plot(steps, r2_test,  color="blue", linewidth=1.2, label="Test  (F05 — unseen)")
ax.axhline(r2_test[-1],  color="blue", linestyle=":", linewidth=1.0,
           label=f"Final test R² = {r2_test[-1]:.4f}")
ax.axhline(r2_train[-1], color="red",  linestyle=":", linewidth=1.0,
           label=f"Final train R² = {r2_train[-1]:.4f}")
ax.set_ylabel("R²")
ax.set_ylim([0, 1.05])
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_title("R²  (higher = better, 1 = perfect)", fontsize=10)

# ---- MAE ----
ax = axes[1]
ax.plot(steps_all, train_mae, color="red",  linewidth=1.0, label="Train MAE")
ax.plot(steps_all, test_mae,  color="blue", linewidth=1.0, label="Test MAE")
ax.axhline(test_mae[-1],  color="blue", linestyle=":", linewidth=1.0,
           label=f"Final test MAE = {test_mae[-1]:.1f} W")
ax.set_ylabel("MAE (W)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_title("MAE — Mean Absolute Error  (lower = better)", fontsize=10)

# ---- RMSE ----
ax = axes[2]
ax.plot(steps_all, train_rmse, color="red",  linewidth=1.0, label="Train RMSE")
ax.plot(steps_all, test_rmse,  color="blue", linewidth=1.0, label="Test RMSE")
ax.axhline(test_rmse[-1], color="blue", linestyle=":", linewidth=1.0,
           label=f"Final test RMSE = {test_rmse[-1]:.1f} W")
ax.set_ylabel("RMSE (W)")
ax.set_xlabel("Number of Trees")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_title("RMSE — Root Mean Squared Error  (lower = better)", fontsize=10)

plt.tight_layout()
out_path = os.path.join(INPUT_FOLDER, "plot_learning_curves.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"\nPlot saved → {out_path}")
print(f"\nFinal metrics at {N_ESTIMATORS} trees:")
print(f"  Train R²   = {r2_train[-1]:.4f}  |  Test R²   = {r2_test[-1]:.4f}")
print(f"  Train MAE  = {train_mae[-1]:.1f} W  |  Test MAE  = {test_mae[-1]:.1f} W")
print(f"  Train RMSE = {train_rmse[-1]:.1f} W  |  Test RMSE = {test_rmse[-1]:.1f} W")
