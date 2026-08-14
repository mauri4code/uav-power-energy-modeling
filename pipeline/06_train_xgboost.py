"""
Step 6 – Train XGBoost model to predict UAV power consumption.

Pipeline:
    1. Load train/test sets from ml_data/
    2. GroupKFold CV (k = n_flights) — one fold per flight, no row mixing
    3. RandomizedSearchCV to find best hyperparameters
    4. Retrain final model on all train flights with best params
    5. Evaluate ONCE on held-out test set
    6. Save model + plots

Input  : ML/ml_data/train.csv, ML/ml_data/test.csv, ML/ml_data/feature_cols.txt
Output : ML/ml_data/xgboost_model.json         (trained model)
         ML/ml_data/xgboost_cv_results.csv      (CV fold scores)
         ML/ml_data/plot_predicted_vs_actual.png
         ML/ml_data/plot_feature_importance.png
         ML/ml_data/plot_residuals.png

Run: python 06_train_xgboost.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =====================================================
# SETTINGS
# =====================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
# Data folder. Override with the ML_DATA_DIR env var to run a variant dataset,
# e.g.  ML_DATA_DIR=ML/ml_data_inflight python 06_train_xgboost.py
INPUT_FOLDER = os.path.join(SCRIPT_DIR, os.environ.get("ML_DATA_DIR",
                                                       os.path.join("ML", "ml_data")))
N_ITER        = 30      # number of random hyperparameter combinations to try
RANDOM_SEED   = 42


# =====================================================
# LOAD DATA
# =====================================================
def load_data():
    train = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
    test  = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
    feature_cols = open(os.path.join(INPUT_FOLDER, "feature_cols.txt")).read().strip().split("\n")

    print(f"  Train : {len(train)} rows — flights {sorted(train['flight_id'].unique().tolist())}")
    print(f"  Test  : {len(test)} rows  — flights {sorted(test['flight_id'].unique().tolist())}")
    print(f"  Features: {len(feature_cols)}")

    return train, test, feature_cols


# =====================================================
# METRICS HELPER
# =====================================================
def print_metrics(label, y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"  {label}")
    print(f"    R²   = {r2:.4f}   (1 = perfect, 0 = useless)")
    print(f"    MAE  = {mae:.2f} W  (average error in Watts)")
    print(f"    RMSE = {rmse:.2f} W  (penalises large errors more)")
    return r2, mae, rmse


# =====================================================
# PLOTS
# =====================================================
def plot_predicted_vs_actual(y_true, y_pred, label, out_path):
    """Scatter plot of predicted vs actual power — points on diagonal = perfect."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=8, color="steelblue", label="samples")

    # Perfect prediction line
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="perfect prediction")

    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    ax.set_xlabel("Actual Power (W)")
    ax.set_ylabel("Predicted Power (W)")
    ax.set_title(f"Predicted vs Actual — {label}\nR² = {r2:.4f}  |  MAE = {mae:.1f} W")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {out_path}")


def plot_residuals(y_true, y_pred, label, out_path):
    """Residual plot — errors should be random around 0, no pattern."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(y_pred, residuals, alpha=0.3, s=8, color="orange")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted Power (W)")
    ax.set_ylabel("Residual (Actual − Predicted) (W)")
    ax.set_title(f"Residuals — {label}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {out_path}")


def plot_feature_importance(model, feature_cols, out_path):
    """Horizontal bar chart of top 20 most important features."""
    importances = model.feature_importances_
    indices     = np.argsort(importances)[-20:]   # top 20

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(
        [feature_cols[i] for i in indices],
        importances[indices],
        color="steelblue", alpha=0.8
    )
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("XGBoost — Top 20 Feature Importances")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {out_path}")


def plot_cv_scores(cv_results, out_path):
    """Bar chart of R² per fold."""
    folds  = [f"Fold {i+1}\n({r['flight']})" for i, r in enumerate(cv_results)]
    scores = [r["r2"] for r in cv_results]
    colors = ["steelblue"] * len(scores)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(folds, scores, color=colors, alpha=0.8)
    ax.axhline(np.mean(scores), color="red", linestyle="--",
               linewidth=1.5, label=f"mean R² = {np.mean(scores):.4f}")
    ax.set_ylabel("R²")
    ax.set_ylim([0, 1])
    ax.set_title("GroupKFold CV — R² per fold (one flight per fold)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Value labels on bars
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{score:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {out_path}")


# =====================================================
# MAIN
# =====================================================
def main():
    print("\n" + "=" * 55)
    print("  [Step 6] XGBoost Training")
    print("=" * 55 + "\n")

    # ---- Load data ----
    train, test, feature_cols = load_data()

    X_train = train[feature_cols].values
    y_train = train["power"].values
    groups  = train["flight_id"].values      # used by GroupKFold

    X_test  = test[feature_cols].values
    y_test  = test["power"].values

    n_flights = len(np.unique(groups))
    print(f"\n  GroupKFold: k = {n_flights} (one fold per flight)\n")

    # ---- Hyperparameter search space ----
    param_dist = {
        "n_estimators":      [100, 200, 300, 500],
        "max_depth":         [3, 4, 5, 6, 7],
        "learning_rate":     [0.01, 0.05, 0.1, 0.2],
        "subsample":         [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":  [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight":  [1, 3, 5, 7],
        "reg_alpha":         [0, 0.01, 0.1, 0.5],
        "reg_lambda":        [0.5, 1.0, 2.0, 5.0],
    }

    # ---- RandomizedSearchCV with GroupKFold ----
    print(f"  Searching {N_ITER} random hyperparameter combinations...\n")

    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )

    cv = GroupKFold(n_splits=n_flights)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=N_ITER,
        scoring="r2",
        cv=cv,
        random_state=RANDOM_SEED,
        verbose=1,
        n_jobs=-1,
    )

    search.fit(X_train, y_train, groups=groups)

    print(f"\n  Best CV R²  : {search.best_score_:.4f}")
    print(f"  Best params :")
    for k, v in search.best_params_.items():
        print(f"    {k:20s}: {v}")

    # ---- Per-fold CV scores with best params ----
    print(f"\n  Per-fold scores (best params):")
    best_model_cv = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
        **search.best_params_,
    )

    cv_results = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups)):
        Xtr, Xval = X_train[train_idx], X_train[val_idx]
        ytr, yval = y_train[train_idx], y_train[val_idx]
        flight_val = np.unique(groups[val_idx])[0]

        best_model_cv.fit(Xtr, ytr)
        y_val_pred = best_model_cv.predict(Xval)

        r2   = r2_score(yval, y_val_pred)
        mae  = mean_absolute_error(yval, y_val_pred)
        rmse = np.sqrt(mean_squared_error(yval, y_val_pred))

        print(f"    Fold {fold+1} (val={flight_val}): R²={r2:.4f}  MAE={mae:.1f}W  RMSE={rmse:.1f}W")
        cv_results.append({"fold": fold+1, "flight": flight_val, "r2": r2, "mae": mae, "rmse": rmse})

    cv_mean_r2 = np.mean([r["r2"] for r in cv_results])
    cv_std_r2  = np.std( [r["r2"] for r in cv_results])
    print(f"\n  CV mean R² = {cv_mean_r2:.4f} ± {cv_std_r2:.4f}")

    # ---- Retrain final model on ALL train data ----
    print(f"\n  Retraining final model on all {n_flights} train flights...")
    final_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
        **search.best_params_,
    )
    final_model.fit(X_train, y_train)

    # Train set performance (check for overfitting)
    y_train_pred = final_model.predict(X_train)
    print()
    r2_tr, mae_tr, rmse_tr = print_metrics("Train set (all 4 flights):", y_train, y_train_pred)

    # ---- Final evaluation on held-out test set ----
    print()
    y_test_pred = final_model.predict(X_test)
    r2_te, mae_te, rmse_te = print_metrics("Test set  (F05 — unseen flight):", y_test, y_test_pred)

    # Overfitting check
    print()
    if r2_tr - r2_te > 0.1:
        print(f"  ⚠️  Overfitting gap: train R²={r2_tr:.4f} vs test R²={r2_te:.4f}")
    else:
        print(f"  ✅ No major overfitting: train R²={r2_tr:.4f} vs test R²={r2_te:.4f}")

    # ---- Save model ----
    model_path = os.path.join(INPUT_FOLDER, "xgboost_model.json")
    final_model.save_model(model_path)
    print(f"\n  Model saved → {model_path}")

    # Save CV results
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(INPUT_FOLDER, "xgboost_cv_results.csv"), index=False)

    # Save best params
    params_path = os.path.join(INPUT_FOLDER, "xgboost_best_params.json")
    with open(params_path, "w") as f:
        json.dump(search.best_params_, f, indent=2)
    print(f"  Best params saved → {params_path}")

    # ---- Plots ----
    print()
    plot_predicted_vs_actual(
        y_test, y_test_pred, "Test set (F05)",
        os.path.join(INPUT_FOLDER, "plot_predicted_vs_actual.png")
    )
    plot_residuals(
        y_test, y_test_pred, "Test set (F05)",
        os.path.join(INPUT_FOLDER, "plot_residuals.png")
    )
    plot_feature_importance(
        final_model, feature_cols,
        os.path.join(INPUT_FOLDER, "plot_feature_importance.png")
    )
    plot_cv_scores(
        cv_results,
        os.path.join(INPUT_FOLDER, "plot_cv_scores.png")
    )

    # ---- Final summary ----
    print(f"\n{'='*55}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*55}")
    print(f"  CV mean R²     : {cv_mean_r2:.4f} ± {cv_std_r2:.4f}")
    print(f"  Test R²        : {r2_te:.4f}")
    print(f"  Test MAE       : {mae_te:.2f} W")
    print(f"  Test RMSE      : {rmse_te:.2f} W")
    test_flight_ids = sorted(test["flight_id"].unique().tolist())
    print(f"  Test flight(s) : {test_flight_ids}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
