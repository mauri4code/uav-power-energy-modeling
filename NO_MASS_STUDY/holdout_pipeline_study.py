"""
HOLD-OUT PIPELINE: tune (CV) -> retrain -> test on unseen flights.

STANDALONE. Reuses the mission-window loader from datasize_study.py.
1 Hz, +/-12 s window, feature set motors + mass, XGBoost (absolute-error loss).

WORKFLOW
--------
1. Split the 14 flights into TRAIN and a held-out TEST set (TEST_SIZE flights that
   are never touched during tuning).
2. TUNE hyperparameters on TRAIN only, by RandomizedSearchCV with GroupKFold on
   the flight id (= leave-one-flight-out over the training flights). 30 candidates.
3. RETRAIN one final model on all TRAIN flights with the best params
   (features z-scored on TRAIN only).
4. TEST once on the unseen TEST flights; report R2 and MAE (overall + per flight).

Output (into HOLDOUT/):
   holdout_summary.json      split, best params, CV score, test metrics
   holdout_test_time.png     measured vs predicted power on the test flights

Run:  python holdout_pipeline_study.py
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

import datasize_study as ds          # load(), COLS, find_flights_dir()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "HOLDOUT")
os.makedirs(OUT, exist_ok=True)

OBJ = "reg:absoluteerror"
TEST_SIZE = 2                        # flights held out for the final test (12 train / 2 test)
SEED = 42
N_ITER = 30
COLS = ds.COLS                       # motors + mass

PARAM_DIST = {
    "n_estimators":     [100, 200, 300, 500],
    "max_depth":        [3, 4, 5, 6, 7],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "reg_alpha":        [0, 0.01, 0.1, 0.5],
    "reg_lambda":       [0.5, 1.0, 2.0, 5.0],
}


def main():
    data = ds.load(ds.find_flights_dir())
    flights = sorted(data.flight_id.unique())
    N = len(flights)

    # 1. split into train / held-out test
    rng = np.random.default_rng(SEED)
    perm = list(rng.permutation(flights))
    test_flights = sorted(perm[:TEST_SIZE])
    train_flights = sorted(perm[TEST_SIZE:])
    tr = data[data.flight_id.isin(train_flights)]
    te = data[data.flight_id.isin(test_flights)]
    print(f"[HOLD-OUT]  {N} flights  |  features {COLS}")
    print(f"  TRAIN ({len(train_flights)}): {train_flights}")
    print(f"  TEST  ({len(test_flights)}): {test_flights}\n")

    # 2. tune on TRAIN only (GroupKFold by flight = LOFO over training flights)
    Xtr, ytr = tr[COLS].values, tr["power"].values
    groups = tr["flight_id"].values
    n_tr_flights = tr["flight_id"].nunique()
    base = XGBRegressor(objective=OBJ, random_state=42, n_jobs=-1, verbosity=0)
    search = RandomizedSearchCV(
        base, PARAM_DIST, n_iter=N_ITER, scoring="neg_mean_absolute_error",
        cv=GroupKFold(n_splits=n_tr_flights), random_state=42, n_jobs=-1)
    print(f"  tuning on TRAIN: {N_ITER} candidates, GroupKFold "
          f"({n_tr_flights} folds)...", flush=True)
    search.fit(Xtr, ytr, groups=groups)
    best = search.best_params_
    cv_mae = float(-search.best_score_)
    print(f"  best CV MAE (train) = {cv_mae:.2f} W")
    print(f"  best params = {best}\n", flush=True)

    # 3. retrain final model on ALL train flights (z-score on train only)
    mu, sd = tr[COLS].mean(), tr[COLS].std().replace(0, 1)
    model = XGBRegressor(objective=OBJ, random_state=42, n_jobs=-1, **best)
    model.fit((tr[COLS] - mu) / sd, ytr)

    # 4. test once on the unseen flights
    pred = model.predict((te[COLS] - mu) / sd)
    r2 = float(r2_score(te["power"], pred))
    mae = float(mean_absolute_error(te["power"], pred))
    print("  TEST on unseen flights:")
    print(f"    overall  R2 = {r2:.3f}   MAE = {mae:.2f} W")
    per_flight = {}
    for f in test_flights:
        m = (te.flight_id == f).values
        fm = float(mean_absolute_error(te["power"].values[m], pred[m]))
        fr = float(r2_score(te["power"].values[m], pred[m]))
        per_flight[f] = {"r2": round(fr, 3), "mae_w": round(fm, 2)}
        print(f"    {f}: R2 = {fr:+.3f}   MAE = {fm:.2f} W")

    with open(os.path.join(OUT, "holdout_summary.json"), "w") as fh:
        json.dump({"rate_hz": 1, "margin_s": 12, "features": COLS,
                   "test_size": TEST_SIZE, "seed": SEED,
                   "train_flights": train_flights, "test_flights": test_flights,
                   "best_params": best, "cv_mae_train_w": round(cv_mae, 2),
                   "test_overall": {"r2": round(r2, 3), "mae_w": round(mae, 2)},
                   "test_per_flight": per_flight}, fh, indent=2)

    # time plot of the test flights
    fig, axes = plt.subplots(len(test_flights), 1,
                             figsize=(12, 3.2 * len(test_flights)), squeeze=False)
    for i, f in enumerate(test_flights):
        m = (te.flight_id == f).values
        d = te[m].sort_values("t")
        p = model.predict((d[COLS] - mu) / sd)
        ax = axes[i][0]
        ax.plot(d["t"], d["power"], color="#334155", lw=1.6, label="measured")
        ax.plot(d["t"], p, color="#dc2626", lw=1.1, label="predicted")
        ax.set_title(f"{f} (unseen)  R^2={r2_score(d['power'], p):+.3f}  "
                     f"MAE={mean_absolute_error(d['power'], p):.0f} W",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel("Power (W)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[-1][0].set_xlabel("Time (s)")
    fig.suptitle("Hold-out pipeline: tune (CV) -> retrain -> test on unseen flights",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "holdout_test_time.png"), dpi=150)
    plt.close()

    print("\n  Saved -> HOLDOUT/holdout_summary.json, holdout_test_time.png")


if __name__ == "__main__":
    main()
