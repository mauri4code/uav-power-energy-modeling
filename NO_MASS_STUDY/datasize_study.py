"""
DATA-SIZE SENSITIVITY (learning curve) on the +/-12 s mission window, 1 Hz.

STANDALONE. Reads only flights/F*/flight_resampled.csv, writes only into DATASIZE/.

THE QUESTION
------------
How many FLIGHTS are needed before more flights stop improving the model?

Why flights, not rows: consecutive 1 Hz rows within a flight are nearly identical,
so a random subset of rows still "sees" every flight and the curve would look
flat for the wrong reason. The honest axis is the number of distinct flights.

WHAT IT DOES
------------
For each training size k = 3, 4, ..., N-1 flights:
  * repeat R times: pick k flights at random for TRAIN, the remaining N-k flights
    are TEST; z-score on train only; fit the tuned XGBoost; predict the test
    flights pooled; record R2 and MAE.
  * store the MEAN and STD over the repeats.
When C(N, k) <= R every distinct subset is used exactly once (no sampling noise);
otherwise R random subsets are drawn.

Model: the tuned XGBoost from mission12_study.py (RandomizedSearchCV result),
feature set motors + mass (the selected set, with mass).

Output (into DATASIZE/):
  datasize_results.csv     size, mean/std of R2 and MAE, n_subsets
  datasize_r2_mean.png     mean R2 vs number of training flights (+/- std band)
  datasize_r2_std.png      std of R2 vs number of training flights
  datasize_mae_mean.png    mean MAE vs number of training flights (+/- std band)

Run:  python datasize_study.py
"""

import os
import sys
import glob
import json
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "DATASIZE")
os.makedirs(OUT, exist_ok=True)

MARGIN_S = 12
OBJ = "reg:absoluteerror"
R_REPEATS = 30          # random subsets per size when there are more than this
K_MIN = 3               # smallest training set (flights)
RNG = np.random.default_rng(42)

# Tuned XGBoost (from NO_MASS_STUDY/MISSION12/summary_1hz_withmass.json)
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.6,
              colsample_bytree=0.8, min_child_weight=5, reg_alpha=0.1,
              reg_lambda=5.0)

# Selected feature set: motors + mass
MOTORS = ["motor_1_front_right", "motor_2_rear_left",
          "motor_3_front_left",  "motor_4_rear_right"]
COLS = MOTORS + ["payload_mass"]


def find_flights_dir():
    d = HERE
    for _ in range(6):
        c = os.path.join(d, "flights")
        if glob.glob(os.path.join(c, "F*", "flight_resampled.csv")):
            return c
        for cand in glob.glob(os.path.join(d, "*", "*", "*", "flights")):
            if glob.glob(os.path.join(cand, "F*", "flight_resampled.csv")):
                return cand
        d = os.path.dirname(d)
    sys.exit("Could not find flights/ containing flight_resampled.csv")


def load(fdir):
    """1 Hz mission window (flight +/- MARGIN_S s), motors + mass + power."""
    out = []
    need = COLS + ["power"]
    for f in sorted(glob.glob(os.path.join(fdir, "F*", "flight_resampled.csv"))):
        d = pd.read_csv(f).sort_values("timestamp").copy()
        thr = 0.5 * (np.percentile(d["power"], 5) + np.percentile(d["power"], 95))
        d["t"] = d["timestamp"] - d["timestamp"].iloc[0]
        d["sec"] = np.floor(d["t"]).astype(int)
        a = d.groupby("sec", as_index=False)[need + ["t"]].mean()
        a["flight_id"] = d["flight_id"].iloc[0]
        hi = (a["power"] > thr).values
        i0, i1 = hi.argmax(), len(hi) - 1 - hi[::-1].argmax()
        idx = np.arange(len(a))
        keep = (idx >= i0 - MARGIN_S) & (idx <= i1 + MARGIN_S)
        out.append(a[keep])
    return pd.concat(out, ignore_index=True)


def fit_predict(data, train_flights, test_flights):
    tr = data[data.flight_id.isin(train_flights)]
    te = data[data.flight_id.isin(test_flights)]
    mu, sd = tr[COLS].mean(), tr[COLS].std().replace(0, 1)
    m = XGBRegressor(objective=OBJ, random_state=42, n_jobs=-1, **PARAMS)
    m.fit((tr[COLS] - mu) / sd, tr["power"])
    p = m.predict((te[COLS] - mu) / sd)
    return r2_score(te["power"], p), mean_absolute_error(te["power"], p)


def subsets_for_size(flights, k):
    """All distinct k-subsets if there are <= R_REPEATS of them, else R random."""
    n_total = len(list(itertools.combinations(range(len(flights)), k)))
    if n_total <= R_REPEATS:
        return [list(c) for c in itertools.combinations(flights, k)]
    subs = set()
    while len(subs) < R_REPEATS:
        subs.add(tuple(sorted(RNG.choice(flights, size=k, replace=False))))
    return [list(s) for s in subs]


def main():
    fdir = find_flights_dir()
    data = load(fdir)
    flights = sorted(data.flight_id.unique())
    N = len(flights)
    print(f"[DATA-SIZE]  1 Hz, +/-{MARGIN_S} s window, features {COLS}")
    print(f"  {len(data)} rows over {N} flights: {flights}\n")

    rows = []
    for k in range(K_MIN, N):                       # up to N-1 (>=1 test flight)
        subs = subsets_for_size(flights, k)
        r2s, maes = [], []
        for train in subs:
            test = [f for f in flights if f not in train]
            r2, mae = fit_predict(data, train, test)
            r2s.append(r2); maes.append(mae)
        rows.append({"n_train_flights": k, "n_subsets": len(subs),
                     "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
                     "mae_mean_w": np.mean(maes), "mae_std_w": np.std(maes)})
        print(f"  k={k:2d} flights | {len(subs):2d} subsets | "
              f"R2 {np.mean(r2s):+.3f} +/- {np.std(r2s):.3f} | "
              f"MAE {np.mean(maes):5.1f} +/- {np.std(maes):4.1f} W", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "datasize_results.csv"), index=False)

    x = df.n_train_flights.values

    # --- mean R2 with +/- std band ---
    plt.figure(figsize=(8, 5))
    plt.plot(x, df.r2_mean, "-o", color="#2563eb", label="mean $R^2$")
    plt.fill_between(x, df.r2_mean - df.r2_std, df.r2_mean + df.r2_std,
                     color="#2563eb", alpha=0.18, label="$\\pm$1 std")
    plt.xlabel("number of training flights"); plt.ylabel("$R^2$ (pooled over held-out flights)")
    plt.title("Data-size sensitivity: mean $R^2$ vs training flights\n"
              "(tuned XGBoost, motors + mass, 1 Hz)", fontweight="bold")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "datasize_r2_mean.png"), dpi=150)
    plt.close()

    # --- std of R2 ---
    plt.figure(figsize=(8, 5))
    plt.plot(x, df.r2_std, "-o", color="#dc2626")
    plt.xlabel("number of training flights"); plt.ylabel("standard deviation of $R^2$")
    plt.title("Data-size sensitivity: stability of $R^2$ vs training flights\n"
              "(lower = more reproducible)", fontweight="bold")
    plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "datasize_r2_std.png"), dpi=150)
    plt.close()

    # --- mean MAE with +/- std band ---
    plt.figure(figsize=(8, 5))
    plt.plot(x, df.mae_mean_w, "-o", color="#16a34a", label="mean MAE")
    plt.fill_between(x, df.mae_mean_w - df.mae_std_w, df.mae_mean_w + df.mae_std_w,
                     color="#16a34a", alpha=0.18, label="$\\pm$1 std")
    plt.xlabel("number of training flights"); plt.ylabel("MAE (W)")
    plt.title("Data-size sensitivity: mean MAE vs training flights\n"
              "(tuned XGBoost, motors + mass, 1 Hz)", fontweight="bold")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "datasize_mae_mean.png"), dpi=150)
    plt.close()

    with open(os.path.join(OUT, "datasize_summary.json"), "w") as fh:
        json.dump({"rate_hz": 1, "margin_s": MARGIN_S, "features": COLS,
                   "hyperparameters": PARAMS, "n_flights": N,
                   "r_repeats_cap": R_REPEATS, "results": rows}, fh, indent=2, default=float)

    print("\n  Saved -> DATASIZE/datasize_results.csv, datasize_summary.json")
    print("  Saved -> DATASIZE/datasize_r2_mean.png, datasize_r2_std.png, datasize_mae_mean.png")


if __name__ == "__main__":
    main()
