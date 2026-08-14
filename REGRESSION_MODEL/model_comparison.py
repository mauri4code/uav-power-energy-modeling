"""
MODEL COMPARISON — tree (XGBoost) vs Ridge vs Ensemble, head to head.

STANDALONE. Reads only flights/F*/flight_resampled.csv, writes only into
MODEL_COMPARISON/. Pulls together the three model families already studied in
this folder and in NO_MASS_STUDY/, and puts their best models on one chart:
accuracy AND computation time.

WHAT IS COMPARED
----------------
Each model is evaluated at ITS OWN best feature combination, as found by the
respective study (so every model is shown at its best, not forced onto a shared
feature set). All on the same +/-12 s mission window, 1 Hz, leave-one-flight-out.

    model        best combination (with mass)          source study
    ----------   ------------------------------------  -----------------------------
    XGBoost      motors + mass + speed + imbalance      NO_MASS_STUDY/MISSION12/
    Ridge        mass + motors + imu                    RIDGE_MISSION12/
    Ensemble     motors + mass + speed + velocity       Ensemble/  (blend w_xgb=0.90)

COMPUTATION TIME
----------------
Measured as the wall-clock to fit AND predict one full leave-one-flight-out
pass (14 folds) at that best combination — i.e. the cost of building the
deployable model, timed identically for all three in this one process. The
median of REPEATS runs is reported, so a single slow fold does not dominate.

This is the model cost, not the feature-search cost. The searches themselves
differ a lot (Ridge affords an exhaustive 127/255-subset search in ~45 s;
XGBoost is limited to greedy because one evaluation is ~100x slower) — that is
discussed in MODEL_COMPARISON/README-equivalent notes, not plotted here.

OUTPUT (in MODEL_COMPARISON/)
------------------------------
    plot_model_comparison.png     3 panels — MAE, R2, computation time (with mass)
    model_comparison.csv          every model x mass setting, all metrics + time
    model_comparison.json         same, machine-readable

Run:
    python model_comparison.py
    python model_comparison.py --repeats 5     # more timing repeats (default 3)
"""

import os
import sys
import glob
import json
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "MODEL_COMPARISON")
os.makedirs(OUT, exist_ok=True)

MARGIN_S = 12
ALPHAS = np.logspace(-2, 4, 25)
BLEND_GRID = np.linspace(0, 1, 21)
OBJ = "reg:absoluteerror"
REPEATS = int(sys.argv[sys.argv.index("--repeats") + 1]) if "--repeats" in sys.argv else 3

MOTORS = ["motor_1_front_right", "motor_2_rear_left",
          "motor_3_front_left",  "motor_4_rear_right"]
GROUPS = {
    "mass":        ["payload_mass"],
    "motors":      MOTORS,
    "imbalance":   ["front_rear_imbalance", "diagonal_imbalance"],
    "orientation": ["roll_rad", "pitch_rad", "yaw_rad"],
    "imu":         ["uav1_hw_api_imu__angular_velocity_x",
                    "uav1_hw_api_imu__angular_velocity_y",
                    "uav1_hw_api_imu__angular_velocity_z",
                    "uav1_hw_api_imu__linear_acceleration_x",
                    "uav1_hw_api_imu__linear_acceleration_y",
                    "uav1_hw_api_imu__linear_acceleration_z"],
    "speed":       ["speed_3d"],
    "velocity":    ["uav1_estimation_manager_uav_state__velocity_linear_x",
                    "uav1_estimation_manager_uav_state__velocity_linear_y",
                    "uav1_estimation_manager_uav_state__velocity_linear_z"],
    "altitude":    ["uav1_mavros_altitude__local"],
}
ALL_COLS = sorted({c for g in GROUPS.values() for c in g})

# Best feature combination per model per mass setting, taken from each study's
# saved winner (summary_*.json). Kept explicit so the comparison is transparent
# and reproducible even if a study is later re-run.
# Tuned XGBoost hyperparameters per mass setting (from NO_MASS_STUDY/MISSION12/
# summary_1hz_*.json, RandomizedSearchCV result). The ensemble keeps the fixed
# base-learner config of its own study; only the standalone XGBoost is tuned here.
XGB_PARAMS = {
    "withmass": dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                     subsample=0.6, colsample_bytree=0.8, min_child_weight=5,
                     reg_alpha=0.1, reg_lambda=5.0),
    "nomass":   dict(n_estimators=300, max_depth=7, learning_rate=0.05,
                     subsample=0.6, colsample_bytree=0.8, min_child_weight=1,
                     reg_alpha=0.5, reg_lambda=5.0),
}
_XGB_PARAMS = XGB_PARAMS["withmass"]     # set per mass setting in main()

BEST = {
    "XGBoost": {
        "withmass": ["motors", "mass"],
        "nomass":   ["motors", "velocity"],
    },
    "Ridge": {
        "withmass": ["mass", "motors", "imu"],
        "nomass":   ["motors", "altitude"],
    },
    "Ensemble": {
        "withmass": ["motors", "mass", "speed", "velocity"],
        "nomass":   ["motors", "imbalance", "velocity", "speed", "orientation", "imu"],
    },
}
COLORS = {"XGBoost": "#f97316", "Ridge": "#2563eb", "Ensemble": "#16a34a"}


def find_flights_dir():
    d = HERE
    for _ in range(5):
        c = os.path.join(d, "flights")
        if glob.glob(os.path.join(c, "F*", "flight_resampled.csv")):
            return c
        for cand in glob.glob(os.path.join(d, "*", "*", "*", "flights")):
            if glob.glob(os.path.join(cand, "F*", "flight_resampled.csv")):
                return cand
        d = os.path.dirname(d)
    sys.exit("Could not find flights/ containing flight_resampled.csv")


def load(fdir):
    out = []
    for f in sorted(glob.glob(os.path.join(fdir, "F*", "flight_resampled.csv"))):
        d = pd.read_csv(f).sort_values("timestamp").copy()
        thr = 0.5 * (np.percentile(d["power"], 5) + np.percentile(d["power"], 95))
        d["t"] = d["timestamp"] - d["timestamp"].iloc[0]
        d["sec"] = np.floor(d["t"]).astype(int)
        a = d.groupby("sec", as_index=False)[ALL_COLS + ["power", "t"]].mean()
        a["flight_id"] = d["flight_id"].iloc[0]
        hi = (a["power"] > thr).values
        i0, i1 = hi.argmax(), len(hi) - 1 - hi[::-1].argmax()
        idx = np.arange(len(a))
        a["phase"] = np.where(hi, "cruise", "other")
        keep = (idx >= i0 - MARGIN_S) & (idx <= i1 + MARGIN_S)
        out.append(a[keep])
    return pd.concat(out, ignore_index=True)


def cols_for(groups):
    return [c for g in groups for c in GROUPS[g]]


def eval_xgb(data, cols):
    # Features are z-scored on the train fold before fitting, exactly as in
    # NO_MASS_STUDY/mission12_study.py's evaluate(), with the tuned
    # (RandomizedSearchCV) hyperparameters, so this reproduces that study's
    # headline number (35.56 W with mass on motors+mass, 39.80 W without mass).
    pred = np.full(len(data), np.nan)
    for h in sorted(data["flight_id"].unique()):
        te = (data.flight_id == h).values
        tr = data[~te]
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
        m = XGBRegressor(objective=OBJ, n_jobs=-1, random_state=42, **_XGB_PARAMS)
        m.fit((tr[cols] - mu) / sd, tr["power"])
        pred[te] = m.predict((data[te][cols] - mu) / sd)
    return pred


def eval_ridge(data, cols):
    pred = np.full(len(data), np.nan)
    for h in sorted(data["flight_id"].unique()):
        te = (data.flight_id == h).values
        tr = data[~te]
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
        m = RidgeCV(alphas=ALPHAS)
        m.fit(((tr[cols] - mu) / sd).values, tr["power"].values)
        pred[te] = m.predict(((data[te][cols] - mu) / sd).values)
    return pred


def eval_ensemble(data, cols):
    # Matches ensemble_mission12_study.py exactly: XGBoost on RAW features (not
    # z-scored) blended with z-scored Ridge, weight grid-searched on the
    # out-of-fold predictions. So this reproduces that study's 36.48 W / w=0.90.
    p_xgb = np.full(len(data), np.nan)
    p_ridge = eval_ridge(data, cols)
    for h in sorted(data["flight_id"].unique()):
        te = (data.flight_id == h).values
        tr = data[~te]
        m = XGBRegressor(objective=OBJ, n_estimators=150, max_depth=5,
                         learning_rate=0.1, subsample=0.9, n_jobs=-1, random_state=42)
        m.fit(tr[cols], tr["power"])
        p_xgb[te] = m.predict(data[te][cols])
    y = data["power"].values
    maes = [mean_absolute_error(y, w * p_xgb + (1 - w) * p_ridge) for w in BLEND_GRID]
    w = float(BLEND_GRID[int(np.argmin(maes))])
    return w * p_xgb + (1 - w) * p_ridge, w


EVALUATORS = {"XGBoost": eval_xgb, "Ridge": eval_ridge, "Ensemble": eval_ensemble}


def timed_eval(name, data, cols):
    """Median wall-clock of REPEATS full leave-one-flight-out passes, + the last prediction."""
    times, pred, w = [], None, None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        r = EVALUATORS[name](data, cols)
        times.append(time.perf_counter() - t0)
        pred, w = (r[0], r[1]) if isinstance(r, tuple) else (r, None)
    return float(np.median(times)), pred, w


def score(data, p):
    y = data["power"].values
    ck = (data.phase == "cruise").values
    return {"r2": round(float(r2_score(y, p)), 4),
            "mae_w": round(float(mean_absolute_error(y, p)), 2),
            "cruise_mae_w": round(float(mean_absolute_error(y[ck], p[ck])), 2)}


def fig_comparison(rows, mass_key, mass_label):
    """Three panels — MAE, R2, computation time — one bar per model."""
    sub = [r for r in rows if r["mass"] == mass_key]
    names = [r["model"] for r in sub]
    x = np.arange(len(sub))
    colors = [COLORS[n] for n in names]

    panels = [("mae_w", "MAE (W) — lower is better", "{:.2f}", False),
              ("r2", "R2 — higher is better", "{:.3f}", False),
              ("time_s", "compute time (s), log scale — 14-fold fit+predict,\n"
                         f"median of {REPEATS} — lower is better", "{:.2f} s", True)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (key, nice, fmt, is_log) in zip(axes, panels):
        vals = [r[key] for r in sub]
        bars = ax.bar(x, vals, color=colors)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel(nice, fontsize=10 if not is_log else 9.5)
        ax.grid(axis="y", alpha=0.3)
        if is_log:
            ax.set_yscale("log")
            ax.set_ylim(min(vals) / 2.5, max(vals) * 2.2)
        else:
            ax.set_ylim(0, max(vals) * 1.18)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, " " + fmt.format(v),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    feat = "   ".join(f"{r['model']}: {' + '.join(r['groups'])}"
                      f"{'  (w=' + format(r['blend_w_xgb'], '.2f') + ')' if r['blend_w_xgb'] is not None else ''}"
                      for r in sub)
    fig.suptitle(f"Tree (XGBoost) vs Ridge vs Ensemble — best model of each, "
                f"{mass_label}\n+/-{MARGIN_S} s window, 1 Hz, leave-one-flight-out"
                f"\n{feat}", fontsize=11.5, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUT, f"plot_model_comparison_{mass_key}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def main():
    data = load(find_flights_dir())
    print(f"[MODEL COMPARISON]  +/-{MARGIN_S} s window, 1 Hz, "
          f"{len(data)} rows, {data.flight_id.nunique()} flights")
    print(f"  timing: median of {REPEATS} full leave-one-flight-out passes each\n")

    rows = []
    global _XGB_PARAMS
    for mass_key, mass_label in [("withmass", "mass included"),
                                 ("nomass", "mass withheld")]:
        print(f"  --- {mass_label} ---")
        _XGB_PARAMS = XGB_PARAMS[mass_key]
        for model in ["XGBoost", "Ridge", "Ensemble"]:
            groups = BEST[model][mass_key]
            cols = cols_for(groups)
            t, pred, w = timed_eval(model, data, cols)
            s = score(data, pred)
            rows.append({"model": model, "mass": mass_key,
                        "groups": groups, "n_features": len(cols),
                        "blend_w_xgb": (round(w, 2) if w is not None else None),
                        "time_s": round(t, 3), **s})
            wtxt = f"  w_xgb={w:.2f}" if w is not None else ""
            print(f"    {model:9s}  MAE {s['mae_w']:6.2f} W   R2 {s['r2']:+.4f}   "
                  f"time {t:6.2f} s{wtxt}   ({' + '.join(groups)})")
        print()

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
    with open(os.path.join(OUT, "model_comparison.json"), "w") as fh:
        json.dump({"margin_s": MARGIN_S, "rate_hz": 1, "repeats": REPEATS,
                   "timing": "median wall-clock of one 14-fold leave-one-flight-out "
                             "fit+predict pass, seconds",
                   "results": rows}, fh, indent=2)

    p1 = fig_comparison(rows, "withmass", "mass included")
    p2 = fig_comparison(rows, "nomass", "mass withheld")
    print(f"  Saved -> {os.path.basename(p1)}")
    print(f"  Saved -> {os.path.basename(p2)}")
    print(f"  Saved -> model_comparison.csv, model_comparison.json\n")


if __name__ == "__main__":
    main()
