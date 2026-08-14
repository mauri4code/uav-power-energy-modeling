"""
EXPERIMENT — does absolute-error loss recover the cruise accuracy lost to transitions?

STANDALONE. Reads only flights/F*/flight_resampled.csv, writes only into this folder.

THE PROBLEM
-----------
Training on the MARGIN dataset (flight + 5 s either side) makes the model WORSE
during cruise than training on cruise alone: 36.1 -> 44.4 W with payload mass.

The reason is the loss function. XGBoost minimises SQUARED error by default, so a
200 W miss during a landing contributes 40,000 while a 40 W miss during cruise
contributes 1,600 — one bad transition sample weighs as much as 25 cruise samples.
There are only 120 transition rows against 1,523 cruise rows, but their errors are
so much larger that they dominate what the model optimises.

THE TEST
--------
Absolute-error loss weighs every sample by its error, not its error squared. A
200 W miss counts as 200 and a 40 W miss as 40 — a factor of 5, not 25. If the
diagnosis is right, this should recover much of the lost cruise accuracy while
keeping the transitions in the training data.

Compared on three datasets x two loss functions, leave-one-flight-out, 1 Hz,
payload mass INCLUDED (matching the best configuration found so far).

Output : loss_function_results.csv
         plot_loss_function.png

Run: python exp_loss_function.py
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
MARGIN_S = 5

MOTORS = ["motor_1_front_right", "motor_2_rear_left",
          "motor_3_front_left",  "motor_4_rear_right"]
FEATS = MOTORS + ["payload_mass"]          # the best set found: motors + mass


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
    sys.exit("flights/ not found")


def load_1hz(fdir):
    out = []
    for f in sorted(glob.glob(os.path.join(fdir, "F*", "flight_resampled.csv"))):
        d = pd.read_csv(f).sort_values("timestamp").copy()
        thr = 0.5 * (np.percentile(d["power"], 5) + np.percentile(d["power"], 95))
        d["t"] = d["timestamp"] - d["timestamp"].iloc[0]
        d["sec"] = np.floor(d["t"]).astype(int)
        d["payload_mass"] = d["payload_mass"]
        a = d.groupby("sec", as_index=False)[MOTORS + ["payload_mass", "power", "t"]].mean()
        a["flight_id"] = d["flight_id"].iloc[0]
        hi = (a["power"] > thr).values
        i0, i1 = hi.argmax(), len(hi) - 1 - hi[::-1].argmax()
        phase = np.where(hi, "cruise", "ground")
        edge = np.zeros(len(a), bool)
        edge[max(i0 - MARGIN_S, 0):i0] = True
        edge[i1 + 1:i1 + 1 + MARGIN_S] = True
        a["phase"] = np.where(edge & ~hi, "transition", phase)
        out.append(a)
    return pd.concat(out, ignore_index=True)


def evaluate(data, objective):
    """Leave-one-flight-out with the given loss; predictions aligned to data order."""
    pred = np.full(len(data), np.nan)
    for h in sorted(data["flight_id"].unique()):
        te_m = (data.flight_id == h).values
        tr, te = data[~te_m], data[te_m]
        mu, sd = tr[FEATS].mean(), tr[FEATS].std().replace(0, 1)
        m = XGBRegressor(objective=objective, n_estimators=150, max_depth=5,
                         learning_rate=0.1, subsample=0.9, n_jobs=-1, random_state=42)
        m.fit((tr[FEATS] - mu) / sd, tr["power"])
        pred[te_m] = m.predict((te[FEATS] - mu) / sd)
    return pred


def main():
    full = load_1hz(find_flights_dir())
    margin = full[full.phase != "ground"].reset_index(drop=True)
    cruise = full[full.phase == "cruise"].reset_index(drop=True)

    print(f"\n[LOSS FUNCTION TEST]  features: motors + payload_mass")
    print(f"  MARGIN {len(margin)} rows  |  CRUISE-ONLY {len(cruise)} rows")
    print(f"  transition rows in MARGIN: {(margin.phase == 'transition').sum()}\n")

    rows = []
    for ds_name, ds in [("trained on MARGIN", margin),
                        ("trained on CRUISE only", cruise)]:
        for obj, nice in [("reg:squarederror", "squared error (default)"),
                          ("reg:absoluteerror", "absolute error")]:
            p = evaluate(ds, obj)
            y = ds["power"].values
            ck = (ds.phase == "cruise").values
            rec = {"trained_on": ds_name, "loss": nice,
                   "overall_mae_w": round(float(mean_absolute_error(y, p)), 1),
                   "cruise_mae_w": round(float(mean_absolute_error(y[ck], p[ck])), 1),
                   "cruise_r2": round(float(r2_score(y[ck], p[ck])), 3)}
            if (~ck).any():
                rec["transition_mae_w"] = round(
                    float(mean_absolute_error(y[~ck], p[~ck])), 1)
            rows.append(rec)
            print(f"  {ds_name:24s} {nice:24s} "
                  f"cruise {rec['cruise_mae_w']:5.1f} W  (R² {rec['cruise_r2']:+.3f})"
                  + (f"   transition {rec['transition_mae_w']:6.1f} W"
                     if "transition_mae_w" in rec else ""), flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "loss_function_results.csv"), index=False)

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    lab = ["squared error\n(default)", "absolute error"]
    x = np.arange(2); w = 0.36
    for j, (ds, color) in enumerate([("trained on MARGIN", "#f59e0b"),
                                     ("trained on CRUISE only", "#2563eb")]):
        sub = R[R.trained_on == ds]
        ax[0].bar(x + (j - 0.5) * w, sub["cruise_mae_w"], w, color=color, label=ds)
    ax[0].set_xticks(x); ax[0].set_xticklabels(lab, fontsize=9)
    ax[0].set_ylabel("cruise MAE (W)", fontsize=10)
    ax[0].set_title("Accuracy during cruise", fontsize=11, fontweight="bold")
    ax[0].grid(axis="y", alpha=0.3); ax[0].legend(fontsize=9)
    for j, ds in enumerate(["trained on MARGIN", "trained on CRUISE only"]):
        for i, v in enumerate(R[R.trained_on == ds]["cruise_mae_w"]):
            ax[0].text(i + (j - 0.5) * w, v, f" {v:.1f}", ha="center",
                       va="bottom", fontsize=9)

    sub = R[R.trained_on == "trained on MARGIN"]
    ax[1].bar(x, sub["transition_mae_w"], 0.5, color="#dc2626")
    ax[1].set_xticks(x); ax[1].set_xticklabels(lab, fontsize=9)
    ax[1].set_ylabel("transition MAE (W)", fontsize=10)
    ax[1].set_title("Accuracy during takeoff / landing\n(MARGIN model only)",
                    fontsize=11, fontweight="bold")
    ax[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(sub["transition_mae_w"]):
        ax[1].text(i, v, f" {v:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Does absolute-error loss stop the transitions dominating?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, "plot_loss_function.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  Saved → {p}")
    print(f"  Saved → {os.path.join(OUT, 'loss_function_results.csv')}\n")


if __name__ == "__main__":
    main()
