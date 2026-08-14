"""
Step 13 – Power over time, with every model's prediction and its predictors named.

Combines the two views that were previously separate:

  * the ROW-LEVEL model (steps 06-08) predicts power every 50 ms from 21 features
    -> drawn as a moving line
  * the FLIGHT-LEVEL models (step 12) predict one average per flight from a few
    named predictors -> drawn as horizontal lines, because that is literally
    what they output

Every prediction is leave-one-flight-out: the flight being drawn never
contributed to the model that predicts it.

The legend names the predictors behind each line, which is the point of the
figure — it shows that a horizontal line from `mass` alone sits closer to the
truth than a 21-feature model wiggling at the wrong level.

IN-FLIGHT ROWS ONLY, consistent with steps 05b and 12.

Output : ML/flight_level_models/plot_prediction_time_<FLIGHT>.png   (one per flight)
         ML/flight_level_models/prediction_time_summary.csv

Run: python 13_prediction_time_by_model.py            # the current test flights
     python 13_prediction_time_by_model.py F05 F12    # or any flights you name
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FLIGHTS_DIR = os.path.join(SCRIPT_DIR, "flights")
OUT_DIR     = os.path.join(SCRIPT_DIR, "ML", "flight_level_models")
ROW_FEATS   = os.path.join(SCRIPT_DIR, "ML", "ml_data", "feature_cols.txt")

MOTOR_COLS = ["motor_1_front_right", "motor_2_rear_left",
              "motor_3_front_left",  "motor_4_rear_right"]
ALT = "uav1_mavros_altitude__local"

# Flight-level models: (label, predictor columns, colour)
FLIGHT_MODELS = [
    ("mass",                     ["mass"],                                      "#16a34a"),
    ("mass + motor",             ["mass", "motor"],                             "#7c3aed"),
    ("mass + position",          ["mass", "pos_front", "pos_diag", "pos_rear"], "#b45309"),
    ("motor only",               ["motor"],                                     "#0891b2"),
]


def load_flights():
    raw, table = {}, []
    for f in sorted(glob.glob(os.path.join(FLIGHTS_DIR, "F*", "flight_resampled.csv"))):
        d = pd.read_csv(f)
        thr = 0.5 * (np.percentile(d["power"], 5) + np.percentile(d["power"], 95))
        g = d[d["power"] > thr].sort_values("timestamp").reset_index(drop=True)
        fid = d["flight_id"].iloc[0]
        raw[fid] = g
        pos = d["position_payload"].iloc[0]
        table.append({"flight": fid, "power": g["power"].mean(),
                      "mass": d["payload_mass"].iloc[0],
                      "motor": g[MOTOR_COLS].mean(axis=1).mean(),
                      "position": pos, "trajectory": d["trajectory"].iloc[0],
                      "pos_front": int(pos == "front"), "pos_diag": int(pos == "diagonal"),
                      "pos_rear": int(pos == "rear")})
    return raw, pd.DataFrame(table).set_index("flight")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw, tab = load_flights()
    row_feats = open(ROW_FEATS).read().strip().split("\n")

    targets = sys.argv[1:] or ["F08", "F09", "F13"]
    targets = [t for t in targets if t in raw]
    if not targets:
        sys.exit("None of the requested flights exist under flights/")

    print(f"\n[Step 13] Prediction over time — flights {targets}")
    rows = []

    for fid in targets:
        te = raw[fid]
        t  = (te["timestamp"] - te["timestamp"].iloc[0]).values
        y  = te["power"].values

        # ---- row-level model: trained on every OTHER flight, all 21 features ----
        tr = pd.concat([g for k, g in raw.items() if k != fid], ignore_index=True)
        mu, sd = tr[row_feats].mean(), tr[row_feats].std().replace(0, 1)
        rm = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                          subsample=0.9, n_jobs=-1, random_state=42)
        rm.fit((tr[row_feats] - mu) / sd, tr["power"])
        y_row = rm.predict((te[row_feats] - mu) / sd)

        # ---- flight-level models: one number, drawn as a horizontal line ----
        flat = {}
        for label, cols, _ in FLIGHT_MODELS:
            trf = tab.drop(index=fid)
            lm  = LinearRegression().fit(trf[cols], trf["power"])
            flat[label] = float(lm.predict(tab.loc[[fid], cols])[0])

        # ---------------- figure ----------------
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(t, y, color="#334155", lw=1.3, label="ACTUAL power", zorder=3)
        ax.axhline(y.mean(), color="#334155", ls=":", lw=1.6,
                   label=f"actual flight average = {y.mean():.0f} W", zorder=2)
        ax.plot(t, y_row, color="#ef4444", lw=1.0, alpha=0.85, zorder=2,
                label=f"row-level XGBoost — 21 features  (MAE {mean_absolute_error(y, y_row):.0f} W)")

        for (label, cols, color) in FLIGHT_MODELS:
            v = flat[label]
            ax.axhline(v, color=color, ls="--", lw=2.0, zorder=4,
                       label=f"flight-level: {label}  →  {v:.0f} W  (err {v - y.mean():+.0f} W)")

        cfg = tab.loc[fid]
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Power (W)", fontsize=11)
        ax.set_xlim(t[0], t[-1])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower left", framealpha=0.94, ncol=2)
        ax.set_title(
            f"{fid} — {cfg['mass']:.2f} kg, {cfg['position']}, {cfg['trajectory']}   "
            f"(unseen flight, in-flight rows only)\n"
            "solid = actual · red = 21-feature row model · dashed = flight-level models, "
            "labelled by predictors used",
            fontsize=12, fontweight="bold")
        plt.tight_layout()
        p = os.path.join(OUT_DIR, f"plot_prediction_time_{fid}.png")
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()

        rec = {"flight": fid, "actual_mean_w": round(float(y.mean()), 1),
               "row_level_mae_w": round(float(mean_absolute_error(y, y_row)), 1),
               "row_level_mean_w": round(float(y_row.mean()), 1)}
        for label, _, _ in FLIGHT_MODELS:
            rec[f"{label} (W)"] = round(flat[label], 1)
        rows.append(rec)

        print(f"  {fid}: actual {y.mean():6.1f} W | row-level mean {y_row.mean():6.1f} W "
              f"(MAE {rec['row_level_mae_w']:5.1f}) | " +
              " | ".join(f"{l} {flat[l]:.0f}" for l, _, _ in FLIGHT_MODELS))
        print(f"        saved -> {p}")

    s = pd.DataFrame(rows)
    csv = os.path.join(OUT_DIR, "prediction_time_summary.csv")
    s.to_csv(csv, index=False)
    print("\n" + s.to_string(index=False))
    print(f"\n  Saved → {csv}\n")


if __name__ == "__main__":
    main()
