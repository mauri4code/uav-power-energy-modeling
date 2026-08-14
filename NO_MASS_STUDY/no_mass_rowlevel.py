"""
NO-MASS, PER-INSTANT — can power be tracked moment to moment without the payload?

STANDALONE. Reads only flights/F*/flight_resampled.csv, writes only into this folder.

WHY THIS EXISTS
---------------
no_mass_study.py builds FLIGHT-LEVEL models: one row per flight, features are that
flight's averages, output is one number per flight. Those necessarily appear as
horizontal lines on a time plot — one number cannot vary.

This file builds PER-INSTANT models instead: one row per moment, features are the
values at that moment, output is a value at that moment. These produce a moving
line and can, in principle, follow the power trace.

Payload mass is still withheld, as is payload position (its "none" category occurs
only on the zero-payload flights and would leak the load).

SAMPLING
--------
Everything runs at 1 Hz. Power is genuinely measured once per second; the 20 Hz
grid in flight_resampled.csv is a forward-filled copy, so each real measurement is
repeated ~20 times. Aggregating features into one-second windows pairs them with
the real measurement and, for large feature sets, roughly halves the error.

TWO COEFFICIENTS OF DETERMINATION
---------------------------------
  within-flight  does the model beat a flat line at THIS flight's own mean?
                 the honest test of whether it tracks variation during a flight
  pooled         computed across all flights at once, so it also gets credit for
                 placing different flights at different levels

They differ a lot, and quoting only the second one overstates what the model does.

Output : rowlevel_results.csv
         plot_no_mass_rowlevel_time.png

Run: python no_mass_rowlevel.py            # F08 F09 F13
     python no_mass_rowlevel.py F01 F02    # or any flights you name
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
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE

MOTORS = ["motor_1_front_right", "motor_2_rear_left",
          "motor_3_front_left",  "motor_4_rear_right"]

GROUPS = {                                   # no mass, no payload position
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

SETS = [
    ("motors",                    ["motors"],                          "#2563eb"),
    ("motors + imbalance",        ["motors", "imbalance"],             "#16a34a"),
    ("motors + orientation",      ["motors", "orientation"],           "#7c3aed"),
    ("ALL features (no mass)",    list(GROUPS),                        "#f59e0b"),
]

ALL = sorted({c for g in GROUPS.values() for c in g})


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
        d = pd.read_csv(f)
        thr = 0.5 * (np.percentile(d["power"], 5) + np.percentile(d["power"], 95))
        g = d[d["power"] > thr].sort_values("timestamp").copy()
        g["t"] = g["timestamp"] - g["timestamp"].iloc[0]
        g["sec"] = np.floor(g["t"]).astype(int)
        a = g.groupby("sec", as_index=False)[ALL + ["power", "t"]].mean()
        a["flight_id"] = d["flight_id"].iloc[0]
        a["true_mass"] = d["payload_mass"].iloc[0]      # reporting only, never a feature
        out.append(a)
    return pd.concat(out, ignore_index=True)


def cols_for(groups):
    return [c for g in groups for c in GROUPS[g]]


def evaluate(data, cols):
    """Leave-one-flight-out per-instant predictions, plus per-flight diagnostics."""
    flights = sorted(data["flight_id"].unique())
    preds, per = {}, []
    for h in flights:
        tr, te = data[data.flight_id != h], data[data.flight_id == h].sort_values("t")
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
        m = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.1,
                         subsample=0.9, n_jobs=-1, random_state=42)
        m.fit((tr[cols] - mu) / sd, tr["power"])
        p = m.predict((te[cols] - mu) / sd)
        y = te["power"].values
        preds[h] = p
        per.append({"flight": h,
                    "mae_w": round(float(mean_absolute_error(y, p)), 1),
                    "r2_within": round(float(r2_score(y, p)), 3),
                    "timing": round(float(np.corrcoef(y, p)[0, 1]), 3)
                              if p.std() > 1e-9 else np.nan})
    Y = np.concatenate([data[data.flight_id == h].sort_values("t")["power"].values
                        for h in flights])
    P = np.concatenate([preds[h] for h in flights])
    return preds, pd.DataFrame(per), r2_score(Y, P), mean_absolute_error(Y, P)


def main():
    data = load_1hz(find_flights_dir())
    flights = sorted(data["flight_id"].unique())
    show = [f for f in (sys.argv[1:] or ["F08", "F09", "F13"]) if f in flights]
    if not show:
        sys.exit(f"None of those flights exist. Available: {flights}")

    print(f"\n[NO-MASS, PER-INSTANT]  {len(data)} one-second rows, "
          f"{len(flights)} flights")
    print("  payload mass and payload position are withheld\n")

    results, allpreds, rows = [], {}, []
    for label, groups, _ in SETS:
        cols = cols_for(groups)
        preds, per, r2p, maep = evaluate(data, cols)
        allpreds[label] = preds
        results.append({"feature_set": label, "n_features": len(cols),
                        "pooled_r2": round(float(r2p), 3),
                        "pooled_mae_w": round(float(maep), 1),
                        "mean_r2_within": round(float(per.r2_within.mean()), 3),
                        "mean_timing": round(float(per.timing.mean()), 3)})
        print(f"  {label:24s} {len(cols):3d} feats | pooled R² {r2p:+.3f}  "
              f"MAE {maep:5.1f} W | within-flight R² {per.r2_within.mean():+.3f}  "
              f"timing {per.timing.mean():+.3f}")
        for _, r in per.iterrows():
            rows.append({"feature_set": label, **r.to_dict()})

    pd.DataFrame(results).to_csv(os.path.join(OUT, "rowlevel_results.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "rowlevel_per_flight.csv"), index=False)

    # flight-level reference line: motors averaged per flight (the no-mass winner)
    tab = data.groupby("flight_id", as_index=False)[cols_for(["motors"]) + ["power"]].mean()
    fl_pred = {}
    for i in range(len(tab)):
        tr = tab.drop(tab.index[i])
        lm = LinearRegression().fit(tr[cols_for(["motors"])], tr["power"])
        fl_pred[tab.iloc[i]["flight_id"]] = float(
            lm.predict(tab.iloc[[i]][cols_for(["motors"])])[0])

    fig, axes = plt.subplots(len(show), 1, figsize=(14, 4.4 * len(show)))
    if len(show) == 1:
        axes = [axes]
    for ax, fid in zip(axes, show):
        te = data[data.flight_id == fid].sort_values("t")
        t, y = te["t"].values, te["power"].values
        ax.plot(t, y, color="#334155", lw=1.6, label="ACTUAL power (1 Hz)", zorder=5)
        ax.axhline(y.mean(), color="#334155", ls=":", lw=1.4, zorder=2,
                   label=f"actual average = {y.mean():.0f} W")
        for (label, _, color) in SETS:
            p = allpreds[label][fid]
            ax.plot(t, p, color=color, lw=1.3, alpha=0.9, zorder=3,
                    label=f"{label}  (MAE {mean_absolute_error(y, p):.0f} W, "
                          f"R² {r2_score(y, p):+.2f}, timing "
                          f"{np.corrcoef(y, p)[0, 1]:+.2f})")
        ax.axhline(fl_pred[fid], color="#dc2626", ls="--", lw=2.0, zorder=4,
                   label=f"FLIGHT-LEVEL motors → {fl_pred[fid]:.0f} W "
                         f"({fl_pred[fid] - y.mean():+.0f} W)")
        ax.set_title(f"{fid} — held out  (true payload "
                     f"{te['true_mass'].iloc[0]:.2f} kg, withheld)",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Power (W)", fontsize=10)
        ax.set_xlim(t[0], t[-1]); ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower left", ncol=2, framealpha=0.94)
    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle("Per-instant models without payload mass\n"
                 "moving lines are trained on one row per moment; "
                 "red dashed is the flight-level model for comparison",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, "plot_no_mass_rowlevel_time.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()

    print(f"\n  Saved → {p}")
    print(f"  Saved → {os.path.join(OUT, 'rowlevel_results.csv')}")
    print(f"  Saved → {os.path.join(OUT, 'rowlevel_per_flight.csv')}\n")


if __name__ == "__main__":
    main()
