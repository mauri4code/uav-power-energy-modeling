"""
Step 16 – Instantaneous prediction on the SIM_mission flight.

Unlike step 15 (which averaged each segment), here the four best Step-12 models
predict power at EVERY 20 ms row of the mission flight, using that row's own
feature values. This shows how each model tracks the takeoff -> high -> medium
-> low -> landing profile in real time.

The four models are the top-4 by MAE from Step 12:
    1. mass + position     (MAE 12.3 W)
    2. mass                (MAE 12.4 W)
    3. mass + motor        (MAE 13.4 W)
    4. mass + motor + alt  (MAE 15.0 W)

Training is unchanged from Step 12 (flight-level: one in-flight-averaged row per
flight). Evaluation is LEAVE-ONE-FLIGHT-OUT: every mission row is predicted by a
model trained on the 13 flights that EXCLUDE that row's source flight, so no
segment is scored by a model that saw its source flight. Inference only — the
mission flight never enters training.

Input  : flights/F*/flight_resampled.csv          (training, 14 real flights)
         SIM_FLIGHTS/SIM_mission/flight_resampled.csv (+ segment_map.csv)
Output : SIM_FLIGHTS/SIM_mission/instantaneous_pred.png
         SIM_FLIGHTS/SIM_mission/instantaneous_pred.csv

Run: python 16_mission_instantaneous.py
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FLIGHTS_DIR = os.path.join(SCRIPT_DIR, "flights")
MISSION_DIR = os.path.join(SCRIPT_DIR, "SIM_FLIGHTS", "SIM_mission")

MOTOR_COLS = ["motor_1_front_right", "motor_2_rear_left",
              "motor_3_front_left",  "motor_4_rear_right"]
ALT = "uav1_mavros_altitude__local"

AIRBORNE_W = 300.0     # rows below this are "on the ground" (idle ~25 W)

# The four best Step-12 models: (name, feature columns).  All linear.
MODELS = [
    ("mass + position",    ["mass", "pos_front", "pos_diag", "pos_rear"]),
    ("mass",               ["mass"]),
    ("mass + motor",       ["mass", "motor"]),
    ("mass + motor + alt", ["mass", "motor", "alt"]),
]
COLORS = {"mass + position": "#2563eb", "mass": "#059669",
          "mass + motor": "#d97706", "mass + motor + alt": "#dc2626"}


def flight_features(d):
    """Step-12 flight-level row: in-flight AVERAGED features + average power."""
    thr = 0.5 * (np.percentile(d["power"], 5) + np.percentile(d["power"], 95))
    g = d[d["power"] > thr]
    pos = d["position_payload"].iloc[0]
    return {
        "flight": d["flight_id"].iloc[0],
        "mass":   d["payload_mass"].iloc[0],
        "motor":  g[MOTOR_COLS].mean(axis=1).mean(),
        "alt":    g[ALT].mean(),
        "speed":  g["speed_3d"].mean(),
        "pitch":  g["pitch_rad"].mean(),
        "roll":   g["roll_rad"].mean(),
        "pos_front": int(pos == "front"),
        "pos_diag":  int(pos == "diagonal"),
        "pos_rear":  int(pos == "rear"),
        "power":  g["power"].mean(),
    }


def training_table():
    rows = [flight_features(pd.read_csv(f))
            for f in sorted(glob.glob(os.path.join(FLIGHTS_DIR, "F*", "flight_resampled.csv")))]
    return pd.DataFrame(rows).sort_values("flight").reset_index(drop=True)


def row_features(df):
    """INSTANTANEOUS per-row features for the mission flight (same names)."""
    pos = df["position_payload"]
    return pd.DataFrame({
        "mass":  df["payload_mass"],
        "motor": df[MOTOR_COLS].mean(axis=1),
        "alt":   df[ALT],
        "speed": df["speed_3d"],
        "pitch": df["pitch_rad"],
        "roll":  df["roll_rad"],
        "pos_front": (pos == "front").astype(int),
        "pos_diag":  (pos == "diagonal").astype(int),
        "pos_rear":  (pos == "rear").astype(int),
    })


def source_per_row(df, seg_map):
    """Return an array giving each row's source flight (for leave-one-out)."""
    src = np.empty(len(df), dtype=object)
    for r in seg_map.itertuples():
        src[int(r.dst_start_idx):int(r.dst_start_idx) + int(r.n_samples)] = r.source_flight
    return src


def main():
    print("\n[Step 16] Instantaneous prediction on SIM_mission")

    train = training_table()
    print(f"  Training table: {len(train)} flights (flight-level averages)")

    mission = pd.read_csv(os.path.join(MISSION_DIR, "flight_resampled.csv"))
    seg_map = pd.read_csv(os.path.join(MISSION_DIR, "segment_map.csv"))
    t   = mission["timestamp"].values
    y   = mission["power"].values
    Xr  = row_features(mission)
    src = source_per_row(mission, seg_map)
    src_flights = sorted(set(src))
    airborne = y > AIRBORNE_W
    print(f"  Mission: {len(mission)} rows, sources={src_flights}, "
          f"{airborne.sum()} airborne rows")

    # ---- leave-one-flight-out instantaneous predictions ----
    preds, metrics = {}, {}
    for name, cols in MODELS:
        # one linear model per held-out source flight
        models = {fk: LinearRegression().fit(train[train.flight != fk][cols],
                                             train[train.flight != fk]["power"])
                  for fk in src_flights}
        p = np.empty(len(mission))
        for fk in src_flights:
            m = src == fk
            p[m] = models[fk].predict(Xr.loc[m, cols])
        preds[name] = p
        # metrics on airborne rows only (ground idle would swamp them)
        metrics[name] = (mean_absolute_error(y[airborne], p[airborne]),
                         r2_score(y[airborne], p[airborne]))
        print(f"  {name:20s}  airborne MAE {metrics[name][0]:6.1f} W   R² {metrics[name][1]:6.3f}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(t, y, color="#0f172a", lw=1.4, label="ACTUAL power", zorder=5)
    for name, _ in MODELS:
        mae, r2 = metrics[name]
        ax.plot(t, preds[name], color=COLORS[name], lw=1.1, alpha=0.9,
                label=f"{name}   (MAE {mae:.0f} W, R² {r2:.2f})")

    # headroom so phase labels sit above the traces but below the title
    ymax = max(y.max(), max(p.max() for p in preds.values()))
    ax.set_ylim(-20, ymax * 1.18)
    label_y = ymax * 1.03

    for r in seg_map.itertuples():
        ax.axvline(r.t_start_s, color="k", ls=":", lw=0.6, alpha=0.4)
        ph = getattr(r, "phase", r.source_flight)
        ax.text((r.t_start_s + r.t_end_s) / 2, label_y,
                f"{ph}\n({r.source_flight})", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#334155")
    ax.axhline(AIRBORNE_W, color="#94a3b8", ls="--", lw=0.7)
    ax.text(t[-1], AIRBORNE_W, " airborne thr", va="center", fontsize=7, color="#64748b")
    ax.set_xlabel("time [s]"); ax.set_ylabel("power [W]")
    ax.set_title("SIM_mission — instantaneous power prediction (4 best Step-12 models)\n"
                 "leave-one-flight-out; metrics on airborne rows only",
                 fontweight="bold", pad=28)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95, ncol=1)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(MISSION_DIR, "instantaneous_pred.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()

    # ---- csv ----
    out = pd.DataFrame({"timestamp": t, "source_flight": src, "actual_power": y.round(1)})
    for name, _ in MODELS:
        out[f"pred__{name}"] = preds[name].round(1)
    out_csv = os.path.join(MISSION_DIR, "instantaneous_pred.csv")
    out.to_csv(out_csv, index=False)

    print(f"\n  Saved -> {out_png}")
    print(f"          {out_csv}\n")


if __name__ == "__main__":
    main()
