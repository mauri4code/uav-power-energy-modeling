"""
Step 17 – No-mass vs with-mass row-level model on the SIM_mission flight.

Reproduces the two winning 20 Hz models from the thesis NO_MASS_STUDY/MISSION12
study and applies them to the synthetic mission flight (step 14), so the two can
be compared on a takeoff -> high -> medium -> low -> landing profile.

Winning 20 Hz feature sets (from summary_20hz_*.json):
  no-mass   : motors + imu                         (10 features)  R² 0.885
  with-mass : motors + mass + velocity + speed      (9 features)  R² 0.914

Model recipe (verbatim from mission12_study.py):
  XGBRegressor(objective="reg:absoluteerror", n_estimators=150, max_depth=5,
               learning_rate=0.1, subsample=0.9, random_state=42)
  per-fold standardization (mean/std from the training flights only)
  trained on the flight ±12 s window (arming / takeoff / landing margin)

Evaluation = leave-one-flight-out: each mission segment is predicted by a model
trained on the 13 real flights that EXCLUDE that segment's source flight. The
mission flight never enters training. No saved model existed in NO_MASS_STUDY,
so the models are rebuilt from the recipe above (not the mission flight).

Output: SIM_FLIGHTS/SIM_mission/nomass_vs_withmass.png
        SIM_FLIGHTS/SIM_mission/nomass_vs_withmass.csv

Run: python 17_mission_nomass_vs_withmass.py
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FLIGHTS_DIR = os.path.join(SCRIPT_DIR, "flights")
MISSION_DIR = os.path.join(SCRIPT_DIR, "SIM_FLIGHTS", "SIM_mission")

MARGIN_S = 12
RATE_STEP = 20          # rows per second at 20 Hz (for the ±12 s window)

MOTORS   = ["motor_1_front_right", "motor_2_rear_left",
            "motor_3_front_left",  "motor_4_rear_right"]
IMU      = ["uav1_hw_api_imu__angular_velocity_x",
            "uav1_hw_api_imu__angular_velocity_y",
            "uav1_hw_api_imu__angular_velocity_z",
            "uav1_hw_api_imu__linear_acceleration_x",
            "uav1_hw_api_imu__linear_acceleration_y",
            "uav1_hw_api_imu__linear_acceleration_z"]
VELOCITY = ["uav1_estimation_manager_uav_state__velocity_linear_x",
            "uav1_estimation_manager_uav_state__velocity_linear_y",
            "uav1_estimation_manager_uav_state__velocity_linear_z"]
SPEED    = ["speed_3d"]
MASS     = ["payload_mass"]

# The two 20 Hz winners from the study.
CONFIGS = {
    "no-mass (motors + imu)":              MOTORS + IMU,
    "with-mass (motors + mass + vel + spd)": MOTORS + MASS + VELOCITY + SPEED,
}
COLORS = {"no-mass (motors + imu)": "#2563eb",
          "with-mass (motors + mass + vel + spd)": "#dc2626"}

ALL_COLS = sorted(set(MOTORS + IMU + VELOCITY + SPEED + MASS))


def make_model():
    return XGBRegressor(objective="reg:absoluteerror", n_estimators=150, max_depth=5,
                        learning_rate=0.1, subsample=0.9, n_jobs=-1, random_state=42)


def load_training_windows():
    """
    Every real flight reduced to its ±12 s mission window at 20 Hz, exactly as
    mission12_study.load(): cruise = power above the mid-percentile threshold,
    plus a 12 s margin either side to include arming / takeoff / landing.
    """
    out = []
    for f in sorted(glob.glob(os.path.join(FLIGHTS_DIR, "F*", "flight_resampled.csv"))):
        d = pd.read_csv(f).sort_values("timestamp").reset_index(drop=True)
        thr = 0.5 * (np.percentile(d["power"], 5) + np.percentile(d["power"], 95))
        a = d[ALL_COLS + ["power"]].copy()
        a["flight_id"] = d["flight_id"].iloc[0]
        hi = (d["power"] > thr).values
        i0, i1 = hi.argmax(), len(hi) - 1 - hi[::-1].argmax()
        idx = np.arange(len(a))
        keep = (idx >= i0 - MARGIN_S * RATE_STEP) & (idx <= i1 + MARGIN_S * RATE_STEP)
        out.append(a[keep])
    return pd.concat(out, ignore_index=True)


def source_per_row(mission, seg_map):
    src = np.empty(len(mission), dtype=object)
    for r in seg_map.itertuples():
        src[int(r.dst_start_idx):int(r.dst_start_idx) + int(r.n_samples)] = r.source_flight
    return src


def predict_lofo(train, mission, src, cols):
    """
    Leave-one-flight-out over the mission's source flights: for each source Fk,
    fit on the training windows of the other flights (standardized), predict the
    mission rows whose source is Fk.
    """
    pred = np.full(len(mission), np.nan)
    for fk in sorted(set(src)):
        tr = train[train.flight_id != fk]
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
        m = make_model()
        m.fit((tr[cols] - mu) / sd, tr["power"])
        rows = src == fk
        pred[rows] = m.predict((mission.loc[rows, cols] - mu) / sd)
    return pred


def main():
    print("\n[Step 17] No-mass vs with-mass on SIM_mission (20 Hz)")

    train = load_training_windows()
    print(f"  Training windows: {len(train)} rows, {train.flight_id.nunique()} flights")

    mission = pd.read_csv(os.path.join(MISSION_DIR, "flight_resampled.csv"))
    seg_map = pd.read_csv(os.path.join(MISSION_DIR, "segment_map.csv"))
    t = mission["timestamp"].values
    y = mission["power"].values
    src = source_per_row(mission, seg_map)
    thr = 0.5 * (np.percentile(y, 5) + np.percentile(y, 95))
    cruise = y > thr
    print(f"  Mission: {len(mission)} rows, sources={sorted(set(src))}, "
          f"{cruise.sum()} cruise rows")

    preds, metrics = {}, {}
    for name, cols in CONFIGS.items():
        p = predict_lofo(train, mission, src, cols)
        preds[name] = p
        metrics[name] = {
            "mae_all":    mean_absolute_error(y, p),
            "r2_all":     r2_score(y, p),
            "mae_cruise": mean_absolute_error(y[cruise], p[cruise]),
            "r2_cruise":  r2_score(y[cruise], p[cruise]),
        }
        m = metrics[name]
        print(f"  {name:42s}  all: MAE {m['mae_all']:5.1f} R² {m['r2_all']:.3f} | "
              f"cruise: MAE {m['mae_cruise']:5.1f} R² {m['r2_cruise']:.3f}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(t, y, color="#0f172a", lw=1.6, label="ACTUAL power", zorder=5)
    for name, cols in CONFIGS.items():
        m = metrics[name]
        ax.plot(t, preds[name], color=COLORS[name], lw=1.2, alpha=0.9,
                label=f"{name}\n   all MAE {m['mae_all']:.0f} W (R² {m['r2_all']:.2f}) | "
                      f"cruise MAE {m['mae_cruise']:.0f} W (R² {m['r2_cruise']:.2f})")
    ymax = max(y.max(), max(p.max() for p in preds.values()))
    ax.set_ylim(-20, ymax * 1.18)
    for r in seg_map.itertuples():
        ax.axvline(r.t_start_s, color="k", ls=":", lw=0.6, alpha=0.4)
        ph = getattr(r, "phase", r.source_flight)
        ax.text((r.t_start_s + r.t_end_s) / 2, ymax * 1.03,
                f"{ph}\n({r.source_flight})", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#334155")
    ax.set_xlabel("time [s]"); ax.set_ylabel("power [W]")
    ax.set_title("SIM_mission — row-level 20 Hz model: no-mass vs with-mass\n"
                 "leave-one-flight-out; rebuilt from NO_MASS_STUDY/MISSION12 recipe",
                 fontweight="bold", pad=28)
    ax.legend(loc="lower center", fontsize=8.5, framealpha=0.95)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(MISSION_DIR, "nomass_vs_withmass.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()

    out = pd.DataFrame({"timestamp": t, "source_flight": src, "actual_power": y.round(1)})
    for name in CONFIGS:
        tag = "nomass" if name.startswith("no-mass") else "withmass"
        out[f"pred_{tag}"] = preds[name].round(1)
    out_csv = os.path.join(MISSION_DIR, "nomass_vs_withmass.csv")
    out.to_csv(out_csv, index=False)

    print(f"\n  Saved -> {out_png}")
    print(f"          {out_csv}\n")


if __name__ == "__main__":
    main()
