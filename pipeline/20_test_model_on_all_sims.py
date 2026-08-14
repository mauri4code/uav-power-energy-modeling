"""
Step 20 – Apply the row-level 20 Hz model (no-mass vs with-mass) to every SIM
flight, not just SIM_mission.

Reuses the exact NO_MASS_STUDY/MISSION12 recipe (see step 17): XGBoost
reg:absoluteerror, ±12 s training window, per-fold standardization,
leave-one-flight-out. The per-fold models are cached and reused across the
different SIM flights so each (feature-set, held-out-flight) is fitted once.

The other SIM flights (progressive / random / varied) are pure cruise segments
(no takeoff/landing), so metrics are reported on all rows.

Input : flights/F*/flight_resampled.csv
        SIM_FLIGHTS/<variant>/flight_resampled.csv (+ segment_map.csv)
Output : SIM_FLIGHTS/<variant>/nomass_vs_withmass.png
         SIM_FLIGHTS/<variant>/nomass_vs_withmass.csv

Run: python 20_test_model_on_all_sims.py
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
SIM_DIR     = os.path.join(SCRIPT_DIR, "SIM_FLIGHTS")

# which SIM variants to score (SIM_mission already done in step 17)
VARIANTS = ["SIM_progressive", "SIM_random", "SIM_varied"]

MARGIN_S, RATE_STEP = 12, 20

MOTORS   = ["motor_1_front_right", "motor_2_rear_left",
            "motor_3_front_left",  "motor_4_rear_right"]
IMU      = ["uav1_hw_api_imu__angular_velocity_x", "uav1_hw_api_imu__angular_velocity_y",
            "uav1_hw_api_imu__angular_velocity_z", "uav1_hw_api_imu__linear_acceleration_x",
            "uav1_hw_api_imu__linear_acceleration_y", "uav1_hw_api_imu__linear_acceleration_z"]
VELOCITY = ["uav1_estimation_manager_uav_state__velocity_linear_x",
            "uav1_estimation_manager_uav_state__velocity_linear_y",
            "uav1_estimation_manager_uav_state__velocity_linear_z"]
SPEED, MASS = ["speed_3d"], ["payload_mass"]

CONFIGS = {
    "no-mass (motors + imu)":                MOTORS + IMU,
    "with-mass (motors + mass + vel + spd)": MOTORS + MASS + VELOCITY + SPEED,
}
COLORS = {"no-mass (motors + imu)": "#2563eb",
          "with-mass (motors + mass + vel + spd)": "#dc2626"}
ALL_COLS = sorted(set(MOTORS + IMU + VELOCITY + SPEED + MASS))


def make_model():
    return XGBRegressor(objective="reg:absoluteerror", n_estimators=150, max_depth=5,
                        learning_rate=0.1, subsample=0.9, n_jobs=-1, random_state=42)


def load_training_windows():
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


# cache: (config_name, held_out_flight) -> (fitted_model, mu, sd)
_CACHE = {}


def get_fold(train, cfg_name, cols, fk):
    key = (cfg_name, fk)
    if key not in _CACHE:
        tr = train[train.flight_id != fk]
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
        m = make_model().fit((tr[cols] - mu) / sd, tr["power"])
        _CACHE[key] = (m, mu, sd)
    return _CACHE[key]


def source_per_row(df, seg_map):
    src = np.empty(len(df), dtype=object)
    for r in seg_map.itertuples():
        src[int(r.dst_start_idx):int(r.dst_start_idx) + int(r.n_samples)] = r.source_flight
    return src


def predict(train, df, src, cfg_name, cols):
    p = np.full(len(df), np.nan)
    for fk in sorted(set(src)):
        m, mu, sd = get_fold(train, cfg_name, cols, fk)
        rows = src == fk
        p[rows] = m.predict((df.loc[rows, cols] - mu) / sd)
    return p


def run_variant(train, variant):
    folder = os.path.join(SIM_DIR, variant)
    df  = pd.read_csv(os.path.join(folder, "flight_resampled.csv"))
    seg = pd.read_csv(os.path.join(folder, "segment_map.csv"))
    t, y = df["timestamp"].values, df["power"].values
    src = source_per_row(df, seg)

    preds, metrics = {}, {}
    for name, cols in CONFIGS.items():
        p = predict(train, df, src, name, cols)
        preds[name] = p
        metrics[name] = (mean_absolute_error(y, p), r2_score(y, p))

    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.plot(t, y, color="#0f172a", lw=1.4, label="ACTUAL power", zorder=5)
    for name in CONFIGS:
        mae, r2 = metrics[name]
        ax.plot(t, preds[name], color=COLORS[name], lw=1.1, alpha=0.9,
                label=f"{name}   MAE {mae:.0f} W (R² {r2:.2f})")
    ymax = max(y.max(), max(p.max() for p in preds.values()))
    ax.set_ylim(min(0, y.min()) - 20, ymax * 1.15)
    for r in seg.itertuples():
        ax.axvline(r.t_start_s, color="k", ls=":", lw=0.5, alpha=0.35)
        ax.text((r.t_start_s + r.t_end_s) / 2, ymax * 1.02, r.source_flight,
                ha="center", va="bottom", fontsize=7, color="#334155")
    ax.set_xlabel("time [s]"); ax.set_ylabel("power [W]")
    ax.set_title(f"{variant} — row-level 20 Hz model: no-mass vs with-mass\n"
                 "leave-one-flight-out; NO_MASS_STUDY/MISSION12 recipe",
                 fontweight="bold", pad=22)
    ax.legend(loc="lower center", fontsize=9, framealpha=0.95, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    png = os.path.join(folder, "nomass_vs_withmass.png")
    plt.savefig(png, dpi=150, bbox_inches="tight"); plt.close()

    out = pd.DataFrame({"timestamp": t, "source_flight": src, "actual_power": y.round(1)})
    for name in CONFIGS:
        tag = "nomass" if name.startswith("no-mass") else "withmass"
        out[f"pred_{tag}"] = preds[name].round(1)
    out.to_csv(os.path.join(folder, "nomass_vs_withmass.csv"), index=False)

    mn, mw = metrics["no-mass (motors + imu)"], metrics["with-mass (motors + mass + vel + spd)"]
    print(f"  {variant:16s}  no-mass MAE {mn[0]:5.1f} R² {mn[1]:.3f} | "
          f"with-mass MAE {mw[0]:5.1f} R² {mw[1]:.3f}   -> {os.path.basename(png)}")


def main():
    print("\n[Step 20] Row-level model on all SIM flights (20 Hz)")
    train = load_training_windows()
    print(f"  Training windows: {len(train)} rows, {train.flight_id.nunique()} flights\n")
    for v in VARIANTS:
        run_variant(train, v)
    print()


if __name__ == "__main__":
    main()
