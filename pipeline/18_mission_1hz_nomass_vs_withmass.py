"""
Step 18 – 1 Hz no-mass vs with-mass row-level model on the SIM_mission flight.

Same idea as step 17, but at 1 Hz — the rate at which the study's best model was
found (summary_1hz_nomass.json: R² 0.914). Both the training flights and the
mission flight are aggregated to one row per second (per-second mean), exactly as
mission12_study.load() does for RATE == 1.

Winning 1 Hz feature sets (from summary_1hz_*.json):
  no-mass   : motors + velocity                     (7 features)  R² 0.914  <-- best overall
  with-mass : motors + mass + speed + imbalance     (8 features)  R² 0.923

Model recipe, ±12 s window, per-fold standardization and leave-one-flight-out are
identical to step 17. Rebuilt from recipe (no saved model existed). Mission never
enters training.

Output: SIM_FLIGHTS/SIM_mission/1hz/nomass_vs_withmass_1hz.png
        SIM_FLIGHTS/SIM_mission/1hz/nomass_vs_withmass_1hz.csv

Run: python 18_mission_1hz_nomass_vs_withmass.py
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
OUT_DIR     = os.path.join(MISSION_DIR, "1hz")

MARGIN_S = 12
RATE_STEP = 1           # 1 row per second at 1 Hz (for the ±12 s window)

MOTORS    = ["motor_1_front_right", "motor_2_rear_left",
             "motor_3_front_left",  "motor_4_rear_right"]
VELOCITY  = ["uav1_estimation_manager_uav_state__velocity_linear_x",
             "uav1_estimation_manager_uav_state__velocity_linear_y",
             "uav1_estimation_manager_uav_state__velocity_linear_z"]
SPEED     = ["speed_3d"]
MASS      = ["payload_mass"]
IMBALANCE = ["front_rear_imbalance", "diagonal_imbalance"]

# The two 1 Hz winners from the study.
CONFIGS = {
    "no-mass (motors + velocity)":              MOTORS + VELOCITY,
    "with-mass (motors + mass + speed + imbal)": MOTORS + MASS + SPEED + IMBALANCE,
}
COLORS = {"no-mass (motors + velocity)": "#2563eb",
          "with-mass (motors + mass + speed + imbal)": "#dc2626"}

ALL_COLS = sorted(set(MOTORS + VELOCITY + SPEED + MASS + IMBALANCE))


def make_model():
    return XGBRegressor(objective="reg:absoluteerror", n_estimators=150, max_depth=5,
                        learning_rate=0.1, subsample=0.9, n_jobs=-1, random_state=42)


def to_1hz(d):
    """Aggregate a 20 Hz frame to one row per second (per-second mean)."""
    d = d.sort_values("timestamp").reset_index(drop=True)
    t = d["timestamp"] - d["timestamp"].iloc[0]
    d = d.assign(sec=np.floor(t).astype(int))
    return d.groupby("sec", as_index=False)[ALL_COLS + ["power"]].mean()


def load_training_windows():
    """Each real flight aggregated to 1 Hz, cut to its ±12 s mission window."""
    out = []
    for f in sorted(glob.glob(os.path.join(FLIGHTS_DIR, "F*", "flight_resampled.csv"))):
        raw = pd.read_csv(f)
        thr = 0.5 * (np.percentile(raw["power"], 5) + np.percentile(raw["power"], 95))
        a = to_1hz(raw)
        a["flight_id"] = raw["flight_id"].iloc[0]
        hi = (a["power"] > thr).values
        i0, i1 = hi.argmax(), len(hi) - 1 - hi[::-1].argmax()
        idx = np.arange(len(a))
        keep = (idx >= i0 - MARGIN_S * RATE_STEP) & (idx <= i1 + MARGIN_S * RATE_STEP)
        out.append(a[keep])
    return pd.concat(out, ignore_index=True)


def mission_1hz():
    """Mission aggregated to 1 Hz, with each second's source flight preserved."""
    df = pd.read_csv(os.path.join(MISSION_DIR, "flight_resampled.csv"))
    seg = pd.read_csv(os.path.join(MISSION_DIR, "segment_map.csv"))
    # per-row source at 20 Hz
    src20 = np.empty(len(df), dtype=object)
    phase20 = np.empty(len(df), dtype=object)
    for r in seg.itertuples():
        s, n = int(r.dst_start_idx), int(r.n_samples)
        src20[s:s + n] = r.source_flight
        phase20[s:s + n] = getattr(r, "phase", r.source_flight)
    df = df.assign(_src=src20, _phase=phase20,
                   sec=np.floor(df["timestamp"] - df["timestamp"].iloc[0]).astype(int))
    agg = df.groupby("sec", as_index=False)[ALL_COLS + ["power"]].mean()
    # segment boundaries are on whole seconds, so each second has a single source
    meta = df.groupby("sec", as_index=False).agg(_src=("_src", "first"),
                                                 _phase=("_phase", "first"))
    agg = agg.merge(meta, on="sec")
    return agg


def predict_lofo(train, mission, cols):
    pred = np.full(len(mission), np.nan)
    for fk in sorted(mission["_src"].unique()):
        tr = train[train.flight_id != fk]
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
        m = make_model()
        m.fit((tr[cols] - mu) / sd, tr["power"])
        rows = (mission["_src"] == fk).values
        pred[rows] = m.predict((mission.loc[rows, cols] - mu) / sd)
    return pred


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("\n[Step 18] No-mass vs with-mass on SIM_mission (1 Hz)")

    train = load_training_windows()
    print(f"  Training windows (1 Hz): {len(train)} rows, {train.flight_id.nunique()} flights")

    mission = mission_1hz()
    t = mission["sec"].values.astype(float)
    y = mission["power"].values
    thr = 0.5 * (np.percentile(y, 5) + np.percentile(y, 95))
    cruise = y > thr
    print(f"  Mission (1 Hz): {len(mission)} rows, sources={sorted(mission['_src'].unique())}, "
          f"{cruise.sum()} cruise rows")

    preds, metrics = {}, {}
    for name, cols in CONFIGS.items():
        p = predict_lofo(train, mission, cols)
        preds[name] = p
        metrics[name] = {
            "mae_all":    mean_absolute_error(y, p),
            "r2_all":     r2_score(y, p),
            "mae_cruise": mean_absolute_error(y[cruise], p[cruise]),
            "r2_cruise":  r2_score(y[cruise], p[cruise]),
        }
        m = metrics[name]
        print(f"  {name:44s}  all: MAE {m['mae_all']:5.1f} R² {m['r2_all']:.3f} | "
              f"cruise: MAE {m['mae_cruise']:5.1f} R² {m['r2_cruise']:.3f}")

    # ---- plot ----
    # segment boundaries (seconds)
    seg = pd.read_csv(os.path.join(MISSION_DIR, "segment_map.csv"))
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(t, y, color="#0f172a", lw=1.8, marker="o", ms=2.5, label="ACTUAL power", zorder=5)
    for name, cols in CONFIGS.items():
        m = metrics[name]
        ax.plot(t, preds[name], color=COLORS[name], lw=1.4, marker="o", ms=2, alpha=0.9,
                label=f"{name}\n   all MAE {m['mae_all']:.0f} W (R² {m['r2_all']:.2f}) | "
                      f"cruise MAE {m['mae_cruise']:.0f} W (R² {m['r2_cruise']:.2f})")
    ymax = max(y.max(), max(p.max() for p in preds.values()))
    ax.set_ylim(-20, ymax * 1.18)
    for r in seg.itertuples():
        ax.axvline(r.t_start_s, color="k", ls=":", lw=0.6, alpha=0.4)
        ph = getattr(r, "phase", r.source_flight)
        ax.text((r.t_start_s + r.t_end_s) / 2, ymax * 1.03,
                f"{ph}\n({r.source_flight})", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#334155")
    ax.set_xlabel("time [s]"); ax.set_ylabel("power [W]")
    ax.set_title("SIM_mission — row-level 1 Hz model: no-mass vs with-mass\n"
                 "leave-one-flight-out; rebuilt from NO_MASS_STUDY/MISSION12 recipe",
                 fontweight="bold", pad=28)
    ax.legend(loc="lower center", fontsize=8.5, framealpha=0.95)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, "nomass_vs_withmass_1hz.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()

    out = pd.DataFrame({"second": t, "source_flight": mission["_src"].values,
                        "actual_power": y.round(1)})
    for name in CONFIGS:
        tag = "nomass" if name.startswith("no-mass") else "withmass"
        out[f"pred_{tag}"] = preds[name].round(1)
    out_csv = os.path.join(OUT_DIR, "nomass_vs_withmass_1hz.csv")
    out.to_csv(out_csv, index=False)

    print(f"\n  Saved -> {out_png}")
    print(f"          {out_csv}\n")


if __name__ == "__main__":
    main()
