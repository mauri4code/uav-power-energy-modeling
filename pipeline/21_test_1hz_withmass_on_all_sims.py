"""
Step 21 – Apply the 1 Hz WITH-MASS row-level model to every SIM flight.

Uses only the 1 Hz with-mass winner from NO_MASS_STUDY/MISSION12
(summary_1hz_withmass.json): motors + mass + speed + imbalance.
Same recipe as steps 18/20: XGBoost reg:absoluteerror, ±12 s window,
per-fold standardization, leave-one-flight-out. Both training flights and each
SIM flight are aggregated to 1 Hz (per-second mean). Fold models are cached and
reused across SIM flights.

Input : flights/F*/flight_resampled.csv
        SIM_FLIGHTS/<variant>/flight_resampled.csv (+ segment_map.csv)
Output : SIM_FLIGHTS/<variant>/1hz/withmass_1hz.png
         SIM_FLIGHTS/<variant>/1hz/withmass_1hz.csv

Run: python 21_test_1hz_withmass_on_all_sims.py
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

VARIANTS = ["SIM_progressive", "SIM_random", "SIM_varied", "SIM_mission"]

MARGIN_S, RATE_STEP = 12, 1          # 1 Hz -> 1 row per second

MOTORS    = ["motor_1_front_right", "motor_2_rear_left",
             "motor_3_front_left",  "motor_4_rear_right"]
SPEED     = ["speed_3d"]
MASS      = ["payload_mass"]
IMBALANCE = ["front_rear_imbalance", "diagonal_imbalance"]

# 1 Hz with-mass winner
FEATURES = MOTORS + MASS + SPEED + IMBALANCE
ALL_COLS = sorted(set(FEATURES))
COLOR    = "#dc2626"


def make_model():
    return XGBRegressor(objective="reg:absoluteerror", n_estimators=150, max_depth=5,
                        learning_rate=0.1, subsample=0.9, n_jobs=-1, random_state=42)


def to_1hz(d):
    d = d.sort_values("timestamp").reset_index(drop=True)
    t = d["timestamp"] - d["timestamp"].iloc[0]
    d = d.assign(sec=np.floor(t).astype(int))
    return d.groupby("sec", as_index=False)[ALL_COLS + ["power"]].mean()


def load_training_windows():
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


def sim_1hz(folder):
    df  = pd.read_csv(os.path.join(folder, "flight_resampled.csv"))
    seg = pd.read_csv(os.path.join(folder, "segment_map.csv"))
    src20 = np.empty(len(df), dtype=object)
    for r in seg.itertuples():
        src20[int(r.dst_start_idx):int(r.dst_start_idx) + int(r.n_samples)] = r.source_flight
    df = df.assign(_src=src20,
                   sec=np.floor(df["timestamp"] - df["timestamp"].iloc[0]).astype(int))
    agg  = df.groupby("sec", as_index=False)[ALL_COLS + ["power"]].mean()
    meta = df.groupby("sec", as_index=False).agg(_src=("_src", "first"))
    return agg.merge(meta, on="sec"), seg


_CACHE = {}


def get_fold(train, fk):
    if fk not in _CACHE:
        tr = train[train.flight_id != fk]
        mu, sd = tr[FEATURES].mean(), tr[FEATURES].std().replace(0, 1)
        m = make_model().fit((tr[FEATURES] - mu) / sd, tr["power"])
        _CACHE[fk] = (m, mu, sd)
    return _CACHE[fk]


def run_variant(train, variant):
    folder = os.path.join(SIM_DIR, variant)
    out_dir = os.path.join(folder, "1hz"); os.makedirs(out_dir, exist_ok=True)
    mission, seg = sim_1hz(folder)
    t, y = mission["sec"].values.astype(float), mission["power"].values

    p = np.full(len(mission), np.nan)
    for fk in sorted(mission["_src"].unique()):
        m, mu, sd = get_fold(train, fk)
        rows = (mission["_src"] == fk).values
        p[rows] = m.predict((mission.loc[rows, FEATURES] - mu) / sd)

    mae, r2 = mean_absolute_error(y, p), r2_score(y, p)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(t, y, color="#0f172a", lw=1.8, marker="o", ms=2.5, label="ACTUAL power", zorder=5)
    ax.plot(t, p, color=COLOR, lw=1.5, marker="o", ms=2, alpha=0.9,
            label=f"with-mass 1 Hz (motors + mass + speed + imbal)   MAE {mae:.0f} W (R² {r2:.2f})")
    ymax = max(y.max(), p.max())
    ax.set_ylim(min(0, y.min()) - 20, ymax * 1.15)
    for r in seg.itertuples():
        ax.axvline(r.t_start_s, color="k", ls=":", lw=0.5, alpha=0.35)
        ax.text((r.t_start_s + r.t_end_s) / 2, ymax * 1.02, r.source_flight,
                ha="center", va="bottom", fontsize=7, color="#334155")
    ax.set_xlabel("time [s]"); ax.set_ylabel("power [W]")
    ax.set_title(f"{variant} — 1 Hz with-mass row-level model\n"
                 "leave-one-flight-out; NO_MASS_STUDY/MISSION12 recipe",
                 fontweight="bold", pad=22)
    ax.legend(loc="lower center", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    png = os.path.join(out_dir, "withmass_1hz.png")
    plt.savefig(png, dpi=150, bbox_inches="tight"); plt.close()

    pd.DataFrame({"second": t, "source_flight": mission["_src"].values,
                  "actual_power": y.round(1), "pred_withmass": p.round(1)}
                 ).to_csv(os.path.join(out_dir, "withmass_1hz.csv"), index=False)

    print(f"  {variant:16s}  MAE {mae:5.1f} W   R² {r2:.3f}   -> 1hz/{os.path.basename(png)}")


def main():
    print("\n[Step 21] 1 Hz with-mass model on all SIM flights")
    print(f"  features: {FEATURES}")
    train = load_training_windows()
    print(f"  Training windows (1 Hz): {len(train)} rows, {train.flight_id.nunique()} flights\n")
    for v in VARIANTS:
        run_variant(train, v)
    print()


if __name__ == "__main__":
    main()
