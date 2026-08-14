"""
EXPERIMENT — where should the mission window start? Fixed seconds, or arming?

STANDALONE. Reads only flights/F*/flight_resampled.csv, writes only here.

THE ISSUE
---------
The MARGIN dataset keeps the flight plus a fixed number of seconds either side.
At 5 s it misses the ARMING event on every flight: motors spin up from zero to
idle throttle 5.3-11.9 s before takeoff, depending on the flight.

Arming is the moment the aircraft starts consuming meaningfully more than
standby, so it is arguably where the mission begins. It is also a physical event
rather than an arbitrary constant, which makes it easier to defend and to
reproduce on a new flight.

NOTE: the power-ON event is NOT in this dataset. Every recording begins with the
aircraft already energised at ~25-27 W (lowest starting value across the 14
flights is 26.8 W), so a 0 -> 25 W transient cannot be captured retrospectively.
Capturing it would require starting the ROS bag before powering the aircraft.

THREE WINDOWS COMPARED
----------------------
  MARGIN 5 s    flight +/- 5 s          (current)
  MARGIN 12 s   flight +/- 12 s         (fixed, wide enough to cover arming)
  ARMED         first motor command > 0 to last, + 3 s of settling

Model: motors + payload_mass, absolute-error loss, leave-one-flight-out, 1 Hz.

Output : arming_window_results.csv
         plot_arming_window.png
Run    : python exp_arming_window.py
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
OBJ = "reg:absoluteerror"
SETTLE_S = 3          # seconds kept after disarm, for the ARMED window

MOTORS = ["motor_1_front_right", "motor_2_rear_left",
          "motor_3_front_left",  "motor_4_rear_right"]
FEATS = MOTORS + ["payload_mass"]


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
    """1 Hz rows with flags for each candidate window and a phase label."""
    out = []
    for f in sorted(glob.glob(os.path.join(fdir, "F*", "flight_resampled.csv"))):
        d = pd.read_csv(f).sort_values("timestamp").copy()
        thr = 0.5 * (np.percentile(d["power"], 5) + np.percentile(d["power"], 95))
        d["t"] = d["timestamp"] - d["timestamp"].iloc[0]
        d["sec"] = np.floor(d["t"]).astype(int)
        a = d.groupby("sec", as_index=False)[MOTORS + ["payload_mass", "power", "t"]].mean()
        a["flight_id"] = d["flight_id"].iloc[0]

        p = a["power"].values
        m = a[MOTORS].mean(axis=1).values
        hi = p > thr
        i0, i1 = hi.argmax(), len(hi) - 1 - hi[::-1].argmax()
        a["phase"] = np.where(hi, "cruise", "ground")

        idx = np.arange(len(a))
        for s in (5, 12):
            a[f"win{s}"] = (idx >= i0 - s) & (idx <= i1 + s)

        # armed window: motors non-zero, plus a settling tail
        armed = m > 0.02
        if armed.any():
            j0 = armed.argmax()
            j1 = len(armed) - 1 - armed[::-1].argmax()
        else:                                   # no arming visible: fall back
            j0, j1 = i0, i1
        a["win_armed"] = (idx >= j0) & (idx <= j1 + SETTLE_S)
        a["arm_gap_s"] = float(a["t"].iloc[i0] - a["t"].iloc[j0])
        out.append(a)
    return pd.concat(out, ignore_index=True)


def evaluate(data):
    pred = np.full(len(data), np.nan)
    for h in sorted(data["flight_id"].unique()):
        te_m = (data.flight_id == h).values
        tr, te = data[~te_m], data[te_m]
        mu, sd = tr[FEATS].mean(), tr[FEATS].std().replace(0, 1)
        mdl = XGBRegressor(objective=OBJ, n_estimators=150, max_depth=5,
                           learning_rate=0.1, subsample=0.9, n_jobs=-1,
                           random_state=42)
        mdl.fit((tr[FEATS] - mu) / sd, tr["power"])
        pred[te_m] = mdl.predict((te[FEATS] - mu) / sd)
    return pred


def main():
    full = load_1hz(find_flights_dir())
    gaps = full.groupby("flight_id")["arm_gap_s"].first()
    print(f"\n[ARMING WINDOW TEST]  arming precedes takeoff by "
          f"{gaps.min():.1f}-{gaps.max():.1f} s (median {gaps.median():.1f} s)\n")

    variants = [("MARGIN 5 s", "win5"), ("MARGIN 12 s", "win12"),
                ("ARMED (motors on)", "win_armed")]
    rows, preds = [], {}
    for name, col in variants:
        ds = full[full[col]].reset_index(drop=True)
        p = evaluate(ds)
        preds[name] = (ds, p)
        y = ds["power"].values
        ck = (ds.phase == "cruise").values
        rec = {"window": name, "n_rows": len(ds),
               "pct_non_cruise": round(100 * (~ck).mean(), 1),
               "r2": round(float(r2_score(y, p)), 3),
               "mae_w": round(float(mean_absolute_error(y, p)), 1),
               "cruise_mae_w": round(float(mean_absolute_error(y[ck], p[ck])), 1),
               "noncruise_mae_w": round(float(mean_absolute_error(y[~ck], p[~ck])), 1)}
        rows.append(rec)
        print(f"  {name:20s} {len(ds):5d} rows ({rec['pct_non_cruise']:4.1f}% not cruise)  "
              f"R² {rec['r2']:+.3f}  MAE {rec['mae_w']:5.1f} W  "
              f"| cruise {rec['cruise_mae_w']:5.1f}  other {rec['noncruise_mae_w']:6.1f}",
              flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "arming_window_results.csv"), index=False)

    # ---- figure: one flight per row, one window per column ----
    show = [f for f in (sys.argv[1:] or ["F08", "F09", "F13"])
            if f in set(full.flight_id)]
    fig, axes = plt.subplots(len(show), 3, figsize=(19, 3.8 * len(show)), squeeze=False)
    for r, fid in enumerate(show):
        for c, (name, _) in enumerate(variants):
            ds, p = preds[name]
            k = (ds.flight_id == fid).values
            d = ds[k].sort_values("t")
            t, y, pp = d["t"].values, d["power"].values, p[k]
            ck = (d.phase == "cruise").values
            ax = axes[r][c]
            ax.fill_between(t, 0, 1000, where=~ck, color="#94a3b8", alpha=0.18,
                            step="mid", zorder=0, label="not cruise")
            ax.plot(t, y, color="#334155", lw=1.4, label="ACTUAL", zorder=3)
            ax.plot(t, pp, color="#dc2626", lw=1.1, alpha=0.9, zorder=2,
                    label=f"predicted (cruise "
                          f"{mean_absolute_error(y[ck], pp[ck]):.0f} W)")
            ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, max(y) * 1.1)
            ax.grid(alpha=0.3)
            ax.set_title(f"{fid} — {name}", fontsize=10, fontweight="bold")
            if c == 0:
                ax.set_ylabel("Power (W)", fontsize=10)
            if r == 0:
                ax.legend(fontsize=7.5, loc="center left", framealpha=0.94)
    for c in range(3):
        axes[-1][c].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle("Where should the mission window start?\n"
                 "5 s misses arming on every flight; ARMED begins at the first "
                 "motor command", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p_out = os.path.join(OUT, "plot_arming_window.png")
    plt.savefig(p_out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  Saved → {p_out}")
    print(f"  Saved → {os.path.join(OUT, 'arming_window_results.csv')}\n")


if __name__ == "__main__":
    main()
