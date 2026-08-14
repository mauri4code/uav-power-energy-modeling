"""
LAG STUDY — does giving the model the recent past reduce the error?

STANDALONE. Reads only flights/F*/flight_resampled.csv, writes into LAGS/.
Nothing else in the project is touched or overwritten.

THE IDEA
--------
Every model so far uses only INSTANTANEOUS values: the motor command at time t
predicts the power at time t. But power does not respond instantly to a command.
There is ESC and battery dynamics, and the 1 Hz battery reading is effectively an
average over the second preceding it. So the value being predicted depends on the
recent past, not just on the present instant.

This is the last untested modelling lever. Everything else has been exhausted:
feature selection (exhaustive at flight level, greedy at row level), sampling rate
(1 Hz beats 20 Hz), loss function (absolute beat squared by ~9 W), window choice
(changes scope, not cruise accuracy), hyperparameters and model class.

WHAT IT TESTS
-------------
Baseline is the best configuration found so far:
    motors + payload_mass, 1 Hz, mission window +/-12 s, absolute-error loss.

Against it, four ways of adding the recent past:
    lag1        motor commands one second earlier
    lag1+2      one and two seconds earlier
    roll3       rolling mean of the motor commands over the last 3 s
    roll5       rolling mean over the last 5 s
    lag1+roll3  both

Lags are computed WITHIN each flight, so no value ever leaks across a flight
boundary. Rows at the start of a flight where the lag is undefined are dropped,
and the same rows are dropped for every variant so the comparison stays fair.

Evaluation is unchanged: leave-one-flight-out, fit on 13 flights, predict the 14th.

Output (in LAGS/):
    lag_results.csv          every variant scored
    plot_lag_comparison.png  MAE and R2 by variant
    plot_lag_time.png        power over time for the held-out flights, best 3 variants

Run:
    python lag_study.py
    python lag_study.py --no-mass       withhold payload mass
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
OUT = os.path.join(HERE, "LAGS")
os.makedirs(OUT, exist_ok=True)

MARGIN_S = 12
OBJ = "reg:absoluteerror"
SHOW = ["F08", "F09", "F13"]
WITH_MASS = "--no-mass" not in sys.argv
TAG = "withmass" if WITH_MASS else "nomass"

MOTORS = ["motor_1_front_right", "motor_2_rear_left",
          "motor_3_front_left",  "motor_4_rear_right"]
BASE = MOTORS + (["payload_mass"] if WITH_MASS else [])

MAX_SHIFT = 5          # rows dropped at the start of every flight, for fairness


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
    """1 Hz mission window, with lagged and rolling motor features per flight."""
    out = []
    for f in sorted(glob.glob(os.path.join(fdir, "F*", "flight_resampled.csv"))):
        d = pd.read_csv(f).sort_values("timestamp").copy()
        thr = 0.5 * (np.percentile(d["power"], 5) + np.percentile(d["power"], 95))
        d["t"] = d["timestamp"] - d["timestamp"].iloc[0]
        d["sec"] = np.floor(d["t"]).astype(int)
        a = d.groupby("sec", as_index=False)[MOTORS + ["payload_mass", "power", "t"]].mean()
        a["flight_id"] = d["flight_id"].iloc[0]

        hi = (a["power"] > thr).values
        i0, i1 = hi.argmax(), len(hi) - 1 - hi[::-1].argmax()
        idx = np.arange(len(a))
        a["phase"] = np.where(hi, "cruise", "other")
        a = a[(idx >= i0 - MARGIN_S) & (idx <= i1 + MARGIN_S)].reset_index(drop=True)

        # lags and rolling means, computed inside this flight only
        for m in MOTORS:
            a[f"{m}_lag1"] = a[m].shift(1)
            a[f"{m}_lag2"] = a[m].shift(2)
            a[f"{m}_roll3"] = a[m].rolling(3, min_periods=3).mean()
            a[f"{m}_roll5"] = a[m].rolling(5, min_periods=5).mean()

        # drop the first MAX_SHIFT rows so every variant sees identical rows
        out.append(a.iloc[MAX_SHIFT:].reset_index(drop=True))
    return pd.concat(out, ignore_index=True)


VARIANTS = [
    ("baseline (instantaneous)", []),
    ("+ lag1",                   [f"{m}_lag1" for m in MOTORS]),
    ("+ lag1 + lag2",            [f"{m}_lag1" for m in MOTORS] +
                                 [f"{m}_lag2" for m in MOTORS]),
    ("+ roll3",                  [f"{m}_roll3" for m in MOTORS]),
    ("+ roll5",                  [f"{m}_roll5" for m in MOTORS]),
    ("+ lag1 + roll3",           [f"{m}_lag1" for m in MOTORS] +
                                 [f"{m}_roll3" for m in MOTORS]),
]


def evaluate(data, cols):
    pred = np.full(len(data), np.nan)
    for h in sorted(data["flight_id"].unique()):
        te_m = (data.flight_id == h).values
        tr, te = data[~te_m], data[te_m]
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
        m = XGBRegressor(objective=OBJ, n_estimators=150, max_depth=5,
                         learning_rate=0.1, subsample=0.9, n_jobs=-1, random_state=42)
        m.fit((tr[cols] - mu) / sd, tr["power"])
        pred[te_m] = m.predict((te[cols] - mu) / sd)
    return pred


def main():
    data = load(find_flights_dir())
    ck = (data.phase == "cruise").values
    y = data["power"].values

    print(f"\n[LAG STUDY]  1 Hz, window ±{MARGIN_S} s, "
          f"mass {'INCLUDED' if WITH_MASS else 'WITHHELD'}")
    print(f"  {len(data)} rows after dropping the first {MAX_SHIFT} s of each flight")
    print(f"  base features: {' + '.join(BASE)}\n")

    rows, preds = [], {}
    for label, extra in VARIANTS:
        cols = BASE + extra
        p = evaluate(data, cols)
        preds[label] = p
        rec = {"variant": label, "n_features": len(cols),
               "r2": round(float(r2_score(y, p)), 4),
               "mae_w": round(float(mean_absolute_error(y, p)), 2),
               "cruise_mae_w": round(float(mean_absolute_error(y[ck], p[ck])), 2),
               "other_mae_w": round(float(mean_absolute_error(y[~ck], p[~ck])), 2)
                              if (~ck).any() else None}
        rows.append(rec)
        print(f"  {label:26s} {len(cols):3d} feats   R² {rec['r2']:+.4f}   "
              f"MAE {rec['mae_w']:6.2f} W   cruise {rec['cruise_mae_w']:6.2f} W",
              flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, f"lag_results_{TAG}.csv"), index=False)

    base = R.iloc[0]
    best = R.loc[R.mae_w.idxmin()]
    delta = base["mae_w"] - best["mae_w"]
    print(f"\n  baseline {base['mae_w']:.2f} W  ->  best {best['mae_w']:.2f} W "
          f"({best['variant']})   change {-delta:+.2f} W")
    if delta < 0.5:
        print("  -> no meaningful gain: the recent past adds nothing here.\n")
    else:
        print(f"  -> temporal features help by {delta:.2f} W.\n")

    # ---- comparison figure ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(R))
    for ax, key, nice in [(axes[0], "mae_w", "MAE (W) — lower is better"),
                          (axes[1], "r2", "R² — higher is better")]:
        cols_ = ["#16a34a" if v == R[key].min() and key == "mae_w"
                 else "#16a34a" if v == R[key].max() and key == "r2"
                 else "#2563eb" for v in R[key]]
        ax.bar(x, R[key], color=cols_)
        ax.axhline(base[key], color="#dc2626", ls="--", lw=2,
                   label=f"baseline = {base[key]:.3f}" if key == "r2"
                   else f"baseline = {base[key]:.2f} W")
        ax.set_xticks(x); ax.set_xticklabels(R["variant"], rotation=30,
                                             ha="right", fontsize=8.5)
        ax.set_ylabel(nice, fontsize=10); ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)
        for i, v in enumerate(R[key]):
            ax.text(i, v, f"{v:.3f}" if key == "r2" else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Does the recent past help? — 1 Hz, window ±{MARGIN_S} s, "
                 f"{'mass included' if WITH_MASS else 'mass withheld'}\n"
                 f"green = best, red dashed = instantaneous baseline",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p1 = os.path.join(OUT, f"plot_lag_comparison_{TAG}.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()

    # ---- time figure, best three variants ----
    top3 = R.nsmallest(3, "mae_w")["variant"].tolist()
    show = [f for f in SHOW if f in set(data.flight_id)]
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(len(show), 1, figsize=(15, 4.3 * len(show)), squeeze=False)
    for r, fid in enumerate(show):
        k = (data.flight_id == fid).values
        d = data[k].sort_values("t")
        t, yy = d["t"].values, d["power"].values
        cck = (d.phase == "cruise").values
        ax = axes[r][0]
        ax.fill_between(t, 0, max(yy) * 1.1, where=~cck, color="#94a3b8",
                        alpha=0.18, step="mid", zorder=0, label="arming / takeoff / landing")
        ax.plot(t, yy, color="#334155", lw=1.5, label="ACTUAL", zorder=5)
        for j, lab in enumerate(top3):
            pp = preds[lab][k]
            ax.plot(t, pp, color=cmap(j), lw=1.0, alpha=0.85, zorder=3,
                    label=f"{lab}  (R² {r2_score(yy, pp):+.3f}, "
                          f"MAE {mean_absolute_error(yy, pp):.0f} W)")
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, max(yy) * 1.1)
        ax.set_ylabel("Power (W)", fontsize=10); ax.grid(alpha=0.3)
        ax.set_title(f"{fid} — held out   (R² is within this flight)",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="center left", framealpha=0.94)
    axes[-1][0].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle(f"Best 3 temporal variants — "
                 f"{'mass included' if WITH_MASS else 'mass withheld'}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p2 = os.path.join(OUT, f"plot_lag_time_{TAG}.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()

    print(f"  Saved → LAGS/{os.path.basename(p1)}")
    print(f"  Saved → LAGS/{os.path.basename(p2)}")
    print(f"  Saved → LAGS/lag_results_{TAG}.csv\n")


if __name__ == "__main__":
    main()
