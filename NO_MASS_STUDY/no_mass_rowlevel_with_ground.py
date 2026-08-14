"""
NO-MASS, PER-INSTANT, INCLUDING TAKEOFF AND LANDING.

STANDALONE. Reads only flights/F*/flight_resampled.csv, writes only into this folder.

THE QUESTION
------------
Every other study here removes ground samples, on the grounds that the ~25 W idle
against ~750 W airborne contrast lets a model score well merely for detecting
whether the aircraft is flying. This file keeps them, so the claim can be checked
rather than asserted.

Two datasets, identical models, identical cross-validation:
  FULL       every sample: ground idle, takeoff, cruise, landing, ground idle
  IN-FLIGHT  the filtered version used elsewhere

Payload mass and payload position remain withheld throughout.

WHAT TO LOOK FOR
----------------
The headline metrics will look far better on FULL. The error breakdown by phase
shows why: almost all of the apparent skill comes from the ground samples, which
are trivially predictable (motors near zero -> power near 25 W). The cruise error
is what actually matters, and it is reported separately.

Everything runs at 1 Hz — power is genuinely measured once per second, and the
20 Hz grid is a forward-filled copy of it.

Output : ground_vs_inflight_results.csv
         ground_vs_inflight_by_phase.csv
         plot_ground_vs_inflight_time.png
         plot_ground_vs_inflight_metrics.png

Run: python no_mass_rowlevel_with_ground.py            # F08 F09 F13
     python no_mass_rowlevel_with_ground.py F01 F02
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

MOTORS = ["motor_1_front_right", "motor_2_rear_left",
          "motor_3_front_left",  "motor_4_rear_right"]

GROUPS = {                                    # no mass, no payload position
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
# WITH_MASS=1 adds payload_mass to every feature set, so the identical three
# dataset variants can be compared with and without the payload known.
# Two ways to switch it on, so it works from a terminal or from an IDE run button:
#     python no_mass_rowlevel_with_ground.py --with-mass
#     WITH_MASS=1 python no_mass_rowlevel_with_ground.py
WITH_MASS = (os.environ.get("WITH_MASS", "0") == "1") or ("--with-mass" in sys.argv)
if WITH_MASS:
    GROUPS["mass"] = ["payload_mass"]

_suffix = "with mass" if WITH_MASS else "no mass"
SETS = [("motors" + (" + mass" if WITH_MASS else ""),
         ["motors"] + (["mass"] if WITH_MASS else [])),
        (f"ALL features ({_suffix})", list(GROUPS))]

# seconds of takeoff/landing kept either side of the flight in the MARGIN variant
MARGIN_S = int(os.environ.get("MARGIN_S", "5"))
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
    """
    Whole recording at 1 Hz, with a phase label per second.

    phase: ground     below the in-flight threshold, outside the flight window
           transition below threshold but inside the takeoff/landing edges
           cruise     above threshold
    """
    out = []
    for f in sorted(glob.glob(os.path.join(fdir, "F*", "flight_resampled.csv"))):
        d = pd.read_csv(f).sort_values("timestamp").copy()
        thr = 0.5 * (np.percentile(d["power"], 5) + np.percentile(d["power"], 95))
        d["t"] = d["timestamp"] - d["timestamp"].iloc[0]
        d["sec"] = np.floor(d["t"]).astype(int)
        a = d.groupby("sec", as_index=False)[ALL + ["power", "t"]].mean()
        a["flight_id"] = d["flight_id"].iloc[0]
        a["true_mass"] = d["payload_mass"].iloc[0]
        a["payload_mass"] = d["payload_mass"].iloc[0]

        hi = (a["power"] > thr).values
        i0, i1 = hi.argmax(), len(hi) - 1 - hi[::-1].argmax()
        phase = np.where(hi, "cruise", "ground")
        edge = np.zeros(len(a), bool)
        edge[max(i0 - MARGIN_S, 0):i0] = True       # MARGIN_S before first crossing
        edge[i1 + 1:i1 + 1 + MARGIN_S] = True       # MARGIN_S after last crossing
        phase = np.where(edge & ~hi, "transition", phase)
        a["phase"] = phase
        a["threshold_w"] = thr
        out.append(a)
    return pd.concat(out, ignore_index=True)


def make_margin(full):
    """
    Middle ground: keep the flight plus MARGIN_S seconds either side.

    Drops the long ground-idle blocks — which are trivially predictable and
    inflate R2 — while retaining the takeoff and landing transitions, which are
    the physically interesting part and the one place a per-instant model can do
    something a flight-level model cannot.

    In phase terms: cruise + transition, without ground.
    """
    return full[full.phase != "ground"].reset_index(drop=True)


def cols_for(groups):
    return [c for g in groups for c in GROUPS[g]]


def evaluate(data, cols):
    """Leave-one-flight-out; returns predictions aligned to `data` row order."""
    pred = np.full(len(data), np.nan)
    for h in sorted(data["flight_id"].unique()):
        tr_m = data.flight_id != h
        te_m = ~tr_m
        tr, te = data[tr_m], data[te_m]
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
        m = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.1,
                         subsample=0.9, n_jobs=-1, random_state=42)
        m.fit((tr[cols] - mu) / sd, tr["power"])
        pred[te_m.values] = m.predict((te[cols] - mu) / sd)
    return pred


def main():
    full = load_1hz(find_flights_dir())
    margin = make_margin(full)
    inflight = full[full.phase == "cruise"].reset_index(drop=True)
    flights = sorted(full["flight_id"].unique())
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show = [f for f in (args or ["F08", "F09", "F13"]) if f in flights]

    print(f"\n[PER-INSTANT, WITH GROUND]  "
          f"payload mass {'INCLUDED' if WITH_MASS else 'WITHHELD'}", flush=True)
    print(f"  flights from: {find_flights_dir()}", flush=True)
    print(f"  FULL      {len(full):5d} rows  "
          f"({100*(full.phase != 'cruise').mean():.0f}% not cruise)")
    print(f"  MARGIN    {len(margin):5d} rows  "
          f"(flight + {MARGIN_S} s either side; ground idle removed)")
    print(f"  IN-FLIGHT {len(inflight):5d} rows\n")

    res, byphase, preds_full, preds_marg = [], [], {}, {}
    for label, groups in SETS:
        cols = cols_for(groups)

        print(f"  fitting '{label}' ({len(cols)} features) ...", flush=True)
        print("     FULL      ", end="", flush=True); p_full = evaluate(full, cols)
        print("done", flush=True)
        print("     MARGIN    ", end="", flush=True); p_marg = evaluate(margin, cols)
        print("done", flush=True)
        print("     IN-FLIGHT ", end="", flush=True); p_infl = evaluate(inflight, cols)
        print("done", flush=True)
        preds_full[label] = p_full
        preds_marg[label] = p_marg

        res.append({"feature_set": label, "dataset": "FULL (with ground)",
                    "n_rows": len(full),
                    "r2": round(float(r2_score(full.power, p_full)), 3),
                    "mae_w": round(float(mean_absolute_error(full.power, p_full)), 1)})
        res.append({"feature_set": label, "dataset": f"MARGIN (+/-{MARGIN_S}s)",
                    "n_rows": len(margin),
                    "r2": round(float(r2_score(margin.power, p_marg)), 3),
                    "mae_w": round(float(mean_absolute_error(margin.power, p_marg)), 1)})
        # the number that matters: how the MARGIN model does on cruise samples alone
        mk = (margin.phase == "cruise").values
        res[-1]["cruise_mae_w"] = round(float(mean_absolute_error(
            margin.power[mk], p_marg[mk])), 1)
        res.append({"feature_set": label, "dataset": "IN-FLIGHT only",
                    "n_rows": len(inflight),
                    "r2": round(float(r2_score(inflight.power, p_infl)), 3),
                    "mae_w": round(float(mean_absolute_error(inflight.power, p_infl)), 1)})

        # where does the FULL model's error actually live?
        for ph in ["ground", "transition", "cruise"]:
            k = (full.phase == ph).values
            if k.sum() < 5:
                continue
            byphase.append({"feature_set": label, "phase": ph, "n_rows": int(k.sum()),
                            "share_pct": round(100 * k.mean(), 1),
                            "mae_w": round(float(mean_absolute_error(
                                full.power[k], p_full[k])), 1),
                            "mean_power_w": round(float(full.power[k].mean()), 0)})

        f_, m_, i_ = res[-3], res[-2], res[-1]
        print(f"  {label:24s}")
        print(f"      FULL      R² {f_['r2']:+.3f}  MAE {f_['mae_w']:5.1f} W")
        print(f"      MARGIN    R² {m_['r2']:+.3f}  MAE {m_['mae_w']:5.1f} W   "
              f"(cruise only: {m_['cruise_mae_w']:.1f} W)")
        print(f"      IN-FLIGHT R² {i_['r2']:+.3f}  MAE {i_['mae_w']:5.1f} W")

    R, B = pd.DataFrame(res), pd.DataFrame(byphase)
    R.to_csv(os.path.join(OUT, "ground_vs_inflight_results.csv"), index=False)
    B.to_csv(os.path.join(OUT, "ground_vs_inflight_by_phase.csv"), index=False)
    print("\n  Where the FULL model's error lives:\n")
    print(B.to_string(index=False))

    # ---------------- time plot over the whole recording ----------------
    fig, axes = plt.subplots(len(show), 1, figsize=(14, 4.4 * len(show)))
    if len(show) == 1:
        axes = [axes]
    for ax, fid in zip(axes, show):
        k = (full.flight_id == fid).values
        te = full[k].sort_values("t")
        t, y = te["t"].values, te["power"].values
        ax.fill_between(t, 0, y.max() * 1.1, where=(te.phase != "cruise").values,
                        color="#94a3b8", alpha=0.18, step="mid", zorder=0,
                        label="ground / transition")
        ax.plot(t, y, color="#334155", lw=1.5, label="ACTUAL power", zorder=4)
        for (label, _), color in zip(SETS, ["#2563eb", "#f59e0b"]):
            ax.plot(t, preds_full[label][k], color=color, lw=1.2, alpha=0.9, zorder=3,
                    label=f"{label}  (MAE {mean_absolute_error(y, preds_full[label][k]):.0f} W)")
        ax.set_title(f"{fid} — whole recording, model trained WITH ground samples "
                     f"(payload {te['true_mass'].iloc[0]:.2f} kg, withheld)",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Power (W)", fontsize=10)
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, y.max() * 1.1)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="center left", framealpha=0.94)
    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle("Per-instant prediction including takeoff and landing",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p1 = os.path.join(OUT, "plot_ground_vs_inflight_time.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()

    # ---------------- MARGIN variant over time ----------------
    # Only the flight plus MARGIN_S seconds either side is drawn, because that is
    # all this model ever saw. The takeoff and landing edges are visible at the
    # extremes; the long ground-idle blocks are gone.
    fig, axes = plt.subplots(len(show), 1, figsize=(14, 4.4 * len(show)))
    if len(show) == 1:
        axes = [axes]
    for ax, fid in zip(axes, show):
        k = (margin.flight_id == fid).values
        te = margin[k].sort_values("t")
        t, y = te["t"].values, te["power"].values
        tr_mask = (te.phase == "transition").values
        ax.fill_between(t, 0, y.max() * 1.1, where=tr_mask, color="#f59e0b",
                        alpha=0.18, step="mid", zorder=0,
                        label=f"takeoff / landing (kept, {MARGIN_S} s each side)")
        ax.plot(t, y, color="#334155", lw=1.5, label="ACTUAL power", zorder=4)
        for (label, _), color in zip(SETS, ["#2563eb", "#16a34a"]):
            pk = preds_marg[label][k]
            cm = mean_absolute_error(y[~tr_mask], pk[~tr_mask])
            ax.plot(t, pk, color=color, lw=1.2, alpha=0.9, zorder=3,
                    label=f"{label}  (all {mean_absolute_error(y, pk):.0f} W, "
                          f"cruise {cm:.0f} W)")
        ax.set_title(f"{fid} — flight + {MARGIN_S} s either side  "
                     f"(payload {te['true_mass'].iloc[0]:.2f} kg, withheld)",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Power (W)", fontsize=10)
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, y.max() * 1.1)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="center left", framealpha=0.94)
    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle(f"MARGIN variant: ground idle removed, takeoff and landing kept",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p3 = os.path.join(OUT, "plot_margin_time.png")
    plt.savefig(p3, dpi=150, bbox_inches="tight"); plt.close()

    # ---------------- metric comparison ----------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    labels = [s[0] for s in SETS]
    x = np.arange(len(labels)); w = 0.27
    for j, (ds, color) in enumerate([("FULL (with ground)", "#94a3b8"),
                                     (f"MARGIN (+/-{MARGIN_S}s)", "#16a34a"),
                                     ("IN-FLIGHT only", "#2563eb")]):
        sub = R[R.dataset == ds].set_index("feature_set").loc[labels]
        ax[0].bar(x + (j - 1) * w, sub["r2"], w, color=color, label=ds)
        ax[1].bar(x + (j - 1) * w, sub["mae_w"], w, color=color, label=ds)
    for a, t_, nice in zip(ax, ["R²", "MAE (W)"],
                           ["R² — higher looks better", "MAE (W) — lower is better"]):
        a.set_xticks(x); a.set_xticklabels(labels, fontsize=9)
        a.set_title(nice, fontsize=11, fontweight="bold")
        a.grid(axis="y", alpha=0.3); a.legend(fontsize=9)
    cruise = B[B.phase == "cruise"].set_index("feature_set").loc[labels, "mae_w"]
    ax[1].plot(x, cruise.values, "r*", ms=16, zorder=5,
               label="FULL model, cruise samples only")
    ax[1].legend(fontsize=9)
    fig.suptitle("Including ground samples inflates R² while the cruise error is unchanged",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p2 = os.path.join(OUT, "plot_ground_vs_inflight_metrics.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()

    print(f"\n  Saved → {p1}")
    print(f"  Saved → {p3}")
    print(f"  Saved → {p2}\n")


if __name__ == "__main__":
    main()
