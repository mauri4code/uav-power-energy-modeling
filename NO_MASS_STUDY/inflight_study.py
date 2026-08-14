"""
IN-FLIGHT STUDY — the same search as mission12_study.py, cruise samples only.

STANDALONE. Reads only flights/F*/flight_resampled.csv, writes only into this folder.

THE WINDOW
----------
Cruise only: every sample where power exceeds the per-flight threshold
(0.5 * (P5 + P95), around 340-415 W). No arming, no takeoff, no landing, no ground.
This is mission12_study.py with the margin set to zero.

WHY BOTH EXIST
--------------
The two studies answer different questions and are NOT comparable by R2:

  mission12  arming -> landing. R2 is scored against a target spanning ~25-960 W,
             15% of which is easy non-cruise material. Higher R2, and the honest
             number if the claim is about a complete mission.
  this file  cruise only. Target spans ~550-850 W, so R2 is scored against a much
             narrower range and looks far worse for the same accuracy in watts.

Compare them by MAE, which is in watts and means the same thing in both. In the
runs done so far, cruise MAE was ~35 W either way — the window changes the scope
of the claim, not the quality of the model.

WHAT IT DOES
------------
1. GREEDY FORWARD SELECTION over feature groups. Starts empty, repeatedly adds the
   group that reduces MAE most, and stops when nothing helps. Reports the whole path,
   so you can see the order in which features earn their place and where it plateaus.
   Cheaper and more readable than an exhaustive search, and at row level exhaustive
   is not tractable.
2. TOP-4 REPORTING. Every combination evaluated during the search is recorded;
   the best four are plotted together, because with 14 flights they usually sit
   within a couple of watts of each other and picking one winner overstates the
   margin.
3. TIME PLOT and PER-FLIGHT R2 for those four sets on the held-out flights.

Everything is leave-one-flight-out: fit on 13 flights, predict the 14th, repeat.
Loss is absolute error (reg:absoluteerror) — it beat squared error in every
configuration tested, and it optimises the metric actually being reported.

FOUR RUNS
---------
    1 Hz  without mass      1 Hz  with mass
   20 Hz  without mass     20 Hz  with mass

1 Hz is one row per real battery reading. 20 Hz is the raw resampled grid, where each
battery reading is forward-filled across ~20 rows — included as a check, not as the
recommended configuration.

Output (names carry the rate and the mass setting):
    search_<rate>_<mass>.csv          the greedy path, every step scored
    plot_search_<rate>_<mass>.png     MAE as groups are added
    plot_time_<rate>_<mass>.png       power over time, best 4 sets overlaid
    plot_top4_<rate>_<mass>.png       the best 4 combinations, MAE and R2
    plot_r2_<rate>_<mass>.png         per-flight R2 and MAE, best 4 sets
    per_flight_<rate>_<mass>.csv      the same numbers as a table
    summary_<rate>_<mass>.json        winning set, metrics, phase breakdown

Outputs go to MISSION13/ so they never collide with mission12_study.py.

Run:
    python inflight_study.py
    python inflight_study.py --with-mass
    python inflight_study.py --rate 20
    python inflight_study.py --rate 20 --with-mass
"""

import os
import sys
import glob
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
# All four runs write into this one subfolder, distinguished by the tag in each
# filename, so the configurations can be compared without opening four folders.
OUT = os.path.join(HERE, "MISSION13")
os.makedirs(OUT, exist_ok=True)

MARGIN_S = 0        # cruise only: no arming, no takeoff, no landing
OBJ = "reg:absoluteerror"
SHOW = ["F08", "F09", "F13"]
TOP_K = 4          # how many of the best combinations to plot together

WITH_MASS = ("--with-mass" in sys.argv) or (os.environ.get("WITH_MASS") == "1")
RATE = 20 if "--rate" in sys.argv and sys.argv[sys.argv.index("--rate") + 1] == "20" else 1
TAG = f"{RATE}hz_{'withmass' if WITH_MASS else 'nomass'}"

MOTORS = ["motor_1_front_right", "motor_2_rear_left",
          "motor_3_front_left",  "motor_4_rear_right"]

GROUPS = {
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

    # TRAJECTORY IS DELIBERATELY EXCLUDED.
    # It was tested and removed. trajectory_3 occurs on exactly one flight (F02),
    # so under leave-one-flight-out that column is all-zero in training whenever
    # F02 is held out, and on every other fold it acts as a dedicated F02
    # indicator letting the model absorb that flight's residual. Any gain would
    # therefore be indistinguishable from flagging one outlier.
    # Independent evidence that it carries nothing: trajectory 1 and 2 differ by
    # 0.07 m/s in mean speed and by +5.7 +/- 10.3 W in mean power, and trajectory
    # never entered a leading model in the earlier flight-level search.
    # The one-hot columns are still built in load() if it is ever re-enabled.
    # "trajectory": ["traj_1", "traj_2", "traj_3"],
}
if WITH_MASS:
    GROUPS["mass"] = ["payload_mass"]

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
    sys.exit("Could not find flights/ containing flight_resampled.csv")


def load(fdir):
    """Mission window at the requested rate, with a cruise/other phase label."""
    out = []
    for f in sorted(glob.glob(os.path.join(fdir, "F*", "flight_resampled.csv"))):
        d = pd.read_csv(f).sort_values("timestamp").copy()
        thr = 0.5 * (np.percentile(d["power"], 5) + np.percentile(d["power"], 95))
        d["t"] = d["timestamp"] - d["timestamp"].iloc[0]

        traj = str(d["trajectory"].iloc[0])          # one-hot, constant per flight
        d["traj_1"] = int(traj.endswith("_1"))
        d["traj_2"] = int(traj.endswith("_2"))
        d["traj_3"] = int(traj.endswith("_3"))

        if RATE == 1:
            d["sec"] = np.floor(d["t"]).astype(int)
            a = d.groupby("sec", as_index=False)[ALL + ["power", "t"]].mean()
            step = 1
        else:
            a = d[ALL + ["power", "t"]].reset_index(drop=True)
            step = int(round(1 / np.median(np.diff(d["t"].values))))   # rows per second

        a["flight_id"] = d["flight_id"].iloc[0]
        hi = (a["power"] > thr).values
        i0, i1 = hi.argmax(), len(hi) - 1 - hi[::-1].argmax()
        idx = np.arange(len(a))
        a["phase"] = np.where(hi, "cruise", "other")
        keep = (idx >= i0 - MARGIN_S * step) & (idx <= i1 + MARGIN_S * step)
        out.append(a[keep])
    return pd.concat(out, ignore_index=True)


def cols_for(groups):
    return [c for g in groups for c in GROUPS[g]]


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


def score(data, p):
    y = data["power"].values
    ck = (data.phase == "cruise").values
    return {"r2": round(float(r2_score(y, p)), 4),
            "mae_w": round(float(mean_absolute_error(y, p)), 2),
            "cruise_mae_w": round(float(mean_absolute_error(y[ck], p[ck])), 2),
            "other_mae_w": round(float(mean_absolute_error(y[~ck], p[~ck])), 2)
                           if (~ck).any() else None}


def greedy(data):
    """
    Forward selection: add the group that reduces MAE most, until none does.

    Every candidate evaluated along the way is recorded, not just the winner, so
    the best few combinations can be reported together. With only 14 flights the
    top few are usually within a couple of watts of each other, and showing them
    side by side is more honest than presenting a single 'best' set as if the
    margin were meaningful.
    """
    remaining, chosen, path, best_mae = list(GROUPS), [], [], np.inf
    seen = {}                       # frozenset(groups) -> score dict
    step = 0
    while remaining:
        step += 1
        trials = []
        for g in remaining:
            combo = chosen + [g]
            s = score(data, evaluate(data, cols_for(combo)))
            seen[frozenset(combo)] = {"groups": combo, **s}
            trials.append((s["mae_w"], g, s, len(cols_for(combo))))
            print(f"    step {step}: try +{g:12s} -> MAE {s['mae_w']:6.2f} W", flush=True)
        trials.sort()
        mae, g, s, nfeat = trials[0]
        improved = mae < best_mae - 0.05          # ignore changes under 0.05 W
        path.append({"step": step, "added": g, "n_features": nfeat,
                     "set": " + ".join(chosen + [g]), "improved": bool(improved), **s})
        if not improved:
            print(f"    step {step}: best was +{g} at {mae:.2f} W — no improvement, stop\n",
                  flush=True)
            break
        chosen.append(g); remaining.remove(g); best_mae = mae
        print(f"    step {step}: KEEP +{g}  (MAE {mae:.2f} W, R² {s['r2']:.3f})\n",
              flush=True)

    ranked = sorted(seen.values(), key=lambda d: d["mae_w"])
    return chosen, pd.DataFrame(path), ranked


def fig_top4(ranked, k=4):
    """The best k combinations found, by MAE and by R²."""
    t = ranked[:k][::-1]
    labels = [" + ".join(d["groups"]) for d in t]
    y = np.arange(len(t))
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.6 * len(t) + 3.6))
    for ax, key, nice in [(axes[0], "mae_w", "MAE (W) — lower is better"),
                          (axes[1], "r2", "R² — higher is better")]:
        vals = [d[key] for d in t]
        ax.barh(y, vals, color=["#16a34a" if i == len(t) - 1 else "#2563eb"
                                for i in range(len(t))])
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(nice, fontsize=10); ax.grid(axis="x", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(v, i, f"  {v:.3f}" if key == "r2" else f"  {v:.2f}",
                    va="center", fontsize=9)
    fig.suptitle(f"Best {k} feature combinations — {RATE} Hz, "
                 f"{'mass included' if WITH_MASS else 'mass withheld'}, "
                 f"IN-FLIGHT only\ngreen = best; note how small the "
                 f"spread between them is", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, f"plot_top4_{TAG}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def fig_search(path, chosen):
    """Selection path shown twice: by MAE (what the search optimises) and by R²."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    kept, rej = path[path.improved], path[~path.improved]
    for ax, key, nice, better in [
            (axes[0], "mae_w", "MAE (W)", "lower is better"),
            (axes[1], "r2",    "R²",      "higher is better")]:
        ax.plot(kept["step"], kept[key], "-o", lw=2, color="#2563eb", label="kept")
        if len(rej):
            ax.plot(rej["step"], rej[key], "x", ms=10, color="#dc2626",
                    label="rejected (no gain)")
        for _, r in path.iterrows():
            ax.annotate(f"+{r['added']}", (r["step"], r[key]),
                        textcoords="offset points", xytext=(6, 6), fontsize=8.5)
        ax.set_xlabel("selection step", fontsize=10)
        ax.set_ylabel(f"{nice} — leave-one-flight-out", fontsize=10)
        ax.set_title(f"{nice}   ({better})", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle(f"Greedy feature selection — {RATE} Hz, "
                 f"{'mass included' if WITH_MASS else 'mass withheld'}, "
                 f"IN-FLIGHT only\nfinal set: {' + '.join(chosen)}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, f"plot_search_{TAG}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def fig_r2(data, preds):
    """
    R² for the winning model, per flight and pooled.

    Two figures of merit, because they answer different questions:
      WITHIN   scored against that flight's own mean — does the model track the
               variation during this flight?
      POOLED   scored across all flights at once, so it also gets credit for
               placing different flights at different levels.
    Pooled is always the flattering one; both are drawn so the gap is visible
    rather than hidden.
    """
    flights = sorted(data.flight_id.unique())
    cmap = plt.get_cmap("tab10")
    x = np.arange(len(flights))
    w = 0.8 / max(len(preds), 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.4))
    rows = []
    for j, (label, p_all) in enumerate(preds.items()):
        within = [r2_score(data.power.values[(data.flight_id == f).values],
                           p_all[(data.flight_id == f).values]) for f in flights]
        mae = [mean_absolute_error(data.power.values[(data.flight_id == f).values],
                                   p_all[(data.flight_id == f).values]) for f in flights]
        pooled = r2_score(data.power.values, p_all)
        off = (j - (len(preds) - 1) / 2) * w
        axes[0].bar(x + off, within, w, color=cmap(j),
                    label=f"{label}  (pooled R² {pooled:.3f})")
        axes[1].bar(x + off, mae, w, color=cmap(j),
                    label=f"{label}  (mean {np.mean(mae):.1f} W)")
        for f, r_, m_ in zip(flights, within, mae):
            rows.append({"feature_set": label, "flight": f,
                         "r2_within": round(float(r_), 4), "mae_w": round(float(m_), 2)})

    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xticks(x); axes[0].set_xticklabels(flights, rotation=45, fontsize=8.5)
    axes[0].set_ylabel("R² within each flight", fontsize=10)
    axes[0].set_title("R² per held-out flight\n"
                      "(negative = worse than that flight's own mean; "
                      "pooled values in the legend are much higher)",
                      fontsize=11, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3); axes[0].legend(fontsize=8)

    axes[1].set_xticks(x); axes[1].set_xticklabels(flights, rotation=45, fontsize=8.5)
    axes[1].set_ylabel("MAE (W)", fontsize=10)
    axes[1].set_title("Error per held-out flight", fontsize=11, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3); axes[1].legend(fontsize=8)

    fig.suptitle(f"Best {len(preds)} feature combinations, per flight — {RATE} Hz, "
                 f"{'mass included' if WITH_MASS else 'mass withheld'}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, f"plot_r2_{TAG}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()

    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"per_flight_{TAG}.csv"), index=False)
    return p


def fig_time(data, preds):
    """Power over time for the held-out flights, with the best k sets overlaid."""
    show = [f for f in SHOW if f in set(data.flight_id)]
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(len(show), 1, figsize=(15, 4.4 * len(show)), squeeze=False)
    for r, fid in enumerate(show):
        k = (data.flight_id == fid).values
        d = data[k].sort_values("t")
        t, y = d["t"].values, d["power"].values
        ck = (d.phase == "cruise").values
        ax = axes[r][0]
        ax.fill_between(t, 0, max(y) * 1.1, where=~ck, color="#94a3b8", alpha=0.18,
                        step="mid", zorder=0, label="excluded")
        ax.plot(t, y, color="#334155", lw=1.6, label="ACTUAL", zorder=5)
        for j, (label, p_all) in enumerate(preds.items()):
            p = p_all[k]
            # R2 scored WITHIN this flight — against this flight's own mean.
            # Harsher than the pooled R2 elsewhere, and the right question for a
            # single-flight panel: does this line track this trace?
            ax.plot(t, p, color=cmap(j), lw=1.0, alpha=0.85, zorder=3,
                    label=f"{label}  (R² {r2_score(y, p):+.3f}, "
                          f"cruise {mean_absolute_error(y[ck], p[ck]):.0f} W, "
                          f"all {mean_absolute_error(y, p):.0f} W)")
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, max(y) * 1.1)
        ax.set_ylabel("Power (W)", fontsize=10); ax.grid(alpha=0.3)
        ax.set_title(f"{fid} — held out   (R² below is within this flight)",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=7.5, loc="center left", ncol=2, framealpha=0.94)
    axes[-1][0].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle(f"IN-FLIGHT only (cruise samples) — {RATE} Hz, "
                 f"{'mass included' if WITH_MASS else 'mass withheld'}\n"
                 f"best {len(preds)} feature combinations overlaid",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, f"plot_time_{TAG}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def main():
    fdir = find_flights_dir()
    data = load(fdir)
    print(f"\n[MISSION-12]  {RATE} Hz  |  mass "
          f"{'INCLUDED' if WITH_MASS else 'WITHHELD'}  |  window ±{MARGIN_S} s")
    print(f"  flights from: {fdir}")
    print(f"  {len(data)} rows, {data.flight_id.nunique()} flights, "
          f"{100 * (data.phase != 'cruise').mean():.1f}% not cruise\n")
    print("  greedy forward selection:")

    chosen, path, ranked = greedy(data)
    cols = cols_for(chosen)
    pred = evaluate(data, cols)
    s = score(data, pred)

    # the best TOP_K distinct combinations seen during the search
    top = ranked[:TOP_K]
    print(f"  best {len(top)} combinations:")
    for i, d in enumerate(top, 1):
        print(f"    {i}. {' + '.join(d['groups']):40s} "
              f"MAE {d['mae_w']:6.2f} W   R² {d['r2']:+.4f}", flush=True)
    preds = {" + ".join(d["groups"]): evaluate(data, cols_for(d["groups"]))
             for d in top}
    print()

    print(f"  FINAL: {' + '.join(chosen)}  ({len(cols)} features)")
    # other_mae_w is None when the window contains no non-cruise rows, which is
    # always the case here (margin 0) and possible in mission12 if MARGIN_S is 0.
    _other = ("n/a (cruise only)" if s["other_mae_w"] is None
              else f"{s['other_mae_w']:.2f} W")
    print(f"         R² {s['r2']:+.4f}   MAE {s['mae_w']:.2f} W   "
          f"cruise {s['cruise_mae_w']:.2f} W   other {_other}\n")

    path.to_csv(os.path.join(OUT, f"search_{TAG}.csv"), index=False)
    with open(os.path.join(OUT, f"summary_{TAG}.json"), "w") as fh:
        json.dump({"rate_hz": RATE, "with_mass": WITH_MASS, "margin_s": MARGIN_S,
                   "objective": OBJ, "n_rows": int(len(data)),
                   "chosen_groups": chosen, "features": cols, "metrics": s,
                   "top_combinations": [
                       {"rank": i + 1, "groups": d["groups"], "mae_w": d["mae_w"],
                        "r2": d["r2"], "cruise_mae_w": d["cruise_mae_w"]}
                       for i, d in enumerate(ranked[:TOP_K])],
                   "selection_path": path.to_dict("records")}, fh, indent=2)

    p1 = fig_search(path, chosen)
    p2 = fig_time(data, preds)
    p3 = fig_r2(data, preds)
    p4 = fig_top4(ranked, TOP_K)
    print(f"  Saved → MISSION13/{os.path.basename(p1)}")
    print(f"  Saved → MISSION13/{os.path.basename(p2)}")
    print(f"  Saved → MISSION13/{os.path.basename(p3)}")
    print(f"  Saved → MISSION13/{os.path.basename(p4)}")
    print(f"  Saved → MISSION13/search_{TAG}.csv, summary_{TAG}.json, "
          f"per_flight_{TAG}.csv\n")


if __name__ == "__main__":
    main()
