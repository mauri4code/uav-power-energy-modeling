"""
NO-MASS STUDY — can power be predicted WITHOUT being told the payload?

STANDALONE. Reads only flights/F*/flight_resampled.csv (located by walking up
from this file) and writes only into this folder. Depends on nothing else.

THE QUESTION
------------
The main feature study found that payload mass alone predicts a flight's mean
airborne power to ~12.4 W. That is useful for mission planning, where the load is
known in advance — but it is close to tautological: you are predicting power from
the thing you already decided.

The harder and more practically interesting question is whether the aircraft can
work it out for itself. If payload mass is withheld, can the flight data alone —
motor commands, attitude, speed, altitude, trajectory — recover the same answer?
Physically it should be able to: a heavier aircraft must command more thrust to
hover, so the motor signals ought to encode the load.

WHAT IS EXCLUDED, AND WHY
-------------------------
  payload_mass       the variable being withheld — this is the point
  position_payload   EXCLUDED because it leaks the answer. The category "none"
                     occurs only on the two zero-payload flights, so including it
                     would hand the model part of the mass information through
                     the back door.

WHAT IS NEWLY INCLUDED
----------------------
  trajectory         one-hot encoded (T1/T2/T3). Never used as a feature in the
                     main pipeline; included here because with mass withheld the
                     mission profile is a legitimate candidate predictor and does
                     not leak load information.

METHOD
------
Target: mean power while in flight, one value per flight (n = 14).
Every non-empty combination of feature groups, leave-one-flight-out, linear
regression. In-flight samples only (per-flight threshold at the midpoint of the
5th and 95th power percentile).

Output : no_mass_results.csv          every combination, ranked
         no_mass_predictions.csv      per-flight predictions for the leaders
         no_mass_summary.json         winners, coefficients, comparison to baseline
         plot_no_mass_ranking.png     top combinations
         plot_no_mass_per_flight.png  every flight, predicted while held out
         plot_no_mass_time.png        top sets against real power over time

Run: python no_mass_study.py
"""

import os
import sys
import glob
import json
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE

MOTORS = ["motor_1_front_right", "motor_2_rear_left",
          "motor_3_front_left",  "motor_4_rear_right"]

# NOTE: no "mass" group, and no payload position. See the docstring.
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
    "trajectory":  ["traj_1", "traj_2", "traj_3"],
}


def find_flights_dir():
    d = HERE
    for _ in range(5):
        c = os.path.join(d, "flights")
        if glob.glob(os.path.join(c, "F*", "flight_resampled.csv")):
            return c
        # also look one level into the codes tree
        for cand in glob.glob(os.path.join(d, "*", "*", "*", "flights")):
            if glob.glob(os.path.join(cand, "F*", "flight_resampled.csv")):
                return cand
        d = os.path.dirname(d)
    sys.exit("Could not find a flights/ folder containing flight_resampled.csv")


def build_table(fdir):
    """One row per flight: in-flight means, plus one-hot trajectory."""
    rows = []
    for f in sorted(glob.glob(os.path.join(fdir, "F*", "flight_resampled.csv"))):
        d = pd.read_csv(f)
        thr = 0.5 * (np.percentile(d["power"], 5) + np.percentile(d["power"], 95))
        g = d[d["power"] > thr]
        traj = str(d["trajectory"].iloc[0])
        rec = {"flight": d["flight_id"].iloc[0],
               "power": g["power"].mean(),
               "true_mass": d["payload_mass"].iloc[0],      # kept for reporting only
               "trajectory": traj,
               "duration_s": round(len(g) * 0.05, 1)}
        for cols in GROUPS.values():
            for c in cols:
                if c in g.columns:
                    rec[c] = g[c].mean()
        rec["traj_1"] = int(traj.endswith("_1"))
        rec["traj_2"] = int(traj.endswith("_2"))
        rec["traj_3"] = int(traj.endswith("_3"))
        rec["mean_motor"] = g[MOTORS].mean(axis=1).mean()
        rows.append(rec)
    return pd.DataFrame(rows).reset_index(drop=True)


def cols_for(groups):
    return [c for g in groups for c in GROUPS[g]]


def loo(tab, cols):
    """Leave-one-flight-out predictions."""
    n = len(tab)
    pred = np.empty(n)
    for i in range(n):
        tr = tab.drop(tab.index[i])
        lm = LinearRegression().fit(tr[cols], tr["power"])
        pred[i] = lm.predict(tab.iloc[[i]][cols])[0]
    return pred


def loo_ridge(tab, cols, alphas=(0.01, 0.1, 1.0, 10.0, 100.0)):
    """
    Leave-one-flight-out with ridge regression.

    Needed because the full feature set has 23 predictors against 13 training
    flights. Ordinary least squares is undefined when predictors outnumber
    observations (it can fit the training data exactly and predicts arbitrarily
    off it), which is why the exhaustive search above skips those combinations.
    Ridge adds an L2 penalty so a solution exists; the penalty strength is chosen
    inside each fold by nested cross-validation, never using the held-out flight.

    Features are standardised on the training flights only.
    """
    n = len(tab)
    pred = np.empty(n)
    for i in range(n):
        tr = tab.drop(tab.index[i])
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
        m = RidgeCV(alphas=alphas).fit((tr[cols] - mu) / sd, tr["power"])
        pred[i] = m.predict((tab.iloc[[i]][cols] - mu) / sd)[0]
    return pred


def full_sets(tab):
    """Named large feature sets, evaluated with ridge (OLS cannot fit them)."""
    named = [
        ("ALL features (no mass)", list(GROUPS)),
        ("all except imu",         [g for g in GROUPS if g != "imu"]),
        ("all except trajectory",  [g for g in GROUPS if g != "trajectory"]),
        ("motors + orientation + imu", ["motors", "orientation", "imu"]),
    ]
    res = []
    for label, groups in named:
        cols = cols_for(groups)
        p = loo_ridge(tab, cols)
        res.append({"feature_set": label, "groups": " + ".join(groups),
                    "n_features": len(cols),
                    "r2": round(r2_score(tab["power"], p), 4),
                    "mae_w": round(mean_absolute_error(tab["power"], p), 2),
                    "max_err_w": round(float(np.abs(p - tab["power"]).max()), 1)})
        print(f"    {label:26s} {len(cols):3d} feats  "
              f"R² {res[-1]['r2']:7.4f}  MAE {res[-1]['mae_w']:6.2f} W")
    return pd.DataFrame(res).sort_values("mae_w").reset_index(drop=True)


def search(tab):
    n, names, res = len(tab), list(GROUPS), []
    for k in range(1, len(names) + 1):
        for combo in itertools.combinations(names, k):
            cols = cols_for(combo)
            if len(cols) >= n - 1:
                continue
            p = loo(tab, cols)
            res.append({"groups": " + ".join(combo), "n_groups": k,
                        "n_features": len(cols),
                        "r2": round(r2_score(tab["power"], p), 4),
                        "mae_w": round(mean_absolute_error(tab["power"], p), 2),
                        "max_err_w": round(float(np.abs(p - tab["power"]).max()), 1)})
    return pd.DataFrame(res).sort_values("mae_w").reset_index(drop=True)


# ------------------------------------------------------------------ figures
def fig_ranking(res, baseline, fs, k=12):
    """
    Top OLS combinations plus the large ridge-fitted sets, on one axis.

    The two families are distinguished because they are not fitted the same way:
    the searched combinations use ordinary least squares (<= 12 features), while
    the large sets need ridge, since 23 predictors against 13 training flights
    leaves OLS undefined.
    """
    a = res.head(k)[["groups", "n_features", "r2", "mae_w"]].copy().reset_index(drop=True)
    a["kind"] = "searched (OLS)"
    # fs already carries its own "groups" column, so drop it before renaming
    b = (fs.drop(columns=["groups"])
           .rename(columns={"feature_set": "groups"})
           [["groups", "n_features", "r2", "mae_w"]].copy().reset_index(drop=True))
    b["kind"] = "full sets (ridge)"
    t = (pd.concat([a, b], axis=0, ignore_index=True)
           .sort_values("mae_w", ascending=False)
           .reset_index(drop=True))

    colors = ["#f59e0b" if k_ == "full sets (ridge)" else "#2563eb"
              for k_ in t["kind"]]
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(t) + 3))
    y = np.arange(len(t))
    ax.barh(y, t["mae_w"], color=colors)
    ax.axvline(baseline, color="#dc2626", ls="--", lw=2.2,
               label=f"payload mass alone (baseline) = {baseline:.2f} W")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g}  [{n}]" for g, n in zip(t["groups"], t["n_features"])],
                       fontsize=8.5)
    ax.set_xlabel("MAE (W) — leave-one-flight-out", fontsize=10)
    for i, (v, r) in enumerate(zip(t["mae_w"], t["r2"])):
        ax.text(v + 0.15, i, f"{v:.2f} W  (R² {r:.3f})", va="center", fontsize=8)
    ax.set_xlim(0, max(t["mae_w"].max(), baseline) * 1.32)

    handles = [plt.Rectangle((0, 0), 1, 1, color="#2563eb"),
               plt.Rectangle((0, 0), 1, 1, color="#f59e0b"),
               plt.Line2D([], [], color="#dc2626", ls="--", lw=2.2)]
    ax.legend(handles, ["searched combinations (OLS)",
                        "full / large sets (ridge)",
                        f"payload mass alone = {baseline:.2f} W"],
              fontsize=9, loc="lower right")
    ax.set_title("Predicting power WITHOUT payload mass\n"
                 "feature count in brackets; red line is the mass-only baseline",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUT, "plot_no_mass_ranking.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def fig_per_flight(tab, preds, tops):
    order = tab.sort_values("power").index
    x = np.arange(len(order))
    cmap = plt.get_cmap("tab10")
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(max(11, 1.0 * len(order)), 10),
                                  gridspec_kw={"height_ratios": [3, 2]}, sharex=True)
    ax.bar(x, tab.loc[order, "power"], 0.68, color="#e2e8f0",
           edgecolor="#94a3b8", label="ACTUAL", zorder=1)
    for j, g in enumerate(tops):
        ax.scatter(x, preds[g][order], s=62, color=cmap(j), zorder=3,
                   edgecolor="white", lw=1.0, label=g)
    ax.set_ylabel("Mean power in flight (W)", fontsize=11)
    ax.legend(fontsize=8.5, ncol=2, loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Every flight, predicted while held out — payload mass withheld",
                 fontsize=12, fontweight="bold")
    for j, g in enumerate(tops):
        ax2.plot(x, (preds[g] - tab["power"].values)[order], "-o", ms=5,
                 color=cmap(j), lw=1.2)
    ax2.axhline(0, color="black", lw=1.0)
    ax2.set_ylabel("prediction − actual (W)", fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{tab.loc[i,'flight']}\n{tab.loc[i,'true_mass']:.2f} kg"
                         for i in order], fontsize=8.5)
    ax2.set_title("Error per flight (labels show the withheld payload)", fontsize=10)
    plt.tight_layout()
    p = os.path.join(OUT, "plot_no_mass_per_flight.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def fig_time(fdir, tab, preds, tops, show=("F08", "F09", "F13")):
    show = [f for f in show if f in set(tab["flight"])]
    if not show:
        return None
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(len(show), 1, figsize=(14, 4.3 * len(show)))
    if len(show) == 1:
        axes = [axes]
    for ax, fid in zip(axes, show):
        d = pd.read_csv(os.path.join(fdir, fid, "flight_resampled.csv"))
        thr = 0.5 * (np.percentile(d["power"], 5) + np.percentile(d["power"], 95))
        g = d[d["power"] > thr].sort_values("timestamp")
        t = (g["timestamp"] - g["timestamp"].iloc[0]).values
        y = g["power"].values
        i = tab.index[tab.flight == fid][0]
        ax.plot(t, y, color="#334155", lw=1.4, label="ACTUAL power", zorder=4)
        ax.axhline(y.mean(), color="#334155", ls=":", lw=1.4,
                   label=f"actual average = {y.mean():.0f} W", zorder=2)
        for j, gname in enumerate(tops):
            v = preds[gname][i]
            ax.axhline(v, color=cmap(j), ls="--", lw=2.0, zorder=3,
                       label=f"{gname}  →  {v:.0f} W  ({v - y.mean():+.0f} W)")
        ax.set_title(f"{fid} — held out  (true payload "
                     f"{tab.loc[i, 'true_mass']:.2f} kg, withheld from the model)",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Power (W)", fontsize=10)
        ax.set_xlim(t[0], t[-1]); ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left", ncol=2, framealpha=0.94)
    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle("Predicting power without knowing the payload",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, "plot_no_mass_time.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def fig_time_full(fdir, tab, preds, tops, show=("F08", "F09", "F13")):
    """
    Same as fig_time, but drawn over the ENTIRE recording rather than the
    in-flight portion only.

    The shaded bands are the samples the filter discards: ground idle and the
    sub-threshold part of the takeoff and landing transitions. Models are still
    fitted and evaluated on in-flight samples only — the shading simply shows
    what was excluded and why, so the reader can judge the choice rather than
    take it on trust.
    """
    show = [f for f in show if f in set(tab["flight"])]
    if not show:
        return None
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(len(show), 1, figsize=(14, 4.3 * len(show)))
    if len(show) == 1:
        axes = [axes]

    for ax, fid in zip(axes, show):
        d = pd.read_csv(os.path.join(fdir, fid, "flight_resampled.csv")).sort_values("timestamp")
        t = (d["timestamp"] - d["timestamp"].iloc[0]).values
        p = d["power"].values
        thr = 0.5 * (np.percentile(p, 5) + np.percentile(p, 95))
        keep = p > thr
        i = tab.index[tab.flight == fid][0]

        # shade the discarded blocks
        ax.fill_between(t, p.min() - 40, p.max() + 40, where=~keep,
                        color="#94a3b8", alpha=0.22, step="mid", zorder=0,
                        label="excluded (ground / below threshold)")
        ax.axhline(thr, color="#64748b", ls="-.", lw=1.1, zorder=1,
                   label=f"in-flight threshold = {thr:.0f} W")

        ax.plot(t, p, color="#334155", lw=1.2, label="ACTUAL power", zorder=4)
        ax.axhline(p[keep].mean(), color="#334155", ls=":", lw=1.4, zorder=2,
                   label=f"in-flight average = {p[keep].mean():.0f} W")
        for j, gname in enumerate(tops):
            v = preds[gname][i]
            ax.axhline(v, color=cmap(j), ls="--", lw=2.0, zorder=3,
                       label=f"{gname}  →  {v:.0f} W")

        ax.set_title(f"{fid} — full recording  (true payload "
                     f"{tab.loc[i, 'true_mass']:.2f} kg, withheld;  "
                     f"{100 * (~keep).mean():.0f}% of samples excluded)",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Power (W)", fontsize=10)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(0, p.max() * 1.08)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="center left", ncol=2, framealpha=0.94)
    axes[-1].set_xlabel("Time (s) — whole recording", fontsize=10)

    fig.suptitle("Full recordings: what the in-flight filter removes, and where "
                 "the predictions sit", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p_out = os.path.join(OUT, "plot_no_mass_time_full.png")
    plt.savefig(p_out, dpi=150, bbox_inches="tight"); plt.close()
    return p_out


def main():
    fdir = find_flights_dir()
    print(f"\n[NO-MASS STUDY]  flights from: {fdir}")
    tab = build_table(fdir)
    print(f"  {len(tab)} flights | trajectories: "
          f"{dict(tab.trajectory.value_counts())}")
    print(f"  power range {tab.power.min():.0f}–{tab.power.max():.0f} W\n")

    # baseline for comparison: the withheld variable, used alone
    base_pred = loo(tab.assign(m=tab["true_mass"]), ["m"]) if False else None
    tmp = tab.copy(); tmp["m"] = tmp["true_mass"]
    bp = loo(tmp, ["m"])
    base_mae = mean_absolute_error(tab["power"], bp)
    base_r2 = r2_score(tab["power"], bp)
    print(f"  BASELINE (payload mass alone): MAE {base_mae:.2f} W, R² {base_r2:.4f}\n")

    res = search(tab)
    print(f"  {len(res)} combinations scored (OLS, <= 12 features) — best 8:\n")
    print(res.head(8).to_string(index=False))

    print("\n  FULL / LARGE feature sets, ridge regression:")
    fs = full_sets(tab)
    fs.to_csv(os.path.join(OUT, "no_mass_full_sets.csv"), index=False)

    tops = res.head(3)["groups"].tolist()
    if "motors" not in tops:
        tops.append("motors")
    preds = {g: loo(tab, cols_for(g.split(" + "))) for g in tops}
    # the full set cannot be fitted by OLS here, so it comes in via ridge
    FULL = "ALL features, ridge (23)"
    preds[FULL] = loo_ridge(tab, cols_for(list(GROUPS)))
    tops.append(FULL)

    out = tab[["flight", "true_mass", "trajectory", "power"]].copy()
    out["pred: payload mass (baseline)"] = bp.round(1)
    for g in tops:
        out[f"pred: {g}"] = preds[g].round(1)
    out.to_csv(os.path.join(OUT, "no_mass_predictions.csv"), index=False)
    res.to_csv(os.path.join(OUT, "no_mass_results.csv"), index=False)

    best = res.iloc[0]
    lm = LinearRegression().fit(tab[cols_for(best["groups"].split(" + "))], tab["power"])
    summary = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "question": "Can mean flight power be predicted WITHOUT payload mass?",
        "excluded": {"payload_mass": "the variable being withheld",
                     "position_payload": "leaks mass: category 'none' occurs only "
                                         "on the two zero-payload flights"},
        "newly_included": {"trajectory": "one-hot T1/T2/T3; does not leak load"},
        "evaluation": "leave-one-flight-out, in-flight samples only",
        "n_flights": int(len(tab)),
        "baseline_payload_mass_alone": {"mae_w": round(float(base_mae), 2),
                                        "r2": round(float(base_r2), 4)},
        "best_without_mass": {"groups": best["groups"],
                              "n_features": int(best["n_features"]),
                              "mae_w": float(best["mae_w"]), "r2": float(best["r2"]),
                              "coefficients": {c: round(float(v), 3) for c, v in
                                               zip(cols_for(best["groups"].split(" + ")),
                                                   lm.coef_)},
                              "intercept_w": round(float(lm.intercept_), 1)},
        "penalty_w": round(float(best["mae_w"] - base_mae), 2),
        "top10": res.head(10).to_dict("records"),
        "full_sets_ridge": fs.to_dict("records"),
        "note_on_full_sets": "23 predictors vs 13 training flights: OLS is undefined, so these use RidgeCV with the penalty chosen inside each fold.",
    }
    with open(os.path.join(OUT, "no_mass_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    p1 = fig_ranking(res, base_mae, fs)
    p2 = fig_per_flight(tab, preds, tops)
    p3 = fig_time(fdir, tab, preds, tops)
    p4 = fig_time_full(fdir, tab, preds, tops)

    print(f"\n  BEST without mass : {best['groups']}  "
          f"({best['n_features']} feats, MAE {best['mae_w']} W, R² {best['r2']})")
    print(f"  Baseline with mass: {base_mae:.2f} W")
    print(f"  Penalty for not knowing the payload: {best['mae_w'] - base_mae:+.2f} W")
    print(f"\n  Saved → {OUT}/")
    for f in sorted(os.listdir(OUT)):
        if f.endswith((".csv", ".png", ".json")):
            print(f"           {f}")
    print()


if __name__ == "__main__":
    main()
