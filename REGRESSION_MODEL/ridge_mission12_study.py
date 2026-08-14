"""
RIDGE MISSION-12 STUDY — with mass vs without mass, exhaustive feature search,
on the +/-12 s mission window.

STANDALONE. Reads only flights/F*/flight_resampled.csv, writes only into this
folder. Sibling study to NO_MASS_STUDY/mission12_study.py — same window, same
feature groups, same leave-one-flight-out protocol — but:
  * model is Ridge (linear, L2-regularised) instead of XGBoost
  * feature search is EXHAUSTIVE over all non-empty group subsets instead of
    greedy forward selection, because Ridge is cheap enough to afford it
    (see FEATURE_STUDY/feature_study.py, which does the same at flight level)
  * both mass settings run in the same invocation and are compared directly

THE WINDOW
----------
+/-12 s around the cruise phase (defined by a per-flight power threshold —
midpoint of the 5th/95th percentile). 12 s is the smallest fixed margin that
captures motor ARMING on all 14 flights; see
NO_MASS_STUDY/mission12_study.py and NO_MASS_STUDY/MISSION12/README-equivalent
notes for the derivation. Not a tuned constant.

MODEL
-----
RidgeCV per leave-one-flight-out fold: fit on 13 flights (z-scored on train
only), predict the held-out 14th. RidgeCV picks its regularisation strength
alpha via the efficient closed-form leave-one-out trick over ALL training
rows for that fold (sklearn's `cv=None` mode) — this is a standard
simplification for choosing a *regularisation strength*, not the reported
score: the reported score always comes from the fully held-out flight, never
seen during fitting or alpha selection.

FEATURE SEARCH
--------------
Groups without mass: motors, imbalance, orientation, imu, speed, velocity,
altitude (7 groups -> 127 non-empty subsets).
With mass: + mass (8 groups -> 255 non-empty subsets).
Every subset is scored by leave-one-flight-out MAE. All of them are kept and
ranked, not just the greedy path, so "best combination" claims can be backed
by the full table (search_ridge_<mass>.csv) rather than a single search
trace.

OUTPUT (in RIDGE_MISSION12/)
-----------------------------
    search_ridge_<mass>.csv           every subset, ranked by MAE
    summary_ridge_<mass>.json         winner, top-4, metrics
    plot_search_ridge_<mass>.png      MAE vs number of features, all subsets
    plot_top4_ridge_<mass>.png        best 4 combinations, MAE and R2
    plot_group_importance_<mass>.png  MAE with vs without each group present
    plot_time_ridge_compare.png       best-with-mass vs best-without-mass,
                                       side by side, over time, held-out flights
    plot_r2_ridge_compare.png         per-flight R2 / MAE, both winners
    comparison_ridge_summary.json     headline with-mass vs without-mass numbers

Run:
    python ridge_mission12_study.py
    python ridge_mission12_study.py --rate 20      # 20 Hz raw grid instead of 1 Hz
    python ridge_mission12_study.py --no-motors    # drop the motors group entirely,
                                                    # writes to RIDGE_MISSION12_NO_MOTORS/
                                                    # instead of RIDGE_MISSION12/
    python ridge_mission12_study.py --show F02,F06,F12   # different held-out
                                                    # flights in the time/r2 plots
    python ridge_mission12_study.py --replot --show F02,F06,F12
                                                    # skip the search entirely and just
                                                    # redraw plot_time/plot_r2 for the
                                                    # winning combos already on disk
                                                    # (comparison_ridge_summary.json
                                                    # must already exist) — seconds,
                                                    # not the full ~45 s search
"""

import os
import sys
import glob
import json
import itertools
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error

MARGIN_S = 12
TOP_K = 4
ALPHAS = np.logspace(-2, 4, 25)  # RidgeCV regularisation-strength grid

RATE = 20 if "--rate" in sys.argv and sys.argv[sys.argv.index("--rate") + 1] == "20" else 1
EXCLUDE_MOTORS = "--no-motors" in sys.argv
REPLOT = "--replot" in sys.argv

# held-out flights shown in the time / comparison plots
SHOW = (sys.argv[sys.argv.index("--show") + 1].split(",") if "--show" in sys.argv
       else ["F08", "F09", "F13"])

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "RIDGE_MISSION12_NO_MOTORS" if EXCLUDE_MOTORS else "RIDGE_MISSION12")
os.makedirs(OUT, exist_ok=True)
TITLE_SUFFIX = " — motors excluded" if EXCLUDE_MOTORS else ""

MOTORS = ["motor_1_front_right", "motor_2_rear_left",
          "motor_3_front_left",  "motor_4_rear_right"]

# "mass" is always defined here (unlike mission12_study.py, which only adds it
# to GROUPS under --with-mass) because this script runs BOTH settings in one
# pass and needs the column loaded either way.
GROUPS = {
    "mass":        ["payload_mass"],
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
    # trajectory deliberately excluded — see NO_MASS_STUDY/mission12_study.py
    # for the reasoning (trajectory_3 occurs on exactly one flight).
}
# --no-motors drops the motors group entirely (e.g. to see what the rest of the
# sensor suite can do on its own, or to model a setup where motor commands are
# not available at inference time). Everything downstream — column set, search
# space, plots, output folder — follows from this dict, so nothing else needs
# a separate code path.
if EXCLUDE_MOTORS:
    del GROUPS["motors"]

NOMASS_GROUPS = [g for g in GROUPS if g != "mass"]
WITHMASS_GROUPS = list(GROUPS)
ALL_COLS = sorted({c for g in GROUPS.values() for c in g})


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
    """Mission window +/-MARGIN_S around cruise, at RATE Hz, with a phase label."""
    out = []
    for f in sorted(glob.glob(os.path.join(fdir, "F*", "flight_resampled.csv"))):
        d = pd.read_csv(f).sort_values("timestamp").copy()
        thr = 0.5 * (np.percentile(d["power"], 5) + np.percentile(d["power"], 95))
        d["t"] = d["timestamp"] - d["timestamp"].iloc[0]

        if RATE == 1:
            d["sec"] = np.floor(d["t"]).astype(int)
            a = d.groupby("sec", as_index=False)[ALL_COLS + ["power", "t"]].mean()
            step = 1
        else:
            a = d[ALL_COLS + ["power", "t"]].reset_index(drop=True)
            step = int(round(1 / np.median(np.diff(d["t"].values))))

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


def ridge_evaluate(data, cols):
    """Leave-one-flight-out predictions with RidgeCV (train-only z-score)."""
    pred = np.full(len(data), np.nan)
    alphas_used = []
    for h in sorted(data["flight_id"].unique()):
        te_m = (data.flight_id == h).values
        tr, te = data[~te_m], data[te_m]
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
        Xtr = ((tr[cols] - mu) / sd).values
        Xte = ((te[cols] - mu) / sd).values
        m = RidgeCV(alphas=ALPHAS)
        m.fit(Xtr, tr["power"].values)
        pred[te_m] = m.predict(Xte)
        alphas_used.append(float(m.alpha_))
    return pred, alphas_used


def score(data, p):
    y = data["power"].values
    ck = (data.phase == "cruise").values
    return {"r2": round(float(r2_score(y, p)), 4),
            "mae_w": round(float(mean_absolute_error(y, p)), 2),
            "cruise_mae_w": round(float(mean_absolute_error(y[ck], p[ck])), 2),
            "other_mae_w": round(float(mean_absolute_error(y[~ck], p[~ck])), 2)
                           if (~ck).any() else None}


def exhaustive_search(data, group_keys, tag):
    """Score every non-empty subset of group_keys. Cheap enough with Ridge."""
    rows = []
    n = len(group_keys)
    total = 2 ** n - 1
    done = 0
    for size in range(1, n + 1):
        for combo in itertools.combinations(group_keys, size):
            combo = list(combo)
            cols = cols_for(combo)
            pred, alphas = ridge_evaluate(data, cols)
            s = score(data, pred)
            rows.append({"groups": combo, "n_groups": len(combo),
                        "n_features": len(cols),
                        "mean_alpha": round(float(np.mean(alphas)), 3), **s})
            done += 1
            if done % 25 == 0 or done == total:
                print(f"    [{tag}] {done}/{total} combinations scored", flush=True)
    df = pd.DataFrame(rows).sort_values("mae_w").reset_index(drop=True)
    return df


def fig_search_all(df, tag, mass_label):
    """MAE vs number of features for every subset, Pareto frontier highlighted."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df.n_features, df.mae_w, s=14, alpha=0.35, color="#94a3b8",
              label="all subsets")
    best_per_size = df.loc[df.groupby("n_features")["mae_w"].idxmin()].sort_values("n_features")
    ax.plot(best_per_size.n_features, best_per_size.mae_w, "-o", color="#2563eb",
           lw=2, label="best at each size")
    win = df.iloc[0]
    ax.scatter([win.n_features], [win.mae_w], s=140, color="#16a34a", zorder=5,
              marker="*", label=f"overall best: {' + '.join(win.groups)}")
    ax.set_xlabel("number of features", fontsize=10)
    ax.set_ylabel("MAE (W) — leave-one-flight-out", fontsize=10)
    ax.set_title(f"Ridge — exhaustive feature-group search — {mass_label}{TITLE_SUFFIX}\n"
                f"window +/-{MARGIN_S} s, {RATE} Hz, {2 ** df.n_groups.max() - 1} "
                f"non-empty subsets scored", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout()
    p = os.path.join(OUT, f"plot_search_ridge_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def fig_top4(df, tag, mass_label, k=TOP_K):
    t = df.iloc[:k].iloc[::-1]
    labels = [" + ".join(g) for g in t.groups]
    y = np.arange(len(t))
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.6 * len(t) + 3.6))
    for ax, key, nice in [(axes[0], "mae_w", "MAE (W) — lower is better"),
                          (axes[1], "r2", "R2 — higher is better")]:
        vals = t[key].values
        ax.barh(y, vals, color=["#16a34a" if i == len(t) - 1 else "#2563eb"
                                for i in range(len(t))])
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(nice, fontsize=10); ax.grid(axis="x", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(v, i, f"  {v:.3f}" if key == "r2" else f"  {v:.2f}",
                    va="center", fontsize=9)
    fig.suptitle(f"Ridge — best {k} feature combinations — {mass_label}{TITLE_SUFFIX}\n"
                f"window +/-{MARGIN_S} s, {RATE} Hz — green = best",
                fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, f"plot_top4_ridge_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def fig_group_importance(df, group_keys, tag, mass_label):
    """
    MAE distribution for subsets that include vs exclude each group.

    Conditioned on the single strongest group already being present (normally
    'motors'; whatever tops the size-1 subsets when it has been excluded via
    --no-motors). Without it, the model is not competitive at all (MAE far
    higher than with it — see the size-1 rows in search_ridge_<mass>.csv), and
    that gap is so large it swamps the x-axis and hides every other
    comparison. Restricting to subsets that already include the anchor group
    shows what actually matters among the models worth deploying.
    """
    singles = df[df.n_groups == 1].sort_values("mae_w")
    anchor = singles.iloc[0].groups[0]
    base = df[df.groups.apply(lambda gs: anchor in gs)]
    other_groups = [g for g in group_keys if g != anchor]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(other_groups) + 2.5))
    data_with, data_without = [], []
    for g in other_groups:
        has = base.groups.apply(lambda gs: g in gs)
        data_with.append(base.loc[has, "mae_w"].values)
        data_without.append(base.loc[~has, "mae_w"].values)
    y = np.arange(len(other_groups))
    w = 0.35
    ax.boxplot(data_with, positions=y - w / 2 - 0.02, widths=w, vert=False,
              patch_artist=True, boxprops=dict(facecolor="#2563eb", alpha=0.6),
              medianprops=dict(color="black"))
    ax.boxplot(data_without, positions=y + w / 2 + 0.02, widths=w, vert=False,
              patch_artist=True, boxprops=dict(facecolor="#94a3b8", alpha=0.6),
              medianprops=dict(color="black"))
    ax.set_yticks(y); ax.set_yticklabels(other_groups, fontsize=9)
    ax.set_xlabel(f"MAE (W), subsets that already include '{anchor}', "
                 "with / without the group", fontsize=9.5)
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#2563eb", alpha=0.6),
              plt.Rectangle((0, 0), 1, 1, fc="#94a3b8", alpha=0.6)]
    ax.legend(handles, ["group present", "group absent"], fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(f"Ridge — does each group help, given {anchor}? — {mass_label}"
                f"{TITLE_SUFFIX}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, f"plot_group_importance_ridge_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def final_coefficients(data, cols):
    """
    Descriptive fit on ALL 14 flights with the winning feature set.

    Not used for any reported metric — those all come from leave-one-flight-out
    predictions on held-out flights. This is purely to show, in one linear
    equation, which direction and how strongly each standardized feature pushes
    predicted power. This readable-coefficient view is the main practical
    advantage Ridge has over XGBoost for this study.
    """
    mu, sd = data[cols].mean(), data[cols].std().replace(0, 1)
    X = ((data[cols] - mu) / sd).values
    m = RidgeCV(alphas=ALPHAS)
    m.fit(X, data["power"].values)
    coef = pd.DataFrame({"feature": cols, "standardized_coef": m.coef_}
                        ).sort_values("standardized_coef", key=np.abs, ascending=False)
    coef["alpha"] = float(m.alpha_)
    coef["intercept_w"] = float(m.intercept_)
    return coef


def run_setting(data, group_keys, tag, mass_label):
    print(f"\n[{mass_label}] searching {2 ** len(group_keys) - 1} non-empty "
          f"subsets of {group_keys}", flush=True)
    df = exhaustive_search(data, group_keys, tag)
    df.to_csv(os.path.join(OUT, f"search_ridge_{tag}.csv"), index=False)

    win = df.iloc[0]
    print(f"  BEST: {' + '.join(win.groups)}  ({win.n_features} features)")
    print(f"        R2 {win.r2:+.4f}   MAE {win.mae_w:.2f} W   "
          f"cruise {win.cruise_mae_w:.2f} W   alpha~{win.mean_alpha:.3g}")
    print(f"  top {TOP_K}:")
    for i, r in df.iloc[:TOP_K].iterrows():
        print(f"    {' + '.join(r.groups):45s} MAE {r.mae_w:6.2f} W   R2 {r.r2:+.4f}")

    pred, alphas = ridge_evaluate(data, cols_for(win.groups))
    p1 = fig_search_all(df, tag, mass_label)
    p2 = fig_top4(df, tag, mass_label)
    p3 = fig_group_importance(df, group_keys, tag, mass_label)
    print(f"  Saved -> {os.path.basename(p1)}, {os.path.basename(p2)}, "
          f"{os.path.basename(p3)}")

    coef = final_coefficients(data, cols_for(win.groups))
    coef.to_csv(os.path.join(OUT, f"coefficients_ridge_{tag}.csv"), index=False)
    print(f"  Standardized coefficients (all-flights fit, intercept "
          f"{coef.intercept_w.iloc[0]:.1f} W):")
    for _, r in coef.iterrows():
        print(f"    {r.feature:45s} {r.standardized_coef:+7.2f} W / sd")

    summary = {"rate_hz": RATE, "margin_s": MARGIN_S, "mass": tag,
              "motors_excluded": EXCLUDE_MOTORS,
              "model": "RidgeCV", "alphas_grid": list(ALPHAS),
              "n_rows": int(len(data)), "n_subsets_scored": int(len(df)),
              "winner": {"groups": win.groups, "n_features": int(win.n_features),
                         "mean_alpha": win.mean_alpha, "r2": win.r2,
                         "mae_w": win.mae_w, "cruise_mae_w": win.cruise_mae_w,
                         "other_mae_w": win.other_mae_w},
              "top_combinations": df.iloc[:TOP_K][
                  ["groups", "n_features", "mae_w", "r2", "cruise_mae_w"]
              ].to_dict("records")}
    with open(os.path.join(OUT, f"summary_ridge_{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    return df, win, pred


def fig_time_compare(data, pred_with, pred_nomass, win_with, win_nomass):
    """Best-with-mass vs best-without-mass, side by side, over time."""
    show = [f for f in SHOW if f in set(data.flight_id)]
    fig, axes = plt.subplots(len(show), 2, figsize=(17, 4.0 * len(show)), squeeze=False)
    rw = r2_score(data.power, pred_with)
    rn = r2_score(data.power, pred_nomass)
    for r, fid in enumerate(show):
        k = (data.flight_id == fid).values
        d = data[k].sort_values("t")
        t, y = d["t"].values, d["power"].values
        ck = (d.phase == "cruise").values
        for c, (p_all, name, groups, tot) in enumerate([
                (pred_with, f"WITH MASS — R2 {rw:.3f} overall", win_with.groups, rw),
                (pred_nomass, f"WITHOUT MASS — R2 {rn:.3f} overall", win_nomass.groups, rn)]):
            p = p_all[k]
            ax = axes[r][c]
            ax.fill_between(t, 0, max(y) * 1.1, where=~ck, color="#94a3b8", alpha=0.18,
                            step="mid", zorder=0, label="arming / takeoff / landing")
            ax.plot(t, y, color="#334155", lw=1.4, label="ACTUAL", zorder=3)
            ax.plot(t, p, color="#dc2626" if c else "#2563eb", lw=1.1, alpha=0.9, zorder=2,
                    label=f"predicted  (R2 {r2_score(y, p):+.3f}, "
                          f"cruise {mean_absolute_error(y[ck], p[ck]):.0f} W, "
                          f"all {mean_absolute_error(y, p):.0f} W)")
            ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, max(y) * 1.1)
            ax.grid(alpha=0.3)
            ax.set_title(f"{fid} — {name}\n{' + '.join(groups)}", fontsize=9.5,
                        fontweight="bold")
            if c == 0:
                ax.set_ylabel("Power (W)", fontsize=10)
            ax.legend(fontsize=7.5, loc="center left", framealpha=0.94)
    for c in range(2):
        axes[-1][c].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle(f"Ridge, leave-one-flight-out, window +/-{MARGIN_S} s, {RATE} Hz{TITLE_SUFFIX}\n"
                f"best model with mass vs best model without mass",
                fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, "plot_time_ridge_compare.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def fig_r2_compare(data, pred_with, pred_nomass):
    """Per-flight R2 and MAE, both winners overlaid."""
    flights = sorted(data.flight_id.unique())
    preds = {"with mass": pred_with, "without mass": pred_nomass}
    colors = {"with mass": "#2563eb", "without mass": "#dc2626"}
    x = np.arange(len(flights))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.4))
    for j, (label, p_all) in enumerate(preds.items()):
        within = [r2_score(data.power.values[(data.flight_id == f).values],
                           p_all[(data.flight_id == f).values]) for f in flights]
        mae = [mean_absolute_error(data.power.values[(data.flight_id == f).values],
                                   p_all[(data.flight_id == f).values]) for f in flights]
        pooled = r2_score(data.power.values, p_all)
        off = (j - 0.5) * w
        axes[0].bar(x + off, within, w, color=colors[label],
                    label=f"{label}  (pooled R2 {pooled:.3f})")
        axes[1].bar(x + off, mae, w, color=colors[label],
                    label=f"{label}  (mean {np.mean(mae):.1f} W)")

    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xticks(x); axes[0].set_xticklabels(flights, rotation=45, fontsize=8.5)
    axes[0].set_ylabel("R2 within each flight", fontsize=10)
    axes[0].set_title("R2 per held-out flight", fontsize=11, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3); axes[0].legend(fontsize=9)

    axes[1].set_xticks(x); axes[1].set_xticklabels(flights, rotation=45, fontsize=8.5)
    axes[1].set_ylabel("MAE (W)", fontsize=10)
    axes[1].set_title("Error per held-out flight", fontsize=11, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3); axes[1].legend(fontsize=9)

    fig.suptitle(f"Ridge — best with-mass model vs best without-mass model, "
                f"per flight{TITLE_SUFFIX}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, "plot_r2_ridge_compare.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def replot(data):
    """
    Skip the full search entirely. Re-read the winning combinations already on
    disk (from a previous full run) and just redraw the two comparison plots
    with a new SHOW list — the leave-one-flight-out predictions already cover
    every flight, so nothing about the model changes, only which held-out
    flights are drawn.
    """
    path = os.path.join(OUT, "comparison_ridge_summary.json")
    if not os.path.exists(path):
        sys.exit(f"--replot needs an existing {path} — run without --replot first.")
    with open(path) as fh:
        prev = json.load(fh)
    groups_mass = prev["best_using_mass"]["groups"]
    groups_no = prev["without_mass"]["groups"]
    print(f"[RIDGE MISSION-12 — REPLOT]  reusing winners from {os.path.basename(path)}")
    print(f"  using mass:   {' + '.join(groups_mass)}")
    print(f"  without mass: {' + '.join(groups_no)}")
    print(f"  new SHOW: {SHOW}")

    pred_mass = ridge_evaluate(data, cols_for(groups_mass))[0]
    pred_no = ridge_evaluate(data, cols_for(groups_no))[0]
    win_mass = SimpleNamespace(groups=groups_mass)
    win_no = SimpleNamespace(groups=groups_no)

    p1 = fig_time_compare(data, pred_mass, pred_no, win_mass, win_no)
    p2 = fig_r2_compare(data, pred_mass, pred_no)
    print(f"  Saved -> {os.path.basename(p1)}")
    print(f"  Saved -> {os.path.basename(p2)}\n")


def main():
    fdir = find_flights_dir()
    data = load(fdir)
    print(f"[RIDGE MISSION-12]  {RATE} Hz  |  window +/-{MARGIN_S} s"
          f"{'  |  MOTORS EXCLUDED' if EXCLUDE_MOTORS else ''}")
    print(f"  flights from: {fdir}")
    print(f"  writing to:   {OUT}")
    print(f"  {len(data)} rows, {data.flight_id.nunique()} flights, "
          f"{100 * (data.phase != 'cruise').mean():.1f}% not cruise")

    if REPLOT:
        replot(data)
        return

    df_with, win_with, pred_with = run_setting(data, WITHMASS_GROUPS, "withmass",
                                                "mass included")
    df_no, win_no, pred_no = run_setting(data, NOMASS_GROUPS, "nomass",
                                          "mass withheld")

    # The unconstrained global best of WITHMASS_GROUPS is not guaranteed to
    # actually use mass (it doesn't, once motors are excluded — see README).
    # For a head-to-head "does mass help" comparison, use the best subset that
    # DOES include mass, not whichever subset happened to win overall.
    mass_rows = df_with[df_with.groups.apply(lambda g: "mass" in g)]
    win_mass = mass_rows.sort_values("mae_w").iloc[0]
    uses_mass = "mass" in win_with.groups

    print("\n[COMPARISON]  (best model that USES mass vs best model without it)")
    print(f"  using mass:   {' + '.join(win_mass.groups):40s} "
          f"MAE {win_mass.mae_w:6.2f} W   R2 {win_mass.r2:+.4f}")
    print(f"  without mass: {' + '.join(win_no.groups):40s} "
          f"MAE {win_no.mae_w:6.2f} W   R2 {win_no.r2:+.4f}")
    print(f"  mass costs {win_no.mae_w - win_mass.mae_w:+.2f} W MAE if withheld")
    if not uses_mass:
        print(f"  NOTE: the unconstrained search over {{{', '.join(WITHMASS_GROUPS)}}} "
              f"picked '{' + '.join(win_with.groups)}' overall (MAE {win_with.mae_w:.2f} W), "
              f"which does not use mass at all — mass actively hurt every "
              f"combination it was added to here. See search_ridge_withmass.csv.")

    pred_mass = pred_with if uses_mass else ridge_evaluate(data, cols_for(win_mass.groups))[0]

    p1 = fig_time_compare(data, pred_mass, pred_no, win_mass, win_no)
    p2 = fig_r2_compare(data, pred_mass, pred_no)
    print(f"\n  Saved -> {os.path.basename(p1)}")
    print(f"  Saved -> {os.path.basename(p2)}")

    with open(os.path.join(OUT, "comparison_ridge_summary.json"), "w") as fh:
        json.dump({"rate_hz": RATE, "margin_s": MARGIN_S, "model": "RidgeCV",
                   "motors_excluded": EXCLUDE_MOTORS,
                   "global_best_offered_mass": {
                       "groups": win_with.groups, "uses_mass": uses_mass,
                       "mae_w": win_with.mae_w, "r2": win_with.r2},
                   "best_using_mass": {"groups": win_mass.groups,
                                 "n_features": int(win_mass.n_features),
                                 "r2": win_mass.r2, "mae_w": win_mass.mae_w,
                                 "cruise_mae_w": win_mass.cruise_mae_w},
                   "without_mass": {"groups": win_no.groups,
                                     "n_features": int(win_no.n_features),
                                     "r2": win_no.r2, "mae_w": win_no.mae_w,
                                     "cruise_mae_w": win_no.cruise_mae_w},
                   "mass_gain_mae_w": round(float(win_no.mae_w - win_mass.mae_w), 2)},
                  fh, indent=2)
    print(f"  Saved -> comparison_ridge_summary.json\n")


if __name__ == "__main__":
    main()
