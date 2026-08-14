"""
ENSEMBLE MISSION-12 STUDY — XGBoost + Ridge blended, with mass vs without mass,
on the +/-12 s mission window.

STANDALONE. Reads only flights/F*/flight_resampled.csv, writes only into
Ensemble/. Third sibling to NO_MASS_STUDY/mission12_study.py (XGBoost) and
ridge_mission12_study.py (Ridge) in this folder — same window, same feature
groups, same leave-one-flight-out protocol, but the prediction at every fold
is a blend of both base models instead of either one alone.

MODEL
-----
Per leave-one-flight-out fold (fit on 13 flights, predict the held-out 14th):
  * XGBRegressor  — same hyperparameters as NO_MASS_STUDY/mission12_study.py
                    (reg:absoluteerror, 150 trees, depth 5, lr 0.1, subsample 0.9)
                    fit on raw (unscaled) features — trees do not need scaling.
  * RidgeCV       — same as ridge_mission12_study.py, fit on train-only
                    z-scored features, alpha chosen by RidgeCV's internal
                    leave-one-out trick.
  * blend = w * xgb_pred + (1-w) * ridge_pred

w is chosen per feature combination by grid search over the fold's own
out-of-fold predictions (21 points, 0.00-1.00), minimising MAE. This is
standard "blend the out-of-fold predictions" practice: it never lets a
held-out flight influence how its own base-model predictions were produced,
it only decides how two already-honest prediction streams are combined
afterward.

FEATURE SEARCH
--------------
GREEDY forward selection over feature groups, same algorithm as
NO_MASS_STUDY/mission12_study.py — NOT exhaustive like ridge_mission12_study.py.
XGBoost is the bottleneck: one leave-one-flight-out evaluation of a single
combination takes ~4 s (benchmarked), so the 255/127 exhaustive subsets Ridge
can afford would take 15-25 minutes here. Greedy still records every candidate
tried at every step (not only the kept path), so a top-4 ranking is still
possible from what was actually evaluated.

OUTPUT (in Ensemble/)
----------------------
    search_ensemble_<mass>.csv          greedy path, every step scored
    summary_ensemble_<mass>.json        winner, top-4, blend weight, metrics
    plot_search_ensemble_<mass>.png     MAE/R2 as groups are added
    plot_top4_ensemble_<mass>.png       best 4 combinations tried during search
    plot_model_breakdown_<mass>.png     XGBoost-alone vs Ridge-alone vs Ensemble,
                                         for the winning combination
    plot_time_ensemble_compare.png      best-using-mass vs best-without-mass,
                                         side by side, over time
    plot_r2_ensemble_compare.png        per-flight R2/MAE, both winners
    coefficients_ridge_component_<mass>.csv   Ridge half of the ensemble, for
                                         interpretability (as in RIDGE_MISSION12/)
    comparison_ensemble_summary.json    headline with-mass vs without-mass numbers

Run:
    python ensemble_mission12_study.py
    python ensemble_mission12_study.py --rate 20
    python ensemble_mission12_study.py --no-motors   # drop motors entirely,
                                                      # writes to Ensemble_NO_MOTORS/
                                                      # instead of Ensemble/
    python ensemble_mission12_study.py --show F02,F06,F12   # different held-out
                                                      # flights in the time/r2 plots
    python ensemble_mission12_study.py --replot --show F02,F06,F12
                                                      # skip the greedy search entirely
                                                      # and just redraw plot_time/plot_r2
                                                      # for the winning combos already on
                                                      # disk (comparison_ensemble_summary.json
                                                      # must already exist) — seconds, not
                                                      # the full multi-minute search
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

from xgboost import XGBRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error

MARGIN_S = 12
TOP_K = 4
ALPHAS = np.logspace(-2, 4, 25)          # RidgeCV regularisation-strength grid
BLEND_GRID = np.linspace(0, 1, 21)       # weight on XGBoost, 0.00 .. 1.00
OBJ = "reg:absoluteerror"

RATE = 20 if "--rate" in sys.argv and sys.argv[sys.argv.index("--rate") + 1] == "20" else 1
EXCLUDE_MOTORS = "--no-motors" in sys.argv
REPLOT = "--replot" in sys.argv

SHOW = (sys.argv[sys.argv.index("--show") + 1].split(",") if "--show" in sys.argv
       else ["F08", "F09", "F13"])

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "Ensemble_NO_MOTORS" if EXCLUDE_MOTORS else "Ensemble")
os.makedirs(OUT, exist_ok=True)
TITLE_SUFFIX = " — motors excluded" if EXCLUDE_MOTORS else ""

MOTORS = ["motor_1_front_right", "motor_2_rear_left",
          "motor_3_front_left",  "motor_4_rear_right"]

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
}
# --no-motors drops the motors group entirely, same convention as
# ridge_mission12_study.py — e.g. to see what the rest of the sensor suite can
# do on its own, or to model a setup where motor commands are not available
# at inference time.
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


def base_predictions(data, cols):
    """
    Leave-one-flight-out out-of-fold predictions from BOTH base models.

    XGBoost fits on raw features (trees do not need scaling). Ridge fits on
    train-only z-scored features, same discipline as ridge_mission12_study.py.
    Returns two full-length arrays, aligned to `data`'s row order.
    """
    p_xgb = np.full(len(data), np.nan)
    p_ridge = np.full(len(data), np.nan)
    for h in sorted(data["flight_id"].unique()):
        te_m = (data.flight_id == h).values
        tr, te = data[~te_m], data[te_m]

        gxb = XGBRegressor(objective=OBJ, n_estimators=150, max_depth=5,
                           learning_rate=0.1, subsample=0.9, n_jobs=-1,
                           random_state=42)
        gxb.fit(tr[cols], tr["power"])
        p_xgb[te_m] = gxb.predict(te[cols])

        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
        rdg = RidgeCV(alphas=ALPHAS)
        rdg.fit(((tr[cols] - mu) / sd).values, tr["power"].values)
        p_ridge[te_m] = rdg.predict(((te[cols] - mu) / sd).values)
    return p_xgb, p_ridge


def best_blend(y, p_xgb, p_ridge):
    """Grid search the blend weight on already-out-of-fold predictions."""
    maes = [mean_absolute_error(y, w * p_xgb + (1 - w) * p_ridge) for w in BLEND_GRID]
    i = int(np.argmin(maes))
    w = float(BLEND_GRID[i])
    return w, w * p_xgb + (1 - w) * p_ridge, maes[i]


def score(data, p):
    y = data["power"].values
    ck = (data.phase == "cruise").values
    return {"r2": round(float(r2_score(y, p)), 4),
            "mae_w": round(float(mean_absolute_error(y, p)), 2),
            "cruise_mae_w": round(float(mean_absolute_error(y[ck], p[ck])), 2),
            "other_mae_w": round(float(mean_absolute_error(y[~ck], p[~ck])), 2)
                           if (~ck).any() else None}


def evaluate_combo(data, groups):
    """Full ensemble evaluation of one feature-group combination."""
    cols = cols_for(groups)
    p_xgb, p_ridge = base_predictions(data, cols)
    w, blend, _ = best_blend(data["power"].values, p_xgb, p_ridge)
    s = score(data, blend)
    return {"groups": groups, "n_features": len(cols), "blend_w_xgb": round(w, 2),
           **s}, p_xgb, p_ridge, blend


def greedy(data, group_keys, tag):
    """
    Forward selection: add the group that reduces ensemble MAE most, until
    none does. Same algorithm as NO_MASS_STUDY/mission12_study.py's greedy(),
    scored on the blended prediction instead of XGBoost alone.
    """
    remaining, chosen, path, best_mae = list(group_keys), [], [], np.inf
    seen = {}
    step = 0
    while remaining:
        step += 1
        trials = []
        for g in remaining:
            combo = chosen + [g]
            s, *_ = evaluate_combo(data, combo)
            seen[frozenset(combo)] = s
            trials.append((s["mae_w"], g, s))
            print(f"    [{tag}] step {step}: try +{g:12s} -> "
                  f"MAE {s['mae_w']:6.2f} W  (w_xgb={s['blend_w_xgb']:.2f})", flush=True)
        trials.sort()
        mae, g, s = trials[0]
        improved = mae < best_mae - 0.05
        path.append({"step": step, "added": g, "n_features": s["n_features"],
                     "set": " + ".join(chosen + [g]), "improved": bool(improved), **s})
        if not improved:
            print(f"    [{tag}] step {step}: best was +{g} at {mae:.2f} W "
                  f"— no improvement, stop\n", flush=True)
            break
        chosen.append(g); remaining.remove(g); best_mae = mae
        print(f"    [{tag}] step {step}: KEEP +{g}  (MAE {mae:.2f} W, "
              f"R2 {s['r2']:.3f})\n", flush=True)

    ranked = sorted(seen.values(), key=lambda d: d["mae_w"])
    return chosen, pd.DataFrame(path), ranked


def fig_search(path, chosen, tag, mass_label):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    kept, rej = path[path.improved], path[~path.improved]
    for ax, key, nice in [(axes[0], "mae_w", "MAE (W)"), (axes[1], "r2", "R2")]:
        ax.plot(kept["step"], kept[key], "-o", lw=2, color="#2563eb", label="kept")
        if len(rej):
            ax.plot(rej["step"], rej[key], "x", ms=10, color="#dc2626",
                    label="rejected (no gain)")
        for _, r in path.iterrows():
            ax.annotate(f"+{r['added']}", (r["step"], r[key]),
                        textcoords="offset points", xytext=(6, 6), fontsize=8.5)
        ax.set_xlabel("selection step", fontsize=10)
        ax.set_ylabel(f"{nice} — leave-one-flight-out", fontsize=10)
        ax.set_title(nice, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle(f"Ensemble (XGBoost + Ridge blend) — greedy search — "
                f"{mass_label}{TITLE_SUFFIX}\n"
                f"window +/-{MARGIN_S} s, {RATE} Hz — final set: {' + '.join(chosen)}",
                fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, f"plot_search_ensemble_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def fig_top4(ranked, tag, mass_label, k=TOP_K):
    t = ranked[:k][::-1]
    labels = [" + ".join(d["groups"]) for d in t]
    y = np.arange(len(t))
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.6 * len(t) + 3.6))
    for ax, key, nice in [(axes[0], "mae_w", "MAE (W) — lower is better"),
                          (axes[1], "r2", "R2 — higher is better")]:
        vals = [d[key] for d in t]
        ax.barh(y, vals, color=["#16a34a" if i == len(t) - 1 else "#2563eb"
                                for i in range(len(t))])
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(nice, fontsize=10); ax.grid(axis="x", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(v, i, f"  {v:.3f}" if key == "r2" else f"  {v:.2f}",
                    va="center", fontsize=9)
    fig.suptitle(f"Ensemble — best {k} combinations tried during the greedy search "
                f"— {mass_label}{TITLE_SUFFIX}\nwindow +/-{MARGIN_S} s, {RATE} Hz — "
                f"green = best", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, f"plot_top4_ensemble_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def fig_model_breakdown(data, p_xgb, p_ridge, p_ens, w, tag, mass_label):
    """XGBoost-alone vs Ridge-alone vs Ensemble, for the winning combination."""
    y = data["power"].values
    ck = (data.phase == "cruise").values
    rows = [("XGBoost alone", p_xgb), ("Ridge alone", p_ridge),
           (f"Ensemble (w={w:.2f})", p_ens)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    colors = ["#f97316", "#2563eb", "#16a34a"]
    for ax, key, nice in [(axes[0], "mae", "MAE (W) — lower is better"),
                          (axes[1], "r2", "R2 — higher is better")]:
        vals = []
        for name, p in rows:
            vals.append(mean_absolute_error(y, p) if key == "mae" else r2_score(y, p))
        ax.bar(range(3), vals, color=colors)
        ax.set_xticks(range(3)); ax.set_xticklabels([n for n, _ in rows], fontsize=9)
        ax.set_ylabel(nice, fontsize=10); ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v, f" {v:.2f}" if key == "mae" else f" {v:.3f}",
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"Does blending help? — {mass_label}{TITLE_SUFFIX}\n"
                f"winning combination, window +/-{MARGIN_S} s, {RATE} Hz",
                fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, f"plot_model_breakdown_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def run_setting(data, group_keys, tag, mass_label):
    print(f"\n[{mass_label}] greedy search over {group_keys}", flush=True)
    chosen, path, ranked = greedy(data, group_keys, tag)
    win = ranked[0]
    print(f"  BEST: {' + '.join(win['groups'])}  ({win['n_features']} features)")
    print(f"        R2 {win['r2']:+.4f}   MAE {win['mae_w']:.2f} W   "
          f"cruise {win['cruise_mae_w']:.2f} W   blend w_xgb={win['blend_w_xgb']:.2f}")
    print(f"  top {TOP_K}:")
    for d in ranked[:TOP_K]:
        print(f"    {' + '.join(d['groups']):45s} MAE {d['mae_w']:6.2f} W   "
              f"R2 {d['r2']:+.4f}   w_xgb={d['blend_w_xgb']:.2f}")

    s, p_xgb, p_ridge, p_ens = evaluate_combo(data, win["groups"])
    path.to_csv(os.path.join(OUT, f"search_ensemble_{tag}.csv"), index=False)

    p1 = fig_search(path, chosen, tag, mass_label)
    p2 = fig_top4(ranked, tag, mass_label)
    p3 = fig_model_breakdown(data, p_xgb, p_ridge, p_ens, s["blend_w_xgb"], tag, mass_label)
    print(f"  Saved -> {os.path.basename(p1)}, {os.path.basename(p2)}, "
          f"{os.path.basename(p3)}")

    # Ridge half of the ensemble, refit on all 14 flights, saved for
    # interpretability — same convention as RIDGE_MISSION12/coefficients_ridge_*.csv
    cols = cols_for(win["groups"])
    mu, sd = data[cols].mean(), data[cols].std().replace(0, 1)
    rdg = RidgeCV(alphas=ALPHAS)
    rdg.fit(((data[cols] - mu) / sd).values, data["power"].values)
    coef = pd.DataFrame({"feature": cols, "standardized_coef": rdg.coef_}
                        ).sort_values("standardized_coef", key=np.abs, ascending=False)
    coef["alpha"] = float(rdg.alpha_)
    coef["intercept_w"] = float(rdg.intercept_)
    coef.to_csv(os.path.join(OUT, f"coefficients_ridge_component_{tag}.csv"), index=False)

    summary = {"rate_hz": RATE, "margin_s": MARGIN_S, "mass": tag,
              "motors_excluded": EXCLUDE_MOTORS,
              "model": "ensemble (XGBoost + RidgeCV, blend weight grid-searched "
                       "on out-of-fold predictions)",
              "n_rows": int(len(data)), "n_subsets_tried": len(ranked),
              "winner": win,
              "top_combinations": ranked[:TOP_K],
              "selection_path": path.to_dict("records")}
    with open(os.path.join(OUT, f"summary_ensemble_{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    return win, p_xgb, p_ridge, p_ens


def fig_time_compare(data, pred_with, pred_nomass, win_with, win_nomass):
    show = [f for f in SHOW if f in set(data.flight_id)]
    fig, axes = plt.subplots(len(show), 2, figsize=(17, 4.0 * len(show)), squeeze=False)
    rw = r2_score(data.power, pred_with)
    rn = r2_score(data.power, pred_nomass)
    for r, fid in enumerate(show):
        k = (data.flight_id == fid).values
        d = data[k].sort_values("t")
        t, y = d["t"].values, d["power"].values
        ck = (d.phase == "cruise").values
        for c, (p_all, name, groups) in enumerate([
                (pred_with, f"USING MASS — R2 {rw:.3f} overall", win_with["groups"]),
                (pred_nomass, f"WITHOUT MASS — R2 {rn:.3f} overall", win_nomass["groups"])]):
            p = p_all[k]
            ax = axes[r][c]
            ax.fill_between(t, 0, max(y) * 1.1, where=~ck, color="#94a3b8", alpha=0.18,
                            step="mid", zorder=0, label="arming / takeoff / landing")
            ax.plot(t, y, color="#334155", lw=1.4, label="ACTUAL", zorder=3)
            ax.plot(t, p, color="#dc2626" if c else "#16a34a", lw=1.1, alpha=0.9, zorder=2,
                    label=f"ensemble predicted  (R2 {r2_score(y, p):+.3f}, "
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
    fig.suptitle(f"Ensemble (XGBoost + Ridge), leave-one-flight-out, "
                f"window +/-{MARGIN_S} s, {RATE} Hz{TITLE_SUFFIX}\n"
                f"best model using mass vs best model without mass",
                fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, "plot_time_ensemble_compare.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def fig_r2_compare(data, pred_with, pred_nomass):
    flights = sorted(data.flight_id.unique())
    preds = {"using mass": pred_with, "without mass": pred_nomass}
    colors = {"using mass": "#16a34a", "without mass": "#dc2626"}
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

    fig.suptitle(f"Ensemble — best using-mass model vs best without-mass model, "
                f"per flight{TITLE_SUFFIX}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, "plot_r2_ensemble_compare.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def replot(data):
    """
    Skip the full greedy search entirely. Re-read the winning combinations
    already on disk (from a previous full run) and just redraw the two
    comparison plots with a new SHOW list — leave-one-flight-out predictions
    already cover every flight, so nothing about the model changes, only which
    held-out flights are drawn.
    """
    path = os.path.join(OUT, "comparison_ensemble_summary.json")
    if not os.path.exists(path):
        sys.exit(f"--replot needs an existing {path} — run without --replot first.")
    with open(path) as fh:
        prev = json.load(fh)
    groups_mass = prev["best_using_mass"]["groups"]
    groups_no = prev["without_mass"]["groups"]
    print(f"[ENSEMBLE MISSION-12 — REPLOT]  reusing winners from {os.path.basename(path)}")
    print(f"  using mass:   {' + '.join(groups_mass)}")
    print(f"  without mass: {' + '.join(groups_no)}")
    print(f"  new SHOW: {SHOW}")

    _, _, _, pred_mass = evaluate_combo(data, groups_mass)
    _, _, _, pred_no = evaluate_combo(data, groups_no)
    win_mass = {"groups": groups_mass}
    win_no = {"groups": groups_no}

    p1 = fig_time_compare(data, pred_mass, pred_no, win_mass, win_no)
    p2 = fig_r2_compare(data, pred_mass, pred_no)
    print(f"  Saved -> {os.path.basename(p1)}")
    print(f"  Saved -> {os.path.basename(p2)}\n")


def main():
    fdir = find_flights_dir()
    data = load(fdir)
    print(f"[ENSEMBLE MISSION-12]  {RATE} Hz  |  window +/-{MARGIN_S} s"
          f"{'  |  MOTORS EXCLUDED' if EXCLUDE_MOTORS else ''}")
    print(f"  flights from: {fdir}")
    print(f"  writing to:   {OUT}")
    print(f"  {len(data)} rows, {data.flight_id.nunique()} flights, "
          f"{100 * (data.phase != 'cruise').mean():.1f}% not cruise")

    if REPLOT:
        replot(data)
        return

    win_with, pxw, prw, pew = run_setting(data, WITHMASS_GROUPS, "withmass",
                                          "mass included")
    win_no, pxn, prn, pen = run_setting(data, NOMASS_GROUPS, "nomass",
                                        "mass withheld")

    # As in ridge_mission12_study.py: greedy search over WITHMASS_GROUPS is not
    # guaranteed to actually use mass. If it didn't, evaluate the best
    # mass-forced combination separately so the comparison plot means something.
    uses_mass = "mass" in win_with["groups"]
    if uses_mass:
        win_mass, pred_mass = win_with, pew
    else:
        forced = ["mass"] + [g for g in win_with["groups"]]
        s, _, _, pred_mass = evaluate_combo(data, forced)
        win_mass = s
        print(f"\n  NOTE: greedy search over {{{', '.join(WITHMASS_GROUPS)}}} picked "
              f"'{' + '.join(win_with['groups'])}' (MAE {win_with['mae_w']:.2f} W), "
              f"which does not use mass. Forced-mass alternative "
              f"'{' + '.join(win_mass['groups'])}' scores {win_mass['mae_w']:.2f} W.")

    print("\n[COMPARISON]  (best ensemble that USES mass vs best ensemble without it)")
    print(f"  using mass:   {' + '.join(win_mass['groups']):40s} "
          f"MAE {win_mass['mae_w']:6.2f} W   R2 {win_mass['r2']:+.4f}")
    print(f"  without mass: {' + '.join(win_no['groups']):40s} "
          f"MAE {win_no['mae_w']:6.2f} W   R2 {win_no['r2']:+.4f}")
    print(f"  mass costs {win_no['mae_w'] - win_mass['mae_w']:+.2f} W MAE if withheld")

    p1 = fig_time_compare(data, pred_mass, pen, win_mass, win_no)
    p2 = fig_r2_compare(data, pred_mass, pen)
    print(f"\n  Saved -> {os.path.basename(p1)}")
    print(f"  Saved -> {os.path.basename(p2)}")

    with open(os.path.join(OUT, "comparison_ensemble_summary.json"), "w") as fh:
        json.dump({"rate_hz": RATE, "margin_s": MARGIN_S,
                   "motors_excluded": EXCLUDE_MOTORS,
                   "model": "ensemble (XGBoost + RidgeCV blend)",
                   "global_best_offered_mass": {"groups": win_with["groups"],
                                                 "uses_mass": uses_mass,
                                                 "mae_w": win_with["mae_w"],
                                                 "r2": win_with["r2"]},
                   "best_using_mass": win_mass,
                   "without_mass": win_no,
                   "mass_gain_mae_w": round(float(win_no["mae_w"] - win_mass["mae_w"]), 2)},
                  fh, indent=2)
    print(f"  Saved -> comparison_ensemble_summary.json\n")


if __name__ == "__main__":
    main()
