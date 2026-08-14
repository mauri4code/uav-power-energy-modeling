"""
LAG-SEARCH STUDY — which features help, and which of them need their recent past?

STANDALONE. Reads only flights/F*/flight_resampled.csv, writes only into LAG_SEARCH/.

WHY THIS EXISTS
---------------
Two earlier studies each answered half a question:

  MISSION12   searched feature groups, but every feature was INSTANTANEOUS.
  LAGS        showed that a one-second lag helps, but only ever on motors + mass.

Neither asked the combined question: given that history helps, WHICH signals need
it? Motors plausibly do, because power responds to a command with ESC and battery
delay. Payload mass cannot — it is constant. Attitude or IMU might, or might not.

HOW THE SEARCH IS SET UP
------------------------
Every physical group is offered to the search TWICE, as two independent candidates:

    motors        the four commands at time t
    motors_lag1   the same four commands at time t-1

A lagged group may only enter once its instantaneous partner is already in. A lag is
a SUPPLEMENT to the present value, never a replacement: "the motor commands one
second ago, and nothing about now" is not a model anyone would deploy, so the search
is not allowed to propose one. Within that rule it is free to take a group with its
lag, or without — so it can still discover that motors need history while velocity
does not, which a global "lag everything" switch could never show.

`mass` is offered once only. It is constant within a flight, so its lag is a copy of
itself and adding it would be a free duplicate column.

WHAT A LAG FEATURE IS — AND IS NOT
----------------------------------
`shift(1)` on a 1 Hz table. For the row at t = 60 s, `motors_lag1` holds the commands
recorded at t = 59 s. It is the same signal, shifted. That is deliberate: XGBoost has
no memory, it sees one row at a time and cannot know a previous row existed, so
shifting the column is the only way to give it access to the past.

Only INPUTS are lagged. Power is never lagged. Feeding past power would make this an
autoregressive model that predicts 750 W because it was 750 W a second ago, which
would be leakage dressed up as a result.

Shifts are computed INSIDE each flight and the first 2 rows of every flight are
dropped, identically for every candidate set, so no value crosses a flight boundary
and all combinations are scored on exactly the same rows.

WHY 1 Hz ONLY
-------------
Power is logged at 1 Hz and forward-filled onto the 20 Hz grid. At 20 Hz a "lag of
one row" is 0.05 s against a target that only updates every 20 rows — it would carry
no information about the target at all. Lag features are only meaningful at the rate
the target is actually measured.

Everything is leave-one-flight-out. Loss is absolute error, which beat squared error
in every configuration tested and optimises the metric being reported.

Output (in LAG_SEARCH/):
    search_<tag>.csv          the greedy path, every step scored
    plot_search_<tag>.png     MAE and R2 as groups are added
    plot_top4_<tag>.png       the best 4 combinations
    plot_time_<tag>.png       power over time, best 4 sets, R2 per line
    plot_r2_<tag>.png         per-flight R2 and MAE
    plot_lag_gain_<tag>.png   each group with vs without its lag, side by side
    per_flight_<tag>.csv
    summary_<tag>.json

Run:
    python lag_search_study.py                # mass withheld
    python lag_search_study.py --with-mass    # mass included
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

MARGIN_S = 12
OBJ = "reg:absoluteerror"
SHOW = ["F08", "F09", "F13"]
TOP_K = 4
DROP_HEAD = 2          # rows dropped at the start of each flight, same for all sets

WITH_MASS = ("--with-mass" in sys.argv) or (os.environ.get("WITH_MASS") == "1")

# ------------------------------------------------------------- --no-motors mode
# Motor commands are a CONTROL OUTPUT, not a mission parameter. A model given them
# is close to reading the throttle: on a real planning problem you would not have
# them, because they are what the aircraft decides in response to the load. This
# mode removes them to ask a different, harder question — how much of the power can
# be recovered from state alone?
#
# IMBALANCE IS ALSO REMOVED, because it is arithmetic on the motor commands:
#     front_rear_imbalance = (M1+M3)/2 - (M2+M4)/2
#     diagonal_imbalance   = (M3+M4)/2 - (M1+M2)/2
# Keeping it while dropping `motors` would put the same information straight back
# in under another name. --allow-imbalance overrides this, for comparison only.
NO_MOTORS = "--no-motors" in sys.argv
ALLOW_IMB = "--allow-imbalance" in sys.argv

OUT = os.path.join(HERE, "NO_MOTORS" if NO_MOTORS else "LAG_SEARCH")
os.makedirs(OUT, exist_ok=True)

TAG = ("withmass" if WITH_MASS else "nomass") + ("_imb" if NO_MOTORS and ALLOW_IMB else "")

MOTORS = ["motor_1_front_right", "motor_2_rear_left",
          "motor_3_front_left",  "motor_4_rear_right"]

# Physical groups. Each is offered to the search both as-is and lagged by 1 s.
BASE_GROUPS = {
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
    # TRAJECTORY REMAINS EXCLUDED, for the reason given in mission12_study.py:
    # trajectory_3 occurs on exactly one flight, so under leave-one-flight-out it
    # would act as a dedicated indicator for that flight rather than a feature.
}

if NO_MOTORS:
    BASE_GROUPS.pop("motors")
    if not ALLOW_IMB:
        BASE_GROUPS.pop("imbalance")

RAW = sorted({c for g in BASE_GROUPS.values() for c in g})

# Candidate groups actually offered to the greedy search
GROUPS = {}
for name, cols in BASE_GROUPS.items():
    GROUPS[name] = list(cols)
    GROUPS[f"{name}_lag1"] = [f"{c}_lag1" for c in cols]
if WITH_MASS:
    # constant within a flight -> its lag is a duplicate, so it is offered once only
    GROUPS["mass"] = ["payload_mass"]


MODE_TXT = ("NO MOTOR COMMANDS" + (" (imbalance allowed)" if ALLOW_IMB else "")
            if NO_MOTORS else "all groups")
MASS_TXT = "mass included" if WITH_MASS else "mass withheld"
SUB = f"1 Hz, window ±{MARGIN_S} s, {MASS_TXT}, {MODE_TXT}"


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
    """1 Hz mission window; every raw column also present shifted by one second."""
    out = []
    need = RAW + (["payload_mass"] if WITH_MASS else [])
    for f in sorted(glob.glob(os.path.join(fdir, "F*", "flight_resampled.csv"))):
        d = pd.read_csv(f).sort_values("timestamp").copy()
        thr = 0.5 * (np.percentile(d["power"], 5) + np.percentile(d["power"], 95))
        d["t"] = d["timestamp"] - d["timestamp"].iloc[0]
        d["sec"] = np.floor(d["t"]).astype(int)
        a = d.groupby("sec", as_index=False)[need + ["power", "t"]].mean()
        a["flight_id"] = d["flight_id"].iloc[0]

        hi = (a["power"] > thr).values
        i0, i1 = hi.argmax(), len(hi) - 1 - hi[::-1].argmax()
        idx = np.arange(len(a))
        a["phase"] = np.where(hi, "cruise", "other")
        a = a[(idx >= i0 - MARGIN_S) & (idx <= i1 + MARGIN_S)].reset_index(drop=True)

        # one-second shift, computed inside this flight only
        for c in RAW:
            a[f"{c}_lag1"] = a[c].shift(1)

        out.append(a.iloc[DROP_HEAD:].reset_index(drop=True))
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


# ---------------------------------------------------------------- score cache
# 15 candidate groups means the greedy search fits a few hundred XGBoost models,
# each over 14 folds. Results are keyed by the SET of groups and written to disk
# after every evaluation, so an interrupted run resumes instead of restarting.
# Delete the cache file to force a clean re-run.
CACHE_PATH = os.path.join(OUT, f"_cache_{TAG}.json")
_cache = {}
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH) as fh:
        _cache = json.load(fh)
    print(f"  resuming: {len(_cache)} combinations already scored "
          f"(delete {os.path.basename(CACHE_PATH)} to start over)")


def _key(groups):
    return "|".join(sorted(groups))


def score_set(data, groups):
    """Score a set of groups, using the on-disk cache when possible."""
    k = _key(groups)
    if k not in _cache:
        _cache[k] = score(data, evaluate(data, cols_for(groups)))
        with open(CACHE_PATH, "w") as fh:
            json.dump(_cache, fh)
    return _cache[k]


def allowed(g, chosen):
    """
    A lagged group may only enter once its instantaneous partner is already in.

    Rationale: a lag is a SUPPLEMENT to the present value, not a replacement for
    it. 'the motor commands one second ago, and nothing about now' is not a model
    anyone would deploy, and letting the search pick such a set would produce
    combinations that score acceptably but cannot be justified physically.
    Constraining it also shrinks the search space and removes a class of
    selection-bias artefacts.
    """
    return not g.endswith("_lag1") or g[:-5] in chosen


def greedy(data):
    """Forward selection over instantaneous and lagged candidates."""
    remaining, chosen, path, best_mae = list(GROUPS), [], [], np.inf
    seen, step = {}, 0
    while remaining:
        step += 1
        trials = []
        for g in [g for g in remaining if allowed(g, chosen)]:
            combo = chosen + [g]
            s = score_set(data, combo)
            seen[frozenset(combo)] = {"groups": combo, **s}
            trials.append((s["mae_w"], g, s, len(cols_for(combo))))
            print(f"    step {step}: try +{g:18s} -> MAE {s['mae_w']:6.2f} W", flush=True)
        if not trials:
            break
        trials.sort()
        mae, g, s, nfeat = trials[0]
        improved = mae < best_mae - 0.05
        path.append({"step": step, "added": g, "n_features": nfeat,
                     "set": " + ".join(chosen + [g]), "improved": bool(improved), **s})
        if not improved:
            print(f"    step {step}: best was +{g} at {mae:.2f} W — no improvement, stop\n",
                  flush=True)
            break
        chosen.append(g); remaining.remove(g); best_mae = mae
        print(f"    step {step}: KEEP +{g}  (MAE {mae:.2f} W, R² {s['r2']:.3f})\n",
              flush=True)
    return chosen, pd.DataFrame(path), sorted(seen.values(), key=lambda d: d["mae_w"])


def lag_gain(data):
    """
    For each physical group, does its own lag add anything?

    Scored on the group alone (plus mass, if in use) so the comparison is clean:
        X            vs   X + X_lag1
    This is the figure that answers 'which signals need history', independently of
    whatever the greedy search happened to pick.
    """
    base = ["mass"] if WITH_MASS else []
    rows = []
    for name in BASE_GROUPS:
        a = score_set(data, base + [name])
        b = score_set(data, base + [name, f"{name}_lag1"])
        rows.append({"group": name, "mae_now": a["mae_w"], "mae_with_lag": b["mae_w"],
                     "gain_w": round(a["mae_w"] - b["mae_w"], 2),
                     "r2_now": a["r2"], "r2_with_lag": b["r2"]})
        print(f"    {name:12s} alone {a['mae_w']:6.2f} W  ->  with lag "
              f"{b['mae_w']:6.2f} W   gain {a['mae_w'] - b['mae_w']:+6.2f} W", flush=True)
    return pd.DataFrame(rows)


def fig_lag_gain(lg):
    lg = lg.sort_values("gain_w")
    y = np.arange(len(lg))
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.55 * len(lg) + 3.8))
    axes[0].barh(y - 0.2, lg["mae_now"], 0.4, color="#94a3b8", label="now only")
    axes[0].barh(y + 0.2, lg["mae_with_lag"], 0.4, color="#2563eb", label="now + lag1")
    axes[0].set_yticks(y); axes[0].set_yticklabels(lg["group"], fontsize=9)
    axes[0].set_xlabel("MAE (W) — lower is better", fontsize=10)
    axes[0].grid(axis="x", alpha=0.3); axes[0].legend(fontsize=9)
    axes[0].set_title("Each group on its own" +
                      (" (plus mass)" if WITH_MASS else ""),
                      fontsize=11, fontweight="bold")

    cols = ["#16a34a" if v > 0 else "#dc2626" for v in lg["gain_w"]]
    axes[1].barh(y, lg["gain_w"], color=cols)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_yticks(y); axes[1].set_yticklabels(lg["group"], fontsize=9)
    axes[1].set_xlabel("MAE reduction from adding the lag (W)", fontsize=10)
    axes[1].grid(axis="x", alpha=0.3)
    axes[1].set_title("green = history helps,  red = history hurts",
                      fontsize=11, fontweight="bold")
    for i, v in enumerate(lg["gain_w"]):
        axes[1].text(v, i, f"  {v:+.2f}", va="center", fontsize=9)

    fig.suptitle(f"Which signals need their recent past? — {SUB}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, f"plot_lag_gain_{TAG}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def fig_search(path, chosen):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    kept, rej = path[path.improved], path[~path.improved]
    for ax, key, nice, better in [(axes[0], "mae_w", "MAE (W)", "lower is better"),
                                  (axes[1], "r2", "R²", "higher is better")]:
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
    fig.suptitle(f"Greedy selection over instantaneous and lagged groups — {SUB}\n"
                 f"final set: {' + '.join(chosen)}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, f"plot_search_{TAG}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def fig_top4(ranked, k=TOP_K):
    t = ranked[:k][::-1]
    labels = [" + ".join(d["groups"]) for d in t]
    y = np.arange(len(t))
    fig, axes = plt.subplots(1, 2, figsize=(16, 0.6 * len(t) + 3.6))
    for ax, key, nice in [(axes[0], "mae_w", "MAE (W) — lower is better"),
                          (axes[1], "r2", "R² — higher is better")]:
        vals = [d[key] for d in t]
        ax.barh(y, vals, color=["#16a34a" if i == len(t) - 1 else "#2563eb"
                                for i in range(len(t))])
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlabel(nice, fontsize=10); ax.grid(axis="x", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(v, i, f"  {v:.3f}" if key == "r2" else f"  {v:.2f}",
                    va="center", fontsize=9)
    fig.suptitle(f"Best {k} combinations — {SUB}\n"
                 f"green = best; note how small the spread is",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, f"plot_top4_{TAG}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def fig_time(data, preds):
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
                        step="mid", zorder=0, label="arming / takeoff / landing")
        ax.plot(t, y, color="#334155", lw=1.6, label="ACTUAL", zorder=5)
        for j, (label, p_all) in enumerate(preds.items()):
            p = p_all[k]
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
    fig.suptitle(f"{SUB}\n"
                 f"best {len(preds)} combinations, instantaneous and lagged features",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, f"plot_time_{TAG}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def fig_r2(data, preds):
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
    axes[0].set_title("R² per held-out flight\n(negative = worse than that flight's "
                      "own mean; pooled values in the legend are much higher)",
                      fontsize=11, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3); axes[0].legend(fontsize=7.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(flights, rotation=45, fontsize=8.5)
    axes[1].set_ylabel("MAE (W)", fontsize=10)
    axes[1].set_title("Error per held-out flight", fontsize=11, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3); axes[1].legend(fontsize=7.5)
    fig.suptitle(f"Best {len(preds)} combinations, per flight — {SUB}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, f"plot_r2_{TAG}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"per_flight_{TAG}.csv"), index=False)
    return p


def main():
    fdir = find_flights_dir()
    data = load(fdir)
    print(f"\n[{'NO-MOTORS SEARCH' if NO_MOTORS else 'LAG SEARCH'}]  1 Hz  |  mass "
          f"{'INCLUDED' if WITH_MASS else 'WITHHELD'}  |  window ±{MARGIN_S} s")
    if NO_MOTORS:
        print("  motor commands EXCLUDED"
              + ("; imbalance allowed back in (it is derived from them)"
                 if ALLOW_IMB else " — and imbalance too, it is derived from them"))
    print(f"  {len(data)} rows, {data.flight_id.nunique()} flights, "
          f"{100 * (data.phase != 'cruise').mean():.1f}% not cruise")
    print(f"  {len(GROUPS)} candidate groups "
          f"({len(BASE_GROUPS)} instantaneous + {len(BASE_GROUPS)} lagged"
          f"{' + mass' if WITH_MASS else ''})\n")

    print("  does each group need its own lag?")
    lg = lag_gain(data)
    lg.to_csv(os.path.join(OUT, f"lag_gain_{TAG}.csv"), index=False)
    print()

    print("  greedy forward selection:")
    chosen, path, ranked = greedy(data)
    cols = cols_for(chosen)
    s = score_set(data, chosen)

    top = ranked[:TOP_K]
    print(f"  best {len(top)} combinations:")
    for i, d in enumerate(top, 1):
        print(f"    {i}. {' + '.join(d['groups']):46s} "
              f"MAE {d['mae_w']:6.2f} W   R² {d['r2']:+.4f}", flush=True)
    preds = {" + ".join(d["groups"]): evaluate(data, cols_for(d["groups"])) for d in top}
    print()

    _other = ("n/a (cruise only)" if s["other_mae_w"] is None
              else f"{s['other_mae_w']:.2f} W")
    print(f"  FINAL: {' + '.join(chosen)}  ({len(cols)} features)")
    print(f"         R² {s['r2']:+.4f}   MAE {s['mae_w']:.2f} W   "
          f"cruise {s['cruise_mae_w']:.2f} W   other {_other}\n")

    path.to_csv(os.path.join(OUT, f"search_{TAG}.csv"), index=False)
    with open(os.path.join(OUT, f"summary_{TAG}.json"), "w") as fh:
        json.dump({"rate_hz": 1, "with_mass": WITH_MASS, "margin_s": MARGIN_S,
                   "objective": OBJ, "n_rows": int(len(data)),
                   "chosen_groups": chosen, "features": cols, "metrics": s,
                   "lag_gain_per_group": lg.to_dict("records"),
                   "top_combinations": [
                       {"rank": i + 1, "groups": d["groups"], "mae_w": d["mae_w"],
                        "r2": d["r2"], "cruise_mae_w": d["cruise_mae_w"]}
                       for i, d in enumerate(ranked[:TOP_K])],
                   "selection_path": path.to_dict("records")}, fh, indent=2)

    for p in (fig_lag_gain(lg), fig_search(path, chosen), fig_top4(ranked),
              fig_time(data, preds), fig_r2(data, preds)):
        print(f"  Saved → {os.path.basename(OUT)}/{os.path.basename(p)}")
    print(f"  Saved → {os.path.basename(OUT)}/search_{TAG}.csv, summary_{TAG}.json, "
          f"per_flight_{TAG}.csv, lag_gain_{TAG}.csv\n")


if __name__ == "__main__":
    main()
