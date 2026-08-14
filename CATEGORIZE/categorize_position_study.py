"""
CATEGORIZE POSITION STUDY — predict payload POSITION (none / front / rear /
diagonal) from flight sensors, with mass vs without mass, exhaustive feature
search, on the +/-12 s mission window.

STANDALONE. Reads only flights/F*/flight_resampled.csv (located by walking up
from this file), writes only into this folder's WITH_MASS/ and WITHOUT_MASS/
subfolders. Classification sibling of the power-regression studies in
14 REGRESION MODEL/ and NO_MASS_STUDY/ — same window, same 1 Hz recommended
rate, same leave-one-flight-out protocol, same train-only z-scoring — but:

  * TARGET is the categorical `position_payload`, not `power`.  So the metrics
    are classification metrics (accuracy, macro-F1, confusion matrix), not
    R2 / MAE.
  * Evaluated PER FLIGHT: the classifier predicts every row of the held-out
    flight, and those row predictions are collapsed to ONE label per flight by
    majority vote. The flight is the unit of truth (the label is constant
    within a flight), and rows within a flight are near-identical in cruise, so
    the honest sample size is 14 flights, not ~thousands of rows. Row-level
    accuracy is also reported, but flagged as optimistic.
  * THREE models are compared: multinomial Logistic Regression and LDA (linear,
    interpretable) plus a Random Forest (tree sanity check), against a
    majority-class baseline.

WHY THESE FEATURES (see PLAN.md for the full argument)
------------------------------------------------------
  * `imbalance` (front_rear, diagonal) is EXCLUDED: those two features were
    engineered specifically to encode load distribution, so predicting position
    from them is circular (leakage-by-design).
  * `motors` is INCLUDED — it is the key signal, and the exhaustive search
    reports which features are actually used. Note in any write-up that
    imbalance is a linear combination of the four motor commands, so keeping
    motors does not remove the load-distribution signal; high accuracy from a
    motor-containing combination means "the motor commands encode the load",
    not independent detection. Whether orientation / IMU add anything on top is
    exactly what the group-importance plot shows.
  * `mass` is the variant axis: WITH_MASS/ offers it to the search, WITHOUT_MASS/
    does not. Mass is confounded with the label (only `none` has 0 g), so it can
    only help separate `none` from the rest — the two subfolders make that
    visible instead of hiding it.
  * `trajectory` excluded (trajectory_3 occurs on exactly one flight — a
    single-flight indicator under leave-one-flight-out), same reason as the
    power studies.

THE 14 FLIGHTS
--------------
  none:     F01, F02            (2 flights)
  front:    F03, F04, F07, F08  (4)
  rear:     F11, F12, F13, F14  (4)
  diagonal: F05, F06, F09, F10  (4)
  full:     never flown -> not in the label set, cannot be predicted or tested.
Majority-class baseline is 4/14 = 0.286; every model is reported against it.

OUTPUT (in WITH_MASS/ and WITHOUT_MASS/)
----------------------------------------
    search_<model>_<tag>.csv          every group subset, ranked by flight accuracy
    summary_<tag>.json                winner per model, metrics, baseline
    per_flight_predictions_<tag>.csv  each flight: truth + each model's prediction
    coefficients_logreg_<tag>.csv     per-class standardized coefficients (winning combo)
    plot_search_<tag>.png             flight accuracy vs #features, all subsets
    plot_confusion_<tag>.png          confusion matrices, the 3 models side by side
    plot_group_importance_<tag>.png   accuracy with vs without each group (Logistic)
  and in this folder:
    comparison_summary.json           headline with-mass vs without-mass numbers

Run:
    python categorize_position_study.py
    python categorize_position_study.py --rate 20    # 20 Hz grid instead of 1 Hz
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

from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

MARGIN_S = 12
RATE = 20 if "--rate" in sys.argv and sys.argv[sys.argv.index("--rate") + 1] == "20" else 1

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = "position_payload"

MOTORS = ["motor_1_front_right", "motor_2_rear_left",
          "motor_3_front_left",  "motor_4_rear_right"]

# Same group definitions as the power studies, minus `imbalance` (leakage-by-
# design) and `trajectory` (single-flight indicator). `mass` is loaded either
# way and only OFFERED to the search in the with-mass run.
GROUPS = {
    "mass":        ["payload_mass"],
    "motors":      MOTORS,
    "orientation": ["roll_rad", "pitch_rad", "yaw_rad"],
    "imu":         ["uav1_hw_api_imu__angular_velocity_x",
                    "uav1_hw_api_imu__angular_velocity_y",
                    "uav1_hw_api_imu__angular_velocity_z",
                    "uav1_hw_api_imu__linear_acceleration_x",
                    "uav1_hw_api_imu__linear_acceleration_y",
                    "uav1_hw_api_imu__linear_acceleration_z"],
    "speed":       ["speed_3d", "speed_horizontal"],
    "velocity":    ["uav1_estimation_manager_uav_state__velocity_linear_x",
                    "uav1_estimation_manager_uav_state__velocity_linear_y",
                    "uav1_estimation_manager_uav_state__velocity_linear_z"],
    "altitude":    ["uav1_mavros_altitude__local"],
}
ALL_COLS = sorted({c for g in GROUPS.values() for c in g})

# Fresh-estimator factories. class_weight="balanced" on LogReg/RF counters the
# uneven class sizes (2 `none` flights vs 4 each of the others) and the uneven
# row counts per flight; LDA uses empirical priors. All deterministic.
MODELS = {
    # RF runs single-threaded on purpose: the exhaustive search is parallelised
    # at the subset level (threading backend), so nested RF parallelism would
    # only oversubscribe cores and thrash. 200 trees is ample on 14 flights.
    "logreg": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "lda":    lambda: LinearDiscriminantAnalysis(),
    "rf":     lambda: RandomForestClassifier(n_estimators=200, random_state=42,
                                             class_weight="balanced", n_jobs=1),
}
MODEL_NICE = {"logreg": "Logistic Regression", "lda": "LDA", "rf": "Random Forest"}


def find_flights_dir():
    d = HERE
    for _ in range(6):
        c = os.path.join(d, "flights")
        if glob.glob(os.path.join(c, "F*", "flight_resampled.csv")):
            return c
        for cand in glob.glob(os.path.join(d, "*", "*", "*", "flights")):
            if glob.glob(os.path.join(cand, "F*", "flight_resampled.csv")):
                return cand
        d = os.path.dirname(d)
    sys.exit("Could not find flights/ containing flight_resampled.csv")


def load(fdir):
    """Mission window +/-MARGIN_S around cruise, at RATE Hz, with the position label."""
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
        a[TARGET] = d[TARGET].iloc[0]           # constant within a flight
        hi = (a["power"] > thr).values
        i0, i1 = hi.argmax(), len(hi) - 1 - hi[::-1].argmax()
        idx = np.arange(len(a))
        keep = (idx >= i0 - MARGIN_S * step) & (idx <= i1 + MARGIN_S * step)
        out.append(a[keep])
    return pd.concat(out, ignore_index=True)


def cols_for(groups):
    return [c for g in groups for c in GROUPS[g]]


def evaluate(data, cols, make_model):
    """
    Leave-one-flight-out row predictions (train-only z-score), then a single
    per-flight label by majority vote over that flight's rows.
    Returns (row_pred array, {flight_id: voted_label}).
    """
    row_pred = np.empty(len(data), dtype=object)
    y = data[TARGET].values
    for h in sorted(data["flight_id"].unique()):
        te_m = (data.flight_id == h).values
        tr, te = data[~te_m], data[te_m]
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
        Xtr = ((tr[cols] - mu) / sd).values
        Xte = ((te[cols] - mu) / sd).values
        m = make_model()
        m.fit(Xtr, y[~te_m])
        row_pred[te_m] = m.predict(Xte)

    votes = {}
    for f in sorted(data["flight_id"].unique()):
        k = (data.flight_id == f).values
        votes[f] = pd.Series(row_pred[k]).mode().iloc[0]   # tie -> alphabetical, deterministic
    return row_pred, votes


def flight_truth(data):
    return {f: data.loc[data.flight_id == f, TARGET].iloc[0]
            for f in sorted(data["flight_id"].unique())}


def score(data, row_pred, votes, labels):
    truth = flight_truth(data)
    fl = sorted(truth)
    yt = [truth[f] for f in fl]
    yp = [votes[f] for f in fl]
    return {
        "flight_acc": round(float(accuracy_score(yt, yp)), 4),
        "flight_macro_f1": round(float(f1_score(yt, yp, labels=labels,
                                                average="macro", zero_division=0)), 4),
        "row_acc": round(float(accuracy_score(data[TARGET].values, row_pred)), 4),
        "n_correct_flights": int(sum(a == b for a, b in zip(yt, yp))),
    }


def baseline(data):
    """
    Majority-class baseline: accuracy of always guessing the single most common
    position. With counts none=2, front=rear=diagonal=4, that is 4/14 = 0.286.

    (A leave-one-flight-out majority vote is intentionally NOT used here: with
    three classes tied at 4 flights, removing one flight leaves a tie that the
    tie-break resolves to a wrong class on every fold, giving a pathological
    0/14 that understates chance. The most-frequent-class rate is the honest,
    standard reference.)
    """
    counts = data.drop_duplicates("flight_id")[TARGET].value_counts()
    return round(float(counts.max() / counts.sum()), 4)


def _score_combo(data, combo, make_model, labels):
    cols = cols_for(combo)
    row_pred, votes = evaluate(data, cols, make_model)
    s = score(data, row_pred, votes, labels)
    return {"groups": combo, "n_groups": len(combo), "n_features": len(cols), **s}


def exhaustive_search(data, group_keys, make_model, labels, tag_model):
    """
    Score every non-empty subset of group_keys, parallelised over subsets with a
    threading backend (RF is single-threaded inside — see MODELS). Threading
    avoids pickling the lambda factories and gives a real speedup because the
    fit calls release the GIL.
    """
    combos = [list(c) for size in range(1, len(group_keys) + 1)
              for c in itertools.combinations(group_keys, size)]
    print(f"    [{tag_model}] scoring {len(combos)} subsets...", flush=True)
    rows = Parallel(n_jobs=-1, backend="threading")(
        delayed(_score_combo)(data, c, make_model, labels) for c in combos)
    df = (pd.DataFrame(rows)
          .sort_values(["flight_acc", "flight_macro_f1", "row_acc"],
                       ascending=False)
          .reset_index(drop=True))
    return df


# ---------------------------------------------------------------- plots -------

def fig_search(searches, tag, mass_label, base_acc, outdir):
    """Flight accuracy vs number of features — best-at-each-size line per model."""
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"logreg": "#2563eb", "lda": "#16a34a", "rf": "#dc2626"}
    for name, df in searches.items():
        ax.scatter(df.n_features, df.flight_acc, s=12, alpha=0.18, color=colors[name])
        best = df.loc[df.groupby("n_features")["flight_acc"].idxmax()].sort_values("n_features")
        ax.plot(best.n_features, best.flight_acc, "-o", color=colors[name], lw=2,
                label=f"{MODEL_NICE[name]} (best {df.flight_acc.max():.2f})")
    ax.axhline(base_acc, color="#64748b", ls="--", lw=1.5,
               label=f"majority-class baseline ({base_acc:.2f})")
    ax.set_xlabel("number of features", fontsize=10)
    ax.set_ylabel("per-flight accuracy — leave-one-flight-out", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Payload-position classification — exhaustive feature search — {mass_label}\n"
                 f"window +/-{MARGIN_S} s, {RATE} Hz, 14 flights", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    p = os.path.join(outdir, f"plot_search_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def fig_confusion(data, winners, labels, tag, mass_label, outdir):
    """Flight-level confusion matrix for each model's winning combination."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    truth = flight_truth(data)
    fl = sorted(truth)
    yt = [truth[f] for f in fl]
    for ax, name in zip(axes, ["logreg", "lda", "rf"]):
        votes = winners[name]["votes"]
        yp = [votes[f] for f in fl]
        cm = confusion_matrix(yt, yp, labels=labels)
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, fontsize=9)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
        acc = winners[name]["flight_acc"]
        ax.set_title(f"{MODEL_NICE[name]}\nacc {acc:.2f} ({winners[name]['n_correct_flights']}/{len(fl)})"
                     f"  —  {' + '.join(winners[name]['groups'])}", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("predicted", fontsize=9); ax.set_ylabel("true", fontsize=9)
    fig.suptitle(f"Payload-position confusion (per flight, held out) — {mass_label}\n"
                 f"window +/-{MARGIN_S} s, {RATE} Hz", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(outdir, f"plot_confusion_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def fig_group_importance(df, group_keys, tag, mass_label, outdir):
    """
    Flight accuracy for subsets that include vs exclude each group, conditioned
    on the single strongest group already being present (normally 'motors').
    Based on the Logistic Regression search. Higher = better.
    """
    singles = df[df.n_groups == 1].sort_values("flight_acc", ascending=False)
    anchor = singles.iloc[0].groups[0]
    base = df[df.groups.apply(lambda gs: anchor in gs)]
    others = [g for g in group_keys if g != anchor]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(others) + 2.5))
    with_, without_ = [], []
    for g in others:
        has = base.groups.apply(lambda gs: g in gs)
        with_.append(base.loc[has, "flight_acc"].values)
        without_.append(base.loc[~has, "flight_acc"].values)
    y = np.arange(len(others)); w = 0.35
    ax.boxplot(with_, positions=y - w / 2 - 0.02, widths=w, vert=False,
               patch_artist=True, boxprops=dict(facecolor="#2563eb", alpha=0.6),
               medianprops=dict(color="black"))
    ax.boxplot(without_, positions=y + w / 2 + 0.02, widths=w, vert=False,
               patch_artist=True, boxprops=dict(facecolor="#94a3b8", alpha=0.6),
               medianprops=dict(color="black"))
    ax.set_yticks(y); ax.set_yticklabels(others, fontsize=9)
    ax.set_xlabel(f"per-flight accuracy, subsets that already include '{anchor}', "
                  "with / without the group", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#2563eb", alpha=0.6),
               plt.Rectangle((0, 0), 1, 1, fc="#94a3b8", alpha=0.6)]
    ax.legend(handles, ["group present", "group absent"], fontsize=9, loc="lower left")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(f"Logistic — does each group help, given {anchor}? — {mass_label}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(outdir, f"plot_group_importance_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def logreg_coefficients(data, cols, labels):
    """
    Descriptive multinomial Logistic fit on ALL 14 flights with the winning
    feature set. Not used for any reported metric (those come from held-out
    flights) — purely to show which standardized feature pushes toward which
    position class. One row per (class, feature).
    """
    mu, sd = data[cols].mean(), data[cols].std().replace(0, 1)
    X = ((data[cols] - mu) / sd).values
    m = LogisticRegression(max_iter=5000, class_weight="balanced")
    m.fit(X, data[TARGET].values)
    recs = []
    for ci, cls in enumerate(m.classes_):
        for fi, feat in enumerate(cols):
            recs.append({"class": cls, "feature": feat,
                         "standardized_coef": round(float(m.coef_[ci, fi]), 4)})
    return pd.DataFrame(recs)


def run_setting(data, group_keys, tag, mass_label, labels, base_acc, outdir):
    os.makedirs(outdir, exist_ok=True)
    print(f"\n[{mass_label}] {2 ** len(group_keys) - 1} non-empty subsets of {group_keys}", flush=True)
    searches, winners = {}, {}
    for name, make in MODELS.items():
        df = exhaustive_search(data, group_keys, make, labels, name)
        df.to_csv(os.path.join(outdir, f"search_{name}_{tag}.csv"), index=False)
        win = df.iloc[0]
        row_pred, votes = evaluate(data, cols_for(win.groups), make)
        winners[name] = {"groups": win.groups, "n_features": int(win.n_features),
                         "flight_acc": win.flight_acc, "flight_macro_f1": win.flight_macro_f1,
                         "row_acc": win.row_acc, "n_correct_flights": int(win.n_correct_flights),
                         "votes": votes}
        searches[name] = df
        print(f"  {MODEL_NICE[name]:20s} best: {' + '.join(win.groups):40s} "
              f"acc {win.flight_acc:.2f} ({win.n_correct_flights}/14)  "
              f"macroF1 {win.flight_macro_f1:.2f}  row {win.row_acc:.2f}")

    p1 = fig_search(searches, tag, mass_label, base_acc, outdir)
    p2 = fig_confusion(data, winners, labels, tag, mass_label, outdir)
    p3 = fig_group_importance(searches["logreg"], group_keys, tag, mass_label, outdir)
    print(f"  Saved -> {os.path.basename(p1)}, {os.path.basename(p2)}, {os.path.basename(p3)}")

    # per-flight predictions table (truth + each model's vote)
    truth = flight_truth(data)
    pf = pd.DataFrame({"flight_id": sorted(truth), "true": [truth[f] for f in sorted(truth)]})
    for name in MODELS:
        pf[f"pred_{name}"] = [winners[name]["votes"][f] for f in sorted(truth)]
        pf[f"correct_{name}"] = pf["true"] == pf[f"pred_{name}"]
    pf.to_csv(os.path.join(outdir, f"per_flight_predictions_{tag}.csv"), index=False)

    # interpretable coefficients for the Logistic winner
    coef = logreg_coefficients(data, cols_for(winners["logreg"]["groups"]), labels)
    coef.to_csv(os.path.join(outdir, f"coefficients_logreg_{tag}.csv"), index=False)

    summary = {"rate_hz": RATE, "margin_s": MARGIN_S, "mass": tag,
               "labels": labels, "n_rows": int(len(data)),
               "n_flights": int(data.flight_id.nunique()),
               "baseline_flight_acc": base_acc,
               "models": {name: {k: v for k, v in w.items() if k != "votes"}
                          for name, w in winners.items()}}
    with open(os.path.join(outdir, f"summary_{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return winners


def main():
    fdir = find_flights_dir()
    data = load(fdir)
    labels = sorted(data[TARGET].unique())
    base_acc = baseline(data)

    print(f"[CATEGORIZE POSITION]  {RATE} Hz  |  window +/-{MARGIN_S} s")
    print(f"  flights from: {fdir}")
    print(f"  {len(data)} rows, {data.flight_id.nunique()} flights, classes {labels}")
    print(f"  majority-class baseline (per flight): {base_acc:.3f}")

    with_groups = list(GROUPS)                      # includes 'mass'
    no_groups = [g for g in GROUPS if g != "mass"]  # excludes 'mass'

    w_with = run_setting(data, with_groups, "withmass", "mass included", labels,
                         base_acc, os.path.join(HERE, "WITH_MASS"))
    w_no = run_setting(data, no_groups, "nomass", "mass withheld", labels,
                       base_acc, os.path.join(HERE, "WITHOUT_MASS"))

    comp = {"rate_hz": RATE, "margin_s": MARGIN_S, "labels": labels,
            "baseline_flight_acc": base_acc,
            "with_mass": {name: {"groups": w_with[name]["groups"],
                                 "flight_acc": w_with[name]["flight_acc"],
                                 "flight_macro_f1": w_with[name]["flight_macro_f1"],
                                 "n_correct_flights": w_with[name]["n_correct_flights"]}
                          for name in MODELS},
            "without_mass": {name: {"groups": w_no[name]["groups"],
                                    "flight_acc": w_no[name]["flight_acc"],
                                    "flight_macro_f1": w_no[name]["flight_macro_f1"],
                                    "n_correct_flights": w_no[name]["n_correct_flights"]}
                             for name in MODELS}}
    with open(os.path.join(HERE, "comparison_summary.json"), "w") as fh:
        json.dump(comp, fh, indent=2)

    print("\n[COMPARISON]  best per-flight accuracy (of 14 flights)")
    for name in MODELS:
        print(f"  {MODEL_NICE[name]:20s} with mass {w_with[name]['flight_acc']:.2f} "
              f"({w_with[name]['n_correct_flights']}/14)   "
              f"without mass {w_no[name]['flight_acc']:.2f} "
              f"({w_no[name]['n_correct_flights']}/14)")
    print(f"  baseline {base_acc:.2f}")
    print("  Saved -> comparison_summary.json\n")


if __name__ == "__main__":
    main()
