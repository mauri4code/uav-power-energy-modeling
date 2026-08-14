"""
Step 11 – Correlation figures.

Produces two complementary figures from flights/*/flight_resampled.csv,
using IN-FLIGHT rows only (ground/idle rows are excluded, otherwise every
correlation is dominated by the 25 W vs 750 W idle/flight contrast).

  1. plot_correlation_matrix.png
     Classic feature-vs-feature correlation heatmap, plus the power column.
     The standard figure a reader expects in a data chapter.

  2. plot_correlation_within_vs_between.png
     The figure that carries this thesis's main methodological point.
     For every feature it compares:
       WITHIN  = |corr(feature, power)| computed inside each flight, averaged
       BETWEEN = |corr(feature, power)| across the 13 flight-mean values
     Between-flight correlations are large; within-flight ones are near zero.
     payload_mass is the clearest case: 0.96 between flights, and undefined
     within a flight because it is constant there. This is why the model
     predicts flight-average power well (~3%) but cannot track power
     instant-to-instant.

Output : ML/figures/plot_correlation_matrix.png
         ML/figures/plot_correlation_within_vs_between.png
         ML/figures/correlation_within_vs_between.csv

Run: python 11_correlation_plots.py
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FLIGHTS_DIR = os.path.join(SCRIPT_DIR, "flights")
OUT_DIR     = os.path.join(SCRIPT_DIR, "ML", "figures")
FEATURES    = os.path.join(SCRIPT_DIR, "ML", "ml_data", "feature_cols.txt")

# Long ROS topic names are unreadable on an axis; shorten for display only.
RENAME = {
    "uav1_estimation_manager_uav_state__velocity_linear_": "vel_",
    "uav1_hw_api_imu__angular_velocity_":                  "gyro_",
    "uav1_hw_api_imu__linear_acceleration_":               "acc_",
    "uav1_mavros_altitude__local":                         "altitude",
}


def short(name):
    for long_, s in RENAME.items():
        if name.startswith(long_):
            return name.replace(long_, s)
    return name


def load_inflight():
    """All flights, in-flight rows only, with a per-flight idle threshold."""
    out = []
    for f in sorted(glob.glob(os.path.join(FLIGHTS_DIR, "F*", "flight_resampled.csv"))):
        d = pd.read_csv(f)
        thr = 0.5 * (np.percentile(d["power"], 5) + np.percentile(d["power"], 95))
        out.append(d[d["power"] > thr])
    df = pd.concat(out, ignore_index=True)
    print(f"  Loaded {len(df)} in-flight rows from {df.flight_id.nunique()} flights")
    return df


def fig_matrix(df, cols):
    c = df[cols + ["power"]].rename(columns={x: short(x) for x in cols}).corr()

    n = len(c)
    fig, ax = plt.subplots(figsize=(0.55 * n + 3, 0.55 * n + 2))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(n)); ax.set_xticklabels(c.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(c.columns, fontsize=8)

    for i in range(n):
        for j in range(n):
            v = c.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.55 else "black")

    # Outline the power row/column — the one everything is compared against.
    k = n - 1
    ax.add_patch(plt.Rectangle((-0.5, k - 0.5), n, 1, fill=False, ec="black", lw=2))
    ax.add_patch(plt.Rectangle((k - 0.5, -0.5), 1, n, fill=False, ec="black", lw=2))

    ax.set_title("Feature correlation matrix — in-flight rows only\n"
                 "(power row/column outlined)", fontsize=12, fontweight="bold", pad=14)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "plot_correlation_matrix.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def fig_within_between(df, cols):
    # WITHIN: correlation with power inside each flight, averaged over flights
    within = {}
    for c in cols:
        vals = []
        for _, g in df.groupby("flight_id"):
            if g[c].std() > 1e-9 and g["power"].std() > 1e-9:
                vals.append(abs(np.corrcoef(g[c], g["power"])[0, 1]))
        within[c] = np.mean(vals) if vals else np.nan

    # BETWEEN: correlation across the 13 flight-mean values
    means   = df.groupby("flight_id")[cols + ["power"]].mean()
    between = means[cols].corrwith(means["power"]).abs()

    t = (pd.DataFrame({"within": pd.Series(within), "between": between})
         .sort_values("between", ascending=True))
    t.index = [short(i) for i in t.index]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(t) + 2.5))
    y = np.arange(len(t)); h = 0.4
    ax.barh(y + h / 2, t["between"], h, color="#2563eb", label="BETWEEN flights")
    ax.barh(y - h / 2, t["within"].fillna(0), h, color="#94a3b8", label="WITHIN a flight")

    # payload_mass is constant inside a flight -> undefined, not zero. Say so.
    for i, (w, name) in enumerate(zip(t["within"], t.index)):
        if np.isnan(w):
            ax.text(0.012, i - h / 2, "constant within a flight — undefined",
                    va="center", fontsize=7.5, style="italic", color="#475569")

    ax.set_yticks(y); ax.set_yticklabels(t.index, fontsize=8.5)
    ax.set_xlabel("|correlation with power|", fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title("Where the information is: between flights, not within them\n"
                 f"mean over all features — within {t['within'].mean():.2f}  vs  "
                 f"between {t['between'].mean():.2f}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "plot_correlation_within_vs_between.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()

    csv = os.path.join(OUT_DIR, "correlation_within_vs_between.csv")
    t.sort_values("between", ascending=False).round(3).to_csv(csv, index_label="feature")
    return p, csv, t


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cols = open(FEATURES).read().strip().split("\n")

    print("\n[Step 11] Correlation figures")
    df = load_inflight()

    p1 = fig_matrix(df, cols)
    p2, csv, t = fig_within_between(df, cols)

    print("\n  Top features by BETWEEN-flight correlation with power:")
    for name, r in t.sort_values("between", ascending=False).head(6).iterrows():
        w = "  n/a" if np.isnan(r["within"]) else f"{r['within']:5.2f}"
        print(f"    {name:34s} between={r['between']:5.2f}   within={w}")

    print(f"\n  Saved → {p1}")
    print(f"  Saved → {p2}")
    print(f"  Saved → {csv}\n")


if __name__ == "__main__":
    main()
