"""
Step 10 – Compare the two dataset variants side by side.

Reads the per-flight metrics produced by 08_plot_power_time.py for both runs:

    ML/ml_data/test_flight_metrics.csv            all rows   (idle + flight)
    ML/ml_data_inflight/test_flight_metrics.csv   in-flight rows only

and produces a single comparison table and figure for the thesis.

WHY BOTH NUMBERS MATTER
-----------------------
The all-rows R2 is inflated: idle (~25 W) and flight (~750 W) differ by 30x, so
most of the target's variance is simply "on the ground vs flying", which is easy.
The in-flight R2 answers the question the thesis actually asks - can power be
predicted while airborne. Reporting both is more defensible than reporting the
larger number alone.

Output : ML/comparison/variant_comparison.csv
         ML/comparison/plot_variant_comparison.png

Run (after both pipelines have been run):
    python 05_prepare_ml.py && python 06_train_xgboost.py && python 08_plot_power_time.py
    python 05b_prepare_ml_inflight.py
    ML_DATA_DIR=ML/ml_data_inflight python 06_train_xgboost.py
    ML_DATA_DIR=ML/ml_data_inflight python 08_plot_power_time.py
    python 10_compare_variants.py
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "ML", "comparison")

VARIANTS = [
    ("All rows (idle + flight)", os.path.join(SCRIPT_DIR, "ML", "ml_data")),
    ("In-flight rows only",      os.path.join(SCRIPT_DIR, "ML", "ml_data_inflight")),
]

METRICS = [("r2", "R²", None), ("mae_w", "MAE (W)", "W"), ("rmse_w", "RMSE (W)", "W")]


def load(label, folder):
    path = os.path.join(folder, "test_flight_metrics.csv")
    if not os.path.exists(path):
        sys.exit(f"Missing: {path}\n"
                 f"Run step 08 for this variant first "
                 f"(see the header of this file for the exact commands).")
    df = pd.read_csv(path)
    df.insert(0, "variant", label)
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    combined = pd.concat([load(lab, f) for lab, f in VARIANTS], ignore_index=True)

    # ---- Wide table: one row per flight, one column pair per variant ----
    wide = combined.pivot(index="flight_id", columns="variant",
                          values=[m for m, _, _ in METRICS])
    wide.columns = [f"{m}__{v}" for m, v in wide.columns]
    wide = wide.reset_index()

    csv_path = os.path.join(OUT_DIR, "variant_comparison.csv")
    wide.to_csv(csv_path, index=False)

    print("\n" + "=" * 78)
    print(combined.to_string(index=False))
    print("=" * 78)

    # ---- Figure: one panel per metric, grouped bars by flight ----
    flights = [f for f in wide["flight_id"] if f != "ALL (pooled)"] + ["ALL (pooled)"]
    labels  = [lab for lab, _ in VARIANTS]
    colors  = ["#94a3b8", "#2563eb"]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4.6))
    for ax, (key, nice, _) in zip(axes, METRICS):
        x = range(len(flights))
        w = 0.38
        for k, (lab, color) in enumerate(zip(labels, colors)):
            vals = [wide.loc[wide.flight_id == f, f"{key}__{lab}"].iloc[0] for f in flights]
            ax.bar([i + (k - 0.5) * w for i in x], vals, w, label=lab, color=color)
        ax.set_xticks(list(x))
        ax.set_xticklabels(flights, rotation=45, ha="right", fontsize=9)
        ax.set_title(nice, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.8)
        if key == "r2":
            ax.set_ylim(min(-1.0, ax.get_ylim()[0]), 1.05)
    axes[0].legend(fontsize=9, loc="lower left")

    fig.suptitle("Held-out performance: all rows vs in-flight rows only",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "plot_variant_comparison.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Saved → {csv_path}")
    print(f"  Saved → {png_path}\n")


if __name__ == "__main__":
    main()
