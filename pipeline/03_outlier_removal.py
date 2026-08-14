"""
Step 3 – IQR-based outlier removal applied to all sensor CSV files.

Input  : <OUTPUT_FOLDER>/<sensor>.csv
Output : <OUTPUT_FOLDER>/<sensor>_clean.csv  (one per sensor file)

Run standalone : python 03_outlier_removal.py
Run via runner : called automatically by run_flight.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # no GUI — save only
import matplotlib.pyplot as plt

MOTOR_COLS = [
    "motor_1_front_right",
    "motor_2_rear_left",
    "motor_3_front_left",
    "motor_4_rear_right",
]
MOTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# Sensor files and the columns to check for outliers.
# "timestamp" is never treated as an outlier target.
#
# Only linear accelerations are checked for outliers — these are the noisiest
# IMU signals and the most prone to real spikes.
# All other sensor files are still processed (to generate *_clean.csv for
# step 4) but no rows are removed from them.
SENSOR_COLS = {
    "uav1_hw_api_imu.csv": [
        "uav1_hw_api_imu__linear_acceleration_x",
        "uav1_hw_api_imu__linear_acceleration_y",
        "uav1_hw_api_imu__linear_acceleration_z",
    ],
    "uav1_estimation_manager_uav_state.csv": [],   # no outlier removal
    "uav1_hw_api_orientation_with_rpy.csv":   [],   # no outlier removal
    "uav1_mavros_altitude.csv":               [],   # no outlier removal
    "uav1_mrs_uav_status_uav_status.csv":     [],   # no outlier removal
    "uav1_motor_commands.csv":                [],   # no outlier removal (normalized 0–1, clean by design)
}


def iqr_keep_mask(series: pd.Series, k: float) -> pd.Series:
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return (series >= Q1 - k * IQR) & (series <= Q3 + k * IQR)


def clean_file(in_path: str, cols: list, k: float, out_path: str) -> None:
    df = pd.read_csv(in_path)
    existing = [c for c in cols if c in df.columns]
    df = df.dropna(subset=existing).copy()

    mask = pd.Series(True, index=df.index)
    for col in existing:
        mask &= iqr_keep_mask(df[col], k)

    n_removed = (~mask).sum()
    df_clean  = df[mask]
    pct       = 100 * n_removed / max(len(df), 1)

    print(f"  {os.path.basename(in_path)}: "
          f"{n_removed} rows removed ({pct:.2f}%), "
          f"{len(df_clean)} remaining")

    df_clean.to_csv(out_path, index=False)


def plot_motor_outlier_check(output_folder: str) -> None:
    """
    Plot motor commands before vs after outlier removal.
    Since no IQR removal is applied to motors (normalized 0–1, clean by design),
    the two lines should overlap perfectly — this plot confirms that.
    Saved to: <output_folder>/plot_motor_outlier_check.png
    """
    raw_path   = os.path.join(output_folder, "uav1_motor_commands.csv")
    clean_path = os.path.join(output_folder, "uav1_motor_commands_clean.csv")

    if not os.path.exists(raw_path) or not os.path.exists(clean_path):
        print("  SKIP motor outlier plot (files not found)")
        return

    raw   = pd.read_csv(raw_path)
    clean = pd.read_csv(clean_path)

    # Relative time from start
    t0 = raw["timestamp"].min()
    raw["time_s"]   = raw["timestamp"]   - t0
    clean["time_s"] = clean["timestamp"] - t0

    n_removed = len(raw) - len(clean)

    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Motor Commands — Outlier Check (before vs after)\n"
        f"Rows removed: {n_removed}  |  Before: {len(raw)}  After: {len(clean)}",
        fontsize=11, fontweight="bold"
    )

    for i, (col, color) in enumerate(zip(MOTOR_COLS, MOTOR_COLORS)):
        axs[i].plot(raw["time_s"],   raw[col],   color=color,   linewidth=1.2,
                    alpha=0.6, label="before (raw)")
        axs[i].plot(clean["time_s"], clean[col], color="black", linewidth=0.8,
                    linestyle="--", alpha=0.9, label="after (clean)")
        axs[i].set_title(col, fontsize=9)
        axs[i].set_ylabel("Command (0–1)", fontsize=8)
        axs[i].set_ylim(-0.05, 1.05)
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(fontsize=7, loc="upper right")

    axs[-1].set_xlabel("Time (s) from flight start", fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(output_folder, "plot_motor_outlier_check.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {out_path}")


def run(output_folder, k):

    print(f"[Step 3] Outlier removal  (k={k})")

    for fname, cols in SENSOR_COLS.items():
        in_path  = os.path.join(output_folder, fname)
        out_path = os.path.join(output_folder, fname.replace(".csv", "_clean.csv"))

        if not os.path.exists(in_path):
            print(f"  SKIP (not found): {fname}")
            continue

        clean_file(in_path, cols, k, out_path)

    plot_motor_outlier_check(output_folder)
    print()


if __name__ == "__main__":
    print("Run via: python run_flight.py <test_folder>")
