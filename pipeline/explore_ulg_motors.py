"""
Standalone script — Extract and plot motor commands from a PX4 .ulg file.

Reads actuator_motors.control[0-3] (Motor 1–4 commands) from a .ulg log file,
plots them over time, and optionally saves them to CSV.

This script is independent of the ROS bag preprocessing pipeline.

Usage:
    python explore_ulg_motors.py

Requirements:
    pip install pyulog pandas matplotlib
"""

from pyulog import ULog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
#  SETTINGS  ← only change these
# ============================================================
ULG_FILE    = "log_34_UnknownDate.ulg"
SAVE_CSV    = True                        # save extracted data to CSV
OUTPUT_CSV  = "ulg_motor_commands.csv"   # output CSV filename
OUTPUT_PLOT = "ulg_motor_commands.png"   # output plot filename

# Motor labels — physical positions on the UAV frame
MOTORS = {
    "control[0]": "motor_1_front_right",
    "control[1]": "motor_2_rear_left",
    "control[2]": "motor_3_front_left",
    "control[3]": "motor_4_rear_right",
}

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# Output plot for derived imbalance features
OUTPUT_PLOT_IMBALANCE = "ulg_motor_imbalances.png"


# ============================================================
#  EXTRACT
# ============================================================
def extract_motor_commands(ulg_file):
    """Extract actuator_motors.control[0-3] from a .ulg file."""
    print(f"\nLoading: {ulg_file}")
    ulog = ULog(ulg_file)

    # Find actuator_motors topic
    motor_data = None
    for d in ulog.data_list:
        if d.name == "actuator_motors" and d.multi_id == 0:
            motor_data = d
            break

    if motor_data is None:
        raise RuntimeError("Topic 'actuator_motors' not found in the .ulg file.")

    # Build dataframe — convert timestamp from microseconds to seconds
    df = pd.DataFrame()
    df["timestamp"] = motor_data.data["timestamp"] * 1e-6
    df["time_s"]    = df["timestamp"] - df["timestamp"].iloc[0]   # relative time from 0

    for field, label in MOTORS.items():
        df[label] = motor_data.data[field]

    print(f"  Extracted {len(df)} samples")
    print(f"  Duration : {df['time_s'].max():.1f} s")
    print(f"  Sampling : ~{len(df) / df['time_s'].max():.1f} Hz")
    print(f"\n  Motor command ranges (normalized 0–1):")
    for label in MOTORS.values():
        print(f"    {label}: min={df[label].min():.4f}  max={df[label].max():.4f}  "
              f"mean={df[label].mean():.4f}")

    return df


# ============================================================
#  DERIVED FEATURES  (imbalances)
# ============================================================
def compute_imbalances(df):
    """
    Compute front/rear and diagonal motor imbalance signals.

    Motor physical positions:
        M3 front_left  |  M1 front_right
        M2 rear_left   |  M4 rear_right

    Payload positions used in this experiment and what they activate:
        none      → all ≈ 0
        front     → M1 + M3  →  front_rear_imbalance > 0
        rear      → M2 + M4  →  front_rear_imbalance < 0
        diagonal  → M3 + M4  →  diagonal_imbalance   > 0
        full      → all 4    →  both ≈ 0 (symmetric)

    Derived features:
        front_motor_mean     = mean(M1_front_right, M3_front_left)
        rear_motor_mean      = mean(M2_rear_left,   M4_rear_right)
        front_rear_imbalance = front_motor_mean - rear_motor_mean
            > 0  → front motors working harder  (front payload)
            < 0  → rear motors working harder   (rear payload)

        diagonal_A_mean      = mean(M3_front_left,  M4_rear_right)   ← loaded diagonal
        diagonal_B_mean      = mean(M1_front_right, M2_rear_left)    ← opposite diagonal
        diagonal_imbalance   = diagonal_A_mean - diagonal_B_mean
            > 0  → M3+M4 diagonal working harder  (diagonal payload)
            ≈ 0  → symmetric (none, front, rear, or full)
    """
    df = df.copy()

    df["front_motor_mean"]     = (df["motor_1_front_right"] + df["motor_3_front_left"]) / 2
    df["rear_motor_mean"]      = (df["motor_2_rear_left"]   + df["motor_4_rear_right"]) / 2
    df["front_rear_imbalance"] = df["front_motor_mean"] - df["rear_motor_mean"]

    df["diagonal_A_mean"]      = (df["motor_3_front_left"]  + df["motor_4_rear_right"]) / 2
    df["diagonal_B_mean"]      = (df["motor_1_front_right"] + df["motor_2_rear_left"])  / 2
    df["diagonal_imbalance"]   = df["diagonal_A_mean"] - df["diagonal_B_mean"]

    print(f"\n  Derived imbalance features:")
    for col in ["front_rear_imbalance", "diagonal_imbalance"]:
        print(f"    {col}: min={df[col].min():.4f}  max={df[col].max():.4f}  "
              f"mean={df[col].mean():.4f}  std={df[col].std():.4f}")

    return df


# ============================================================
#  PLOT — individual motor commands
# ============================================================
def plot_motor_commands(df, output_plot):
    """Plot all 4 motor commands on the same axis and individually."""
    fig, axs = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    fig.suptitle("Motor Commands — actuator_motors.control[0–3]", fontsize=13, fontweight="bold")

    motor_labels = list(MOTORS.values())

    # ---- Top plot: all 4 motors overlaid ----
    for i, (label, color) in enumerate(zip(motor_labels, COLORS)):
        axs[0].plot(df["time_s"], df[label], color=color, linewidth=0.8,
                    alpha=0.85, label=label)
    axs[0].set_ylabel("Command (0–1)", fontsize=9)
    axs[0].set_title("All Motors — Overlay", fontsize=10)
    axs[0].legend(fontsize=8, loc="upper right")
    axs[0].set_ylim(-0.05, 1.05)
    axs[0].grid(True, alpha=0.3)

    # ---- Individual plots ----
    for i, (label, color) in enumerate(zip(motor_labels, COLORS)):
        ax = axs[i + 1]
        ax.plot(df["time_s"], df[label], color=color, linewidth=0.8)
        ax.set_ylabel("Command (0–1)", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # Annotate mean
        mean_val = df[label].mean()
        ax.axhline(mean_val, color="black", linestyle="--", linewidth=0.8,
                   label=f"mean={mean_val:.3f}")
        ax.legend(fontsize=8, loc="upper right")

    axs[-1].set_xlabel("Time (s) from start of log", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {output_plot}")
    plt.show()


# ============================================================
#  PLOT — imbalance features
# ============================================================
def plot_imbalances(df, output_plot):
    """
    Plot front/rear and diagonal imbalance signals alongside the
    motor pairs used to compute them.
    """
    fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Motor Imbalance Features", fontsize=13, fontweight="bold")

    # ---- Front vs Rear means ----
    axs[0].plot(df["time_s"], df["front_motor_mean"], color="#1f77b4",
                linewidth=0.8, label="front_motor_mean  (M1 + M3)")
    axs[0].plot(df["time_s"], df["rear_motor_mean"],  color="#ff7f0e",
                linewidth=0.8, label="rear_motor_mean   (M2 + M4)")
    axs[0].set_ylabel("Command (0–1)", fontsize=9)
    axs[0].set_title("Front vs Rear Motor Means", fontsize=10)
    axs[0].legend(fontsize=8, loc="upper right")
    axs[0].set_ylim(-0.05, 1.05)
    axs[0].grid(True, alpha=0.3)

    # ---- Front–Rear imbalance ----
    axs[1].plot(df["time_s"], df["front_rear_imbalance"], color="#9467bd", linewidth=0.8)
    axs[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axs[1].fill_between(df["time_s"], df["front_rear_imbalance"], 0,
                        where=df["front_rear_imbalance"] > 0,
                        alpha=0.25, color="#1f77b4", label="front heavier")
    axs[1].fill_between(df["time_s"], df["front_rear_imbalance"], 0,
                        where=df["front_rear_imbalance"] < 0,
                        alpha=0.25, color="#ff7f0e", label="rear heavier")
    mean_fr = df["front_rear_imbalance"].mean()
    axs[1].axhline(mean_fr, color="#9467bd", linestyle=":", linewidth=1.0,
                   label=f"mean={mean_fr:.4f}")
    axs[1].set_ylabel("Imbalance (0–1)", fontsize=9)
    axs[1].set_title("front_rear_imbalance  =  front_mean − rear_mean", fontsize=10)
    axs[1].legend(fontsize=8, loc="upper right")
    axs[1].grid(True, alpha=0.3)

    # ---- Diagonal A (M3+M4) vs Diagonal B (M1+M2) ----
    axs[2].plot(df["time_s"], df["diagonal_A_mean"], color="#2ca02c",
                linewidth=0.8, label="diagonal_A_mean  (M3 front_left + M4 rear_right)")
    axs[2].plot(df["time_s"], df["diagonal_B_mean"], color="#d62728",
                linewidth=0.8, label="diagonal_B_mean  (M1 front_right + M2 rear_left)")
    axs[2].set_ylabel("Command (0–1)", fontsize=9)
    axs[2].set_title("Diagonal A (M3+M4) vs Diagonal B (M1+M2)", fontsize=10)
    axs[2].legend(fontsize=8, loc="upper right")
    axs[2].set_ylim(-0.05, 1.05)
    axs[2].grid(True, alpha=0.3)

    # ---- Diagonal imbalance ----
    axs[3].plot(df["time_s"], df["diagonal_imbalance"], color="#8c564b", linewidth=0.8)
    axs[3].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axs[3].fill_between(df["time_s"], df["diagonal_imbalance"], 0,
                        where=df["diagonal_imbalance"] > 0,
                        alpha=0.25, color="#2ca02c", label="M3+M4 diagonal heavier")
    axs[3].fill_between(df["time_s"], df["diagonal_imbalance"], 0,
                        where=df["diagonal_imbalance"] < 0,
                        alpha=0.25, color="#d62728", label="M1+M2 diagonal heavier")
    mean_d = df["diagonal_imbalance"].mean()
    axs[3].axhline(mean_d, color="#8c564b", linestyle=":", linewidth=1.0,
                   label=f"mean={mean_d:.4f}")
    axs[3].set_ylabel("Imbalance (0–1)", fontsize=9)
    axs[3].set_title("diagonal_imbalance  =  diagonal_A_mean − diagonal_B_mean", fontsize=10)
    axs[3].legend(fontsize=8, loc="upper right")
    axs[3].grid(True, alpha=0.3)

    axs[-1].set_xlabel("Time (s) from start of log", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {output_plot}")
    plt.show()


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    # Extract
    df = extract_motor_commands(ULG_FILE)

    # Compute imbalance features
    df = compute_imbalances(df)

    # Save CSV — only the 6 useful columns (4 individual motors + 2 imbalances)
    if SAVE_CSV:
        cols_to_save = (
            ["timestamp", "time_s"]
            + list(MOTORS.values())
            + ["front_rear_imbalance", "diagonal_imbalance"]
        )
        df[cols_to_save].to_csv(OUTPUT_CSV, index=False)
        print(f"  CSV  saved → {OUTPUT_CSV}")
        print(f"  Columns : {cols_to_save}")

    # Plot individual motor commands
    plot_motor_commands(df, OUTPUT_PLOT)

    # Plot imbalance features
    plot_imbalances(df, OUTPUT_PLOT_IMBALANCE)
