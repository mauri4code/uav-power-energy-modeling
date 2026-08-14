"""
Plot power vs time — all flights on a single figure.

Color   → payload position + mass (7 unique groups)
Linestyle → trajectory_1 (solid) / trajectory_2 (dashed)
Each line is labelled with full flight metadata.

Output: ML/ml_data/plot_power_all_flights.png

Run: python plot_power_all_flights.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── PATHS ────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FLIGHTS_DIR = os.path.join(SCRIPT_DIR, "flights")
OUT_DIR     = os.path.join(SCRIPT_DIR, "ML", "ml_data")
os.makedirs(OUT_DIR, exist_ok=True)

# ── COLOUR MAP: (position_payload, payload_mass) → hex colour ────────────
GROUP_COLORS = {
    ("none",     0.00): "#6B7280",   # gray          – no payload
    ("front",    0.24): "#60A5FA",   # light blue    – front 0.24 kg
    ("front",    0.48): "#1D4ED8",   # dark blue     – front 0.48 kg
    ("diagonal", 0.24): "#FB923C",   # light orange  – diagonal 0.24 kg
    ("diagonal", 0.48): "#C2410C",   # dark orange   – diagonal 0.48 kg
    ("rear",     0.24): "#4ADE80",   # light green   – rear 0.24 kg
    ("rear",     0.48): "#166534",   # dark green    – rear 0.48 kg
}

LINESTYLES = {
    "trajectory_1": "-",
    "trajectory_2": "--",
}

# ── LOAD ─────────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(FLIGHTS_DIR, "*", "flight_resampled.csv")))
if not files:
    raise FileNotFoundError(f"No flight_resampled.csv found under {FLIGHTS_DIR}")

flights = []
for path in files:
    df = pd.read_csv(path)
    df["time_s"] = df["timestamp"] - df["timestamp"].iloc[0]
    flights.append(df)
    print(f"  Loaded {df['flight_id'].iloc[0]:4s}  "
          f"pos={df['position_payload'].iloc[0]:10s}  "
          f"mass={df['payload_mass'].iloc[0]:.2f} kg  "
          f"traj={df['trajectory'].iloc[0]}  "
          f"rows={len(df)}")

# ── PLOT ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor("#F8FAFC")
ax.set_facecolor("#F8FAFC")

for df in flights:
    fid      = df["flight_id"].iloc[0]
    pos      = df["position_payload"].iloc[0]
    mass     = df["payload_mass"].iloc[0]
    traj     = df["trajectory"].iloc[0]
    t        = df["time_s"].values
    p        = df["power"].values

    color = GROUP_COLORS.get((pos, round(mass, 2)), "#999999")
    ls    = LINESTYLES.get(traj, "-")

    # Smooth with a rolling mean for readability (window = 1 s = 20 samples @ 20 Hz)
    p_smooth = pd.Series(p).rolling(20, center=True, min_periods=1).mean().values

    traj_num = traj.replace("trajectory_", "T")
    mass_str = f"{mass:.2f} kg" if mass > 0 else "no payload"
    label    = f"{fid}  |  {pos:8s}  {mass_str}  {traj_num}"

    ax.plot(t, p_smooth,
            color=color, linestyle=ls, linewidth=1.4,
            alpha=0.85, label=label)

# ── AXES ─────────────────────────────────────────────────────────────────
ax.set_xlabel("Time (s) from flight start", fontsize=12)
ax.set_ylabel("Power (W)", fontsize=12)
ax.set_title("UAV Power Consumption — All Flights\n"
             "Color = payload position + mass  |  Line style = trajectory",
             fontsize=13, fontweight="bold", pad=14)
ax.grid(True, alpha=0.3, linestyle="--")
ax.tick_params(labelsize=10)

# ── LEGEND ───────────────────────────────────────────────────────────────
# Flight legend (auto from labels)
flight_legend = ax.legend(
    title="Flight  |  Position      Mass    Traj",
    title_fontsize=9,
    fontsize=8.5,
    loc="upper right",
    framealpha=0.92,
    edgecolor="#CBD5E1",
    ncol=1,
)
ax.add_artist(flight_legend)

# Extra legend: line style = trajectory
ls_handles = [
    mlines.Line2D([], [], color="black", linestyle="-",  linewidth=1.5, label="Trajectory 1"),
    mlines.Line2D([], [], color="black", linestyle="--", linewidth=1.5, label="Trajectory 2"),
]
ax.legend(handles=ls_handles, loc="upper left",
          fontsize=9, framealpha=0.92, edgecolor="#CBD5E1",
          title="Line style", title_fontsize=9)
ax.add_artist(flight_legend)  # re-add flight legend (legend() replaced it)

# ── COLOUR KEY (payload groups) ─────────────────────────────────────────
group_handles = [
    mlines.Line2D([], [], color=c, linewidth=4,
                  label=f"{pos}  {mass:.2f} kg" if mass > 0 else f"{pos}")
    for (pos, mass), c in GROUP_COLORS.items()
]
color_legend = ax.legend(
    handles=group_handles,
    title="Color = position + mass",
    title_fontsize=9,
    fontsize=8.5,
    loc="lower right",
    framealpha=0.92,
    edgecolor="#CBD5E1",
)
ax.add_artist(color_legend)

# Trajectory legend (top-left)
ax.legend(handles=ls_handles, loc="upper left",
          fontsize=9, framealpha=0.92, edgecolor="#CBD5E1",
          title="Line style", title_fontsize=9)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "plot_power_all_flights.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved → {out_path}")
