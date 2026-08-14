"""
Step 9 – Motor usage heatmap overlaid on UAV rotor diagram.

Reads flight_resampled.csv for one or all flights, computes mean motor
command per rotor, and draws a colour-coded heatmap on top of Rotors_poss.jpeg
so you can immediately see which motors are working hardest.

Usage
-----
    python 09_motor_heatmap.py              # all flights in a grid
    python 09_motor_heatmap.py F03          # single flight
    python 09_motor_heatmap.py F03 F05      # two specific flights

Output
------
    ML/ml_data/plot_motor_heatmap_<id>.png  (one per flight shown)
    ML/ml_data/plot_motor_heatmap_all.png   (grid panel, only when >1 flight)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# =====================================================
# PATHS
# =====================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
FLIGHTS_DIR  = os.path.join(SCRIPT_DIR, "flights")
IMAGE_PATH   = os.path.join(SCRIPT_DIR, "Rotors_poss.jpeg")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "ML", "ml_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# MOTOR LAYOUT
# Motor positions as fractions of image (width, height).
# Measured from Rotors_poss.jpeg:
#   M3 front-left  → top-left
#   M1 front-right → top-right
#   M2 rear-left   → bottom-left
#   M4 rear-right  → bottom-right
# =====================================================
MOTOR_LAYOUT = {
    "motor_1_front_right": {"pos": (0.845, 0.185), "label": "M1\nfront-right"},
    "motor_2_rear_left":   {"pos": (0.145, 0.830), "label": "M2\nrear-left"},
    "motor_3_front_left":  {"pos": (0.155, 0.185), "label": "M3\nfront-left"},
    "motor_4_rear_right":  {"pos": (0.830, 0.830), "label": "M4\nrear-right"},
}

# Radius of the heatmap circle as a fraction of image width
CIRCLE_RADIUS_FRAC = 0.115

# Colour map: blue (low) → red (high)
CMAP = plt.cm.RdYlBu_r

# =====================================================
# HELPER: compute per-motor means for one flight CSV
# =====================================================
def load_flight_means(flight_id: str) -> dict | None:
    csv_path = os.path.join(FLIGHTS_DIR, flight_id, "flight_resampled.csv")
    if not os.path.exists(csv_path):
        print(f"  [WARNING] Not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    motor_cols = list(MOTOR_LAYOUT.keys())
    missing = [c for c in motor_cols if c not in df.columns]
    if missing:
        print(f"  [WARNING] Missing columns in {flight_id}: {missing}")
        return None
    means = {col: df[col].mean() for col in motor_cols}
    meta  = {
        "flight_id":        df["flight_id"].iloc[0],
        "position_payload": df["position_payload"].iloc[0],
        "payload_mass":     df["payload_mass"].iloc[0],
        "trajectory":       df["trajectory"].iloc[0],
        "n_samples":        len(df),
    }
    return {"means": means, "meta": meta}


# =====================================================
# HELPER: draw one heatmap panel onto an existing axes
# =====================================================
def draw_heatmap(ax, img: np.ndarray, means: dict, meta: dict,
                 vmin: float, vmax: float) -> None:
    """Overlay motor heatmap circles on the rotor diagram image."""
    h, w = img.shape[:2]
    radius_px = CIRCLE_RADIUS_FRAC * w

    ax.imshow(img)
    ax.axis("off")

    norm   = Normalize(vmin=vmin, vmax=vmax)
    mapper = ScalarMappable(norm=norm, cmap=CMAP)

    for col, layout in MOTOR_LAYOUT.items():
        xf, yf   = layout["pos"]
        cx, cy   = xf * w, yf * h
        value    = means[col]
        rgba     = mapper.to_rgba(value)
        # semi-transparent filled circle
        circle = plt.Circle(
            (cx, cy), radius_px,
            color=rgba, alpha=0.72, zorder=3
        )
        ax.add_patch(circle)
        # thin black outline
        outline = plt.Circle(
            (cx, cy), radius_px,
            fill=False, edgecolor="black", linewidth=1.5, zorder=4
        )
        ax.add_patch(outline)
        # text: label + mean value
        ax.text(
            cx, cy - radius_px * 0.18,
            layout["label"],
            ha="center", va="center",
            fontsize=7.5, fontweight="bold",
            color="black", zorder=5
        )
        ax.text(
            cx, cy + radius_px * 0.52,
            f"{value:.3f}",
            ha="center", va="center",
            fontsize=8.5, fontweight="bold",
            color="black", zorder=5
        )

    # Title with flight info
    ax.set_title(
        f"{meta['flight_id']}  |  payload: {meta['position_payload']}"
        f"  ({meta['payload_mass']:.2f} kg)\n"
        f"traj: {meta['trajectory']}  |  {meta['n_samples']} samples",
        fontsize=9, pad=6
    )


# =====================================================
# MAIN: single-panel save per flight
# =====================================================
def save_single(flight_data: dict, img: np.ndarray,
                vmin: float, vmax: float) -> None:
    fid = flight_data["meta"]["flight_id"]
    fig, ax = plt.subplots(figsize=(5.5, 6))
    draw_heatmap(ax, img, flight_data["means"], flight_data["meta"], vmin, vmax)

    # Colorbar
    norm   = Normalize(vmin=vmin, vmax=vmax)
    mapper = ScalarMappable(norm=norm, cmap=CMAP)
    mapper.set_array([])
    cbar = fig.colorbar(mapper, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean motor command (0–1)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle("Motor Usage Heatmap", fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"plot_motor_heatmap_{fid}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# =====================================================
# MAIN: multi-panel grid (all flights)
# =====================================================
def save_grid(all_data: list, img: np.ndarray,
              vmin: float, vmax: float) -> None:
    n = len(all_data)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols,
                             figsize=(5.5 * cols, 6.2 * rows),
                             squeeze=False)

    for i, fd in enumerate(all_data):
        ax = axes[i // cols][i % cols]
        draw_heatmap(ax, img, fd["means"], fd["meta"], vmin, vmax)

    # Hide unused axes
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    # Shared colorbar on the right
    norm   = Normalize(vmin=vmin, vmax=vmax)
    mapper = ScalarMappable(norm=norm, cmap=CMAP)
    mapper.set_array([])
    cbar = fig.colorbar(mapper, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Mean motor command (0–1)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Motor Usage Heatmap — All Flights",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "plot_motor_heatmap_all.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# =====================================================
# ENTRY POINT
# =====================================================
def main():
    # ---- Determine which flights to process ----
    args = sys.argv[1:]
    if args:
        flight_ids = args
    else:
        # Auto-discover all available flights
        dirs = sorted(glob.glob(os.path.join(FLIGHTS_DIR, "F*")))
        flight_ids = [os.path.basename(d) for d in dirs]
        if not flight_ids:
            print("No flight folders found under flights/. "
                  "Run the pipeline first.")
            sys.exit(1)

    print(f"\nMotor heatmap — flights: {flight_ids}")

    # ---- Load image ----
    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: Image not found → {IMAGE_PATH}")
        sys.exit(1)
    img = np.array(Image.open(IMAGE_PATH).convert("RGB"))
    print(f"  Image loaded: {img.shape[1]}×{img.shape[0]} px")

    # ---- Load flight data ----
    all_data = []
    for fid in flight_ids:
        result = load_flight_means(fid)
        if result:
            all_data.append(result)
            m = result["meta"]
            print(f"  {fid}: {m['position_payload']:10s}  "
                  + "  ".join(f"{k}={v:.3f}"
                              for k, v in result["means"].items()))

    if not all_data:
        print("No valid flight data found.")
        sys.exit(1)

    # ---- Global colour scale (consistent across all panels) ----
    all_values = [v for fd in all_data for v in fd["means"].values()]
    vmin = min(all_values)
    vmax = max(all_values)
    # Expand range slightly so extreme colours are visible
    margin = (vmax - vmin) * 0.05
    vmin  -= margin
    vmax  += margin
    print(f"\n  Colour scale: {vmin:.3f} → {vmax:.3f}")

    # ---- Save individual plots ----
    for fd in all_data:
        save_single(fd, img, vmin, vmax)

    # ---- Save grid if more than one flight ----
    if len(all_data) > 1:
        save_grid(all_data, img, vmin, vmax)

    print("\nDone.")


if __name__ == "__main__":
    main()
