"""
Step 10 - Per-configuration AVERAGE motor heatmap.

For each payload configuration (none / front / rear / diagonal), averages the
cruise motor commands across all flights of that configuration and draws the
rotor-diagram heatmap. Also prints the % increase of each motor relative to the
unloaded (none) baseline.

Cruise = samples with power above the per-flight midpoint of the 5th/95th power
percentiles (same definition as the rest of the project).

Output (ML/ml_data/):
    heatmap_config_none.png / _front.png / _rear.png / _diagonal.png
    heatmap_config_grid.png   (2x2 panel, all four)

Run: python 10_config_heatmap.py
"""

import os, glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.cm import ScalarMappable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FLIGHTS_DIR = os.path.join(SCRIPT_DIR, "flights")
IMAGE_PATH = os.path.join(SCRIPT_DIR, "HEAT_MAP", "Rotors_poss.jpeg")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "ML", "ml_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hough-measured rotor positions (same as HEAT_MAP/make_heatmap_animation.py)
MOTOR_LAYOUT = {
    "motor_1_front_right": {"pos": (0.7960, 0.2198), "label": "M1\nfront-right"},
    "motor_2_rear_left":   {"pos": (0.1696, 0.8366), "label": "M2\nrear-left"},
    "motor_3_front_left":  {"pos": (0.1685, 0.2223), "label": "M3\nfront-left"},
    "motor_4_rear_right":  {"pos": (0.7810, 0.8347), "label": "M4\nrear-right"},
}
MOTORS = list(MOTOR_LAYOUT.keys())
CIRCLE_RADIUS_FRAC = 0.114
CMAP = plt.cm.jet          # SolidWorks-style sim legend (HTML look)
COLOR_GAMMA = 0.6          # warps the scale so warm colours trigger sooner
ORDER = ["none", "front", "rear", "diagonal"]


def cruise_means():
    """Per-flight cruise motor means, grouped by configuration (equal weight per flight)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(FLIGHTS_DIR, "F*", "flight_resampled.csv"))):
        d = pd.read_csv(f)
        thr = 0.5 * (np.percentile(d["power"], 5) + np.percentile(d["power"], 95))
        cr = d[d["power"] > thr]
        rows.append({"pos": str(d["position_payload"].iloc[0]),
                     **{m: cr[m].mean() for m in MOTORS}})
    df = pd.DataFrame(rows)
    by = df.groupby("pos")[MOTORS].mean()
    return by


def _norm(vmin, vmax):
    return PowerNorm(gamma=COLOR_GAMMA, vmin=vmin, vmax=vmax)


def draw(ax, img, means, title, vmin, vmax, base=None):
    h, w = img.shape[:2]
    r = CIRCLE_RADIUS_FRAC * w
    ax.imshow(img); ax.axis("off")
    mapper = ScalarMappable(norm=_norm(vmin, vmax), cmap=CMAP)
    bbox = dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.78, edgecolor="none")
    for col, lay in MOTOR_LAYOUT.items():
        cx, cy = lay["pos"][0] * w, lay["pos"][1] * h
        v = means[col]
        ax.add_patch(plt.Circle((cx, cy), r, color=mapper.to_rgba(float(np.clip(v, vmin, vmax))),
                                 alpha=0.78, zorder=3))
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, edgecolor="black", lw=1.5, zorder=4))
        ax.text(cx, cy - r * 1.32, lay["label"], ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", zorder=5)
        txt = f"{v:.3f}"
        if base is not None:
            pct = 100 * (v - base[col]) / base[col]
            txt += f"\n{pct:+.1f}%"
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=9,
                fontweight="bold", zorder=5, bbox=bbox)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=6)


def main():
    by = cruise_means()
    base = by.loc["none"]
    print("Per-config cruise motor means:\n", by.round(4))
    img = np.array(Image.open(IMAGE_PATH).convert("RGB"))
    vals = by.values.flatten()
    m = (vals.max() - vals.min()) * 0.05
    vmin, vmax = vals.min() - m, vals.max() + m

    # individual figures
    for cfg in ORDER:
        fig, ax = plt.subplots(figsize=(5.5, 6))
        b = None if cfg == "none" else base
        draw(ax, img, by.loc[cfg], f"Payload: {cfg}", vmin, vmax, base=b)
        mp = ScalarMappable(norm=_norm(vmin, vmax), cmap=CMAP); mp.set_array([])
        cb = fig.colorbar(mp, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("Mean motor command (0-1)", fontsize=8)
        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, f"heatmap_config_{cfg}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print("saved", out)

    # 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(11, 12.4))
    for ax, cfg in zip(axes.flat, ORDER):
        b = None if cfg == "none" else base
        draw(ax, img, by.loc[cfg], f"Payload: {cfg}", vmin, vmax, base=b)
    mp = ScalarMappable(norm=_norm(vmin, vmax), cmap=CMAP); mp.set_array([])
    cb = fig.colorbar(mp, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("Mean motor command (0-1)", fontsize=9)
    fig.suptitle("Average motor usage by payload configuration (cruise)",
                 fontsize=13, fontweight="bold", y=1.00)
    out = os.path.join(OUTPUT_DIR, "heatmap_config_grid.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print("saved", out)


if __name__ == "__main__":
    main()
