"""
make_heatmap_animation.py — animated motor heatmap over the mission window.

Generates two self-contained HTML files (no build step, no external
dependencies at view time) that animate the four PX4 rotor commands over
the airframe photo (Rotors_poss.jpeg):

    motor_animation.html       one flight at a time, with a flight selector
    motor_animation_grid.html  all 14 flights animating together on a
                                shared, normalised (0-100%) clock

Why a 12-second margin around the "mission window"
----------------------------------------------------
Each raw recording contains a stretch of ground idle before the motors are
armed. Its length is arbitrary — it depends only on when the operator
started the ROS bag, and ranges from 0 to 59 s across the 14 flights in
this dataset. Animating that dead time is pointless and makes flights hard
to compare, so instead we detect the actual mission (arm -> flight ->
land/disarm) and animate only that, padded by a fixed margin on each side.

The margin is derived, not guessed: for every flight we compute a power
threshold `thr = 0.5 * (P5 + P95)` (it separates ~25 W idle from ~700 W
airborne draw, landing around 340-415 W depending on payload) and find the
first sample where `power > thr`. That crossing is a proxy for "airborne",
which lags "armed" by however long the operator held on the ground before
takeoff. Across all 14 flights the largest such arm-to-takeoff gap is
11.9 s (see CODEX_PROMPT.md). Padding the window by 12 s on both sides is
therefore the smallest symmetric margin that is guaranteed to include the
arming transient (motors spinning at idle) on every flight, while adding a
matching 12 s tail so landing and shutdown are visible too. Flights whose
recording starts mid-flight (F03-F06 — no idle stretch was captured) simply
clip to the start of the data; there is nothing before it to show.

Usage
-----
    python make_heatmap_animation.py

Reads
-----
    ../flights/F01/flight_resampled.csv ... F14  (this repo's convention:
        no F02... actually F01..F14 exist, see CODEX_PROMPT.md)
    Rotors_poss.jpeg

Writes
------
    motor_animation.html
    motor_animation_grid.html
"""

import base64
import glob
import io
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import PowerNorm

# =====================================================
# PATHS
# =====================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# this copy lives one level deeper than the original (HEAT_MAP/with_power_plot/),
# so the flights data is two levels up instead of one
FLIGHTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "flights"))
IMAGE_PATH = os.path.join(SCRIPT_DIR, "Rotors_poss.jpeg")
OUT_SINGLE = os.path.join(SCRIPT_DIR, "motor_animation.html")
OUT_GRID = os.path.join(SCRIPT_DIR, "motor_animation_grid.html")

# =====================================================
# ROTOR LAYOUT
# 09_motor_heatmap.py's constants were a rough eyeball estimate; they sat
# visibly off-centre from the drawn rotor rings once overlaid at full
# resolution. Re-measured here with a small Hough-circle search (find the
# (cx, cy, r) that maximises dark-ring coverage) against Rotors_poss.jpeg
# — see scratch analysis in this PR/session. Coverage was 1.000 for all
# four rotors, i.e. an exact fit to the drawn ring.
# =====================================================
MOTOR_LAYOUT = {
    "motor_1_front_right": {"pos": (0.7960, 0.2198), "label": "M1 front-right"},
    "motor_2_rear_left":   {"pos": (0.1696, 0.8366), "label": "M2 rear-left"},
    "motor_3_front_left":  {"pos": (0.1685, 0.2223), "label": "M3 front-left"},
    "motor_4_rear_right":  {"pos": (0.7810, 0.8347), "label": "M4 rear-right"},
}
MOTOR_ORDER = list(MOTOR_LAYOUT.keys())
CIRCLE_RADIUS_FRAC = 0.114

# =====================================================
# TUNABLES
# =====================================================
WINDOW_MARGIN_S = 12.0          # see docstring above
# 20 Hz matches the source PX4 log rate exactly (no decimation at all) --
# denser control points for the JS interpolator to work from, so the
# interpolated curve hugs the true motor-command trajectory instead of
# smoothing over real detail between samples that used to be 0.1s apart.
SINGLE_DECIMATION_HZ = 20.0     # frames/s of flight-time for the single view
GRID_NORM_STEPS = 400           # samples across the normalised 0-100% clock
# Both HTML players interpolate continuously between data samples (see the JS
# render loop) rather than snapping to the next sample, so playback is smooth
# regardless of this decimation rate; the JS side also eases each segment
# with smoothstep (zero velocity at each sample) instead of straight linear
# interpolation, removing the small rate-of-change "kink" linear would leave
# at every sample boundary. PLAYBACK_TIME_SCALE additionally slows the
# default wall-clock playback down (2.0 = takes twice as long to watch),
# independent of the Speed selector which multiplies on top of this baseline.
PLAYBACK_TIME_SCALE = 2.0
COLOR_LUT_STOPS = 1024          # was 256; finer table removes any visible colour banding
# SolidWorks-style simulation legend: blue (low) -> cyan -> green -> yellow -> red (high)
CMAP = plt.cm.jet
# Colour mapping is warped by t -> t**COLOR_GAMMA (t = normalised value in [0,1])
# before indexing into the colormap. gamma < 1 pushes the warm half of the
# scale (yellow/orange/red) to trigger at lower raw values, since motor
# commands rarely sit near the very top of the vmin-vmax range in practice.
# vmin/vmax themselves are untouched, so the printed numbers and the stated
# legend range stay exact — only the colour a given value renders as shifts.
COLOR_GAMMA = 0.6


def extract_change_points(t_rel, values):
    """Collapse a forward-filled staircase back to its true update points.

    flight_resampled.csv holds motor commands (and power) on a uniform 20 Hz
    grid, but 04_resampling.py builds that column with a *forward-fill*
    (`forwardfill_to_grid`), not interpolation: PX4 only reports a new motor
    command every ~0.1s (raw ~10 Hz) and ROS battery telemetry only every
    ~2s (~0.5 Hz), so most adjacent 20 Hz rows are exact repeats, with the
    real change compressed into a single 1/20s grid step. Interpolating
    against that grid directly (as the first version of this script did)
    reproduces that same instant-jump-then-hold shape at render time, which
    reads as flickery/"digital". Extracting just the rows where the value
    actually changes and interpolating between *those* reconstructs a ramp
    that spans the true ~0.1s / ~2s update interval instead.
    """
    values = np.asarray(values, dtype=float)
    keep = np.zeros(len(values), dtype=bool)
    keep[0] = True
    keep[1:] = values[1:] != values[:-1]
    return t_rel[keep], values[keep]


# =====================================================
# STEP 1 — load one flight and locate its mission window
# =====================================================
def load_flight(flight_id: str):
    csv_path = os.path.join(FLIGHTS_DIR, flight_id, "flight_resampled.csv")
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    missing = [c for c in MOTOR_ORDER if c not in df.columns]
    if missing or "power" not in df.columns:
        print(f"  [WARNING] {flight_id}: missing columns {missing}, skipping")
        return None

    df = df.sort_values("timestamp").reset_index(drop=True)
    t_rel = (df["timestamp"] - df["timestamp"].iloc[0]).to_numpy()
    power = df["power"].to_numpy()

    thr = 0.5 * (np.percentile(power, 5) + np.percentile(power, 95))
    above = np.where(power > thr)[0]
    if len(above) == 0:
        print(f"  [WARNING] {flight_id}: power never exceeds threshold, skipping")
        return None
    first_above = float(t_rel[above[0]])
    last_above = float(t_rel[above[-1]])

    window_start = max(0.0, first_above - WINDOW_MARGIN_S)
    window_end = min(float(t_rel[-1]), last_above + WINDOW_MARGIN_S)

    motor_keypoints = {c: extract_change_points(t_rel, df[c].to_numpy()) for c in MOTOR_ORDER}
    power_keypoints = extract_change_points(t_rel, power)

    return {
        "flight_id": flight_id,
        "df": df,
        "t_rel": t_rel,
        "power": power,
        "motor_keypoints": motor_keypoints,
        "power_keypoints": power_keypoints,
        "thr": thr,
        "first_above": first_above,
        "last_above": last_above,
        "window_start": window_start,
        "window_end": window_end,
        "meta": {
            "payload_mass": float(df["payload_mass"].iloc[0]),
            "position_payload": str(df["position_payload"].iloc[0]),
            "trajectory": str(df["trajectory"].iloc[0]),
        },
    }


def phase_at(flight, t_abs: float) -> int:
    """0 = arming, 1 = flight, 2 = landing/shutdown (relative to recording start)."""
    if t_abs < flight["first_above"]:
        return 0
    if t_abs <= flight["last_above"]:
        return 1
    return 2


# =====================================================
# STEP 2 — colour scale (fixed globally, computed once)
# =====================================================
def compute_global_color_scale(flights):
    in_flight_vals = []
    for fl in flights:
        mask = fl["power"] > fl["thr"]
        for col in MOTOR_ORDER:
            in_flight_vals.append(fl["df"][col].to_numpy()[mask])
    allv = np.concatenate(in_flight_vals)
    vmin = float(np.percentile(allv, 1))
    vmax = float(np.percentile(allv, 99))
    return vmin, vmax


def build_color_lut(n=COLOR_LUT_STOPS):
    return [
        "#%02x%02x%02x" % tuple(int(255 * c) for c in CMAP(i / (n - 1))[:3])
        for i in range(n)
    ]


# =====================================================
# STEP 3 — per-flight frames for the SINGLE-flight view
# decimated to SINGLE_DECIMATION_HZ, values rounded for size
# =====================================================
def build_single_frames(flight):
    ws, we = flight["window_start"], flight["window_end"]
    duration = we - ws
    n_frames = max(2, int(round(duration * SINGLE_DECIMATION_HZ)) + 1)
    sample_times = ws + np.linspace(0.0, duration, n_frames)

    # Interpolate against the true update points (see extract_change_points),
    # not the forward-filled 20 Hz grid -- reconstructs the real ramp shape
    # instead of the grid's instant-jump-then-hold artifact.
    pt, pv = flight["power_keypoints"]
    power_i = np.interp(sample_times, pt, pv)
    motor_i = {}
    for c in MOTOR_ORDER:
        kt, kv = flight["motor_keypoints"][c]
        motor_i[c] = np.interp(sample_times, kt, kv)

    frames = []
    for k, t_abs in enumerate(sample_times):
        elapsed = round(t_abs - ws, 2)
        p = round(float(power_i[k]), 1)
        ph = phase_at(flight, t_abs)
        m = [round(float(motor_i[c][k]), 3) for c in MOTOR_ORDER]
        frames.append([elapsed] + [p, ph] + m)
    return frames


# =====================================================
# STEP 4 — frames for the GRID view: every flight resampled
# onto a shared 0-100% clock (each flight normalised to its
# own mission-window duration) so all 14 stay in step.
# =====================================================
def build_grid_frames(flight, n_steps=GRID_NORM_STEPS):
    ws, we = flight["window_start"], flight["window_end"]
    duration = we - ws
    norm = np.linspace(0.0, 1.0, n_steps)
    sample_times = ws + norm * duration

    # Same true-update-point interpolation as build_single_frames (see
    # extract_change_points) instead of the forward-filled 20 Hz grid.
    pt, pv = flight["power_keypoints"]
    power_i = np.interp(sample_times, pt, pv)
    motor_i = {}
    for c in MOTOR_ORDER:
        kt, kv = flight["motor_keypoints"][c]
        motor_i[c] = np.interp(sample_times, kt, kv)

    frames = []
    for k, t_abs in enumerate(sample_times):
        elapsed = round(t_abs - ws, 2)
        p = round(float(power_i[k]), 1)
        ph = phase_at(flight, t_abs)
        m = [round(float(motor_i[c][k]), 3) for c in MOTOR_ORDER]
        frames.append([elapsed] + [p, ph] + m)
    return frames, duration


# =====================================================
# STEP 5 — static "average heatmap" image, one panel per flight
# Mean of each rotor's command over the mission window (not the whole
# recording, so idle-time dilution doesn't vary flight to flight). Uses the
# exact same jet + gamma colour mapping as the animations, via
# PowerNorm(gamma=...), so the static image and the animated one always
# agree on what a given value looks like.
# =====================================================
def build_average_heatmap_png(flights, vmin, vmax, gamma):
    img = np.array(Image.open(IMAGE_PATH).convert("RGB"))
    h, w = img.shape[:2]
    radius_px = CIRCLE_RADIUS_FRAC * w

    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    mapper = ScalarMappable(norm=norm, cmap=CMAP)

    n = len(flights)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.3 * rows), squeeze=False)
    axes = axes.ravel()

    for i, fl in enumerate(flights):
        ax = axes[i]
        mask = (fl["t_rel"] >= fl["window_start"]) & (fl["t_rel"] <= fl["window_end"])
        means = {c: float(fl["df"][c].to_numpy()[mask].mean()) for c in MOTOR_ORDER}

        ax.imshow(img)
        ax.axis("off")
        for col in MOTOR_ORDER:
            layout = MOTOR_LAYOUT[col]
            cx, cy = layout["pos"][0] * w, layout["pos"][1] * h
            val = means[col]
            rgba = mapper.to_rgba(float(np.clip(val, vmin, vmax)))
            ax.add_patch(plt.Circle((cx, cy), radius_px, color=rgba, alpha=0.78, zorder=3))
            ax.add_patch(plt.Circle((cx, cy), radius_px, fill=False, edgecolor="black",
                                     linewidth=1.4, zorder=4))
            ax.text(cx, cy + radius_px * 0.42, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="black", zorder=5)

        ax.set_title(
            f"{fl['flight_id']}  |  {fl['meta']['payload_mass']:.2f} kg — {fl['meta']['position_payload']}\n"
            f"{fl['meta']['trajectory']}",
            fontsize=8.5, pad=4,
        )

    for j in range(n, rows * cols):
        axes[j].axis("off")

    mapper.set_array([])
    cbar = fig.colorbar(mapper, ax=list(axes), fraction=0.02, pad=0.015)
    cbar.set_label(f"Mean motor command over mission window (gamma={gamma})", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Average motor command per flight — mission window", fontsize=13, fontweight="bold", y=1.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


# =====================================================
# HTML TEMPLATES
# =====================================================
SHARED_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 16px; background: #14161a; color: #eaeaea;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
h1 { font-size: 1.15rem; font-weight: 600; margin: 0 0 12px; }
h2 { font-size: 1rem; font-weight: 600; margin: 28px 0 6px; }
.avg-section img { max-width: 100%; border-radius: 10px; border: 1px solid #2c3038; display: block; }
.toolbar {
  display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
  background: #1e2126; border: 1px solid #2c3038; border-radius: 10px;
  padding: 10px 14px; margin-bottom: 14px;
}
.toolbar label { font-size: 0.8rem; color: #9aa2ad; margin-right: 4px; }
select, button {
  background: #262a31; color: #eaeaea; border: 1px solid #3a4048;
  border-radius: 6px; padding: 6px 10px; font-size: 0.85rem; cursor: pointer;
}
button:hover, select:hover { background: #30353d; }
button.primary { background: #3a6ea5; border-color: #3a6ea5; }
button.primary:hover { background: #4a7eb5; }
input[type=range] { flex: 1 1 200px; accent-color: #3a6ea5; }
.note { font-size: 0.72rem; color: #7c8590; }
.legend { font-size: 0.75rem; color: #9aa2ad; margin-top: 6px; }
"""

SINGLE_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Motor heatmap — single flight</title>
<style>
__SHARED_CSS__
.stage { display: flex; gap: 18px; flex-wrap: wrap; }
.canvas-wrap { position: relative; border-radius: 10px; overflow: hidden; background: #000; }
canvas#stage { display: block; max-width: 100%; }
.labels {
  display: grid; grid-template-columns: repeat(2, minmax(160px, 1fr));
  gap: 8px 20px; background: #1e2126; border: 1px solid #2c3038;
  border-radius: 10px; padding: 14px 18px; min-width: 300px; align-content: start;
}
.labels .item .k { font-size: 0.7rem; color: #7c8590; text-transform: uppercase; letter-spacing: .04em; }
.labels .item .v { font-size: 1.05rem; font-weight: 600; }
.phase-arming { color: #e0b400; }
.phase-flight { color: #4caf50; }
.phase-landing { color: #4aa3df; }
.colorbar-wrap { margin-top: 14px; }
.colorbar { height: 14px; border-radius: 4px; border: 1px solid #3a4048; }
.colorbar-ticks { display: flex; justify-content: space-between; font-size: 0.72rem; color: #9aa2ad; margin-top: 3px; }
.chart-wrap {
  margin-top: 18px; background: #1e2126; border: 1px solid #2c3038;
  border-radius: 10px; padding: 14px 18px;
}
.chart-wrap canvas { width: 100%; height: 260px; display: block; border-radius: 6px; }
.chart-legend { display: flex; flex-wrap: wrap; gap: 4px 16px; margin-top: 8px; font-size: 0.72rem; color: #cfd4da; }
.chart-legend .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: middle; }
</style>
</head>
<body>
<h1>Animated motor heatmap — single flight</h1>

<div class="toolbar">
  <label for="flightSelect">Flight</label>
  <select id="flightSelect"></select>

  <button id="playBtn" class="primary">&#9654; Play</button>

  <label for="speedSelect">Speed</label>
  <select id="speedSelect">
    <option value="1">1x</option>
    <option value="2">2x</option>
    <option value="5">5x</option>
  </select>

  <input type="range" id="scrub" min="0" max="1000" value="0" step="1">
  <span class="note" id="decimationNote"></span>
</div>

<div class="stage">
  <div class="canvas-wrap">
    <canvas id="stage" width="843" height="855"></canvas>
  </div>

  <div class="labels">
    <div class="item"><div class="k">Flight</div><div class="v" id="lblFlight">-</div></div>
    <div class="item"><div class="k">Trajectory</div><div class="v" id="lblTraj">-</div></div>
    <div class="item"><div class="k">Payload</div><div class="v" id="lblPayload">-</div></div>
    <div class="item"><div class="k">Phase</div><div class="v" id="lblPhase">-</div></div>
    <div class="item"><div class="k">Elapsed time</div><div class="v" id="lblElapsed">-</div></div>
    <div class="item"><div class="k">Power</div><div class="v" id="lblPower">-</div></div>

    <div class="colorbar-wrap" style="grid-column: 1 / -1;">
      <div class="k">Motor command colour scale</div>
      <canvas class="colorbar" id="colorbar" width="300" height="14"></canvas>
      <div class="colorbar-ticks"><span id="cbMin"></span><span id="cbMax"></span></div>
      <div class="legend" id="cbLegend"></div>
    </div>
  </div>
</div>

<div class="chart-wrap">
  <div class="k">Power &amp; motor commands over time — how the motors react, then power follows</div>
  <canvas id="chart" width="1200" height="260"></canvas>
  <div class="chart-legend" id="chartLegend"></div>
</div>

<script>
const IMAGE_B64 = "__IMAGE_B64__";
const MOTOR_LAYOUT = __MOTOR_LAYOUT_JSON__;
const MOTOR_ORDER = __MOTOR_ORDER_JSON__;
const CIRCLE_RADIUS_FRAC = __CIRCLE_RADIUS_FRAC__;
const COLOR_SCALE = __COLOR_SCALE_JSON__;
const COLOR_LUT = __COLOR_LUT_JSON__;
const COLOR_GAMMA = __COLOR_GAMMA__;
const DECIMATION_HZ = __DECIMATION_HZ__;
const TIME_SCALE = __TIME_SCALE__;
const POWER_MAX = __POWER_MAX__;
const FLIGHTS = __FLIGHTS_JSON__;
const PHASE_NAMES = ["arming", "flight", "landing/shutdown"];
const PHASE_CLASS = ["phase-arming", "phase-flight", "phase-landing"];
const MOTOR_LINE_COLORS = { motor_1_front_right: "#ff6b6b", motor_2_rear_left: "#4dabf7", motor_3_front_left: "#51cf66", motor_4_rear_right: "#f7c948" };
const MOTOR_SHORT = { motor_1_front_right: "M1", motor_2_rear_left: "M2", motor_3_front_left: "M3", motor_4_rear_right: "M4" };
const PHASE_BAND_COLOR = ["#e0b400", "#4caf50", "#4aa3df"]; // arming, flight, landing

// ---- image ----
const img = new Image();
img.src = IMAGE_B64;

const canvas = document.getElementById("stage");
const ctx = canvas.getContext("2d");

// Warping t by **COLOR_GAMMA (gamma < 1) pushes the warm end of the scale
// (yellow/orange/red) to trigger at lower raw values -- exaggerates red
// without moving vmin/vmax, so printed numbers and the legend stay exact.
function lutColor(value) {
  let t = (value - COLOR_SCALE.vmin) / (COLOR_SCALE.vmax - COLOR_SCALE.vmin);
  t = Math.max(0, Math.min(1, t));
  t = Math.pow(t, COLOR_GAMMA);
  const idx = Math.round(t * (COLOR_LUT.length - 1));
  return COLOR_LUT[idx];
}

function drawFrame(frame) {
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);

  const radius = CIRCLE_RADIUS_FRAC * w;
  const [elapsed, power, phase, m1, m2, m3, m4] = frame;
  const values = { motor_1_front_right: m1, motor_2_rear_left: m2, motor_3_front_left: m3, motor_4_rear_right: m4 };

  for (const col of MOTOR_ORDER) {
    const layout = MOTOR_LAYOUT[col];
    const cx = layout.pos[0] * w, cy = layout.pos[1] * h;
    const val = values[col];
    const color = lutColor(val);

    ctx.globalAlpha = 0.72;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();

    ctx.globalAlpha = 1;
    ctx.lineWidth = 2;
    ctx.strokeStyle = "black";
    ctx.stroke();

    ctx.fillStyle = "black";
    ctx.font = "bold 14px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(layout.label, cx, cy - radius * 0.32);
    ctx.font = "bold 22px sans-serif";
    ctx.fillText(val.toFixed(2), cx, cy + radius * 0.28);
  }

  document.getElementById("lblElapsed").textContent = elapsed.toFixed(2) + " s";
  document.getElementById("lblPower").textContent = power.toFixed(1) + " W";
  const phaseEl = document.getElementById("lblPhase");
  phaseEl.textContent = PHASE_NAMES[phase];
  phaseEl.className = "v " + PHASE_CLASS[phase];
}

// ---- colour bar (drawn once, scale is fixed globally) ----
// Sampled by raw value (not by raw LUT index) so the bar shows the same
// gamma-warped mapping the circles use -- what you see here is what you get.
function drawColorbar() {
  const cb = document.getElementById("colorbar");
  const cctx = cb.getContext("2d");
  const grad = cctx.createLinearGradient(0, 0, cb.width, 0);
  const n = 128;
  for (let i = 0; i < n; i++) {
    const frac = i / (n - 1);
    const value = COLOR_SCALE.vmin + frac * (COLOR_SCALE.vmax - COLOR_SCALE.vmin);
    grad.addColorStop(frac, lutColor(value));
  }
  cctx.fillStyle = grad;
  cctx.fillRect(0, 0, cb.width, cb.height);
  document.getElementById("cbMin").textContent = COLOR_SCALE.vmin.toFixed(3);
  document.getElementById("cbMax").textContent = COLOR_SCALE.vmax.toFixed(3);
  document.getElementById("cbLegend").textContent =
    "Fixed across all flights: 1st-99th percentile of in-flight motor commands. Colour scale is gamma-warped (gamma=" + COLOR_GAMMA +
    ") to bring out red at more moderate values. Values outside the range are clipped in colour (numeric label still exact).";
}

// ---- power/motor-over-time chart ----
// Static series (phase bands, gridlines, power + motor lines) are drawn
// once per flight onto an offscreen canvas; each render just blits that
// image back and draws a moving playhead line + value dots on top, so a
// full redraw of ~thousands of points never happens per animation frame.
const chart = document.getElementById("chart");
const chartCtx = chart.getContext("2d");
const chartBase = document.createElement("canvas");
chartBase.width = chart.width;
chartBase.height = chart.height;
const chartBaseCtx = chartBase.getContext("2d");
const CHART_MARGIN = { left: 50, right: 44, top: 14, bottom: 26 };

function chartX(t, dur) {
  const pw = chart.width - CHART_MARGIN.left - CHART_MARGIN.right;
  return CHART_MARGIN.left + (dur > 0 ? (t / dur) * pw : 0);
}
function chartYPower(p) {
  const ph = chart.height - CHART_MARGIN.top - CHART_MARGIN.bottom;
  return chart.height - CHART_MARGIN.bottom - Math.max(0, Math.min(p / POWER_MAX, 1)) * ph;
}
function chartYMotor(m) {
  const ph = chart.height - CHART_MARGIN.top - CHART_MARGIN.bottom;
  return chart.height - CHART_MARGIN.bottom - Math.max(0, Math.min(m, 1)) * ph;
}

function buildChartBase(fl) {
  const c = chartBaseCtx;
  const W = chart.width, H = chart.height;
  c.clearRect(0, 0, W, H);
  const frames = fl.frames;
  const dur = frames[frames.length - 1][0];

  // Phase band boundaries, from the same phase field the labels use.
  let t1 = dur, t2 = dur;
  for (const f of frames) { if (f[2] >= 1) { t1 = f[0]; break; } }
  for (const f of frames) { if (f[2] >= 2) { t2 = f[0]; break; } }
  [[0, t1, PHASE_BAND_COLOR[0]], [t1, t2, PHASE_BAND_COLOR[1]], [t2, dur, PHASE_BAND_COLOR[2]]]
    .forEach(([a, b, color]) => {
      if (b <= a) return;
      c.fillStyle = color; c.globalAlpha = 0.08;
      c.fillRect(chartX(a, dur), CHART_MARGIN.top, chartX(b, dur) - chartX(a, dur),
                 H - CHART_MARGIN.top - CHART_MARGIN.bottom);
    });
  c.globalAlpha = 1;

  // Gridlines + axis labels: power (left, W) and motor command (right, 0-1).
  c.font = "10px sans-serif";
  c.strokeStyle = "#2c3038";
  [0, 0.5, 1].forEach((f) => {
    const y = chartYPower(f * POWER_MAX);
    c.beginPath(); c.lineWidth = 1; c.moveTo(CHART_MARGIN.left, y); c.lineTo(W - CHART_MARGIN.right, y); c.stroke();
    c.fillStyle = "#7c8590"; c.textAlign = "left"; c.textBaseline = "middle";
    c.fillText((f * POWER_MAX).toFixed(0) + "W", 2, y);
  });
  [0, 0.5, 1].forEach((f) => {
    const y = chartYMotor(f);
    c.fillStyle = "#7c8590"; c.textAlign = "right"; c.textBaseline = "middle";
    c.fillText(f.toFixed(1), W - 4, y);
  });
  [0, 0.25, 0.5, 0.75, 1].forEach((f) => {
    const t = f * dur, x = chartX(t, dur);
    c.beginPath(); c.strokeStyle = "#22262c"; c.moveTo(x, CHART_MARGIN.top); c.lineTo(x, H - CHART_MARGIN.bottom); c.stroke();
    c.fillStyle = "#7c8590"; c.textAlign = "center"; c.textBaseline = "top";
    c.fillText(t.toFixed(0) + "s", x, H - CHART_MARGIN.bottom + 4);
  });

  // Motor command lines (thin) then power (thicker, on top).
  MOTOR_ORDER.forEach((col, mi) => {
    c.beginPath();
    c.strokeStyle = MOTOR_LINE_COLORS[col];
    c.lineWidth = 1.3;
    c.globalAlpha = 0.9;
    frames.forEach((f, i) => {
      const x = chartX(f[0], dur), y = chartYMotor(f[3 + mi]);
      if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
    });
    c.stroke();
  });
  c.globalAlpha = 1;
  c.beginPath();
  c.strokeStyle = "#ffa94d";
  c.lineWidth = 2.4;
  frames.forEach((f, i) => {
    const x = chartX(f[0], dur), y = chartYPower(f[1]);
    if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
  });
  c.stroke();
}

function drawChartOverlay(t, frame) {
  chartCtx.clearRect(0, 0, chart.width, chart.height);
  chartCtx.drawImage(chartBase, 0, 0);
  const x = chartX(t, duration);

  chartCtx.strokeStyle = "rgba(255,255,255,0.85)";
  chartCtx.lineWidth = 1.5;
  chartCtx.beginPath();
  chartCtx.moveTo(x, CHART_MARGIN.top);
  chartCtx.lineTo(x, chart.height - CHART_MARGIN.bottom);
  chartCtx.stroke();

  const [, power, , m1, m2, m3, m4] = frame;
  const dot = (y, color) => {
    chartCtx.fillStyle = color;
    chartCtx.beginPath();
    chartCtx.arc(x, y, 3.2, 0, Math.PI * 2);
    chartCtx.fill();
  };
  dot(chartYPower(power), "#ffa94d");
  dot(chartYMotor(m1), MOTOR_LINE_COLORS.motor_1_front_right);
  dot(chartYMotor(m2), MOTOR_LINE_COLORS.motor_2_rear_left);
  dot(chartYMotor(m3), MOTOR_LINE_COLORS.motor_3_front_left);
  dot(chartYMotor(m4), MOTOR_LINE_COLORS.motor_4_rear_right);
}

function buildChartLegend() {
  const el = document.getElementById("chartLegend");
  const items = [["Power (W, left axis)", "#ffa94d"]];
  MOTOR_ORDER.forEach((col) => items.push([MOTOR_SHORT[col] + " command (0-1, right axis)", MOTOR_LINE_COLORS[col]]));
  items.push(["arming / flight / landing (background tint)", "#7c8590"]);
  el.innerHTML = items.map(([label, color]) =>
    '<span><i class="sw" style="background:' + color + '"></i>' + label + "</span>").join("");
}

// ---- state ----
// flightTime is a continuous float (seconds into the current flight's
// mission window). Rendering interpolates linearly between the two
// decimated data samples straddling it, so colour/value transitions are
// smooth at the display's native frame rate instead of snapping every
// 1/DECIMATION_HZ seconds ("digital lights").
let currentFlightId = null;
let flightTime = 0;
let duration = 0;
let dt = 1;
let playing = false;
let lastTs = null;

function setFlight(fid) {
  currentFlightId = fid;
  const fl = FLIGHTS[fid];
  duration = fl.frames[fl.frames.length - 1][0];
  dt = fl.frames.length > 1 ? duration / (fl.frames.length - 1) : 1;
  flightTime = 0;
  document.getElementById("lblFlight").textContent = fid;
  document.getElementById("lblTraj").textContent = fl.meta.trajectory;
  document.getElementById("lblPayload").textContent =
    fl.meta.payload_mass.toFixed(2) + " kg — " + fl.meta.position_payload;
  buildChartBase(fl);
  renderAtTime(0);
}

// Smoothstep-eased interpolation: zero velocity at each sample instead of a
// constant rate that changes abruptly at the next one, so a chain of
// segments reads as one continuous ease rather than piecewise-linear kinks.
function slerp(a, b, t) { const s = t * t * (3 - 2 * t); return a + (b - a) * s; }

// Interpolate the frame at continuous time `t` (seconds) via the two
// nearest decimated samples.
function interpolatedFrame(t) {
  const fl = FLIGHTS[currentFlightId];
  const frames = fl.frames;
  const idxFloat = Math.max(0, Math.min(t / dt, frames.length - 1));
  const i0 = Math.floor(idxFloat);
  const i1 = Math.min(i0 + 1, frames.length - 1);
  const frac = idxFloat - i0;
  const f0 = frames[i0], f1 = frames[i1];
  return [
    t,
    slerp(f0[1], f1[1], frac),        // power
    frac < 0.5 ? f0[2] : f1[2],       // phase (not interpolated, just nearest)
    slerp(f0[3], f1[3], frac),        // m1
    slerp(f0[4], f1[4], frac),        // m2
    slerp(f0[5], f1[5], frac),        // m3
    slerp(f0[6], f1[6], frac),        // m4
  ];
}

function renderAtTime(t) {
  flightTime = Math.max(0, Math.min(t, duration));
  const frame = interpolatedFrame(flightTime);
  drawFrame(frame);
  drawChartOverlay(flightTime, frame);
  const scrub = document.getElementById("scrub");
  scrub.value = duration > 0 ? Math.round((flightTime / duration) * 1000) : 0;
}

function tick(ts) {
  if (!playing) return;
  if (lastTs === null) lastTs = ts;
  const dtMs = ts - lastTs;
  lastTs = ts;
  const speed = parseFloat(document.getElementById("speedSelect").value);
  const next = flightTime + (dtMs / 1000) * speed / TIME_SCALE;
  if (next >= duration) {
    renderAtTime(duration);
    setPlaying(false);
    return;
  }
  renderAtTime(next);
  requestAnimationFrame(tick);
}

function setPlaying(p) {
  playing = p;
  document.getElementById("playBtn").innerHTML = playing ? "&#10074;&#10074; Pause" : "&#9654; Play";
  if (playing) {
    lastTs = null;
    requestAnimationFrame(tick);
  }
}

document.getElementById("playBtn").addEventListener("click", () => setPlaying(!playing));
document.getElementById("scrub").addEventListener("input", (e) => {
  setPlaying(false);
  const frac = parseInt(e.target.value, 10) / 1000;
  renderAtTime(frac * duration);
});
document.getElementById("flightSelect").addEventListener("change", (e) => setFlight(e.target.value));

function init() {
  const select = document.getElementById("flightSelect");
  Object.keys(FLIGHTS).sort().forEach((fid) => {
    const opt = document.createElement("option");
    opt.value = fid; opt.textContent = fid;
    select.appendChild(opt);
  });
  document.getElementById("decimationNote").textContent =
    "Data decimated to " + DECIMATION_HZ + " fps of flight time, smoothly interpolated for playback. " +
    "Played at " + (1 / TIME_SCALE).toFixed(2) + "x real flight time by default (Speed multiplies on top).";
  drawColorbar();
  buildChartLegend();
  setFlight(Object.keys(FLIGHTS).sort()[0]);
}

img.onload = init;
</script>

<div class="avg-section">
  <h2>Average motor command per flight — mission window</h2>
  <p class="note">
    Mean of each rotor's command while inside its mission window (arm-to-shutdown, +/- 12 s
    margin), one static snapshot per flight — not the whole recording, so idle time doesn't
    dilute the comparison. Same jet colour map and gamma warp as the animation above.
  </p>
  <img src="__AVERAGE_HEATMAP_B64__" alt="Average motor command per flight">
</div>
</body>
</html>
"""

GRID_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Motor heatmap — all flights (shared clock)</title>
<style>
__SHARED_CSS__
.grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
}
.panel { background: #1e2126; border: 1px solid #2c3038; border-radius: 10px; padding: 8px; }
.panel canvas.rotor { width: 100%; height: auto; display: block; border-radius: 6px; }
.panel canvas.spark { width: 100%; height: 42px; display: block; border-radius: 4px; margin-top: 6px; }
.panel .title { font-size: 0.78rem; font-weight: 600; margin-bottom: 4px; }
.panel .sub { font-size: 0.68rem; color: #9aa2ad; margin-bottom: 6px; }
.panel .stat { font-size: 0.68rem; color: #cfd4da; display: flex; justify-content: space-between; margin-top: 4px; }
.panel .sparklabel { font-size: 0.62rem; color: #7c8590; margin-top: 4px; }
.colorbar-wrap { margin: 4px 0 14px; max-width: 420px; }
.colorbar { height: 14px; border-radius: 4px; border: 1px solid #3a4048; width: 100%; }
.colorbar-ticks { display: flex; justify-content: space-between; font-size: 0.72rem; color: #9aa2ad; margin-top: 3px; }
</style>
</head>
<body>
<h1>Animated motor heatmap — all 14 flights, shared normalised clock</h1>

<div class="toolbar">
  <button id="playBtn" class="primary">&#9654; Play</button>

  <label for="speedSelect">Speed</label>
  <select id="speedSelect">
    <option value="1">1x</option>
    <option value="2">2x</option>
    <option value="5">5x</option>
  </select>

  <input type="range" id="scrub" min="0" max="1000" value="0" step="1">
  <span class="note" id="pctLabel">0%</span>
</div>
<div class="note" style="margin-bottom: 10px;">
  Each flight's own mission window (arm-to-shutdown, +/- 12 s margin) is normalised to 0-100% so
  flights of different length stay in step. __N_STEPS__ samples across the clock, smoothly
  interpolated for playback, at half real-time speed by default (Speed multiplies on top).
</div>

<div class="colorbar-wrap">
  <div class="k note">Motor command colour scale (fixed across all flights and frames)</div>
  <canvas class="colorbar" id="colorbar" width="420" height="14"></canvas>
  <div class="colorbar-ticks"><span id="cbMin"></span><span id="cbMax"></span></div>
</div>

<div class="grid" id="grid"></div>

<script>
const IMAGE_B64 = "__IMAGE_B64__";
const MOTOR_LAYOUT = __MOTOR_LAYOUT_JSON__;
const MOTOR_ORDER = __MOTOR_ORDER_JSON__;
const CIRCLE_RADIUS_FRAC = __CIRCLE_RADIUS_FRAC__;
const COLOR_SCALE = __COLOR_SCALE_JSON__;
const COLOR_LUT = __COLOR_LUT_JSON__;
const COLOR_GAMMA = __COLOR_GAMMA__;
const TIME_SCALE = __TIME_SCALE__;
const POWER_MAX = __POWER_MAX__;
const FLIGHTS = __FLIGHTS_JSON__;
const N_STEPS = __N_STEPS__;
const PHASE_NAMES = ["arming", "flight", "landing/shutdown"];

const img = new Image();
img.src = IMAGE_B64;

// Warping t by **COLOR_GAMMA (gamma < 1) pushes the warm end of the scale
// (yellow/orange/red) to trigger at lower raw values -- exaggerates red
// without moving vmin/vmax, so printed numbers and the legend stay exact.
function lutColor(value) {
  let t = (value - COLOR_SCALE.vmin) / (COLOR_SCALE.vmax - COLOR_SCALE.vmin);
  t = Math.max(0, Math.min(1, t));
  t = Math.pow(t, COLOR_GAMMA);
  const idx = Math.round(t * (COLOR_LUT.length - 1));
  return COLOR_LUT[idx];
}

const flightIds = Object.keys(FLIGHTS).sort();
const canvases = {};
const sparkCanvases = {};
const sparkBases = {};   // offscreen canvas per flight: static power line drawn once
const SPARK_W = 300, SPARK_H = 44, SPARK_MARGIN = { left: 2, right: 2, top: 4, bottom: 4 };

function buildGrid() {
  const grid = document.getElementById("grid");
  flightIds.forEach((fid) => {
    const fl = FLIGHTS[fid];
    const panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML =
      '<div class="title">' + fid + ' — ' + fl.meta.trajectory + '</div>' +
      '<div class="sub">' + fl.meta.payload_mass.toFixed(2) + ' kg — ' + fl.meta.position_payload + '</div>' +
      '<canvas class="rotor" width="843" height="855"></canvas>' +
      '<div class="stat"><span class="phase">-</span><span class="power">-</span></div>' +
      '<canvas class="spark" width="' + SPARK_W + '" height="' + SPARK_H + '"></canvas>' +
      '<div class="sparklabel">Power over time (0-' + POWER_MAX.toFixed(0) + ' W, fixed scale)</div>';
    grid.appendChild(panel);
    canvases[fid] = panel.querySelector("canvas.rotor");
    sparkCanvases[fid] = panel.querySelector("canvas.spark");
    sparkBases[fid] = buildSparkBase(fl);
  });
}

// Static power-vs-time line for one flight, drawn once onto an offscreen
// canvas so the per-frame render is just a cheap image blit + playhead line.
function buildSparkBase(fl) {
  const off = document.createElement("canvas");
  off.width = SPARK_W; off.height = SPARK_H;
  const c = off.getContext("2d");
  const frames = fl.frames;
  const dur = frames[frames.length - 1][0];
  const pw = SPARK_W - SPARK_MARGIN.left - SPARK_MARGIN.right;
  const ph = SPARK_H - SPARK_MARGIN.top - SPARK_MARGIN.bottom;
  const x = (t) => SPARK_MARGIN.left + (dur > 0 ? (t / dur) * pw : 0);
  const y = (p) => SPARK_H - SPARK_MARGIN.bottom - Math.max(0, Math.min(p / POWER_MAX, 1)) * ph;

  c.strokeStyle = "#ffa94d";
  c.lineWidth = 1.6;
  c.beginPath();
  frames.forEach((f, i) => {
    const px = x(f[0]), py = y(f[1]);
    if (i === 0) c.moveTo(px, py); else c.lineTo(px, py);
  });
  c.stroke();
  return off;
}

function drawSparkOverlay(fid, clock) {
  const canvas = sparkCanvases[fid];
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, SPARK_W, SPARK_H);
  ctx.drawImage(sparkBases[fid], 0, 0);
  const pw = SPARK_W - SPARK_MARGIN.left - SPARK_MARGIN.right;
  const x = SPARK_MARGIN.left + clock * pw;
  ctx.strokeStyle = "rgba(255,255,255,0.8)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, SPARK_MARGIN.top);
  ctx.lineTo(x, SPARK_H - SPARK_MARGIN.bottom);
  ctx.stroke();
}

// Smoothstep-eased interpolation: zero velocity at each sample instead of a
// constant rate that changes abruptly at the next one, so a chain of
// segments reads as one continuous ease rather than piecewise-linear kinks.
function slerp(a, b, t) { const s = t * t * (3 - 2 * t); return a + (b - a) * s; }

// Interpolate flight `fid`'s frame at continuous shared clock position
// `clock` (0-1). All flights share the same number of normalised steps at
// the same clock fractions, so a single float index works for everyone.
function interpolatedFrame(fid, clock) {
  const frames = FLIGHTS[fid].frames;
  const idxFloat = Math.max(0, Math.min(clock * (N_STEPS - 1), frames.length - 1));
  const i0 = Math.floor(idxFloat);
  const i1 = Math.min(i0 + 1, frames.length - 1);
  const frac = idxFloat - i0;
  const f0 = frames[i0], f1 = frames[i1];
  return [
    slerp(f0[0], f1[0], frac),
    slerp(f0[1], f1[1], frac),
    frac < 0.5 ? f0[2] : f1[2],
    slerp(f0[3], f1[3], frac),
    slerp(f0[4], f1[4], frac),
    slerp(f0[5], f1[5], frac),
    slerp(f0[6], f1[6], frac),
  ];
}

function drawPanel(fid, clock) {
  const frame = interpolatedFrame(fid, clock);
  const canvas = canvases[fid];
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);

  const radius = CIRCLE_RADIUS_FRAC * w;
  const [elapsed, power, phase, m1, m2, m3, m4] = frame;
  const values = { motor_1_front_right: m1, motor_2_rear_left: m2, motor_3_front_left: m3, motor_4_rear_right: m4 };

  for (const col of MOTOR_ORDER) {
    const layout = MOTOR_LAYOUT[col];
    const cx = layout.pos[0] * w, cy = layout.pos[1] * h;
    const val = values[col];

    ctx.globalAlpha = 0.75;
    ctx.fillStyle = lutColor(val);
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.lineWidth = 3;
    ctx.strokeStyle = "black";
    ctx.stroke();

    ctx.fillStyle = "black";
    ctx.font = "bold 36px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(val.toFixed(2), cx, cy + 12);
  }

  const panel = canvas.closest(".panel");
  panel.querySelector(".phase").textContent = PHASE_NAMES[phase];
  panel.querySelector(".power").textContent = power.toFixed(0) + " W";
}

// Sampled by raw value (not by raw LUT index) so the bar shows the same
// gamma-warped mapping the circles use -- what you see here is what you get.
function drawColorbar() {
  const cb = document.getElementById("colorbar");
  const cctx = cb.getContext("2d");
  const grad = cctx.createLinearGradient(0, 0, cb.width, 0);
  const n = 128;
  for (let i = 0; i < n; i++) {
    const frac = i / (n - 1);
    const value = COLOR_SCALE.vmin + frac * (COLOR_SCALE.vmax - COLOR_SCALE.vmin);
    grad.addColorStop(frac, lutColor(value));
  }
  cctx.fillStyle = grad;
  cctx.fillRect(0, 0, cb.width, cb.height);
  document.getElementById("cbMin").textContent = COLOR_SCALE.vmin.toFixed(3);
  document.getElementById("cbMax").textContent = COLOR_SCALE.vmax.toFixed(3);
}

// clock is a continuous float in [0,1] across the shared normalised mission
// window. Panels interpolate between their two nearest samples every
// animation frame, so playback is smooth instead of snapping every step.
let clock = 0;
let playing = false;
let lastTs = null;
const BASE_FPS = __GRID_FPS__;
// fraction of the clock covered per wall-clock second at speed=1, before
// TIME_SCALE slows it down further (see PLAYBACK_TIME_SCALE in the generator)
const CLOCK_RATE = BASE_FPS / (N_STEPS - 1);

function renderAll() {
  flightIds.forEach((fid) => { drawPanel(fid, clock); drawSparkOverlay(fid, clock); });
  document.getElementById("scrub").value = Math.round(clock * 1000);
  document.getElementById("pctLabel").textContent = (100 * clock).toFixed(0) + "%";
}

function tick(ts) {
  if (!playing) return;
  if (lastTs === null) lastTs = ts;
  const dtMs = ts - lastTs;
  lastTs = ts;
  const speed = parseFloat(document.getElementById("speedSelect").value);
  const next = clock + (dtMs / 1000) * CLOCK_RATE * speed / TIME_SCALE;
  if (next >= 1) {
    clock = 1;
    renderAll();
    setPlaying(false);
    return;
  }
  clock = next;
  renderAll();
  requestAnimationFrame(tick);
}

function setPlaying(p) {
  playing = p;
  document.getElementById("playBtn").innerHTML = playing ? "&#10074;&#10074; Pause" : "&#9654; Play";
  if (playing) { lastTs = null; requestAnimationFrame(tick); }
}

document.getElementById("playBtn").addEventListener("click", () => setPlaying(!playing));
document.getElementById("scrub").addEventListener("input", (e) => {
  setPlaying(false);
  clock = parseInt(e.target.value, 10) / 1000;
  renderAll();
});

img.onload = () => {
  buildGrid();
  drawColorbar();
  renderAll();
};
</script>

<div class="avg-section">
  <h2>Average motor command per flight — mission window</h2>
  <p class="note">
    Mean of each rotor's command while inside its mission window (arm-to-shutdown, +/- 12 s
    margin), one static snapshot per flight — not the whole recording, so idle time doesn't
    dilute the comparison. Same jet colour map and gamma warp as the animation above.
  </p>
  <img src="__AVERAGE_HEATMAP_B64__" alt="Average motor command per flight">
</div>
</body>
</html>
"""


def render_template(template: str, mapping: dict) -> str:
    html = template.replace("__SHARED_CSS__", SHARED_CSS)
    for key, value in mapping.items():
        html = html.replace(key, value)
    return html


# =====================================================
# MAIN
# =====================================================
def main():
    dirs = sorted(glob.glob(os.path.join(FLIGHTS_DIR, "F*")))
    flight_ids = [os.path.basename(d) for d in dirs]
    if not flight_ids:
        raise SystemExit(f"No flight folders found under {FLIGHTS_DIR}")

    print(f"Loading {len(flight_ids)} flights: {flight_ids}")
    flights = []
    for fid in flight_ids:
        fl = load_flight(fid)
        if fl is not None:
            flights.append(fl)
            print(
                f"  {fid}: window {fl['window_start']:.1f}-{fl['window_end']:.1f}s "
                f"(mission {fl['first_above']:.1f}-{fl['last_above']:.1f}s, thr={fl['thr']:.0f} W)"
            )
    if not flights:
        raise SystemExit("No usable flights found.")

    vmin, vmax = compute_global_color_scale(flights)
    print(f"Global colour scale (1st-99th pct of in-flight commands): {vmin:.3f} - {vmax:.3f}")
    color_lut = build_color_lut()

    # Fixed global power axis (like the colour scale) so the power-over-time
    # chart is comparable flight to flight instead of auto-scaling per panel.
    power_max = float(max(fl["power"].max() for fl in flights)) * 1.05
    print(f"Global power axis max: {power_max:.0f} W")

    with open(IMAGE_PATH, "rb") as f:
        image_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("ascii")

    motor_layout_json = json.dumps(
        {k: {"pos": list(v["pos"]), "label": v["label"]} for k, v in MOTOR_LAYOUT.items()}
    )
    motor_order_json = json.dumps(MOTOR_ORDER)
    color_scale_json = json.dumps({"vmin": vmin, "vmax": vmax})
    color_lut_json = json.dumps(color_lut)

    print("Rendering average-heatmap-per-flight PNG...")
    average_heatmap_b64 = build_average_heatmap_png(flights, vmin, vmax, COLOR_GAMMA)

    # ---- single-flight view ----
    single_flights = {}
    for fl in flights:
        single_flights[fl["flight_id"]] = {
            "meta": fl["meta"],
            "frames": build_single_frames(fl),
        }
    single_html = render_template(SINGLE_HTML_TEMPLATE, {
        "__IMAGE_B64__": image_b64,
        "__MOTOR_LAYOUT_JSON__": motor_layout_json,
        "__MOTOR_ORDER_JSON__": motor_order_json,
        "__CIRCLE_RADIUS_FRAC__": json.dumps(CIRCLE_RADIUS_FRAC),
        "__COLOR_SCALE_JSON__": color_scale_json,
        "__COLOR_LUT_JSON__": color_lut_json,
        "__COLOR_GAMMA__": json.dumps(COLOR_GAMMA),
        "__TIME_SCALE__": json.dumps(PLAYBACK_TIME_SCALE),
        "__DECIMATION_HZ__": json.dumps(SINGLE_DECIMATION_HZ),
        "__POWER_MAX__": json.dumps(power_max),
        "__FLIGHTS_JSON__": json.dumps(single_flights),
        "__AVERAGE_HEATMAP_B64__": average_heatmap_b64,
    })
    with open(OUT_SINGLE, "w") as f:
        f.write(single_html)
    size_mb = os.path.getsize(OUT_SINGLE) / 1e6
    print(f"Wrote {OUT_SINGLE} ({size_mb:.2f} MB)")

    # ---- grid view ----
    grid_flights = {}
    for fl in flights:
        frames, duration = build_grid_frames(fl)
        grid_flights[fl["flight_id"]] = {
            "meta": dict(fl["meta"], window_duration_s=round(duration, 1)),
            "frames": frames,
        }
    grid_fps = max(5, min(15, GRID_NORM_STEPS // 20))  # gentle default playback rate
    grid_html = render_template(GRID_HTML_TEMPLATE, {
        "__IMAGE_B64__": image_b64,
        "__MOTOR_LAYOUT_JSON__": motor_layout_json,
        "__MOTOR_ORDER_JSON__": motor_order_json,
        "__CIRCLE_RADIUS_FRAC__": json.dumps(CIRCLE_RADIUS_FRAC),
        "__COLOR_SCALE_JSON__": color_scale_json,
        "__COLOR_LUT_JSON__": color_lut_json,
        "__COLOR_GAMMA__": json.dumps(COLOR_GAMMA),
        "__TIME_SCALE__": json.dumps(PLAYBACK_TIME_SCALE),
        "__POWER_MAX__": json.dumps(power_max),
        "__FLIGHTS_JSON__": json.dumps(grid_flights),
        "__N_STEPS__": str(GRID_NORM_STEPS),
        "__GRID_FPS__": str(grid_fps),
        "__AVERAGE_HEATMAP_B64__": average_heatmap_b64,
    })
    with open(OUT_GRID, "w") as f:
        f.write(grid_html)
    size_mb = os.path.getsize(OUT_GRID) / 1e6
    print(f"Wrote {OUT_GRID} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
