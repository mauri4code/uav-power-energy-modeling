# Animated motor heatmap

Two self-contained HTML animations of the four PX4 rotor commands
(`motor_1_front_right`, `motor_2_rear_left`, `motor_3_front_left`,
`motor_4_rear_right`) overlaid on the airframe photo `Rotors_poss.jpeg`,
built from `../flights/F01..F14/flight_resampled.csv`.

```
make_heatmap_animation.py     generator (reads the CSVs, writes the HTML)
motor_animation.html          one flight at a time, with a flight dropdown
motor_animation_grid.html     all 14 flights animating together
Rotors_poss.jpeg              airframe photo, 843 x 855 px
```

## Regenerating

```bash
conda activate 00_EXPORT_TOPICS   # needs pandas, numpy, matplotlib, Pillow
cd HEAT_MAP
python make_heatmap_animation.py
```

This overwrites `motor_animation.html` and `motor_animation_grid.html`.
Both are plain files — open them directly in a browser, no server needed.

## The mission window

Each raw recording contains a stretch of ground idle before the motors are
armed, and its length is arbitrary: it depends only on when the operator
started the ROS bag, and ranges from 0 to 59 s across the 14 flights.
Animating that dead time is pointless and makes flights hard to compare, so
both HTML files animate only the **mission window**, computed per flight as:

1. `thr = 0.5 * (P5 + P95)` of that flight's `power` column — this lands
   around 340-415 W, cleanly between ~25 W idle and ~700 W airborne draw.
2. Find the first and last sample where `power > thr`.
3. Keep from **12 s before the first** crossing to **12 s after the last**.

The 12 s margin is not arbitrary: `power > thr` is a proxy for "airborne",
which lags "armed" (motors spinning at idle) by however long the operator
held on the ground before takeoff. The largest such arm-to-takeoff gap
across all 14 flights is 11.9 s, so 12 s is the smallest symmetric margin
guaranteed to capture the arming transient on every flight, and it adds a
matching tail so landing and shutdown are visible too. F03-F06 start mid-
flight (no idle stretch was captured), so their windows simply start at the
first sample — there's nothing before it to show.

Elapsed time (`arming` / `flight` / `landing/shutdown`) shown in the UI is
derived from the same threshold crossings.

## Colour scale

Fixed globally so flights are visually comparable — it is **not** rescaled
per flight or per frame. The limits are the 1st and 99th percentile of all
in-flight motor commands (i.e. samples where `power > thr`) pooled across
all 14 flights, currently **0.528 - 0.830**. Values outside that range
(mostly during arming/landing, near 0) are clipped in colour only; the
printed numeric value is always exact. The limits are restated on the
colour bar in both HTML files. The colormap is `matplotlib`'s `jet` — the
blue -> cyan -> green -> yellow -> red rainbow legend used by SolidWorks
Simulation (and most FEA/CFD tools), sampled to a 256-colour lookup table
and embedded in the HTML so the JS side never needs matplotlib.

In practice motor commands rarely sit near the very top of the vmin-vmax
range, so a plain linear mapping stayed mostly blue/green and red was rare.
To make the warm end easier to read, the normalised value `t` is warped by
`t ** COLOR_GAMMA` (`COLOR_GAMMA = 0.6`, set in `make_heatmap_animation.py`)
before indexing the colour table — this exaggerates red/orange at more
moderate values without touching `vmin`/`vmax` or the printed numbers. The
colour bar in both HTML files is drawn with the same warp, so it always
matches what the circles show. Lower `COLOR_GAMMA` (e.g. 0.4) for even more
aggressive red; `1.0` restores a plain linear scale.

## Decimation and playback smoothness

`flight_resampled.csv` sits on a uniform 20 Hz grid, but that grid is *not*
the sensor's native rate for motors or power. Looking at `04_resampling.py`:
PX4 only reports a new motor command every ~0.1s (raw ~10 Hz) and the ROS
battery topic only every ~2s (~0.5 Hz), and both are written onto the 20 Hz
grid with `forwardfill_to_grid` — a hold, not an interpolation. So most
adjacent 20 Hz rows are exact repeats, with each real change compressed
into a single 1/20s grid step. Animating straight off that column (even
with interpolation at render time) reproduces the same instant-jump-then-
hold shape — which is the "digital lights" look.

The fix is in `extract_change_points()`: it collapses each forward-filled
column back to the rows where the value actually changes, and the frame
builders (`build_single_frames` / `build_grid_frames`) interpolate against
*those* points instead of the raw 20 Hz grid. That reconstructs a ramp
spanning the true ~0.1s (motors) / ~2s (power) update interval instead of
squeezing it into 1/20s. `motor_animation.html` then samples that
reconstructed curve at `SINGLE_DECIMATION_HZ = 20` (matching the grid, so
nothing is lost) and rounds motor commands to 3 decimals to keep the file
small (~2.5 MB for all 14 flights). `motor_animation_grid.html` samples it
onto a shared 400-step 0-100% clock (see below) instead of a fixed Hz,
since panels need to stay in lock-step regardless of each flight's real
duration.

On top of that, both players drive the canvas from `requestAnimationFrame`
(display refresh rate, not data rate) and ease every channel — motor
commands, power, circle colour — between the two samples straddling the
current playhead with `smoothstep` (`t*t*(3-2*t)`) rather than straight
linear interpolation, so each segment has zero velocity at its endpoints
instead of an abrupt rate change at the next sample. `COLOR_LUT_STOPS` was
also raised from 256 to 1024 so the colour table itself has no visible
banding. `PLAYBACK_TIME_SCALE = 2.0` in `make_heatmap_animation.py`
additionally slows the default wall-clock playback to half real-time (i.e.
it takes twice as long to watch), on top of which the Speed selector
(1x/2x/5x) still multiplies. Lower `PLAYBACK_TIME_SCALE` back to `1.0` for
real-time-paced default playback.

## Grid view / shared clock

`motor_animation_grid.html` shows all 14 flights animating together to make
payload-position effects visible at a glance — e.g. whether a front payload
consistently loads `motor_1_front_right` / `motor_3_front_left` harder than
the rear pair. Because flights have different mission-window durations,
each flight's window is independently normalised to 0-100% (400 steps) and
the UI drives a single shared step index across all panels, so "50%" always
means "halfway through this flight's own mission" regardless of how long
that mission actually took.

## Rotor layout

`../09_motor_heatmap.py`'s original positions were a rough eyeball estimate
and sit visibly off-centre from the drawn rotor rings once overlaid at full
resolution. They were re-measured here with a small Hough-circle search
(scan candidate `(cx, cy, r)` and keep the one whose perimeter has maximum
overlap with the image's dark ink) against `Rotors_poss.jpeg`, hitting
100% ring coverage for all four rotors:

```python
MOTOR_LAYOUT = {
    "motor_1_front_right": {"pos": (0.7960, 0.2198), "label": "M1 front-right"},
    "motor_2_rear_left":   {"pos": (0.1696, 0.8366), "label": "M2 rear-left"},
    "motor_3_front_left":  {"pos": (0.1685, 0.2223), "label": "M3 front-left"},
    "motor_4_rear_right":  {"pos": (0.7810, 0.8347), "label": "M4 rear-right"},
}
CIRCLE_RADIUS_FRAC = 0.114
```

`../09_motor_heatmap.py` still uses the old constants and will inherit the
same drift if it's regenerated — not changed here since it wasn't part of
this task.

## Average heatmap per flight

Both HTML files end with a static "Average motor command per flight —
mission window" image: one panel per flight, each rotor coloured by its
mean command **while inside that flight's mission window** (not the whole
recording — idle time before/after would dilute the comparison unevenly
since idle-stretch length varies by flight). It's a single PNG generated by
`build_average_heatmap_png()` in `make_heatmap_animation.py` using
`matplotlib` with `PowerNorm(gamma=COLOR_GAMMA)` over the same `jet`
colormap, so it always matches what the animation's colours mean — a quick
way to sanity-check payload-position effects (e.g. front payload -> higher
`M1`/`M3`) without having to scrub through the animation.
