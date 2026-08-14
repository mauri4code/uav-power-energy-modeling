"""
Step 14 – Build synthetic test flights by stitching intact segments.

Idea:
  A synthetic flight is made of contiguous slices ("segments") taken from
  DIFFERENT real flights and concatenated in time. Each segment comes from a
  single real flight, so the physics inside a segment (features -> power) stays
  intact — we never mix rows across flights within a segment. What changes over
  time, segment to segment, is the operating condition: payload_mass,
  position_payload, trajectory, and therefore speed / altitude / power.

Two variants are produced (in separate folders):
  1. "progressive" : one segment per flight, ordered by payload (0 -> 0.24 ->
                     0.48). Payload / speed / altitude change in clear stages.
  2. "random"      : N segments from random flights in random order (fixed seed)
                     — an "unseen mixed" stress test.

Input  : flights/*/flight_resampled.csv        (the resampled real flights)
Output : SIM_FLIGHTS/SIM_progressive/flight_resampled.csv
         SIM_FLIGHTS/SIM_random/flight_resampled.csv
         + a segment_map.csv and a preview plot per variant

These folders sit OUTSIDE flights/, so step 5 (05_prepare_ml.py) does NOT pick
them up automatically. To score the model on one, point the model script at the
file explicitly.

Run: python 14_make_synthetic_flight.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Paths ----
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FLIGHTS_DIR = os.path.join(SCRIPT_DIR, "flights")
OUT_DIR     = os.path.join(SCRIPT_DIR, "SIM_FLIGHTS")

# ---- Settings ----
DT              = 0.05          # 20 Hz — matches the resampled grid
SEGMENT_SECONDS = 20.0          # length of each stitched segment
SEG_LEN         = int(round(SEGMENT_SECONDS / DT))   # = 400 samples
EDGE_TRIM       = 40            # skip this many samples at start/end of a flight
                                # (avoids takeoff / landing transients)
RANDOM_SEED     = 42
N_RANDOM_SEGS   = 12            # segments in the "random" variant


# ===========================================================
#  HELPERS
# ===========================================================

def load_flights():
    """Load every flight_resampled.csv keyed by flight_id, sorted by id."""
    files = sorted(glob.glob(os.path.join(FLIGHTS_DIR, "*", "flight_resampled.csv")))
    if not files:
        raise FileNotFoundError(f"No flight_resampled.csv under {FLIGHTS_DIR}/")
    flights = {}
    for f in files:
        df = pd.read_csv(f)
        fid = str(df["flight_id"].iloc[0])
        flights[fid] = df.reset_index(drop=True)
    print(f"  Loaded {len(flights)} flights: {sorted(flights)}")
    return flights


def take_segment(df, start=None, rng=None):
    """
    Return a contiguous SEG_LEN slice from df, taken from the usable middle
    region [EDGE_TRIM, len-EDGE_TRIM]. If `start` is None a random valid start
    is drawn from `rng`; otherwise the given start is clamped to a valid range.
    """
    lo = EDGE_TRIM
    hi = len(df) - EDGE_TRIM - SEG_LEN
    if hi < lo:                      # flight too short to trim: use whatever fits
        lo, hi = 0, max(0, len(df) - SEG_LEN)
    if start is None:
        start = int(rng.integers(lo, hi + 1)) if hi > lo else lo
    else:
        start = int(np.clip(start, lo, hi if hi > lo else lo))
    seg = df.iloc[start:start + SEG_LEN].copy().reset_index(drop=True)
    return seg, start


def stitch(segments, new_flight_id):
    """
    Concatenate a list of segment dataframes into one synthetic flight and
    re-base the timestamp to a single continuous 20 Hz timeline. Returns the
    stitched dataframe and a segment map (provenance per segment).
    """
    stitched = pd.concat(segments, ignore_index=True)
    n = len(stitched)
    stitched["timestamp"] = np.arange(n) * DT     # continuous, starts at 0
    stitched["flight_id"] = new_flight_id         # single new id for the whole flight

    # provenance table
    rows, cursor = [], 0
    for seg in segments:
        entry = {
            "source_flight" : seg["_src_flight"].iloc[0],
            "src_start_idx" : int(seg["_src_start"].iloc[0]),
            "dst_start_idx" : cursor,
            "n_samples"     : len(seg),
            "t_start_s"     : cursor * DT,
            "t_end_s"       : (cursor + len(seg)) * DT,
            "payload_mass"  : seg["payload_mass"].iloc[0],
            "position_payload": seg["position_payload"].iloc[0],
            "trajectory"    : seg["trajectory"].iloc[0],
        }
        if "phase" in seg.columns:               # mission builder tags phases
            entry["phase"] = seg["phase"].iloc[0]
        rows.append(entry)
        cursor += len(seg)
    seg_map = pd.DataFrame(rows)

    # drop temporary / helper columns so the flight keeps the real-flight schema
    drop = [c for c in ["_src_flight", "_src_start", "phase"] if c in stitched.columns]
    stitched = stitched.drop(columns=drop)
    return stitched, seg_map


def tag(seg, src_flight, src_start):
    seg["_src_flight"] = src_flight
    seg["_src_start"]  = src_start
    return seg


def build_progressive(flights):
    """One segment per flight, ordered by (payload_mass, position, trajectory)."""
    order = sorted(
        flights,
        key=lambda fid: (
            flights[fid]["payload_mass"].iloc[0],
            str(flights[fid]["position_payload"].iloc[0]),
            str(flights[fid]["trajectory"].iloc[0]),
        ),
    )
    segments = []
    for fid in order:
        df = flights[fid]
        # deterministic: middle of the flight
        start = (len(df) - SEG_LEN) // 2
        seg, s = take_segment(df, start=start)
        segments.append(tag(seg, fid, s))
    return stitch(segments, "SIM_PROG")


def raw_segment(df, start):
    """A plain SEG_LEN slice at an explicit index — NO edge-trimming, so genuine
    takeoff/landing transitions at a flight's edges are preserved."""
    start = int(np.clip(start, 0, max(0, len(df) - SEG_LEN)))
    return df.iloc[start:start + SEG_LEN].copy().reset_index(drop=True), start


def take_clean_segment(df):
    """
    Pick the SEG_LEN window whose power is cleanest (highest 5th-percentile
    power), scanning candidate starts across the usable middle region. This
    avoids brief power dropouts / transients (e.g. F03's dip to ~30 W) so each
    segment has a stable, representative power level. Returns (segment, start).
    """
    lo = EDGE_TRIM
    hi = len(df) - EDGE_TRIM - SEG_LEN
    if hi < lo:
        lo, hi = 0, max(0, len(df) - SEG_LEN)
    best_start, best_score = lo, -np.inf
    for start in range(lo, hi + 1, 20):          # scan every 1 s
        p = df["power"].iloc[start:start + SEG_LEN]
        score = np.percentile(p, 5)              # penalises windows with dropouts
        if score > best_score:
            best_score, best_start = score, start
    return take_segment(df, start=best_start)


def build_varied(flights):
    """
    Order segments to maximise the up/down swing in power between neighbours:
    sort flights by mean power, then interleave from the extremes
    (highest, lowest, 2nd-highest, 2nd-lowest, ...). Clean windows are used so
    each step is a clear, stable power level.
    """
    # clean segment + its mean power for every flight
    segs = {}
    for fid, df in flights.items():
        seg, s = take_clean_segment(df)
        segs[fid] = (seg, s, seg["power"].mean())

    by_power = sorted(segs, key=lambda fid: segs[fid][2])   # ascending mean power
    # interleave: highest, lowest, next-highest, next-lowest, ...
    order, left, right = [], 0, len(by_power) - 1
    while left <= right:
        order.append(by_power[right]); right -= 1
        if left <= right:
            order.append(by_power[left]); left += 1

    segments = []
    for fid in order:
        seg, s, _ = segs[fid]
        segments.append(tag(seg, fid, s))
    return stitch(segments, "SIM_VARIED")


def build_mission(flights):
    """
    One realistic mission profile stitched from real segments:
        takeoff (climb, high power) -> high cruise -> medium cruise ->
        low cruise -> landing (descent to ground).

    Takeoff and landing use RAW edge windows (real ground->climb and
    descent->ground transitions). The cruise phases use clean mid-flight
    windows at descending power levels (0.48 -> 0.24 -> 0.00 kg).

    Each entry: (flight_id, start_index_or_None, phase_label).
    start=None -> clean mid-flight window; otherwise raw window at that index.
    """
    plan = [
        ("F07", 181,  "takeoff"),   # 0.48 kg: ground idle -> climb (power ramps up)
        ("F12", None, "high"),      # 0.48 kg: highest cruise power (~778 W)
        ("F13", None, "medium"),    # 0.24 kg: medium cruise power (~715 W)
        ("F01", None, "low"),       # 0.00 kg: lowest cruise power (~587 W)
        ("F01", 2300, "landing"),   # cruise -> power cut -> glide down to ground
    ]
    segments = []
    for fid, start, phase in plan:
        df = flights[fid]
        if start is None:
            seg, s = take_clean_segment(df)
        else:
            seg, s = raw_segment(df, start)
        seg["phase"] = phase
        segments.append(tag(seg, fid, s))
    return stitch(segments, "SIM_MISSION")


def build_random(flights):
    """N random segments from random flights in random order (fixed seed)."""
    rng = np.random.default_rng(RANDOM_SEED)
    fids = sorted(flights)
    picks = rng.choice(fids, size=N_RANDOM_SEGS, replace=True)
    segments = []
    for fid in picks:
        df = flights[fid]
        seg, s = take_segment(df, rng=rng)
        segments.append(tag(seg, fid, s))
    return stitch(segments, "SIM_RAND")


def preview_plot(stitched, seg_map, title, path):
    """Plot payload / speed / altitude / power over time with segment boundaries."""
    t = stitched["timestamp"].values
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    series = [
        ("payload_mass",              "Payload [kg]",   "tab:purple"),
        ("speed_3d",                  "Speed 3D [m/s]", "tab:blue"),
        ("uav1_mavros_altitude__local", "Altitude [m]", "tab:green"),
        ("power",                     "Power [W]",      "tab:red"),
    ]
    for ax, (col, label, color) in zip(axes, series):
        ax.plot(t, stitched[col].values, color=color, lw=0.8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        for _, r in seg_map.iterrows():
            ax.axvline(r["t_start_s"], color="k", ls=":", lw=0.6, alpha=0.5)
    # label each segment along the top axis (phase name if present, else flight)
    for _, r in seg_map.iterrows():
        lbl = r["source_flight"]
        if "phase" in seg_map.columns:
            lbl = f"{r['phase'].upper()}\n({r['source_flight']})"
        axes[0].text(
            (r["t_start_s"] + r["t_end_s"]) / 2,
            axes[0].get_ylim()[1],
            lbl, ha="center", va="bottom", fontsize=7.5, rotation=0,
        )
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_variant(stitched, seg_map, folder, plot_title):
    os.makedirs(folder, exist_ok=True)
    fcsv = os.path.join(folder, "flight_resampled.csv")
    mcsv = os.path.join(folder, "segment_map.csv")
    png  = os.path.join(folder, "preview.png")
    stitched.to_csv(fcsv, index=False)
    seg_map.to_csv(mcsv, index=False)
    preview_plot(stitched, seg_map, plot_title, png)
    print(f"  Saved -> {fcsv}  ({len(stitched)} rows, {len(seg_map)} segments)")
    print(f"          {mcsv}")
    print(f"          {png}")


# ===========================================================
#  MAIN
# ===========================================================

def run():
    print("\n" + "=" * 55)
    print("  [Step 14] Build synthetic test flights")
    print("=" * 55)
    print(f"  Segment length : {SEGMENT_SECONDS:.0f} s  ({SEG_LEN} samples @ {1/DT:.0f} Hz)")

    flights = load_flights()

    # Single realistic mission profile:
    #   takeoff -> high -> medium -> low -> landing
    print("\n  -- Mission profile (takeoff / high / medium / low / landing) --")
    mission, mission_map = build_mission(flights)
    save_variant(mission, mission_map, os.path.join(OUT_DIR, "SIM_mission"),
                 "Synthetic mission flight — takeoff → high → medium → low → landing")

    # The alternative multi-segment variants (progressive / random / varied) are
    # still available as functions above; call build_progressive / build_random /
    # build_varied here if you want them again.

    print("\n  Done.\n")


if __name__ == "__main__":
    run()
