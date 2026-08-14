"""
Step 4 – Resample all cleaned sensor CSVs to a common time grid and merge.

Input  : <OUTPUT_FOLDER>/<sensor>_clean.csv
Output : <OUTPUT_FOLDER>/flight_resampled.csv

Run standalone : python 04_resampling.py
Run via runner : called automatically by run_flight.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # no GUI — save only
import matplotlib.pyplot as plt

MOTOR_COLS   = ["motor_1_front_right", "motor_2_rear_left",
                "motor_3_front_left",  "motor_4_rear_right"]
MOTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def interpolate_to_grid(df: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Linear interpolation to a uniform time grid (high-rate signals)."""
    df = (
        df.sort_values("timestamp")
          .drop_duplicates("timestamp")
          .set_index("timestamp")
    )
    return (
        df.reindex(df.index.union(grid))
          .sort_index()
          .interpolate(method="index", limit_direction="both")
          .reindex(grid)
          .reset_index()
          .rename(columns={"index": "timestamp"})
    )


def interpolate_to_grid_forward(df: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """
    Linear interpolation — forward direction only.
    NaN at the START (before the sensor's first reading) are left as NaN,
    so the caller can decide how to fill them (e.g. fill with 0).
    Used for velocity: estimator starts late, so initial rows = 0 (UAV at rest).
    """
    df = (
        df.sort_values("timestamp")
          .drop_duplicates("timestamp")
          .set_index("timestamp")
    )
    return (
        df.reindex(df.index.union(grid))
          .sort_index()
          .interpolate(method="index", limit_direction="forward")
          .reindex(grid)
          .reset_index()
          .rename(columns={"index": "timestamp"})
    )


def forwardfill_to_grid(df: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Forward-fill to a uniform time grid (slow/discrete signals like battery)."""
    grid_df = pd.DataFrame({"timestamp": grid})
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    return (
        pd.merge_asof(grid_df, df, on="timestamp", direction="backward")
          .ffill()
          .bfill()
    )


def _last_active_edge(sig: np.ndarray, thresh: float):
    """Index of the last sample above `thresh` (i.e. the shutdown / landing edge)."""
    hi = sig > thresh
    if not hi.any():
        return None
    return int(len(sig) - 1 - np.argmax(hi[::-1]))


def estimate_motor_lag(motors_rel: pd.DataFrame, grid_rel: np.ndarray,
                       power_ref: np.ndarray, dt: float,
                       max_lag_s: float = 90.0,
                       refine_window_s: float = 5.0,
                       accept_refine_r: float = 0.50) -> tuple:
    """
    Estimate the constant time offset between the PX4 motor log and the ROS
    battery stream, and return the shift needed to align them.

    WHY THIS IS NEEDED
    ------------------
    PX4 writes to the SD card from boot/arm; the ROS bag is started separately
    by the operator. Zeroing each stream to its own first sample (the original
    approach) silently assumes both recordings began at the same instant. They
    did not - measured offsets on this dataset run from -30 s to +53 s, and on
    the worst flights the motor commands ended up ANTI-correlated with power.

    METHOD
    ------
    1. Coarse: align the LANDING edge - the last sample above threshold in each
       stream. Motor shutdown and the collapse of current draw are the same
       physical event, and unlike takeoff it is present in every flight here.
    2. Refine: search +/- `refine_window_s` around the coarse estimate for the
       shift maximising correlation, scored ONLY on the genuine overlap (no
       ffill/bfill padding, which would otherwise reward absurd shifts).
    3. Accept the refinement only if it reaches `accept_refine_r`. On flights
       where the ROS bag started mid-flight the motor stream contains a takeoff
       step that power does not, so overall correlation stays low no matter what
       - there the raw landing-edge estimate is the more trustworthy answer.

    Note on interpreting `r`: in cruise both signals vary by only 2-10%, so
    almost all correlation comes from the takeoff/landing steps. A low r means
    "few shared step events in this window", NOT "bad data".

    Returns
    -------
    (lag_s, r_after, r_before)
        lag_s > 0 means power occurs LATER than the motor signal, so the motor
        timestamps must be shifted FORWARD by lag_s.
    """
    probe = forwardfill_to_grid(motors_rel, grid_rel)
    m = probe[MOTOR_COLS].mean(axis=1).values.astype(float)
    p = np.asarray(power_ref, dtype=float)

    n = min(len(m), len(p))
    m, p = m[:n], p[:n]

    if n < 200 or m.std() < 1e-9 or p.std() < 1e-9:
        return 0.0, float("nan"), float("nan")

    r_before = float(np.corrcoef(p, m)[0, 1])

    def score(lag_s):
        """Correlation at `lag_s`, computed on the true overlap only."""
        sh = int(round(lag_s / dt))
        if sh >= 0:
            a, b = p[sh:], m[:n - sh] if sh else m
        else:
            a, b = p[:n + sh], m[-sh:]
        k = min(len(a), len(b))
        if k < 200:
            return -np.inf
        a, b = a[:k], b[:k]
        if a.std() < 1e-9 or b.std() < 1e-9:
            return -np.inf
        return float(np.corrcoef(a, b)[0, 1])

    # ---- 1. Coarse: landing edge ----
    # The motor edge MUST be found on the raw PX4 stream, not on the gridded
    # probe. When the bag started mid-flight the motor shutdown falls outside
    # the grid, and a gridded detector silently returns the last sample -
    # producing a plausible-looking but wrong lag.
    m_raw = motors_rel[MOTOR_COLS].mean(axis=1).values.astype(float)
    t_raw = motors_rel["timestamp"].values.astype(float)

    p_thresh = 0.5 * (np.percentile(p, 5) + np.percentile(p, 95))
    i_m = _last_active_edge(m_raw, 0.35)
    i_p = _last_active_edge(p, p_thresh)
    if i_m is None or i_p is None:
        return 0.0, r_before, r_before

    coarse = float(grid_rel[i_p] - grid_rel[0]) - float(t_raw[i_m] - t_raw[0])
    if abs(coarse) > max_lag_s:
        return 0.0, r_before, r_before

    # ---- 2. Refine locally ----
    cands = np.arange(coarse - refine_window_s, coarse + refine_window_s + 1e-9, dt)
    cands = cands[np.abs(cands) <= max_lag_s]
    scores = np.array([score(c) for c in cands])

    if not np.isfinite(scores).any():
        r_c = score(coarse)
        return float(coarse), (r_c if np.isfinite(r_c) else float("nan")), r_before

    best_i = int(np.nanargmax(scores))
    refined, r_refined = float(cands[best_i]), float(scores[best_i])

    # ---- 3. Accept refinement only if it is actually informative ----
    if r_refined >= accept_refine_r:
        return refined, r_refined, r_before
    return float(coarse), score(coarse), r_before


def plot_motor_resampling(motors_raw: pd.DataFrame, motors_resampled: pd.DataFrame,
                          output_folder: str, flight_id: str) -> None:
    """
    Plot raw motor commands (~2.29 Hz) vs resampled (20 Hz, forward-fill).
    The raw signal appears as scatter dots; the resampled as a continuous staircase line.
    This confirms the forward-fill is working correctly before merging into the dataset.
    Saved to: <output_folder>/plot_motor_resampling.png
    """
    t0 = motors_raw["timestamp"].min()
    raw_t  = motors_raw["timestamp"]  - t0
    res_t  = motors_resampled["timestamp"] - t0

    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Motor Commands — Resampling Check  [{flight_id}]\n"
        f"Raw: ~2.29 Hz (dots)   →   Resampled: 20 Hz forward-fill (line)",
        fontsize=11, fontweight="bold"
    )

    for i, (col, color) in enumerate(zip(MOTOR_COLS, MOTOR_COLORS)):
        # Resampled line first (background)
        axs[i].step(res_t, motors_resampled[col], where="post",
                    color=color, linewidth=0.8, alpha=0.7, label="resampled 20 Hz")
        # Raw dots on top
        axs[i].scatter(raw_t, motors_raw[col],
                       color="black", s=10, zorder=5, label=f"raw ~2.29 Hz  (n={len(motors_raw)})")
        axs[i].set_title(col, fontsize=9)
        axs[i].set_ylabel("Command (0–1)", fontsize=8)
        axs[i].set_ylim(-0.05, 1.05)
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(fontsize=7, loc="upper right")

    axs[-1].set_xlabel("Time (s) from flight start", fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(output_folder, "plot_motor_resampling.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {out_path}")


def plot_power_vs_motors(final: pd.DataFrame, output_folder: str, flight_id: str) -> None:
    """
    Plot resampled power alongside resampled motor commands (all at 20 Hz).
    Both signals come from the merged final dataframe — confirms they are
    temporally aligned and that motor load correlates with power consumption.
    Saved to: <output_folder>/plot_power_vs_motors_resampled.png
    """
    time_s = final["timestamp"] - final["timestamp"].min()

    fig, axs = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    fig.suptitle(
        f"Resampled Power vs Motor Commands — 20 Hz  [{flight_id}]\n"
        f"All signals from flight_resampled.csv — verifies temporal alignment",
        fontsize=11, fontweight="bold"
    )

    # ---- Power ----
    axs[0].plot(time_s, final["power"], color="#e377c2", linewidth=0.8)
    axs[0].set_title("power  (W)  =  battery_volt × |battery_curr|", fontsize=9)
    axs[0].set_ylabel("Power (W)", fontsize=8)
    axs[0].grid(True, alpha=0.3)
    mean_p = final["power"].mean()
    axs[0].axhline(mean_p, color="black", linestyle="--", linewidth=0.8,
                   label=f"mean = {mean_p:.1f} W")
    axs[0].legend(fontsize=7, loc="upper right")

    # ---- Motor commands ----
    for i, (col, color) in enumerate(zip(MOTOR_COLS, MOTOR_COLORS)):
        ax = axs[i + 1]
        ax.plot(time_s, final[col], color=color, linewidth=0.8)
        ax.set_title(col, fontsize=9)
        ax.set_ylabel("Command (0–1)", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        mean_m = final[col].mean()
        ax.axhline(mean_m, color="black", linestyle="--", linewidth=0.8,
                   label=f"mean = {mean_m:.3f}")
        ax.legend(fontsize=7, loc="upper right")

    axs[-1].set_xlabel("Time (s) from flight start", fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(output_folder, "plot_power_vs_motors_resampled.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {out_path}")


def load(folder: str, fname: str, cols: list) -> pd.DataFrame:
    path = os.path.join(folder, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)[cols]


def run(output_folder, flight_id, payload_mass, position_payload, trajectory, dt):

    print(f"[Step 4] Resampling  (dt={dt}s = {int(1/dt)} Hz)")

    # ---- Load cleaned sensor streams ----
    state = load(output_folder, "uav1_estimation_manager_uav_state_clean.csv", [
        "timestamp",
        "uav1_estimation_manager_uav_state__velocity_linear_x",
        "uav1_estimation_manager_uav_state__velocity_linear_y",
        "uav1_estimation_manager_uav_state__velocity_linear_z",
    ])
    imu = load(output_folder, "uav1_hw_api_imu_clean.csv", [
        "timestamp",
        "uav1_hw_api_imu__angular_velocity_x",
        "uav1_hw_api_imu__angular_velocity_y",
        "uav1_hw_api_imu__angular_velocity_z",
        "uav1_hw_api_imu__linear_acceleration_x",
        "uav1_hw_api_imu__linear_acceleration_y",
        "uav1_hw_api_imu__linear_acceleration_z",
    ])
    orientation = load(output_folder, "uav1_hw_api_orientation_with_rpy_clean.csv", [
        "timestamp", "roll_rad", "pitch_rad", "yaw_rad",
    ])
    altitude = load(output_folder, "uav1_mavros_altitude_clean.csv", [
        "timestamp", "uav1_mavros_altitude__local",
    ])
    battery = load(output_folder, "uav1_mrs_uav_status_uav_status_clean.csv", [
        "timestamp",
        "uav1_mrs_uav_status_uav_status__battery_volt",
        "uav1_mrs_uav_status_uav_status__battery_curr",
    ])
    motors = load(output_folder, "uav1_motor_commands_clean.csv", [
        "timestamp",
        "motor_1_front_right",
        "motor_2_rear_left",
        "motor_3_front_left",
        "motor_4_rear_right",
    ])

    # ---- Common time window (ROS streams only) ----
    streams = [state, imu, orientation, altitude, battery]
    t0   = max(df["timestamp"].min() for df in streams)
    t1   = min(df["timestamp"].max() for df in streams)
    grid = np.arange(t0, t1, dt)
    print(f"  Time window : {t0:.3f} → {t1:.3f} s  ({len(grid)} samples)")

    # ---- Resample ROS streams ----
    state_r       = interpolate_to_grid(state,       grid)
    imu_r         = interpolate_to_grid(imu,         grid)
    orientation_r = interpolate_to_grid(orientation, grid)
    altitude_r    = interpolate_to_grid(altitude,    grid)
    battery_r     = forwardfill_to_grid(battery,     grid)

    # ---- Resample motor commands (PX4 time ≠ ROS Unix time) ----
    # PX4 timestamps are seconds since boot (~0–600 s).
    # ROS timestamps are Unix epoch (~1.7 × 10⁹ s).
    # They cannot be aligned directly — normalize both to relative time (0 = start).
    #
    # NOTE: zeroing each stream to its own first sample is NOT sufficient. The
    # PX4 SD-card log and the ROS bag are started independently, so the two
    # origins differ by an unknown constant (0–69 s on this dataset). The lag is
    # therefore measured from the data itself by cross-correlating mean motor
    # command against power, and the motor stream is shifted to match.
    motors_rel  = motors.copy()
    motors_rel["timestamp"] = motors_rel["timestamp"] - motors_rel["timestamp"].min()
    grid_rel    = grid - grid[0]   # relative grid: 0 → flight duration

    power_ref = (
        battery_r["uav1_mrs_uav_status_uav_status__battery_volt"]
        * battery_r["uav1_mrs_uav_status_uav_status__battery_curr"].abs()
    ).values
    lag_s, r_lag, r_zero = estimate_motor_lag(motors_rel, grid_rel, power_ref, dt)

    if lag_s != 0.0:
        motors_rel["timestamp"] = motors_rel["timestamp"] + lag_s

    motors_r    = forwardfill_to_grid(motors_rel, grid_rel)   # forward-fill, slow signal
    print(f"  Motors aligned: PX4 relative time  "
          f"({motors_rel['timestamp'].min():.1f} → {motors_rel['timestamp'].max():.1f} s)  "
          f"vs grid ({grid_rel[0]:.1f} → {grid_rel[-1]:.1f} s)")
    print(f"  Motor↔power lag: {lag_s:+.2f} s   "
          f"(corr {r_zero:+.3f} → {r_lag:+.3f})")
    # Does this window contain a takeoff, or did the bag start mid-flight?
    # Correlation is driven almost entirely by the takeoff/landing steps: in
    # cruise both signals vary by only 2-10%. A partial-coverage flight will
    # therefore show a low r even when correctly aligned — that is expected,
    # not a sign of bad data, so it is reported rather than warned about.
    _p = power_ref
    has_takeoff = bool(_p[:len(_p) // 2].min() < 0.5 * (np.percentile(_p, 5)
                                                        + np.percentile(_p, 95)))
    if not has_takeoff:
        print(f"  Note [{flight_id}]: no takeoff in the ROS window (bag started "
              f"mid-flight) — aligned on the landing edge; low corr is expected.")

    align_report = {"flight_id": flight_id, "lag_s": round(lag_s, 2),
                    "corr_before": round(r_zero, 4), "corr_after": round(r_lag, 4),
                    "has_takeoff": has_takeoff}
    pd.DataFrame([align_report]).to_csv(
        os.path.join(output_folder, "motor_alignment.csv"), index=False)

    # ---- Merge ----
    final = pd.DataFrame({"timestamp": grid})
    for df_r in [state_r, imu_r, orientation_r, altitude_r, battery_r, motors_r]:
        final = final.merge(df_r.drop(columns=["timestamp"]), left_index=True, right_index=True)

    # ---- Derived + metadata ----
    final["power"] = (
        final["uav1_mrs_uav_status_uav_status__battery_volt"]
        * final["uav1_mrs_uav_status_uav_status__battery_curr"].abs()
    )
    final["speed_3d"] = np.sqrt(
        final["uav1_estimation_manager_uav_state__velocity_linear_x"] ** 2
        + final["uav1_estimation_manager_uav_state__velocity_linear_y"] ** 2
        + final["uav1_estimation_manager_uav_state__velocity_linear_z"] ** 2
    )
    final["speed_horizontal"] = np.sqrt(
        final["uav1_estimation_manager_uav_state__velocity_linear_x"] ** 2
        + final["uav1_estimation_manager_uav_state__velocity_linear_y"] ** 2
    )

    # ---- Motor imbalance features (computed after resampling) ----
    # Motor physical positions:
    #     M3 front_left  |  M1 front_right
    #     M2 rear_left   |  M4 rear_right
    #
    # front_rear_imbalance > 0 → front motors working harder  (front payload)
    # front_rear_imbalance < 0 → rear motors working harder   (rear payload)
    # diagonal_imbalance   > 0 → M3+M4 diagonal working harder (diagonal payload)
    final["front_rear_imbalance"] = (
        (final["motor_1_front_right"] + final["motor_3_front_left"]) / 2
        - (final["motor_2_rear_left"] + final["motor_4_rear_right"]) / 2
    )
    final["diagonal_imbalance"] = (
        (final["motor_3_front_left"] + final["motor_4_rear_right"]) / 2
        - (final["motor_1_front_right"] + final["motor_2_rear_left"]) / 2
    )

    final.insert(0, "flight_id",        flight_id)
    final.insert(1, "payload_mass",     payload_mass)
    final.insert(2, "position_payload", position_payload)
    final.insert(3, "trajectory",       trajectory)

    # ---- Diagnostic plots ----
    # Pass relative-time versions so both axes share the same reference (0 = flight start)
    plot_motor_resampling(motors_rel, motors_r, output_folder, flight_id)
    plot_power_vs_motors(final, output_folder, flight_id)

    # ---- Save ----
    out_path = os.path.join(output_folder, "flight_resampled.csv")
    final.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}  ({len(final)} rows × {len(final.columns)} cols)\n")


if __name__ == "__main__":
    print("Run via: python run_flight.py <test_folder>")
