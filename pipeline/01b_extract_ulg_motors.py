"""
Step 1b – Extract motor commands from a PX4 .ulg log file.

Reads actuator_motors.control[0-3] (Motor 1–4 normalized commands)
and saves them as a CSV in the flight output folder, ready for the
outlier removal and resampling steps.

Input  : <ULG_FILE>                                  (PX4 .ulg log)
Output : <OUTPUT_FOLDER>/uav1_motor_commands.csv     (4 motor columns, ~2.29 Hz)

Run standalone : python 01b_extract_ulg_motors.py
Run via runner : called automatically by run_flight.py

Notes:
  - Motor commands are normalized (0–1): 0 = off, 1 = full throttle
  - Acquisition rate is ~2.29 Hz — much lower than the 20 Hz pipeline
  - Resampling (forward-fill) is handled in step 4 (04_resampling.py)
  - Imbalance features (front_rear, diagonal) are computed in step 4
    after resampling, not here

Motor physical positions on the UAV frame:
    M3 front_left  |  M1 front_right
    M2 rear_left   |  M4 rear_right
"""

import os
import pandas as pd
from pyulog import ULog

# Motor field → column name mapping
MOTORS = {
    "control[0]": "motor_1_front_right",
    "control[1]": "motor_2_rear_left",
    "control[2]": "motor_3_front_left",
    "control[3]": "motor_4_rear_right",
}

OUTPUT_FILENAME = "uav1_motor_commands.csv"


def run(ulg_file, output_folder):

    print(f"[Step 1b] Extracting motor commands from: {ulg_file}")

    if not os.path.exists(ulg_file):
        raise FileNotFoundError(f"ULG file not found: {ulg_file}")

    os.makedirs(output_folder, exist_ok=True)

    # ---- Load .ulg ----
    ulog = ULog(ulg_file)

    motor_data = None
    for d in ulog.data_list:
        if d.name == "actuator_motors" and d.multi_id == 0:
            motor_data = d
            break

    if motor_data is None:
        raise RuntimeError("Topic 'actuator_motors' not found in the .ulg file.")

    # ---- Build dataframe ----
    df = pd.DataFrame()
    df["timestamp"] = motor_data.data["timestamp"] * 1e-6   # µs → seconds

    for field, label in MOTORS.items():
        df[label] = motor_data.data[field]

    # ---- Summary ----
    duration   = df["timestamp"].max() - df["timestamp"].min()
    sample_hz  = len(df) / duration if duration > 0 else 0
    print(f"  Samples  : {len(df)}")
    print(f"  Duration : {duration:.1f} s")
    print(f"  Rate     : ~{sample_hz:.2f} Hz")
    print(f"  Motor command ranges (normalized 0–1):")
    for label in MOTORS.values():
        print(f"    {label}: "
              f"min={df[label].min():.4f}  "
              f"max={df[label].max():.4f}  "
              f"mean={df[label].mean():.4f}")

    # ---- Save ----
    out_path = os.path.join(output_folder, OUTPUT_FILENAME)
    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}  ({len(df)} rows × {len(df.columns)} cols)\n")


if __name__ == "__main__":
    print("Run via: python run_flight.py <test_folder>")
