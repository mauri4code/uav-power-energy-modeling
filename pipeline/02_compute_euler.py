"""
Step 2 – Convert quaternion orientation to roll / pitch / yaw (radians and degrees).

Input  : <OUTPUT_FOLDER>/uav1_hw_api_orientation.csv
Output : <OUTPUT_FOLDER>/uav1_hw_api_orientation_with_rpy.csv

Run standalone : python 02_compute_euler.py
Run via runner : called automatically by run_flight.py
"""

import os
import numpy as np
import pandas as pd

INPUT_FILE  = "uav1_hw_api_orientation.csv"
OUTPUT_FILE = "uav1_hw_api_orientation_with_rpy.csv"

QX = "uav1_hw_api_orientation__quaternion_x"
QY = "uav1_hw_api_orientation__quaternion_y"
QZ = "uav1_hw_api_orientation__quaternion_z"
QW = "uav1_hw_api_orientation__quaternion_w"


def quat_to_euler(x, y, z, w):
    roll  = np.arctan2(2.0 * (w * x + y * z),  1.0 - 2.0 * (x**2 + y**2))
    sinp  = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw   = np.arctan2(2.0 * (w * z + x * y),  1.0 - 2.0 * (y**2 + z**2))
    return roll, pitch, yaw


def run(output_folder):

    in_path  = os.path.join(output_folder, INPUT_FILE)
    out_path = os.path.join(output_folder, OUTPUT_FILE)

    print(f"[Step 2] Computing Euler angles from {in_path}")

    df = pd.read_csv(in_path)
    roll, pitch, yaw = quat_to_euler(
        df[QX].to_numpy(), df[QY].to_numpy(),
        df[QZ].to_numpy(), df[QW].to_numpy(),
    )

    df["roll_rad"]   = roll
    df["pitch_rad"]  = pitch
    df["yaw_rad"]    = yaw
    df["roll_deg"]   = np.degrees(roll)
    df["pitch_deg"]  = np.degrees(pitch)
    df["yaw_deg"]    = np.degrees(yaw)

    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}  ({len(df)} rows)\n")


if __name__ == "__main__":
    print("Run via: python run_flight.py <test_folder>")
