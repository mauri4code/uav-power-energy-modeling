"""
Full preprocessing pipeline for one flight.

Usage:
    python run_flight.py 2026_04_29_TEST3

Reads flight_config.json from the given folder and runs all steps:
    1.  Export bag topics         → flights/<ID>/*.csv
    1b. Extract ULG motor cmds    → flights/<ID>/uav1_motor_commands.csv
    2.  Compute Euler angles      → flights/<ID>/uav1_hw_api_orientation_with_rpy.csv
    3.  Outlier removal           → flights/<ID>/*_clean.csv
    4.  Resample & merge          → flights/<ID>/flight_resampled.csv
"""

import sys
import json
import importlib
from pathlib import Path

# ---- Pipeline settings (shared across all flights, rarely change) ----
IQR_K = 3.0    # IQR multiplier for outlier removal (higher = less aggressive)
DT    = 0.05   # resampling period in seconds (0.05 s = 20 Hz)


def load_config(folder: str) -> dict:
    """
    Reads flight_config.json from the given test folder.
    Returns a dictionary with all flight metadata (bag file, flight ID, etc.)
    Raises a clear error if the file is not found.
    """
    json_path = Path(folder) / "flight_config.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"No flight_config.json found in: {folder}\n"
            f"Expected path: {json_path.resolve()}"
        )
    with open(json_path) as f:
        return json.load(f)


def main():
    # ---- Check that a folder name was passed as argument ----
    # e.g. python run_flight.py 2026_04_29_TEST3
    if len(sys.argv) != 2:
        print("Usage  : python run_flight.py <test_folder>")
        print("Example: python run_flight.py 2026_04_29_TEST3")
        sys.exit(1)

    # ---- Load flight metadata from JSON ----
    folder = sys.argv[1]          # folder name passed in the command line
    cfg    = load_config(folder)  # reads flight_config.json inside that folder

    # Extract each field from the JSON into a named variable
    bag_file         = cfg["BAG_FILE"]          # path to the ROS .bag file
    ulg_file         = cfg["ULG_FILE"]          # path to the PX4 .ulg log file
    flight_id        = cfg["FLIGHT_ID"]         # label used in the final dataset (e.g. F03)
    payload_mass     = cfg["PAYLOAD_MASS"]      # payload weight in grams
    position_payload = cfg["POSITION_PAYLOAD"]  # payload position (none/front/diagonal/...)
    trajectory       = cfg["TRAJECTORY"]        # trajectory flown (trajectory_1/trajectory_2)
    output_folder    = f"flights/{flight_id}"   # output folder is auto-built from the flight ID

    # ---- Print a summary before starting ----
    print("=" * 55)
    print(f"  FOLDER   : {folder}")
    print(f"  BAG FILE : {bag_file}")
    print(f"  ULG FILE : {ulg_file}")
    print(f"  FLIGHT ID: {flight_id}")
    print(f"  OUTPUT   : {output_folder}/")
    print("=" * 55)

    # ---- Load all pipeline step modules ----
    # importlib is used because module names start with numbers (01_, 02_, ...)
    # which Python does not allow with a regular "import" statement.
    step1  = importlib.import_module("01_export_bag")
    step1b = importlib.import_module("01b_extract_ulg_motors")
    step2  = importlib.import_module("02_compute_euler")
    step3  = importlib.import_module("03_outlier_removal")
    step4  = importlib.import_module("04_resampling")

    # ---- Run each step in order, passing the config values explicitly ----
    step1.run(bag_file=bag_file, output_folder=output_folder)       # ROS bag → CSVs
    step1b.run(ulg_file=ulg_file, output_folder=output_folder)      # PX4 .ulg → motor CSV
    step2.run(output_folder=output_folder)                          # quaternion → roll/pitch/yaw
    step3.run(output_folder=output_folder, k=IQR_K)                 # outlier removal
    step4.run(                                                       # resample + merge
        output_folder=output_folder,
        flight_id=flight_id,
        payload_mass=payload_mass,
        position_payload=position_payload,
        trajectory=trajectory,
        dt=DT,
    )

    # ---- Done ----
    print("=" * 55)
    print(f"  Pipeline complete.")
    print(f"  Final dataset → {output_folder}/flight_resampled.csv")
    print("=" * 55)


if __name__ == "__main__":
    main()
