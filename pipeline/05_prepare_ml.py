"""
Step 5 – ML preparation: load all flights, encode, shuffle, split, scale.
dsa
Input  : flights/*/flight_resampled.csv   (all available flights)
Output : ML/ml_data/train.csv                (training set, scaled)
         ML/ml_data/test.csv                 (test set, scaled)
         ML/ml_data/scaler.pkl               (fitted StandardScaler — reuse for new flights)
         ML/ml_data/feature_cols.txt         (list of feature columns used)

Run standalone : python 05_prepare_ml.py
Run via runner : called automatically by run_flight.py (when all flights ready)

Notes:
  - Categorical columns are One-Hot Encoded into separate binary columns
  - Flights are shuffled BEFORE  (not individual rows)dsa
  - Scaler is fit on TRAIN data only → prevents data leakaged
  - TARGET column (power) is NOT scaled → output stays interpretable in Watts
  - timestamp and flight_id are kept in CSV for reference but NOT used as features
"""

import os
import glob
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---- Paths (relative to this script, works from any working directory) ----
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
FLIGHTS_DIR   = os.path.join(SCRIPT_DIR, "flights")       # flights/ sits next to this script
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "ML", "ml_data") # outputs go into ML/ml_data/

# ---- Settings ----
RANDOM_SEED      = 42
TEST_SIZE        = 0.2        # 20% of flights → test set (used only when TEST_FLIGHTS is None)

# ---- Override: set to a list of flight IDs to fix the test set, e.g. ["F06"] ----
# Set to None to fall back to the automatic 80/20 random split.
TEST_FLIGHTS     =  None

# ---- Column roles ----
TARGET_COL       = "power"
CATEGORICAL_COLS = [] #["position_payload", "trajectory"] # RESTORE_PREVIOUS_00
EXCLUDE_COLS     = [
    "flight_id", "timestamp", TARGET_COL,             # reference / target
    "uav1_mrs_uav_status_uav_status__battery_volt",   # removed — power = V×I would make prediction trivial
    "uav1_mrs_uav_status_uav_status__battery_curr",   # removed — same reason
    "trajectory",
    "speed_horizontal",
    #"speed_3d",
    "position_payload",    # RESTORE_PREVIOUS_00  JUST DELETE THE LINE IF WE WANT TO USE TRAJECTORY
]


# ===========================================================
#  HELPERS
# ===========================================================

def load_all_flights(flights_folder="flights"):
    """Load every flight_resampled.csv and concatenate into one dataframe."""
    pattern = os.path.join(flights_folder, "*", "flight_resampled.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No flight_resampled.csv files found under '{flights_folder}/'.\n"
            f"Run the preprocessing pipeline first (run_flight.py)."
        )

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  Loaded: {f}  ({len(df)} rows, flight={df['flight_id'].iloc[0]})")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total: {len(combined)} rows from {len(files)} flights")
    return combined


def one_hot_encode(df, categorical_cols):
    """One-Hot Encode categorical columns into separate binary columns."""
    print(f"\n  One-Hot Encoding: {categorical_cols}")
    df = pd.get_dummies(df, columns=categorical_cols, dtype=int)
    new_cols = [c for c in df.columns if any(c.startswith(cat) for cat in categorical_cols)]
    print(f"  New binary columns created: {new_cols}")
    return df


def shuffle_and_split(df, test_size=0.2, seed=42, test_flights_override=None):
    """
    Split flights into train and test at flight level (never splits individual rows).

    If test_flights_override is given (e.g. ["F06"]), those flights go to test
    and all others go to train — ignores test_size and seed.

    Otherwise, shuffle the full flight list and take the first test_size fraction
    as the test set (reproducible with random seed).
    """
    flights = df["flight_id"].unique().tolist()

    if test_flights_override:
        test_flights  = test_flights_override
        train_flights = [f for f in flights if f not in test_flights]
        print(f"\n  Split mode     : manual override")
    else:
        random.seed(seed)
        random.shuffle(flights)
        n_test        = max(1, round(len(flights) * test_size))
        test_flights  = flights[:n_test]
        train_flights = flights[n_test:]
        print(f"\n  Split mode     : random 80/20 (seed={seed})")

    print(f"  Total flights  : {len(flights)}")
    print(f"  Train flights  : {sorted(train_flights)}")
    print(f"  Test flights   : {test_flights}")

    train = df[df["flight_id"].isin(train_flights)].copy()
    test  = df[df["flight_id"].isin(test_flights)].copy()

    print(f"\n  Train rows : {len(train)}")
    print(f"  Test rows  : {len(test)}")
    return train, test


def scale(train, test, feature_cols):
    """
    Fit StandardScaler on training data only, then apply to both train and test.
    TARGET column (power) is intentionally NOT scaled.
    """
    print(f"\n  Fitting StandardScaler on {len(feature_cols)} feature columns...")
    scaler = StandardScaler()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    test[feature_cols]  = scaler.transform(test[feature_cols])
    print(f"  Scaler fitted. Mean and std computed from training data only.")
    return train, test, scaler


# ===========================================================
#  MAIN
# ===========================================================

def run(flights_folder=None):
    flights_folder = flights_folder or FLIGHTS_DIR
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("\n" + "=" * 55)
    print("  [Step 5] ML Preparation")
    print("=" * 55)

    # ---- 1. Load all flights ----
    df = load_all_flights(flights_folder)

    # ---- 2. One-Hot Encode categoricals ----
    df = one_hot_encode(df, CATEGORICAL_COLS)

    # ---- 3. Split flights into train / test ----
    train, test = shuffle_and_split(df, test_size=TEST_SIZE, seed=RANDOM_SEED,
                                    test_flights_override=TEST_FLIGHTS)

    # ---- 4. Define feature columns ----
    feature_cols = [c for c in train.columns if c not in EXCLUDE_COLS]
    print(f"\n  Feature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"    {col}")

    # ---- 5. Scale features (train only → apply to both) ----
    train, test, scaler = scale(train, test, feature_cols)

    # ---- 6. Save outputs ----
    train_path   = os.path.join(OUTPUT_FOLDER, "train.csv")
    test_path    = os.path.join(OUTPUT_FOLDER, "test.csv")
    scaler_path  = os.path.join(OUTPUT_FOLDER, "scaler.pkl")
    features_path = os.path.join(OUTPUT_FOLDER, "feature_cols.txt")

    train.to_csv(train_path, index=False)
    test.to_csv(test_path,   index=False)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    with open(features_path, "w") as f:
        f.write("\n".join(feature_cols))

    print(f"\n  Saved → {train_path}  ({len(train)} rows × {len(train.columns)} cols)")
    print(f"  Saved → {test_path}   ({len(test)} rows × {len(test.columns)} cols)")
    print(f"  Saved → {scaler_path}  (StandardScaler fitted on train)")
    print(f"  Saved → {features_path}  ({len(feature_cols)} features)")
    print("\n  Done. Ready for model training.\n")


if __name__ == "__main__":
    run()
