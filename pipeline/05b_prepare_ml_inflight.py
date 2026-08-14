"""
Step 5b – ML preparation, IN-FLIGHT ROWS ONLY.

Same pipeline as 05_prepare_ml.py, with one difference: rows recorded while the
UAV is on the ground are removed before splitting.

WHY THIS VARIANT EXISTS
-----------------------
A full recording contains two very different regimes: idle on the ground
(~25 W) and flight (~750 W). That 30x gap dominates the variance of the target,
so a model is rewarded almost entirely for telling "on the ground" from "flying"
- an easy task. Measured by leave-one-flight-out on this dataset:

    R2 over all rows       =  0.947
    R2 over in-flight rows = -0.856   (negative on 9 of 13 flights)

The headline 0.96 therefore describes the idle/flight separation, not the
ability to predict power during flight. This file produces the stricter dataset
so both numbers can be reported side by side.

Threshold: per flight, the midpoint between the 5th and 95th percentile of that
flight's power. Idle sits near 25 W and flight near 700 W, so the gap is wide
and the exact cut is not sensitive. Using a per-flight value (rather than one
global constant) keeps it robust to differences in payload and battery state.

Input  : flights/*/flight_resampled.csv
Output : ML/ml_data_inflight/train.csv, test.csv, scaler.pkl, feature_cols.txt
         ML/ml_data_inflight/inflight_filter_report.csv

Run: python 05b_prepare_ml_inflight.py

Then point the downstream steps at this folder:
    ML_DATA_DIR=ML/ml_data_inflight python 06_train_xgboost.py
    ML_DATA_DIR=ML/ml_data_inflight python 07_learning_curves.py
    ML_DATA_DIR=ML/ml_data_inflight python 08_plot_power_time.py
"""

import os
import pickle
import importlib.util

import numpy as np
import pandas as pd

# ---- Reuse everything from step 05 so the two variants cannot drift apart ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "prep05", os.path.join(SCRIPT_DIR, "05_prepare_ml.py"))
p5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p5)

OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "ML", "ml_data_inflight")

# Fraction between the 5th and 95th power percentile used as the idle/flight cut.
THRESHOLD_FRAC = 0.5


def filter_inflight(df: pd.DataFrame) -> tuple:
    """Drop ground/idle rows, per flight. Returns (filtered_df, report_df)."""
    print(f"\n  Removing idle rows (per-flight threshold, "
          f"frac={THRESHOLD_FRAC} between P5 and P95)")

    keep, rows = [], []
    for fid, g in df.groupby("flight_id", sort=True):
        lo, hi = np.percentile(g["power"], 5), np.percentile(g["power"], 95)
        thr = lo + THRESHOLD_FRAC * (hi - lo)
        sel = g["power"] > thr
        keep.append(g[sel])

        rows.append({
            "flight_id":   fid,
            "rows_before": len(g),
            "rows_after":  int(sel.sum()),
            "rows_dropped": int((~sel).sum()),
            "pct_dropped": round(100.0 * (~sel).sum() / len(g), 1),
            "threshold_w": round(float(thr), 1),
            "mean_power_after_w": round(float(g.loc[sel, "power"].mean()), 1),
        })
        print(f"    {fid}: {len(g):6d} → {int(sel.sum()):6d} rows "
              f"({rows[-1]['pct_dropped']:4.1f}% dropped, cut at {thr:6.1f} W)")

    out    = pd.concat(keep, ignore_index=True)
    report = pd.DataFrame(rows)
    print(f"\n  Total: {len(df)} → {len(out)} rows "
          f"({100.0 * (len(df) - len(out)) / len(df):.1f}% removed)")
    return out, report


def run(flights_folder=None):
    flights_folder = flights_folder or p5.FLIGHTS_DIR
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("\n" + "=" * 55)
    print("  [Step 5b] ML Preparation — IN-FLIGHT ROWS ONLY")
    print("=" * 55)

    df = p5.load_all_flights(flights_folder)
    df, report = filter_inflight(df)
    df = p5.one_hot_encode(df, p5.CATEGORICAL_COLS)

    # Identical split settings to step 05, so the two variants stay comparable.
    train, test = p5.shuffle_and_split(df,
                                       test_size=p5.TEST_SIZE,
                                       seed=p5.RANDOM_SEED,
                                       test_flights_override=p5.TEST_FLIGHTS)

    feature_cols = [c for c in train.columns if c not in p5.EXCLUDE_COLS]
    print(f"\n  Feature columns ({len(feature_cols)})")

    train, test, scaler = p5.scale(train, test, feature_cols)

    train_path    = os.path.join(OUTPUT_FOLDER, "train.csv")
    test_path     = os.path.join(OUTPUT_FOLDER, "test.csv")
    scaler_path   = os.path.join(OUTPUT_FOLDER, "scaler.pkl")
    features_path = os.path.join(OUTPUT_FOLDER, "feature_cols.txt")
    report_path   = os.path.join(OUTPUT_FOLDER, "inflight_filter_report.csv")

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    report.to_csv(report_path, index=False)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(features_path, "w") as f:
        f.write("\n".join(feature_cols))

    print(f"\n  Saved → {train_path}  ({len(train)} rows × {len(train.columns)} cols)")
    print(f"  Saved → {test_path}   ({len(test)} rows × {len(test.columns)} cols)")
    print(f"  Saved → {scaler_path}")
    print(f"  Saved → {features_path}  ({len(feature_cols)} features)")
    print(f"  Saved → {report_path}")
    print("\n  Done. Run step 06 with ML_DATA_DIR=ML/ml_data_inflight\n")


if __name__ == "__main__":
    run()
