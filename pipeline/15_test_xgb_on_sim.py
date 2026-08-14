"""
Step 15 – Test the Step-12 XGBoost model on the synthetic SIM flights.

Step 12's winning-family XGBoost is a FLIGHT-LEVEL model: it maps six in-flight
AVERAGED features -> average airborne power (one number per flight). The SIM
flights (built in step 14) are made of intact 20 s segments, each copied from a
real flight, so each segment is effectively a mini-flight at one operating
condition.

So the faithful test is per SEGMENT:
  1. Train the Step-12 XGBoost ("all 6") on all 14 real flights (flight-level).
  2. For each SIM flight, split it back into its segments (segment_map.csv),
     compute the same in-flight-averaged features per segment, predict the
     segment's average power, and compare to the segment's ACTUAL average power.
  3. Save a predicted-vs-actual plot + a CSV inside each SIM subfolder.

Input  : flights/F*/flight_resampled.csv        (training, 14 real flights)
         SIM_FLIGHTS/*/flight_resampled.csv      (+ segment_map.csv)
Output : SIM_FLIGHTS/<variant>/pred_vs_actual_power.png
         SIM_FLIGHTS/<variant>/segment_predictions.csv

Run: python 15_test_xgb_on_sim.py
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FLIGHTS_DIR = os.path.join(SCRIPT_DIR, "flights")
SIM_DIR     = os.path.join(SCRIPT_DIR, "SIM_FLIGHTS")

MOTOR_COLS = ["motor_1_front_right", "motor_2_rear_left",
              "motor_3_front_left",  "motor_4_rear_right"]
ALT = "uav1_mavros_altitude__local"

# The Step-12 "all 6, XGBoost" feature set and hyper-parameters (verbatim).
FEATURES = ["mass", "motor", "alt", "speed", "pitch", "roll"]


def make_xgb():
    return XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.1,
                        subsample=1.0, reg_lambda=1.0, n_jobs=-1, random_state=42)


def inflight_features(d):
    """
    Reduce a block of raw rows to the six averaged features + average power,
    using exactly the Step-12 recipe: keep only 'airborne' rows, where the
    threshold is the midpoint of the 5th and 95th power percentiles.
    """
    thr = 0.5 * (np.percentile(d["power"], 5) + np.percentile(d["power"], 95))
    g = d[d["power"] > thr]
    if len(g) == 0:
        g = d
    return {
        "mass":  d["payload_mass"].iloc[0],
        "motor": g[MOTOR_COLS].mean(axis=1).mean(),
        "alt":   g[ALT].mean(),
        "speed": g["speed_3d"].mean(),
        "pitch": g["pitch_rad"].mean(),
        "roll":  g["roll_rad"].mean(),
        "power": g["power"].mean(),
        "n_rows": len(g),
    }


def build_training_table():
    """One row per real flight (Step-12 build_table, reduced to what we need)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(FLIGHTS_DIR, "F*", "flight_resampled.csv"))):
        d = pd.read_csv(f)
        r = inflight_features(d)
        r["flight"] = d["flight_id"].iloc[0]
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("flight").reset_index(drop=True)
    print(f"  Training table: {len(df)} real flights")
    return df


def segment_table(sim_csv, seg_map_csv):
    """One row per SIM segment: averaged features + actual average power."""
    df  = pd.read_csv(sim_csv)
    seg = pd.read_csv(seg_map_csv)
    rows = []
    for r in seg.itertuples():
        block = df.iloc[int(r.dst_start_idx):int(r.dst_start_idx) + int(r.n_samples)]
        feat = inflight_features(block)
        feat["source_flight"] = r.source_flight
        feat["t_start_s"] = r.t_start_s
        feat["t_end_s"]   = r.t_end_s
        rows.append(feat)
    return df, pd.DataFrame(rows)


def plot_variant(raw, seg_pred, variant, out_png, pred_col="pred", mode_label=""):
    """Predicted vs actual average power per segment, over the flight timeline."""
    t_raw = raw["timestamp"].values
    mae = mean_absolute_error(seg_pred["power"], seg_pred[pred_col])
    r2  = r2_score(seg_pred["power"], seg_pred[pred_col])

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [3, 1]})

    # ---- left: time series ----
    ax0.plot(t_raw, raw["power"].values, color="#cbd5e1", lw=0.7,
             label="raw power (20 Hz)", zorder=1)
    # step lines for actual & predicted segment averages
    edges = np.append(seg_pred["t_start_s"].values, seg_pred["t_end_s"].values[-1])
    ax0.stairs(seg_pred["power"].values, edges, color="#1e293b", lw=2.2,
               label="actual segment avg", baseline=None, zorder=3)
    ax0.stairs(seg_pred[pred_col].values, edges, color="#dc2626", lw=2.2, ls="--",
               label="XGBoost predicted", baseline=None, zorder=4)
    for _, s in seg_pred.iterrows():
        ax0.axvline(s["t_start_s"], color="k", ls=":", lw=0.5, alpha=0.4)
        ax0.text((s["t_start_s"] + s["t_end_s"]) / 2, ax0.get_ylim()[1],
                 s["source_flight"], ha="center", va="bottom", fontsize=7)
    ax0.set_xlabel("time [s]"); ax0.set_ylabel("power [W]")
    ax0.set_title(f"{variant} — XGBoost (Step 12, all-6){mode_label}\n"
                  f"MAE = {mae:.1f} W    R² = {r2:.3f}", fontweight="bold")
    ax0.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    ax0.grid(alpha=0.3)

    # ---- right: predicted vs actual scatter ----
    lo = min(seg_pred["power"].min(), seg_pred[pred_col].min()) - 20
    hi = max(seg_pred["power"].max(), seg_pred[pred_col].max()) + 20
    ax1.plot([lo, hi], [lo, hi], color="#94a3b8", lw=1, ls="--")
    ax1.scatter(seg_pred["power"], seg_pred[pred_col], s=55, color="#dc2626",
                edgecolor="white", zorder=3)
    for _, s in seg_pred.iterrows():
        ax1.annotate(s["source_flight"], (s["power"], s[pred_col]),
                     fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi); ax1.set_aspect("equal")
    ax1.set_xlabel("actual avg power [W]"); ax1.set_ylabel("predicted [W]")
    ax1.set_title("per-segment", fontweight="bold")
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    return mae, r2


def main():
    print("\n[Step 15] Test Step-12 XGBoost on SIM flights")

    # ---- build the 14-flight training table ----
    train = build_training_table()

    # (a) leakage version: one model trained on ALL 14 flights
    model_all = make_xgb().fit(train[FEATURES], train["power"])

    # (b) leave-one-flight-out: one model per held-out flight, trained on the
    #     other 13. To predict a segment whose source is Fk, use the model that
    #     never saw Fk -> no leakage.
    lofo_models = {}
    for fk in train["flight"]:
        sub = train[train["flight"] != fk]
        lofo_models[fk] = make_xgb().fit(sub[FEATURES], sub["power"])
    print(f"  Fitted 1 all-flights model + {len(lofo_models)} leave-one-out models")

    # ---- score every SIM variant ----
    variants = sorted(glob.glob(os.path.join(SIM_DIR, "*", "flight_resampled.csv")))
    if not variants:
        raise FileNotFoundError(f"No SIM flights under {SIM_DIR}/ (run step 14 first)")

    print("\n  variant               |  trained-on-all  |  leave-one-out (honest)")
    print("                        |  MAE(W)    R²    |  MAE(W)    R²    segments")
    for sim_csv in variants:
        folder  = os.path.dirname(sim_csv)
        variant = os.path.basename(folder)
        seg_map = os.path.join(folder, "segment_map.csv")

        raw, seg = segment_table(sim_csv, seg_map)

        # (a) leakage prediction
        seg["pred_all"] = model_all.predict(seg[FEATURES]).round(1)
        # (b) leave-one-flight-out prediction (per-segment source flight held out)
        seg["pred_lofo"] = [
            round(float(lofo_models[src].predict(seg.loc[[i], FEATURES])[0]), 1)
            for i, src in zip(seg.index, seg["source_flight"])
        ]

        # ---- plots (both versions saved in the subfolder) ----
        png_all  = os.path.join(folder, "pred_vs_actual_power_trainedALL.png")
        png_lofo = os.path.join(folder, "pred_vs_actual_power_LOFO.png")
        mae_a, r2_a = plot_variant(raw, seg, variant, png_all,  pred_col="pred_all",
                                   mode_label="  [trained on ALL 14 — leaky]")
        mae_l, r2_l = plot_variant(raw, seg, variant, png_lofo, pred_col="pred_lofo",
                                   mode_label="  [leave-one-flight-out — honest]")

        # ---- CSV with both predictions ----
        out_csv = os.path.join(folder, "segment_predictions.csv")
        cols = (["source_flight", "t_start_s", "t_end_s"] + FEATURES
                + ["power", "pred_all", "pred_lofo", "n_rows"])
        seg[cols].rename(columns={"power": "actual_power"}).to_csv(out_csv, index=False)

        print(f"  {variant:20s}  |  {mae_a:6.1f}  {r2_a:6.3f} |  {mae_l:6.1f}  {r2_l:6.3f}   {len(seg)}")
        print(f"      saved -> {os.path.basename(png_all)}, {os.path.basename(png_lofo)}, {os.path.basename(out_csv)}")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
