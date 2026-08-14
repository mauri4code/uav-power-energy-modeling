"""
LSTM power-prediction study on the +/-12 s mission window, 1 Hz.

STANDALONE. Reads only flights/F*/flight_resampled.csv (found by walking up),
writes only into this folder. PyTorch.

WHY / HOW
---------
The state of the art uses LSTM / Bi-LSTM sequence models for UAV power (Muli 2022,
Ait Saadi 2025). This script tries the same idea on our data so it can be compared
head-to-head with the tree/linear models, under the SAME protocol:
  * 1 Hz, +/-12 s mission window, per-flight cruise threshold (P5/P95 midpoint)
  * leave-one-flight-out: train on 13 flights, predict the held-out 14th, x14
  * features standardised on the training flights only (target too, inverted back)

The LSTM sees a sliding window of SEQ_LEN seconds of telemetry and predicts the
power at the last step. Loss is L1 (absolute error), matching the metric reported
and the XGBoost objective.

Feature set: the dynamic-state families (motors, mass, velocity, speed, IMU,
orientation, altitude) --- LSTMs benefit from temporal context, so it is given the
broad set rather than only motors+mass.

Output (into this folder):
  lstm_summary.json         metrics (overall + cruise) + per-flight
  lstm_per_flight.csv       per held-out flight R2 / MAE
  lstm_time.png             measured vs predicted power for a few held-out flights

Run:  python lstm_study.py
"""

import os
import sys
import glob
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE

MARGIN_S = 12
SEQ_LEN = 10          # seconds of history fed to the LSTM
HIDDEN = 48
LAYERS = 1
EPOCHS = 60
BATCH = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
SHOW = ["F08", "F09", "F13"]

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")

MOTORS = ["motor_1_front_right", "motor_2_rear_left",
          "motor_3_front_left",  "motor_4_rear_right"]
VELOCITY = ["uav1_estimation_manager_uav_state__velocity_linear_x",
            "uav1_estimation_manager_uav_state__velocity_linear_y",
            "uav1_estimation_manager_uav_state__velocity_linear_z"]
IMU = ["uav1_hw_api_imu__angular_velocity_x", "uav1_hw_api_imu__angular_velocity_y",
       "uav1_hw_api_imu__angular_velocity_z", "uav1_hw_api_imu__linear_acceleration_x",
       "uav1_hw_api_imu__linear_acceleration_y", "uav1_hw_api_imu__linear_acceleration_z"]
ORIENT = ["roll_rad", "pitch_rad", "yaw_rad"]
ALT = ["uav1_mavros_altitude__local"]
FEATURES = MOTORS + ["payload_mass"] + VELOCITY + ["speed_3d"] + IMU + ORIENT + ALT


def find_flights_dir():
    d = HERE
    for _ in range(6):
        c = os.path.join(d, "flights")
        if glob.glob(os.path.join(c, "F*", "flight_resampled.csv")):
            return c
        for cand in glob.glob(os.path.join(d, "*", "*", "*", "flights")):
            if glob.glob(os.path.join(cand, "F*", "flight_resampled.csv")):
                return cand
        d = os.path.dirname(d)
    sys.exit("Could not find flights/ containing flight_resampled.csv")


def load(fdir):
    """1 Hz mission window per flight, with a cruise/transition phase label."""
    need = FEATURES + ["power"]
    out = []
    for f in sorted(glob.glob(os.path.join(fdir, "F*", "flight_resampled.csv"))):
        d = pd.read_csv(f).sort_values("timestamp").copy()
        thr = 0.5 * (np.percentile(d["power"], 5) + np.percentile(d["power"], 95))
        d["t"] = d["timestamp"] - d["timestamp"].iloc[0]
        d["sec"] = np.floor(d["t"]).astype(int)
        a = d.groupby("sec", as_index=False)[need + ["t"]].mean()
        a["flight_id"] = d["flight_id"].iloc[0]
        hi = (a["power"] > thr).values
        i0, i1 = hi.argmax(), len(hi) - 1 - hi[::-1].argmax()
        idx = np.arange(len(a))
        a["phase"] = np.where(hi, "cruise", "other")
        keep = (idx >= i0 - MARGIN_S) & (idx <= i1 + MARGIN_S)
        out.append(a[keep].reset_index(drop=True))
    return out                                     # list of per-flight DataFrames


def make_sequences(df, mu, sd, y_mu, y_sd):
    """Sliding windows of SEQ_LEN; predict power at the last step."""
    X = ((df[FEATURES].values - mu) / sd).astype(np.float32)
    y = ((df["power"].values - y_mu) / y_sd).astype(np.float32)
    phase = df["phase"].values
    seqs, targ, ph, ypow = [], [], [], []
    for i in range(SEQ_LEN - 1, len(df)):
        seqs.append(X[i - SEQ_LEN + 1:i + 1])
        targ.append(y[i])
        ph.append(phase[i])
        ypow.append(df["power"].values[i])
    if not seqs:
        return None
    return (np.stack(seqs), np.array(targ, np.float32),
            np.array(ph), np.array(ypow, np.float32))


class LSTMReg(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, HIDDEN, num_layers=LAYERS, batch_first=True)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def train_one(Xtr, ytr):
    model = LSTMReg(Xtr.shape[2]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.L1Loss()
    Xt = torch.tensor(Xtr, device=DEVICE)
    yt = torch.tensor(ytr, device=DEVICE)
    n = len(Xt)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    return model


def main():
    fdir = find_flights_dir()
    flights = load(fdir)
    ids = [d["flight_id"].iloc[0] for d in flights]
    print(f"[LSTM]  1 Hz, +/-{MARGIN_S}s, seq_len={SEQ_LEN}, {len(FEATURES)} features")
    print(f"  flights: {ids}\n")

    all_pred, all_true, all_phase, all_fid = [], [], [], []
    per_flight = {}
    time_data = {}

    for h, dfh in zip(ids, flights):
        tr_df = pd.concat([d for d, i in zip(flights, ids) if i != h], ignore_index=True)
        mu, sd = tr_df[FEATURES].mean().values, tr_df[FEATURES].std().replace(0, 1).values
        y_mu, y_sd = tr_df["power"].mean(), tr_df["power"].std()

        Xtr, ytr = [], []
        for d, i in zip(flights, ids):
            if i == h:
                continue
            s = make_sequences(d, mu, sd, y_mu, y_sd)
            if s is not None:
                Xtr.append(s[0]); ytr.append(s[1])
        Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)

        model = train_one(Xtr, ytr)

        s = make_sequences(dfh, mu, sd, y_mu, y_sd)
        model.eval()
        with torch.no_grad():
            pred_std = model(torch.tensor(s[0], device=DEVICE)).cpu().numpy()
        pred = pred_std * y_sd + y_mu           # back to watts
        true = s[3]; phase = s[2]

        r2 = r2_score(true, pred); mae = mean_absolute_error(true, pred)
        per_flight[h] = {"r2": round(float(r2), 3), "mae_w": round(float(mae), 2)}
        print(f"  {h}: R2 {r2:+.3f}  MAE {mae:5.1f} W", flush=True)

        all_pred.append(pred); all_true.append(true)
        all_phase.append(phase); all_fid.append(np.array([h] * len(true)))
        # time trace (align to t of the predicted indices)
        t = dfh["t"].values[SEQ_LEN - 1:]
        time_data[h] = (t, true, pred)

    P = np.concatenate(all_pred); T = np.concatenate(all_true)
    PH = np.concatenate(all_phase)
    r2 = float(r2_score(T, P)); mae = float(mean_absolute_error(T, P))
    ck = PH == "cruise"
    cruise_mae = float(mean_absolute_error(T[ck], P[ck]))
    other_mae = float(mean_absolute_error(T[~ck], P[~ck])) if (~ck).any() else None
    print(f"\n  POOLED: R2 {r2:.3f}   MAE {mae:.2f} W   "
          f"cruise {cruise_mae:.2f} W   transition "
          f"{other_mae:.2f} W" if other_mae else "")

    pd.DataFrame([{"flight": k, **v} for k, v in per_flight.items()]).to_csv(
        os.path.join(OUT, "lstm_per_flight.csv"), index=False)
    with open(os.path.join(OUT, "lstm_summary.json"), "w") as fh:
        json.dump({"model": "LSTM", "rate_hz": 1, "margin_s": MARGIN_S,
                   "seq_len": SEQ_LEN, "hidden": HIDDEN, "layers": LAYERS,
                   "epochs": EPOCHS, "n_features": len(FEATURES),
                   "features": FEATURES,
                   "pooled": {"r2": round(r2, 3), "mae_w": round(mae, 2),
                              "cruise_mae_w": round(cruise_mae, 2),
                              "transition_mae_w": round(other_mae, 2) if other_mae else None},
                   "per_flight": per_flight}, fh, indent=2)

    show = [f for f in SHOW if f in time_data]
    fig, axes = plt.subplots(len(show), 1, figsize=(13, 3.2 * len(show)), squeeze=False)
    for i, f in enumerate(show):
        t, tr, pr = time_data[f]
        ax = axes[i][0]
        ax.plot(t, tr, color="#334155", lw=1.6, label="measured")
        ax.plot(t, pr, color="#7c3aed", lw=1.1, label="LSTM predicted")
        ax.set_title(f"{f} (held out)  R^2={r2_score(tr, pr):+.3f}  "
                     f"MAE={mean_absolute_error(tr, pr):.0f} W",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel("Power (W)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[-1][0].set_xlabel("Time (s)")
    fig.suptitle(f"LSTM (seq_len={SEQ_LEN}, {len(FEATURES)} features), "
                 f"leave-one-flight-out", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "lstm_time.png"), dpi=150)
    plt.close()

    print("\n  Saved -> 15 LSTM/lstm_summary.json, lstm_per_flight.csv, lstm_time.png")


if __name__ == "__main__":
    main()
