"""
Step 19 – Summary figure: 20 Hz vs 1 Hz, no-mass vs with-mass, on SIM_mission.

Reads the prediction CSVs written by steps 17 and 18 and stacks them into one
comparison figure (top = 20 Hz, bottom = 1 Hz), plus a compact metrics bar panel.
Purely a presentation step — no models are trained here.

Input : SIM_FLIGHTS/SIM_mission/nomass_vs_withmass.csv        (20 Hz, step 17)
        SIM_FLIGHTS/SIM_mission/1hz/nomass_vs_withmass_1hz.csv (1 Hz,  step 18)
Output: SIM_FLIGHTS/SIM_mission/rate_summary.png

Run: python 19_mission_rate_summary.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MISSION_DIR = os.path.join(SCRIPT_DIR, "SIM_FLIGHTS", "SIM_mission")

C_NOMASS, C_WITH = "#2563eb", "#dc2626"


def metrics(y, p, cruise):
    return {"mae_all": mean_absolute_error(y, p), "r2_all": r2_score(y, p),
            "mae_cru": mean_absolute_error(y[cruise], p[cruise]),
            "r2_cru":  r2_score(y[cruise], p[cruise])}


def panel(ax, t, df, seg, title, show_phase):
    y = df["actual_power"].values
    thr = 0.5 * (np.percentile(y, 5) + np.percentile(y, 95))
    cru = y > thr
    mn = metrics(y, df["pred_nomass"].values, cru)
    mw = metrics(y, df["pred_withmass"].values, cru)

    ax.plot(t, y, color="#0f172a", lw=1.6, label="ACTUAL", zorder=5)
    ax.plot(t, df["pred_nomass"].values,  color=C_NOMASS, lw=1.2, alpha=0.9,
            label=f"no-mass    cruise MAE {mn['mae_cru']:.0f} W (R² {mn['r2_cru']:.2f})")
    ax.plot(t, df["pred_withmass"].values, color=C_WITH, lw=1.2, alpha=0.9,
            label=f"with-mass  cruise MAE {mw['mae_cru']:.0f} W (R² {mw['r2_cru']:.2f})")
    ymax = max(y.max(), df["pred_nomass"].max(), df["pred_withmass"].max())
    ax.set_ylim(-20, ymax * 1.18)
    for r in seg.itertuples():
        ax.axvline(r.t_start_s, color="k", ls=":", lw=0.6, alpha=0.35)
        if show_phase:
            ph = getattr(r, "phase", r.source_flight)
            ax.text((r.t_start_s + r.t_end_s) / 2, ymax * 1.03,
                    f"{ph}\n({r.source_flight})", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#334155")
    ax.set_ylabel("power [W]")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.legend(loc="lower center", fontsize=8.5, framealpha=0.95, ncol=1)
    ax.grid(alpha=0.3)
    return mn, mw


def main():
    seg = pd.read_csv(os.path.join(MISSION_DIR, "segment_map.csv"))
    d20 = pd.read_csv(os.path.join(MISSION_DIR, "nomass_vs_withmass.csv"))
    d01 = pd.read_csv(os.path.join(MISSION_DIR, "1hz", "nomass_vs_withmass_1hz.csv"))

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1], hspace=0.3, wspace=0.22)
    ax20 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[1, 0], sharex=ax20)
    axm  = fig.add_subplot(gs[:, 1])

    m20 = panel(ax20, d20["timestamp"].values, d20, seg,
                "20 Hz  (raw resampled grid)", show_phase=True)
    m01 = panel(ax01, d01["second"].values, d01, seg,
                "1 Hz  (per-second mean — best config)", show_phase=False)
    ax01.set_xlabel("time [s]")

    # ---- metrics bar panel: cruise MAE, four bars ----
    labels = ["20 Hz\nno-mass", "20 Hz\nwith-mass", "1 Hz\nno-mass", "1 Hz\nwith-mass"]
    vals   = [m20[0]["mae_cru"], m20[1]["mae_cru"], m01[0]["mae_cru"], m01[1]["mae_cru"]]
    cols   = [C_NOMASS, C_WITH, C_NOMASS, C_WITH]
    x = np.arange(4)
    axm.bar(x, vals, color=cols, alpha=0.9)
    for xi, v in zip(x, vals):
        axm.text(xi, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axm.set_xticks(x); axm.set_xticklabels(labels, fontsize=8.5)
    axm.set_ylabel("cruise MAE [W] — lower is better", fontsize=9.5)
    axm.set_title("Cruise error", fontsize=11, fontweight="bold")
    axm.grid(axis="y", alpha=0.3)
    axm.set_ylim(0, max(vals) * 1.25)

    fig.suptitle("SIM_mission — row-level power model: rate & mass comparison\n"
                 "leave-one-flight-out; rebuilt from NO_MASS_STUDY/MISSION12 recipe",
                 fontsize=13, fontweight="bold")
    out = os.path.join(MISSION_DIR, "rate_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {out}")
    print(f"  20 Hz cruise MAE: no-mass {vals[0]:.1f}  with-mass {vals[1]:.1f}")
    print(f"  1  Hz cruise MAE: no-mass {vals[2]:.1f}  with-mass {vals[3]:.1f}")


if __name__ == "__main__":
    main()
