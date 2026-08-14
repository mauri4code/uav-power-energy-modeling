"""
FULL vs MARGIN, side by side over time — identical model, identical loss.

STANDALONE. Reads only flights/F*/flight_resampled.csv, writes only here.

WHY
---
FULL scores R2 0.933 and MARGIN scores R2 0.818, yet their cruise accuracy is
identical (35.0 vs 34.8 W). The difference is not model quality — it is that FULL
contains ~284 rows of ground idle that are trivially predictable and that widen
the spread R2 is measured against.

This figure shows both on the same time axis so the cause is visible rather than
argued: the extra rows FULL gets to score on are the flat sections at either end,
where the prediction sits exactly on the measurement.

Model: motors + payload_mass, absolute-error loss, leave-one-flight-out, 1 Hz.

Output : plot_full_vs_margin_time.png
Run    : python plot_full_vs_margin_time.py            # F08 F09 F13
         python plot_full_vs_margin_time.py F01 F02
"""

import os
import sys
import importlib.util

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "lossexp", os.path.join(HERE, "exp_loss_function.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

OBJ = "reg:absoluteerror"


def main():
    full = E.load_1hz(E.find_flights_dir())
    margin = full[full.phase != "ground"].reset_index(drop=True)

    print(f"\nFULL   {len(full)} rows   MARGIN {len(margin)} rows")
    print("fitting both (absolute-error loss) ...", flush=True)
    p_full = E.evaluate(full, OBJ)
    p_marg = E.evaluate(margin, OBJ)

    def stats(d, p):
        ck = (d.phase == "cruise").values
        return (r2_score(d.power, p), mean_absolute_error(d.power, p),
                mean_absolute_error(d.power[ck], p[ck]))

    rf, mf, cf = stats(full, p_full)
    rm, mm, cm = stats(margin, p_marg)
    print(f"  FULL    R² {rf:+.3f}  MAE {mf:5.1f} W  cruise {cf:5.1f} W")
    print(f"  MARGIN  R² {rm:+.3f}  MAE {mm:5.1f} W  cruise {cm:5.1f} W\n")

    flights = sorted(full.flight_id.unique())
    show = [f for f in (sys.argv[1:] or ["F08", "F09", "F13"]) if f in flights]

    fig, axes = plt.subplots(len(show), 2, figsize=(17, 4.0 * len(show)),
                             squeeze=False)
    for r, fid in enumerate(show):
        kf = (full.flight_id == fid).values
        km = (margin.flight_id == fid).values
        tf = full[kf].sort_values("t"); tm = margin[km].sort_values("t")

        for c, (d, p, k, name, tot) in enumerate([
                (tf, p_full[kf], kf, f"FULL — R² {rf:.3f} overall", rf),
                (tm, p_marg[km], km, f"MARGIN ±5 s — R² {rm:.3f} overall", rm)]):
            ax = axes[r][c]
            t, y = d["t"].values, d["power"].values
            ck = (d.phase == "cruise").values
            ax.fill_between(t, 0, 1000, where=~ck, color="#94a3b8", alpha=0.18,
                            step="mid", zorder=0,
                            label="ground / transition (the 'free' rows)")
            ax.plot(t, y, color="#334155", lw=1.4, label="ACTUAL", zorder=3)
            ax.plot(t, p, color="#dc2626", lw=1.1, alpha=0.9, zorder=2,
                    label=f"predicted (cruise MAE "
                          f"{mean_absolute_error(y[ck], p[ck]):.0f} W)")
            ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, max(y) * 1.1)
            ax.grid(alpha=0.3)
            ax.set_title(f"{fid} — {name}", fontsize=10, fontweight="bold")
            if c == 0:
                ax.set_ylabel("Power (W)", fontsize=10)
            if r == 0:
                ax.legend(fontsize=7.5, loc="center left", framealpha=0.94)
    for c in range(2):
        axes[-1][c].set_xlabel("Time (s)", fontsize=10)

    fig.suptitle("Same model, same loss, same flights — only the scored rows differ\n"
                 f"cruise accuracy is identical ({cf:.1f} vs {cm:.1f} W); "
                 f"R² differs ({rf:.3f} vs {rm:.3f}) because FULL includes the shaded idle",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(HERE, "plot_full_vs_margin_time.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {out}\n")


if __name__ == "__main__":
    main()
