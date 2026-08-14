# 14 REGRESION MODEL — Ridge, Ensemble and tree-vs-Ridge comparison on the +/-12 s mission window

Two standalone scripts, each runnable with or without `--no-motors`, all
reading only `flights/F*/flight_resampled.csv` (located automatically by
walking up from their own folder) and each run writing only into its own
subfolder. Nothing else in the thesis codebase is touched.

| script | writes to | model |
|---|---|---|
| `ridge_mission12_study.py` | `RIDGE_MISSION12/` | Ridge (linear) |
| `ridge_mission12_study.py --no-motors` | `RIDGE_MISSION12_NO_MOTORS/` | Ridge, motors excluded |
| `ensemble_mission12_study.py` | `Ensemble/` | XGBoost + Ridge, blended |
| `ensemble_mission12_study.py --no-motors` | `Ensemble_NO_MOTORS/` | XGBoost + Ridge, motors excluded |
| `model_comparison.py` | `MODEL_COMPARISON/` | tree vs Ridge vs ensemble, head to head + timing |

The **head-to-head summary is in `MODEL_COMPARISON/`** (see the section at the
bottom) — it puts the best model of each family, plus their computation time,
on one chart.

Both scripts use the same **+/-12 s mission window** as
`NO_MASS_STUDY/mission12_study.py` (the smallest fixed margin that captures
motor arming on all 14 flights; not a tuned constant), the same
leave-one-flight-out protocol, and the same feature groups (`motors`,
`imbalance`, `orientation`, `imu`, `speed`, `velocity`, `altitude`, `mass`).

Every leave-one-flight-out run already predicts all 14 flights (each one is
held out in its own fold) — the `SHOW` list only controls which 3 of them get
drawn in `plot_time_*_compare.png` for readability. Change it with
`--show F02,F06,F12` (any comma-separated flight IDs); it defaults to
`F08,F09,F13`. Add `--replot` to skip the search/greedy-selection entirely and
just redraw the two comparison plots from the winning combinations already
saved in `comparison_*_summary.json` — seconds instead of the full run, useful
purely to look at different held-out flights. `--replot` needs a prior full
run to exist. The plots currently on disk show **F05, F06, F01**.

---

# RIDGE MISSION-12 STUDY — with mass vs without mass

## The question

Every earlier feature/lag study in `NO_MASS_STUDY/` used **XGBoost**. This study
asks the same "with mass vs without mass, best feature combination" question with
a **Ridge regression** (linear, L2-regularised) instead, on the same
**+/-12 s mission window** used throughout the project (`MISSION12/` /
`mission12_study.py` — the smallest fixed margin that captures motor arming on
all 14 flights; not a tuned constant).

Ridge gives up XGBoost's ability to model non-linear motor-to-power response and
interactions, but in exchange the fitted model is one linear equation with
interpretable, sign-and-magnitude coefficients per standardized feature — useful
as a sanity check against the tree-based results and for the "physically what is
driving power" narrative in the thesis.

## Setup

- **Window:** +/-12 s around cruise (per-flight power threshold = midpoint of
  P5/P95), same definition as `NO_MASS_STUDY/mission12_study.py`.
- **Rate:** 1 Hz (one row per real battery reading; the recommended
  configuration throughout the project — 20 Hz forward-fills the same battery
  reading across ~20 rows and is not used here).
- **Model:** `RidgeCV` per fold. Regularisation strength (`alpha`) is chosen by
  sklearn's efficient closed-form leave-one-out trick over the fold's own
  training rows (`cv=None`) — a standard simplification for picking a
  regularisation strength; it never touches the held-out flight, so it does not
  affect the reported score.
- **Evaluation:** leave-one-flight-out (14 folds). Features are z-scored using
  train-fold mean/std only, applied to both train and the held-out flight — same
  data-leakage discipline as everywhere else in the project (see
  `04 CODES/02 ML MODELS/CLAUDE.md`).
- **Feature search:** EXHAUSTIVE over every non-empty subset of feature groups,
  not greedy forward selection. Ridge is cheap enough to afford it (255 subsets
  with mass, 127 without, ~5,350 fold-fits total, ~45 s on a laptop) — unlike the
  XGBoost studies, which use greedy search because exhaustive search with trees
  is too slow.
- **Groups:** `motors` (4 commands), `imbalance`, `orientation`, `imu`, `speed`,
  `velocity`, `altitude`, and `mass` (with-mass run only) — identical group
  definitions to `NO_MASS_STUDY/mission12_study.py`. `trajectory` is excluded for
  the same reason documented there (trajectory_3 occurs on exactly one flight and
  would act as a single-flight indicator under leave-one-flight-out).

## Results

| | best combination | features | R² | MAE (W) | cruise MAE (W) |
|---|---|---|---|---|---|
| **mass included** | `mass + motors + imu` | 11 | 0.895 | **47.11** | 37.14 |
| **mass withheld** | `motors + altitude` | 5 | 0.892 | **49.70** | 39.79 |

Mass costs **+2.59 W MAE** if withheld — a real but small gain, much smaller than
the ~10-40 W gaps reported in the XGBoost feature/lag studies. With a linear
model, mass mostly acts as a near-constant per-flight offset; XGBoost can exploit
it more (e.g. interacting with motor commands), which is consistent with mass
mattering less here than in `FEATURE_STUDY/` or `NO_MASS_STUDY/mission12_study.py`.

Both Ridge winners land close to **0.89 R² / ~48 W MAE**, well behind the XGBoost
mission-12 numbers (`motors + mass + motors_lag1` = 34.8 W in `LAG_SEARCH/`,
`motors + mass + speed + imbalance` = 35.5 W in `MISSION12/`). The time plot
(`plot_time_ridge_compare.png`) shows why: the Ridge line is visibly smoother than
the actual trace and than XGBoost's predictions in the equivalent plots — a linear
model cannot bend to follow the wobble in cruise power the way a tree ensemble
can. This is the expected, honest trade-off for using Ridge: less accurate, more
interpretable.

**Standardized coefficients** (descriptive fit on all 14 flights, saved in
`coefficients_ridge_<mass>.csv`): the four motor commands dominate, mass adds a
modest positive +22 W/sd, and IMU terms contribute a few watts each. One motor
coefficient (`motor_1_front_right`) comes out negative in the with-mass fit —
this is a collinearity artifact, not a physical effect: the four motor commands
are near-duplicates in cruise (documented in `FEATURE_STUDY/PIPELINE.md`), so
Ridge can trade weight between them while their *sum* stays informative. Read the
motor group's contribution together, not coefficient by coefficient.

## RIDGE_MISSION12_NO_MOTORS/ — the same study with `motors` dropped entirely

Run with `python3 ridge_mission12_study.py --no-motors`. This removes the
`motors` group from the search space from the start (not just from the winning
combination) — a check of what the rest of the sensor suite (imbalance,
orientation, imu, speed, velocity, altitude, mass) can do without motor
commands at all, e.g. for a deployment where they are not available at
inference time.

| | best combination | features | R² | MAE (W) |
|---|---|---|---|---|
| **using mass** (best subset that includes it) | `mass + imbalance + speed + velocity` | 7 | 0.427 | 149.88 |
| **without mass** | `imbalance + speed + velocity` | 6 | 0.468 | **141.63** |

Both collapse to ~0.43-0.47 R² / ~142-150 W MAE — roughly 3x worse than with
motors (47-50 W). Confirms the group-importance finding from the main study:
`motors` alone accounts for almost all of the predictive power; nothing else in
the sensor suite comes close to replacing it.

**Mass makes it worse here, not better.** The *unconstrained* search over all 7
remaining groups (mass included as an option) never selects mass — its global
optimum is `imbalance + speed + velocity`, exactly what the without-mass search
also finds. Forcing mass into that same combination costs +8.25 W MAE (149.88 vs
141.63). Read together with the main study, this says mass's contribution to
predicted power runs almost entirely *through* the motor commands (a heavier
aircraft needs more thrust, i.e. higher motor commands, to hover) — once motor
commands are unavailable, mass on its own does not carry a usable linear signal
across only 14 flights, and Ridge's regularisation prefers to leave it out.
`comparison_ridge_summary.json` in this subfolder records both the true
unconstrained winner (`global_best_offered_mass`) and the best mass-using
combination (`best_using_mass`) so this isn't hidden by only reporting one.

The time plot (`plot_time_ridge_compare.png`) makes the accuracy gap obvious:
without motor commands the model has almost nothing to track — predictions
oscillate around the actual trace rather than following it, because imbalance,
speed and velocity barely vary in cruise (documented in `FEATURE_STUDY/PIPELINE.md`:
these flights never exceed 0.28 m/s).

## Reading the plots

- `plot_search_ridge_<mass>.png` — MAE vs number of features, every one of the
  127/255 subsets scored. Shows the same shape both settings share: MAE falls
  off a cliff once `motors` enters (3→4 features), then is flat — extra groups
  neither help nor hurt much once motors are in.
- `plot_top4_ridge_<mass>.png` — best 4 combinations, MAE and R² bars.
- `plot_group_importance_ridge_<mass>.png` — MAE with vs without each group,
  **conditioned on `motors` already being present** (unconditioned, the
  motors-absent subsets are so much worse — ~150-250 W vs ~50 W — that they
  swamp the scale and hide every other comparison).
- `plot_time_ridge_compare.png` — best with-mass model vs best without-mass
  model, side by side, over time, for F08/F09/F13 held out. This is the "plot
  them in time" deliverable.
- `plot_r2_ridge_compare.png` — per-flight R² and MAE, both winners overlaid.

## Files

| file | contents |
|---|---|
| `search_ridge_<mass>.csv` | every non-empty subset, ranked by MAE |
| `summary_ridge_<mass>.json` | winner, top-4, metrics |
| `coefficients_ridge_<mass>.csv` | standardized coefficients of the winning combo, fit on all 14 flights |
| `comparison_ridge_summary.json` | headline with-mass vs without-mass numbers |
| `plot_search_ridge_<mass>.png` | MAE vs number of features, all subsets |
| `plot_top4_ridge_<mass>.png` | best 4 combinations |
| `plot_group_importance_ridge_<mass>.png` | does each group help, given the strongest single group (normally motors)? |
| `plot_time_ridge_compare.png` | power over time, best-using-mass vs without-mass |
| `plot_r2_ridge_compare.png` | per-flight R²/MAE, both winners |

`<mass>` is `withmass` or `nomass`. Same file set in both `RIDGE_MISSION12/` and
`RIDGE_MISSION12_NO_MOTORS/`.

## Running it

```bash
python3 ridge_mission12_study.py                       # 1 Hz, all sensor groups (recommended)
python3 ridge_mission12_study.py --rate 20              # 20 Hz raw grid, sanity check
python3 ridge_mission12_study.py --no-motors             # drop motors -> RIDGE_MISSION12_NO_MOTORS/
python3 ridge_mission12_study.py --show F02,F06,F12      # different held-out flights in the plots
python3 ridge_mission12_study.py --replot --show F02,F06,F12   # same, but skip the search (seconds)
```

About 30-55 s per run (fewer groups to search without motors is faster).
Deterministic (fixed `random_state` is not needed — Ridge has no random
component beyond `RidgeCV`'s deterministic alpha search).

---

# ENSEMBLE MISSION-12 STUDY — XGBoost + Ridge blended

Standalone. `ensemble_mission12_study.py` writes into `Ensemble/`. Third
sibling script in this folder: same window, same feature groups, same
leave-one-flight-out protocol as the two studies above, but the prediction at
every fold is a blend of an XGBoost model and a Ridge model instead of either
one alone.

## The question

XGBoost (`NO_MASS_STUDY/mission12_study.py`) wins on accuracy; Ridge
(`ridge_mission12_study.py`, above) loses ~10-15 W MAE but is interpretable.
Averaging their predictions is the simplest possible way to ask whether they
are wrong in different enough places that combining them beats either one —
and to have a single model that carries a bit of both: mostly XGBoost's
accuracy, with a Ridge component still available for the standardized
coefficients.

## Setup

- **Window / rate / groups:** identical to the Ridge study — +/-12 s, 1 Hz,
  same 7 (+mass) feature groups.
- **Base models per fold:** an `XGBRegressor` with the same hyperparameters as
  `NO_MASS_STUDY/mission12_study.py` (`reg:absoluteerror`, 150 trees, depth 5,
  learning rate 0.1, subsample 0.9) fit on raw features, and a `RidgeCV` fit on
  train-only z-scored features — both leave-one-flight-out, both never seeing
  the held-out flight.
- **Blend:** `prediction = w * xgb_pred + (1-w) * ridge_pred`. For each feature
  combination, `w` is chosen by a 21-point grid search (0.00 to 1.00) that
  minimises MAE over the leave-one-flight-out out-of-fold predictions. This is
  standard "blend the out-of-fold predictions" practice — it decides how to
  combine two already-honest prediction streams after the fact, it does not
  let a held-out flight leak into how its own predictions were produced.
- **Feature search:** GREEDY forward selection, same algorithm as
  `NO_MASS_STUDY/mission12_study.py` — **not** exhaustive like the Ridge study.
  XGBoost is the bottleneck: one leave-one-flight-out evaluation of a single
  combination takes ~4 s (benchmarked on this data), so the 255/127 exhaustive
  subsets Ridge can afford would take 15-25 minutes here. A full run (both mass
  settings) takes ~6 minutes.

## Results

| | best combination | features | R² | MAE (W) | blend weight (XGBoost) |
|---|---|---|---|---|---|
| **using mass** | `motors + mass + speed + velocity` | 9 | 0.926 | **36.48** | 0.90 |
| **without mass** | `motors + imbalance + velocity + speed + orientation + imu` | 19 | 0.919 | **39.17** | 0.95 |

Mass costs **+2.69 W MAE** if withheld — very close to the plain-XGBoost
mission-12 numbers (34.8-35.5 W, `NO_MASS_STUDY/MISSION12/` and `LAG_SEARCH/`),
confirming the ensemble is doing essentially the same job, just slightly
refined by a small Ridge contribution.

**Does blending actually help?** `plot_model_breakdown_<mass>.png` compares the
winning combination's three variants directly:

| | MAE (W) | R² |
|---|---|---|
| XGBoost alone | 36.73 | 0.925 |
| Ridge alone | 48.30 | 0.895 |
| **Ensemble (w=0.90)** | **36.48** | **0.926** |

The ensemble beats pure XGBoost, but only by **0.25 W** — inside the noise of
this dataset (cruise floor is ~34 W, set by flight-to-flight repeatability, not
by the model; see `FEATURE_STUDY/PIPELINE.md`). The blend weight landing at
0.90 rather than 1.00 says Ridge is not useless — it pulls the prediction
slightly on some folds — but the honest read is that **XGBoost alone already
captures nearly everything the ensemble captures**; blending is a small,
legitimate refinement here, not a breakthrough. The one place the search
consistently gives Ridge more weight (`w_xgb` as low as 0.00-0.45, see the
`altitude`-containing rows in `search_ensemble_<mass>.csv`) is exactly where
XGBoost is worst — altitude alone is a poor predictor and XGBoost overfits it
more than Ridge does with only 14 flights, so the grid search leans on Ridge
to stabilise those specific combinations.

As in the Ridge-without-motors study, the unconstrained greedy search here
does pick mass on its own (step 2, MAE 40.60 → 37.05 W) — unlike the
no-motors case, mass is genuinely useful once motor commands are present, so
no forced-mass fallback was needed for the comparison plots.

## Ensemble_NO_MOTORS/ — the same study with `motors` dropped entirely

Run with `python3 ensemble_mission12_study.py --no-motors`. Same idea as
`RIDGE_MISSION12_NO_MOTORS/`: removes the `motors` group from the search space
entirely, to see what the rest of the sensor suite can do on its own.

| | best combination | features | R² | MAE (W) | blend weight (XGBoost) |
|---|---|---|---|---|---|
| **using mass** | `imu + mass + speed + velocity + imbalance` | 13 | 0.880 | **42.75** | 1.00 |
| **without mass** | `imu + velocity` | 9 | 0.859 | **62.09** | 1.00 |

Both are far worse than with motors (43-62 W vs 36-39 W) — the same
"motors carries almost everything" conclusion as every other study in this
project. But the comparison to `RIDGE_MISSION12_NO_MOTORS/` is the interesting
part:

**Here, mass helps a lot; in the pure-Ridge no-motors study, it hurt.** Mass
buys **+19.34 W MAE** in this ensemble (42.75 vs 62.09 W) — the largest mass
gain measured anywhere in this folder — while the Ridge-only no-motors study
found the opposite (forcing mass in cost +8.25 W, see
`RIDGE_MISSION12_NO_MOTORS/`). The blend weight explains why: it is **1.00 —
pure XGBoost — on every combination tried here** (see
`search_ensemble_<mass>.csv`; `plot_model_breakdown_<mass>.png` shows Ridge
alone scoring R² 0.424 against the ensemble's 0.880). Without motor commands,
mass only helps if the model can combine it *non-linearly* with the remaining
weak signals (IMU, velocity, imbalance, speed) — something a linear Ridge fit
on 14 flights cannot do, but XGBoost can, even if what it is fitting here is
partly a per-flight identifier in disguise (mass is constant within a flight)
rather than a physical effect. Read this result as *"XGBoost can extract more
from mass than Ridge can once motors are gone,"* not as evidence that a linear
mass effect exists independent of motors — the Ridge study already showed it
does not.

## Reading the plots

- `plot_search_ensemble_<mass>.png` — greedy path, MAE and R² as groups are
  added, same style as `NO_MASS_STUDY/mission12_study.py`'s search plot.
- `plot_top4_ensemble_<mass>.png` — best 4 combinations *actually evaluated*
  during the greedy search (not exhaustive — see Setup).
- `plot_model_breakdown_<mass>.png` — XGBoost alone vs Ridge alone vs Ensemble,
  for the winning combination. The "does blending help" figure.
- `plot_time_ensemble_compare.png` — best using-mass model vs best
  without-mass model, side by side, over time, for F08/F09/F13 held out.
- `plot_r2_ensemble_compare.png` — per-flight R² and MAE, both winners overlaid.

## Files

| file | contents |
|---|---|
| `search_ensemble_<mass>.csv` | greedy path, every step scored |
| `summary_ensemble_<mass>.json` | winner, top-4, blend weight, metrics |
| `coefficients_ridge_component_<mass>.csv` | standardized coefficients of the Ridge half of the ensemble, fit on all 14 flights — same convention as `RIDGE_MISSION12/coefficients_ridge_<mass>.csv` |
| `comparison_ensemble_summary.json` | headline using-mass vs without-mass numbers |
| `plot_search_ensemble_<mass>.png` | MAE/R² as groups are added |
| `plot_top4_ensemble_<mass>.png` | best 4 combinations tried |
| `plot_model_breakdown_<mass>.png` | XGBoost alone vs Ridge alone vs Ensemble |
| `plot_time_ensemble_compare.png` | power over time, using-mass vs without-mass |
| `plot_r2_ensemble_compare.png` | per-flight R²/MAE, both winners |

`<mass>` is `withmass` or `nomass`. Same file set in both `Ensemble/` and
`Ensemble_NO_MOTORS/`.

## Running it

```bash
python3 ensemble_mission12_study.py                      # 1 Hz, all sensor groups (recommended)
python3 ensemble_mission12_study.py --rate 20             # 20 Hz raw grid, sanity check
python3 ensemble_mission12_study.py --no-motors           # drop motors -> Ensemble_NO_MOTORS/
python3 ensemble_mission12_study.py --show F02,F06,F12    # different held-out flights in the plots
python3 ensemble_mission12_study.py --replot --show F02,F06,F12   # same, but skip the greedy search
                                                            # (seconds instead of minutes)
```

About 3-6 minutes per run (XGBoost-bound; fewer groups without motors is
faster). Deterministic (`random_state=42` on XGBoost; Ridge and the
blend-weight grid search have no random component).

---

# MODEL COMPARISON — tree vs Ridge vs ensemble, head to head

Standalone. `model_comparison.py` writes into `MODEL_COMPARISON/`. Pulls the
three model families onto one chart: the **best model of each** (at its own
best feature combination, from its own study) plus **computation time**, all on
the same +/-12 s / 1 Hz / leave-one-flight-out setup.

## Results (mass included — each model's overall best)

| model | best combination | R² | MAE (W) | compute time |
|---|---|---|---|---|
| **XGBoost (tree)** | `motors + mass + speed + imbalance` | 0.923 | **35.53** | ~3.2 s |
| **Ridge (linear)** | `mass + motors + imu` | 0.895 | 47.11 | **~0.08 s** |
| **Ensemble (blend, w=0.90)** | `motors + mass + speed + velocity` | **0.926** | 36.48 | ~3.2 s |

Mass withheld (same three models, second panel / `..._nomass.png`): XGBoost
39.96 W, Ridge 49.70 W, Ensemble 39.17 W.

**The trade-off in one line.** The tree and the ensemble are ~11-13 W more
accurate than Ridge (35-36 vs 47 W MAE) but ~40x slower to fit
(~3.2 s vs ~0.08 s per 14-fold pass). The ensemble is the most accurate by a
hair (R² 0.926 vs 0.923) but costs the same as XGBoost and is dominated by it
(blend weight 0.90), so **XGBoost alone is the sensible default**; Ridge is the
one to reach for only if fit/refit speed or an interpretable equation matters
more than ~11 W of accuracy.

## What "compute time" means here

Wall-clock to **fit and predict one full leave-one-flight-out pass (14 folds)**
at the model's best combination — the cost of building the deployable model,
timed identically for all three in one process (median of 3 runs; there is
run-to-run variance of a few tenths of a second from system load, which does
not change the ~40x ordering). It is the *model* cost, not the
*feature-search* cost — those differ for a different reason: Ridge is cheap
enough per fit that its study runs an **exhaustive** 127/255-subset search in
~45 s, whereas XGBoost is ~40x slower per fit so its studies (and the ensemble)
use **greedy** forward selection instead. So Ridge is both the fastest model
*and* the only one that can afford to look at every feature combination.

## Files

| file | contents |
|---|---|
| `plot_model_comparison_withmass.png` | 3 panels — MAE, R², compute time — best model of each (mass included) |
| `plot_model_comparison_nomass.png` | same, mass withheld |
| `model_comparison.csv` | every model × mass setting: groups, features, blend weight, time, R², MAE |
| `model_comparison.json` | same, machine-readable |

## Running it

```bash
python3 model_comparison.py               # median of 3 timing runs (~50 s total)
python3 model_comparison.py --repeats 5    # steadier timing, a bit slower
```

The best feature combinations are hard-coded from each study's saved winner
(`BEST` dict at the top of the script), so the comparison is transparent and
does not silently re-run each search. If a study is later re-run and its winner
changes, update that dict. Accuracy numbers reproduce each source study exactly
(XGBoost is z-scored to match `NO_MASS_STUDY/mission12_study.py`; the ensemble's
XGBoost half is on raw features to match `ensemble_mission12_study.py`).
