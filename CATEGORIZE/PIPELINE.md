# 12 CATEGORIZE — Methodology / pipeline

How the payload-position classifier was built, end to end: where the data comes
from, how it was preprocessed, how the model was chosen, the cross-validation
scheme, and every parameter used. Companion to `README.md` (results) and
`PLAN.md` (design rationale).

**One-line summary:** 4-class classification of `position_payload`, evaluated by
**14-fold leave-one-flight-out** (GroupKFold with one flight per fold), model and
feature set chosen by an **exhaustive feature-group search** scored on that same
cross-validation. Hyperparameters were **fixed, not tuned** (see §6 for why, and
the honesty caveat in §8).

---

## 1. Task definition

| | |
|---|---|
| **Target** | `position_payload` — a categorical label, constant within a flight |
| **Classes** | `none`, `front`, `rear`, `diagonal` (4). `full` was never flown → not in the label set, cannot be predicted or tested |
| **Type** | Multi-class classification (not regression — so accuracy / macro-F1 / confusion matrix, never R²/MAE) |
| **Unit of prediction** | one label **per flight** (see §4) |

---

## 2. Data provenance

### 2.1 Upstream (not part of this study)

The classifier consumes `flights/F*/flight_resampled.csv`, the final output of
the preprocessing pipeline documented in
`04 CODES/02 ML MODELS/00 EXPORT_TOPICS/CONTEXT.md`. In brief, per flight:

1. `01_export_bag.py` — ROS `.bag` → per-topic CSVs (velocity, IMU, orientation,
   altitude, battery).
2. `01b_extract_ulg_motors.py` — PX4 `.ulg` → 4 motor commands (normalized 0–1, ~2.29 Hz).
3. `02_compute_euler.py` — quaternion → roll / pitch / yaw.
4. `03_outlier_removal.py` — IQR cleaning (`IQR_K = 3.0`) on **linear
   accelerations only**; motors pass through untouched.
5. `04_resampling.py` — everything resampled onto a uniform **20 Hz** grid (ROS
   streams linearly interpolated; battery + motors forward-filled), merged into
   one CSV, plus derived features: `speed_3d`, `speed_horizontal`, `power`,
   `front_rear_imbalance`, `diagonal_imbalance`.

The label (`position_payload`) and `payload_mass` come from each flight's
`flight_config.json` and are written into every row of that flight's CSV.

### 2.2 The 14 flights

| class | flights | n flights |
|---|---|---|
| `none` | F01, F02 | 2 |
| `front` | F03, F04, F07, F08 | 4 |
| `rear` | F11, F12, F13, F14 | 4 |
| `diagonal` | F05, F06, F09, F10 | 4 |

`payload_mass` ∈ {0, 0.24, 0.48} kg. Mass is **confounded** with the label: only
`none` has 0 g, and both 0.24 and 0.48 kg appear across `front`/`rear`/`diagonal`.
So mass can separate `none` from the rest but carries no front-vs-rear-vs-diagonal
information — which is why it is the variant axis (§3.3), not a core feature.

---

## 3. Preprocessing inside this study (`load()` in `categorize_position_study.py`)

All of the following happens per flight, then flights are concatenated.

### 3.1 Mission window (±12 s)

Each raw recording has an arbitrary amount of ground idle. It is trimmed to the
**mission window**, identically to the power studies:

1. Per-flight power threshold `thr = 0.5 · (P5 + P95)` of that flight's `power`
   column (lands ~340–415 W, cleanly between ~25 W idle and ~700 W airborne).
2. Find the first and last row where `power > thr`.
3. Keep from **12 s before the first** crossing to **12 s after the last**.

12 s is the smallest fixed margin that captures motor arming on all 14 flights
(largest arm-to-takeoff gap = 11.9 s) — not a tuned constant.

### 3.2 Downsample 20 Hz → 1 Hz

The 20 Hz grid forward-fills the ~1 Hz battery and ~2.29 Hz motor signals, so
most adjacent rows are exact repeats. Rows are aggregated to **1 Hz by taking the
mean of each column within each integer second** (`groupby(floor(t))`). 1 Hz is
the recommended rate throughout the project (one row per real battery reading).
`--rate 20` keeps the raw grid as a sanity check.

Result: **1,792 rows across the 14 flights**, unevenly distributed (F02 = 426
rows, F03 = 44) — one reason the flight, not the row, is the unit of truth (§4)
and why class weighting is used (§6).

### 3.3 Feature groups

Features are grouped (not selected individually), exactly as in the power
studies, because the four motor commands are near-duplicates in cruise and
grouping keeps the exhaustive search tractable.

| group | columns | n features |
|---|---|---|
| `motors` | motor_1_front_right, motor_2_rear_left, motor_3_front_left, motor_4_rear_right | 4 |
| `orientation` | roll_rad, pitch_rad, yaw_rad | 3 |
| `imu` | angular_velocity x/y/z, linear_acceleration x/y/z | 6 |
| `speed` | speed_3d, speed_horizontal | 2 |
| `velocity` | velocity_linear x/y/z | 3 |
| `altitude` | altitude__local | 1 |
| `mass` | payload_mass | 1 |

**Excluded on purpose:**

- `imbalance` (front_rear, diagonal) — **leakage by design**: these were
  engineered specifically to encode load distribution (`front` → front_rear +,
  etc.), so predicting position from them is circular.
- `trajectory` — a single-flight indicator (`trajectory_3` occurs on exactly one
  flight), same reason as the power studies.
- `power`, `battery_volt`, `battery_curr` — never features here (they were the
  target of the *other* study; not relevant to position).

**Variant axis = mass.** `mass` is offered to the search in the `WITH_MASS/` run
and dropped in `WITHOUT_MASS/`, so its effect on position prediction is measured,
not assumed.

---

## 4. Prediction unit: per-flight majority vote

The label is constant within a flight, and 1 Hz rows are individually noisy
(a heavy flight contributes hundreds of correlated rows). So:

1. The classifier predicts a label for **every row** of the held-out flight.
2. Those row predictions are collapsed to **one label per flight by majority
   vote** (`pandas.Series.mode()`; ties broken alphabetically → deterministic).

The **honest sample size is 14 flights, not 1,792 rows.** Per-flight accuracy is
the headline; row-level accuracy is recorded too but flagged as optimistic
(e.g. Logistic on `motors` scores 0.63 per row yet 1.00 per flight).

---

## 5. Cross-validation: leave-one-flight-out (14-fold)

**Yes — this uses k-fold cross-validation**, specifically **leave-one-flight-out
= GroupKFold with `k = 14` folds, grouped by `flight_id`**, one flight per fold.

- Each fold: fit on **13 flights**, predict the **1 held-out flight**, repeat 14×.
- **No flight ever appears on both sides of a fold.** This is the same discipline
  as `06_train_xgboost.py`'s `GroupKFold(n_splits=n_flights)` and every study in
  `NO_MASS_STUDY/`.
- **There is no separate final test set.** With only 14 flights, holding some out
  would leave too little to fit; leave-one-flight-out already reports a score on
  every flight while held out, so all 14 serve as test exactly once.

### 5.1 Scaling — fit on train fold only

Inside each fold, features are z-scored (`(x − μ) / σ`) using the **training
flights' mean and std only**, applied to both the training rows and the held-out
flight. Zero-variance columns get σ→1. This prevents the held-out flight from
leaking into the scaling. (Random Forest is scale-invariant and does not need it,
but scaling is applied uniformly to all three models to keep one code path.)

### 5.2 What "held out" means concretely — by flight, not by row

Each flight is hundreds of 1 Hz rows. A fold pulls out **all rows of one flight
together** — never some rows of a flight for training and others for testing.
Holding out the whole flight as a group is what makes the test honest: it mimics
"a brand-new flight arrives," not "a few unseen instants of a flight the model
already knows."

```
Fold 5 (hold out F05):
   TRAIN = F01 F02 F03 F04  F06 F07 F08 F09 F10 F11 F12 F13 F14   (13 flights, all rows)
   TEST  =                F05                                      (1 flight, every row unseen)
   scaler mean/std computed on the 13 TRAIN flights only, then applied to F05
   → predict every F05 row → majority vote → one label for F05
```

repeated 14 times, each flight held out exactly once (Fold 1 → F01, …, Fold 14 → F14).

### 5.3 Two levels of holding out — the nested version

The nested cross-validation (`NESTED_CV/`) stacks a second hold-out **inside** the
training set, so the test flight takes no part in choosing the model:

```
OUTER: hold out F05 as the final TEST → 13 training flights remain (F05 sealed away)

    INNER (using ONLY those 13, to CHOOSE the model + features):
        hold out F01 → train on 12, score
        hold out F02 → train on 12, score
           ⋮   (each of the 13 held out once — a 13-fold hold-out within the training set)
        → pick the model + feature subset that scores best across these inner hold-outs

    refit that winner on all 13 training flights → predict F05
```

F05 is never used to train **and** never used to select. Then the whole thing
repeats with F06 as the outer test, F07, and so on. See `NESTED_CV/README.md`.

---

## 6. Model selection — how the model was found

Two axes were searched; **hyperparameters were not** (see §6.3).

### 6.1 Feature-group selection — exhaustive search

For each model, **every non-empty subset of the offered feature groups** is
scored by the leave-one-flight-out protocol above and ranked by per-flight
accuracy (ties broken by macro-F1, then row accuracy).

- `WITHOUT_MASS/`: 6 groups → **2⁶ − 1 = 63** subsets.
- `WITH_MASS/`: 7 groups → **2⁷ − 1 = 127** subsets.

Exhaustive (not greedy) is affordable because the models are cheap on 14 flights.
The full ranked table is saved (`search_<model>_<tag>.csv`) so a "best
combination" claim can be checked against every alternative, not just a search
path. The search is parallelised over subsets (`joblib`, threading backend) so
Random Forest — the only slow model — uses all cores.

### 6.2 Model-family comparison — three classifiers + baseline

| role | model |
|---|---|
| interpretable, linear | multinomial **Logistic Regression** |
| interpretable, linear | **Linear Discriminant Analysis (LDA)** |
| non-linear sanity check | **Random Forest** |
| reference | **majority-class baseline** |

This mirrors the Ridge-vs-XGBoost pairing on the power side: a linear,
coefficient-readable model plus a tree model to check whether non-linearity buys
anything. The winner reported per model is the top row of its search table.

### 6.3 Hyperparameters were fixed, not tuned — and why

Unlike the XGBoost power model (which used `RandomizedSearchCV`, 30 candidates),
**no hyperparameter search was run here.** The reasons:

- With **14 flights**, a nested tuning loop would select hyperparameters on a
  handful of flights and overfit the selection badly.
- The classes turned out **linearly separable from the motor commands** (Logistic
  on `motors` = 14/14), so there was nothing for extra model capacity to gain.
- Sensible library defaults + class balancing are the defensible, low-variance
  choice at this data scale.

Consequently there is **no nested cross-validation**: the single 14-fold
leave-one-flight-out loop is used both to score feature subsets and to report the
final numbers. The optimism this introduces is discussed in §8.

---

## 7. Exact parameters used

### 7.1 Windowing / sampling

| parameter | value |
|---|---|
| mission-window margin (`MARGIN_S`) | 12 s |
| power threshold | `0.5 · (P5 + P95)` of per-flight `power` |
| sample rate (`RATE`) | 1 Hz (per-second column mean); `--rate 20` optional |
| rows after windowing | 1,792 across 14 flights |

### 7.2 Cross-validation

| parameter | value |
|---|---|
| scheme | leave-one-flight-out = GroupKFold, group = `flight_id` |
| folds `k` | 14 (one flight per fold) |
| scaling | StandardScaler-equivalent, fit on train flights only |
| aggregation | per-flight majority vote over row predictions |
| selection metric | per-flight accuracy (tie-break: macro-F1, then row accuracy) |

### 7.3 Models (scikit-learn 1.8, all deterministic)

**Logistic Regression** — `LogisticRegression(max_iter=2000, class_weight="balanced")`

| param | value | note |
|---|---|---|
| penalty | `l2` | default |
| C (inverse reg. strength) | `1.0` | default — untuned |
| solver | `lbfgs` | default; multinomial softmax automatically for >2 classes |
| max_iter | `2000` | raised from default 100 for convergence on standardized data |
| class_weight | `balanced` | counters the 2-vs-4 class sizes and uneven per-flight row counts |

*(The descriptive all-flights coefficient fit in `coefficients_logreg_*.csv` uses
`max_iter=5000`; it feeds no reported metric — only the interpretable coefficients.)*

**LDA** — `LinearDiscriminantAnalysis()` (all defaults)

| param | value |
|---|---|
| solver | `svd` |
| shrinkage | `None` |
| priors | `None` → estimated from class frequencies in each training fold |

**Random Forest** — `RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=1)`

| param | value | note |
|---|---|---|
| n_estimators | `200` | ample for 14 flights; lower it to speed up re-runs |
| criterion | `gini` | default |
| max_depth | `None` | default (grow until pure) |
| max_features | `sqrt` | default |
| min_samples_split / leaf | `2` / `1` | default |
| bootstrap | `True` | default |
| class_weight | `balanced` | as above |
| random_state | `42` | reproducibility |
| n_jobs | `1` | single-threaded on purpose — the *search* is parallelised over subsets instead |

**Baseline** — always predict the single most frequent class → 4/14 = **0.286**.

---

## 8. Honesty caveats (methodological)

- **Selection is not independently validated.** The same 14-fold loop both picks
  the best feature subset (out of 63/127) and reports its score. Searching many
  subsets on 14 flights can hand you a favourable winner by chance — the
  individual folds are honest, but the *selection* over subsets is mildly
  optimistic. The defensible headline is the *simplest strong model*
  (`motors` → Logistic 14/14), not whichever multi-group subset edged it out.
  Same caveat the `FEATURE_STUDY/` README raises on the power side.
- **N = 14, classes of 2–4.** One misclassified flight moves accuracy by ~7
  points; rankings are suggestive, not settled. `none` has only 2 flights;
  `full` is untested.
- **No held-out final test set** (too few flights) — every number is a
  leave-one-flight-out cross-validation estimate.
- **Motors carry the excluded imbalance signal.** `imbalance` is a linear
  combination of the motor commands, so high accuracy from a motor-containing
  subset means "the commands encode the load," not independent detection. The
  motor-free result (orientation alone ≈ 64%) is the genuine detection claim.

---

## 9. Reproducing

```bash
python3 categorize_position_study.py            # 1 Hz, all groups (recommended)
python3 categorize_position_study.py --rate 20  # 20 Hz raw grid, sanity check
```

~15 min (Random-Forest exhaustive search dominates; the linear models are
seconds). Deterministic — same inputs give the same numbers on every run.
Outputs are described in `README.md` §Files.
