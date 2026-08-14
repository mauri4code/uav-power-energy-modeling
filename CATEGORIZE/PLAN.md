# 12 CATEGORIZE — Predicting payload *position* from flight sensors (PLAN)

**Status:** proposal for approval. No code written yet.

This study is the **classification** sibling of the power-regression work in
`14 REGRESION MODEL/` and `NO_MASS_STUDY/`. Same 14 flights, same
`flight_resampled.csv` inputs, same leave-one-flight-out discipline — but the
target is now the categorical **`position_payload`**, not `power`.

---

## 1. The question

Given a flight's sensor signals, can we predict **where the payload was placed**
(`none` / `front` / `rear` / `diagonal`)? And — the interesting version — can we
do it *without* reading the motor commands, i.e. from how the airframe actually
behaves (orientation, IMU, speed)?

This is **4-class classification**, so the regression metrics (R², MAE) do not
apply. The reported metrics are **per-flight accuracy, a confusion matrix,
per-class accuracy, and macro-F1**.

---

## 2. The data — and why it constrains everything

Target distribution across the 14 flights:

| Position | Flights | Count |
|---|---|---|
| `none` | F01, F02 | 2 |
| `front` | F03, F04, F07, F08 | 4 |
| `rear` | F11, F12, F13, F14 | 4 |
| `diagonal` | F05, F06, F09, F10 | 4 |
| `full` | — | **0** |

Consequences baked into the design:

- **`full` is never flown** → it cannot be predicted or tested. It is dropped
  from the label set; only 4 classes are in play.
- **14 flights = 14 independent samples.** Rows within a flight are near-identical
  in cruise (motor commands barely move), so the *effective* sample size is the
  number of flights, not the ~30k rows. **The honest metric is per-flight, not
  per-row.**
- **`none` has only 2 flights.** Under leave-one-flight-out, holding out F01
  leaves exactly one `none` example (F02) in training — workable, but the class
  is fragile. Every other class has 3 in training when one is held out.
- **Chance level is not 25%.** The majority-class baseline (always guess one of
  the 4-flight classes) already scores 4/14 ≈ **29%**. Any model must be reported
  against that baseline, not against zero.

---

## 3. Features — what goes in, what is excluded, and why

Same 20 Hz `flight_resampled.csv`, same **±12 s mission window** and **1 Hz**
recommended rate as the power studies, so this study sits on identical rows.

Feature groups **offered** to the exhaustive search:

| group | features | included? |
|---|---|---|
| `motors` | 4 motor commands | ✅ **included** — the key signal; the search decides which help |
| `orientation` | roll, pitch, yaw | ✅ |
| `imu` | 3 gyro + 3 accel | ✅ |
| `speed` | speed_3d, speed_horizontal | ✅ |
| `velocity` | vx, vy, vz | ✅ |
| `altitude` | local altitude | ✅ |
| `mass` | payload_mass | ✅ **in the `WITH_MASS/` run only** (variant axis — see below) |
| `imbalance` | front_rear, diagonal | ❌ **excluded** (your call — it is engineered from position) |
| `trajectory` | one-hot | ❌ excluded (trajectory_3 = single-flight indicator, same reason as the power studies) |

**Why exclude `imbalance` (your instruction):** the two imbalance features were
constructed specifically to encode load distribution — `front` → front_rear +,
`diagonal` → diagonal_imbalance +, etc. Feeding them back to predict position is
circular (leakage-by-design).

**Motors are kept in — and we let the search tell us which features are used.**
You want motor commands in because they are the key signal, and it is not yet
known which ones matter. So `motors` is offered to the exhaustive feature-group
search alongside the airframe-behaviour groups, and the search reports which
combination actually predicts position best. Worth stating in the write-up:
`imbalance` is a linear combination of the four motor commands, so excluding
`imbalance` while keeping `motors` does not remove the load-distribution signal —
the model can reconstruct it. That means high accuracy from a motor-containing
combination should be read as "the motor commands encode the load distribution,"
not as independent detection. Whether the *non-motor* groups (orientation, IMU)
add anything on top is exactly what the group-importance output will show.

**Mass is the variant axis — two subfolders** (mirroring `withmass` / `nomass`
in the Ridge and Ensemble power studies):

- **`WITH_MASS/`** — `mass` offered to the search alongside every other group.
- **`WITHOUT_MASS/`** — identical search with `mass` removed from the pool.

Note for interpretation: mass is confounded with the label — only `none` has 0 g,
while `front`/`rear`/`diagonal` each appear at both 0.24 and 0.48 kg. So mass can
only help separate `none` from the rest; it says nothing about
front-vs-rear-vs-diagonal. The two subfolders make that effect visible instead of
hiding it.

---

## 4. The models

Directly parallel to the Ridge-vs-XGBoost pairing in the power studies:

| role | model | why |
|---|---|---|
| **Primary (interpretable)** | Multinomial **Logistic Regression** (L2) — and/or **LDA** | Linear, gives per-class coefficients you can read the same way as the Ridge coefficients. The classes are near-linearly separable, so this should already be strong. This is the classification analogue of your Ridge study. |
| **Tree counterpart (sanity check)** | **Random Forest** classifier | Non-linear comparison, the way XGBoost pairs with Ridge for power. RF preferred over XGBoost at N=14 — less overfitting, no hyperparameter search drama. |
| **Baselines (mandatory, for honesty)** | Majority-class (≈29%) + stratified-random | So any accuracy number is judged against chance, not zero. |

Features are z-scored with **train-fold statistics only** (same leakage discipline
as everywhere in the project). Deterministic (`random_state=42`).

---

## 5. Evaluation protocol

1. **Leave-one-flight-out**, 14 folds — fit on 13 flights, predict the held-out
   one. No flight on both sides of a fold. Identical to the power studies.
2. Classifier predicts **per row**, then rows are **aggregated to one prediction
   per flight by majority vote** (probability-averaging as a secondary option).
   The label is constant within a flight, so the flight is the unit of truth.
3. **Reported metrics:**
   - Per-flight accuracy (the headline — 14 predictions, X correct).
   - Confusion matrix over the 4 classes.
   - Per-class accuracy / recall + macro-F1 (so the 2-flight `none` class is
     not hidden by the larger classes).
   - Row-level accuracy too, but flagged as optimistic (correlated rows).
4. **Feature-group search** (optional, cheap here): exhaustive over the offered
   groups (2⁵−1 = 31 subsets without motors, 2⁶−1 = 63 with) since Logistic/RF
   are fast — no need for greedy. Reports which groups actually carry position.

---

## 6. Honest caveats to state in the thesis

- **N is tiny.** 14 flights, 2–4 per class. A single misclassified flight swings
  accuracy by 7 points. Confidence intervals will be wide; treat rankings as
  suggestive, not definitive — same limit as the power side.
- **`full` untested, `none` fragile** (2 flights).
- **With-motors accuracy is near-trivial** and must be labelled as such.
- **The real claim is the without-motors variant** — and it may well be weak,
  because in steady cruise a small CoG offset produces only a tiny, possibly
  noise-level roll/pitch bias. That would itself be a legitimate, reportable
  finding: "position is not recoverable from airframe behaviour alone at this
  data scale / in steady cruise."

---

## 7. Proposed deliverables (once approved)

A single standalone script `categorize_position_study.py` in this folder,
following the same conventions as your other studies (reads
`flights/F*/flight_resampled.csv` by walking up, writes only into subfolders
here):

| output | contents |
|---|---|
| `WITH_MASS/` and `WITHOUT_MASS/` | one results folder per variant |
| `confusion_<model>.png` | confusion matrix, per variant/model |
| `per_flight_predictions.csv` | each flight, true vs predicted while held out |
| `search_<model>.csv` | every feature-group subset scored, ranked by per-flight accuracy |
| `plot_group_importance.png` | accuracy with vs without each group (does orientation/IMU add anything over motors?) |
| `coefficients_logreg.csv` | per-class standardized coefficients of the winning combo (interpretable story) |
| `summary.json` | headline accuracies, baselines, best feature set, per model |
| `README.md` | written up in the same style as your other study READMEs |

Runtime: seconds to a couple of minutes (Logistic/LDA/RF on 14 flights are cheap).

---

## 8. Decisions — locked in

1. **Mass:** run **both** — `WITH_MASS/` and `WITHOUT_MASS/` subfolders.
2. **Motors:** **included** as a candidate group; the exhaustive search reports
   which features are actually used.
3. **Models:** **Logistic Regression + LDA** (both, interpretable) + **Random
   Forest** (tree check) + majority-class/random baselines.
4. **Feature-group search:** **on** — exhaustive over every offered group subset,
   ranked by per-flight accuracy, with a group-importance plot.

**Awaiting green light to build `categorize_position_study.py`.**
