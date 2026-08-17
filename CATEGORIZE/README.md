# 12 CATEGORIZE — Predicting payload *position* from flight sensors

Classification sibling of the power-regression studies (`14 REGRESION MODEL/`,
`NO_MASS_STUDY/`). Same 14 flights, same `flight_resampled.csv` inputs, same
**±12 s mission window**, **1 Hz**, **leave-one-flight-out** protocol and
**train-only z-scoring** — but the target is the categorical **`position_payload`**
(`none` / `front` / `rear` / `diagonal`), so this is **4-class classification**,
scored by accuracy / macro-F1 / confusion matrix instead of R² / MAE.

Standalone: `categorize_position_study.py` reads only
`flights/F*/flight_resampled.csv` (found by walking up) and writes only into
`WITH_MASS/`, `WITHOUT_MASS/`, and `comparison_summary.json` here.

See `PLAN.md` for the design rationale. This README reports the results.

---

## The question

Given a flight's sensors, can we predict **where the payload sat**? And — the
part that is an actual result rather than a tautology — can we do it **without
the motor commands**, from how the airframe behaves (orientation, IMU, speed)?

---

## Setup

- **Target:** `position_payload`, 4 classes. `full` was never flown, so it is
  not in the label set and cannot be predicted or tested.
- **Unit of truth is the flight.** The classifier predicts every row of the
  held-out flight, then those rows are collapsed to **one label per flight by
  majority vote**. Rows within a flight are near-identical in cruise, so the
  honest sample size is **14 flights, not ~1,800 rows**. Row-level accuracy is
  reported too, but it is optimistic and not the headline.
- **Evaluation:** leave-one-flight-out (14 folds), z-scored on train flights
  only — identical discipline to the power studies.
- **Models:** multinomial **Logistic Regression** and **LDA** (linear,
  interpretable) + **Random Forest** (tree sanity check), vs a **majority-class
  baseline = 4/14 = 0.286**.
- **Features:** `imbalance` excluded (engineered from position — leakage by
  design); `trajectory` excluded (single-flight indicator). `motors`,
  `orientation`, `imu`, `speed`, `velocity`, `altitude` offered to an
  **exhaustive** search over every group subset. `mass` is the variant axis:
  offered in `WITH_MASS/`, dropped in `WITHOUT_MASS/`.

### The 14 flights

| class | flights | n |
|---|---|---|
| `none` | F01, F02 | 2 |
| `front` | F03, F04, F07, F08 | 4 |
| `rear` | F11, F12, F13, F14 | 4 |
| `diagonal` | F05, F06, F09, F10 | 4 |

---

## Results — headline

| model | best feature set | per-flight accuracy | macro-F1 |
|---|---|---|---|
| **Logistic Regression** | `motors` | **14 / 14 (1.00)** | 1.00 |
| **Random Forest** | `motors` | 13 / 14 (0.93) | 0.94 |
| **LDA** | `motors + orientation + velocity` | 10 / 14 (0.71) | 0.69 |
| majority-class baseline | — | 4 / 14 (0.29) | — |

**Identical with and without mass** (`comparison_summary.json`): LogReg 14/14
both, RF 13/14 both. This is expected — mass is confounded with the label (only
`none` has 0 g) and carries no information to separate front vs rear vs diagonal,
so the two subfolders come out the same. **Mass is irrelevant to position.**

**Position is essentially fully recoverable from the motor commands.** A plain
multinomial logistic regression on the four commands places all 14 held-out
flights correctly. This is the *expected* result, not a surprise: `imbalance`
(which we excluded) is a linear combination of the four motor commands, so the
motor group still carries the full load-distribution signal. The right reading
is **"the motor commands encode where the load is,"** not "we built a clever
detector." RF gets 13/14 (it confuses one `front` flight, F03, with `diagonal`);
LogReg's linear decision boundary happens to be the cleaner fit here.

Note the row-vs-flight gap: LogReg on `motors` scores only **0.63 row accuracy**
but **1.00 per flight** — individual 1 Hz rows are noisy, but each flight's
majority vote is decisive. This is exactly why the flight is the unit of truth.

---

## Results — the honest question: no motor commands

Best subset that uses **no motor-derived signal at all** (orientation / IMU /
speed / velocity / altitude only):

| model | best airframe-only set | accuracy | macro-F1 |
|---|---|---|---|
| Logistic Regression | `orientation` | 9 / 14 (0.64) | 0.63 |
| Random Forest | `orientation + altitude` | 9 / 14 (0.64) | 0.52 |
| LDA | `orientation` | 8 / 14 (0.57) | 0.55 |

**This is the real finding.** With the motor commands removed, **orientation
alone still recovers position at ~64% — more than double the 29% baseline, but
far below the ~100% the motor commands give.** A payload placed off the airframe
centre biases the steady-state roll/pitch (the aircraft leans slightly toward
the load), and that tilt is a genuine, measurable position signature. But at 14
flights and in steady cruise the signal is weak: it separates the broad cases
but is not reliable enough to deploy on its own. `orientation` is consistently the
one non-motor group that carries anything — `imu`, `speed`, `velocity` and
`altitude` add little or nothing on top of it (see the group-importance plots).

Honest caveat, same as everywhere in this project: **14 flights, 2–4 per class.**
One misclassified flight moves accuracy by 7 points, so treat the airframe-only
rankings as suggestive, not settled. The `none` class has only 2 flights, and
`full` is untested. More repeat flights per configuration is the single most
valuable next step — the same conclusion the power studies reach.

---

## Files

Per variant, in `WITH_MASS/` (tag `withmass`) and `WITHOUT_MASS/` (tag `nomass`):

| file | contents |
|---|---|
| `search_<model>_<tag>.csv` | every group subset scored, ranked by per-flight accuracy (models: logreg, lda, rf) |
| `summary_<tag>.json` | winner per model, metrics, baseline |
| `per_flight_predictions_<tag>.csv` | each flight: truth + each model's predicted label + correct flag |
| `coefficients_logreg_<tag>.csv` | per-class standardized Logistic coefficients of the winning combo (all-flights descriptive fit) |
| `plot_search_<tag>.png` | per-flight accuracy vs #features, all subsets, all 3 models + baseline |
| `plot_confusion_<tag>.png` | flight-level confusion matrices, the 3 models side by side |
| `plot_group_importance_<tag>.png` | accuracy with vs without each group (Logistic), conditioned on the strongest group |

And in this folder: `comparison_summary.json` — headline with-mass vs
without-mass numbers for all three models.

---

## Running it

```bash
python3 categorize_position_study.py            # 1 Hz, all sensor groups (recommended)
python3 categorize_position_study.py --rate 20  # 20 Hz raw grid, sanity check
```

About **15 minutes** (the exhaustive Random Forest search over 127 + 63 subsets
is the cost; the two linear models are seconds). Deterministic
(`random_state=42`). To speed up re-runs, lower `n_estimators` in the `rf`
factory in `categorize_position_study.py` — 200 trees is already well past what
14 flights need.
