# UAV Power & Energy Modeling — Analysis Code

Code used to produce the results in the thesis *"UAV Energy Consumption Modeling
Accounting for Payload Distribution and Mission Profiles."* This repository holds
the **code only** — no flight data, raw logs, or generated results/figures are
included (see [Data](#data) below).

## Structure

```
pipeline/               Raw-log -> clean-dataset pipeline (run in numeric order)
  01_export_bag.py        ROS bag -> per-topic CSV
  01b_extract_ulg_motors.py   PX4 .ulg -> motor commands CSV
  02_compute_euler.py     attitude quaternion -> roll/pitch/yaw
  03_outlier_removal.py   IQR filter on linear accelerations
  04_resampling.py        heterogeneous-rate streams -> common 20 Hz grid
  05_prepare_ml.py        assemble the ML-ready per-flight table
  06_train_xgboost.py     first XGBoost training pass (GroupKFold + RandomizedSearchCV)
  07-22_*.py               downstream analyses (learning curves, correlation
                           plots, flight-level models, mission-window studies,
                           synthetic-flight checks, ...)
  heatmap/
    09_motor_heatmap.py    per-flight motor-usage heatmap (rotor diagram overlay)
    10_config_heatmap.py   per-CONFIGURATION average heatmap (none/front/rear/diagonal)
    animation/
      make_heatmap_animation.py   generates the two HTML files below
      motor_animation.html        interactive per-flight animated heatmap
      motor_animation_grid.html   all 14 flights animating together
      Rotors_poss.jpeg            airframe photo used as the heatmap base

NO_MASS_STUDY/           XGBoost power-prediction studies on the +/-12 s mission
                         window (1 Hz), incl. the tuned model (mission12_study.py),
                         the data-size sensitivity curve (datasize_study.py), and
                         the tune->retrain->test-on-unseen check (holdout_pipeline_study.py)

REGRESSION_MODEL/        Ridge regression, the XGBoost+Ridge stacking ensemble,
                         and the head-to-head model comparison

CATEGORIZE/              Payload-POSITION classification (Logistic Regression,
                         LDA, Random Forest), with/without payload mass, and the
                         airframe-only (no motor commands) variant

LSTM/                    LSTM power-prediction comparison (same protocol as
                         NO_MASS_STUDY, deep-learning baseline)
```

## Protocol used throughout

Every study in `NO_MASS_STUDY/`, `REGRESSION_MODEL/`, `CATEGORIZE/`, and `LSTM/`
follows the same discipline, so results are directly comparable:

- **1 Hz**, one row per real battery reading, over a **±12 s mission window**
  around cruise (the smallest fixed margin that captures motor arming on every
  flight).
- **Leave-one-flight-out cross-validation**: fit on 13 (or 13-of-14) flights,
  predict the held-out flight, repeat so every flight is held out exactly once.
- **Train-only standardisation**: the scaler is fit on the training flights of
  each fold only, never on the held-out flight.
- Each script is **standalone**: it locates `flights/F*/flight_resampled.csv` by
  walking up from its own location, and writes only into its own output folder.

## Data

The flight data (`flights/F*/flight_resampled.csv`) is **not included** in this
repository. To run any script, place the per-flight CSVs (produced by the
`pipeline/` stage) under a `flights/` folder that the script can find by walking
up from its own directory — see each script's docstring (`find_flights_dir()`)
for the exact lookup logic.

## Requirements

Python 3, with `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`,
`joblib`, and `torch` (for `LSTM/lstm_study.py` only). No GPU required.

## Notes

- Result artifacts (CSVs, JSON summaries, plots) are intentionally excluded via
  `.gitignore` — each script regenerates them into its own subfolder when run.
- Several scripts print their reasoning as they run (feature-selection paths,
  cross-validation folds, etc.) — the printed output is part of the
  documentation, not just logging.
