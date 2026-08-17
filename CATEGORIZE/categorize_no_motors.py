"""
CATEGORIZE — airframe-only variant: exhaustive search over feature groups with
`motors` (and `mass`) EXCLUDED, to answer "can position be recovered from how the
aircraft flies, with no motor/throttle signal and without being told the mass?"

Reuses every function from categorize_position_study.py unchanged (same window,
same 1 Hz, same leave-one-flight-out + per-flight majority vote protocol,
including run_setting() so this variant gets the same plots/CSVs as WITH_MASS/
WITHOUT_MASS); only the candidate group pool is restricted to orientation, imu,
speed, velocity, altitude.

Output (in NO_MOTORS/), same layout as WITH_MASS/WITHOUT_MASS:
    search_<model>_no_motors.csv          every group subset, ranked by flight accuracy
    summary_no_motors.json                winner per model, metrics, baseline
    per_flight_predictions_no_motors.csv  each flight: truth + each model's prediction
    coefficients_logreg_no_motors.csv     per-class standardized coefficients (winning combo)
    plot_search_no_motors.png             flight accuracy vs #features, all subsets
    plot_confusion_no_motors.png          confusion matrices, the 3 models side by side
    plot_group_importance_no_motors.png   accuracy with vs without each group (Logistic)
"""
import categorize_position_study as cat

OUT = cat.os.path.join(cat.HERE, "NO_MOTORS")

AIRFRAME_GROUPS = ["orientation", "imu", "speed", "velocity", "altitude"]

data = cat.load(cat.find_flights_dir())
labels = sorted(data[cat.TARGET].unique())
base_acc = cat.baseline(data)
print(f"[NO-MOTORS] baseline (majority class) = {base_acc}")
print(f"  groups offered: {AIRFRAME_GROUPS}\n")

cat.run_setting(data, AIRFRAME_GROUPS, "no_motors", "no motors (airframe only)",
                 labels, base_acc, OUT)
print(f"\nSaved -> {OUT}/")
