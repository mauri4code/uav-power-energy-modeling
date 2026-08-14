"""
CATEGORIZE — airframe-only variant: exhaustive search over feature groups with
`motors` (and `mass`) EXCLUDED, to answer "can position be recovered from how the
aircraft flies, with no motor/throttle signal and without being told the mass?"

Reuses every function from categorize_position_study.py unchanged (same window,
same 1 Hz, same leave-one-flight-out + per-flight majority vote protocol); only
the candidate group pool is restricted to orientation, imu, speed, velocity,
altitude.

Output: prints the best (group-subset, accuracy) per model and saves
NO_MOTORS/summary_no_motors.json
"""
import os, json
import categorize_position_study as cat

OUT = os.path.join(cat.HERE, "NO_MOTORS")
os.makedirs(OUT, exist_ok=True)

AIRFRAME_GROUPS = ["orientation", "imu", "speed", "velocity", "altitude"]

data = cat.load(cat.find_flights_dir())
labels = sorted(data[cat.TARGET].unique())
base_acc = cat.baseline(data)
print(f"[NO-MOTORS] baseline (majority class) = {base_acc}")
print(f"  groups offered: {AIRFRAME_GROUPS}\n")

results = {}
for name, make_model in cat.MODELS.items():
    df = cat.exhaustive_search(data, AIRFRAME_GROUPS, make_model, labels, name)
    best = df.iloc[0]
    results[name] = {"groups": best["groups"], "flight_acc": best["flight_acc"],
                      "flight_macro_f1": best["flight_macro_f1"],
                      "n_correct_flights": int(best["n_correct_flights"])}
    print(f"  {cat.MODEL_NICE[name]:20s} best={' + '.join(best['groups']):40s} "
          f"acc={best['flight_acc']:.3f} ({best['n_correct_flights']}/14)  "
          f"macro-F1={best['flight_macro_f1']:.3f}")

with open(os.path.join(OUT, "summary_no_motors.json"), "w") as fh:
    json.dump({"baseline_flight_acc": base_acc, "groups_offered": AIRFRAME_GROUPS,
               "results": results}, fh, indent=2)
print(f"\nSaved -> {OUT}/summary_no_motors.json")
