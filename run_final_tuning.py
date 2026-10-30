"""
run_final_tuning.py - averaged sweep (n_repeats=7).

Usage:
    python run_final_tuning.py                      # all four models
    python run_final_tuning.py logistic_regression random_forest xgboost
    python run_final_tuning.py transformer          # launch this one from LLMAI (GPU)

Results go to OUTPUT_DIR below, a NEW directory on purpose: neither the
sweep checkpoint's fingerprint nor <model>_final_result.json records
n_repeats, so pointing resume=True at the old "tuning_results" would
reuse the n_repeats=1 pilot results and finish almost instantly.
"""

import sys

from model_tuning import run_all_models_from_csvs

OUTPUT_DIR = "tuning_results_n7"

models = sys.argv[1:] or None  # None = all four

# The importance refit runs locally and has no checkpoint (a single
# transformer refit has taken 2h+), so the transformer gets fewer
# importance repeats than the sweep's 7.
includes_transformer = models is None or "transformer" in models
importance_n_repeats = 3 if includes_transformer else 7

results = run_all_models_from_csvs(
    "tuning_data", OUTPUT_DIR,
    models=models,
    n_repeats=7,
    importance_n_repeats=importance_n_repeats,
    permutation_repeats=2,
    # One-time group-aware split by match: 60% train / 20% validation /
    # 20% held-out test. The sweep (n_repeats=7) runs inside the 80%
    # train+val pool; the untouched 20% is scored once per model at the
    # end with the winning params. This is the number to report, not
    # the sweep's best_score (which is the max over hundreds of
    # combinations and therefore optimistic). Same seed + same CSVs =
    # same test matches for every model and every launch.
    held_out_train_size=0.6,
    held_out_val_size=0.2,
    held_out_test_size=0.2,
    held_out_random_state=42,
    use_ray=True,
    ray_address="10.2.1.189:6379",
    # 6 = the whole transformer grid on the GPU lane, serialized:
    # 6 combos x 7 repeats = 42 fits, one at a time.
    num_gpu_combos=6,
    memory_per_task_by_model={
        "transformer": 4_294_967_296,      # 4 GiB, measured
        "random_forest": 5_368_709_120,    # 5 GiB; observed peak ~4.45 GB
    },
)
