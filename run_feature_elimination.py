"""
run_feature_elimination.py - recursive feature elimination for the
four tuned models, then one final refit on the selected features.

Usage (from the project folder, same one as run_final_tuning.py):
    python run_feature_elimination.py xgboost
    python run_feature_elimination.py logistic_regression random_forest xgboost
    python run_feature_elimination.py transformer      # launch from LLMAI (GPU)

Optional overrides (defaults are in SETTINGS below):
    --drop-fraction 0.25   --n-repeats 7   --importance-n-repeats 3
    --permutation-repeats 2   --tolerance 0.002
    --selection best|smallest_within_tolerance
    --importance-scoring roc_auc_ovr|f1_macro
    --no-ray               run locally instead of on the cluster
    --data-dir tuning_data --results-dir tuning_results_n7
    --ray-address 10.2.1.189:6379

What it does, per model
-----------------------
1. Loads <data-dir>/<model>_features.csv and recomputes the SAME
   60/20/20 match-level split as run_final_tuning.py (seed 42), so
   the held-out test matches are identical to the ones scored there.
2. Reads the model's best_params from
   <results-dir>/<model>/<model>_final_result.json and holds them
   fixed.
3. Runs recursive feature elimination on the 80% train+validation
   pool ONLY - the test matches never influence which features are
   kept.
4. Refits once on train+validation with the selected features and
   scores the held-out test set once.

Outputs, in <results-dir>/<model>/elimination/:
    rfe/rfe_history.jsonl, rfe_curve.csv, rfe_summary.json
        every step: features remaining, what was dropped, validation
        metrics - resumable if interrupted
    <model>_held_out_test_result.json, <model>_held_out_predictions.npz
        the AFTER-elimination test result and predictions (same
        format as the before-elimination files, which are NOT
        touched - they live one folder up)
    elimination_summary.json
        n features before/after/eliminated, elimination order,
        selected features, and before-vs-after test metrics

Do not use the before-vs-after test numbers to decide whether to
keep the elimination - that would turn the test set into a selection
set. Report both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import model_tuning
from model_tuning import split_train_val_test, evaluate_held_out_test
from feature_engineering import load_model_csv
from feature_elimination import run_recursive_feature_elimination

DATA_DIR = "tuning_data"
RESULTS_DIR = "tuning_results_n7"
RAY_ADDRESS = "10.2.1.189:6379"

# Must match run_final_tuning.py so the test matches are identical.
HELD_OUT_SIZES = (0.6, 0.2, 0.2)
HELD_OUT_SEED = 42

MEMORY_PER_TASK = {
    "transformer": 4_294_967_296,      # 4 GiB, measured
    "random_forest": 6_442_450_944,    # 6 GiB
}

# Per-model cost settings. Each elimination step costs n_repeats fits
# (on the cluster) PLUS importance_n_repeats fits and a permutation
# pass over every remaining feature (both run locally on the machine
# you launch from). The transformer is by far the slowest: its fits
# take over an hour each and its permutation pass scores the 80k-row
# validation fold once per feature, so it gets bigger steps and fewer
# repeats. Adjust after watching the first step's wall-clock time.
SETTINGS = {
    "logistic_regression": dict(
        drop_fraction=0.25, n_repeats=7,
        importance_n_repeats=3, permutation_repeats=2),
    "random_forest": dict(
        drop_fraction=0.25, n_repeats=7,
        importance_n_repeats=2, permutation_repeats=2),
    "xgboost": dict(
        drop_fraction=0.25, n_repeats=7,
        importance_n_repeats=3, permutation_repeats=2),
    "transformer": dict(
        drop_fraction=0.30, n_repeats=3,
        importance_n_repeats=1, permutation_repeats=1),
}

# primary_metric matches the proposal's primary metric. tolerance is
# in ROC-AUC units: the split-to-split std in the sweeps was about
# 0.007-0.01, so 0.002 is deliberately tighter than one std - the
# repeats use the same seeds at every step, so step-to-step
# differences are less noisy than that, but widen it if elimination
# keeps stopping after one or two steps.
COMMON = dict(
    primary_metric="roc_auc_mean",
    importance_scoring="roc_auc_ovr",
    tolerance=0.002,
    min_features=5,
    selection="best",
)

TEST_METRICS = ("accuracy", "f1", "precision", "recall", "roc_auc")


def run_one(model_name: str, args: argparse.Namespace) -> None:

    settings = {**SETTINGS[model_name], **COMMON}

    for key, value in (
        ("drop_fraction", args.drop_fraction),
        ("n_repeats", args.n_repeats),
        ("importance_n_repeats", args.importance_n_repeats),
        ("permutation_repeats", args.permutation_repeats),
        ("tolerance", args.tolerance),
        ("selection", args.selection),
        ("importance_scoring", args.importance_scoring),
    ):
        if value is not None:
            settings[key] = value

    results_root = Path(args.results_dir) / model_name
    elim_dir = results_root / "elimination"
    elim_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.data_dir) / f"{model_name}_features.csv"
    X, y, groups = load_model_csv(csv_path)

    split = split_train_val_test(
        X, y, groups,
        train_size=HELD_OUT_SIZES[0],
        val_size=HELD_OUT_SIZES[1],
        test_size=HELD_OUT_SIZES[2],
        random_state=HELD_OUT_SEED,
    )

    X_tv = split["X_trainval"]
    y_tv = split["y_trainval"]
    g_tv = split["groups_trainval"]
    X_test = split["X_test"]
    y_test = split["y_test"]

    with open(results_root / f"{model_name}_final_result.json") as file:
        best_params = json.load(file)["best_params"]

    print(
        f"\n{'=' * 60}\nFeature elimination: {model_name} "
        f"({X.shape[1]} features)\nbest_params: {best_params}\n"
        f"settings: {settings}\n{'=' * 60}"
    )

    # The transformer's builder bakes in its exact column list, so it
    # has to be rebuilt for every feature subset. Category
    # cardinalities are sized from the FULL dataset's columns (all
    # splits), not the train+val pool - same reason as the original
    # run: the test set can contain category codes the pool lacks.
    builder_factory = None

    if model_name == "transformer":

        from tabular_transformer import make_transformer_builder

        def builder_factory(X_subset, _X_full=X):
            return make_transformer_builder(_X_full[list(X_subset.columns)])

    summary = run_recursive_feature_elimination(
        model_name, best_params, X_tv, y_tv,
        output_dir=elim_dir / "rfe",
        n_repeats=settings["n_repeats"],
        test_size=split["sweep_test_size"],
        groups=g_tv,
        base_random_state=HELD_OUT_SEED,
        primary_metric=settings["primary_metric"],
        drop_fraction=settings["drop_fraction"],
        min_features=settings["min_features"],
        tolerance=settings["tolerance"],
        importance_n_repeats=settings["importance_n_repeats"],
        permutation_repeats=settings["permutation_repeats"],
        importance_scoring=settings["importance_scoring"],
        selection=settings["selection"],
        builder_factory=builder_factory,
        use_ray=not args.no_ray,
        ray_address=None if args.no_ray else args.ray_address,
        memory_per_task=MEMORY_PER_TASK.get(model_name),
        num_gpu_combos=1 if model_name == "transformer" else 0,
    )

    selected = summary["selected_features"]

    # ----------------------------------------------------------
    # One final refit on the selected features, scored once on the
    # untouched test matches. Skipped if a previous run already
    # produced it (it is the expensive, one-shot step).
    # ----------------------------------------------------------

    after_path = elim_dir / f"{model_name}_held_out_test_result.json"

    if after_path.exists():
        print(f"[{model_name}] after-elimination test result already "
              f"exists at {after_path} - not refitting.")
        with open(after_path) as file:
            after = json.load(file)
    else:
        if builder_factory is not None:
            model_tuning.MODEL_BUILDERS[model_name] = builder_factory(
                X_tv[selected]
            )

        after = evaluate_held_out_test(
            model_name, best_params,
            X_tv[selected], y_tv,
            X_test[selected], y_test,
            output_dir=elim_dir,
            random_state=HELD_OUT_SEED,
        )

    with open(results_root / f"{model_name}_held_out_test_result.json") as file:
        before = json.load(file)

    report = {
        "model": model_name,
        "n_features_before": summary["n_features_start"],
        "n_features_after": summary["n_selected"],
        "n_eliminated": summary["n_eliminated"],
        "eliminated_in_order": summary["eliminated_in_order"],
        "selected_features": selected,
        "validation_primary_metric": summary["primary_metric"],
        "validation_at_selected": summary["final_performance"],
        "validation_best_seen": summary["best_performance"],
        "rfe_settings": {
            key: settings[key] for key in (
                "drop_fraction", "n_repeats", "importance_n_repeats",
                "permutation_repeats", "tolerance", "selection",
                "min_features", "importance_scoring",
            )
        },
        "test_before": {k: before[k] for k in TEST_METRICS},
        "test_after": {k: after[k] for k in TEST_METRICS},
    }

    with open(elim_dir / "elimination_summary.json", "w") as file:
        json.dump(report, file, indent=2)

    print(
        f"\n[{model_name}] eliminated {report['n_eliminated']} of "
        f"{report['n_features_before']} features "
        f"({report['n_features_after']} kept)."
    )
    for metric in TEST_METRICS:
        print(
            f"    test {metric:9s} before={before[metric]:.4f} "
            f"after={after[metric]:.4f}"
        )


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("models", nargs="+", choices=list(SETTINGS))
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--ray-address", default=RAY_ADDRESS)
    parser.add_argument("--no-ray", action="store_true")
    parser.add_argument("--drop-fraction", type=float, default=None)
    parser.add_argument("--n-repeats", type=int, default=None)
    parser.add_argument("--importance-n-repeats", type=int, default=None)
    parser.add_argument("--permutation-repeats", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument(
        "--importance-scoring", default=None,
        help="metric permutation importance ranks features by "
             "(default roc_auc_ovr; f1_macro is the repo's multiclass "
             "default and is robust to everything roc_auc_ovr is not)",
    )
    parser.add_argument(
        "--selection", choices=["best", "smallest_within_tolerance"],
        default=None,
    )
    args = parser.parse_args()

    for model_name in args.models:
        run_one(model_name, args)


if __name__ == "__main__":
    main()
