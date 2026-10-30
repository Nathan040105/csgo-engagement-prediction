"""
feature_elimination.py

Backward recursive feature elimination: fit, rank features by
permutation importance, drop the least important, retrain - keep
going as long as performance holds, stop (and revert to the best
feature set found) the moment it drops.

Built entirely on top of model_tuning.py's existing run_grid_sweep
and run_feature_importance_analysis, not a separate implementation -
this reuses the same multiclass-aware metrics, NaN-safe handling,
per-repeat seeding, and checkpointing already validated there,
rather than risking a second, subtly different way of fitting and
scoring a model.

Stopping rule: each step is compared against the BEST step seen so
far (not just the immediately preceding one) - a single noisy dip
followed by recovery shouldn't trigger a premature stop, and
n_repeats-averaged metrics still have some run-to-run variance even
though far less than a single split would. If you specifically want
"stop at the very first step that's worse than its immediate
predecessor" instead, set tolerance high enough that only genuine
regressions matter, or ask - it's a one-line change.

Changes in this version (all backward compatible - every new
argument has a default that reproduces the old behaviour, except
primary_metric, see below):

    - primary_metric now defaults to "roc_auc_mean" (the project's
      primary metric per the proposal; now well-defined because the
      full corpus has 204 mutual_kill rows). Pass "f1_mean" for the
      old default.
    - Resume bug fixed: resuming used to re-run the last recorded
      step's feature set under a new step number instead of
      continuing from the features that step had decided to drop,
      and resuming a run that had already finished re-ran the
      stopping step. Both wasted a full step of compute.
    - drop_fraction: drop a fraction of the remaining features per
      step (needed to get through 159-549 features in a sensible
      number of steps for the slow models).
    - builder_factory: re-registers the model builder from each
      step's feature subset. REQUIRED for the transformer, whose
      builder bakes in the exact column list at construction (see
      tabular_transformer.make_transformer_builder) and would
      KeyError the moment a column is dropped.
    - Ray / GPU / permutation_repeats arguments are forwarded to
      run_grid_sweep and run_feature_importance_analysis (before,
      every step ran locally and sequentially).
    - selection: "best" (old behaviour) or
      "smallest_within_tolerance".
    - Writes n_selected / n_eliminated / eliminated_in_order to
      rfe_summary.json and a per-step rfe_curve.csv.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import model_tuning
from model_tuning import run_grid_sweep, run_feature_importance_analysis


_STEP_METRIC_KEYS = (
    "roc_auc_mean", "roc_auc_std",
    "f1_mean", "f1_std",
    "accuracy_mean", "accuracy_std",
    "precision_mean", "recall_mean",
)


def run_recursive_feature_elimination(
    model_name: str,
    best_params: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    output_dir: Path | str,
    n_repeats: int = 20,
    test_size: float = 0.2,
    groups: np.ndarray | None = None,
    base_random_state: int = 42,
    primary_metric: str = "roc_auc_mean",
    n_features_to_drop_per_step: int = 1,
    min_features: int = 5,
    tolerance: float = 0.0,
    importance_n_repeats: int = 10,
    importance_scoring: str | None = None,
    resume: bool = True,
    show_progress: bool = True,
    drop_fraction: float | None = None,
    builder_factory: Callable[[pd.DataFrame], Callable] | None = None,
    permutation_repeats: int = 10,
    use_ray: bool = False,
    ray_address: str | None = None,
    max_concurrent_tasks: int = 64,
    memory_per_task: int | None = None,
    ray_py_modules: list[str] | None = None,
    num_gpu_combos: int = 0,
    selection: str = "best",
) -> dict[str, Any]:
    """
    Parameters
    ----------
    X, y, groups
        Pass the TRAIN+VALIDATION pool only
        (model_tuning.split_train_val_test's X_trainval /
        y_trainval / groups_trainval), never the full dataset. If
        the held-out test matches are in here, they influence which
        features get kept and the final test score is no longer
        untouched.
    test_size
        The sweep's internal validation fraction WITHIN the pool you
        pass. Use split_train_val_test's returned sweep_test_size
        (0.25 for a 60/20/20 split) so each step is validated the
        same way the hyperparameter sweep was.
    best_params
        Hyperparameters used for EVERY elimination step, held fixed
        throughout - typically the best_params from a prior
        run_full_tuning() call. Re-tuning hyperparameters at every
        elimination step (hundreds of features x many steps) would
        be prohibitively expensive, and would also confound "did
        performance change because of the features or because of
        the hyperparameters" - fixing hyperparameters isolates the
        feature-set effect, which is the whole point of this.
    primary_metric
        Which of run_grid_sweep's output columns decides "did
        performance get worse". Defaults to roc_auc_mean, matching
        the project's primary evaluation metric. Use "f1_mean" if a
        thin class ever makes roc_auc undefined again (see
        model_tuning.py's own primary_metric guidance).
    n_features_to_drop_per_step
        How many of the least-important remaining features get
        removed each round (used when drop_fraction is None). 1 gives
        the finest-grained (and most defensible) elimination order
        but is the most expensive to run all the way down.
    drop_fraction
        If given (e.g. 0.25), each step drops
        max(1, ceil(drop_fraction * features_remaining)) features
        instead of a fixed count - big steps while there are many
        features, finer steps as the set shrinks. Overrides
        n_features_to_drop_per_step. With 159 features and
        min_features=5, 0.25 takes about 12 steps, 0.3 about 10.
    min_features
        Elimination stops once this few features remain, regardless
        of performance - a safety floor so it can't eliminate itself
        down to nothing. A step never drops below this floor, and the
        floor-sized set is itself evaluated before stopping.
    tolerance
        How much worse than the best-seen performance is tolerated
        before stopping. 0.0 (default) stops at literally any
        decrease. A small positive value (e.g. 0.002) tolerates
        minor noise before calling it a real regression. Also the
        margin used by selection="smallest_within_tolerance".
    importance_n_repeats, permutation_repeats, importance_scoring
        Passed to run_feature_importance_analysis. Pass
        importance_scoring="roc_auc_ovr" to rank features by the same
        metric the steps are judged on (the default for multiclass is
        f1_macro).
    builder_factory
        Optional callable (X_subset) -> builder. When given, it is
        called at the start of every step and its result is
        registered as model_tuning.MODEL_BUILDERS[model_name] before
        that step's fits. Needed for the transformer (see module
        docstring). Size anything that depends on category
        cardinality from the FULL dataset's columns, not the
        train+val pool's, or the held-out test set can contain codes
        the embedding tables don't cover.
    use_ray, ray_address, max_concurrent_tasks, memory_per_task,
    ray_py_modules, num_gpu_combos
        Forwarded to run_grid_sweep for each step's repeats. For the
        transformer, num_gpu_combos=1 routes the step's single
        combination onto the GPU lane.
    selection
        "best": revert to the step with the highest primary_metric
        (the original behaviour).
        "smallest_within_tolerance": the step with the FEWEST
        features whose primary_metric is within `tolerance` of the
        best step - the most parsimonious set that is statistically
        no worse, usually the more useful answer to "how many
        features can be eliminated".
    resume
        If True and output_dir already has a partial run, picks up
        from the features the last completed step decided to drop,
        rather than starting over - each step's own model_tuning.py
        checkpoint (in output_dir/step_<n>/) also independently
        resumes if that particular step was interrupted mid-sweep.
        The importance refit inside a step has no checkpoint of its
        own, so an interruption there redoes that refit.

    Returns
    -------
    dict with:
        selected_features   - the winning feature subset
        eliminated_features - every feature removed (in X's column
                               order)
        eliminated_in_order - the same features in the order they
                               were eliminated
        n_features_start / n_selected / n_eliminated
        final_performance   - primary_metric at the chosen step
        n_steps_this_run    - steps run this call (not counting
                               resumed-from-checkpoint steps)
    Also writes, to output_dir:
        rfe_history.jsonl - one line per step (features remaining,
                             what was dropped, performance and the
                             step's metrics)
        rfe_curve.csv     - the same, one row per step, for plotting
                             performance against number of features
        rfe_summary.json  - the final selected/eliminated split.
    """

    if selection not in ("best", "smallest_within_tolerance"):
        raise ValueError(
            f"selection must be 'best' or 'smallest_within_tolerance', "
            f"got {selection!r}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "rfe_history.jsonl"

    history: list[dict[str, Any]] = []

    if resume and history_path.exists():
        with open(history_path) as file:
            for line in file:
                line = line.strip()
                if line:
                    history.append(json.loads(line))

    finished = False

    if history:
        last = history[-1]
        step = last["step"]
        best_step = max(history, key=lambda h: h["performance"])

        if last["dropped_this_step"]:
            # The last recorded step decided to drop these AFTER it
            # was recorded - continue from what's left, not from the
            # feature set that step was evaluated on.
            dropped = set(last["dropped_this_step"])
            current_features = [
                f for f in last["remaining_features"] if f not in dropped
            ]
        else:
            # An empty dropped list is only ever written by the
            # stopping step - the run already finished.
            current_features = list(last["remaining_features"])
            finished = True

        if show_progress:
            state = "already finished" if finished else (
                f"{len(current_features)} features remaining"
            )
            print(
                f"[RFE] Resuming after step {step} ({state}; best so "
                f"far: step {best_step['step']}, "
                f"{primary_metric}={best_step['performance']:.4f})"
            )
    else:
        current_features = list(X.columns)
        step = 0
        best_step = None

    n_new_steps = 0

    if not history and X.shape[1] < min_features:
        raise ValueError(
            f"X has {X.shape[1]} features, fewer than "
            f"min_features={min_features}."
        )

    while not finished and len(current_features) >= min_features:

        step += 1
        X_subset = X[current_features]

        if builder_factory is not None:
            model_tuning.MODEL_BUILDERS[model_name] = builder_factory(X_subset)

        param_grid = {name: [value] for name, value in best_params.items()}
        step_dir = output_dir / f"step_{step}"

        results_df, _ = run_grid_sweep(
            model_name, X_subset, y,
            param_grid=param_grid,
            n_repeats=n_repeats,
            test_size=test_size,
            groups=groups,
            base_random_state=base_random_state,
            output_dir=step_dir,
            resume=resume,
            show_progress=False,
            use_ray=use_ray,
            ray_address=ray_address,
            max_concurrent_tasks=max_concurrent_tasks,
            memory_per_task=memory_per_task,
            ray_py_modules=ray_py_modules,
            num_gpu_combos=num_gpu_combos,
        )

        row = results_df.iloc[0]
        performance = float(row[primary_metric])

        if math.isnan(performance):
            raise RuntimeError(
                f"[RFE] {primary_metric} is NaN at step {step} "
                f"({len(current_features)} features) - every repeat "
                f"failed or was undefined, so this step can't be "
                f"compared. Check the step's sweep output in "
                f"{step_dir} (or switch primary_metric)."
            )

        step_record = {
            "step": step,
            "n_features": len(current_features),
            "remaining_features": current_features,
            "dropped_this_step": [],
            "performance": performance,
            "primary_metric": primary_metric,
            "metrics": {
                key: float(row[key])
                for key in _STEP_METRIC_KEYS
                if key in row.index and not pd.isna(row[key])
            },
        }

        if show_progress:
            print(
                f"[RFE] Step {step}: {len(current_features)} features "
                f"-> {primary_metric}={performance:.4f}"
            )

        if best_step is None or performance > best_step["performance"]:
            best_step = step_record

        elif performance < best_step["performance"] - tolerance:

            if show_progress:
                print(
                    f"[RFE] Performance dropped from "
                    f"{best_step['performance']:.4f} (step "
                    f"{best_step['step']}, {best_step['n_features']} "
                    f"features) to {performance:.4f} - stopping."
                )

            history.append(step_record)
            with open(history_path, "a") as file:
                file.write(json.dumps(step_record) + "\n")

            n_new_steps += 1
            break

        if drop_fraction is not None:
            n_to_drop = max(1, math.ceil(drop_fraction * len(current_features)))
        else:
            n_to_drop = n_features_to_drop_per_step

        # Never step below the floor.
        n_to_drop = min(n_to_drop, len(current_features) - min_features)

        if n_to_drop <= 0:

            # At the floor: this set has just been evaluated, and
            # there is nothing left to drop. (An empty dropped list
            # is also how a finished run is recognised on resume.)
            if show_progress:
                print(
                    f"[RFE] Reached min_features={min_features} - "
                    f"stopping."
                )

            history.append(step_record)
            with open(history_path, "a") as file:
                file.write(json.dumps(step_record) + "\n")

            n_new_steps += 1
            break

        # Only computed when we are actually going to drop something -
        # the importance refit is the slowest part of a step for the
        # transformer, and the stopping step has no use for it.
        importance_df = run_feature_importance_analysis(
            model_name, best_params, X_subset, y,
            n_repeats=importance_n_repeats,
            test_size=test_size,
            groups=groups,
            base_random_state=base_random_state,
            permutation_repeats=permutation_repeats,
            scoring=importance_scoring,
        )

        ranked = importance_df.sort_values("importance_mean")
        to_drop = ranked["feature"].head(n_to_drop).tolist()
        step_record["dropped_this_step"] = to_drop

        history.append(step_record)
        with open(history_path, "a") as file:
            file.write(json.dumps(step_record) + "\n")

        n_new_steps += 1

        if show_progress:
            print(f"[RFE] Dropping {len(to_drop)}: {to_drop}")

        current_features = [f for f in current_features if f not in to_drop]

    # ----------------------------------------------------------
    # Choose the final feature set
    # ----------------------------------------------------------

    best_step = max(history, key=lambda h: h["performance"])

    if selection == "smallest_within_tolerance":
        eligible = [
            h for h in history
            if h["performance"] >= best_step["performance"] - tolerance
        ]
        chosen = min(
            eligible, key=lambda h: (h["n_features"], -h["performance"])
        )
    else:
        chosen = best_step

    if show_progress:
        print(
            f"\n[RFE] Final selection ({selection}): "
            f"{len(chosen['remaining_features'])} of {X.shape[1]} "
            f"features (step {chosen['step']}), "
            f"{primary_metric}={chosen['performance']:.4f} "
            f"(best seen {best_step['performance']:.4f} at step "
            f"{best_step['step']})"
        )

    selected = list(chosen["remaining_features"])
    selected_set = set(selected)

    eliminated_features = [f for f in X.columns if f not in selected_set]

    eliminated_in_order: list[str] = []
    for record in history:
        if record["step"] < chosen["step"]:
            eliminated_in_order.extend(record["dropped_this_step"])

    summary = {
        "selected_features": selected,
        "eliminated_features": eliminated_features,
        "eliminated_in_order": eliminated_in_order,
        "n_features_start": int(X.shape[1]),
        "n_selected": len(selected),
        "n_eliminated": len(eliminated_features),
        "final_performance": chosen["performance"],
        "final_step": chosen["step"],
        "best_performance": best_step["performance"],
        "best_step": best_step["step"],
        "primary_metric": primary_metric,
        "selection": selection,
        "tolerance": tolerance,
        "n_steps_this_run": n_new_steps,
    }

    with open(output_dir / "rfe_summary.json", "w") as file:
        json.dump(summary, file, indent=2)

    curve_rows = []
    for record in history:
        curve_row = {
            "step": record["step"],
            "n_features": record["n_features"],
            "performance": record["performance"],
            "n_dropped_after": len(record["dropped_this_step"]),
        }
        curve_row.update(record.get("metrics", {}))
        curve_rows.append(curve_row)

    pd.DataFrame(curve_rows).to_csv(output_dir / "rfe_curve.csv", index=False)

    return summary
