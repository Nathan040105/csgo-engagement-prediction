"""
model_tuning.py

Fully automated hyperparameter sweep, repeated-split evaluation, and
feature importance analysis, generalized across model types.

For each hyperparameter combination in a grid (full cartesian
product, enumerated in nested-loop order - e.g. for params A, B, C
each with values [1,2,3]: (1,1,1),(1,1,2),(1,1,3),(1,2,1),...):

    1. Repeat N_REPEATS times:
        - draw a fresh random train/test split
        - fit the model with that hyperparameter combination
        - record ROC-AUC, F1, precision, recall, accuracy
        - record cheap "native" feature importance (coefficients /
          Gini / gain - whichever the model provides for free)
    2. Average all of the above across the N_REPEATS splits.

Produces:
    - a results CSV, one row per hyperparameter combination
    - one marginal-effect plot per hyperparameter (metric vs that
      parameter's value, averaged over every other parameter)
    - a feature importance bar chart for the best-performing
      combination, using proper permutation importance (a separate,
      lighter-weight pass - see note in run_feature_importance_analysis)

IMPORTANT - COMPUTE COST:
A grid of P parameters x V values each, with N_REPEATS repeats,
fits the model (V^P) * N_REPEATS times. 3 params x 3 values x 100
repeats = 2,700 fits. This is fine for logistic regression / random
forest, slower for gradient boosting, and likely impractical for a
neural architecture (e.g. a Tabular Transformer) without reducing
N_REPEATS and/or the grid size for that model specifically. Start
with a small grid and a small N_REPEATS to sanity check runtime
before committing to a full sweep.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier

# Ray is optional - everything in this file works without it
# installed (plain sequential execution, the original behavior).
# It's only needed to pass use_ray=True to run_grid_sweep /
# run_full_tuning / run_all_models for distributed execution
# across a cluster - see run_grid_sweep's docstring.
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


# ==========================================================
# Model Registry
#
# Each builder takes a dict of hyperparameters and returns a
# fresh, unfitted, sklearn-compatible estimator (fit / predict /
# predict_proba). To add a model, add an entry here - the rest of
# this module (sweeping, splitting, evaluation, plotting) is
# generic and doesn't need to change.
#
# A future Tabular Transformer can plug in here too, as long as
# it's wrapped to expose the same fit/predict/predict_proba
# interface (e.g. via skorch, or a small custom wrapper class).
# ==========================================================

def _build_logistic_regression(params: dict[str, Any], seed: int):
    # Wrapped in a scaling pipeline: logistic regression coefficient
    # magnitudes are only comparable to each other (and thus usable
    # as a "native" importance signal) if features are on the same
    # scale. random_state is set here (not just on the split) so a
    # given repeat's fit is fully reproducible - some solvers
    # (liblinear, saga) have their own internal randomness that the
    # split's seed alone doesn't control.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=seed, **params)),
    ])


def _build_random_forest(params: dict[str, Any], seed: int):
    # random_state controls bootstrap sampling and per-split feature
    # subsampling - without it, two calls with the identical
    # train/test split still produce different fitted trees and
    # therefore different metrics, which defeats the point of
    # checkpointing a repeat's result as if it were reproducible.
    return RandomForestClassifier(random_state=seed, **params)


class _EarlyStoppingXGBClassifier(XGBClassifier):
    """
    XGBClassifier with early stopping against its OWN internal
    validation slice, carved out of whatever X/y it's given at
    fit() time - never the sweep's own held-out test portion (see
    evaluate_held_out_test) and never even the sweep's per-repeat
    validation split. This matters: if early stopping instead used
    either of those, the number of boosting rounds would have been
    CHOSEN using the same data later used to score the result,
    reintroducing the exact selection-bias problem
    split_train_val_test/evaluate_held_out_test exist to prevent -
    just moved one level down, from "which hyperparameters" to
    "how many rounds". A model gets to pick its own stopping point
    from data nobody else ever sees the score against.

    Mirrors TabularTransformerClassifier's own internal
    train/validation carve-out in tabular_transformer.py - same
    pattern, same reasoning, applied to XGBoost's native
    early_stopping_rounds instead of a hand-rolled training loop.

    Falls back to fitting on 100% of the data with no early stopping
    if there isn't enough of it to carve out a meaningful internal
    slice (n < 20) - same safety floor as the transformer's fit(),
    for the same reason: an early-stopping decision made on a
    handful of rows is noise, not signal.
    """

    def __init__(
        self,
        early_stopping_val_fraction: float = 0.15,
        **kwargs: Any,
    ) -> None:
        self.early_stopping_val_fraction = early_stopping_val_fraction
        super().__init__(**kwargs)

    def _wrapper_params(self) -> set[str]:
        # Tells XGBoost's OWN get_xgb_params() (distinct from generic
        # sklearn get_params() - this is the method actually used to
        # build the dict handed to the native C++ booster) to exclude
        # this attribute, the same way it already excludes
        # early_stopping_rounds/n_estimators/etc. Without this,
        # get_xgb_params() includes early_stopping_val_fraction as if
        # it were a real training parameter, and the native booster
        # warns "Parameters: { early_stopping_val_fraction } are not
        # used" on every single fit - confirmed by triggering it
        # directly. Cosmetically harmless on its own, but that's one
        # warning per fit across an entire sweep, which is real noise
        # to dig through, not just an ignorable one-off.
        return super()._wrapper_params() | {"early_stopping_val_fraction"}

    def fit(self, X, y, **kwargs):

        n = len(X)

        if n < 20:
            return super().fit(X, y, **kwargs)

        seed = self.random_state if self.random_state is not None else 0
        rng = np.random.default_rng(seed)

        n_val = max(1, int(n * self.early_stopping_val_fraction))
        shuffled = rng.permutation(n)
        val_pos, fit_pos = shuffled[:n_val], shuffled[n_val:]

        if hasattr(X, "iloc"):
            X_fit, X_val = X.iloc[fit_pos], X.iloc[val_pos]
        else:
            X_fit, X_val = X[fit_pos], X[val_pos]

        if hasattr(y, "iloc"):
            y_fit, y_val = y.iloc[fit_pos], y.iloc[val_pos]
        else:
            y_fit, y_val = y[fit_pos], y[val_pos]

        return super().fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )


def _build_xgboost(params: dict[str, Any], seed: int):
    # eval_metric was "logloss" (binary) - silently inert before,
    # since eval_metric does nothing without an eval_set, but
    # actively wrong now that early stopping needs one: it raises
    # ("label and prediction size not match, hint: use merror or
    # mlogloss") the moment it's actually exercised against this
    # dataset's real 6 classes - confirmed by triggering it
    # directly, not assumed. "mlogloss" is the multiclass-correct
    # metric early stopping should have been watching all along.
    #
    # early_stopping_rounds is fixed here rather than swept in
    # DEFAULT_PARAM_GRIDS: it's a training-mechanics knob (how much
    # patience before giving up), not a model-capacity hyperparameter
    # the research questions care about comparing.
    return _EarlyStoppingXGBClassifier(
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        random_state=seed,
        **params,
    )


MODEL_BUILDERS: dict[str, Callable[[dict[str, Any], int], Any]] = {
    "logistic_regression": _build_logistic_regression,
    "random_forest": _build_random_forest,
    "xgboost": _build_xgboost,
}

# Widened for cluster-scale compute (was a much smaller grid sized
# for single-core sequential runs) - see each model's comment for
# the reasoning behind its specific range.
DEFAULT_PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "logistic_regression": {
        # Wider span, still log-spaced - the useful signal in C is
        # its order of magnitude, not fine-grained values within one.
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300, 400, 500],
        # Added a shallow (3) and a deep (20) option beyond the
        # original 5/10/None span, to actually bracket the
        # under/overfitting boundary rather than just sampling
        # around the middle of it.
        "max_depth": [3, 5, 8, 12, 20, None],
        "min_samples_leaf": [1, 2, 5, 10, 20],
        # Not swept before - controls how many features each split
        # considers. "sqrt" is sklearn's classification default;
        # including None (all features, higher correlation between
        # trees) and a fractional option gives real range either
        # side of that default rather than just confirming it.
        "max_features": ["sqrt", "log2", 0.5, None],
    },
    "xgboost": {
        # n_estimators is deliberately a single fixed ceiling here,
        # NOT swept like random_forest's - now that _build_xgboost's
        # early stopping picks the actual number of rounds
        # automatically per combination of the OTHER hyperparameters
        # (see _EarlyStoppingXGBClassifier). This is NOT the same
        # situation as random_forest's n_estimators sweep just above:
        # random forest's trees are independent (bagging) and adding
        # more of them essentially only reduces variance - there's no
        # peak to overshoot, so sweeping discrete values to find
        # diminishing returns is the right, and only sensible, way to
        # treat it. XGBoost's trees are additive (boosting) - each
        # round fits the errors of the previous ones, and it CAN
        # overshoot a peak the way the training-rounds loss curve
        # shows. A shared discrete grid of guesses (previously
        # [100, 200, 300, 400, 500], same shape as random_forest's)
        # can only ever land near the right stopping point for
        # whichever combinations happen to peak close to one of those
        # 5 values - every other combination gets a systematically
        # wrong answer. Early stopping finds each combination's own
        # answer via a validation slice instead of guessing one
        # shared list for all of them.
        #
        # 500 here is a MEASURED ceiling, not a guess - checked
        # directly against the real 402,873-row dataset before
        # picking a number: at max_depth=5, learning_rate=0.1, each
        # round costs ~1.16s, and validation mlogloss was STILL
        # improving at round 400 (0.877, down from 1.203 at round 0),
        # nowhere near the 20-round-no-improvement patience needed to
        # stop early - and even a fast-looking combination
        # (max_depth=3, learning_rate=0.3) used 496 of 500 rounds.
        # In practice this means most combinations use close to the
        # full ceiling rather than stopping cleanly early on this
        # dataset - so don't expect this to make individual fits
        # fast. What it still buys you: the reported model for every
        # combination is always the best-validated checkpoint up to
        # that ceiling (confirmed - XGBoost's predict() uses
        # get_booster().best_iteration, not whichever round training
        # happened to end on), and the grid shrinks 5x (this axis
        # dropping from 5 values to 1) since n_estimators is no
        # longer combined multiplicatively with every other
        # hyperparameter. Raise this if you want closer to genuine
        # convergence for slow combinations at the cost of more time;
        # lower early_stopping_rounds' patience (currently 20, set in
        # _build_xgboost) if you'd rather stop more aggressively on
        # small improvements instead of raising the ceiling.
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [3, 4, 5, 6, 7, 9],
        "learning_rate": [0.01, 0.03, 0.1, 0.2, 0.3],
        # Not swept before - row/column subsampling per tree, the
        # two standard XGBoost regularization knobs beyond depth.
        # 1.0 = no subsampling (original behavior still included).
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
}

# The transformer lives in its own module (tabular_transformer.py)
# specifically so torch is an optional dependency - everything
# above works fine without it installed. Its MODEL_BUILDERS entry
# is NOT registered here (unlike the three above) because building
# it requires the actual dataset (to size embedding tables to
# every category's cardinality) - see
# tabular_transformer.make_transformer_builder()'s docstring for
# how to register it before use.
try:
    from tabular_transformer import TRANSFORMER_PARAM_GRID
    DEFAULT_PARAM_GRIDS["transformer"] = TRANSFORMER_PARAM_GRID
except ImportError:
    pass


# ==========================================================
# Splitting
# ==========================================================


def _make_split(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float,
    random_state: int,
    groups: np.ndarray | None,
):
    """
    Produces one train/test split.

    If `groups` is provided (e.g. a match/demo identifier), uses
    GroupShuffleSplit so that no group's rows are split across both
    train and test - important for this dataset, where multiple
    engagements from the same match/round could otherwise leak
    information across the split. Falls back to a plain random
    split if no groups are given.
    """

    if groups is not None:

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )

        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

        return X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# ==========================================================
# Single-Split Evaluation
# ==========================================================


def _extract_native_importance(
    fitted_model: Any,
    feature_names: list[str],
) -> np.ndarray:
    """
    Pulls out whichever "free" importance signal a fitted model
    already computed as part of training - no extra cost, but not
    strictly comparable across model types (coefficients vs. Gini
    vs. gain are different quantities). See
    run_feature_importance_analysis() for a proper, uniform
    permutation-importance alternative.
    """

    if isinstance(fitted_model, Pipeline):
        estimator = fitted_model.named_steps["clf"]
    else:
        estimator = fitted_model

    if hasattr(estimator, "coef_"):
        return np.abs(estimator.coef_[0])

    if hasattr(estimator, "feature_importances_"):
        return estimator.feature_importances_

    return np.full(len(feature_names), np.nan)


def _compute_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    all_classes: list[Any],
) -> dict[str, float]:
    """
    Computes roc_auc/f1/precision/recall/accuracy for one split,
    handling both binary (2 classes) and multiclass (3+) targets.

    all_classes is the FULL set of classes present in y overall
    (computed once from the whole dataset, not per-split - see
    callers) - passed explicitly to f1/precision/recall so every
    split's macro average is computed over the same class set,
    rather than silently varying if a split happens to be missing
    a class from y_test or y_pred.

    Multiclass roc_auc (one-vs-rest, macro) is NOT well-defined
    when a class in all_classes has zero examples in this split's
    y_test - sklearn raises ValueError in that case. Rather than
    letting that crash the whole grid sweep (which is what
    happens with a thin class like a 1-row mutual_kill: nearly
    every split will hit this), that failure is caught here and
    roc_auc is recorded as NaN for this split only. pandas' mean()/
    std() in run_grid_sweep skip NaN automatically, so a
    hyperparameter combination's roc_auc_mean ends up averaged over
    however many splits actually had a usable test fold - see
    run_grid_sweep's printed valid_roc_auc_splits count. This is a
    symptom, not a fix - if most splits are getting skipped, that's
    the thin-class problem feature_engineering.py already warns
    about, surfacing here as "the metric this run is optimizing
    for is only defined some of the time."
    """

    result: dict[str, float] = {
        "accuracy": accuracy_score(y_test, y_pred),
    }

    if len(all_classes) == 2:

        result["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
        result["f1"] = f1_score(y_test, y_pred, zero_division=0)
        result["precision"] = precision_score(y_test, y_pred, zero_division=0)
        result["recall"] = recall_score(y_test, y_pred, zero_division=0)

        return result

    # Multiclass: labels= pins every split's macro average to the
    # same full class set, so combinations stay comparable across
    # splits even when a particular split's test fold happens to
    # be missing a rare class.
    result["f1"] = f1_score(
        y_test, y_pred, labels=all_classes, average="macro", zero_division=0
    )
    result["precision"] = precision_score(
        y_test, y_pred, labels=all_classes, average="macro", zero_division=0
    )
    result["recall"] = recall_score(
        y_test, y_pred, labels=all_classes, average="macro", zero_division=0
    )

    try:
        result["roc_auc"] = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro", labels=all_classes
        )
    except ValueError:
        result["roc_auc"] = np.nan

    return result


def _evaluate_one_split(
    model_builder: Callable[[dict[str, Any], int], Any],
    params: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    all_classes: list[Any],
    seed: int,
) -> dict[str, Any]:
    """
    Takes the already-resolved builder callable (not a model_name
    string looked up against MODEL_BUILDERS internally) so this
    function is a pure, self-contained unit of work: everything it
    needs is in its arguments, nothing implicit from module-level
    global state. That matters beyond tidiness once Ray is
    involved - a remote worker process has its own separate copy
    of MODEL_BUILDERS, so a builder registered dynamically on the
    driver (e.g. tabular_transformer.make_transformer_builder(X),
    assigned into MODEL_BUILDERS at runtime rather than defined at
    import time like the other three) would silently not exist on
    the worker if this function looked it up by name instead of
    receiving the actual callable directly.
    """

    model = model_builder(params, seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    result = _compute_metrics(y_test, y_pred, y_proba, all_classes)

    importance = _extract_native_importance(model, feature_names)

    for name, value in zip(feature_names, importance):
        result[f"importance__{name}"] = value

    return result


# ==========================================================
# Grid Sweep
# ==========================================================


# ==========================================================
# Checkpointing
#
# A full sweep is n_repeats x every hyperparameter combination -
# for 100 repeats and even a modest grid, that's thousands of
# fits and can run long enough to get interrupted (crash, time
# limit, closing the laptop). Rather than losing everything and
# restarting from combination 1, each combination's result is
# written to a JSONL checkpoint file the moment it finishes, and
# a resumed run skips straight past whatever's already there.
# ==========================================================


def _json_default(obj: Any) -> Any:
    """
    json.dumps default= handler for the numpy scalar types that
    show up in a results row (np.float64 from .mean()/.std(),
    np.int64 from a hyperparameter, etc.) - plain json can't
    serialize these on its own.
    """

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _fingerprint_dataset(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None,
) -> str:
    """
    Cheap fingerprint identifying the exact data a checkpoint was
    computed against - shape, column names, class set, and a
    sample of actual values (not just shape, so two same-shaped
    but genuinely different datasets - e.g. the 9-match sample
    versus a later full-corpus rebuild with an identical column
    schema - don't collide). Checked before trusting any
    checkpoint: reusing a saved result against different data
    than it was computed on would silently produce numbers that
    look valid but aren't, which is worse than just being slow.
    """

    hasher = hashlib.sha256()
    hasher.update(str(X.shape).encode())
    hasher.update(",".join(X.columns).encode())
    hasher.update(str(sorted(np.unique(y).tolist())).encode())

    sample_n = min(50, len(X))
    hasher.update(
        pd.util.hash_pandas_object(X.head(sample_n)).values.tobytes()
    )
    hasher.update(
        pd.util.hash_pandas_object(pd.Series(y[:sample_n])).values.tobytes()
    )

    if groups is not None:
        hasher.update(str(sorted(set(map(str, groups)))).encode())

    return hasher.hexdigest()[:16]


def _load_checkpoint(
    checkpoint_path: Path,
    fingerprint: str,
) -> dict[str, dict[str, Any]]:
    """
    Loads a JSONL sweep checkpoint, keyed by each combination's
    hyperparameters (as a canonical sorted JSON string). Returns
    {} - i.e. start fresh - if the file doesn't exist, is
    unreadable, or was computed against different data than the
    current call (fingerprint mismatch, see _fingerprint_dataset).
    A mismatched file is renamed aside with a timestamp rather
    than deleted, so nothing is silently lost.
    """

    if not checkpoint_path.exists():
        return {}

    completed: dict[str, dict[str, Any]] = {}
    stored_fingerprint = None

    try:

        with open(checkpoint_path, "r") as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                entry = json.loads(line)

                if "__fingerprint__" in entry:
                    stored_fingerprint = entry["__fingerprint__"]
                    continue

                key = json.dumps(entry["params"], sort_keys=True, default=str)
                completed[key] = entry

    except (json.JSONDecodeError, KeyError) as error:

        print(
            f"[checkpoint] {checkpoint_path} is corrupted ({error}) - "
            f"starting fresh rather than trusting a partially-written file"
        )
        return {}

    if stored_fingerprint != fingerprint:

        backup_path = checkpoint_path.with_name(
            checkpoint_path.stem + f".stale-{int(time.time())}.jsonl"
        )
        checkpoint_path.rename(backup_path)

        print(
            f"[checkpoint] {checkpoint_path.name} was computed against "
            f"different data (fingerprint mismatch) - ignoring it and "
            f"starting fresh rather than reusing stale results. Old "
            f"file preserved at {backup_path.name}"
        )
        return {}

    return completed


def _append_checkpoint(
    checkpoint_path: Path,
    params: dict[str, Any],
    row: dict[str, Any],
    fingerprint: str,
    write_header: bool,
) -> None:
    """
    Appends one completed combination to the JSONL checkpoint,
    flushing and fsyncing so the write survives a crash
    immediately after - the whole point of checkpointing is
    surviving unexpected interruption, so a buffered write that's
    still sitting in an OS cache when the process dies defeats it.
    """

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    with open(checkpoint_path, "a") as file:

        if write_header:
            file.write(json.dumps({"__fingerprint__": fingerprint}) + "\n")

        file.write(
            json.dumps({"params": params, "row": row}, default=_json_default)
            + "\n"
        )
        file.flush()
        os.fsync(file.fileno())


# ==========================================================
# Distributed Execution (Ray)
#
# The unit of parallel work is one (hyperparameter combination,
# repeat) pair - each is fully independent given (X, y, groups,
# params, seed), which is exactly what makes this "embarrassingly
# parallel" and a good fit for Ray. _split_and_evaluate below
# combines _make_split + _evaluate_one_split into one function so
# a single Ray task does a full unit of work rather than needing
# two round-trips.
# ==========================================================


def _split_and_evaluate(
    model_builder: Callable[[dict[str, Any], int], Any],
    params: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None,
    test_size: float,
    seed: int,
    feature_names: list[str],
    all_classes: list[Any],
) -> dict[str, Any]:

    X_train, X_test, y_train, y_test = _make_split(X, y, test_size, seed, groups)

    return _evaluate_one_split(
        model_builder, params, X_train, X_test,
        y_train, y_test, feature_names, all_classes, seed,
    )


if RAY_AVAILABLE:
    _split_and_evaluate_remote = ray.remote(max_calls=1)(_split_and_evaluate)


def _run_distributed_sweep(
    model_builder: Callable[[dict[str, Any], int], Any],
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None,
    combinations: list[tuple[Any, ...]],
    param_names: list[str],
    completed: dict[str, dict[str, Any]],
    n_repeats: int,
    test_size: float,
    base_random_state: int,
    feature_names: list[str],
    all_classes: list[Any],
    checkpoint_path: Path | None,
    fingerprint: str,
    checkpoint_header_written: bool,
    max_concurrent_tasks: int,
    show_progress: bool,
    memory_per_task: int | None = None,
    num_gpu_combos: int = 0,
) -> tuple[list[dict[str, Any]], int, int]:
    """
    Ray-backed replacement for run_grid_sweep's sequential combo
    loop. Submits every (combination, repeat) pair not already in
    `completed` as a Ray task, keeps at most max_concurrent_tasks
    in flight at once (submitting a new one each time one
    finishes, rather than dumping the entire backlog on the
    cluster at once - a grid like the widened random_forest default
    is 600 combinations x up to 100 repeats = 60,000 tasks, and
    handing all of those to the scheduler simultaneously is real
    memory/scheduling pressure worth avoiding), and checkpoints
    each combination the moment all of its repeats have returned -
    same per-combination checkpoint granularity and file format as
    the sequential path, so a checkpoint written by one is resumable
    by the other and vice versa.

    Completion order is NOT combination order - with 3 machines of
    possibly different speeds, combination 47 can easily finish
    before combination 3. Progress is printed as combinations
    complete, and the final row list is re-sorted back into
    combinations' original order before returning, so downstream
    consumers (plot_marginal_effects, results_df row order) see the
    exact same shape as the sequential path regardless of what
    order the cluster happened to finish things in.

    X, y, and groups are ray.put() exactly once here (not per
    task) - every task references the same object-store copy, so
    the dataset is transferred to a given worker node at most once
    no matter how many tasks run there, rather than being
    re-serialized into every individual task's arguments.

    num_gpu_combos
        Routes the first num_gpu_combos combinations (by combination
        order, not submission/completion order) onto a GPU-exclusive
        lane instead of the default CPU-only lane. Concretely: every
        (combo, repeat) task belonging to one of those combinations
        is submitted with num_gpus=1 - so Ray only ever runs ONE of
        them at a time, on whichever node actually has a GPU, and
        queues the rest - while every other combination's tasks keep
        running num_gpus=0, spread across all remaining CPU capacity
        on every node (including the GPU node's own other cores) in
        parallel with whatever's currently running on the GPU.

        0 (default) requests no GPU anywhere - identical to the
        original behavior, fully backward compatible.

        This is deliberately NOT "give every task a fractional GPU"
        (e.g. num_gpus=0.5 to run two at once). With one physical
        GPU with limited VRAM, sharing it between concurrent tasks
        risks the exact CUDA OOM crash loop this was built to avoid
        in the first place (see TabularTransformerClassifier's
        device auto-detection and model_tuning's Ray docs) - the
        whole point of routing by combo, not by fraction, is that
        the GPU lane stays genuinely exclusive.

        Choosing how many combinations to route here is a real
        tradeoff, not something this function can decide for you:
        one physical GPU is exactly one lane, serialized, no matter
        how many combinations you assign it, while CPU-only tasks
        get roughly (node CPU count - 1) genuinely parallel lanes.
        The GPU lane only helps total sweep time if a GPU fit is
        fast enough per-combo to still beat what that combo would
        have cost sharing the CPU lanes with everything else -
        measure one epoch on each device for your actual hardware
        rather than assuming.

    memory_per_task
        Each task requests only {CPU: 1} by default in Ray, which
        means the scheduler packs tasks onto a node purely by CPU
        count - with zero awareness of how much RAM each one
        actually needs. That's fine for a narrow, ordinal-encoded
        frame, but for a wide one-hot encoding (hundreds of columns)
        it's a real problem: sklearn's Pipeline has to convert the
        whole frame to a dense float64 array before it can scale it
        (StandardScaler can't operate on the compact bool columns
        one-hot produces directly), and that conversion plus the
        scaled output are each a full n_rows x n_features float64
        array - independently, per task, since each worker
        deserializes its OWN copy from the shared object-store
        reference. A node with plenty of free CPUs but not enough
        RAM for that many simultaneous multi-GB tasks gets an OOM
        kill, not a slowdown - confirmed in practice (569 one-hot
        columns, ~3.7-4GB actual RSS per task, 6 tasks landing on
        one node's free CPU slots with no memory accounting at all).

        Defaults (None) to
        4 * X.shape[0] * X.shape[1] * 8 bytes - modeling the
        worst case (every column densifies to float64 during
        fitting, regardless of how compactly it's stored in the
        driver's own memory right now) with roughly one multiple
        each for the raw converted copy, the scaled/transformed
        output, and general overhead/safety margin. This is a
        heuristic, not a measurement of any specific model's actual
        peak usage - override it directly if you've profiled a
        particular model_name and know better, or if the default
        is still causing OOMs (raise it) or leaving the cluster
        under-parallelized (lower it).

        Pass 0 explicitly to skip requesting a memory resource for
        this sweep's tasks entirely - not "request 0 bytes", but
        "don't pass memory= to .options() at all", so Ray schedules
        purely on {CPU, GPU} the way it did before memory_per_task
        existed. This is a diagnostic escape hatch, not a normal
        setting: Ray's memory resource is accounted per-node, not
        from the cluster-wide total ray status reports, so a task
        can sit PENDING_NODE_ASSIGNMENT indefinitely - no error,
        no crash - even when aggregate CPU and memory both show
        idle. If tasks stuck that way schedule immediately once
        their memory_per_task is set to 0, that confirms the memory
        resource specifically was the blocker (and the right fix is
        a more accurate byte value, not abandoning memory awareness
        for real runs) - if they're still stuck, memory was never
        the actual constraint and it's worth looking elsewhere
        (num_cpus, node availability, something else in what that
        task requests).
    """

    if memory_per_task is None:
        memory_per_task = 4 * X.shape[0] * X.shape[1] * 8

    X_ref = ray.put(X)
    y_ref = ray.put(y)
    groups_ref = ray.put(groups)

    # Flat backlog of every (combo_index, repeat) pair still
    # needing computation - combo order preserved for the final
    # re-sort, but repeats from different combinations are
    # naturally interleaved in submission since they're all drawn
    # from one shared pool under the concurrency cap, which is what
    # keeps the cluster saturated rather than starving on whichever
    # single combination happens to be in progress.
    pending_work: list[tuple[int, int, dict[str, Any], int]] = []

    combo_params: list[dict[str, Any]] = []
    combo_keys: list[str] = []
    already_done_rows: dict[int, dict[str, Any]] = {}

    for combo_index, combo_values in enumerate(combinations):

        params = dict(zip(param_names, combo_values))
        combo_params.append(params)

        combo_key = json.dumps(params, sort_keys=True, default=str)
        combo_keys.append(combo_key)

        if combo_key in completed:
            already_done_rows[combo_index] = completed[combo_key]["row"]
            continue

        for repeat in range(n_repeats):
            seed = base_random_state + repeat
            pending_work.append((combo_index, repeat, params, seed))

    total_combos = len(combinations)
    n_resumed = len(already_done_rows)

    if show_progress:
        gpu_note = (
            f" - first {min(num_gpu_combos, total_combos)}/{total_combos} "
            f"combination(s) routed to the exclusive GPU lane, the rest "
            f"CPU-only"
            if num_gpu_combos > 0
            else ""
        )
        # memory_per_task is resolved (None -> heuristic) above, so by
        # this point it's always an int - 0 means the diagnostic
        # "no memory resource requested" path (see its docstring),
        # anything else is the actual per-task byte request Ray will
        # schedule against.
        memory_note = (
            " - no memory resource requested per task (diagnostic mode)"
            if memory_per_task == 0
            else f" - {memory_per_task / 1e9:.2f}GB memory requested per task"
        )
        print(
            f"[{model_name}] Distributed sweep: {total_combos} "
            f"combinations, {n_resumed} already complete (resumed), "
            f"{len(pending_work)} split-tasks to run across the "
            f"cluster (max {max_concurrent_tasks} concurrent){gpu_note}"
            f"{memory_note}."
        )

    # Per-combo accumulation state
    split_results_by_combo: dict[int, list[dict[str, Any]]] = {
        i: [] for i in range(total_combos) if i not in already_done_rows
    }
    remaining_by_combo: dict[int, int] = {
        i: n_repeats for i in range(total_combos) if i not in already_done_rows
    }

    finished_rows: dict[int, dict[str, Any]] = dict(already_done_rows)
    n_computed = 0

    future_to_combo: dict[Any, int] = {}
    work_iter = iter(pending_work)

    def _submit_next() -> Any | None:
        try:
            combo_index, repeat, params, seed = next(work_iter)
        except StopIteration:
            return None

        # See num_gpu_combos' docstring above: the first
        # num_gpu_combos combinations (by combination order - stable
        # regardless of submission/completion order) get num_gpus=1,
        # so Ray runs at most one of them at a time on the GPU node
        # and queues the rest; every other combination stays
        # num_gpus=0 and is free to run on any node's spare CPU
        # capacity, including that same GPU node's other cores, in
        # parallel with whatever the GPU lane is currently doing.
        use_gpu = combo_index < num_gpu_combos

        # See memory_per_task's docstring above for the 0 sentinel:
        # build task_options without "memory" at all in that case,
        # rather than passing memory=0 - the point is testing
        # whether Ray's per-node memory-resource placement is what's
        # blocking scheduling, which means removing the request
        # entirely, not asking for zero of it (which may not be the
        # same thing to Ray's scheduler).
        task_options: dict[str, Any] = {"num_gpus": 1 if use_gpu else 0}
        if memory_per_task > 0:
            task_options["memory"] = memory_per_task

        if num_gpu_combos > 0:
            # Confirmed in practice, not just theoretical: relying on
            # TabularTransformerClassifier's own
            # torch.cuda.is_available() auto-detection let CPU-lane
            # tasks (num_gpus=0) end up running on device=cuda anyway
            # - Ray's per-task CUDA-visibility environment doesn't
            # reliably reset when a worker process gets reused across
            # tasks with a DIFFERENT num_gpus shape (the exact same
            # class of gotcha num_threads already had to work around
            # by being reasserted every fit()/predict_proba() call,
            # rather than trusted from process start). Four separate
            # CPU-lane tasks ended up CUDA-visible this way, piling
            # onto a GPU already down to ~1.6GB free because of an
            # unrelated llama-server process permanently using
            # ~2.3GB of it - straightforward OOM cascade, not a
            # scheduling problem. Only fires when this sweep actually
            # uses GPU routing (num_gpu_combos > 0), so it's a no-op
            # for every other model_name's params, which don't accept
            # a "device" kwarg at all.
            params = {**params, "device": "cuda" if use_gpu else "cpu"}

        future = _split_and_evaluate_remote.options(**task_options).remote(
            model_builder, params, X_ref, y_ref, groups_ref,
            test_size, seed, feature_names, all_classes,
        )
        future_to_combo[future] = combo_index
        return future

    in_flight: list[Any] = []
    for _ in range(min(max_concurrent_tasks, len(pending_work))):
        future = _submit_next()
        if future is not None:
            in_flight.append(future)

    while in_flight:

        done, in_flight = ray.wait(in_flight, num_returns=1)
        future = done[0]
        combo_index = future_to_combo.pop(future)

        try:
            split_result = ray.get(future)
            split_results_by_combo[combo_index].append(split_result)
        except Exception as error:
            print(
                f"[{model_name}] WARNING: a split task for combination "
                f"{combo_index} ({combo_params[combo_index]}) failed: "
                f"{error}. Continuing without it - that combination's "
                f"averages will be over fewer than {n_repeats} repeats."
            )

        remaining_by_combo[combo_index] -= 1

        next_future = _submit_next()
        if next_future is not None:
            in_flight.append(next_future)

        if remaining_by_combo[combo_index] == 0 and combo_index not in finished_rows:

            split_df = pd.DataFrame(split_results_by_combo[combo_index])
            params = combo_params[combo_index]

            row = dict(params)
            row["n_valid_roc_auc_splits"] = int(split_df["roc_auc"].notna().sum()) if len(split_df) else 0

            for col in split_df.columns:
                row[f"{col}_mean"] = split_df[col].mean()
                row[f"{col}_std"] = split_df[col].std()

            finished_rows[combo_index] = row
            n_computed += 1

            if checkpoint_path is not None:
                _append_checkpoint(
                    checkpoint_path, params, row, fingerprint,
                    write_header=not checkpoint_header_written,
                )
                checkpoint_header_written = True

            if show_progress:
                n_done_total = len(finished_rows)
                print(
                    f"  [{n_done_total}/{total_combos}] {params} "
                    f"-> roc_auc={row.get('roc_auc_mean', float('nan')):.4f} "
                    f"(combination complete)"
                )

    ordered_rows = [finished_rows[i] for i in range(total_combos)]

    return ordered_rows, n_resumed, n_computed


def run_grid_sweep(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    param_grid: dict[str, list[Any]] | None = None,
    n_repeats: int = 100,
    test_size: float = 0.2,
    groups: np.ndarray | None = None,
    base_random_state: int = 42,
    show_progress: bool = True,
    output_dir: Path | str | None = None,
    resume: bool = True,
    use_ray: bool = False,
    ray_address: str | None = None,
    max_concurrent_tasks: int = 64,
    memory_per_task: int | None = None,
    ray_py_modules: list[str] | None = None,
    num_gpu_combos: int = 0,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Runs the full hyperparameter grid, n_repeats random splits per
    combination, and returns:

        (results_df, param_dicts)

    results_df has one row per combination with metrics and feature
    importances averaged (and std-deviated) across the repeats.

    param_dicts is a list of the exact original hyperparameter
    dicts, in the same order as results_df's rows. Use this - not
    results_df's own columns - to recover a specific combination's
    hyperparameters afterwards: pandas silently upcasts a mixed
    int/float/None row to a single dtype when read back out (e.g.
    an integer n_estimators becomes np.float64, and a None max_depth
    becomes NaN), which then fails strict sklearn parameter
    validation. param_dicts preserves the original Python types
    exactly, since it's never round-tripped through the DataFrame.

    Parameters
    ----------
    model_name
        Key into MODEL_BUILDERS ("logistic_regression",
        "random_forest", "xgboost").
    X, y
        Already-encoded feature matrix and target (binary or
        multiclass - detected automatically from y). This
        function does not do any encoding itself - categorical
        columns must already be numeric.
    param_grid
        Dict of param_name -> list of values to sweep. Defaults to
        DEFAULT_PARAM_GRIDS[model_name] if not given.
    n_repeats
        Number of random train/test splits averaged per
        hyperparameter combination.
    groups
        Optional array-like of the same length as X, giving a
        match/demo identifier per row, to keep each match's rows
        together on one side of the split. Strongly recommended for
        this dataset - see _make_split().
    output_dir
        If given, each hyperparameter combination's result is
        checkpointed to
        <output_dir>/<model_name>_sweep_checkpoint.jsonl the moment
        it finishes. If this function is interrupted and called
        again with the same output_dir, already-completed
        combinations are loaded from the checkpoint instead of
        being recomputed. None (default) disables checkpointing -
        everything stays in memory only, matching the original
        behavior.
    resume
        If True (default) and a checkpoint exists at output_dir,
        reuse whatever combinations it already has. If False,
        ignores any existing checkpoint and recomputes everything
        from scratch (still overwrites the checkpoint file with
        fresh results as it goes, so a subsequent resumed run
        would pick up from this one).
    use_ray
        If True, distributes (combination, repeat) pairs across a
        Ray cluster instead of running them in a sequential local
        loop. Requires ray to be installed. Connects to an existing
        cluster if ray_address is given, otherwise starts (or
        reuses) a local Ray instance - useful for testing this on
        one machine before pointing it at a real cluster. Per-
        combination checkpointing works identically either way
        (same file format), so a run can be started sequentially
        and resumed distributed, or vice versa.
    ray_address
        Passed to ray.init(address=...). None connects to/starts a
        local Ray instance. For an actual cluster, this is normally
        "auto" (if this process is launched via `ray submit` /
        already on a cluster node) or the head node's address
        (e.g. "ray://<head-node-ip>:10001").
    max_concurrent_tasks
        Caps how many (combination, repeat) tasks are in flight on
        the cluster at once. Only used when use_ray=True. Prevents
        handing the entire grid's worth of tasks to Ray's scheduler
        at once on a large sweep (the widened random_forest default
        grid is 600 combinations x up to 100 repeats = 60,000
        tasks) - a new task is submitted each time one completes,
        rather than all at once upfront. Tune this to roughly your
        cluster's total core count for good utilization without
        overwhelming the scheduler.

        NOTE: this caps task COUNT, not memory - it won't by itself
        prevent an OOM on a wide/one-hot-encoded X, since Ray's
        default scheduling only looks at CPU. See memory_per_task.
    memory_per_task
        Only used when use_ray=True - see _run_distributed_sweep's
        docstring for the full reasoning and the default formula.
        Tells Ray's scheduler how much RAM each task actually needs,
        so it naturally limits how many run concurrently on one node
        based on real available memory rather than just free CPU
        slots. Without this, a wide one-hot encoding (hundreds of
        columns) can OOM a node that has plenty of spare CPUs but
        not enough RAM for that many simultaneous multi-GB fits -
        confirmed in practice, not just a theoretical risk.
    ray_py_modules
        _split_and_evaluate is shipped to worker nodes BY REFERENCE
        (module name + function name, since it's a real top-level
        function in this file), not by value - so a worker resolving
        that reference has to be able to `import model_tuning`
        itself. If that worker's own copy of this file is missing,
        stale, or just not on its Python path for whatever reason
        (e.g. `ray start` was launched from a different working
        directory on that machine than where this file actually
        lives), the task fails with a confusing
        "ModuleNotFoundError: No module named 'model_tuning'" from
        inside Ray's deserialization internals, on the worker, not
        the driver - which looks nothing like an import error at
        first glance and is easy to mistake for a corrupted install.
        Confirmed happening in practice: this is not a theoretical
        edge case.

        Defaults (None) to this file's own path, plus
        tabular_transformer.py's path if it exists alongside it
        (needed if any of the models being tuned resolve to a
        closure referencing TabularTransformerClassifier) - passed
        as runtime_env={"py_modules": [...]} to ray.init(), which
        packages exactly these files and ships them to every worker
        for this job specifically, regardless of what each worker's
        own launch-time working directory or manually-copied files
        happen to be. This means a worker node no longer needs its
        own manually-synced copy of these two files at all - correct
        code always travels with the job now, not with whatever was
        last scp'd there. Pass an explicit list to ship additional
        local modules a custom model_builder might need.
    num_gpu_combos
        Only meaningful when use_ray=True and this cluster actually
        has a GPU node. Routes the first num_gpu_combos combinations
        onto an exclusive GPU lane (num_gpus=1, so Ray serializes
        them one at a time on the GPU node) while every other
        combination runs CPU-only (num_gpus=0) in parallel across
        the rest of the cluster's cores. 0 (default) requests no GPU
        anywhere - identical to the original behavior. See
        _run_distributed_sweep's docstring for the full reasoning,
        including why this is combo-exclusive rather than a
        fractional GPU share. Has no effect when use_ray=False - the
        sequential local path already auto-detects whatever device
        is available on that one machine, with no concurrency to
        arbitrate.
    """

    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"'{model_name}' is not in MODEL_BUILDERS. The three "
            f"sklearn/XGBoost models are registered automatically - "
            f"if this is 'transformer', register it first: "
            f"MODEL_BUILDERS['transformer'] = "
            f"tabular_transformer.make_transformer_builder(X)"
        )

    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRIDS[model_name]

    # Resolved once, here, rather than looked up by name inside
    # _evaluate_one_split - see that function's docstring for why
    # this matters once Ray is involved (a worker process has its
    # own separate MODEL_BUILDERS, so a dynamically-registered
    # builder like the transformer's needs to be shipped to workers
    # as an actual callable, not looked up by name on the worker
    # side where it wouldn't exist).
    model_builder = MODEL_BUILDERS[model_name]

    feature_names = list(X.columns)

    # Computed once from the FULL y, not per-split - see
    # _compute_metrics for why every split needs to agree on the
    # same class set.
    all_classes = sorted(np.unique(y))

    if len(all_classes) > 2 and show_progress:
        print(
            f"[{model_name}] Multiclass target detected "
            f"({len(all_classes)} classes: {all_classes}). "
            f"Using one-vs-rest macro ROC AUC."
        )

    param_names = list(param_grid.keys())
    value_lists = [param_grid[name] for name in param_names]

    # itertools.product enumerates in exactly the nested-loop order
    # requested: the last parameter varies fastest.
    combinations = list(itertools.product(*value_lists))

    # Built once, up front, identically regardless of use_ray - the
    # exact original hyperparameter dicts in combination order, see
    # this function's docstring for why callers should read
    # hyperparameters back from this rather than from results_df's
    # own columns.
    param_dicts: list[dict[str, Any]] = [
        dict(zip(param_names, combo_values)) for combo_values in combinations
    ]

    total_combos = len(combinations)
    total_fits = total_combos * n_repeats

    if show_progress:
        print(
            f"[{model_name}] {total_combos} hyperparameter combinations "
            f"x {n_repeats} repeats = {total_fits} total model fits"
        )

    # ------------------------------------------------------------
    # Checkpoint setup
    # ------------------------------------------------------------

    checkpoint_path = None
    fingerprint = ""
    completed: dict[str, dict[str, Any]] = {}
    checkpoint_header_written = True

    if output_dir is not None:

        checkpoint_path = Path(output_dir) / f"{model_name}_sweep_checkpoint.jsonl"
        fingerprint = _fingerprint_dataset(X, y, groups)

        if resume:
            completed = _load_checkpoint(checkpoint_path, fingerprint)

        # Re-checked after _load_checkpoint, which may have renamed a
        # stale (fingerprint-mismatched) file aside - if so, the path
        # no longer exists and this run's first write needs a fresh
        # header, same as a first-ever run.
        checkpoint_header_written = checkpoint_path.exists()

        if completed and show_progress:
            print(
                f"[{model_name}] Resuming from checkpoint: "
                f"{len(completed)} combination(s) already completed at "
                f"{checkpoint_path.name}"
            )

    # ------------------------------------------------------------
    # Distributed (Ray) execution
    # ------------------------------------------------------------

    if use_ray:

        if not RAY_AVAILABLE:
            raise ImportError(
                "use_ray=True but ray isn't installed. "
                "pip install \"ray[default]\""
            )

        if not ray.is_initialized():

            if ray_py_modules is None:

                # Ship this file itself, plus tabular_transformer.py
                # if it's sitting alongside it - see ray_py_modules'
                # docstring above for why this needs to happen at
                # all. Only checked/built once, the first time this
                # process actually connects to Ray.
                ray_py_modules = [str(Path(__file__).resolve())]

                sibling_transformer = (
                    Path(__file__).resolve().parent / "tabular_transformer.py"
                )

                if sibling_transformer.exists():
                    ray_py_modules.append(str(sibling_transformer))

            ray.init(
                address=ray_address,
                ignore_reinit_error=True,
                runtime_env={
                    "py_modules": ray_py_modules,
                    # Every task below is scheduled with num_cpus=1 (or
                    # explicitly num_cpus=1, num_gpus=0/1 - see
                    # num_gpu_combos). That's a scheduling WEIGHT, not an
                    # OS-level thread cap: nothing stops the process
                    # inside a "1 CPU" task from spawning as many
                    # OMP/MKL/OpenBLAS threads as the node has cores. With
                    # several such tasks landing on the same node - the
                    # whole point of combo-level parallelism - that's
                    # N tasks x (node's full core count) threads
                    # competing for that same core count: the identical
                    # oversubscription problem the driver-side
                    # OMP_NUM_THREADS fix solved for the local sequential
                    # path, just relocated to the workers, where nothing
                    # was setting it. This makes each worker use ~1
                    # thread internally, so N concurrent tasks actually
                    # use ~N cores instead of thrashing.
                    "env_vars": {
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                    },
                },
            )

        rows, n_resumed, n_computed = _run_distributed_sweep(
            model_builder=model_builder,
            model_name=model_name,
            X=X, y=y, groups=groups,
            combinations=combinations,
            param_names=param_names,
            completed=completed,
            n_repeats=n_repeats,
            test_size=test_size,
            base_random_state=base_random_state,
            feature_names=feature_names,
            all_classes=all_classes,
            checkpoint_path=checkpoint_path,
            fingerprint=fingerprint,
            checkpoint_header_written=checkpoint_header_written,
            max_concurrent_tasks=max_concurrent_tasks,
            show_progress=show_progress,
            memory_per_task=memory_per_task,
            num_gpu_combos=num_gpu_combos,
        )

        if checkpoint_path is not None and show_progress and (n_resumed or n_computed):
            print(
                f"[{model_name}] Sweep complete: {n_resumed} combination(s) "
                f"resumed from checkpoint, {n_computed} computed this run."
            )

        return pd.DataFrame(rows), param_dicts

    # ------------------------------------------------------------
    # Sequential (local, single-process) execution - original
    # behavior, unchanged.
    # ------------------------------------------------------------

    rows = []
    n_resumed = 0
    n_computed = 0

    for combo_index, combo_values in enumerate(combinations, start=1):

        params = param_dicts[combo_index - 1]

        combo_key = json.dumps(params, sort_keys=True, default=str)

        if combo_key in completed:

            row = completed[combo_key]["row"]
            rows.append(row)
            n_resumed += 1

            if show_progress:
                print(
                    f"  [{combo_index}/{total_combos}] {params} "
                    f"-> roc_auc={row['roc_auc_mean']:.4f} (resumed from checkpoint)"
                )

            continue

        combo_start = time.time()

        split_results = []

        for repeat in range(n_repeats):

            seed = base_random_state + repeat

            X_train, X_test, y_train, y_test = _make_split(
                X, y, test_size, seed, groups
            )

            split_result = _evaluate_one_split(
                model_builder, params, X_train, X_test,
                y_train, y_test, feature_names, all_classes, seed,
            )

            split_results.append(split_result)

        split_df = pd.DataFrame(split_results)

        row = dict(params)

        row["n_valid_roc_auc_splits"] = int(split_df["roc_auc"].notna().sum())

        for col in split_df.columns:
            row[f"{col}_mean"] = split_df[col].mean()
            row[f"{col}_std"] = split_df[col].std()

        rows.append(row)
        n_computed += 1

        if checkpoint_path is not None:

            _append_checkpoint(
                checkpoint_path, params, row, fingerprint,
                write_header=not checkpoint_header_written,
            )
            checkpoint_header_written = True

        if show_progress:

            elapsed = time.time() - combo_start

            valid_note = ""
            if row["n_valid_roc_auc_splits"] < n_repeats:
                valid_note = (
                    f" [roc_auc valid on {row['n_valid_roc_auc_splits']}"
                    f"/{n_repeats} splits - some test folds were "
                    f"missing a class]"
                )

            print(
                f"  [{combo_index}/{total_combos}] {params} "
                f"-> roc_auc={row['roc_auc_mean']:.4f} "
                f"({elapsed:.1f}s for {n_repeats} repeats)"
                f"{valid_note}"
            )

    if checkpoint_path is not None and show_progress and (n_resumed or n_computed):
        print(
            f"[{model_name}] Sweep complete: {n_resumed} combination(s) "
            f"resumed from checkpoint, {n_computed} computed this run."
        )

    return pd.DataFrame(rows), param_dicts


# ==========================================================
# Marginal Effect Plots
# ==========================================================


def plot_marginal_effects(
    results_df: pd.DataFrame,
    param_names: list[str],
    metric: str = "roc_auc_mean",
    output_dir: Path | str | None = None,
    model_name: str = "",
) -> None:
    """
    For each hyperparameter, plots that parameter's values against
    the metric, averaged over every combination of the OTHER
    hyperparameters - i.e. "holding this parameter fixed, how does
    performance change as I move to the next value, marginalizing
    over everything else." One figure per hyperparameter.
    """

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for param in param_names:

        grouped = (
            results_df
            .groupby(param)[metric]
            .agg(["mean", "std"])
            .reset_index()
        )

        # Non-numeric hyperparameter values (e.g. "l2", None) still
        # plot fine on a categorical x-axis.
        x_labels = grouped[param].astype(str)

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.errorbar(
            x_labels,
            grouped["mean"],
            yerr=grouped["std"],
            marker="o",
            capsize=4,
        )

        ax.set_xlabel(param)
        ax.set_ylabel(metric)
        ax.set_title(f"{model_name}: {metric} vs {param}")
        ax.grid(alpha=0.3)

        fig.tight_layout()

        if output_dir is not None:
            fig.savefig(output_dir / f"{model_name}_{param}_{metric}.png", dpi=150)
            plt.close(fig)
        else:
            plt.show()


# ==========================================================
# Feature Importance (proper, uniform method)
#
# Run separately from the grid sweep, on the best hyperparameter
# combination only, using a smaller number of repeats. Permutation
# importance is model-agnostic and directly comparable across model
# types, unlike the native coefficients/Gini/gain captured during
# the sweep - but it's expensive per call (it refits nothing, but
# it re-predicts the test set once per feature per repeat), so it's
# kept out of the main 100x-per-combo loop deliberately.
# ==========================================================


def run_feature_importance_analysis(
    model_name: str,
    best_params: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int = 20,
    test_size: float = 0.2,
    groups: np.ndarray | None = None,
    base_random_state: int = 42,
    permutation_repeats: int = 10,
    scoring: str | None = None,
) -> pd.DataFrame:
    """
    Fits the given model/hyperparameter combination across
    n_repeats random splits and computes proper permutation
    importance on the held-out test set each time, then averages.

    scoring
        Passed to sklearn's permutation_importance. Defaults to
        None, which auto-selects: "roc_auc" for binary targets,
        "f1_macro" for multiclass. f1_macro (not "roc_auc_ovr") is
        the multiclass default specifically because it uses
        zero_division=0 internally and stays defined even when a
        split's test fold is missing a class - "roc_auc_ovr" is
        not, and with a thin class (see feature_engineering.py's
        thin-class warning) can fail on every single repeat. Pass
        "roc_auc_ovr" explicitly if you specifically want ROC AUC
        and have already resolved the thin-class problem.

    Returns a DataFrame with one row per feature: mean and std
    importance across all repeats.
    """

    feature_names = list(X.columns)

    all_classes = sorted(np.unique(y))

    if scoring is None:
        scoring = "roc_auc" if len(all_classes) == 2 else "f1_macro"

    all_importances = []
    n_skipped = 0
    last_error: ValueError | None = None

    for repeat in range(n_repeats):

        seed = base_random_state + repeat

        X_train, X_test, y_train, y_test = _make_split(
            X, y, test_size, seed, groups
        )

        model = MODEL_BUILDERS[model_name](best_params, seed)
        model.fit(X_train, y_train)

        try:

            result = permutation_importance(
                model, X_test, y_test,
                n_repeats=permutation_repeats,
                random_state=seed,
                scoring=scoring,
            )

        except ValueError as error:

            # Usually: this split's y_test is missing a class that
            # roc_auc_ovr needs to be defined. Skip this repeat's
            # contribution rather than crashing the whole importance
            # analysis. BUT this except catches ANY ValueError, so
            # the real message is printed and kept - a different
            # ValueError (e.g. non-finite probabilities from the
            # model) used to be reported as "missing a class" and
            # cost hours of debugging.
            n_skipped += 1
            last_error = error
            print(
                f"[{model_name}] permutation importance repeat {repeat} "
                f"(seed {seed}) raised ValueError: {error}"
            )
            continue

        all_importances.append(result.importances_mean)

    if n_skipped:
        print(
            f"[{model_name}] feature importance: skipped {n_skipped}"
            f"/{n_repeats} repeats (test fold missing a class - see "
            f"_compute_metrics docstring)"
        )

    if not all_importances:
        raise RuntimeError(
            f"[{model_name}] every repeat's test fold was missing a "
            f"class - permutation importance couldn't be computed at "
            f"all. This means at least one class is too thin to "
            f"reliably land in a {test_size:.0%} test split across "
            f"{n_repeats} repeats; resolve that class (drop/merge) "
            f"in feature_engineering.py before re-running. "
            f"The actual last error was: {last_error!r} - if that is "
            f"not about the number of classes, it is the real cause."
        )

    importance_matrix = np.array(all_importances)

    return pd.DataFrame({
        "feature": feature_names,
        "importance_mean": importance_matrix.mean(axis=0),
        "importance_std": importance_matrix.std(axis=0),
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    output_dir: Path | str | None = None,
    model_name: str = "",
) -> None:
    """
    Horizontal bar chart of the top_n most important features,
    with error bars from importance_std.
    """

    top = importance_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * top_n)))

    ax.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
        capsize=3,
    )

    ax.set_xlabel("Permutation importance (ROC-AUC drop)")
    ax.set_title(f"{model_name}: top {top_n} features")

    fig.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{model_name}_feature_importance.png", dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ==========================================================
# Train / Validation / Test (held-out) split
#
# Everything above this point (run_grid_sweep, run_full_tuning,
# run_all_models, run_all_models_from_csvs) works on a single X/y/
# groups and repeatedly re-splits it internally to select a best
# hyperparameter combination - see run_grid_sweep's docstring and
# _make_split. That internal repeated split is doing a VALIDATION
# job: it's what picks the winner among many candidates. But the
# score it reports for that winner is measured on the same kind of
# split used to pick it - across many combinations, the one that
# looks best is, on average, flattered by whichever random splits
# happened to favor it. That's not a bug in the sweep itself, it's
# an unavoidable property of using held-out performance to choose
# among candidates (see Cawley & Talbot, 2010, JMLR - the same
# citation feature_elimination.py already uses for the equivalent
# problem one level up, at the feature-selection stage).
#
# The fix is the standard one: a THIRD partition that the sweep
# never sees at all, held out before any hyperparameter is even
# considered, touched exactly once - after a winner is chosen - to
# report a number that hasn't been used to pick anything.
#
# split_train_val_test() does that split once, up front.
# evaluate_held_out_test() does the one-time final evaluation.
# Everything in between - the actual hyperparameter sweep - is
# still just run_full_tuning()/run_all_models_from_csvs(), fed the
# smaller train+validation pool this produces, completely unchanged
# from everything already tested above. No changes to that
# machinery were needed to do this correctly.
# ==========================================================


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    groups: np.ndarray,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    One-time, group-aware three-way split. Done ONCE, here, before
    any hyperparameter sweep starts - not re-drawn per repeat, since
    the entire point of the test partition is that it stays exactly
    the same, untouched set of matches from here until
    evaluate_held_out_test() is called at the very end.

    Group-aware in the same sense as _make_split: split on match_id
    (groups), not on raw rows, so no match's engagements end up
    split across two partitions - otherwise the model could see
    other rounds from the same match during training and be
    evaluated on rounds from that same match during "held-out"
    evaluation, which isn't really held out at all.

    Parameters
    ----------
    train_size, val_size, test_size
        Must sum to 1.0. Defaults are 0.6/0.2/0.2.

    random_state
        Fixes which matches land in which partition. Reusing the
        same value across repeated calls (e.g. a resumed session)
        reproduces the identical three partitions - changing it
        produces a genuinely different split, not a bug.

    Returns
    -------
    dict with:
        X_trainval, y_trainval, groups_trainval
            The train_size + val_size portion (e.g. 80% of the
            original data) - pass these straight in as the X/y/
            groups arguments to run_full_tuning() / 
            run_all_models_from_csvs(). The sweep's own internal
            repeated splitting (see run_grid_sweep) operates WITHIN
            this pool, which is exactly what makes it a validation
            mechanism rather than a leak: every one of those
            internal splits is still choosing among candidates using
            only data from this 80%, never touching the held-out
            20% below.
        X_test, y_test
            The held-out test_size portion (e.g. 20%). Do not fit,
            score, or otherwise inspect anything using this until
            you have a single, final chosen model - pass it to
            evaluate_held_out_test() exactly once, at the end.
        sweep_test_size
            Pass this as run_full_tuning's own `test_size` argument.
            Needed because run_full_tuning's internal splits operate
            on X_trainval (already train_size + val_size of the
            original), not on the original X - so achieving exactly
            train_size/val_size of the ORIGINAL data requires
            rescaling: val_size / (train_size + val_size), not
            val_size itself. Using val_size directly here would
            silently give you a smaller effective validation
            fraction of the original data than you asked for.
    """

    total = train_size + val_size + test_size

    if not np.isclose(total, 1.0):
        raise ValueError(
            f"train_size + val_size + test_size must sum to 1.0, "
            f"got {train_size} + {val_size} + {test_size} = {total}"
        )

    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    groups_arr = np.asarray(groups)

    # Split 1: carve off the held-out test partition first, by group.
    test_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    trainval_idx, test_idx = next(
        test_splitter.split(X, y_arr, groups=groups_arr)
    )

    X_trainval = X.iloc[trainval_idx].reset_index(drop=True)
    y_trainval = y_arr[trainval_idx]
    groups_trainval = groups_arr[trainval_idx]

    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y_arr[test_idx]

    n_trainval_groups = len(set(groups_trainval))
    n_test_groups = len(set(groups_arr[test_idx]))

    print(
        f"[split_train_val_test] {len(X)} rows, "
        f"{len(set(groups_arr))} groups (matches) -> "
        f"train+val: {len(X_trainval)} rows / {n_trainval_groups} groups, "
        f"test: {len(X_test)} rows / {n_test_groups} groups"
    )

    overlap = set(groups_trainval) & set(groups_arr[test_idx])
    if overlap:
        # Should be structurally impossible given GroupShuffleSplit,
        # but this is exactly the kind of silent leak that's bitten
        # this project before elsewhere - checked explicitly rather
        # than assumed.
        raise RuntimeError(
            f"{len(overlap)} group(s) leaked across the train+val/"
            f"test split - this should never happen with "
            f"GroupShuffleSplit and indicates a real bug, not a "
            f"data issue. Groups: {sorted(overlap)[:10]}"
        )

    sweep_test_size = val_size / (train_size + val_size)

    return {
        "X_trainval": X_trainval,
        "y_trainval": y_trainval,
        "groups_trainval": groups_trainval,
        "X_test": X_test,
        "y_test": y_test,
        "sweep_test_size": sweep_test_size,
    }


def evaluate_held_out_test(
    model_name: str,
    best_params: dict[str, Any],
    X_trainval: pd.DataFrame,
    y_trainval: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    output_dir: Path | str | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    The one-time final step: refits model_name with best_params on
    the FULL train+validation pool (X_trainval/y_trainval - i.e.
    everything split_train_val_test() didn't hold out), then scores
    it EXACTLY ONCE against X_test/y_test.

    Refits on train+validation COMBINED, not on the original
    train_size-only portion, deliberately: validation's job was to
    help choose best_params, and that job is done by the time this
    is called - there's no remaining reason to withhold that 20%
    from the final model, and using more data for the model you're
    actually going to report on on is strictly better than not.
    If you specifically want the final model trained on ONLY the
    original train_size portion for comparability with the sweep's
    own per-repeat fits, split X_trainval/y_trainval yourselves
    first with the same sweep_test_size before calling this.

    X_test/y_test must be the untouched output of
    split_train_val_test() - never previously fit on, predicted on,
    or otherwise inspected. Calling this more than once for the
    same model, e.g. after going back and tweaking best_params by
    hand because this first result was disappointing, defeats the
    entire purpose - at that point you are using the test set to
    make a decision, exactly the thing this function exists to
    prevent. Treat this as a one-shot report, not a diagnostic to
    iterate against.

    Returns
    -------
    dict with model_name, best_params, n_train, n_test, and every
    metric _compute_metrics produces (accuracy, f1, precision,
    recall, roc_auc - all computed on X_test/y_test only). Also
    written to <output_dir>/<model_name>_held_out_test_result.json
    if output_dir is given.
    """

    y_trainval = (
        y_trainval.to_numpy() if hasattr(y_trainval, "to_numpy")
        else np.asarray(y_trainval)
    )
    y_test = (
        y_test.to_numpy() if hasattr(y_test, "to_numpy")
        else np.asarray(y_test)
    )

    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"model_name {model_name!r} not in MODEL_BUILDERS "
            f"({list(MODEL_BUILDERS.keys())}) - for 'transformer', "
            f"register it first: "
            f"MODEL_BUILDERS['transformer'] = "
            f"make_transformer_builder(X_trainval)"
        )

    model = MODEL_BUILDERS[model_name](best_params, random_state)
    model.fit(X_trainval, y_trainval)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    all_classes = np.unique(np.concatenate([y_trainval, y_test]))

    metrics = _compute_metrics(y_test, y_pred, y_proba, all_classes)

    # Per-class breakdown, confusion matrix and the raw predictions,
    # saved alongside the macro metrics. This function is a one-shot
    # (see docstring), so anything the report might need later - a
    # per-class table, a confusion-matrix figure, ROC/calibration
    # curves - has to be captured on this single call rather than
    # by refitting afterwards. Class labels are the integer codes
    # from feature_engineering.build_label (alphabetical order of
    # engagement_status).
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    class_labels = [int(c) for c in all_classes]

    p_c, r_c, f_c, s_c = precision_recall_fscore_support(
        y_test, y_pred, labels=class_labels, zero_division=0
    )

    per_class = {
        str(label): {
            "precision": float(p_c[i]),
            "recall": float(r_c[i]),
            "f1": float(f_c[i]),
            "support": int(s_c[i]),
        }
        for i, label in enumerate(class_labels)
    }

    cm = confusion_matrix(y_test, y_pred, labels=class_labels)

    result = {
        "model_name": model_name,
        "best_params": best_params,
        "n_train": int(len(X_trainval)),
        "n_test": int(len(X_test)),
        **metrics,
        "class_labels": class_labels,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }

    print(f"\n[{model_name}] HELD-OUT TEST (never used for selection):")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")

    if output_dir is not None:

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result_path = output_dir / f"{model_name}_held_out_test_result.json"

        with open(result_path, "w") as file:
            json.dump(result, file, indent=2, default=_json_default)

        predictions_path = output_dir / f"{model_name}_held_out_predictions.npz"

        np.savez_compressed(
            predictions_path,
            y_test=y_test,
            y_pred=np.asarray(y_pred),
            y_proba=np.asarray(y_proba),
            class_labels=np.asarray(class_labels),
        )

        print(f"[{model_name}] wrote {result_path}")
        print(f"[{model_name}] wrote {predictions_path}")

    return result


# ==========================================================
# Orchestrator
# ==========================================================


def run_full_tuning(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    output_dir: Path | str,
    param_grid: dict[str, list[Any]] | None = None,
    n_repeats: int = 100,
    importance_n_repeats: int = 20,
    test_size: float = 0.2,
    groups: np.ndarray | None = None,
    base_random_state: int = 42,
    primary_metric: str = "roc_auc_mean",
    importance_scoring: str | None = None,
    resume: bool = True,
    use_ray: bool = False,
    ray_address: str | None = None,
    max_concurrent_tasks: int = 64,
    permutation_repeats: int = 10,
    memory_per_task: int | None = None,
    ray_py_modules: list[str] | None = None,
    num_gpu_combos: int = 0,
) -> dict[str, Any]:
    """
    End-to-end: grid sweep -> marginal effect plots -> best
    combination -> feature importance -> feature importance plot.
    Saves everything to output_dir. Returns a summary dict.

    use_ray, ray_address, max_concurrent_tasks
        Passed straight through to run_grid_sweep - see its
        docstring for what each does. Only the grid sweep itself
        runs distributed; the feature-importance pass afterward
        (importance_n_repeats) always runs locally on whichever
        machine calls this function, since it's one fit on the
        already-chosen best combination, not thousands of
        independent fits - not worth the Ray scheduling overhead.

    num_gpu_combos
        Passed straight through to run_grid_sweep - see its
        docstring. Only affects the grid sweep; the feature-
        importance refit afterward runs locally regardless (see
        above) and will auto-detect whatever device is available on
        that one machine, same as always.

    importance_n_repeats, permutation_repeats
        Both drive the cost of the feature-importance pass, and
        BOTH matter for how long it takes - lowering only one still
        leaves the other as a full-cost multiplier. Each of
        importance_n_repeats outer iterations does a full model
        refit on the training split (not a cheap operation on a
        large dataset) followed by permutation_repeats-many shuffles
        PER FEATURE or sklearn's permutation_importance - so total
        cost is roughly n_features x permutation_repeats x
        importance_n_repeats predict-and-score calls, all
        sequential (sklearn's own n_jobs is never overridden here).
        For a fast pilot on a wide feature set (e.g.
        logistic_regression's one-hot encoding), drop BOTH to small
        values, e.g. importance_n_repeats=1, permutation_repeats=2 -
        not just one of them.

    Persistence (resume=True, the default) works at two levels:

    1. Whole-model level: if a previous call already finished
       completely for this model_name/output_dir/primary_metric,
       a <model_name>_final_result.json marker is present and this
       returns the saved result immediately without recomputing
       anything - grid sweep, importance, and plots all skipped.
    2. Mid-sweep level: if the previous call was interrupted partway
       through the grid sweep, run_grid_sweep's own checkpoint (see
       its docstring) resumes from whichever combinations already
       finished, rather than restarting the sweep from scratch.

    Both checks are keyed to a fingerprint of the actual X/y/groups
    data (see _fingerprint_dataset), so pointing this at a rebuilt
    dataset - e.g. swapping the 9-match sample for the full corpus -
    is detected automatically and triggers a fresh run rather than
    silently reusing results computed on different data.

    Pass resume=False to ignore any existing checkpoint/marker and
    force a completely fresh run (still writes fresh checkpoints as
    it goes, so this is also how you'd deliberately restart).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRIDS[model_name]

    final_result_path = output_dir / f"{model_name}_final_result.json"

    if resume and final_result_path.exists():

        results_csv = output_dir / f"{model_name}_grid_results.csv"
        importance_csv = output_dir / f"{model_name}_feature_importance.csv"

        try:

            with open(final_result_path) as file:
                saved = json.load(file)

            fingerprint = _fingerprint_dataset(X, y, groups)

            if (
                saved.get("primary_metric") == primary_metric
                and saved.get("fingerprint") == fingerprint
                and results_csv.exists()
                and importance_csv.exists()
            ):

                print(
                    f"[{model_name}] Found a completed run in {output_dir} "
                    f"(same data, primary_metric={primary_metric}) - "
                    f"loading saved results instead of re-running. Pass "
                    f"resume=False to force a fresh run."
                )

                return {
                    "model_name": model_name,
                    "best_params": saved["best_params"],
                    "best_score": saved["best_score"],
                    "results_df": pd.read_csv(results_csv),
                    "importance_df": pd.read_csv(importance_csv),
                }

            elif saved.get("fingerprint") != _fingerprint_dataset(X, y, groups):
                print(
                    f"[{model_name}] Found a previous completed run in "
                    f"{output_dir}, but it was computed against different "
                    f"data - re-running rather than reusing it."
                )
            else:
                print(
                    f"[{model_name}] Found a previous completed run in "
                    f"{output_dir}, but primary_metric or output files "
                    f"don't match - re-running."
                )

        except (json.JSONDecodeError, KeyError) as error:
            print(
                f"[{model_name}] {final_result_path.name} is corrupted "
                f"({error}) - re-running."
            )

    results_df, param_dicts = run_grid_sweep(
        model_name, X, y,
        param_grid=param_grid,
        n_repeats=n_repeats,
        test_size=test_size,
        groups=groups,
        base_random_state=base_random_state,
        output_dir=output_dir,
        resume=resume,
        use_ray=use_ray,
        ray_address=ray_address,
        max_concurrent_tasks=max_concurrent_tasks,
        memory_per_task=memory_per_task,
        ray_py_modules=ray_py_modules,
        num_gpu_combos=num_gpu_combos,
    )

    results_df.to_csv(output_dir / f"{model_name}_grid_results.csv", index=False)

    plot_marginal_effects(
        results_df,
        param_names=list(param_grid.keys()),
        metric=primary_metric,
        output_dir=output_dir,
        model_name=model_name,
    )

    if results_df[primary_metric].isna().all():

        total_valid = (
            results_df["n_valid_roc_auc_splits"].sum()
            if "n_valid_roc_auc_splits" in results_df.columns
            else 0
        )

        raise RuntimeError(
            f"[{model_name}] primary_metric '{primary_metric}' is NaN "
            f"for every hyperparameter combination ({total_valid} total "
            f"valid splits across the whole sweep) - picking a 'best' "
            f"combination here would be arbitrary, not a real result, "
            f"so this stops rather than silently returning one. This "
            f"happens when a class never appears in any test fold "
            f"across every repeat - with match-aware grouping and only "
            f"a handful of matches, a rare class (see "
            f"feature_engineering.py's thin-class warning) can miss "
            f"every single split. Options: (1) resolve the thin "
            f"class(es) - drop or merge them in feature_engineering.py "
            f"- before re-running, (2) rebuild the dataset from more "
            f"demos so rare classes have more chances to land in a "
            f"test fold, or (3) pass primary_metric='f1_mean' instead, "
            f"which uses zero_division=0 and stays defined even when a "
            f"class is missing from a fold (see _compute_metrics)."
        )

    n_nan_combos = int(results_df[primary_metric].isna().sum())

    if n_nan_combos:
        print(
            f"\n[{model_name}] WARNING: {n_nan_combos}/{len(results_df)} "
            f"combination(s) have a NaN {primary_metric} (every repeat's "
            f"test fold missed a class for that specific hyperparameter "
            f"setting - see _compute_metrics) and are excluded from "
            f"'best combination' selection below. This can happen "
            f"systematically for settings that assign a rare class "
            f"(e.g. a thin class) near-zero probability across the "
            f"board, not just from bad luck in the split - worth a "
            f"look at which combinations these were "
            f"(results_df[results_df['{primary_metric}'].isna()]) "
            f"before trusting the sweep as a whole.\n"
        )

    # np.argmax (and pandas' own idxmax) get "stuck" on the FIRST NaN
    # they encounter and never consider anything after it - including
    # a genuinely higher real score two rows later - because any
    # comparison against NaN evaluates to False. With hundreds of
    # combinations and a rare class in play, some rows being NaN while
    # others aren't is a real, not just theoretical, occurrence (see
    # the warning above), so this has to be NaN-aware rather than a
    # bare argmax. np.nanargmax raises ValueError only when EVERY
    # value is NaN, which is already handled by the check above this
    # block, so it's always safe to call here.
    best_position = np.nanargmax(results_df[primary_metric].to_numpy())
    best_row = results_df.iloc[best_position]
    best_params = param_dicts[best_position]

    print(f"\n[{model_name}] Best combination by {primary_metric}: {best_params}")
    print(f"[{model_name}] {primary_metric} = {best_row[primary_metric]:.4f}")

    importance_df = run_feature_importance_analysis(
        model_name, best_params, X, y,
        n_repeats=importance_n_repeats,
        test_size=test_size,
        groups=groups,
        base_random_state=base_random_state,
        scoring=importance_scoring,
        permutation_repeats=permutation_repeats,
    )

    importance_df.to_csv(
        output_dir / f"{model_name}_feature_importance.csv", index=False
    )

    plot_feature_importance(
        importance_df, output_dir=output_dir, model_name=model_name
    )

    best_score = float(best_row[primary_metric])

    with open(final_result_path, "w") as file:
        json.dump(
            {
                "best_params": best_params,
                "best_score": best_score,
                "primary_metric": primary_metric,
                "fingerprint": _fingerprint_dataset(X, y, groups),
            },
            file,
            indent=2,
            default=_json_default,
        )

    return {
        "model_name": model_name,
        "best_params": best_params,
        "best_score": best_score,
        "results_df": results_df,
        "importance_df": importance_df,
    }


# ==========================================================
# Run for Every Model Type
# ==========================================================


def run_all_models(
    X: pd.DataFrame,
    y: np.ndarray,
    output_dir: Path | str,
    n_repeats: int = 100,
    groups: np.ndarray | None = None,
    models: list[str] | None = None,
    resume: bool = True,
    use_ray: bool = False,
    ray_address: str | None = None,
    max_concurrent_tasks: int = 64,
    importance_n_repeats: int = 20,
    importance_scoring: str | None = None,
    permutation_repeats: int = 10,
    memory_per_task: int | None = None,
    ray_py_modules: list[str] | None = None,
    num_gpu_combos: int = 0,
    memory_per_task_by_model: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Runs run_full_tuning() for every model in `models` (defaults to
    every model in MODEL_BUILDERS), using each model's default grid
    from DEFAULT_PARAM_GRIDS. Returns a dict of model_name ->
    summary dict.

    importance_n_repeats
        Passed straight through to every model's run_full_tuning()
        call. Worth lowering for a quick pilot run: the feature-
        importance pass runs entirely LOCALLY on whichever machine
        calls this function, never distributed across Ray regardless
        of use_ray - see run_full_tuning's docstring. It also scales
        with feature count, so it's the slow part specifically for
        logistic_regression's one-hot encoding (hundreds of columns)
        even when the Ray-distributed grid sweep itself is fast.
    from DEFAULT_PARAM_GRIDS. Returns a dict of model_name ->
    summary dict.

    use_ray, ray_address, max_concurrent_tasks
        Passed straight through to every model's run_full_tuning()
        call (see its docstring, and run_grid_sweep's for what each
        actually does). Models run one after another, not
        concurrently with each other - each model's own grid sweep
        is what gets distributed across the cluster.

    num_gpu_combos
        Passed to run_full_tuning(), but ONLY for model_name ==
        "transformer" - every other builder here (logistic
        regression, random forest, XGBoost) has no GPU code path at
        all, so requesting num_gpus=1 for their tasks would just
        reserve the GPU node without ever using it, starving the
        transformer's own GPU lane for nothing. Set this only when
        "transformer" is in `models`; it's silently ignored (passed
        as 0) for every other model_name regardless of what you pass
        here.

    memory_per_task_by_model
        run_grid_sweep's own memory_per_task default -
        4 * X.shape[0] * X.shape[1] * 8 bytes - is sized around
        sklearn's Pipeline: a wide one-hot frame gets densified to
        float64 twice over (the raw conversion, then the scaled
        output) before fitting even starts. That reasoning doesn't
        transfer to every model_name here - the transformer never
        builds that pair of full-X float64 copies at all: it
        tensorizes only the (smaller) train split, as float32 for
        continuous columns and int64 for categorical ones, not
        float64 for everything. Applying the sklearn-sized default
        to it doesn't just waste cluster memory headroom - it can
        make individual tasks request more of Ray's per-node memory
        resource than any single node actually has free, which
        stalls them as "pending forever, no error" rather than
        merely running a bit under-parallelized (confirmed in
        practice: a transformer sweep stuck at 0 running, 0 CPU
        used anywhere, while `ray status` showed the remaining
        combinations as infeasible-pending on a memory request
        sized for one-hot data the transformer never actually
        needed).

        A dict of model_name -> explicit memory_per_task override,
        keyed exactly like `models`/MODEL_BUILDERS. Only applies to
        the model_name(s) present as keys - every other model still
        gets the shared `memory_per_task` above (None -> its own
        automatic per-model heuristic, computed fresh from that
        model's own X). Like num_gpu_combos, this only matters for
        use_ray=True; the sequential local path has no per-task
        memory reservation to get wrong.

    IMPORTANT: every model in `models` is fit against this SAME X -
    correct only when every model you're running shares one encoding
    (e.g. models=["random_forest", "xgboost", "transformer"], which
    all use "ordinal" by default). Mixing in "logistic_regression"
    here would silently fit it on ordinal-coded categoricals (a
    linear model reading attacker_callout=47 as a number twice
    attacker_callout=23), or fit the tree/transformer models on
    onehot data instead (works, but defeats the point of ordinal
    encoding, and breaks the transformer's categorical-column
    detection entirely - see run_all_models_from_csvs()'s docstring
    for exactly how). Neither mistake raises an exception. For the
    common case of running all four models together - which
    unavoidably spans both encodings - use
    run_all_models_from_csvs() instead, which loads each model's own
    correctly-encoded CSV automatically.

    resume=True (default) is passed through to each model's
    run_full_tuning() call - see its docstring for the two levels
    of persistence this gives you. In practice this means: if this
    function gets interrupted partway through, e.g. after fully
    finishing logistic_regression and getting partway through
    random_forest's sweep, calling run_all_models() again with the
    same output_dir skips logistic_regression entirely and resumes
    random_forest from wherever its sweep left off, rather than
    starting all three models over from nothing.
    """

    if models is None:
        models = list(MODEL_BUILDERS.keys())

    all_results = {}

    for model_name in models:
        print(f"\n{'='*60}\nRunning {model_name}\n{'='*60}")
        all_results[model_name] = run_full_tuning(
            model_name, X, y,
            output_dir=Path(output_dir) / model_name,
            n_repeats=n_repeats,
            groups=groups,
            resume=resume,
            use_ray=use_ray,
            ray_address=ray_address,
            max_concurrent_tasks=max_concurrent_tasks,
            importance_n_repeats=importance_n_repeats,
            importance_scoring=importance_scoring,
            permutation_repeats=permutation_repeats,
            memory_per_task=(memory_per_task_by_model or {}).get(model_name, memory_per_task),
            ray_py_modules=ray_py_modules,
            # See num_gpu_combos' docstring above - the GPU lane only
            # makes sense for the transformer's own PyTorch code path.
            num_gpu_combos=num_gpu_combos if model_name == "transformer" else 0,
        )

    summary = pd.DataFrame([
        {"model": name, "best_score": r["best_score"], "best_params": r["best_params"]}
        for name, r in all_results.items()
    ]).sort_values("best_score", ascending=False)

    print("\n" + "=" * 60)
    print("Summary across all models:")
    print(summary.to_string(index=False))

    summary.to_csv(Path(output_dir) / "model_comparison_summary.csv", index=False)

    return all_results


def run_all_models_from_csvs(
    data_dir: Path | str,
    output_dir: Path | str,
    model_encodings: dict[str, str] | None = None,
    models: list[str] | None = None,
    n_repeats: int = 100,
    resume: bool = True,
    use_ray: bool = False,
    ray_address: str | None = None,
    max_concurrent_tasks: int = 64,
    importance_n_repeats: int = 20,
    importance_scoring: str | None = None,
    permutation_repeats: int = 10,
    memory_per_task: int | None = None,
    ray_py_modules: list[str] | None = None,
    held_out_train_size: float | None = None,
    held_out_val_size: float | None = None,
    held_out_test_size: float | None = None,
    held_out_random_state: int = 42,
    num_gpu_combos: int = 0,
    memory_per_task_by_model: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Encoding-aware counterpart to run_all_models(): loads each
    model's OWN correctly-encoded CSV from data_dir - one
    <model_name>_features.csv per model, exactly what
    feature_engineering.write_model_feature_csvs() writes - instead
    of handing every model the same shared X.

    num_gpu_combos
        Passed to run_full_tuning(), but ONLY for model_name ==
        "transformer" - see run_all_models()'s docstring for why
        (no other model here has a GPU code path, so requesting one
        for their tasks would just reserve the GPU node without
        using it). Silently passed as 0 for every other model_name
        regardless of what you set here.

    memory_per_task_by_model
        Same as run_all_models()'s param of the same name - a dict
        of model_name -> explicit memory_per_task override,
        applying only to model_name(s) present as keys. Particularly
        relevant here since each model_name already loads its OWN
        differently-shaped CSV (that's this function's whole point -
        see above), so the sklearn-Pipeline-sized default heuristic
        being wrong for a leaner model like the transformer isn't a
        hypothetical - it's sized fresh per model_name's own X.shape
        every time, and the transformer's shape doesn't change what
        it actually needs in memory the way the one-hot models' does.
        See run_all_models()'s docstring for the full reasoning and
        the infeasible-pending failure mode this avoids.

    use_ray, ray_address, max_concurrent_tasks
        Passed straight through to every model's run_full_tuning()
        call - see run_grid_sweep's docstring for what each does.
        For an actual multi-node cluster (not just local-machine
        Ray), ray_address should be the head node's address (e.g.
        "10.2.1.189:6379") - or "auto" if this script is itself
        running ON the head node, which is the normal case: connect
        to the cluster, then call this function from there, rather
        than from a third machine outside it.

    importance_n_repeats, importance_scoring
        Also passed straight through to run_full_tuning(). Worth
        lowering for a quick pilot run specifically because this
        step is NOT helped by use_ray/ray_address at all - permutation
        importance always runs locally on whichever machine calls
        this function (see run_full_tuning's docstring), and its
        cost scales with feature count, so it's disproportionately
        slow for logistic_regression's one-hot columns (hundreds of
        features) even when that same model's Ray-distributed grid
        sweep finishes quickly. A fast timing pilot wants both
        n_repeats and importance_n_repeats small - lowering only
        n_repeats still leaves this the long pole.

    This exists because run_all_models() fits every model_name in
    `models` against one shared X, which is silently wrong the
    moment that list spans models needing different encodings.
    logistic_regression needs onehot; random_forest, xgboost, and
    transformer need ordinal (see feature_engineering.py's
    docstring, point 4) - and "all four together" is exactly the
    case that needs both. Concretely, what goes wrong with a single
    shared X, verified rather than assumed:

    - onehot X handed to random_forest/xgboost: fits without error,
      but each category (e.g. one of attacker_callout's ~111 values)
      only gets to compete for a split as its own near-empty column
      instead of one shared column with room for every category -
      exactly the dilution problem ordinal encoding exists to avoid.
    - onehot X handed to the transformer: worse.
      make_transformer_builder's categorical-column detection is
      `[c for c in CATEGORICAL_COLUMNS if c in X.columns]` - against
      onehot data this evaluates to an empty list (onehot explodes
      "attacker_callout" into columns like "attacker_callout_LongA",
      so the original name is never a column). Every one of the 549
      columns silently falls through to "continuous", meaning zero
      categorical embedding tables get built at all - a completely
      different, undocumented architecture, not a crash.
    - ordinal X handed to logistic_regression: the mirror-image
      mistake - a linear model reads attacker_callout=47 as a
      number, twice attacker_callout=23, which is meaningless for a
      nominal category. Also fits without error.

    None of the above raises an exception, which is exactly why this
    function exists rather than a warning in a docstring being
    enough.

    Also auto-registers MODEL_BUILDERS["transformer"] (via
    tabular_transformer.make_transformer_builder) if "transformer"
    is requested and not already registered, sized from that model's
    own correctly-loaded ordinal CSV - so you don't have to remember
    to do that by hand before calling this.

    Parameters
    ----------
    data_dir
        Directory containing <model_name>_features.csv per model -
        i.e. wherever you pointed
        feature_engineering.write_model_feature_csvs()'s output_dir.
    model_encodings
        Only consulted to know which model names to expect - not
        used to re-encode anything, since the CSVs are already
        encoded. Defaults to
        feature_engineering.DEFAULT_MODEL_ENCODINGS if importable,
        otherwise the same four names that module has always
        written (logistic_regression/random_forest/xgboost/
        transformer), so this still works if feature_engineering.py
        genuinely isn't on the path - only its CSVs are required.
    models
        Which of model_encodings's keys to actually run. Defaults to
        all of them.
    held_out_train_size, held_out_val_size, held_out_test_size, held_out_random_state
        Opt-in three-way split - see split_train_val_test() and
        evaluate_held_out_test() for the full reasoning. Leave all
        three size arguments as None (the default) to keep this
        function's original behavior exactly: the sweep runs on
        100% of each model's loaded data, and best_score is a
        validation score with the usual mild optimism that comes
        from being the score that picked the winner (see
        run_full_tuning's own docstring on this).

        Set all three together (e.g. 0.6/0.2/0.2) to instead: hold
        out held_out_test_size of each model's data ONCE before its
        sweep even starts, run the sweep only on the remaining
        train+validation pool, then automatically call
        evaluate_held_out_test() on the winning combination once the
        sweep finishes - so best_score in the returned dict is still
        the validation score (comparable across models, same as
        always), but a separate held_out_test entry with the genuine
        untouched-data score is added alongside it, and
        model_comparison_summary.csv gets both columns instead of
        just one. Must be given all three or none - a partial
        specification (e.g. only held_out_test_size) is rejected
        rather than guessing sensible values for the other two.

        The same held_out_random_state is used for every model, so
        all four models hold out the same matches - not required
        for correctness (each model's split is independently valid
        either way), but it does mean the four held-out scores are
        being computed on the same test matches, which is the fairer
        comparison for RQ3.

    Returns
    -------
    Same shape as run_all_models(): {model_name: summary dict}, plus
    a "held_out_test" key in each model's dict when the held-out
    split arguments are used.
    Also writes model_comparison_summary.csv into output_dir.
    """

    held_out_args = {
        "train": held_out_train_size,
        "val": held_out_val_size,
        "test": held_out_test_size,
    }
    n_given = sum(v is not None for v in held_out_args.values())

    if n_given not in (0, 3):
        raise ValueError(
            "held_out_train_size/held_out_val_size/held_out_test_size "
            "must be given all three together or not at all - got "
            f"{ {k: v for k, v in held_out_args.items() if v is not None} }, "
            f"missing {[k for k, v in held_out_args.items() if v is None]}. "
            "Partial specification isn't rejected because guessing "
            "the missing proportions could silently give you a split "
            "you didn't ask for."
        )

    use_held_out_split = n_given == 3

    if model_encodings is None:
        try:
            from feature_engineering import DEFAULT_MODEL_ENCODINGS
            model_encodings = DEFAULT_MODEL_ENCODINGS
        except ImportError:
            model_encodings = {
                "logistic_regression": "onehot",
                "random_forest": "ordinal",
                "xgboost": "ordinal",
                "transformer": "ordinal",
            }

    if models is None:
        models = list(model_encodings.keys())

    unknown = [name for name in models if name not in model_encodings]
    if unknown:
        raise ValueError(
            f"models {unknown} not found in model_encodings "
            f"{list(model_encodings.keys())} - pass model_encodings "
            f"explicitly if these are custom model names."
        )

    try:
        from feature_engineering import load_model_csv
    except ImportError as error:
        raise ImportError(
            "run_all_models_from_csvs() needs feature_engineering.py "
            "importable (to read the <model_name>_features.csv files "
            "the same way they were written) - either put it on the "
            "path, or load each CSV yourself and call run_all_models() "
            "per encoding group instead."
        ) from error

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Read once per model up front (not lazily inside the loop below)
    # so a missing file is caught for every requested model before
    # any tuning starts, rather than failing halfway through after
    # already spending time on earlier models.
    loaded: dict[str, tuple[pd.DataFrame, pd.Series, np.ndarray]] = {}

    for model_name in models:

        csv_path = data_dir / f"{model_name}_features.csv"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Expected {csv_path} (encoding="
                f"{model_encodings[model_name]!r} for model "
                f"{model_name!r}) - run "
                f"feature_engineering.write_model_feature_csvs() "
                f"first, or point data_dir at wherever those files "
                f"actually are."
            )

        loaded[model_name] = load_model_csv(csv_path)

        print(
            f"[run_all_models_from_csvs] {model_name}: loaded "
            f"{csv_path.name} ({model_encodings[model_name]}, "
            f"{loaded[model_name][0].shape[1]} features)"
        )

    if "transformer" in models and "transformer" not in MODEL_BUILDERS:

        try:
            from tabular_transformer import make_transformer_builder
        except ImportError as error:
            raise ImportError(
                "models includes 'transformer' but tabular_transformer.py "
                "isn't importable (torch not installed?) - install torch, "
                "or drop 'transformer' from models."
            ) from error

        X_transformer = loaded["transformer"][0]
        MODEL_BUILDERS["transformer"] = make_transformer_builder(X_transformer)

        print(
            "[run_all_models_from_csvs] Registered "
            "MODEL_BUILDERS['transformer'], sized from "
            f"{data_dir / 'transformer_features.csv'}"
        )

    all_results = {}
    held_out_test_sets: dict[str, tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]] = {}

    for model_name in models:

        X, y, groups = loaded[model_name]
        sweep_test_size = None

        if use_held_out_split:

            split = split_train_val_test(
                X, y, groups,
                train_size=held_out_train_size,
                val_size=held_out_val_size,
                test_size=held_out_test_size,
                random_state=held_out_random_state,
            )

            X, y, groups = (
                split["X_trainval"], split["y_trainval"], split["groups_trainval"]
            )
            sweep_test_size = split["sweep_test_size"]

            # Kept for evaluate_held_out_test() after this model's
            # sweep finishes - X/y above have already been replaced
            # with the trainval-only pool, so the untouched test
            # portion has to be remembered separately here.
            held_out_test_sets[model_name] = (X, y, split["X_test"], split["y_test"])

        print(
            f"\n{'='*60}\nRunning {model_name} "
            f"({model_encodings[model_name]} encoding, "
            f"{X.shape[1]} features)\n{'='*60}"
        )

        run_kwargs = dict(
            output_dir=output_dir / model_name,
            n_repeats=n_repeats,
            groups=groups,
            resume=resume,
            use_ray=use_ray,
            ray_address=ray_address,
            max_concurrent_tasks=max_concurrent_tasks,
            importance_n_repeats=importance_n_repeats,
            importance_scoring=importance_scoring,
            permutation_repeats=permutation_repeats,
            memory_per_task=(memory_per_task_by_model or {}).get(model_name, memory_per_task),
            ray_py_modules=ray_py_modules,
            # See num_gpu_combos' docstring above - only the
            # transformer has a GPU code path.
            num_gpu_combos=num_gpu_combos if model_name == "transformer" else 0,
        )

        if sweep_test_size is not None:
            run_kwargs["test_size"] = sweep_test_size

        all_results[model_name] = run_full_tuning(model_name, X, y, **run_kwargs)

        if use_held_out_split:

            X_trainval, y_trainval, X_test, y_test = held_out_test_sets[model_name]

            all_results[model_name]["held_out_test"] = evaluate_held_out_test(
                model_name,
                all_results[model_name]["best_params"],
                X_trainval, y_trainval,
                X_test, y_test,
                output_dir=output_dir / model_name,
                random_state=held_out_random_state,
            )

    if use_held_out_split:

        summary = pd.DataFrame([
            {
                "model": name,
                "validation_roc_auc": r["best_score"],
                "held_out_test_roc_auc": r["held_out_test"]["roc_auc"],
                "best_params": r["best_params"],
            }
            for name, r in all_results.items()
        ]).sort_values("held_out_test_roc_auc", ascending=False)

    else:

        summary = pd.DataFrame([
            {"model": name, "best_score": r["best_score"], "best_params": r["best_params"]}
            for name, r in all_results.items()
        ]).sort_values("best_score", ascending=False)

    print("\n" + "=" * 60)
    print("Summary across all models:")
    print(summary.to_string(index=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "model_comparison_summary.csv", index=False)

    return all_results
