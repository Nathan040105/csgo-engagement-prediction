"""
grouped_permutation_importance.py - how much does each KIND of feature matter,
measured on the 278 held-out TEST matches with the project's metric (ROC-AUC)?

Why this exists (what feature_importance_analysis.py cannot tell you)
---------------------------------------------------------------------
The saved importances come from shuffling ONE feature at a time and watching
macro-F1 on validation folds. Two problems for the report:

  1. Features here are heavily redundant (attacker_hp / hp_difference /
     victim_hp, has_flash / flashbangs, the two *_side features, ...). Shuffle
     one member of a group of near-duplicates and the model simply reads the
     other, so each looks less important than the group really is; or the
     shuffle builds a row that never occurs in real data and the score drops
     for the wrong reason (see the *_side check below).
  2. Macro-F1 is dominated by the thin classes (mutual_kill has recall 0 for
     every model); ROC-AUC is the project's primary metric.

This script refits each model with its tuned hyperparameters on the 80%
train+val pool (exactly what evaluate_held_out_test does), then for every
CATEGORY of features shuffles all of the category's columns TOGETHER (one
shared row permutation, so the columns stay consistent with each other and only
their link to everything else is broken) on the untouched 20% test matches and
records the fall in ROC-AUC and macro-F1. It also runs:

  * the *_side check: attacker_side alone, victim_side alone, and both together.
    The transformer's single-feature importance is dominated by these two
    columns, which are mirror images in 98.2% of rows and are also implied by
    the alive-teammate counts, so single-column shuffles are suspect.
  * a floor: shuffling EVERY feature should push ROC-AUC to about 0.5. If it
    does not, something is wrong.
  * optionally (--ablate-sides) a refit WITHOUT the two side columns, the clean
    answer to "does the model need them?" (costs one more refit).

How to run  (from the folder that holds tuning_results_n7/ and tuning_data/)
-----------------------------------------------------------------------------
  nathanserver (Linux, 8 CPU):
      python grouped_permutation_importance.py xgboost random_forest logistic_regression --n-jobs 8
  LLMAI (Linux, GPU) - the transformer refit needs the GPU and takes hours:
      tmux new -s importance
      python -u grouped_permutation_importance.py transformer
  Add --ablate-sides to ALSO refit without the two side columns. That is a second
  full refit; the grouped results are written to disk before it starts, so an
  interrupted ablation does not lose them.
  After copying the transformer's CSV into the same output folder, rebuild the
  combined table and figure on any machine (no models needed):
      python grouped_permutation_importance.py --plot-only

  Quick plumbing test (a few minutes, NOT results):
      python grouped_permutation_importance.py xgboost random_forest logistic_regression --smoke

Needs: model_tuning.py, feature_engineering.py, feature_importance_analysis.py
(and tabular_transformer.py + torch for the transformer) in the same folder.

Outputs (default <results-dir>/grouped_permutation_importance/):
  <model>_grouped_importance.csv        one row per feature group
  <model>_grouped_importance_meta.json  baseline vs saved test score, timings, versions
  grouped_importance_all_models.csv     every model together
  grouped_importance_auc.png            the figure for the report

Reading the numbers: "auc_drop" is baseline test ROC-AUC minus the AUC with that
group shuffled; bigger = the model leans on that group more. The std is over the
shuffle repeats only (it does NOT include the uncertainty from having just 278
test matches, which was not measured), so treat two groups whose drops lie within a
couple of stds of each other as tied, and do not add the drops up: groups share
information, so the drops of all categories sum to LESS than the drop when everything
is shuffled.

The test matches are used here only to DESCRIBE fitted models. Do not use any number
from this script (including the --ablate-sides comparison) to choose features or
hyperparameters; that decision belongs on validation data.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_importance_analysis import (
    CATEGORIES, CATEGORY_SHORT, MODEL_LABELS, MODELS, base_of, category_of,
)

# Must match run_final_tuning.py / run_feature_elimination.py / accuracy_over_time.py
HELD_OUT_SIZES = (0.6, 0.2, 0.2)
HELD_OUT_SEED = 42
SAVED_SCORE_TOLERANCE = 0.002       # refit vs the saved held-out result

SIDE_BASES = ("attacker_side", "victim_side")

# label codes are the alphabetical order of engagement_status (feature_engineering.build_label)
CLASS_NAMES = {0: "attacker_won", 1: "interrupted", 2: "mutual_kill",
               3: "round_end", 4: "timedout", 5: "victim_won"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAR = "#2a78d6"


# ----------------------------------------------------------------------------
# Column groups
# ----------------------------------------------------------------------------

def column_groups(columns: list[str]):
    """Returns (categories, extras, floor): each {group name: [model columns]}.

    A model column is whatever the model actually sees (for logistic
    regression a categorical feature is several one-hot columns; they are all
    shuffled together with the same permutation).
    """
    by_category: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    by_base: dict[str, list[str]] = {}
    for column in columns:
        base = base_of(column)
        by_category[category_of(base)].append(column)
        by_base.setdefault(base, []).append(column)

    categories = {c: cols for c, cols in by_category.items() if cols}
    extras = {
        "attacker_side alone": by_base.get("attacker_side", []),
        "victim_side alone": by_base.get("victim_side", []),
        "attacker_side + victim_side together": (
            by_base.get("attacker_side", []) + by_base.get("victim_side", [])),
    }
    extras = {name: cols for name, cols in extras.items() if cols}
    floor = {"ALL features (sanity floor)": list(columns)}
    return categories, extras, floor


def n_original_features(columns: list[str]) -> int:
    return len({base_of(c) for c in columns})


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

def score(model, X, y_true, all_classes, compute_metrics, return_proba=False):
    proba = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    y_pred = classes[np.argmax(proba, axis=1)]
    metrics = compute_metrics(y_true, y_pred, proba, all_classes)
    if not np.isfinite(metrics["roc_auc"]):
        raise RuntimeError("ROC-AUC is undefined on the test set (a class is missing?)")
    return (metrics, proba) if return_proba else metrics


def per_class_auc(y_true, proba, classes) -> dict:
    """One-vs-rest ROC-AUC for every class. The macro ROC-AUC is the mean of
    these six numbers, and mutual_kill (40 test rows) is so thin that its AUC
    alone can move the macro average by several thousandths between two
    otherwise equivalent models."""
    from sklearn.metrics import roc_auc_score   # lazy: --plot-only needs no scikit-learn
    return {CLASS_NAMES.get(int(c), str(c)):
            float(roc_auc_score((y_true == c).astype(int), proba[:, i]))
            for i, c in enumerate(classes)}


def shuffle_and_score(model, X, columns, permutation, y_true, all_classes, compute_metrics):
    """Shuffle `columns` of X in place with one shared permutation, score,
    restore. Per-column arrays keep their own dtypes (no float upcasting)."""
    originals = {c: X[c].to_numpy(copy=True) for c in columns}
    try:
        for c in columns:
            X[c] = originals[c][permutation]
        return score(model, X, y_true, all_classes, compute_metrics)
    finally:
        for c in columns:
            X[c] = originals[c]


# ----------------------------------------------------------------------------
# One model
# ----------------------------------------------------------------------------

def library_versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "sklearn", "xgboost", "torch"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "?")
        except ImportError:
            pass
    return versions


def run_model(model_name: str, args, out_dir: Path) -> None:
    # Heavy imports only when actually fitting (so --plot-only needs just
    # pandas, numpy and matplotlib).
    import model_tuning
    from model_tuning import MODEL_BUILDERS, _compute_metrics, split_train_val_test
    from feature_engineering import load_model_csv

    results_dir, data_dir = Path(args.results_dir), Path(args.data_dir)

    final_path = results_dir / model_name / f"{model_name}_final_result.json"
    saved_path = results_dir / model_name / f"{model_name}_held_out_test_result.json"
    csv_path = data_dir / f"{model_name}_features.csv"
    for path in (final_path, saved_path, csv_path):
        if not path.exists():
            raise SystemExit(f"[{model_name}] missing {path}")

    with open(final_path, encoding="utf-8") as handle:
        best_params = json.load(handle)["best_params"]
    with open(saved_path, encoding="utf-8") as handle:
        saved = json.load(handle)

    print(f"\n{'=' * 70}\n{MODEL_LABELS[model_name]}: best params {best_params}\n{'=' * 70}")

    X, y, groups = load_model_csv(csv_path)
    split = split_train_val_test(
        X, y, groups,
        train_size=HELD_OUT_SIZES[0], val_size=HELD_OUT_SIZES[1],
        test_size=HELD_OUT_SIZES[2], random_state=HELD_OUT_SEED,
    )
    X_trainval, y_trainval = split["X_trainval"], np.asarray(split["y_trainval"])
    X_test, y_test = split["X_test"].copy(), np.asarray(split["y_test"])
    del split, groups
    if model_name != "transformer":
        # only the transformer's builder needs the full frame (to size its
        # embedding tables); freeing it matters for the 549-column LR file
        del X

    if args.smoke:
        rng = np.random.default_rng(0)
        tr = rng.choice(len(X_trainval), size=min(20000, len(X_trainval)), replace=False)
        te = rng.choice(len(X_test), size=min(20000, len(X_test)), replace=False)
        X_trainval = X_trainval.iloc[tr].reset_index(drop=True)
        y_trainval = y_trainval[tr]
        X_test = X_test.iloc[te].reset_index(drop=True)
        y_test = y_test[te]
        best_params = dict(best_params)
        if "n_estimators" in best_params:
            best_params["n_estimators"] = min(int(best_params["n_estimators"]), 30)
        if best_params.get("max_depth") is None and model_name == "random_forest":
            best_params["max_depth"] = 8
        elif "max_depth" in best_params:
            best_params["max_depth"] = min(int(best_params["max_depth"]), 6)
        print(f"[smoke] {len(X_trainval)} train rows, {len(X_test)} test rows, params {best_params}")

    if model_name == "transformer":
        try:
            from tabular_transformer import make_transformer_builder
        except ImportError as error:
            raise SystemExit("the transformer needs torch and tabular_transformer.py") from error
        MODEL_BUILDERS["transformer"] = make_transformer_builder(X)

    all_classes = np.unique(np.concatenate([y_trainval, y_test]))

    def build():
        estimator = MODEL_BUILDERS[model_name](best_params, HELD_OUT_SEED)
        if args.n_jobs and model_name in ("random_forest", "xgboost"):
            estimator.set_params(n_jobs=args.n_jobs)
        return estimator

    # ---- refit (same as evaluate_held_out_test) ---------------------------
    started = time.time()
    estimator = build()
    print(f"[{model_name}] refitting on {len(X_trainval)} train+val rows ...", flush=True)
    estimator.fit(X_trainval, y_trainval)
    fit_seconds = time.time() - started
    print(f"[{model_name}] fit took {fit_seconds / 60:.1f} min", flush=True)

    if not np.array_equal(np.asarray(estimator.classes_), all_classes):
        raise SystemExit(f"[{model_name}] model classes {estimator.classes_} != {all_classes}")

    baseline, baseline_proba = score(estimator, X_test, y_test, all_classes,
                                     _compute_metrics, return_proba=True)
    baseline_class_auc = per_class_auc(y_test, baseline_proba, estimator.classes_)
    del baseline_proba
    delta = baseline["roc_auc"] - float(saved["roc_auc"])
    print(f"[{model_name}] baseline test ROC-AUC {baseline['roc_auc']:.4f} "
          f"(saved held-out result {saved['roc_auc']:.4f}, difference {delta:+.4f}); "
          f"macro-F1 {baseline['f1']:.4f}, accuracy {baseline['accuracy']:.4f}")
    reproduced = abs(delta) <= SAVED_SCORE_TOLERANCE
    if not args.smoke and not reproduced:
        print(f"[{model_name}] WARNING: the refit differs from the saved held-out result by more "
              f"than {SAVED_SCORE_TOLERANCE} - usually a different library version or thread "
              f"count. The importances below are valid for THIS refit; say so in the report.")

    # ---- grouped shuffles ---------------------------------------------------
    categories, extras, floor = column_groups(list(X_test.columns))
    tests = ([(name, cols, "category") for name, cols in categories.items()]
             + [(name, cols, "side check") for name, cols in extras.items()]
             + [(name, cols, "floor") for name, cols in floor.items()])

    records = []
    loop_started = time.time()
    for repeat in range(args.repeats):
        permutation = np.random.default_rng(args.seed + repeat).permutation(len(X_test))
        for name, columns, kind in tests:
            metrics = shuffle_and_score(estimator, X_test, columns, permutation,
                                        y_test, all_classes, _compute_metrics)
            records.append({
                "group": name, "kind": kind, "repeat": repeat,
                "auc": metrics["roc_auc"], "f1": metrics["f1"], "accuracy": metrics["accuracy"],
            })
        done = repeat + 1
        eta = (time.time() - loop_started) / done * (args.repeats - done)
        print(f"[{model_name}] repeat {done}/{args.repeats} done "
              f"({(time.time() - loop_started) / 60:.1f} min, about {eta / 60:.1f} min left)",
              flush=True)

    raw = pd.DataFrame(records)
    chance = 0.5
    rows = []
    for name, columns, kind in tests:
        part = raw[raw["group"] == name]
        auc_drop = baseline["roc_auc"] - part["auc"]
        f1_drop = baseline["f1"] - part["f1"]
        acc_drop = baseline["accuracy"] - part["accuracy"]
        rows.append({
            "model": model_name,
            "group": name,
            "kind": kind,
            "n_features": n_original_features(columns),
            "n_columns": len(columns),
            "baseline_auc": baseline["roc_auc"],
            "auc_drop": float(auc_drop.mean()),
            "auc_drop_std": float(auc_drop.std(ddof=1)) if len(part) > 1 else np.nan,
            "pct_of_above_chance_auc_lost": 100.0 * float(auc_drop.mean()) / (baseline["roc_auc"] - chance),
            "baseline_f1": baseline["f1"],
            "f1_drop": float(f1_drop.mean()),
            "f1_drop_std": float(f1_drop.std(ddof=1)) if len(part) > 1 else np.nan,
            "accuracy_drop": float(acc_drop.mean()),
            "repeats": int(len(part)),
        })
    table = pd.DataFrame(rows)

    # ---- write the grouped results FIRST ------------------------------------------
    # (the optional ablation below is a second full refit; if it is interrupted or
    # crashes, the shuffle results above must already be safe on disk)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"{model_name}_grouped_importance.csv", index=False)

    def write_meta(ablation):
        meta = {
            "model": model_name,
            "best_params_used": best_params,
            "smoke_test": bool(args.smoke),
            "n_train_rows": int(len(X_trainval)), "n_test_rows": int(len(X_test)),
            "baseline_test": {k: float(v) for k, v in baseline.items()},
            "baseline_per_class_auc": baseline_class_auc,
            "saved_held_out_roc_auc": float(saved["roc_auc"]),
            "saved_held_out_f1": float(saved["f1"]),
            "difference_vs_saved_roc_auc": float(delta),
            "reproduces_saved_result": bool(reproduced),
            "repeats": args.repeats, "seed": args.seed, "n_jobs": args.n_jobs,
            "fit_seconds": fit_seconds, "total_seconds": time.time() - started,
            "versions": library_versions(),
            "side_ablation": ablation,
        }
        with open(out_dir / f"{model_name}_grouped_importance_meta.json", "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)

    write_meta(None)

    show = table[["group", "n_features", "auc_drop", "auc_drop_std", "pct_of_above_chance_auc_lost", "f1_drop"]]
    print(f"\n[{model_name}] grouped permutation importance on the test matches:")
    print(show.round(4).to_string(index=False), flush=True)

    # ---- optional ablation (a second full refit, without the side columns) ---------
    if args.ablate_sides:
        side_columns = [c for c in X_test.columns if base_of(c) in SIDE_BASES]
        keep = [c for c in X_trainval.columns if c not in set(side_columns)]
        print(f"\n[{model_name}] ablation: refitting without {len(side_columns)} side column(s) ...", flush=True)
        if model_name == "transformer":
            from tabular_transformer import make_transformer_builder
            reduced_builder = make_transformer_builder(X[keep])
        else:
            reduced_builder = MODEL_BUILDERS[model_name]
        reduced = reduced_builder(best_params, HELD_OUT_SEED)
        if args.n_jobs and model_name in ("random_forest", "xgboost"):
            reduced.set_params(n_jobs=args.n_jobs)
        reduced.fit(X_trainval[keep], y_trainval)
        without, without_proba = score(reduced, X_test[keep], y_test, all_classes,
                                       _compute_metrics, return_proba=True)
        without_class_auc = per_class_auc(y_test, without_proba, reduced.classes_)
        ablation = {
            "columns_removed": side_columns,
            "auc_with_sides": baseline["roc_auc"], "auc_without_sides": without["roc_auc"],
            "auc_change_when_removed": without["roc_auc"] - baseline["roc_auc"],
            "f1_with_sides": baseline["f1"], "f1_without_sides": without["f1"],
            "accuracy_with_sides": baseline["accuracy"], "accuracy_without_sides": without["accuracy"],
            "per_class_auc_with_sides": baseline_class_auc,
            "per_class_auc_without_sides": without_class_auc,
        }
        write_meta(ablation)
        print(f"[{model_name}] ablation: test ROC-AUC {baseline['roc_auc']:.4f} -> "
              f"{without['roc_auc']:.4f} without the side columns "
              f"({without['roc_auc'] - baseline['roc_auc']:+.4f}); macro-F1 "
              f"{baseline['f1']:.4f} -> {without['f1']:.4f}; accuracy "
              f"{baseline['accuracy']:.4f} -> {without['accuracy']:.4f}")
        print(f"[{model_name}] ablation, change in one-vs-rest AUC per class (the macro AUC is their "
              f"mean; mutual_kill has only 40 test rows, so it is the noisy one):")
        for name in baseline_class_auc:
            print(f"      {name:<14}{baseline_class_auc[name]:.4f} -> {without_class_auc[name]:.4f} "
                  f"({without_class_auc[name] - baseline_class_auc[name]:+.4f})")


# ----------------------------------------------------------------------------
# Combine + figure
# ----------------------------------------------------------------------------

def build_combined(out_dir: Path) -> pd.DataFrame | None:
    frames = []
    for model in MODELS:
        path = out_dir / f"{model}_grouped_importance.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_dir / "grouped_importance_all_models.csv", index=False)
    return combined


def plot_combined(out_dir: Path, combined: pd.DataFrame) -> None:
    cats = combined[combined["kind"] == "category"]
    models = [m for m in MODELS if m in set(cats["model"])]
    mean_drop = cats.groupby("group")["auc_drop"].mean().sort_values(ascending=False)
    order = [g for g in mean_drop.index]
    n = len(models)

    fig_w, fig_h = 3.9 * n + 3.0, 0.5 * len(order) + 2.6
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), facecolor=SURFACE,
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    spread = cats["auc_drop_std"].fillna(0)
    reach = float((cats["auc_drop"] + spread).max())
    # a group whose shuffle HELPED by chance has a small negative drop: leave
    # room on the left so the bar is drawn instead of being clipped at zero
    low = min(0.0, float((cats["auc_drop"] - spread).min())) * 1.15
    y = np.arange(len(order))[::-1]

    for ax, model in zip(axes, models):
        part = cats[cats["model"] == model].set_index("group").reindex(order)
        ax.set_facecolor(SURFACE)
        ax.barh(y, part["auc_drop"], height=0.62, color=BAR, zorder=2)
        ax.errorbar(part["auc_drop"], y, xerr=part["auc_drop_std"].fillna(0), fmt="none",
                    ecolor=INK2, elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=3)
        for yi, (_, row) in zip(y, part.iterrows()):
            spread = 0.0 if np.isnan(row["auc_drop_std"]) else row["auc_drop_std"]
            ax.text(max(row["auc_drop"], 0.0) + spread + reach * 0.02, yi,
                    f"{row['auc_drop']:.3f}".replace("-0.000", "0.000"), va="center", ha="left",
                    fontsize=8.5, color=INK2, zorder=4)
        baseline = float(part["baseline_auc"].iloc[0])
        ax.set_title(MODEL_LABELS[model], loc="left", fontsize=10.5,
                     fontweight="bold", color=INK, pad=20)
        ax.text(0, 1.02, f"baseline test AUC {baseline:.3f}", transform=ax.transAxes,
                fontsize=8.5, color=MUTED, ha="left", va="bottom")
        ax.set_xlim(low, reach * 1.22)
        ax.set_yticks(y)
        ax.set_yticklabels(order, fontsize=9.5, color=INK)
        ax.tick_params(axis="x", labelsize=8.5, colors=INK2, length=0)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.set_xlabel("ROC-AUC lost when the category is shuffled", fontsize=8.5, color=INK2)

    fig.subplots_adjust(left=2.45 / fig_w, right=0.985, top=1 - 1.25 / fig_h,
                        bottom=0.95 / fig_h, wspace=0.12)
    fig.text(0.3 / fig_w, 1 - 0.3 / fig_h, "How much each kind of feature matters, on the held-out test matches",
             ha="left", va="center", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.3 / fig_w, 1 - 0.6 / fig_h,
             "All of a category's columns shuffled together (mean ± std over repeats). "
             "Same x-scale in every panel.", ha="left", va="center", fontsize=9, color=INK2)
    fig.text(0.3 / fig_w, 0.3 / fig_h,
             "Models refit on the train+val matches; scored on the 278 test matches. "
             "The std covers shuffle noise only, not the choice of test matches.",
             ha="left", va="center", fontsize=8, color=MUTED)
    fig.savefig(out_dir / "grouped_importance_auc.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("models", nargs="*", choices=MODELS,
                        help="models to run (default: all four)")
    parser.add_argument("--results-dir", default="tuning_results_n7")
    parser.add_argument("--data-dir", default="tuning_data")
    parser.add_argument("--out-dir", default=None,
                        help="default: <results-dir>/grouped_permutation_importance")
    parser.add_argument("--repeats", type=int, default=5,
                        help="shuffle repeats per group (default 5)")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=0,
                        help="threads for random forest / XGBoost (0 = leave as tuned)")
    parser.add_argument("--ablate-sides", action="store_true",
                        help="also refit WITHOUT attacker_side/victim_side and report the AUC change")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny plumbing test: small data, small ensembles, 1 repeat. NOT results.")
    parser.add_argument("--plot-only", action="store_true",
                        help="rebuild the combined CSV and figure from existing per-model CSVs")
    args = parser.parse_args()

    if args.smoke:
        args.repeats = 1
    default_out = Path(args.results_dir) / "grouped_permutation_importance"
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    if args.smoke:
        out_dir = out_dir / "smoke_test"

    if not args.plot_only:
        for model_name in (args.models or MODELS):
            run_model(model_name, args, out_dir)

    combined = build_combined(out_dir)
    if combined is None:
        raise SystemExit(f"no *_grouped_importance.csv files in {out_dir}")
    plot_combined(out_dir, combined)
    print(f"\nWrote {out_dir / 'grouped_importance_all_models.csv'} and grouped_importance_auc.png")


if __name__ == "__main__":
    main()
