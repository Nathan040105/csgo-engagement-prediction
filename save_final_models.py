"""
save_final_models.py - save the trained models that the report's test results
come from.

Why this script exists
----------------------
No run in this project saved a fitted model. run_final_tuning.py (through
model_tuning.evaluate_held_out_test) and run_feature_elimination.py refit each
tuned model on the 80% train+validation matches, score the 20% held-out test
matches once, and keep only the scores and predictions. This script repeats
exactly that refit (same feature CSV, same seed-42 match split, same tuned
hyperparameters, same model seed), checks that the refit reproduces the saved
test predictions, and then saves the fitted model.

Usage (from the project folder that holds tuning_data/ and tuning_results_n7/)
------------------------------------------------------------------------------
    nathanserver:  python -u save_final_models.py logistic_regression random_forest xgboost --n-jobs 8
    LLMAI (GPU):   python -u save_final_models.py transformer
    any machine:   python save_final_models.py --manifest-only
                   (after copying both machines' models/ folders into one:
                   rebuilds models/manifest.csv and models/.gitignore and
                   re-checks every file's SHA-256)

Run each model on the machine its original held-out refit ran on (the three
scikit-learn/XGBoost models on nathanserver, the transformer on LLMAI's GPU),
so the refit can reproduce the saved predictions exactly.

Options
    --variant both|before|after  "before" = all features (the main results);
                                 "after" = the features kept by feature
                                 elimination. Default: both. Random forest's
                                 elimination removed nothing, so it has no
                                 separate "after" model.
    --n-jobs N     threads for random forest / XGBoost (0 = leave as tuned).
                   The fitted model does not depend on it.
    --overwrite    refit even if the model is already saved
    --smoke        quick plumbing test on a small sample, written to
                   <out-dir>/smoke_test/ - NOT the real models

Run time: minutes each for logistic regression, random forest and XGBoost on
nathanserver. The transformer takes about as long as its original held-out
refit (hours) for each variant. Check nvidia-smi first: if llama-server is
holding the GPU's memory, stop it for the duration of the run.

Outputs (in --out-dir, default models/)
---------------------------------------
    <name>.joblib | <name>.ubj | <name>.pt
        the fitted model; <name> is <model> or <model>_after_elimination
    <name>.json
        the columns the model expects (in order), hyperparameters, seed, data
        and split, library versions, its test metrics and both checks below
    manifest.csv
        one row per saved model: file, size, SHA-256, test ROC-AUC, checks
    .gitignore
        lists the model files over 95 MB so `git add .` skips them (GitHub
        rejects files over 100 MB); upload those as release assets instead

Two checks per model, both recorded in <name>.json and manifest.csv
    reproduces_saved_predictions  the refit's test predictions equal the ones
                                  saved by the original run (same predicted
                                  class on every test engagement, class
                                  probabilities within 1e-6)
    reload_matches                the file read back from disk predicts the
                                  same as the fitted model (same classes,
                                  probabilities within 1e-6)

Formats
-------
    logistic_regression   joblib (the scikit-learn Pipeline: scaler + model)
    random_forest         joblib, compressed
    xgboost               XGBoost's own .ubj format
    transformer           PyTorch file with the weights, the input
                          standardisation and the constructor settings;
                          readable with torch.load(..., weights_only=True)
The joblib files are pickles: they need the scikit-learn version in
requirements.txt and, like any pickle, should only be loaded if you trust
where they came from.

Loading a saved model
---------------------
    from feature_engineering import load_model_csv
    from save_final_models import load_final_model

    model, columns = load_final_model("models/xgboost.json")
    X, y, groups = load_model_csv("tuning_data/xgboost_features.csv")
    proba = model.predict_proba(X[columns])      # one column per class code

Class codes are alphabetical: 0 attacker_won, 1 interrupted, 2 mutual_kill,
3 round_end, 4 timedout, 5 victim_won.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

MODELS = ["logistic_regression", "random_forest", "xgboost", "transformer"]

# Must match run_final_tuning.py / run_feature_elimination.py.
HELD_OUT_SIZES = (0.6, 0.2, 0.2)
HELD_OUT_SEED = 42

CLASS_NAMES = {0: "attacker_won", 1: "interrupted", 2: "mutual_kill",
               3: "round_end", 4: "timedout", 5: "victim_won"}

FORMATS = {
    "logistic_regression": ("joblib", ".joblib"),
    "random_forest": ("joblib", ".joblib"),
    "xgboost": ("xgboost-ubj", ".ubj"),
    "transformer": ("torch", ".pt"),
}

GITHUB_SAFE_BYTES = 95 * 1024 * 1024     # GitHub rejects files over 100 MiB
PROBA_TOLERANCE = 1e-6

# Constructor arguments of TabularTransformerClassifier that are stored in the
# .pt file (everything except device and verbose, which are set at load time).
TRANSFORMER_INIT_KEYS = (
    "continuous_columns", "categorical_columns", "categorical_cardinalities",
    "d_token", "n_layers", "n_heads", "dropout", "learning_rate",
    "weight_decay", "batch_size", "max_epochs", "patience",
    "validation_fraction", "random_state", "num_threads",
    "progress_every_n_batches",
)

MANIFEST_FIELDS = [
    "name", "model", "variant", "file", "size_mb", "sha256", "sha256_ok",
    "in_git", "n_features", "test_roc_auc", "saved_test_roc_auc",
    "reproduces_saved_predictions", "n_test_predictions_different",
    "max_abs_probability_difference", "reload_matches", "smoke_test",
]


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def _plain(value):
    """numpy / Path values -> plain Python values that json.dump accepts."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_plain(data), handle, indent=2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _library_versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "sklearn", "xgboost", "torch", "joblib"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "?")
        except ImportError:
            pass
    return versions


def _compare(y_pred_a, proba_a, y_pred_b, proba_b) -> dict:
    """How far apart two sets of test predictions are."""
    proba_a = np.asarray(proba_a, dtype=np.float64)
    proba_b = np.asarray(proba_b, dtype=np.float64)
    if proba_a.shape != proba_b.shape:
        return {"n_test_predictions_different": int(len(y_pred_a)),
                "max_abs_probability_difference": float("inf"),
                "note": f"probability shapes differ: {proba_a.shape} vs {proba_b.shape}"}
    return {
        "n_test_predictions_different":
            int(np.sum(np.asarray(y_pred_a) != np.asarray(y_pred_b))),
        "max_abs_probability_difference":
            float(np.max(np.abs(proba_a - proba_b))) if proba_a.size else 0.0,
    }


def _smoke_rows(y: np.ndarray, n: int, rng) -> np.ndarray:
    """A small sample that still contains every class (XGBoost needs the
    class codes it is trained on to be 0..k-1 with none missing)."""
    keep = [rng.choice(np.flatnonzero(y == c), size=min(50, int(np.sum(y == c))),
                       replace=False) for c in np.unique(y)]
    taken = np.concatenate(keep)
    rest = np.setdiff1d(np.arange(len(y)), taken)
    extra = rng.choice(rest, size=max(0, min(n - len(taken), len(rest))), replace=False)
    return np.sort(np.concatenate([taken, extra]))


# ----------------------------------------------------------------------------
# Saving and loading
# ----------------------------------------------------------------------------

def _save_transformer(estimator, path: Path) -> None:
    import torch

    init_params = {key: _plain(getattr(estimator, key)) for key in TRANSFORMER_INIT_KEYS}
    payload = {
        "format_version": 1,
        "init_params": init_params,
        "state_dict": {key: value.detach().cpu().clone()
                       for key, value in estimator.model_.state_dict().items()},
        "classes": [int(c) for c in estimator.classes_],
        "cont_mean": torch.from_numpy(np.ascontiguousarray(estimator._cont_mean).copy()),
        "cont_std": torch.from_numpy(np.ascontiguousarray(estimator._cont_std).copy()),
    }
    torch.save(payload, path)


def _load_transformer(path: Path, device: str | None = None):
    import torch
    from tabular_transformer import TabularTransformer, TabularTransformerClassifier

    payload = torch.load(path, map_location="cpu", weights_only=True)
    params = dict(payload["init_params"])

    estimator = TabularTransformerClassifier(**params, device=device, verbose=False)
    estimator.classes_ = np.asarray(payload["classes"], dtype=np.int64)
    estimator._cont_mean = payload["cont_mean"].numpy()
    estimator._cont_std = payload["cont_std"].numpy()

    network = TabularTransformer(
        n_continuous=len(params["continuous_columns"]),
        categorical_cardinalities=[params["categorical_cardinalities"][c]
                                   for c in params["categorical_columns"]],
        n_classes=len(estimator.classes_),
        d_token=params["d_token"],
        n_layers=params["n_layers"],
        n_heads=params["n_heads"],
        dropout=params["dropout"],
    )
    network.load_state_dict(payload["state_dict"])
    estimator.model_ = network.to(estimator.device).eval()
    return estimator


def _save_estimator(model_name: str, estimator, path: Path) -> None:
    fmt = FORMATS[model_name][0]
    if fmt == "joblib":
        import joblib
        joblib.dump(estimator, path, compress=3)
    elif fmt == "xgboost-ubj":
        estimator.save_model(str(path))
    elif fmt == "torch":
        _save_transformer(estimator, path)
    else:
        raise ValueError(f"unknown format {fmt!r}")


def load_final_model(path, device: str | None = None):
    """Load a model saved by this script.

    path: the model's .json file (or the model file next to it).
    device: transformer only - "cpu", "cuda", or None for automatic.
    Returns (model, columns); call model.predict_proba(X[columns]).
    """
    path = Path(path)
    meta_path = path if path.suffix == ".json" else path.with_suffix(".json")
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)

    model_path = meta_path.parent / meta["file"]
    fmt = meta["format"]

    if fmt == "joblib":
        import joblib
        model = joblib.load(model_path)
    elif fmt == "xgboost-ubj":
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(str(model_path))
    elif fmt == "torch":
        model = _load_transformer(model_path, device)
    else:
        raise ValueError(f"{meta_path}: unknown format {fmt!r}")

    return model, list(meta["feature_columns"])


# ----------------------------------------------------------------------------
# Refit, check, save
# ----------------------------------------------------------------------------

def run_model(model_name: str, variants: list[str], args, out_dir: Path) -> None:
    from model_tuning import MODEL_BUILDERS, _compute_metrics, split_train_val_test
    from feature_engineering import load_model_csv

    results_dir, data_dir = Path(args.results_dir), Path(args.data_dir)

    final_path = results_dir / model_name / f"{model_name}_final_result.json"
    csv_path = data_dir / f"{model_name}_features.csv"
    for path in (final_path, csv_path):
        if not path.exists():
            raise SystemExit(f"[{model_name}] missing {path}")

    with open(final_path, encoding="utf-8") as handle:
        best_params = json.load(handle)["best_params"]

    todo = []
    for variant in variants:
        name = model_name if variant == "before" else f"{model_name}_after_elimination"
        result_dir = results_dir / model_name / ("" if variant == "before" else "elimination")
        model_path = out_dir / f"{name}{FORMATS[model_name][1]}"
        meta_path = out_dir / f"{name}.json"

        if variant == "after":
            summary_path = result_dir / "elimination_summary.json"
            if not summary_path.exists():
                print(f"[{name}] no {summary_path} - skipped.")
                continue
            with open(summary_path, encoding="utf-8") as handle:
                summary = json.load(handle)
            if summary["n_eliminated"] == 0:
                print(f"[{name}] feature elimination removed nothing, so this is the same "
                      f"model as {model_name} - not saved twice.")
                continue
            columns = list(summary["selected_features"])
        else:
            columns = None          # all columns, in the CSV's order

        if model_path.exists() and meta_path.exists() and not args.overwrite:
            print(f"[{name}] already saved at {model_path} - skipped (use --overwrite to refit).")
            continue
        todo.append((variant, name, result_dir, model_path, meta_path, columns))

    if not todo:
        return

    print(f"\n{'=' * 70}\n{model_name}: best params {best_params}\n{'=' * 70}", flush=True)

    X, y, groups = load_model_csv(csv_path)
    split = split_train_val_test(
        X, y, groups,
        train_size=HELD_OUT_SIZES[0], val_size=HELD_OUT_SIZES[1],
        test_size=HELD_OUT_SIZES[2], random_state=HELD_OUT_SEED,
    )
    X_trainval, y_trainval = split["X_trainval"], np.asarray(split["y_trainval"])
    X_test, y_test = split["X_test"], np.asarray(split["y_test"])
    n_trainval_matches = len(set(split["groups_trainval"]))
    n_test_matches = len(set(groups)) - n_trainval_matches
    del split
    all_classes = np.unique(np.concatenate([y_trainval, y_test]))

    if args.smoke:
        rng = np.random.default_rng(0)
        train_rows = _smoke_rows(y_trainval, 5000, rng)
        test_rows = _smoke_rows(y_test, 2000, rng)
        X_trainval = X_trainval.iloc[train_rows].reset_index(drop=True)
        y_trainval = y_trainval[train_rows]
        X_test = X_test.iloc[test_rows].reset_index(drop=True)
        y_test = y_test[test_rows]

    for variant, name, result_dir, model_path, meta_path, columns in todo:

        all_columns = columns is None
        if all_columns:
            columns = list(X.columns)
        missing = [c for c in columns if c not in X.columns]
        if missing:
            raise SystemExit(f"[{name}] selected features not in {csv_path}: {missing[:5]}")

        X_fit = X_trainval if all_columns else X_trainval[columns]
        X_eval = X_test if all_columns else X_test[columns]

        params = dict(best_params)
        if args.smoke:
            if model_name == "random_forest":
                params["n_estimators"] = min(int(params["n_estimators"]), 10)
            elif model_name == "xgboost":
                # enough rounds, fast enough learning, for early stopping to
                # trigger, so the saved model's best_iteration is exercised
                params["n_estimators"] = min(int(params["n_estimators"]), 80)
                params["learning_rate"] = max(float(params["learning_rate"]), 0.3)
            elif model_name == "transformer":
                params["max_epochs"] = 1
                params["batch_size"] = 64      # keeps a CPU-only smoke test small

        if model_name == "transformer":
            # Embedding tables are sized from the FULL dataset's columns (all
            # splits), exactly as run_all_models_from_csvs and
            # run_feature_elimination.py did.
            from tabular_transformer import make_transformer_builder
            builder = make_transformer_builder(X if all_columns else X[columns])
        else:
            builder = MODEL_BUILDERS[model_name]

        estimator = builder(params, HELD_OUT_SEED)
        if args.n_jobs and model_name in ("random_forest", "xgboost"):
            estimator.set_params(n_jobs=args.n_jobs)

        print(f"[{name}] refitting on {len(X_fit)} train+val rows x {X_fit.shape[1]} "
              f"columns ...", flush=True)
        started = time.time()
        estimator.fit(X_fit, y_trainval)
        fit_seconds = time.time() - started
        print(f"[{name}] fit took {fit_seconds / 60:.1f} min", flush=True)

        if args.n_jobs and model_name == "random_forest":
            estimator.set_params(n_jobs=None)     # save it with its tuned settings

        proba = estimator.predict_proba(X_eval)
        y_pred = estimator.predict(X_eval)
        metrics = _compute_metrics(y_test, y_pred, proba, all_classes)

        # ---- check 1: does the refit reproduce the saved test predictions? ----
        saved_path = result_dir / f"{model_name}_held_out_predictions.npz"
        saved_result_path = result_dir / f"{model_name}_held_out_test_result.json"
        reproduction = {"reproduces_saved_predictions": None}
        saved_roc_auc = None
        if saved_result_path.exists():
            with open(saved_result_path, encoding="utf-8") as handle:
                saved_roc_auc = float(json.load(handle)["roc_auc"])
        if args.smoke:
            reproduction["note"] = "smoke test: not compared with the saved predictions"
        elif not saved_path.exists():
            reproduction["note"] = f"no saved predictions at {saved_path}"
        else:
            saved = np.load(saved_path)
            if not np.array_equal(saved["y_test"], y_test):
                raise SystemExit(f"[{name}] the recomputed test split differs from "
                                 f"{saved_path} - wrong CSV, seed or split sizes.")
            reproduction.update(_compare(saved["y_pred"], saved["y_proba"], y_pred, proba))
            reproduction["reproduces_saved_predictions"] = bool(
                reproduction["n_test_predictions_different"] == 0
                and reproduction["max_abs_probability_difference"] <= PROBA_TOLERANCE)

        # ---- save, then read it back (check 2) ---------------------------------
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_estimator(model_name, estimator, model_path)

        meta = {
            "name": name,
            "model": model_name,
            "variant": "all features" if variant == "before" else "after feature elimination",
            "file": model_path.name,
            "format": FORMATS[model_name][0],
            "smoke_test": bool(args.smoke),
            "feature_columns": columns,
            "n_features": len(columns),
            "best_params": best_params,
            "params_used": params,
            "seed": HELD_OUT_SEED,
            "class_codes": CLASS_NAMES,
            "trained_on": {
                "data_file": csv_path.name,
                "split": ("match-level GroupShuffleSplit on match_id, test_size 0.2, "
                          "random_state 42 (model_tuning.split_train_val_test); fitted on "
                          "the train+validation part, scored on the test part"),
                "n_train_rows": int(len(X_fit)),
                "n_train_matches": int(n_trainval_matches),
                "n_test_rows": int(len(X_eval)),
                "n_test_matches": int(n_test_matches),
            },
            "test_metrics": {k: float(v) for k, v in metrics.items()},
            "saved_held_out_result": saved_result_path.as_posix(),
            "saved_test_roc_auc": saved_roc_auc,
            **reproduction,
            "fit_seconds": fit_seconds,
            "n_jobs_used_for_fit": args.n_jobs or None,
            "versions": _library_versions(),
            "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        _write_json(meta_path, meta)

        reloaded, reloaded_columns = load_final_model(meta_path)
        reload = _compare(y_pred, proba,
                          reloaded.predict(X_eval[reloaded_columns]),
                          reloaded.predict_proba(X_eval[reloaded_columns]))
        meta["reload_matches"] = bool(reload["n_test_predictions_different"] == 0
                                      and reload["max_abs_probability_difference"] <= PROBA_TOLERANCE)
        meta["reload_check"] = reload
        meta["file_size_bytes"] = model_path.stat().st_size
        meta["sha256"] = _sha256(model_path)
        _write_json(meta_path, meta)
        del reloaded

        size_mb = meta["file_size_bytes"] / 2**20
        verdict = {True: "YES", False: "NO", None: "not checked"}[meta["reproduces_saved_predictions"]]
        print(f"[{name}] saved {model_path} ({size_mb:.1f} MB)\n"
              f"    test ROC-AUC {metrics['roc_auc']:.4f} (saved result: "
              f"{saved_roc_auc if saved_roc_auc is None else round(saved_roc_auc, 4)})\n"
              f"    reproduces the saved test predictions: {verdict}"
              f"{'' if verdict != 'NO' else ' - ' + json.dumps(_plain(reproduction))}\n"
              f"    reloaded file predicts the same: {'YES' if meta['reload_matches'] else 'NO'}",
              flush=True)
        if verdict == "NO":
            print(f"[{name}] WARNING: the refit does not match the original run. Usually a "
                  f"different machine, library version or thread count. The file is saved "
                  f"anyway; say so in the README if you keep it.", flush=True)
        if meta["file_size_bytes"] > GITHUB_SAFE_BYTES:
            print(f"[{name}] {size_mb:.0f} MB is over GitHub's limit - upload it as a "
                  f"release asset (it is listed in {out_dir / '.gitignore'}).", flush=True)


# ----------------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------------

def write_manifest(out_dir: Path) -> None:
    rows = []
    for meta_path in sorted(out_dir.glob("*.json")):
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
        if "format" not in meta or "file" not in meta:
            continue
        model_path = out_dir / meta["file"]
        size = model_path.stat().st_size if model_path.exists() else None
        sha = _sha256(model_path) if model_path.exists() else None
        rows.append({
            "name": meta["name"],
            "model": meta["model"],
            "variant": meta["variant"],
            "file": meta["file"],
            "size_mb": None if size is None else round(size / 2**20, 2),
            "sha256": sha,
            "sha256_ok": None if sha is None else sha == meta.get("sha256"),
            "in_git": None if size is None else size <= GITHUB_SAFE_BYTES,
            "n_features": meta["n_features"],
            "test_roc_auc": round(meta["test_metrics"]["roc_auc"], 6),
            "saved_test_roc_auc": (None if meta.get("saved_test_roc_auc") is None
                                   else round(meta["saved_test_roc_auc"], 6)),
            "reproduces_saved_predictions": meta.get("reproduces_saved_predictions"),
            "n_test_predictions_different": meta.get("n_test_predictions_different"),
            "max_abs_probability_difference": meta.get("max_abs_probability_difference"),
            "reload_matches": meta.get("reload_matches"),
            "smoke_test": meta.get("smoke_test"),
        })

    if not rows:
        print(f"No saved models in {out_dir}.")
        return

    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    too_big = [row["file"] for row in rows if row["in_git"] is False]
    lines = ["# Written by save_final_models.py: model files over 95 MB.",
             "# GitHub rejects files over 100 MB, so these go in a release instead."]
    (out_dir / ".gitignore").write_text("\n".join(lines + too_big) + "\n", encoding="utf-8")

    print(f"\n{'name':<36}{'MB':>9}  {'test AUC':>8}  {'saved':>8}  reproduces  reload  sha ok  in git")
    for row in rows:
        print(f"{row['name']:<36}{row['size_mb'] if row['size_mb'] is not None else 'missing':>9}  "
              f"{row['test_roc_auc']:>8.4f}  "
              f"{row['saved_test_roc_auc'] if row['saved_test_roc_auc'] is not None else '-':>8}  "
              f"{str(row['reproduces_saved_predictions']):<10}  {str(row['reload_matches']):<6}  "
              f"{str(row['sha256_ok']):<6}  {row['in_git']}")
    if too_big:
        print(f"\nUpload as release assets (not committed): {', '.join(too_big)}")
    bad = [row["name"] for row in rows if row["sha256_ok"] is False]
    if bad:
        print(f"WARNING: these files' SHA-256 no longer match what was saved "
              f"(damaged or partial copy?): {', '.join(bad)}")
    print(f"Wrote {out_dir / 'manifest.csv'} and {out_dir / '.gitignore'}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refit the tuned models, check them against the saved test "
                    "predictions and save them (see the module docstring).")
    parser.add_argument("models", nargs="*", choices=MODELS)
    parser.add_argument("--variant", choices=["both", "before", "after"], default="both")
    parser.add_argument("--data-dir", default="tuning_data")
    parser.add_argument("--results-dir", default="tuning_results_n7")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--n-jobs", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / ("smoke_test" if args.smoke else "")

    if not args.manifest_only:
        if not args.models:
            parser.error("name the model(s) to save, or use --manifest-only")
        variants = ["before", "after"] if args.variant == "both" else [args.variant]
        for model_name in args.models:
            run_model(model_name, variants, args, out_dir)

    write_manifest(out_dir)


if __name__ == "__main__":
    main()
