"""
accuracy_over_time.py - accuracy over match progression, before and
after feature elimination, for each model, for one or several window
sizes, in a single run.

Method (per the supervisor's specification)
-------------------------------------------
1. Take the held-out TEST matches only (the ones no model was tuned
   or selected on).
2. Within each match, put the engagements in chronological order
   (round number, then tick within the round). Each engagement is one
   time step, and each model's prediction for it is either correct or
   not.
3. Normalise each match's time to match progression, 0-100%, so
   matches of different lengths are comparable.
4. For each match, compute accuracy at a common set of progression
   points (every --step-pct percent). Accuracy at a point is the
   fraction of correct predictions among the match's engagements
   within +/- half a window of that point. This is the "interpolate
   onto common progression points" step: a single engagement is
   either right or wrong, so accuracy only has meaning over a window,
   and evaluating the window at fixed points puts every match on the
   same axis.
5. At each point, take the MEAN and VARIANCE (population variance,
   as in the worked example) across matches. The plots show the mean
   with a shaded band of +/- one standard deviation across matches.

Window-size experiment
----------------------
By default this runs windows of 10%, 20% and 30% (--window-pct) and
writes everything for each, plus a comparison. The comparison also
reports, per model, how much of the across-match spread is explained
by sampling noise alone: each match's accuracy in a window is a
proportion over only n engagements, so even if every match were
equally predictable it would scatter by about sqrt(acc*(1-acc)/n).
"estimated between-match std" is sqrt(observed variance - that
sampling variance), floored at 0 - a rough estimate (it treats
engagements as independent) of genuine match-to-match differences
in accuracy.

Usage (from the project folder, with the results folder and the raw
engagement_dataset.csv available):

    python accuracy_over_time.py
    python accuracy_over_time.py --window-pct 20
    python accuracy_over_time.py --models xgboost transformer

Reads
    <results-dir>/<model>/<model>_held_out_predictions.npz
        BEFORE elimination (written by run_final_tuning.py)
    <results-dir>/<model>/elimination/<model>_held_out_predictions.npz
        AFTER elimination (written by run_feature_elimination.py);
        models without it are plotted "before" only
    <raw-csv> (data/engagement_dataset.csv) for round order
    <data-dir>/<ref-model>_features.csv for labels and match ids

Writes, to <results-dir>/accuracy_over_time/
    window_<N>pct/accuracy_over_time.csv
    window_<N>pct/accuracy_over_time_<model>.png     (one per model)
    window_<N>pct/accuracy_over_time_all_models.png  (combined)
    window_comparison_<model>.png   same model, windows side by side
    window_size_effect.png          how the spread changes with window
    window_size_summary.csv         the numbers behind that comparison
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_tuning import split_train_val_test

MODELS = ["xgboost", "transformer", "random_forest", "logistic_regression"]

# Must match run_final_tuning.py / run_feature_elimination.py.
HELD_OUT_SIZES = (0.6, 0.2, 0.2)
HELD_OUT_SEED = 42

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "transformer": "Transformer",
    "random_forest": "Random forest",
    "logistic_regression": "Logistic regression",
}

VARIANT_COLOURS = {"before": "#1f4e79", "after": "#e07b00"}
MODEL_COLOURS = {
    "xgboost": "#1f4e79", "transformer": "#e07b00",
    "random_forest": "#2e8b57", "logistic_regression": "#b03060",
}


# ----------------------------------------------------------------
# Locating the test rows in the raw dataset
# ----------------------------------------------------------------

def load_test_rows(raw_csv: Path, ref_csv: Path):
    """
    Returns (test_frame, y_test, majority_class) where test_frame has
    one row per held-out test engagement, in the SAME order the
    models' saved predictions are in, with its match id, round number
    and tick.

    The split is recomputed (it is deterministic: match-level
    GroupShuffleSplit, seed 42), and every assumption it relies on is
    checked rather than trusted.
    """

    raw = pd.read_csv(
        raw_csv,
        usecols=["match_id", "round_num", "start_tick", "engagement_status"],
    )
    ref = pd.read_csv(ref_csv, usecols=["label", "match_id"])

    if len(raw) != len(ref):
        raise SystemExit(
            f"Row count mismatch: {raw_csv} has {len(raw)} rows but "
            f"{ref_csv} has {len(ref)}. The model CSVs must have been "
            f"built from this exact raw dataset."
        )

    if not (raw["match_id"].astype(str).to_numpy()
            == ref["match_id"].astype(str).to_numpy()).all():
        raise SystemExit(
            "match_id differs between the raw dataset and the model "
            "CSV - the rows are not in the same order."
        )

    categories = pd.Categorical(raw["engagement_status"])
    if not (categories.codes == ref["label"].to_numpy()).all():
        raise SystemExit(
            "The model CSV's label column is not the alphabetical "
            "encoding of the raw engagement_status - wrong CSV, or a "
            "binary label_mode file."
        )

    # Adding a row-position column does not change the split, which
    # depends only on the groups and the seed - it just lets us recover
    # which original rows landed in the test set (the split resets the
    # index).
    positions = pd.DataFrame({"__row__": np.arange(len(ref))})

    split = split_train_val_test(
        positions,
        ref["label"].to_numpy(),
        ref["match_id"].to_numpy(),
        train_size=HELD_OUT_SIZES[0],
        val_size=HELD_OUT_SIZES[1],
        test_size=HELD_OUT_SIZES[2],
        random_state=HELD_OUT_SEED,
    )

    rows = split["X_test"]["__row__"].to_numpy()
    y_test = np.asarray(split["y_test"])

    test_frame = raw.iloc[rows][["match_id", "round_num", "start_tick"]]
    test_frame = test_frame.reset_index(drop=True)

    majority_class = int(np.bincount(ref["label"].to_numpy()).argmax())

    return test_frame, y_test, majority_class


# ----------------------------------------------------------------
# Accuracy over progression
# ----------------------------------------------------------------

def per_match_curves(
    test_frame: pd.DataFrame,
    correct: np.ndarray,
    grid: np.ndarray,
    window: float,
    min_engagements: int,
):
    """
    Returns (curves, counts, n_matches_used, n_matches_skipped).
    curves has shape (n_matches_used, len(grid)): each row is one
    match's windowed accuracy at each progression point. counts has
    the same shape: how many engagements fell inside each window
    (needed to say how much of the across-match spread is just
    sampling noise - see summarise).
    """

    frame = test_frame.assign(correct=correct.astype(float))

    # Chronological order within each match. Stable sort so ties keep
    # their original order.
    frame = frame.sort_values(
        ["match_id", "round_num", "start_tick"], kind="mergesort"
    )

    half = window / 2.0
    curves = []
    counts = []
    skipped = 0

    for _, match in frame.groupby("match_id", sort=False):

        n = len(match)

        if n < min_engagements:
            skipped += 1
            continue

        progress = np.arange(n) / (n - 1)
        hits = match["correct"].to_numpy()

        curve = np.full(len(grid), np.nan)
        count = np.zeros(len(grid))

        for i, point in enumerate(grid):
            inside = np.abs(progress - point) <= half + 1e-9
            count[i] = inside.sum()
            if inside.any():
                curve[i] = hits[inside].mean()

        curves.append(curve)
        counts.append(count)

    return np.vstack(curves), np.vstack(counts), len(curves), skipped


def summarise(curves: np.ndarray, counts: np.ndarray):
    """
    Mean, population variance and std across matches, per point, plus
    the std you would expect from sampling noise ALONE and a rough
    estimate of the genuine between-match std.

    Each match's accuracy at a point is a proportion over only
    counts[m, p] engagements, so even if every match were equally
    predictable its accuracy would scatter around the mean with
    standard deviation about sqrt(acc * (1 - acc) / n). The estimated
    between-match std removes that from the observed variance
    (floored at zero). It treats engagements as independent, so it is
    approximate.
    """
    mean = np.nanmean(curves, axis=0)
    variance = np.nanvar(curves, axis=0)       # ddof=0, as in the example
    n_eff = counts.mean(axis=0)
    sampling_var = mean * (1.0 - mean) / np.maximum(n_eff, 1.0)
    between = np.sqrt(np.maximum(variance - sampling_var, 0.0))
    return {
        "mean": mean,
        "variance": variance,
        "std": np.sqrt(variance),
        "expected_sampling_std": np.sqrt(sampling_var),
        "est_between_match_std": between,
    }


# ----------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------

def draw(ax, grid_pct, stats, label, colour, linestyle="-"):
    mean, std = stats["mean"], stats["std"]
    ax.plot(grid_pct, mean, linestyle, color=colour, linewidth=2, label=label)
    ax.fill_between(
        grid_pct,
        np.clip(mean - std, 0, 1),
        np.clip(mean + std, 0, 1),
        color=colour, alpha=0.15, linewidth=0,
    )


def style(ax, title):
    ax.set_xlabel("Match progression (%)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)


# ----------------------------------------------------------------
# One window size: compute everything
# ----------------------------------------------------------------

def compute_window(test_frame, y_test, majority_class, results_dir, models,
                   grid, window, min_engagements, step_pct):
    """Returns (baseline_stats, results, n_used, notes) for one window."""

    baseline_correct = (y_test == majority_class)
    curves, counts, n_used, skipped = per_match_curves(
        test_frame, baseline_correct, grid, window, min_engagements)
    baseline_stats = summarise(curves, counts)

    results = {}
    notes = []

    if skipped:
        notes.append(f"Skipped {skipped} matches with fewer than "
                     f"{min_engagements} engagements.")

    for model in models:
        for variant, folder in (("before", results_dir / model),
                                ("after", results_dir / model / "elimination")):

            path = folder / f"{model}_held_out_predictions.npz"

            if not path.exists():
                notes.append(f"[{model}] no {variant}-elimination "
                             f"predictions yet ({path}).")
                continue

            saved = np.load(path)

            if not np.array_equal(saved["y_test"], y_test):
                raise SystemExit(
                    f"[{model}/{variant}] saved y_test does not match the "
                    f"recomputed test split - the CSVs, seed or split sizes "
                    f"differ from the run that produced {path}."
                )

            correct = (saved["y_pred"] == y_test)

            curves, counts, n_used, _ = per_match_curves(
                test_frame, correct, grid, window, min_engagements)
            stats = summarise(curves, counts)

            label = MODEL_LABELS[model]
            if variant == "after":
                summary_path = folder / "elimination_summary.json"
                if summary_path.exists():
                    with open(summary_path) as file:
                        info = json.load(file)
                    label += (f" - after elimination "
                              f"({info['n_eliminated']} of "
                              f"{info['n_features_before']} features removed)")
                else:
                    label += " - after elimination"
            else:
                label += " - all features"

            results[(model, variant)] = {
                "stats": stats, "label": label,
                "overall_accuracy": float(correct.mean()),
            }

    return baseline_stats, results, n_used, notes


# ----------------------------------------------------------------
# One window size: write csv + per-model + combined figures
# ----------------------------------------------------------------

def write_window_outputs(out_dir, results_dir, models, grid_pct, window_pct,
                         baseline_stats, results, n_used, no_combined):

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    def add_rows(model, variant, stats):
        for i, pct in enumerate(grid_pct):
            rows.append({
                "model": model, "variant": variant,
                "window_pct": window_pct, "progress_pct": pct,
                "mean_accuracy": stats["mean"][i],
                "variance": stats["variance"][i],
                "std": stats["std"][i],
                "expected_sampling_std": stats["expected_sampling_std"][i],
                "est_between_match_std": stats["est_between_match_std"][i],
                "n_matches": n_used,
            })

    add_rows("majority_class", "baseline", baseline_stats)
    for (model, variant), entry in results.items():
        add_rows(model, variant, entry["stats"])

    pd.DataFrame(rows).to_csv(out_dir / "accuracy_over_time.csv", index=False)

    # One figure per model: before vs after.
    for model in models:
        present = [v for v in ("before", "after") if (model, v) in results]
        if not present:
            continue

        with open(results_dir / model / f"{model}_final_result.json") as file:
            best_params = json.load(file)["best_params"]

        fig, ax = plt.subplots(figsize=(8, 5.6))
        for variant in present:
            entry = results[(model, variant)]
            draw(ax, grid_pct, entry["stats"], entry["label"],
                 VARIANT_COLOURS[variant])
        draw(ax, grid_pct, baseline_stats,
             "Always predict most common class", "#777777", "--")
        style(ax, f"{MODEL_LABELS[model]}: accuracy over match progression\n"
                  f"(mean +/- 1 std across {n_used} held-out matches, "
                  f"{window_pct:g}% window)")
        ax.legend(loc="lower right", fontsize=8)

        caption = "Best hyperparameters (tuning_results_n7): " + ", ".join(
            f"{key}={value}" for key, value in best_params.items())
        fig.text(0.5, 0.012, textwrap.fill(caption, 95), ha="center",
                 va="bottom", fontsize=7, color="#444444")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(out_dir / f"accuracy_over_time_{model}.png", dpi=200)
        plt.close(fig)

    if no_combined:
        return

    # One combined figure: all models, before (and after if present).
    variants = [v for v in ("before", "after")
                if any(k[1] == v for k in results)]
    fig, axes = plt.subplots(1, len(variants), figsize=(7 * len(variants), 5),
                             sharey=True, squeeze=False)

    for ax, variant in zip(axes[0], variants):
        for model in models:
            if (model, variant) in results:
                draw(ax, grid_pct, results[(model, variant)]["stats"],
                     MODEL_LABELS[model], MODEL_COLOURS[model])
        draw(ax, grid_pct, baseline_stats,
             "Always predict most common class", "#777777", "--")
        style(ax, "All features" if variant == "before"
                  else "After feature elimination")
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"Accuracy over match progression (mean +/- 1 std across "
                 f"{n_used} held-out matches, {window_pct:g}% window)")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_over_time_all_models.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------
# Window-size comparison
# ----------------------------------------------------------------

def write_comparison(out_dir, models, grid_pct, windows, per_window, n_used):

    summary_rows = []

    for window_pct in windows:
        baseline_stats, results = per_window[window_pct]
        for (model, variant), entry in results.items():
            stats = entry["stats"]
            mean = stats["mean"]
            summary_rows.append({
                "window_pct": window_pct,
                "model": model, "variant": variant,
                "overall_accuracy": entry["overall_accuracy"],
                "mean_accuracy_first_20pct":
                    float(np.nanmean(mean[grid_pct <= 20])),
                "mean_accuracy_last_20pct":
                    float(np.nanmean(mean[grid_pct >= 80])),
                "mean_across_match_std": float(np.nanmean(stats["std"])),
                "mean_expected_sampling_std":
                    float(np.nanmean(stats["expected_sampling_std"])),
                "mean_est_between_match_std":
                    float(np.nanmean(stats["est_between_match_std"])),
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "window_size_summary.csv", index=False)

    if len(windows) < 2:
        return summary

    # Per model: the windows side by side.
    for model in models:
        if not any((model, v) in per_window[windows[0]][1]
                   for v in ("before", "after")):
            continue

        fig, axes = plt.subplots(1, len(windows), figsize=(6 * len(windows), 5),
                                 sharey=True, squeeze=False)

        for ax, window_pct in zip(axes[0], windows):
            baseline_stats, results = per_window[window_pct]
            for variant in ("before", "after"):
                if (model, variant) in results:
                    entry = results[(model, variant)]
                    draw(ax, grid_pct, entry["stats"], entry["label"],
                         VARIANT_COLOURS[variant])
            draw(ax, grid_pct, baseline_stats,
                 "Always predict most common class", "#777777", "--")
            style(ax, f"{window_pct:g}% window")
            ax.legend(loc="lower right", fontsize=7)

        fig.suptitle(f"{MODEL_LABELS[model]}: effect of window size "
                     f"(mean +/- 1 std across {n_used} held-out matches)")
        fig.tight_layout()
        fig.savefig(out_dir / f"window_comparison_{model}.png", dpi=200)
        plt.close(fig)

    # How the across-match spread changes with window size.
    fig, (ax_obs, ax_est) = plt.subplots(1, 2, figsize=(12, 4.8))

    for model in models:
        for variant, linestyle in (("before", "-"), ("after", "-.")):
            part = summary[(summary["model"] == model)
                           & (summary["variant"] == variant)]
            if part.empty:
                continue
            label = MODEL_LABELS[model] + (
                "" if variant == "before" else " (after elimination)")
            ax_obs.plot(part["window_pct"], part["mean_across_match_std"],
                        linestyle, marker="o", color=MODEL_COLOURS[model],
                        label=label)
            ax_est.plot(part["window_pct"], part["mean_est_between_match_std"],
                        linestyle, marker="o", color=MODEL_COLOURS[model],
                        label=label)

    expected = summary.groupby("window_pct")["mean_expected_sampling_std"].mean()
    ax_obs.plot(expected.index, expected.values, "--", color="#777777",
                marker="s", label="Expected from sampling noise alone")

    ax_obs.set_title("Observed across-match std")
    ax_est.set_title("Estimated genuine between-match std\n"
                     "(observed variance minus sampling variance)")
    for ax in (ax_obs, ax_est):
        ax.set_xlabel("Window size (% of match)")
        ax.set_ylabel("Std of accuracy across matches")
        ax.set_xticks(windows)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "window_size_effect.png", dpi=200)
    plt.close(fig)

    return summary


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--results-dir", default="tuning_results_n7")
    parser.add_argument("--data-dir", default="tuning_data")
    parser.add_argument("--raw-csv", default="data/engagement_dataset.csv")
    parser.add_argument("--ref-model", default="xgboost",
                        help="any model whose <name>_features.csv is in "
                             "--data-dir; only its label/match_id are used")
    parser.add_argument("--step-pct", type=float, default=5.0,
                        help="spacing of the common progression points")
    parser.add_argument("--window-pct", type=float, nargs="+",
                        default=[10.0, 20.0, 30.0],
                        help="window width(s) around each point; several "
                             "values run the window-size experiment")
    parser.add_argument("--min-engagements", type=int, default=30,
                        help="skip test matches shorter than this")
    parser.add_argument("--no-combined", action="store_true",
                        help="skip the combined all-models figure")
    args = parser.parse_args()

    windows = sorted(set(args.window_pct))
    results_dir = Path(args.results_dir)
    out_dir = results_dir / "accuracy_over_time"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_frame, y_test, majority_class = load_test_rows(
        Path(args.raw_csv),
        Path(args.data_dir) / f"{args.ref_model}_features.csv",
    )

    print(f"{len(test_frame)} test engagements across "
          f"{test_frame['match_id'].nunique()} held-out matches")

    grid_pct = np.arange(0.0, 100.0 + 1e-9, args.step_pct)
    grid = grid_pct / 100.0

    per_window = {}
    n_used = None

    for window_pct in windows:

        baseline_stats, results, n_used, notes = compute_window(
            test_frame, y_test, majority_class, results_dir, args.models,
            grid, window_pct / 100.0, args.min_engagements, args.step_pct)

        if not results:
            raise SystemExit("No prediction files found - nothing to plot.")

        per_window[window_pct] = (baseline_stats, results)

        print(f"\n--- {window_pct:g}% window ({n_used} matches) ---")
        if window_pct == windows[0]:
            for note in sorted(set(notes)):
                print(note)

        for (model, variant), entry in results.items():
            stats = entry["stats"]
            print(f"[{model}/{variant}] accuracy "
                  f"{entry['overall_accuracy']:.4f} | across-match std "
                  f"{np.nanmean(stats['std']):.3f} vs "
                  f"{np.nanmean(stats['expected_sampling_std']):.3f} "
                  f"from sampling noise alone -> estimated genuine "
                  f"between-match std "
                  f"{np.nanmean(stats['est_between_match_std']):.3f}")

        write_window_outputs(
            out_dir / f"window_{window_pct:g}pct", results_dir, args.models,
            grid_pct, window_pct, baseline_stats, results, n_used,
            args.no_combined)

    write_comparison(out_dir, args.models, grid_pct, windows, per_window,
                     n_used)

    print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()
