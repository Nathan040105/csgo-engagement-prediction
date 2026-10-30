"""
feature_importance_analysis.py - which features, and which KINDS of feature,
drive each model's predictions?

Answers research question 2 ("which gameplay features contribute most
strongly?") from the permutation importances the final sweep already saved.
Needs only pandas, numpy and matplotlib: no model refits, no raw data, no GPU,
no Ray. Runs in a few seconds on any laptop.

Usage (PowerShell or bash), from the folder that contains tuning_results_n7:

    python feature_importance_analysis.py --results-dir tuning_results_n7

Reads, for each model (xgboost, transformer, random_forest, logistic_regression):
    <results>/<model>/<model>_feature_importance.csv
    <results>/<model>/elimination/elimination_summary.json      (optional)

Writes to <results>/feature_importance_analysis/ (or --out-dir):
    feature_categories.csv        every feature and the category it was put in
    importance_by_feature.csv     importance and rank per feature, one column per model
    importance_by_category.csv    summed importance and % share per category per model
    importance_by_owner.csv       the same by whose feature it is: attacker, victim, or shared
    top_features.csv              each model's top-N features
    consensus_top_features.csv    features ranked by their average rank across models
    model_agreement.csv           Spearman correlation and top-N overlap between models
    retention_by_category.csv     % of each category kept by feature elimination
    category_importance_share.png   top_features.png   category_retention.png
    summary.txt                   the same numbers as printed in the terminal

WHAT THE NUMBERS ARE (read before quoting them)
-----------------------------------------------
* Unit: the DROP IN MACRO-F1 when one feature's values are shuffled among the
  rows of a held-out fold (permutation importance). 0.010 = shuffling that
  feature costs one hundredth of macro-F1. model_tuning.run_feature_importance_
  analysis uses macro-F1 for a multiclass target (ROC-AUC OvR can be undefined
  on a fold that misses a class). Its own *_feature_importance.png charts are
  titled "ROC-AUC drop", which is the wrong label for these files; the charts
  made here are labelled correctly.
* Measured on the sweep's VALIDATION folds, not on the final test matches: in
  each repeat the model is refit on a random 75% of the 80% train+val matches
  and scored on the other 25% of matches, then averaged over the repeats
  (7 for logistic regression / random forest / XGBoost, 3 for the transformer,
  2 shuffles each). The 20% test matches are never touched.
* Negative values (shuffling helped by chance) are set to 0 before anything is
  summed, so noise cannot cancel real importance.
* Logistic regression sees one-hot columns (549). They are folded back to the
  original feature by summing. That adds up per-column drops, which is only an
  approximation of shuffling the whole categorical variable at once (and the
  std shown for those features treats the columns as independent).
* Importances are not additive and are not causal. Strongly correlated or
  duplicated features share credit (each looks less important than the pair
  really is), and shuffling ONE of two mirror-image features (attacker_side and
  victim_side) builds rows that never occur in real data, which can inflate
  both. grouped_permutation_importance.py checks both effects directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

MODELS = ["xgboost", "transformer", "random_forest", "logistic_regression"]

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "transformer": "Transformer",
    "random_forest": "Random forest",
    "logistic_regression": "Logistic regression",
}

# Same colours as accuracy_over_time.py, so a model looks the same in every
# figure of the report.
MODEL_COLOURS = {
    "xgboost": "#1f4e79", "transformer": "#e07b00",
    "random_forest": "#2e8b57", "logistic_regression": "#b03060",
}

# ----------------------------------------------------------------------------
# Feature categories
#
# Every one of the 159 model inputs is put in exactly one category. The names
# are neutral; PROPOSAL_MAP says where each sits in the proposal's wording
# (spatial / temporal / combat / contextual). Two honest differences from the
# proposal: there is no acceleration feature, so "temporal" is velocity and
# posture only; and nav-mesh geometry and ping / LAN are their own groups.
# ----------------------------------------------------------------------------

CATEGORIES = [
    "Spatial",
    "Movement",
    "Combat state",
    "Economy & utility",
    "Round & team context",
    "Map geometry (nav mesh)",
    "Connection / other",
]

CATEGORY_SHORT = {
    "Spatial": "Spatial",
    "Movement": "Movement",
    "Combat state": "Combat",
    "Economy & utility": "Economy",
    "Round & team context": "Round/team",
    "Map geometry (nav mesh)": "Nav mesh",
    "Connection / other": "Connection",
}

PROPOSAL_MAP = {
    "Spatial": "Spatial",
    "Movement": "Temporal (velocity and posture only; no acceleration)",
    "Combat state": "Combat",
    "Economy & utility": "Contextual (team economy)",
    "Round & team context": "Contextual (round time, bomb status, map)",
    "Map geometry (nav mesh)": "Spatial (positional advantage / cover)",
    "Connection / other": "not in the proposal",
}

# Per-player features are named attacker_<suffix> / victim_<suffix>.
PLAYER_SUFFIX = {
    "Spatial": {
        "x", "y", "z", "view_x", "view_y",
        "closest_enemy", "closest_enemy_missing",
        "closest_teammate", "closest_teammate_missing",
        "average_enemy_distance", "average_enemy_distance_missing",
        "average_teammate_distance", "average_teammate_distance_missing",
    },
    "Movement": {"speed", "ducking", "standing", "walking", "airborne"},
    "Combat state": {
        "hp", "armor", "has_helmet", "active_weapon", "active_weapon_class",
        "primary_weapon", "secondary_weapon", "magazine_pct",
        "magazine_pct_missing", "reserve_pct", "reserve_pct_missing",
        "reloading", "scoped", "zoom_level", "blinded",
    },
    "Economy & utility": {
        "cash", "equipment", "inventory_size", "flashbangs", "smokes",
        "hegrenades", "molotovs", "incendiaries", "decoys", "has_flash",
        "has_smoke", "has_he", "has_molotov", "has_incendiary", "has_decoy",
        "has_zeus", "total_utility",
    },
    "Round & team context": {
        "side", "alive_enemies", "alive_teammates", "has_bomb", "planting",
        "defusing", "bombzone", "buyzone",
    },
    "Map geometry (nav mesh)": {
        "callout", "area_connectivity", "area_connectivity_missing",
        "near_cover", "exposed",
    },
    "Connection / other": {"ping", "ping_missing"},
}

# Features that belong to the engagement as a whole (not to one player).
SHARED = {
    "Spatial": {
        "delta_x", "delta_y", "delta_z", "height_advantage",
        "horizontal_angle_difference", "vertical_angle_difference",
        "engagement_distance",
    },
    "Movement": {"speed_difference"},
    "Combat state": {"hp_difference", "armor_difference"},
    "Economy & utility": {
        "t_equipment", "ct_equipment", "t_total_utility", "ct_total_utility",
        "equipment_difference", "utility_difference",
    },
    "Round & team context": {
        "start_time", "clock_time_seconds", "bomb_planted", "map_name",
        "t_alive", "ct_alive", "alive_difference",
    },
    "Map geometry (nav mesh)": {
        "same_callout", "nav_path_distance", "nav_los_proxy_ratio",
        "nav_path_distance_missing", "nav_los_proxy_ratio_missing",
    },
    "Connection / other": {"is_lan"},
}

# The categorical features that logistic regression sees one-hot encoded
# (columns named <feature>_<level>); the other three models see them as one
# ordinal column each.
CATEGORICAL_BASES = [
    "attacker_active_weapon", "attacker_active_weapon_class",
    "attacker_callout", "attacker_primary_weapon",
    "attacker_secondary_weapon", "attacker_side",
    "victim_active_weapon", "victim_active_weapon_class",
    "victim_callout", "victim_primary_weapon",
    "victim_secondary_weapon", "victim_side",
    "map_name",
]


# The two side columns are mirror images of each other (98.2% of rows) and are
# implied by the alive-teammate counts, so they get a robustness check of their
# own (see category_rows / grouped_permutation_importance.py).
SIDE_BASES = ("attacker_side", "victim_side")


def base_of(column: str, known_bases=CATEGORICAL_BASES) -> str:
    """The original feature a model input column came from.

    One-hot columns are named <feature>_<level> (e.g. attacker_side_CT), so the
    longest categorical feature name that prefixes the column is its source.
    Every other column is its own feature.
    """
    if column in known_bases:
        return column
    best = None
    for base in known_bases:
        if column.startswith(base + "_") and (best is None or len(base) > len(best)):
            best = base
    return best if best is not None else column


def category_of(base: str) -> str:
    """Category of an original feature; raises if the feature is not mapped."""
    for category, names in SHARED.items():
        if base in names:
            return category
    for prefix in ("attacker_", "victim_"):
        if base.startswith(prefix):
            suffix = base[len(prefix):]
            for category, names in PLAYER_SUFFIX.items():
                if suffix in names:
                    return category
    raise KeyError(
        f"feature {base!r} has no category - add it to PLAYER_SUFFIX or "
        f"SHARED in feature_importance_analysis.py"
    )


# ----------------------------------------------------------------------------
# Loading and aggregating
# ----------------------------------------------------------------------------

def load_importance(results_dir: Path, model: str) -> pd.DataFrame | None:
    path = results_dir / model / f"{model}_feature_importance.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    missing = {"feature", "importance_mean", "importance_std"} - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns {sorted(missing)}")
    return frame


def load_elimination(results_dir: Path, model: str) -> dict | None:
    path = results_dir / model / "elimination" / "elimination_summary.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def to_original_features(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per ORIGINAL feature, importance floored at 0 and summed."""
    work = frame.copy()
    work["base"] = [base_of(column) for column in work["feature"]]
    work["positive"] = work["importance_mean"].clip(lower=0.0)
    work["variance"] = work["importance_std"] ** 2

    grouped = work.groupby("base", sort=False)
    out = pd.DataFrame({
        "importance": grouped["positive"].sum(),
        "importance_signed": grouped["importance_mean"].sum(),
        "std": np.sqrt(grouped["variance"].sum()),
        "n_columns": grouped["feature"].size(),
    }).reset_index()
    out["category"] = [category_of(base) for base in out["base"]]
    out = out.sort_values(["importance", "base"], ascending=[False, True])
    return out.reset_index(drop=True)


OWNERS = ["attacker", "victim", "shared"]


def owner_of(base: str) -> str:
    """Whose feature it is: attacker_* , victim_* , or shared (a difference or distance
    between the two players, a team total, or round context such as the clock)."""
    return "attacker" if base.startswith("attacker_") else "victim" if base.startswith("victim_") else "shared"


def owner_rows(model: str, features: pd.DataFrame) -> list[dict]:
    """Importance by owner. The two side columns are an attacker and a victim feature,
    so they are also reported set aside (see the side-column note in the docstring)."""
    total = features["importance"].sum()
    without_sides = features[~features["base"].isin(SIDE_BASES)]
    total_without_sides = without_sides["importance"].sum()
    owner = features["base"].map(owner_of)
    owner_without_sides = without_sides["base"].map(owner_of)
    rows = []
    for name in OWNERS:
        part = features[owner == name]
        part_without_sides = without_sides[owner_without_sides == name]
        summed = float(part["importance"].sum())
        summed_without_sides = float(part_without_sides["importance"].sum())
        rows.append({
            "model": model,
            "owner": name,
            "n_features": int(len(part)),
            "importance_sum": summed,
            "share_pct": 100.0 * summed / total if total > 0 else np.nan,
            "n_features_excl_sides": int(len(part_without_sides)),
            "share_excl_sides_pct": (100.0 * summed_without_sides / total_without_sides
                                     if total_without_sides > 0 else np.nan),
            "importance_per_feature_excl_sides": (summed_without_sides / len(part_without_sides)
                                                  if len(part_without_sides) else np.nan),
        })
    return rows


def category_rows(model: str, features: pd.DataFrame) -> list[dict]:
    total = features["importance"].sum()
    without_sides = features[~features["base"].isin(SIDE_BASES)]
    total_without_sides = without_sides["importance"].sum()
    rows = []
    for category in CATEGORIES:
        part = features[features["category"] == category]
        summed = float(part["importance"].sum())
        summed_without_sides = float(
            without_sides.loc[without_sides["category"] == category, "importance"].sum())
        rows.append({
            "model": model,
            "category": category,
            "proposal_category": PROPOSAL_MAP[category],
            "n_features": int(len(part)),
            "feature_share_pct": 100.0 * len(part) / len(features),
            "importance_sum": summed,
            "importance_std": float(np.sqrt((part["std"] ** 2).sum())),
            "share_pct": 100.0 * summed / total if total > 0 else np.nan,
            "share_excl_sides_pct": (100.0 * summed_without_sides / total_without_sides
                                     if total_without_sides > 0 else np.nan),
            "importance_per_feature": summed / len(part) if len(part) else np.nan,
        })
    return rows


def retention_rows(model: str, frame: pd.DataFrame, elimination: dict) -> list[dict]:
    """Share of each category's features that survived feature elimination."""
    selected = set(elimination["selected_features"])
    all_columns = list(frame["feature"])

    if not selected <= set(all_columns):
        extra = sorted(selected - set(all_columns))[:5]
        raise SystemExit(
            f"[{model}] elimination kept columns that are not in the "
            f"importance file (e.g. {extra}) - the two files are from "
            f"different runs."
        )

    table = pd.DataFrame({"column": all_columns})
    table["base"] = [base_of(column) for column in table["column"]]
    table["category"] = [category_of(base) for base in table["base"]]
    table["kept"] = table["column"].isin(selected)

    per_base = table.groupby("base").agg(
        category=("category", "first"),
        kept=("kept", "any"),
        columns=("column", "size"),
        columns_kept=("kept", "sum"),
    )

    rows = []
    for category in CATEGORIES:
        part = per_base[per_base["category"] == category]
        n_features = int(len(part))
        n_kept = int(part["kept"].sum())
        rows.append({
            "model": model,
            "category": category,
            "n_features": n_features,
            "n_features_kept": n_kept,
            "pct_features_kept": 100.0 * n_kept / n_features if n_features else np.nan,
            "n_columns": int(part["columns"].sum()),
            "n_columns_kept": int(part["columns_kept"].sum()),
        })
    return rows


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAR = "#2a78d6"

# One-hue sequential ramp (light = near zero, dark = large).
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def _luminance(rgb) -> float:
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _titles(fig, fig_w, fig_h, title, subtitle, footnote=None):
    fig.text(0.35 / fig_w, 1 - 0.32 / fig_h, title, ha="left", va="center",
             fontsize=13, fontweight="bold", color=INK)
    fig.text(0.35 / fig_w, 1 - 0.62 / fig_h, subtitle, ha="left", va="center",
             fontsize=9, color=INK2)
    if footnote:
        fig.text(0.35 / fig_w, 0.30 / fig_h, footnote, ha="left", va="center",
                 fontsize=8, color=MUTED)


def heatmap(path, matrix, row_labels, col_labels, text_fn, title, subtitle,
            footnote=None, vmin=0.0, vmax=None):
    """Annotated one-hue heatmap; every cell carries its own number."""
    n_rows, n_cols = matrix.shape
    extra = 0.17 * (footnote.count("\n") if footnote else 0)   # inches for extra footnote lines
    fig_w = 1.7 * n_cols + 3.3
    fig_h = 0.62 * n_rows + 2.15 + extra
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    vmax = float(np.nanmax(matrix)) if vmax is None else vmax
    norm = plt.Normalize(vmin, vmax)
    ax.imshow(matrix, cmap=SEQ, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=10, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10, color=INK)

    for i in range(n_rows):
        for j in range(n_cols):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            rgb = to_rgb(SEQ(norm(value)))
            colour = "#ffffff" if _luminance(rgb) < 0.18 else INK
            ax.text(j, i, text_fn(i, j), ha="center", va="center",
                    fontsize=10, color=colour)

    fig.subplots_adjust(left=2.55 / fig_w, right=1 - 0.3 / fig_w,
                        top=1 - 1.15 / fig_h, bottom=(0.55 + extra) / fig_h)
    _titles(fig, fig_w, fig_h, title, subtitle, footnote)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_category_share(path, shares: pd.DataFrame, counts: dict, models: list[str],
                        side_share: dict | None = None):
    order = shares.mean(axis=1).sort_values(ascending=False).index.tolist()
    matrix = shares.loc[order, models].to_numpy(dtype=float)
    rows = [f"{c}  ({counts[c]})" for c in order]
    footnote = ("(n) = input features in the category. Importance is measured on validation "
                "folds inside the train+val matches, not on the test matches.")
    # Say so when two mirror-image columns carry a large part of one model's total,
    # because that inflates the category they sit in (see the side check in the README).
    flagged = [m for m in models if side_share and side_share.get(m, 0.0) >= 5.0]
    if flagged:
        side_category = category_of(SIDE_BASES[0])
        footnote += "\n" + "; ".join(
            f"{MODEL_LABELS[m]}: attacker_side + victim_side alone hold {side_share[m]:.1f}% "
            f"of its importance, all counted under {side_category}" for m in flagged) + "."
    heatmap(
        path, matrix, rows, [MODEL_LABELS[m] for m in models],
        text_fn=lambda i, j: f"{matrix[i, j]:.1f}%",
        title="Which kinds of feature drive each model",
        subtitle="Share of each model's total permutation importance (macro-F1 drop), "
                 "summed over the features in a category",
        footnote=footnote,
    )


def plot_retention(path, retention: pd.DataFrame, counts: dict, models: list[str]):
    pct = retention.pivot(index="category", columns="model", values="pct_features_kept")
    kept = retention.pivot(index="category", columns="model", values="n_features_kept")
    total = retention.pivot(index="category", columns="model", values="n_features")
    order = [c for c in CATEGORIES if c in pct.index]
    matrix = pct.loc[order, models].to_numpy(dtype=float)
    rows = [f"{c}  ({counts[c]})" for c in order]

    def text(i, j):
        category, model = order[i], models[j]
        return (f"{matrix[i, j]:.0f}%\n{int(kept.loc[category, model])}"
                f"/{int(total.loc[category, model])}")

    heatmap(
        path, matrix, rows, [MODEL_LABELS[m] for m in models], text_fn=text,
        title="What feature elimination kept, by category",
        subtitle="Percentage of each category's features still in the model after backward "
                 "elimination (kept / total below)",
        footnote="Logistic regression: a categorical feature counts as kept if any of its "
                 "one-hot columns survived. Random forest: elimination stopped at step 1 "
                 "(nothing removed).",
        vmin=0.0, vmax=100.0,
    )


def plot_top_features(path, top_by_model: dict[str, pd.DataFrame], n: int, models: list[str]):
    fig_w, fig_h = 15.0, 0.34 * n * 2 + 3.4
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), facecolor=SURFACE)

    for ax, model in zip(axes.ravel(), models):
        top = top_by_model[model].head(n).iloc[::-1].reset_index(drop=True)
        y = np.arange(len(top))
        ax.set_facecolor(SURFACE)
        ax.barh(y, top["importance"], height=0.62, color=BAR, zorder=2)
        ax.errorbar(top["importance"], y, xerr=top["std"], fmt="none",
                    ecolor=INK2, elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=3)

        reach = float((top["importance"] + top["std"]).max())
        ax.set_xlim(0, reach * 1.32)
        for yi, row in top.iterrows():
            ax.text(row["importance"] + row["std"] + reach * 0.015, yi,
                    CATEGORY_SHORT[row["category"]], va="center", ha="left",
                    fontsize=8, color=MUTED, zorder=4)

        ax.set_yticks(y)
        ax.set_yticklabels(top["base"], fontsize=9, color=INK)
        ax.set_title(MODEL_LABELS[model], loc="left", fontsize=11,
                     fontweight="bold", color=INK, pad=8)
        ax.tick_params(axis="x", labelsize=8.5, colors=INK2, length=0)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.set_xlabel("Macro-F1 drop when shuffled (mean ± std over repeats)",
                      fontsize=8.5, color=INK2)

    fig.subplots_adjust(left=0.2, right=0.985, top=1 - 1.15 / fig_h,
                        bottom=1.15 / fig_h, wspace=0.62, hspace=0.32)
    _titles(fig, fig_w, fig_h, f"Top {n} features by permutation importance, per model",
            "Grey word = the feature's category. Each panel has its own x-scale; "
            "compare the ranking within a panel, not bar lengths across panels.",
            "Validation folds inside the train+val matches (not the test matches). "
            "Logistic regression one-hot columns are summed back to the original feature.")
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-dir", default="tuning_results_n7",
                        help="folder holding <model>/<model>_feature_importance.csv")
    parser.add_argument("--out-dir", default=None,
                        help="default: <results-dir>/feature_importance_analysis")
    parser.add_argument("--top-n", type=int, default=20,
                        help="features per model in top_features.csv and the overlap test")
    parser.add_argument("--plot-top-n", type=int, default=15,
                        help="features per model drawn in top_features.png")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise SystemExit(f"{results_dir} is not a folder")
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "feature_importance_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    # ---- load -------------------------------------------------------------
    raw, features, eliminations = {}, {}, {}
    for model in MODELS:
        frame = load_importance(results_dir, model)
        if frame is None:
            print(f"[skip] no feature importance file for {model}")
            continue
        raw[model] = frame
        features[model] = to_original_features(frame)
        eliminations[model] = load_elimination(results_dir, model)

    models = [m for m in MODELS if m in features]
    if not models:
        raise SystemExit(f"no *_feature_importance.csv files found under {results_dir}")

    all_features = sorted(set().union(*[set(f["base"]) for f in features.values()]))
    for model in models:
        missing = set(all_features) - set(features[model]["base"])
        if missing:
            raise SystemExit(f"[{model}] is missing features other models have: {sorted(missing)[:5]}")

    categories = pd.DataFrame({"feature": all_features})
    categories["category"] = [category_of(f) for f in all_features]
    categories["proposal_category"] = categories["category"].map(PROPOSAL_MAP)
    categories["owner"] = [owner_of(f) for f in all_features]
    categories = categories.sort_values(["category", "feature"]).reset_index(drop=True)
    categories.to_csv(out_dir / "feature_categories.csv", index=False)
    counts = categories["category"].value_counts().to_dict()

    say(f"Results folder: {results_dir}")
    say(f"Models found: {', '.join(MODEL_LABELS[m] for m in models)}")
    say(f"Original features: {len(all_features)}  "
        f"(attacker {sum(categories.owner == 'attacker')}, "
        f"victim {sum(categories.owner == 'victim')}, "
        f"shared {sum(categories.owner == 'shared')})")
    say("Features per category: " + ", ".join(f"{c} {counts.get(c, 0)}" for c in CATEGORIES))
    say()
    for model in models:
        positive = int((raw[model]["importance_mean"] > 0).sum())
        say(f"{MODEL_LABELS[model]:<20} input columns {len(raw[model]):>4}   "
            f"with importance > 0: {positive:>4}   "
            f"total (floored) importance {features[model]['importance'].sum():.4f}")

    # ---- per feature ------------------------------------------------------
    wide = pd.DataFrame({"feature": all_features}).merge(
        categories[["feature", "category"]], on="feature")
    for model in models:
        part = features[model][["base", "importance", "std"]].rename(columns={
            "base": "feature",
            "importance": f"{model}_importance",
            "std": f"{model}_std",
        })
        part[f"{model}_rank"] = part[f"{model}_importance"].rank(ascending=False, method="min")
        wide = wide.merge(part, on="feature", how="left")
    rank_columns = [f"{m}_rank" for m in models]
    wide["mean_rank"] = wide[rank_columns].mean(axis=1)
    wide["best_rank"] = wide[rank_columns].min(axis=1)
    wide["worst_rank"] = wide[rank_columns].max(axis=1)
    wide = wide.sort_values("mean_rank").reset_index(drop=True)
    wide.to_csv(out_dir / "importance_by_feature.csv", index=False)

    # ---- per category -----------------------------------------------------
    category_table = pd.DataFrame(
        [row for model in models for row in category_rows(model, features[model])])
    category_table.to_csv(out_dir / "importance_by_category.csv", index=False)
    shares = category_table.pivot(index="category", columns="model", values="share_pct")[models]
    shares = shares.loc[CATEGORIES]

    say()
    say("Share of each model's total importance (%), by category:")
    printable = shares.copy()
    printable.insert(0, "n_features", [counts.get(c, 0) for c in printable.index])
    printable["mean"] = shares.mean(axis=1)
    printable.columns = ["n_features"] + [MODEL_LABELS[m] for m in models] + ["mean"]
    say(printable.sort_values("mean", ascending=False).round(1).to_string())

    side_share = {}
    for model in models:
        side_total = features[model].loc[features[model]["base"].isin(SIDE_BASES), "importance"].sum()
        side_share[model] = 100.0 * side_total / features[model]["importance"].sum()
    say()
    say("Share of total importance held by attacker_side + victim_side alone: "
        + ", ".join(f"{MODEL_LABELS[m]} {side_share[m]:.1f}%" for m in models))
    excl = category_table.pivot(index="category", columns="model",
                                values="share_excl_sides_pct")[models].loc[CATEGORIES]
    excl.columns = [MODEL_LABELS[m] for m in models]
    say("Category shares (%) with those two features set aside and the rest re-normalised:")
    say(excl.loc[printable.sort_values("mean", ascending=False).index].round(1).to_string())

    # ---- by owner (attacker / victim / shared) -----------------------------
    owner_table = pd.DataFrame(
        [row for model in models for row in owner_rows(model, features[model])])
    owner_table.to_csv(out_dir / "importance_by_owner.csv", index=False)
    owner_share = owner_table.pivot(index="owner", columns="model",
                                    values="share_excl_sides_pct")[models].loc[OWNERS]
    owner_per_feature = owner_table.pivot(index="owner", columns="model",
                                          values="importance_per_feature_excl_sides")[models].loc[OWNERS]
    owner_counts = owner_table[owner_table["model"] == models[0]].set_index("owner")["n_features_excl_sides"]
    say()
    say("Share of total importance (%) by whose feature it is, the two side columns set aside "
        "(shared = differences, distances, angles, team totals, round context):")
    printable_owner = owner_share.copy()
    printable_owner.insert(0, "n_features", owner_counts.loc[OWNERS].values)
    printable_owner.columns = ["n_features"] + [MODEL_LABELS[m] for m in models]
    say(printable_owner.round(1).to_string())
    say("Mean importance per feature, in thousandths of macro-F1 (same grouping):")
    per_feature = (owner_per_feature * 1000.0).copy()
    per_feature.columns = [MODEL_LABELS[m] for m in models]
    say(per_feature.round(2).to_string())

    # ---- top features -----------------------------------------------------
    top_rows, top_by_model = [], {}
    for model in models:
        top = features[model].head(args.top_n).copy()
        selected = eliminations[model]["selected_features"] if eliminations[model] else None
        total = features[model]["importance"].sum()
        top["share_of_total_pct"] = 100.0 * top["importance"] / total
        if selected is not None:
            kept_bases = {base_of(column) for column in selected}
            top["kept_after_elimination"] = top["base"].isin(kept_bases)
        else:
            top["kept_after_elimination"] = np.nan
        top.insert(0, "rank", np.arange(1, len(top) + 1))
        top.insert(0, "model", model)
        top_rows.append(top.rename(columns={"base": "feature"}))
        top_by_model[model] = features[model]
    top_table = pd.concat(top_rows, ignore_index=True)[[
        "model", "rank", "feature", "category", "importance", "std",
        "share_of_total_pct", "kept_after_elimination", "n_columns"]]
    top_table.to_csv(out_dir / "top_features.csv", index=False)

    say()
    say("Top 5 features per model (macro-F1 drop, mean +/- std):")
    for model in models:
        head = features[model].head(5)
        text = "; ".join(f"{r.base} {r.importance:.4f}+/-{r.std:.4f}" for r in head.itertuples())
        say(f"  {MODEL_LABELS[model]:<20} {text}")

    # share of the single most important feature, for context
    say()
    for model in models:
        top1 = features[model].iloc[0]
        top2 = features[model].iloc[:2]["importance"].sum()
        total = features[model]["importance"].sum()
        say(f"  {MODEL_LABELS[model]:<20} #1 feature = {100 * top1['importance'] / total:.1f}% of "
            f"total importance; top 2 = {100 * top2 / total:.1f}%")

    # ---- consensus ----------------------------------------------------------
    consensus = wide[["feature", "category", "mean_rank", "best_rank", "worst_rank"]].copy()
    for model in models:
        consensus[f"{model}_importance"] = wide[f"{model}_importance"]
    in_top = pd.DataFrame({
        m: wide[f"{m}_rank"] <= args.top_n for m in models})
    consensus["n_models_in_top_n"] = in_top.sum(axis=1)
    consensus = consensus.sort_values(["mean_rank", "feature"]).head(30).reset_index(drop=True)
    consensus.insert(0, "consensus_rank", np.arange(1, len(consensus) + 1))
    consensus.to_csv(out_dir / "consensus_top_features.csv", index=False)

    say()
    say(f"Features in the top {args.top_n} of EVERY model:")
    common = set.intersection(*[
        set(features[m].head(args.top_n)["base"]) for m in models])
    say("  " + (", ".join(sorted(common)) if common else "(none)"))
    say()
    say("Consensus top 10 (lowest average rank across models):")
    for row in consensus.head(10).itertuples():
        say(f"  {row.consensus_rank:>2}. {row.feature:<32} {CATEGORY_SHORT[row.category]:<11} "
            f"mean rank {row.mean_rank:>5.1f}  (best {int(row.best_rank)}, worst {int(row.worst_rank)}; "
            f"in top-{args.top_n} of {int(row.n_models_in_top_n)} models)")

    # ---- agreement ----------------------------------------------------------
    importance_matrix = pd.DataFrame({m: wide.set_index("feature")[f"{m}_importance"] for m in models})
    spearman = importance_matrix.corr(method="spearman")
    agreement = []
    for i, first in enumerate(models):
        for second in models[i + 1:]:
            overlap = (set(features[first].head(args.top_n)["base"])
                       & set(features[second].head(args.top_n)["base"]))
            agreement.append({
                "model_a": first, "model_b": second,
                "spearman_all_features": float(spearman.loc[first, second]),
                f"top{args.top_n}_overlap": len(overlap),
            })
    agreement = pd.DataFrame(agreement)
    agreement.to_csv(out_dir / "model_agreement.csv", index=False)

    say()
    say(f"Agreement between models (Spearman over all {len(all_features)} features; "
        f"overlap of the top {args.top_n}):")
    for row in agreement.itertuples():
        say(f"  {MODEL_LABELS[row.model_a]:<20} vs {MODEL_LABELS[row.model_b]:<20} "
            f"rho = {row.spearman_all_features:.3f}   overlap {getattr(row, f'top{args.top_n}_overlap')}/{args.top_n}")

    # ---- retention ----------------------------------------------------------
    retention = None
    with_elimination = [m for m in models if eliminations[m] is not None]
    if with_elimination:
        retention = pd.DataFrame(
            [row for model in with_elimination
             for row in retention_rows(model, raw[model], eliminations[model])])
        retention.to_csv(out_dir / "retention_by_category.csv", index=False)

        say()
        say("Share of each category kept by feature elimination (%):")
        pivot = retention.pivot(index="category", columns="model", values="pct_features_kept")
        pivot = pivot.loc[CATEGORIES, with_elimination]
        pivot.columns = [MODEL_LABELS[m] for m in with_elimination]
        say(pivot.round(0).astype(int).to_string())
        for model in with_elimination:
            info = eliminations[model]
            say(f"  {MODEL_LABELS[model]:<20} {info['n_features_before']} -> {info['n_features_after']} "
                f"columns ({info['n_eliminated']} removed)")

        # important-but-removed check
        say()
        say(f"Top-{args.top_n} features that elimination REMOVED (should be none or few):")
        removed_any = False
        for model in with_elimination:
            kept_bases = {base_of(c) for c in eliminations[model]["selected_features"]}
            lost = [b for b in features[model].head(args.top_n)["base"] if b not in kept_bases]
            if lost:
                removed_any = True
                say(f"  {MODEL_LABELS[model]:<20} {', '.join(lost)}")
        if not removed_any:
            say("  none")

    # ---- figures ------------------------------------------------------------
    if not args.no_plots:
        plot_category_share(out_dir / "category_importance_share.png", shares, counts, models,
                            side_share)
        plot_top_features(out_dir / "top_features.png", top_by_model, args.plot_top_n, models)
        if retention is not None:
            plot_retention(out_dir / "category_retention.png", retention, counts, with_elimination)

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    say()
    say(f"Wrote everything to {out_dir}")


if __name__ == "__main__":
    main()
