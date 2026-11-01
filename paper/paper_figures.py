"""
paper_figures.py - the four data figures in the paper, drawn from the saved
results (no refits).

Usage (from the repository root, after paper_stats.py):
    python paper/scripts/paper_figures.py --results-dir tuning_results_n7 \
        --numbers paper/paper_numbers.json --out-dir paper/figures

Writes vector PDFs sized for an IEEE two-column page (3.5 in column):
    confusion_xgboost.pdf      row-normalised confusion matrix, XGBoost
    grouped_importance.pdf     test ROC-AUC lost per feature category
    accuracy_progression.pdf   accuracy over match progression (20% window)
    rfe_curves.pdf             validation ROC-AUC during feature elimination
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
from matplotlib.colors import LinearSegmentedColormap

MODELS = ["xgboost", "transformer", "logistic_regression", "random_forest"]
LABEL = {"xgboost": "XGBoost", "transformer": "FT-Transformer",
         "logistic_regression": "Logistic regression", "random_forest": "Random forest"}
# Okabe-Ito subset, validated all-pairs (CVD dE 11.0, normal-vision 15.6);
# the orange sits below 3:1 on white, so every series also has its own marker
# and line style and is labelled directly.
COLOUR = {"xgboost": "#0072b2", "transformer": "#d55e00",
          "logistic_regression": "#009e73", "random_forest": "#e69f00"}
MARKER = {"xgboost": "o", "transformer": "s", "logistic_regression": "^", "random_forest": "D"}
STYLE = {"xgboost": "-", "transformer": "--", "logistic_regression": "-.", "random_forest": ":"}
INK, INK2, MUTED, GRID = "#111111", "#4a4a48", "#7a7975", "#dedcd5"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

CLASS_ORDER = ["attacker_won", "victim_won", "mutual_kill", "interrupted", "timedout", "round_end"]
CLASS_CODE = {"attacker_won": 0, "interrupted": 1, "mutual_kill": 2, "round_end": 3,
              "timedout": 4, "victim_won": 5}

COL_W = 3.45   # inches, IEEE column is 3.5 in


def setup() -> None:
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
        "axes.edgecolor": "#9a9890", "axes.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 150,
    })


def clean(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
def confusion(numbers: dict, out: Path) -> None:
    cm = np.asarray(numbers["test"]["xgboost/before"]["confusion_matrix"], dtype=float)
    idx = [CLASS_CODE[c] for c in CLASS_ORDER]
    cm = cm[np.ix_(idx, idx)]
    support = cm.sum(axis=1)
    rows = cm / support[:, None]

    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    ax.imshow(rows, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(rows.shape[0]):
        for j in range(rows.shape[1]):
            v = rows[i, j]
            text = f"{100 * v:.1f}" if v >= 0.0005 else "0"
            ax.text(j, i, text, ha="center", va="center", fontsize=6.6,
                    color="white" if v > 0.55 else INK)
    names = [c.replace("_", r"\_") for c in CLASS_ORDER]
    ax.set_xticks(range(6))
    ax.set_xticklabels([c.replace("_", "_​") for c in CLASS_ORDER], rotation=28, ha="right")
    ax.set_yticks(range(6))
    ax.set_yticklabels([f"{c} ({int(n):,})" for c, n in zip(CLASS_ORDER, support)])
    ax.set_xlabel("Predicted outcome")
    ax.set_ylabel("True outcome (test rows)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
def grouped_importance(results: Path, out: Path) -> None:
    g = pd.read_csv(results / "grouped_permutation_importance" / "grouped_importance_all_models.csv")
    cats = g[g["kind"] == "category"]
    order = cats.groupby("group")["auc_drop"].mean().sort_values(ascending=False).index.tolist()
    short = {"Map geometry (nav mesh)": "Map geometry (nav mesh)",
             "Round & team context": "Round & team context",
             "Economy & utility": "Economy & utility",
             "Connection / other": "Connection (ping, LAN)"}
    nfeat = cats.groupby("group")["n_features"].first()

    fig, ax = plt.subplots(figsize=(COL_W, 2.18))
    n = len(MODELS)
    height = 0.19
    y0 = np.arange(len(order))[::-1]
    for k, m in enumerate(MODELS):
        part = cats[cats["model"] == m].set_index("group").reindex(order)
        y = y0 + (1.5 - k) * height
        ax.barh(y, part["auc_drop"], height=height * 0.86, color=COLOUR[m],
                edgecolor="white", linewidth=0.4, zorder=2, label=LABEL[m])
        ax.errorbar(part["auc_drop"], y, xerr=part["auc_drop_std"], fmt="none",
                    ecolor=INK2, elinewidth=0.5, capsize=1.2, capthick=0.5, zorder=3)
    ax.set_yticks(y0)
    ax.set_yticklabels([f"{short.get(c, c)} ({int(nfeat[c])})" for c in order])
    ax.set_xlabel("Test ROC-AUC lost when the category is shuffled")
    ax.set_xlim(0, 0.115)
    ax.tick_params(axis="y", length=0)
    clean(ax, "x")
    ax.legend(loc="lower right", frameon=False, handlelength=1.2, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
def accuracy_progression(results: Path, out: Path) -> None:
    a = pd.read_csv(results / "accuracy_over_time" / "window_20pct" / "accuracy_over_time.csv")
    fig, ax = plt.subplots(figsize=(COL_W, 2.06))
    xb = a[(a["model"] == "xgboost") & (a["variant"] == "before")]
    ax.fill_between(xb["progress_pct"], xb["mean_accuracy"] - xb["std"], xb["mean_accuracy"] + xb["std"],
                    color=COLOUR["xgboost"], alpha=0.12, linewidth=0, zorder=1,
                    label=r"XGBoost $\pm$1 s.d.")
    ends = {}
    for m in MODELS:
        p = a[(a["model"] == m) & (a["variant"] == "before")]
        ax.plot(p["progress_pct"], p["mean_accuracy"], STYLE[m], color=COLOUR[m], linewidth=1.3,
                marker=MARKER[m], markersize=2.6, markevery=4, zorder=3, label=LABEL[m])
        ends[m] = float(p["mean_accuracy"].iloc[-1])
    b = a[a["model"] == "majority_class"]
    ax.plot(b["progress_pct"], b["mean_accuracy"], color=MUTED, linewidth=1.0, linestyle=(0, (4, 2)),
            zorder=2, label="Majority class")
    ax.set_xlim(0, 100)
    ax.set_ylim(0.40, 0.75)
    ax.set_xlabel("Match progression (%)")
    ax.set_ylabel("Accuracy (20% window)")
    clean(ax, "y")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01, 1.0, 0.2), mode="expand", ncol=3,
              frameon=False, columnspacing=0.6, handlelength=1.9, borderaxespad=0.0, fontsize=6.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
def rfe_curves(numbers: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(COL_W, 1.85))
    for m in MODELS:
        e = numbers["elimination"][m]
        c = e["curve"]
        n0 = c["n_features"][0]
        x = 100 * np.asarray(c["n_features"]) / n0
        y = np.asarray(c["performance"])
        ax.plot(x, y, STYLE[m], color=COLOUR[m], linewidth=1.2, marker=MARKER[m], markersize=3.2,
                zorder=3, label=f"{LABEL[m]} ({n0}" + (" columns" if m == "logistic_regression" else "") + ")")
        sel = c["n_features"].index(e["n_after"])
        ax.plot(x[sel], y[sel], marker=MARKER[m], markersize=7.5, markerfacecolor="none",
                markeredgecolor=COLOUR[m], markeredgewidth=0.9, linestyle="none", zorder=4)
    ax.set_xlim(102, 15)
    ax.set_xlabel("Inputs remaining (% of the starting set)")
    ax.set_ylabel("Validation ROC-AUC")
    clean(ax, "y")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.56), frameon=False, fontsize=6.3,
              handlelength=2.2, borderaxespad=0.1)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="tuning_results_n7")
    ap.add_argument("--numbers", default="paper_numbers.json")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    setup()
    res, out = Path(args.results_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    numbers = json.loads(Path(args.numbers).read_text(encoding="utf-8"))
    confusion(numbers, out / "confusion_xgboost.pdf")
    grouped_importance(res, out / "grouped_importance.pdf")
    accuracy_progression(res, out / "accuracy_progression.pdf")
    rfe_curves(numbers, out / "rfe_curves.pdf")
    print(f"wrote 4 figures to {out}")


if __name__ == "__main__":
    main()
