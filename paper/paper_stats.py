"""
paper_stats.py - every number quoted in the paper, computed from the saved
results (no model is refit, nothing is tuned).

Usage (from the repository root):
    python paper/scripts/paper_stats.py --results-dir tuning_results_n7 \
        --raw-csv data/engagement_dataset.csv --out paper/paper_numbers.json

Reads
    <raw-csv>                       match ids, maps, LAN flag, labels, ticks
    <results-dir>/<model>/...       held-out results and predictions (.npz),
                                    sweep grids, elimination summaries
Writes
    <out>                           a JSON file with all the numbers

Adds two things the result folders do not already contain, both computed
from the saved test predictions only:
    * 95% match-level bootstrap intervals for test ROC-AUC (2,000 resamples of
      the 278 test matches, seed 2026), for each model and for the
      differences between models;
    * the paired per-match accuracy change after minus before elimination.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

MODELS = ["xgboost", "transformer", "logistic_regression", "random_forest"]
CLASSES = ["attacker_won", "interrupted", "mutual_kill", "round_end", "timedout", "victim_won"]
N_BOOT = 2000
BOOT_SEED = 2026


# --------------------------------------------------------------------------
# Weighted one-vs-rest ROC-AUC (fast: each class is sorted once, every
# bootstrap resample only re-weights the rows)
# --------------------------------------------------------------------------

class WeightedOvrAuc:
    def __init__(self, y: np.ndarray, proba: np.ndarray, n_classes: int = 6):
        self.parts = []
        for c in range(n_classes):
            score = proba[:, c]
            # tie groups in ascending score order
            uniq, group = np.unique(score, return_inverse=True)
            self.parts.append((group, len(uniq), (y == c).astype(float)))

    def __call__(self, w: np.ndarray) -> float:
        aucs = []
        for group, n_groups, pos in self.parts:
            p = np.bincount(group, weights=w * pos, minlength=n_groups)
            n = np.bincount(group, weights=w * (1.0 - pos), minlength=n_groups)
            n_below = np.cumsum(n) - n
            total_p, total_n = p.sum(), n.sum()
            aucs.append(float(np.sum(p * (n_below + 0.5 * n)) / (total_p * total_n)))
        return float(np.mean(aucs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="tuning_results_n7")
    ap.add_argument("--raw-csv", default="data/engagement_dataset.csv")
    ap.add_argument("--out", default="paper_numbers.json")
    args = ap.parse_args()
    res = Path(args.results_dir)
    out: dict = {}

    # ---------------------------------------------------------------- data
    raw = pd.read_csv(args.raw_csv, usecols=[
        "match_id", "round_num", "start_tick", "frame_tick", "start_time",
        "engagement_status", "map_name", "is_lan"])
    labels = pd.Categorical(raw["engagement_status"], categories=CLASSES).codes.astype(np.int64)
    assert (labels >= 0).all()
    is_lan = raw["is_lan"].astype(str).eq("True")

    per_match = raw.groupby("match_id").agg(map=("map_name", "first"), lan=("is_lan", "first"))
    out["data"] = {
        "n_engagements": int(len(raw)),
        "n_matches": int(raw["match_id"].nunique()),
        "n_rounds": int(raw[["match_id", "round_num"]].drop_duplicates().shape[0]),
        "n_lan_matches": int((per_match["lan"].astype(str) == "True").sum()),
        "n_online_matches": int((per_match["lan"].astype(str) != "True").sum()),
        "engagements_per_map": raw["map_name"].value_counts().to_dict(),
        "matches_per_map": per_match["map"].value_counts().to_dict(),
        "class_counts": {c: int((labels == i).sum()) for i, c in enumerate(CLASSES)},
        "class_share_pct": {c: round(100 * float((labels == i).mean()), 2) for i, c in enumerate(CLASSES)},
        "engagements_per_match": raw.groupby("match_id").size().describe().round(1).to_dict(),
        "engagements_per_round_mean": round(float(raw.groupby(["match_id", "round_num"]).size().mean()), 2),
        "lan_engagement_share_pct": round(100 * float(is_lan.mean()), 1),
    }

    # feature frame offset: ticks between the feature frame and the first damage
    gap = (raw["start_tick"] - raw["frame_tick"]).to_numpy()
    # tick rate per match from engagements in the same round
    r = raw.sort_values(["match_id", "round_num", "start_tick"])
    d_tick = r.groupby(["match_id", "round_num"])["start_tick"].diff()
    d_time = r.groupby(["match_id", "round_num"])["start_time"].diff()
    rate = (d_tick / d_time).where(d_time > 1.0)
    match_rate = rate.groupby(r["match_id"]).median().round()
    rate_row = raw["match_id"].map(match_rate)
    gap_s = gap / rate_row.to_numpy()
    out["data"]["feature_frame_gap"] = {
        "ticks_median": float(np.median(gap)),
        "ticks_p5_p95": [float(np.percentile(gap, 5)), float(np.percentile(gap, 95))],
        "seconds_median": float(np.nanmedian(gap_s)),
        "seconds_p5_p95": [float(np.nanpercentile(gap_s, 5)), float(np.nanpercentile(gap_s, 95))],
        "seconds_max": float(np.nanmax(gap_s)),
        "tick_rates": match_rate.value_counts().to_dict(),
    }

    # ---------------------------------------------------------------- split
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tv_idx, te_idx = next(splitter.split(raw, labels, groups=raw["match_id"].to_numpy()))
    y_test = labels[te_idx]
    test_match = raw["match_id"].to_numpy()[te_idx]
    test_codes, test_match_idx = np.unique(test_match, return_inverse=True)
    n_test_matches = len(test_codes)
    out["split"] = {
        "n_trainval_rows": int(len(tv_idx)), "n_test_rows": int(len(te_idx)),
        "n_trainval_matches": int(raw["match_id"].iloc[tv_idx].nunique()),
        "n_test_matches": int(n_test_matches),
        "test_class_counts": {c: int((y_test == i).sum()) for i, c in enumerate(CLASSES)},
        "test_majority_accuracy": float((y_test == 0).mean()),
        "test_lan_matches": int((per_match.loc[test_codes, "lan"].astype(str) == "True").sum()),
        "mutual_kill_test_matches": int(len(np.unique(test_match[y_test == 2]))),
    }

    # ---------------------------------------------------------- predictions
    preds = {}
    for m in MODELS:
        for variant, sub in (("before", ""), ("after", "elimination/")):
            z = np.load(res / m / f"{sub}{m}_held_out_predictions.npz")
            assert np.array_equal(z["y_test"], y_test), f"{m}/{variant}: split differs"
            preds[(m, variant)] = (z["y_pred"].astype(int), z["y_proba"].astype(float))

    out["test"] = {}
    for m in MODELS:
        for variant, sub in (("before", ""), ("after", "elimination/")):
            h = json.load(open(res / m / f"{sub}{m}_held_out_test_result.json"))
            y_pred, proba = preds[(m, variant)]
            sk = roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=list(range(6)))
            fast = WeightedOvrAuc(y_test, proba)(np.ones(len(y_test)))
            assert abs(sk - h["roc_auc"]) < 1e-9 and abs(fast - sk) < 1e-9, (m, variant, sk, fast, h["roc_auc"])
            cm = np.asarray(h["confusion_matrix"])
            per_class = {CLASSES[int(k)]: {kk: round(v, 6) for kk, v in d.items()} for k, d in h["per_class"].items()}
            out["test"][f"{m}/{variant}"] = {
                "accuracy": h["accuracy"], "f1_macro": h["f1"], "precision_macro": h["precision"],
                "recall_macro": h["recall"], "roc_auc": h["roc_auc"],
                "per_class": per_class,
                "per_class_auc": {CLASSES[c]: round(float(roc_auc_score((y_test == c).astype(int), proba[:, c])), 4)
                                  for c in range(6)},
                "confusion_matrix": cm.tolist(),
                "predicted_share_pct": {CLASSES[c]: round(100 * float((y_pred == c).mean()), 2) for c in range(6)},
            }

    # ---------------------------------------------- match-level bootstrap
    rng = np.random.default_rng(BOOT_SEED)
    scorers = {k: WeightedOvrAuc(y_test, v[1]) for k, v in preds.items()}
    boots = {k: np.empty(N_BOOT) for k in preds}
    for b in range(N_BOOT):
        counts = rng.multinomial(n_test_matches, np.full(n_test_matches, 1.0 / n_test_matches))
        w = counts[test_match_idx].astype(float)
        for k, f in scorers.items():
            boots[k][b] = f(w)

    def ci(x):
        return [round(float(np.percentile(x, 2.5)), 5), round(float(np.percentile(x, 97.5)), 5)]

    out["bootstrap"] = {"n_resamples": N_BOOT, "seed": BOOT_SEED, "unit": "test match"}
    for k, v in boots.items():
        out["bootstrap"][f"{k[0]}/{k[1]}"] = {"ci95": ci(v), "std": round(float(v.std(ddof=1)), 4)}
    pairs = [("xgboost", "transformer"), ("xgboost", "logistic_regression"), ("xgboost", "random_forest"),
             ("transformer", "logistic_regression"), ("transformer", "random_forest"),
             ("logistic_regression", "random_forest")]
    for variant in ("before", "after"):
        for a, b_ in pairs:
            diff = boots[(a, variant)] - boots[(b_, variant)]
            point = out["test"][f"{a}/{variant}"]["roc_auc"] - out["test"][f"{b_}/{variant}"]["roc_auc"]
            out["bootstrap"][f"diff/{variant}/{a}-{b_}"] = {
                "point": round(point, 5), "ci95": ci(diff),
                "share_of_resamples_above_0": round(float((diff > 0).mean()), 4)}
    for m in MODELS:
        diff = boots[(m, "after")] - boots[(m, "before")]
        point = out["test"][f"{m}/after"]["roc_auc"] - out["test"][f"{m}/before"]["roc_auc"]
        out["bootstrap"][f"diff/after-before/{m}"] = {"point": round(point, 5), "ci95": ci(diff)}

    # ------------------------------------- paired per-match accuracy change
    out["elimination_paired_accuracy"] = {}
    for m in MODELS:
        yb, _ = preds[(m, "before")]
        ya, _ = preds[(m, "after")]
        acc_b = np.bincount(test_match_idx, weights=(yb == y_test)) / np.bincount(test_match_idx)
        acc_a = np.bincount(test_match_idx, weights=(ya == y_test)) / np.bincount(test_match_idx)
        d = acc_a - acc_b
        half = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        out["elimination_paired_accuracy"][m] = {
            "mean_change": round(float(d.mean()), 4),
            "ci95_normal": [round(float(d.mean() - half), 4), round(float(d.mean() + half), 4)],
            "share_predictions_changed_pct": round(100 * float((ya != yb).mean()), 1),
        }

    # ------------------------------------------------------- tuning sweeps
    out["sweep"] = {}
    for m in MODELS:
        grid = pd.read_csv(res / m / f"{m}_grid_results.csv")
        fin = json.load(open(res / m / f"{m}_final_result.json"))
        best = fin["best_params"]
        mask = np.ones(len(grid), bool)
        for k, v in best.items():
            col = grid[k]
            if v is None:
                mask &= col.isna().to_numpy()
            elif isinstance(v, str):
                mask &= (col.astype(str) == v).to_numpy()
            else:
                mask &= np.isclose(pd.to_numeric(col, errors="coerce").to_numpy(dtype=float), float(v))
        row = grid[mask]
        assert len(row) == 1, (m, len(row))
        row = row.iloc[0]
        out["sweep"][m] = {
            "n_combinations": int(len(grid)),
            "best_params": best,
            "val_roc_auc_mean": float(row["roc_auc_mean"]), "val_roc_auc_std": float(row["roc_auc_std"]),
            "val_accuracy_mean": float(row["accuracy_mean"]), "val_f1_mean": float(row["f1_mean"]),
            "n_valid_splits": int(row["n_valid_roc_auc_splits"]),
            "val_roc_auc_range_over_grid": [float(grid["roc_auc_mean"].min()), float(grid["roc_auc_mean"].max())],
        }

    # --------------------------------------------------------- elimination
    out["elimination"] = {}
    for m in MODELS:
        s = json.load(open(res / m / "elimination" / "elimination_summary.json"))
        curve = pd.read_csv(res / m / "elimination" / "rfe" / "rfe_curve.csv")
        out["elimination"][m] = {
            "n_before": s["n_features_before"], "n_after": s["n_features_after"],
            "n_eliminated": s["n_eliminated"],
            "val_selected": s["validation_at_selected"], "val_best_seen": s["validation_best_seen"],
            "test_before": s["test_before"], "test_after": s["test_after"],
            "curve": curve[["step", "n_features", "performance", "roc_auc_std"]].round(5).to_dict("list"),
            "settings": s["rfe_settings"],
            "first_dropped": s["eliminated_in_order"][:12],
        }

    diag = pd.read_csv(res / "removed_feature_diagnostics.csv")
    out["removed_features"] = {
        "n_rows": int(len(diag)),
        "by_removed_by": diag["removed_by"].value_counts().to_dict(),
        "by_reason_both": diag[diag["removed_by"] == "both"]["category"].value_counts().to_dict(),
        "by_reason_all": diag["category"].value_counts().to_dict(),
    }

    # -------------------------------------------- grouped importance (test)
    g = pd.read_csv(res / "grouped_permutation_importance" / "grouped_importance_all_models.csv")
    out["grouped_importance"] = {
        f"{r.model}/{r.group}": {"auc_drop": round(r.auc_drop, 4), "std": round(r.auc_drop_std, 4),
                                 "pct_above_chance_lost": round(r.pct_of_above_chance_auc_lost, 1),
                                 "n_features": int(r.n_features), "n_columns": int(r.n_columns)}
        for r in g.itertuples()}
    out["side_ablation"] = {}
    for m in MODELS:
        meta = json.load(open(res / "grouped_permutation_importance" / f"{m}_grouped_importance_meta.json"))
        ab = meta.get("side_ablation") or {}
        out["side_ablation"][m] = {k: ab.get(k) for k in ("auc_with_sides", "auc_without_sides",
                                                          "auc_change_when_removed")}
        out["side_ablation"][m]["reproduces_saved"] = meta["reproduces_saved_result"]

    # ------------------------------------------- accuracy over progression
    w = pd.read_csv(res / "accuracy_over_time" / "window_size_summary.csv")
    out["accuracy_over_time_summary"] = w.round(4).to_dict("records")
    a20 = pd.read_csv(res / "accuracy_over_time" / "window_20pct" / "accuracy_over_time.csv")
    out["accuracy_over_time_20"] = {
        f"{k[0]}/{k[1]}": grp.set_index("progress_pct")["mean_accuracy"].round(4).to_dict()
        for k, grp in a20.groupby(["model", "variant"])}

    Path(args.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
