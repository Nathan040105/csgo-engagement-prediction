# Predicting 1v1 engagement outcomes in professional CS:GO

Code, results and trained models for the Honours research project **"A Machine
Learning Approach to Predicting Player Engagement Success in CS:GO"**
(University of the Witwatersrand, 2026).

**Author:** Nathan Le Roux · **Supervisor:** Branden Ingram
<!-- Add the co-supervisor here if the final report lists one. -->

The project predicts how a one-on-one fight between two players ends, using only
the game state just before it starts. It extracts **402,873 engagements from
1,388 professional matches** in the [ESTA dataset](https://github.com/pnxenopoulos/esta)
(seven maps), describes each with 159 features (positions, view angles, health,
weapons, economy, round state and nav-mesh map geometry), and compares
**logistic regression, random forest, XGBoost and an FT-Transformer**.

An engagement starts at the first damage between two players and ends in one of
six ways, which are the classes predicted:

| Class | Meaning |
|---|---|
| `attacker_won` | the player who dealt the first damage gets the kill |
| `victim_won` | the other player gets the kill |
| `mutual_kill` | both players kill each other (a trade) |
| `interrupted` | one of the two starts fighting someone else |
| `timedout` | no damage between them for 5 seconds |
| `round_end` | the round ends first |

---

## Results at a glance

All numbers are on **278 held-out test matches (81,693 engagements)** that were
not used for any choice of model, hyperparameter or feature. Always predicting
the most common class (`attacker_won`) gives 49.4% accuracy.

| Model | Validation ROC-AUC | Test ROC-AUC | Test accuracy | Test macro F1 |
|---|---|---|---|---|
| XGBoost | 0.8275 | **0.8199** | 0.6344 | 0.3584 |
| Transformer (FT-Transformer) | 0.8255 | **0.8184** | 0.6295 | 0.3580 |
| Logistic regression | 0.8035 | **0.7921** | 0.6031 | 0.3291 |
| Random forest | 0.8022 | **0.7909** | 0.6070 | 0.2989 |

ROC-AUC is the macro one-vs-rest average over the six classes. XGBoost and the
transformer are effectively tied (0.0015 apart, against a split-to-split
standard deviation of 0.007–0.009 in the sweeps); both lead logistic regression
and random forest by 0.026–0.029. The validation column is the best of many
hyperparameter combinations and therefore optimistic; the test column is the
result.

**Recall per class on the test matches** (test rows in brackets):

| Class | XGBoost | Transformer | Logistic regression | Random forest |
|---|---|---|---|---|
| attacker_won (40,335) | 0.88 | 0.86 | 0.85 | 0.93 |
| interrupted (21,464) | 0.47 | 0.49 | 0.43 | 0.34 |
| timedout (11,466) | 0.49 | 0.46 | 0.42 | 0.41 |
| victim_won (7,931) | 0.10 | 0.09 | 0.09 | 0.02 |
| round_end (457) | 0.09 | 0.11 | 0.07 | 0.04 |
| mutual_kill (40) | 0.00 | 0.00 | 0.00 | 0.00 |

**Feature elimination** (backward elimination on the train+validation matches
only, hyperparameters held fixed):

| Model | Features before → after | Validation ROC-AUC before → after | Test ROC-AUC before → after |
|---|---|---|---|
| Logistic regression | 549 → 308 (one-hot columns) | 0.8035 → 0.8092 | 0.7921 → 0.7922 |
| XGBoost | 159 → 89 | 0.8275 → 0.8294 | 0.8199 → 0.8209 |
| Transformer | 159 → 111 | 0.8256 → 0.8306 | 0.8184 → 0.8209 |
| Random forest | 159 → 159 (first step lost more than the tolerance) | 0.8022 | 0.7909 |

**Which kinds of feature matter** (test ROC-AUC lost when all features in a
category are shuffled together; mean of 5 shuffles):

| Category (features) | XGBoost | Transformer | Logistic regression | Random forest |
|---|---|---|---|---|
| Spatial (33) | 0.094 | 0.082 | 0.073 | 0.076 |
| Combat state (32) | 0.072 | 0.104 | 0.100 | 0.070 |
| Round & team context (23) | 0.041 | 0.055 | 0.041 | 0.047 |
| Economy & utility (40) | 0.026 | 0.015 | 0.022 | 0.012 |
| Map geometry, nav mesh (15) | 0.016 | 0.028 | 0.008 | 0.017 |
| Movement (11) | 0.007 | 0.010 | 0.005 | 0.008 |
| Connection / other (5) | 0.000 | 0.000 | 0.000 | 0.000 |
| *All 159 features at once* | 0.317 | 0.316 | 0.287 | 0.286 |

Spatial and combat-state features drive every model; ping and LAN/online carry
nothing. Categories share information, so the drops do not add up to the
all-features drop, and the shuffles describe fitted models rather than causes.

Figures: `tuning_results_n7/accuracy_over_time/`,
`tuning_results_n7/feature_importance_analysis/` and
`tuning_results_n7/grouped_permutation_importance/grouped_importance_auc.png`.

### Where each research question is answered

<!-- TODO: update the RQ4 row once the with/without-nav-mesh comparison has run. -->

| Research question | Evidence in this repository |
|---|---|
| RQ1: can the pre-engagement state predict the outcome? | `tuning_results_n7/<model>/<model>_held_out_test_result.json`, `model_comparison_summary.csv`, `accuracy_over_time/` |
| RQ2: which feature categories matter most? | `feature_importance_analysis/`, `grouped_permutation_importance/` |
| RQ3: traditional models versus a transformer | `model_comparison_summary.csv`, the per-model result files |
| RQ4: do nav-mesh features help? | the nav-mesh row of the grouped importance (evidence of use, not yet a with/without comparison) |

---

## Repository layout

The scripts sit in one folder because they import each other and find their
data folders by relative path; run them from the repository root.

```
.
├── config.py, utils.py              settings (paths, timeout, weapon tables), shared helpers
├── parser.py                        CS:GO .nav file -> JSON (adapted from fpapouin/pynav, see Licence)
├── nav_data.py                      nav-mesh lookups: area, callout, walkable path distance, connectivity, cover
├── engagement_extractor.py          finds 1v1 engagements and their outcome in each round
├── feature_extractor.py             features from the frame before each engagement
├── build_dataset.py                 ESTA demos -> data/engagement_dataset.csv
├── feature_engineering.py           raw CSV -> per-model feature CSVs (one-hot / ordinal)
├── model_tuning.py                  grid sweeps, repeated match-level splits, held-out test, Ray execution
├── tabular_transformer.py           FT-Transformer with a scikit-learn interface
├── run_final_tuning.py              the final 7-repeat sweep for all four models
├── feature_elimination.py           backward recursive feature elimination
├── run_feature_elimination.py       elimination driver + one refit scored on the test matches
├── accuracy_over_time.py            accuracy over match progression (10/20/30% windows)
├── feature_importance_analysis.py   importance by category (RQ2), from saved importances
├── grouped_permutation_importance.py  category shuffles on the test matches (RQ2)
├── check_side_features.py           check that attacker_side / victim_side are redundant
├── save_final_models.py             refits, checks and saves the trained models
├── clean_checkpoint.py, merge_summaries.py, test_ray_cluster.py
│                                    small utilities used along the way
├── requirements.txt
├── nav_data/
│   ├── maps/*.nav                   navigation meshes from the CS:GO Legacy game files (Valve)
│   └── parsed/*.json                the same, parsed by parser.py (what the pipeline reads)
├── tuning_data/
│   └── categorical_mappings_ordinal.json   ordinal code -> category name
│                                    (the feature CSVs themselves are rebuilt, see Data)
├── tuning_results_n7/               every result: grids, sweep checkpoints, held-out
│   ├── <model>/                     test results and predictions, importances, plots,
│   │   └── elimination/             elimination steps and after-elimination results
│   ├── accuracy_over_time/
│   ├── feature_importance_analysis/
│   ├── grouped_permutation_importance/
│   └── model_comparison_summary.csv
└── models/                          trained models + manifest.csv (the random forest is in the release)
```

Not in the repository, because they are too large: the ESTA demos (`esta/`),
`data/engagement_dataset.csv`, the four `tuning_data/*_features.csv` files and
the random forest model. All except the ESTA demos are attached to the
[`data-v1` release](https://github.com/Nathan040105/csgo-engagement-prediction/releases/tag/data-v1); the next section says how to get each.

---

## Data

| What | Where to get it | Size |
|---|---|---|
| ESTA demos (1,558 professional matches, Jan 2021 – May 2022) | `git clone --depth 1 https://github.com/pnxenopoulos/esta.git esta` | 3.9 GB |
| `data/engagement_dataset.csv` (402,873 rows × 154 columns) | [`data-v1` release](https://github.com/Nathan040105/csgo-engagement-prediction/releases/tag/data-v1): `engagement_dataset.zip` (120 MB), unzip into `data/`; or rebuild with `python build_dataset.py` | 451 MB |
| `tuning_data/<model>_features.csv` | [`data-v1` release](https://github.com/Nathan040105/csgo-engagement-prediction/releases/tag/data-v1): `tuning_data.zip` (233 MB), unzip into `tuning_data/`; or rebuild with `python feature_engineering.py data/engagement_dataset.csv --per-model-dir tuning_data` | 2.2 GB in total |
| Nav meshes | included (`nav_data/`) | 86 MB |

Sizes are as GitHub and Windows show them (1 MB = 1,048,576 bytes).

SHA-256 of `engagement_dataset.csv`:
`68e2bf131488ecc3224d94fbafbcdd92f45c4867e8f4d1c1fa5dfc24ff451e01`

`tuning_data.zip` holds `logistic_regression_features.csv` and
`xgboost_features.csv`. The random forest and transformer CSVs are
byte-identical copies of `xgboost_features.csv`: copy it to
`random_forest_features.csv` and `transformer_features.csv`.

- **Seven maps:** de_ancient, de_dust2, de_inferno, de_mirage, de_nuke,
  de_overpass, de_train. **de_vertigo is excluded** (`config.py`,
  `NAV_MISMATCH_EXCLUDED_MAPS`): a December 2022 update changed its geometry,
  after the ESTA matches were played but before the CS:GO Legacy files the nav
  mesh comes from were frozen, so its nav mesh does not match the demos.
- **Matches:** 602 LAN and 786 online. Engagements per map: inferno 89,208,
  mirage 77,057, nuke 73,166, dust2 56,509, overpass 52,380, ancient 40,176,
  train 14,377.
- **Classes (all matches):** attacker_won 199,401, interrupted 105,506,
  timedout 56,610, victim_won 38,952, round_end 2,200, mutual_kill 204.
- Label codes are alphabetical: 0 attacker_won, 1 interrupted, 2 mutual_kill,
  3 round_end, 4 timedout, 5 victim_won.
- `engagement_dataset.csv` contains player names and SteamIDs as they appear in
  the public ESTA demos. They are dropped before modelling.

---

## Trained models

| File | Model | Features | Test ROC-AUC | Size | Where |
|---|---|---|---|---|---|
| `models/logistic_regression.joblib` | logistic regression | 549 | 0.7921 | 40 KB | repository |
| `models/logistic_regression_after_elimination.joblib` | logistic regression, after elimination | 308 | 0.7922 | 24 KB | repository |
| `models/random_forest.joblib` | random forest | 159 | 0.7909 | 265 MB | [`data-v1` release](https://github.com/Nathan040105/csgo-engagement-prediction/releases/tag/data-v1) |
| `models/xgboost.ubj` | XGBoost | 159 | 0.8199 | 49.1 MB | repository |
| `models/xgboost_after_elimination.ubj` | XGBoost, after elimination | 89 | 0.8209 | 49.8 MB | repository |
| `models/transformer.pt` | FT-Transformer | 159 | 0.8184 | 1.4 MB | repository |
| `models/transformer_after_elimination.pt` | FT-Transformer, after elimination | 111 | 0.8209 | 1.4 MB | repository |

The random forest file is over GitHub's 100 MB limit, so it is attached to the
release instead; download it into `models/` next to its `random_forest.json`.

No training run saved a model, so `save_final_models.py` refits each one
exactly as the held-out test step did (same data, same seed-42 match split, same
tuned hyperparameters, seed 42), compares the refit's test predictions with the
saved ones, and saves it. All seven refits reproduced the original runs' test
predictions exactly (the same predicted class for every one of the 81,693 test
engagements, and identical class probabilities), and each saved file reloads to
the same predictions. `models/manifest.csv` records that comparison, the
file size and the SHA-256 of every file; `models/<name>.json` lists the columns
each model expects, in order. An exact match needs the pinned library versions:
with newer versions the refits differ slightly.

Loading one:

```python
from feature_engineering import load_model_csv
from save_final_models import load_final_model

model, columns = load_final_model("models/xgboost.json")
X, y, groups = load_model_csv("tuning_data/xgboost_features.csv")
proba = model.predict_proba(X[columns])        # one column per class code
```

The `.joblib` files are pickles: load them with the scikit-learn version in
`requirements.txt`, and only from a copy you trust. The XGBoost `.ubj` and the
PyTorch `.pt` files are not pickles.

---

## Reproducing the results

**Getting the repository:**

```bash
git clone https://github.com/Nathan040105/csgo-engagement-prediction.git
cd csgo-engagement-prediction
```

Then download what you need from the [`data-v1` release](https://github.com/Nathan040105/csgo-engagement-prediction/releases/tag/data-v1) (see Data).

**Environment.** Python 3.12 with the pinned versions in `requirements.txt`
(pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0, xgboost 3.4.0, matplotlib
3.10.8, torch 2.13.0, ray 2.57.0):

```bash
pip install -r requirements.txt
```

For the transformer on a GPU, install the torch build that matches your CUDA
driver from pytorch.org (this project used the CUDA 12.6 build on the GPU
machine).

**Compute used.** A two-machine Ray cluster: an 8-CPU, 32 GB head node without a
GPU, and a 4-CPU worker with a GeForce GTX 1050 Ti (4 GB). The tuning scripts
contain that cluster's address; set `ray_address` in `run_final_tuning.py` and
`--ray-address` for `run_feature_elimination.py` to your own, or run without
Ray (`use_ray=False` / `--no-ray`). Steps that refit the transformer should run
on a machine with a GPU, because the refit runs on the machine that launches
the script.

**Steps** (from the repository root):

1. **Nav meshes** (already done; `nav_data/parsed/` is included):
   `python parser.py`
2. **Dataset:** unzip `engagement_dataset.zip` from the release into `data/`,
   or rebuild it: clone ESTA into `esta/` (table above) and run
   `python build_dataset.py`.
3. **Feature CSVs:** unzip `tuning_data.zip` from the release into
   `tuning_data/` and make the two copies described under Data, or rebuild all
   four: `python feature_engineering.py data/engagement_dataset.csv --per-model-dir tuning_data`
4. **Tuning and held-out test** (1,962 hyperparameter combinations × 7 repeats):
   `python run_final_tuning.py logistic_regression random_forest xgboost`, and
   `python run_final_tuning.py transformer` on the GPU machine. With
   `tuning_results_n7/` present, the sweeps and importance steps are skipped
   (same data, saved checkpoints), but each model's final held-out refit runs
   again and rewrites its held-out result and prediction files. To rerun
   everything without touching the published results, change `OUTPUT_DIR` in
   `run_final_tuning.py` first.
5. **Feature elimination:**
   `python run_feature_elimination.py logistic_regression random_forest xgboost`
   and `python run_feature_elimination.py transformer` on the GPU machine.
6. **Accuracy over match progression:** `python accuracy_over_time.py`
7. **Feature importance (RQ2):**
   `python feature_importance_analysis.py --results-dir tuning_results_n7`
   (seconds; reads saved importances only), then
   `python grouped_permutation_importance.py xgboost random_forest logistic_regression --ablate-sides --n-jobs 8`
   and `python grouped_permutation_importance.py transformer --ablate-sides` on
   the GPU machine.
8. **Trained models:**
   `python save_final_models.py logistic_regression random_forest xgboost --n-jobs 8`
   and `python save_final_models.py transformer` on the GPU machine.

---

## Method notes

- **Split.** One match-level split (`GroupShuffleSplit` on `match_id`,
  seed 42): 1,110 matches (321,180 engagements) for training and validation and
  278 matches (81,693 engagements) held out for testing. No match appears on
  both sides.
- **Validation.** There is no fixed validation set. Each hyperparameter
  combination is scored on 7 repeated match-level 75/25 splits of the
  1,110 training matches (seeds 42–48, the same seven splits for every
  combination), so training/validation/test is 60/20/20 of the matches on
  average. The combination with the highest mean ROC-AUC wins, is refit on all
  1,110 matches with seed 42, and is scored once on the test matches.
- **Metric.** Macro one-vs-rest ROC-AUC over the six classes is the primary
  metric; accuracy and macro precision, recall and F1 are also reported.
- **Grids.** Logistic regression: C ∈ {0.001, 0.01, 0.1, 1, 10, 100}, with
  standardised inputs. Random forest: n_estimators {100…500} × max_depth
  {3, 5, 8, 12, 20, none} × min_samples_leaf {1, 2, 5, 10, 20} × max_features
  {sqrt, log2, 0.5, all} (600). XGBoost: n_estimators {100…500} × max_depth
  {3, 4, 5, 6, 7, 9} × learning_rate {0.01, 0.03, 0.1, 0.2, 0.3} × subsample
  {0.6, 0.8, 1.0} × colsample_bytree {0.6, 0.8, 1.0} (1,350), with early
  stopping (20 rounds without improvement in multiclass log-loss on an internal
  15% of the training rows). Transformer: d_token {64, 128} × learning_rate
  {3e-4, 1e-3, 3e-3}, with 2 layers, 8 heads and dropout 0.1 (6), trained with
  AdamW, batch size 512 and early stopping on an internal 15% of the training
  rows.
  <!-- TODO: state max_epochs and patience as they were in the final run.
       tabular_transformer.py's defaults are max_epochs=50, patience=5, but its
       docstring says 20 and 4; the training log lines "epoch k/N" show N. -->
- **Best hyperparameters.** XGBoost: n_estimators 500, max_depth 9,
  learning_rate 0.03, subsample 0.6, colsample_bytree 0.8. Transformer:
  d_token 128, learning_rate 3e-4. Random forest: n_estimators 500, no depth
  limit, min_samples_leaf 20, max_features sqrt. Logistic regression: C = 0.001.
  Several of these sit at the edge of their grid (LR's C, XGBoost's max_depth,
  n_estimators and subsample, RF's min_samples_leaf, the transformer's d_token
  and learning_rate).
- **Feature elimination.** Backward elimination ranked by permutation
  importance (ROC-AUC) on validation folds, with hyperparameters fixed. Each
  step drops 25% of the remaining features (30% for the transformer) and is
  scored on 7 repeated splits (3 for the transformer). Elimination stops when
  validation ROC-AUC falls more than 0.002 below the best step, and the best
  step's features are kept. Only the training matches are used; the selected
  set is then refit once and scored on the test matches.
- **Accuracy over match progression.** Test matches only: engagements in
  chronological order, time normalised to 0–100% of the match, accuracy in a
  window around every 5% point (10%, 20% and 30% windows), then the mean and
  population standard deviation across the 278 matches.
- **Grouped permutation importance.** Each tuned model is refit on the training
  matches; all columns of one category are shuffled together (one shared row
  permutation, 5 repeats, seeds 1000–1004) on the test matches and the fall in
  ROC-AUC is recorded. Every refit reproduced its saved test ROC-AUC exactly.
- **Use of the test matches.** Beyond the one scoring of each final model, the
  test matches were used for the before/after-elimination comparison and the
  importance diagnostics. No model, hyperparameter or feature set was chosen
  from them.

---

## Known issues

- The `<model>_feature_importance.png` charts written by the sweep are labelled
  "ROC-AUC drop", but they plot macro-F1 drops (the CSVs next to them are
  right). Use the figures in `feature_importance_analysis/` instead.
- The comment above the XGBoost grid in `model_tuning.py` says `n_estimators`
  is fixed at 500; the grid actually sweeps 100–500, which is what ran.
- `build_dataset.py` marks a demo as LAN if the text `lan` appears anywhere in
  its file path, so cloning into a folder whose path contains those letters
  (for example `C:\Users\Roland\...`) would mark every demo as LAN. The released
  dataset is not affected. A safer test is
  `"lan" in demo_path.relative_to(DATASET_DIR).parts`.
- `attacker_side` and `victim_side` mirror each other in 98.2% of rows, so
  shuffling one of them creates rows that never occur. This distorts the
  transformer's single-feature importances; the category-level results are
  not affected (see `check_side_features.py` and the side checks in
  `grouped_permutation_importance/`).
- Every model has recall 0 on `mutual_kill` (204 engagements in total, 40 in
  the test matches).
- The transformer's feature elimination used 3 repeats and 30% steps (cost),
  against 7 repeats and 25% for the other models. The logistic regression,
  random forest and XGBoost eliminations ran with an earlier `model_tuning.py`
  that differs from this one only in an error message.

---

## Licence

- **Code:** MIT (see `LICENSE`), except `parser.py`.
- **`parser.py`** is adapted from [fpapouin/pynav](https://github.com/fpapouin/pynav)
  (itself based on [mrazza/gonav](https://github.com/mrazza/gonav)). pynav has
  no licence, so its original parts remain its author's. The changes made here
  are listed at the top of `parser.py`: chiefly the fix for unnamed nav areas
  (place id 0) being silently dropped, which affected 6–17% of each map's
  areas.
  <!-- If the pynav author grants permission or adds a licence, say so here. -->
- **Data, results and models** (`engagement_dataset.csv`, `tuning_data/`,
  `tuning_results_n7/`, `models/`) are derived from ESTA, which is licensed
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), and are
  shared under the same licence.
- **`nav_data/`** contains Valve's CS:GO navigation meshes and their parsed
  form, included only so the pipeline can be rerun.

---

## References

- Xenopoulos, P., & Silva, C. (2022). *ESTA: An Esports Trajectory and Action
  Dataset.* arXiv:2209.09861.
- Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). *Revisiting
  Deep Learning Models for Tabular Data.* NeurIPS 2021.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.*
  KDD 2016.
- Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.
- Cawley, G. C., & Talbot, N. L. C. (2010). *On Over-fitting in Model Selection
  and Subsequent Selection Bias in Performance Evaluation.* JMLR, 11, 2079–2107.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.*
  JMLR, 12, 2825–2830.

## Use of AI tools

<!-- Write this to match the AI-use declaration in the report (Wits form):
     which tools, for what (e.g. code development, debugging, analysis), and
     that the experiments were run and the results checked by the author. -->
