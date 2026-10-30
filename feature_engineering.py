"""
feature_engineering.py

Phase B preprocessing: turns the raw engagement_dataset.csv (one row
per engagement, ~154 raw columns) into model-ready (X, y, groups)
for model_tuning.py.

This file makes the preprocessing decisions that were previously just
discussed, not implemented, explicit and in one place:

    1. Identifier / non-generalizable columns are dropped from X.
    2. match_id is pulled out separately as the `groups` array for
       match-aware train/test splitting (see build_dataset.py /
       feature_extractor.py, which now write match_id + round_num
       per row).
    3. Raw nav_area_id columns are dropped (arbitrary integers, no
       ordinal meaning); the *derived* nav features (same_callout,
       nav_path_distance, nav_los_proxy_ratio, connectivity,
       near_cover, exposed) are kept.
    4. Categorical columns are encoded one of two ways, selected via
       the `encoding` parameter:

       "onehot" - for logistic regression. One-hot rather than
       label/ordinal encoding because a linear model reads integer
       codes as distances - label-encoding a nominal category (e.g.
       attacker_callout) would tell it "callout 12" is twice
       "callout 6", which is meaningless.

       "ordinal" - for random forest and XGBoost, and this is also
       the correct encoding to hand a future Tabular Transformer.
       Trees split on thresholds, not distances, so an arbitrary
       integer code costs a tree nothing - and it means each
       category gets to compete for a *single* split instead of
       being diluted across dozens of near-empty one-hot columns
       (attacker_callout alone is ~7-15+ categories per map). This
       is NOT a three-way split by model type: embeddings (what a
       Transformer's categorical inputs actually want) are looked
       up by integer index, not one-hot vector - so "ordinal" is
       simultaneously the right encoding for trees today and
       embedding-ready for the Transformer later. Only the linear
       baseline needs its own branch.

       Practically: call prepare_features() twice, once per
       encoding, and route each model to the matching X - see the
       usage example below.
    5. Boolean columns (round-tripped through CSV as "True"/"False"
       strings) are coerced to 0/1 ints.
    6. clock_time ("M:SS" string) is converted to clock_time_seconds
       (numeric, seconds remaining).
    7. Known -1 sentinel columns (see MISSING_SENTINEL_COLUMNS) each
       get a companion `<col>_missing` flag rather than being
       silently imputed. Some of these -1s are real data-quality
       gaps (e.g. ping, nav lookups); others are legitimate game
       states (-1 for "no living teammate" on
       attacker_closest_teammate). Either way, a linear model
       shouldn't be handed a bare -1 as if it were a normal small
       distance - the flag makes the special case explicit instead
       of silently distorting the coefficient. The original -1
       values are left as-is (no imputation decision is silently
       made here).
    8. Label strategy defaults to "multiclass": the full
       engagement_status column (all six statuses, all rows kept)
       as the target. "binary" (attacker_won, rows filtered to
       attacker_won/victim_won engagements only) is still available
       for comparison but is no longer the default. See
       build_label() - multiclass mode prints per-class counts and
       warns loudly about any class too thin to survive a
       train/test split (as of the last real dataset checked
       against this pipeline, mutual_kill was a single row -
       structurally unusable as a target class until that's
       addressed, either by dropping it or merging it into another
       category).

Usage
-----
    from feature_engineering import write_model_feature_csvs, load_model_csv
    from model_tuning import run_full_tuning

    # label_mode defaults to "multiclass" (engagement_status, all
    # six statuses) - writes logistic_regression_features.csv,
    # random_forest_features.csv, xgboost_features.csv,
    # transformer_features.csv into tuning_data/
    write_model_feature_csvs("data/engagement_dataset.csv", "tuning_data")

    for model_name in ["logistic_regression", "random_forest", "xgboost"]:
        X, y, groups = load_model_csv(f"tuning_data/{model_name}_features.csv")
        run_full_tuning(model_name, X, y, output_dir="tuning_results",
                         n_repeats=100, groups=groups)

    # pass label_mode="binary" explicitly for the attacker_won comparison instead:
    write_model_feature_csvs("data/engagement_dataset.csv", "tuning_data_binary",
                              label_mode="binary")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


# ==============================================================
# Column Groupings
#
# Written out explicitly against the real column list produced by
# feature_extractor.py, rather than inferred from dtypes at runtime,
# so a schema change in the extractor is a loud KeyError here rather
# than a silent behavior change in what gets encoded vs. dropped.
# ==============================================================

# Columns identifying *who*/*which exact demo tick*, not the
# engagement's game-state - these don't generalize across matches
# and risk the model keying off player/demo identity rather than
# the actual situation.
IDENTIFIER_COLUMNS = [
    "match_id",        # pulled out separately as `groups`, not part of X
    "round_num",        # metadata only, not a model feature
    "start_tick",        # raw per-demo tick counter, arbitrary across demos
    "frame_tick",        # same as start_tick
    "attacker_name",
    "victim_name",
    "winner_name",
    "attacker_steamid",
    "victim_steamid",
]

# Raw nav mesh area ids - arbitrary integers with no ordinal
# meaning (area 500 isn't "more" of anything than area 10). The
# *derived* nav features below are kept.
RAW_NAV_ID_COLUMNS = [
    "attacker_nav_area_id",
    "victim_nav_area_id",
]

# The label lives here in one form or another - always excluded
# from X regardless of label_mode (either it *is* y, or it would
# leak y).
LABEL_SOURCE_COLUMNS = [
    "engagement_status",
    "attacker_won",
]

# Needs numeric conversion before it can be used as a feature.
CLOCK_TIME_COLUMN = "clock_time"

CATEGORICAL_COLUMNS = [
    "map_name",
    "attacker_side",
    "victim_side",
    "attacker_active_weapon",
    "victim_active_weapon",
    "attacker_active_weapon_class",
    "victim_active_weapon_class",
    "attacker_primary_weapon",
    "attacker_secondary_weapon",
    "victim_primary_weapon",
    "victim_secondary_weapon",
    "attacker_callout",
    "victim_callout",
]

# Written to CSV as Python bool -> "True"/"False" strings by
# csv.DictWriter. Coerced to 0/1 int columns.
BOOLEAN_COLUMNS = [
    "is_lan",
    "attacker_has_helmet", "victim_has_helmet",
    "attacker_scoped", "victim_scoped",
    "attacker_has_bomb", "victim_has_bomb", "bomb_planted",
    "attacker_airborne", "victim_airborne",
    "attacker_ducking", "victim_ducking",
    "attacker_walking", "victim_walking",
    "attacker_reloading", "victim_reloading",
    "attacker_blinded", "victim_blinded",
    "attacker_near_cover", "victim_near_cover",
    "attacker_exposed", "victim_exposed",
    "attacker_standing", "victim_standing",
    "attacker_planting", "victim_planting",
    "attacker_defusing", "victim_defusing",
    "attacker_buyzone", "victim_buyzone",
    "attacker_bombzone", "victim_bombzone",
    "attacker_has_flash", "attacker_has_smoke", "attacker_has_he",
    "attacker_has_molotov", "attacker_has_incendiary", "attacker_has_decoy",
    "victim_has_flash", "victim_has_smoke", "victim_has_he",
    "victim_has_molotov", "victim_has_incendiary", "victim_has_decoy",
    "same_callout",
]

# Columns where -1 is a documented sentinel (see feature_extractor.py
# / nav_data.py) rather than a genuine data value. Each gets a
# companion `<col>_missing` flag; the original column is left
# untouched (still -1 where it was -1).
MISSING_SENTINEL_COLUMNS = [
    "attacker_magazine_pct", "attacker_reserve_pct",
    "victim_magazine_pct", "victim_reserve_pct",
    "attacker_closest_teammate", "victim_closest_teammate",
    "attacker_closest_enemy", "victim_closest_enemy",
    "attacker_average_teammate_distance", "victim_average_teammate_distance",
    "attacker_average_enemy_distance", "victim_average_enemy_distance",
    "nav_path_distance", "nav_los_proxy_ratio",
    "attacker_area_connectivity", "victim_area_connectivity",
    "attacker_ping", "victim_ping",
]


# ==============================================================
# Step Functions
# ==============================================================

def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """
    Loads the raw engagement CSV.

    Uses keep_default_na=False + na_values=[] so that legitimate
    empty-string feature values (e.g. winner_name for a non-kill
    engagement) aren't silently coerced to NaN and confused with
    the pipeline's explicit -1 sentinel convention.
    """

    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[])

    missing_expected = [
        col for col in (
            IDENTIFIER_COLUMNS
            + RAW_NAV_ID_COLUMNS
            + LABEL_SOURCE_COLUMNS
            + CATEGORICAL_COLUMNS
            + BOOLEAN_COLUMNS
            + MISSING_SENTINEL_COLUMNS
            + [CLOCK_TIME_COLUMN]
        )
        if col not in df.columns
    ]

    if missing_expected:
        raise ValueError(
            "CSV is missing columns this module expects: "
            f"{missing_expected}. Either the extractor schema has "
            "changed (update the column lists at the top of this "
            "file to match), or this CSV predates the match_id/"
            "round_num addition to build_dataset.py."
        )

    return df


def parse_clock_time(series: pd.Series) -> pd.Series:
    """
    Converts "M:SS" round-clock strings to numeric seconds
    remaining. Falls back to -1 (consistent with this pipeline's
    existing sentinel convention) for anything that doesn't parse,
    rather than raising and killing the whole run over one bad row.
    """

    def _parse(value: str) -> float:

        try:
            minutes_str, seconds_str = str(value).split(":")
            return int(minutes_str) * 60 + int(seconds_str)
        except (ValueError, AttributeError):
            return -1.0

    return series.apply(_parse)


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a `<col>_missing` 0/1 column for every column in
    MISSING_SENTINEL_COLUMNS, flagging exactly the rows where that
    column holds the pipeline's -1 sentinel. The original column is
    left unmodified - no imputation decision is made here.
    """

    df = df.copy()

    for col in MISSING_SENTINEL_COLUMNS:

        df[f"{col}_missing"] = (
            pd.to_numeric(df[col], errors="coerce") == -1
        ).astype(int)

    return df


def encode_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerces "True"/"False" string columns to 0/1 ints.
    """

    df = df.copy()

    for col in BOOLEAN_COLUMNS:
        df[col] = (
            df[col]
            .astype(str)
            .map({"True": 1, "False": 0})
            .fillna(0)
            .astype(int)
        )

    return df


def encode_categoricals_onehot(
    df: pd.DataFrame,
    columns: list[str] = CATEGORICAL_COLUMNS,
) -> pd.DataFrame:
    """
    One-hot encodes the given categorical columns. For logistic
    regression - see the module docstring (point 4) for why.
    """

    return pd.get_dummies(df, columns=columns, prefix=columns)


def encode_categoricals_ordinal(
    df: pd.DataFrame,
    columns: list[str] = CATEGORICAL_COLUMNS,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """
    Integer-codes the given categorical columns (0..k-1 per column).
    For random forest, XGBoost, and as embedding-ready indices for a
    future Tabular Transformer - see the module docstring (point 4).

    Codes are assigned from each column's *sorted* unique values, so
    the mapping is deterministic given the same input data (matters
    because this runs once on the full dataset before model_tuning's
    100 repeated splits - every split needs to see the same codes).

    Returns (df, mappings) where mappings is
    {column: {category: code}}, so a category can be recovered from
    a code later (e.g. reading back which callout a tree split on).
    """

    df = df.copy()
    mappings: dict[str, dict[str, int]] = {}

    for col in columns:

        categories = sorted(df[col].unique())
        mapping = {category: code for code, category in enumerate(categories)}

        mappings[col] = mapping
        df[col] = df[col].map(mapping).astype(int)

    return df, mappings


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops match_id, round_num, raw tick counters, player/steamid
    identity columns, and raw nav area ids from X.

    match_id is expected to have already been pulled out separately
    (see prepare_features) before this is called - it's dropped
    here too as a safety net in case this function is called on its
    own.
    """

    drop_cols = [
        col
        for col in (IDENTIFIER_COLUMNS + RAW_NAV_ID_COLUMNS)
        if col in df.columns
    ]

    return df.drop(columns=drop_cols)


def build_label(
    df: pd.DataFrame,
    label_mode: Literal["binary", "multiclass"],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Filters rows (if needed) and returns (df_filtered, y) for the
    requested label strategy.

    binary
        Keeps only attacker_won/victim_won engagements (a decisive
        1v1 kill happened). Drops mutual_kill/interrupted/timedout/
        round_end rows, since "who won" isn't well-defined for
        those. y = attacker_won, as 0/1 int.

    multiclass
        Keeps every row. y = engagement_status, encoded to
        0..5 ints (mapping printed so it's not implicit).
    """

    if label_mode == "binary":

        filtered = df[
            df["engagement_status"].isin(["attacker_won", "victim_won"])
        ].copy()

        y = (
            filtered["attacker_won"]
            .astype(str)
            .map({"True": 1, "False": 0})
            .astype(int)
        )

        return filtered, y

    if label_mode == "multiclass":

        filtered = df.copy()

        categories = sorted(filtered["engagement_status"].unique())
        mapping = {status: i for i, status in enumerate(categories)}

        counts = filtered["engagement_status"].value_counts()

        print(f"[feature_engineering] multiclass label mapping: {mapping}")
        print("[feature_engineering] class counts:")
        for status in categories:
            print(f"    {status} ({mapping[status]}): {counts[status]}")

        # A class needs to survive being split across ~100 repeated
        # train/test resamples (see model_tuning.py) with enough
        # examples on both sides to produce a defined per-class
        # metric. There's no single "safe" threshold, but anything
        # under 10 is loud-warning territory, and a class with 1-2
        # examples literally cannot appear in both a train split and
        # a test split - it's not "rare", it's structurally unusable
        # as one of the target classes. Not raised as an error: this
        # is a labeling/methodology call (drop the class? merge it
        # into another? keep it and caveat the metric?), not a bug in
        # preprocessing, and it shouldn't silently decide that for you.
        thin_classes = counts[counts < 10]

        if not thin_classes.empty:

            print(
                "\n[feature_engineering] WARNING: the following "
                "classes have very few examples and may not "
                "survive train/test splitting with a defined metric:"
            )

            for status, count in thin_classes.items():

                severity = (
                    "UNUSABLE - cannot appear in both train and test"
                    if count < 3
                    else "risky - may be missing from some splits"
                )

                print(f"    {status}: {count} rows ({severity})")

            print(
                "Consider dropping and/or merging these before "
                "modeling - see build_label()'s docstring.\n"
            )

        y = filtered["engagement_status"].map(mapping).astype(int)

        return filtered, y

    raise ValueError(
        f"label_mode must be 'binary' or 'multiclass', got {label_mode!r}"
    )


# ==============================================================
# Public API
# ==============================================================

def prepare_features(
    csv_path: str | Path,
    label_mode: Literal["binary", "multiclass"] = "multiclass",
    encoding: Literal["onehot", "ordinal"] = "onehot",
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, dict[str, Any]]:
    """
    Full Phase B pipeline: raw CSV -> model-ready (X, y, groups).

    Parameters
    ----------
    csv_path
        Path to the raw engagement_dataset.csv.

    label_mode
        "multiclass" (default) uses the full engagement_status
        column (all six statuses, all rows kept) as the target.
        "binary" instead uses attacker_won, filtered to decisive
        attacker_won/victim_won engagements only - kept available
        for comparison, but no longer the default. See
        build_label() - multiclass mode prints per-class counts
        and warns about classes too thin to survive a train/test
        split (mutual_kill was a single row in the last real CSV
        checked against this pipeline).

    encoding
        "onehot" for logistic regression, "ordinal" for random
        forest / XGBoost / a future embedding-based Transformer -
        see the module docstring (point 4) for why these are the
        only two encodings needed, not one per model.

    Returns
    -------
    X
        DataFrame of encoded, numeric-only features. Safe to pass
        straight to model_tuning.run_all_models.

    y
        Integer-encoded label Series, aligned to X's index.

    groups
        numpy array of match_id values, aligned to X's index - pass
        this straight through as model_tuning's `groups` parameter
        for match-aware splitting.

    meta
        Dict with bookkeeping info: n_rows_before_label_filter,
        n_rows_after, dropped_columns, categorical_columns_encoded,
        n_features, encoding, and (for encoding="ordinal" only)
        categorical_mappings - {column: {category: code}}, so a
        code can be traced back to the category it came from. Print/
        inspect this before trusting a run - it's the cheapest
        sanity check available.
    """

    df = load_dataset(csv_path)
    n_rows_before = len(df)

    df, y = build_label(df, label_mode)
    n_rows_after = len(df)

    groups = df["match_id"].to_numpy()

    df = add_missing_indicators(df)
    df = encode_booleans(df)

    df["clock_time_seconds"] = parse_clock_time(df[CLOCK_TIME_COLUMN])

    df = drop_identifier_columns(df)
    df = df.drop(columns=[CLOCK_TIME_COLUMN] + LABEL_SOURCE_COLUMNS)

    if encoding == "onehot":
        df = encode_categoricals_onehot(df)
        categorical_mappings = None
    elif encoding == "ordinal":
        df, categorical_mappings = encode_categoricals_ordinal(df)
    else:
        raise ValueError(
            f"encoding must be 'onehot' or 'ordinal', got {encoding!r}"
        )

    non_numeric = df.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    if non_numeric:
        raise ValueError(
            "The following columns are still non-numeric after "
            f"encoding - this means the column lists at the top of "
            f"this file are out of sync with the real CSV schema: "
            f"{non_numeric}"
        )

    X = df.reset_index(drop=True)
    y = y.reset_index(drop=True)

    meta = {
        "label_mode": label_mode,
        "encoding": encoding,
        "n_rows_before_label_filter": n_rows_before,
        "n_rows_after_label_filter": n_rows_after,
        "n_features": X.shape[1],
        "n_unique_matches": len(set(groups)),
        "categorical_columns_encoded": CATEGORICAL_COLUMNS,
        "categorical_mappings": categorical_mappings,
        "missing_indicator_columns_added": [
            f"{c}_missing" for c in MISSING_SENTINEL_COLUMNS
        ],
    }

    return X, y, groups, meta


# ==============================================================
# Per-Model CSVs
#
# Rather than making every caller re-run prepare_features() and
# juggle DataFrames in memory, this writes one self-contained CSV
# per model family - each file is the encoded features plus a
# `label` column and a `match_id` column, so it's a complete,
# inspectable artifact on its own (open it, and everything's
# there), not just an in-memory intermediate.
#
# `label` and `match_id` are bookkeeping, not features - see
# load_model_csv(), which drops them back out into (X, y, groups)
# the same way prepare_features() does.
# ==============================================================

# Which encoding each model family needs - see the module
# docstring (point 4). random_forest/xgboost/transformer all get
# "ordinal": trees split on thresholds so the same integer codes
# work for RF and XGBoost, and those same codes are exactly what a
# future Transformer's embedding lookup wants. transformer is
# included provisionally - same encoding as the tree models costs
# nothing extra to produce, but the exact input contract isn't
# confirmed until the architecture is finalized with the
# supervisor, so treat that one file as a placeholder.
DEFAULT_MODEL_ENCODINGS: dict[str, Literal["onehot", "ordinal"]] = {
    "logistic_regression": "onehot",
    "random_forest": "ordinal",
    "xgboost": "ordinal",
    "transformer": "ordinal",
}


def write_model_feature_csvs(
    csv_path: str | Path,
    output_dir: str | Path,
    label_mode: Literal["binary", "multiclass"] = "multiclass",
    model_encodings: dict[str, Literal["onehot", "ordinal"]] | None = None,
) -> dict[str, Path]:
    """
    Writes one <model>_features.csv per model family into
    output_dir, ready to hand straight to model_tuning.run_full_tuning().

    Only computes each distinct encoding once - random_forest,
    xgboost, and transformer all share the "ordinal" encoding by
    default, so that's one prepare_features() call, written out to
    three files, rather than three redundant (and potentially
    inconsistent) recomputations.

    Parameters
    ----------
    model_encodings
        {model_name: "onehot"|"ordinal"}. Defaults to
        DEFAULT_MODEL_ENCODINGS. Pass a subset to only write some
        models' files.

    Returns
    -------
    {model_name: path_written}, plus one extra
    "categorical_mappings_<encoding>" entry per ordinal-encoded
    group (see below) - callers that only expect model names as
    keys can safely ignore any key not in model_encodings.

    Also writes, per ordinal-encoded group
    -----------------------------------------------------------
    categorical_mappings_<encoding>.json - the {column: {category:
    code}} dict prepare_features() produces for that group (e.g.
    random_forest/xgboost/transformer, which all share "ordinal"
    by default). This isn't needed to train or eliminate features -
    run_feature_importance_analysis / run_recursive_feature_elimination
    in feature_elimination.py both work on the encoded columns as-is
    and never need to know what a given code means. It's needed for
    the step *after* that: once elimination or permutation importance
    tells you e.g. "attacker_callout matters a lot" or a fitted
    tree's top split is "attacker_callout <= 4.5", the encoded column
    name alone doesn't tell you which callout that is, and codes are
    assigned by sorted-alphabetical order (see
    encode_categoricals_ordinal) - not memorable, not stable in your
    head, and not written down anywhere else. Without this file,
    that information only ever existed in memory for the process
    that ran prepare_features() and is gone the moment that process
    exits - the same "silently lost, not silently wrong" failure
    shape as the earlier place_id sentinel bug, just at the
    interpretation stage instead of the parsing stage. Not written
    for onehot groups - one-hot columns already carry the category
    in their own name (e.g. attacker_callout_LongA), so there's
    nothing to reconstruct.

    To read a code back to its category later:

        import json
        mappings = json.load(open("tuning_data/categorical_mappings_ordinal.json"))
        code_to_category = {v: k for k, v in mappings["attacker_callout"].items()}
        code_to_category[4]  # -> whatever category sorted to code 4
    """

    if model_encodings is None:
        model_encodings = DEFAULT_MODEL_ENCODINGS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoding_to_models: dict[str, list[str]] = {}
    for model_name, encoding in model_encodings.items():
        encoding_to_models.setdefault(encoding, []).append(model_name)

    written: dict[str, Path] = {}

    for encoding, model_names in encoding_to_models.items():

        X, y, groups, meta = prepare_features(csv_path, label_mode, encoding)

        out_df = X.copy()
        out_df["label"] = y.values
        out_df["match_id"] = groups

        for model_name in model_names:

            out_path = output_dir / f"{model_name}_features.csv"
            out_df.to_csv(out_path, index=False)
            written[model_name] = out_path

            print(
                f"[feature_engineering] wrote {out_path} "
                f"({out_df.shape[0]} rows x {X.shape[1]} features, "
                f"encoding={encoding})"
            )

        categorical_mappings = meta["categorical_mappings"]

        if categorical_mappings is not None:

            mapping_path = output_dir / f"categorical_mappings_{encoding}.json"

            with open(mapping_path, "w", encoding="utf-8") as file:
                json.dump(categorical_mappings, file, indent=2, sort_keys=True)

            written[f"categorical_mappings_{encoding}"] = mapping_path

            print(
                f"[feature_engineering] wrote {mapping_path} "
                f"(category -> code lookup for {', '.join(model_names)})"
            )

    return written


def load_model_csv(
    path: str | Path,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Reads a CSV written by write_model_feature_csvs() back into
    (X, y, groups) - drops the `label`/`match_id` bookkeeping
    columns out of X so they can never accidentally get fit on.
    """

    df = pd.read_csv(path)

    y = df["label"]
    groups = df["match_id"].to_numpy()
    X = df.drop(columns=["label", "match_id"])

    return X, y, groups


# ==============================================================
# Standalone CLI
# ==============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Phase B preprocessing for the CS:GO engagement dataset."
    )
    parser.add_argument("csv_path", help="Path to engagement_dataset.csv")
    parser.add_argument(
        "--label-mode",
        choices=["binary", "multiclass"],
        default="multiclass",
    )
    parser.add_argument(
        "--per-model-dir",
        default=None,
        help="Write one <model>_features.csv per model family "
             "(logistic_regression/random_forest/xgboost/transformer) "
             "into this directory. This is the default mode.",
    )
    parser.add_argument(
        "--encoding",
        choices=["onehot", "ordinal"],
        default=None,
        help="Only used with --output-prefix (single-file mode). "
             "onehot for logistic regression; ordinal for random "
             "forest/XGBoost/Transformer embeddings.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Single-file mode instead of --per-model-dir: writes "
             "<prefix>_X.csv / _y.csv / _groups.csv for one encoding.",
    )

    args = parser.parse_args()

    if args.output_prefix:

        encoding = args.encoding or "onehot"
        X, y, groups, meta = prepare_features(
            args.csv_path, args.label_mode, encoding
        )

        print("\n=== Feature Engineering Summary ===")
        for key, value in meta.items():
            print(f"{key}: {value}")
        print(f"\nX shape: {X.shape}")
        print(f"y value counts:\n{y.value_counts()}")

        X.to_csv(f"{args.output_prefix}_X.csv", index=False)
        y.to_csv(f"{args.output_prefix}_y.csv", index=False)
        pd.Series(groups, name="match_id").to_csv(
            f"{args.output_prefix}_groups.csv", index=False
        )
        print(f"\nWrote {args.output_prefix}_X.csv / _y.csv / _groups.csv")

    else:

        output_dir = args.per_model_dir or "tuning_data"
        written = write_model_feature_csvs(
            args.csv_path, output_dir, args.label_mode
        )
        print(f"\nWrote {len(written)} per-model feature CSVs to {output_dir}/")
