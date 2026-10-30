"""
tabular_transformer.py

A small, from-scratch PyTorch implementation of an FT-Transformer
(Feature Tokenizer + Transformer - Gorishniy et al. 2021,
"Revisiting Deep Learning Models for Tabular Data") for the CS:GO
engagement dataset, wrapped in a scikit-learn-compatible class
(fit/predict/predict_proba) so it drops straight into
model_tuning.py's MODEL_BUILDERS dict alongside logistic
regression, random forest, and XGBoost.

Why this architecture, not a heavier framework (pytorch-tabular,
etc.): FT-Transformer is the design the tabular deep-learning
literature treats as the standard strong baseline - stronger and
better-benchmarked than the older TabTransformer, which only
tokenizes categorical features and just concatenates continuous
ones raw. A full AutoML-style framework brings its own Trainer/
DataModule abstractions and a much heavier dependency chain
(pytorch-lightning, omegaconf, ...) that fights the fit/predict/
predict_proba contract the rest of this pipeline is built around.
This file is ~one architecture, hand-written, so every line is
inspectable - consistent with how the rest of the pipeline is
built.

Deliberately small architecture (2 layers, 4 heads, 32-dim tokens
by default): at a few thousand rows this model is one baseline
among several (logistic regression, random forest, XGBoost), not
a from-scratch attempt at a state-of-the-art result - a deeper
network here would mostly just overfit and burn compute budget.

How feature tokenization works here
------------------------------------
Every continuous column gets its own learned (weight, bias) pair
that projects its single scalar value into a d_token-dimensional
vector - not just concatenated raw. Every categorical column
(already ordinal-encoded by feature_engineering.py's
encoding="ordinal" - see its docstring for why that encoding is
also what an embedding table wants) gets its own nn.Embedding
table. A learned [CLS] token is prepended, all tokens attend to
each other through a standard Transformer encoder, and the [CLS]
token's output representation is fed through a small classification
head. This is the "feature tokenizer" design specifically because
it lets every feature - continuous or categorical - participate in
self-attention on equal footing, rather than only letting
categoricals interact through attention and bolting continuous
features on afterward.
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin


# ==============================================================
# Model
# ==============================================================


class _FeatureTokenizer(nn.Module):
    """
    Projects continuous scalars and categorical codes into a shared
    d_token-dimensional embedding space, plus a learned [CLS] token.
    """

    def __init__(
        self,
        n_continuous: int,
        categorical_cardinalities: list[int],
        d_token: int,
    ) -> None:

        super().__init__()

        self.n_continuous = n_continuous

        # One (weight, bias) pair per continuous feature - each
        # feature's scalar value gets its own learned linear
        # projection into d_token space, not a shared one.
        self.continuous_weight = nn.Parameter(torch.empty(n_continuous, d_token))
        self.continuous_bias = nn.Parameter(torch.empty(n_continuous, d_token))
        nn.init.kaiming_uniform_(self.continuous_weight, a=math.sqrt(5))
        nn.init.zeros_(self.continuous_bias)

        # One combined embedding table instead of a ModuleList of many
        # small ones - see forward() for why. Column i's raw code (in
        # [0, cardinality_i)) gets shifted by the sum of every earlier
        # column's cardinality before the lookup, so each column's codes
        # land in its own disjoint slice of one shared table. This is
        # mathematically identical to per-column embeddings (each slice
        # is trained independently, same as a separate nn.Embedding
        # would be) but replaces a Python-level loop over every
        # categorical column, on every single batch, with one lookup.
        # At ~100+ categorical columns and ~2,000+ batches/epoch, that
        # loop's interpreter overhead was a real cost, not a rounding
        # error.
        self.n_categorical = len(categorical_cardinalities)

        if self.n_categorical > 0:
            total_cardinality = sum(categorical_cardinalities)
            self.categorical_embedding = nn.Embedding(total_cardinality, d_token)
            offsets = torch.tensor(
                list(itertools.accumulate([0] + categorical_cardinalities[:-1])),
                dtype=torch.long,
            )
            self.register_buffer("cat_offsets", offsets)
        else:
            self.categorical_embedding = None

        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        x_continuous: torch.Tensor,
        x_categorical: torch.Tensor,
    ) -> torch.Tensor:

        batch_size = x_continuous.shape[0]

        tokens = [self.cls_token.expand(batch_size, -1, -1)]

        if self.n_continuous > 0:
            # (batch, n_continuous, 1) * (n_continuous, d_token) -> (batch, n_continuous, d_token)
            cont_tokens = (
                x_continuous.unsqueeze(-1) * self.continuous_weight
                + self.continuous_bias
            )
            tokens.append(cont_tokens)

        if self.categorical_embedding is not None:
            # Single lookup against the combined table (see __init__)
            # instead of one Python-level nn.Embedding call per column.
            # x_categorical + self.cat_offsets shifts each column's codes
            # into its own slice of the table; nn.Embedding on a
            # (batch, n_categorical) index tensor returns
            # (batch, n_categorical, d_token) directly, so no torch.stack
            # is needed either.
            cat_tokens = self.categorical_embedding(x_categorical + self.cat_offsets)
            tokens.append(cat_tokens)

        return torch.cat(tokens, dim=1)


class TabularTransformer(nn.Module):
    """
    Feature Tokenizer + standard nn.TransformerEncoder + a small
    classification head on the [CLS] token's output.
    """

    def __init__(
        self,
        n_continuous: int,
        categorical_cardinalities: list[int],
        n_classes: int,
        d_token: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        ff_multiplier: int = 2,
    ) -> None:

        super().__init__()

        self.tokenizer = _FeatureTokenizer(
            n_continuous, categorical_cardinalities, d_token
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, n_classes),
        )

    def forward(
        self,
        x_continuous: torch.Tensor,
        x_categorical: torch.Tensor,
    ) -> torch.Tensor:

        tokens = self.tokenizer(x_continuous, x_categorical)
        encoded = self.encoder(tokens)
        cls_output = encoded[:, 0]

        return self.head(self.norm(cls_output))


# ==============================================================
# scikit-learn-Compatible Wrapper
# ==============================================================


class TabularTransformerClassifier(ClassifierMixin, BaseEstimator):
    """
    Wraps TabularTransformer with fit/predict/predict_proba, so it
    drops into model_tuning.py's MODEL_BUILDERS exactly like
    LogisticRegression/RandomForestClassifier/XGBClassifier do.
    Inherits sklearn's ClassifierMixin/BaseEstimator - not just for
    style, but because sklearn's own permutation_importance (used
    by model_tuning.py's run_feature_importance_analysis) requires
    __sklearn_tags__, which only comes from that inheritance; a
    plain hand-written class with the right method names isn't
    enough.

    ORDER MATTERS: ClassifierMixin must come BEFORE BaseEstimator.
    With the order reversed, sklearn (1.6+) never sets the
    "classifier" estimator tag, is_classifier() returns False, and
    any scorer that needs predict_proba (roc_auc_ovr) raises
    "Got a regressor with response_method=predict_proba" - while
    scorers that only call predict (f1_macro) keep working, which
    is how this went unnoticed until roc_auc_ovr was used for the
    feature-elimination importance step.

    Handles internally, so the caller never has to think about it:
      - Standardizing continuous columns (fit on training data only,
        applied consistently at predict time) - a Transformer trains
        far worse on raw mixed-scale inputs (e.g. engagement_distance
        in the hundreds next to attacker_magazine_pct in [0, 1]).
      - Mapping y's raw label values to a contiguous 0..(k-1) index
        space for CrossEntropyLoss, then mapping predictions back -
        matters when a particular training fold doesn't happen to
        contain every class (see feature_engineering.py's thin-class
        warning): self.classes_ only covers what was actually seen
        in training, matching sklearn's own convention, so
        model_tuning.py's existing NaN-not-crash handling for a
        resulting predict_proba shape mismatch (see _compute_metrics)
        applies here exactly as it does for the other three models -
        no special-casing needed for that scenario.
      - An internal train/validation carve-out for early stopping.
        This split is a plain random split, not match-aware - a
        known, deliberate simplification: the *outer* split (done by
        model_tuning.py's _make_split) is what's group-aware and
        actually evaluated, this internal one only decides when to
        stop training and never contributes to a reported metric.

    Parameters split into two groups:
      - True hyperparameters (meant to be swept): d_token, n_layers,
        n_heads, dropout, learning_rate.
      - Fixed training config (deliberately NOT swept, to keep a
        grid sweep computationally tractable - each "fit" here is a
        real multi-epoch training loop, not an instant sklearn
        .fit()): batch_size, max_epochs, patience, num_threads,
        progress_every_n_batches.

    batch_size, max_epochs, patience defaults were raised/lowered
    from an earlier 128/50/8 after measuring real per-epoch cost
    against the actual ~322,000-row training split this project
    uses: ~790-800s/epoch even using every core on an 8-core
    machine. 512 (vs 128) means 4x fewer Python-level batch
    iterations per epoch for the same data, cutting per-batch
    dispatch overhead without changing what data the model sees.
    20/4 (vs 50/8) bounds worst-case per-combination cost at under
    half what it was, and stops chasing epochs of marginal
    improvement sooner - measured behavior showed the loss still
    inching down at epoch 14 of 50 with a patience of 8, exactly the
    slow-plateau pattern a tighter patience is meant to cut short.

    num_threads
        Calls torch.set_num_threads(num_threads) at the start of
        fit()/predict()/predict_proba(), every time - torch's thread
        pool is process-global and Ray reuses worker processes
        across many tasks, so this can't be set once at import time
        the way an environment variable could; it has to be
        re-asserted per call to be reliable regardless of what ran
        on this worker process before it. Defaults to 1 deliberately:
        measured intra-op scaling using all 8 cores of a single
        machine for one fit came to roughly 5.6x, not 8x - a
        transformer this size doesn't have enough matmul work per
        operation to keep 8 threads busy, so most of that "5.6x" is
        actually wasted thread-coordination overhead. The better use
        of an 8-core (or 12-CPU cluster) machine is running up to 8
        (or 12) independent single-threaded fits in parallel via
        Ray's own task scheduling instead - true task-level
        parallelism instead of a mediocre intra-op speedup on one
        task at a time. Raise this only if you're intentionally
        running fewer concurrent tasks than you have cores for.

    progress_every_n_batches
        With batch_size=512 on the real ~226,000-row (post-internal-
        validation-split) training set, that's ~442 batches/epoch -
        enough that waiting for a whole epoch to complete before any
        output appears can look indistinguishable from a hang on a
        slow combination. Prints running training loss every this
        many batches, plus one line the moment fit() actually starts
        (row counts, batches/epoch, device) so the gap between a
        combination starting and its first real output is never
        silent. 0 disables mid-epoch printing entirely (epoch-level
        printing from verbose is unaffected either way).
    """

    def __init__(
        self,
        continuous_columns: list[str],
        categorical_columns: list[str],
        categorical_cardinalities: dict[str, int],
        d_token: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        max_epochs: int = 50,
        patience: int = 5,
        validation_fraction: float = 0.15,
        random_state: int = 42,
        device: str | None = None,
        verbose: bool = True,
        num_threads: int = 1,
        progress_every_n_batches: int = 200,
    ) -> None:

        self.continuous_columns = continuous_columns
        self.categorical_columns = categorical_columns
        self.categorical_cardinalities = categorical_cardinalities

        self.d_token = d_token
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose
        self.num_threads = num_threads
        self.progress_every_n_batches = progress_every_n_batches

        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model_: TabularTransformer | None = None
        self.classes_: np.ndarray | None = None
        self._cont_mean: np.ndarray | None = None
        self._cont_std: np.ndarray | None = None

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _to_tensors(
        self, X: pd.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.continuous_columns:
            x_cont = X[self.continuous_columns].to_numpy(dtype=np.float32)
            x_cont = (x_cont - self._cont_mean) / self._cont_std
        else:
            x_cont = np.zeros((len(X), 0), dtype=np.float32)

        if self.categorical_columns:
            x_cat = X[self.categorical_columns].to_numpy(dtype=np.int64).copy()
        else:
            x_cat = np.zeros((len(X), 0), dtype=np.int64)

        return (
            torch.from_numpy(x_cont).to(self.device),
            torch.from_numpy(x_cat).to(self.device),
        )

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Any) -> "TabularTransformerClassifier":

        # See num_threads' docstring in __init__ for why this is set
        # here rather than once via an environment variable - torch's
        # thread pool is process-global, and Ray reuses worker
        # processes across tasks, so this has to be reasserted on
        # every call rather than trusted to have been set correctly
        # by whatever ran on this process before.
        torch.set_num_threads(self.num_threads)

        torch.manual_seed(self.random_state)

        y = np.asarray(y)
        self.classes_ = np.sort(np.unique(y))
        class_to_index = {c: i for i, c in enumerate(self.classes_)}
        y_index = np.array([class_to_index[label] for label in y])

        # Internal train/validation carve-out for early stopping
        # only - see class docstring for why this is a plain random
        # split rather than match-aware.
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        indices = rng.permutation(n)
        n_val = max(1, int(n * self.validation_fraction)) if n >= 10 else 0

        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_fit = X.iloc[train_idx].reset_index(drop=True)
        y_fit = y_index[train_idx]
        X_val = X.iloc[val_idx].reset_index(drop=True) if n_val else None
        y_val = y_index[val_idx] if n_val else None

        if self.continuous_columns:
            cont_values = X_fit[self.continuous_columns].to_numpy(dtype=np.float32)
            self._cont_mean = cont_values.mean(axis=0)
            self._cont_std = cont_values.std(axis=0)
            self._cont_std[self._cont_std < 1e-6] = 1.0
        else:
            self._cont_mean = np.zeros(0, dtype=np.float32)
            self._cont_std = np.ones(0, dtype=np.float32)

        cardinalities = [
            self.categorical_cardinalities[col] for col in self.categorical_columns
        ]

        self.model_ = TabularTransformer(
            n_continuous=len(self.continuous_columns),
            categorical_cardinalities=cardinalities,
            n_classes=len(self.classes_),
            d_token=self.d_token,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss()

        x_cont_fit, x_cat_fit = self._to_tensors_raw(X_fit)
        y_fit_t = torch.from_numpy(y_fit).long().to(self.device)

        if X_val is not None:
            x_cont_val, x_cat_val = self._to_tensors_raw(X_val)
            y_val_t = torch.from_numpy(y_val).long().to(self.device)

        n_fit = len(X_fit)
        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        # "gpu"/"cpu" rather than the raw torch.device string (which
        # already appears below) - short and grep-able, so runs from
        # both lanes of num_gpu_combos routing (see model_tuning.py)
        # are easy to tell apart in an interleaved, multi-task Ray log
        # without cross-referencing pids against the dashboard.
        device_tag = "gpu" if self.device.type == "cuda" else "cpu"

        if self.verbose:
            print(
                f"    [{device_tag}] fit starting: {n_fit} training rows, "
                f"{len(X_val) if X_val is not None else 0} internal validation rows, "
                f"{math.ceil(n_fit / self.batch_size)} batches/epoch, "
                f"device={self.device}"
            )

        generator = torch.Generator().manual_seed(self.random_state)

        fit_start = time.time()

        for epoch in range(self.max_epochs):

            epoch_start = time.time()

            self.model_.train()
            permutation = torch.randperm(n_fit, generator=generator)

            train_loss_sum = 0.0
            train_n = 0
            n_batches = math.ceil(n_fit / self.batch_size)

            for batch_i, start in enumerate(range(0, n_fit, self.batch_size), start=1):

                batch_idx = permutation[start:start + self.batch_size]

                optimizer.zero_grad()
                logits = self.model_(x_cont_fit[batch_idx], x_cat_fit[batch_idx])
                loss = loss_fn(logits, y_fit_t[batch_idx])
                loss.backward()
                optimizer.step()

                batch_n = len(batch_idx)
                train_loss_sum += loss.item() * batch_n
                train_n += batch_n

                if (
                    self.verbose
                    and self.progress_every_n_batches > 0
                    and batch_i % self.progress_every_n_batches == 0
                ):
                    print(
                        f"      [{device_tag}] epoch {epoch + 1} batch {batch_i}/{n_batches} "
                        f"running_train_loss={train_loss_sum / train_n:.4f} "
                        f"({time.time() - epoch_start:.1f}s into this epoch)"
                    )

            train_loss = train_loss_sum / train_n

            if X_val is None:
                if self.verbose:
                    print(
                        f"    [{device_tag}] epoch {epoch + 1}/{self.max_epochs} "
                        f"train_loss={train_loss:.4f} "
                        f"({time.time() - epoch_start:.1f}s)"
                    )
                continue

            self.model_.eval()

            # Same mini-batching as the training loop above, and for
            # the same reason it's needed there: passing the WHOLE
            # slice through self.model_() in one call makes standard
            # multi-head attention allocate an intermediate tensor
            # sized (batch x n_heads x seq_len x seq_len) - fine at
            # the batch sizes this was tested at during development,
            # catastrophic at real scale. Confirmed directly against
            # a real crash: an internal validation slice of ~48,000
            # rows (15% of a real ~322,000-row training split) tried
            # to allocate ~39.6GB in one shot for exactly this reason
            # (48344 x 8 heads x 160 seq_len^2 x 4 bytes lines up with
            # the reported allocation almost exactly) - this isn't a
            # hypothetical, it's what actually happened the first
            # time this ran against real data instead of a small
            # local sample. Batching by self.batch_size, same as
            # training, keeps peak memory bounded regardless of how
            # large X_val is.
            val_loss_sum = 0.0
            val_n = 0

            with torch.no_grad():

                for start in range(0, len(X_val), self.batch_size):

                    end = start + self.batch_size

                    batch_logits = self.model_(
                        x_cont_val[start:end], x_cat_val[start:end]
                    )
                    batch_loss = loss_fn(batch_logits, y_val_t[start:end])

                    # loss_fn defaults to mean reduction WITHIN a
                    # batch - weighting by each batch's actual size
                    # before averaging keeps this mathematically
                    # identical to computing the loss over the whole
                    # validation slice at once (which is what this
                    # replaces), not an approximation of it.
                    batch_n = end - start
                    if start + batch_n > len(X_val):
                        batch_n = len(X_val) - start

                    val_loss_sum += batch_loss.item() * batch_n
                    val_n += batch_n

            val_loss = val_loss_sum / val_n
            is_best = val_loss < best_val_loss - 1e-4

            if is_best:
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().clone() for k, v in self.model_.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.verbose:
                print(
                    f"    [{device_tag}] epoch {epoch + 1}/{self.max_epochs} "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"({time.time() - epoch_start:.1f}s)"
                    f"{'  * new best' if is_best else f'  ({epochs_without_improvement}/{self.patience} no improvement)'}"
                )

            if epochs_without_improvement >= self.patience:
                if self.verbose:
                    print(
                        f"    [{device_tag}] early stopping at epoch {epoch + 1} "
                        f"(best val_loss={best_val_loss:.4f}, "
                        f"{time.time() - fit_start:.1f}s total)"
                    )
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        if self.verbose:
            print(f"    [{device_tag}] fit finished in {time.time() - fit_start:.1f}s")

        return self

    def _to_tensors_raw(
        self, X: pd.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same as _to_tensors but assumes _cont_mean/_cont_std already set."""
        return self._to_tensors(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:

        if self.model_ is None:
            raise RuntimeError("Call fit() before predict_proba().")

        torch.set_num_threads(self.num_threads)

        self.model_.eval()
        x_cont, x_cat = self._to_tensors(X)

        # Same mini-batching as fit()'s validation loss, and the same
        # reason: one unbatched forward call over all of X makes
        # self-attention allocate a (batch x n_heads x seq_len^2)
        # tensor that scales with however many rows X has. This
        # matters MORE here than in fit() - predict_proba is called
        # on the sweep's full per-repeat test split (tens of
        # thousands of rows), on evaluate_held_out_test's held-out
        # set, and repeatedly inside permutation_importance, all
        # substantially larger than the 15% internal slice that
        # crashed fit() in the first place. Batching keeps peak
        # memory bounded regardless of how large X is; the
        # concatenated result is identical to the unbatched output,
        # just computed in bounded-size pieces.
        probs_batches = []

        with torch.no_grad():

            for start in range(0, len(X), self.batch_size):

                end = start + self.batch_size

                batch_logits = self.model_(x_cont[start:end], x_cat[start:end])
                batch_probs = torch.softmax(batch_logits, dim=1)
                probs_batches.append(batch_probs)

        probs = torch.cat(probs_batches, dim=0)

        return probs.cpu().numpy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:

        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


# ==============================================================
# Factory: wires a TabularTransformerClassifier builder into
# model_tuning.py's MODEL_BUILDERS calling convention
# ==============================================================


def make_transformer_builder(
    X: pd.DataFrame,
    categorical_columns: list[str] | None = None,
):
    """
    Returns a builder function with the signature
    model_tuning.py's MODEL_BUILDERS expects: (params, seed) -> model.

    This exists because a neural net's architecture (embedding
    table sizes, input width) has to be fixed before fit() is ever
    called, unlike the sklearn models which infer everything from
    whatever data they're given at fit time. Column cardinalities
    are computed once here from the FULL dataset (X, before any
    train/test splitting) and captured in a closure - not
    recomputed per-split - because a category code's embedding
    table needs to be sized for every code that appears ANYWHERE
    in the data, not just whichever split a given repeat happens
    to draw.

    categorical_columns defaults to feature_engineering.py's
    CATEGORICAL_COLUMNS list (imported lazily to avoid a hard
    module-level dependency) intersected with whatever's actually
    present in X, which is exactly right when X came from
    feature_engineering.py's encoding="ordinal" output (i.e.
    random_forest_features.csv / xgboost_features.csv /
    transformer_features.csv - all three share that encoding, see
    feature_engineering.py's docstring). Pass it explicitly if X
    came from somewhere else.

    Usage
    -----
        from tabular_transformer import make_transformer_builder
        from model_tuning import MODEL_BUILDERS, run_full_tuning

        MODEL_BUILDERS["transformer"] = make_transformer_builder(X)
        run_full_tuning("transformer", X, y, output_dir=..., groups=groups)
    """

    if categorical_columns is None:

        try:
            from feature_engineering import CATEGORICAL_COLUMNS
            categorical_columns = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
        except ImportError:
            raise ValueError(
                "categorical_columns must be given explicitly when "
                "feature_engineering.py isn't importable."
            )

    continuous_columns = [c for c in X.columns if c not in categorical_columns]

    categorical_cardinalities = {
        col: int(X[col].max()) + 1 for col in categorical_columns
    }

    def build(params: dict[str, Any], seed: int) -> TabularTransformerClassifier:

        return TabularTransformerClassifier(
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
            categorical_cardinalities=categorical_cardinalities,
            random_state=seed,
            **params,
        )

    return build


# Widened for cluster-scale compute, but still far smaller than the
# sklearn/XGBoost grids - each combination here is a real
# multi-epoch training run, not an instant sklearn fit, so combo
# count matters much more. n_layers and n_heads are now swept too
# (were fixed constants before) since real compute makes an actual
# architecture-depth comparison feasible rather than just assumed.
#
# d_token values were chosen to all be evenly divisible by every
# n_heads value (32/64/128 by 4/8) - nn.TransformerEncoderLayer
# requires d_model % nhead == 0, so an uneven combination would
# crash mid-sweep on whichever combo hit it first. If you add
# values to either list, keep that divisibility true for all pairs
# or filter invalid combinations before passing the grid in.
#
# 3 x 3 x 2 x 3 x 3 = 162 combinations was the estimate before any
# real timing existed. Measured against this project's actual
# ~322,000-row training split: ~790-800s/epoch even using every
# core of an 8-core machine (see num_threads' docstring above for
# why that "every core" part is itself part of the problem, not a
# mitigation). At up to max_epochs epochs per combination, even
# after tightening max_epochs/patience (see __init__), 162
# combinations is not a one-day sweep on a 12-CPU, one-modest-GPU
# cluster - it's several days at best, closer to a week in the
# worst case where combinations don't stop early.
#
# Reduced here to 3 x 2 x 2 x 1 x 3 = 36 by cutting the two
# dimensions least likely to be worth their cost at this scale, not
# by shrinking every dimension evenly:
#   - dropout fixed to 0.1 (was swept 0.1/0.2/0.3): a secondary
#     regularization knob, not a capacity question RQ2/RQ3 care
#     about the way d_token/n_layers/learning_rate are. 0.1 is the
#     standard default for transformer-family models generally.
#   - n_layers narrowed to [1, 2] (was [1, 2, 3]): each added layer
#     is a full extra self-attention + feed-forward block, so this
#     is one of the most expensive dimensions per step, and a
#     ~160-token tabular input (not a long NLP sequence) is a weak
#     case for needing real depth in the first place.
# d_token, n_heads, and learning_rate - the three dimensions most
# directly tied to what RQ2/RQ3 actually want to compare - are
# untouched. Widen this back once a real pilot run on the cluster
# confirms actual per-combination time with the fixes above in
# place - this is a starting point informed by measurement, not a
# permanent ceiling on what's worth exploring.
TRANSFORMER_PARAM_GRID: dict[str, list[Any]] = {
    "d_token": [64, 128],
    "n_layers": [2],
    "n_heads": [8],
    "dropout": [0.1],
    "learning_rate": [3e-4, 1e-3, 3e-3],
}
