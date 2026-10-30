"""
check_side_features.py - are attacker_side / victim_side redundant, and do they
carry any outcome signal of their own?

The transformer's single-feature permutation importance puts attacker_side and
victim_side at #1 and #2 (about a third of its total importance) while XGBoost,
random forest and logistic regression give them ~0. This script collects the
evidence for why, from one model CSV (ordinal encoding: CT = 0, T = 1).

Usage (from the folder that holds tuning_data/):
    python check_side_features.py tuning_data/xgboost_features.csv

Needs pandas, numpy, scikit-learn, xgboost. About two minutes on 2 cores.

Prints:
  1. how often the two sides are opposite (mirror images)
  2. the outcome mix by attacker side (is side informative on its own?)
  3. in what share of rows the side can be read off the alive counts
     (t_alive, ct_alive, attacker_alive_teammates, attacker_alive_enemies)
  4. how well the OTHER 157 features predict attacker_side (a held-out-match
     test) - if they predict it almost perfectly, the column is redundant
"""

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

CLASS_NAMES = ["attacker_won", "interrupted", "mutual_kill", "round_end", "timedout", "victim_won"]

if len(sys.argv) != 2:
    sys.exit(__doc__)

df = pd.read_csv(sys.argv[1])
side, vside = df["attacker_side"], df["victim_side"]
print(f"{len(df)} rows, {df['match_id'].nunique()} matches")

# 1 ---------------------------------------------------------------------------
print("\n1. attacker_side x victim_side (0 = CT, 1 = T)")
print(pd.crosstab(side, vside, margins=True))
print(f"opposite sides in {(side != vside).mean():.2%} of rows "
      f"({int((side == vside).sum())} rows have both players on the same side)")

# 2 ---------------------------------------------------------------------------
print("\n2. outcome mix by attacker side (row %)")
mix = pd.crosstab(side.map({0: "CT", 1: "T"}), df["label"].map(dict(enumerate(CLASS_NAMES))),
                  normalize="index") * 100
print(mix.round(1).to_string())

# 3 ---------------------------------------------------------------------------
print("\n3. can the side be read off the alive counts?")
own = df["attacker_alive_teammates"] + 1
enemy = df["attacker_alive_enemies"]
is_t = (own == df["t_alive"]) & (enemy == df["ct_alive"])
is_ct = (own == df["ct_alive"]) & (enemy == df["t_alive"])
decided = is_t ^ is_ct
correct = ((is_t & (side == 1)) | (is_ct & (side == 0)))[decided].mean()
print(f"side is determined by the counts in {decided.mean():.1%} of rows "
      f"(correct whenever it is determined: {correct:.1%}); the rest are rows where both "
      f"teams have the same number alive")

# 4 ---------------------------------------------------------------------------
print("\n4. predicting attacker_side from every OTHER feature (held-out matches)")
features = [c for c in df.columns if c not in ("label", "match_id", "attacker_side", "victim_side")]
train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
                   .split(df, side, groups=df["match_id"]))
model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=0)
model.fit(df.iloc[train][features], side.iloc[train])
proba = model.predict_proba(df.iloc[test][features])[:, 1]
accuracy = accuracy_score(side.iloc[test], proba > 0.5)
majority = max(side.iloc[test].mean(), 1 - side.iloc[test].mean())
print(f"accuracy {accuracy:.1%} (always guessing the majority side: {majority:.1%}); "
      f"ROC-AUC {roc_auc_score(side.iloc[test], proba):.3f}")
