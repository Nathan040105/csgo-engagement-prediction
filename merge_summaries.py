"""
merge_summaries.py - combine the two model_comparison_summary.csv files
(LR/RF/XGBoost launch + transformer launch) into one.

Usage (PowerShell or bash, from any folder):
    python merge_summaries.py <summary_1.csv> <summary_2.csv> <output.csv>

Example:
    python merge_summaries.py a_summary.csv b_summary.csv model_comparison_summary.csv
"""

import sys

import pandas as pd

if len(sys.argv) != 4:
    sys.exit(__doc__)

first, second, output = sys.argv[1:]

combined = pd.concat(
    [pd.read_csv(first), pd.read_csv(second)], ignore_index=True
).sort_values("held_out_test_roc_auc", ascending=False)

combined.to_csv(output, index=False)

print(combined[["model", "validation_roc_auc", "held_out_test_roc_auc"]].to_string(index=False))
print(f"\nWrote {output}")
