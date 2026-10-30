"""
clean_checkpoint.py

One-time cleanup: the CUDA-OOM cascade wrote several combinations to
transformer_sweep_checkpoint.jsonl with roc_auc_mean=NaN (a task that
raised an exception still gets checkpointed as "combination complete"
- see _run_distributed_sweep's remaining_by_combo logic in
model_tuning.py - so a failed run and a successful one look identical
to resume=True unless something distinguishes them). Left as-is, the
next run's resume=True will treat every one of those as done and
permanently skip re-attempting them, baking in garbage results.

Run this ONCE, after applying the device-injection fix and BEFORE
rerunning tuning_test.py, from wherever transformer_sweep_checkpoint.jsonl
actually lives (adjust CHECKPOINT_PATH below if your output_dir differs
from "tuning_results").

What it does: keeps the header line (__fingerprint__) and any row
whose roc_auc_mean is a real number, drops any row whose roc_auc_mean
is NaN, and writes the result back - after saving the original
untouched as a .bak file first, so nothing is destroyed if this
script itself has a bug.
"""

import json
import math
import shutil
from pathlib import Path

CHECKPOINT_PATH = Path("tuning_results/transformer/transformer_sweep_checkpoint.jsonl")


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        print(f"No checkpoint found at {CHECKPOINT_PATH} - nothing to clean.")
        return

    backup_path = CHECKPOINT_PATH.with_suffix(".jsonl.bak")
    shutil.copy2(CHECKPOINT_PATH, backup_path)
    print(f"Backed up original checkpoint to {backup_path}")

    kept_lines: list[str] = []
    n_kept = 0
    n_dropped = 0

    with open(CHECKPOINT_PATH) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)

            if "__fingerprint__" in entry:
                kept_lines.append(line)
                continue

            roc_auc_mean = entry.get("row", {}).get("roc_auc_mean")
            is_nan = roc_auc_mean is None or (
                isinstance(roc_auc_mean, float) and math.isnan(roc_auc_mean)
            )

            if is_nan:
                n_dropped += 1
                print(f"  dropping (roc_auc_mean=NaN): {entry.get('params')}")
            else:
                kept_lines.append(line)
                n_kept += 1

    with open(CHECKPOINT_PATH, "w") as file:
        for line in kept_lines:
            file.write(line + "\n")

    print(
        f"\nDone: kept {n_kept} valid combination(s), dropped {n_dropped} "
        f"NaN combination(s). They'll be re-attempted on the next run."
    )


if __name__ == "__main__":
    main()
