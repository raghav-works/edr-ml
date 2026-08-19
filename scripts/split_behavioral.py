"""
Stratified train/val/test split for the behavioral dataset, grouped by exact
api_calls sequence -- not a plain row-independent split.

Two reasons this matters:
  1. Stratify by (label, source) jointly, not just label. A label-only split
     could easily push most or all of Carpenter's 101 benign rows into one
     split by chance, since it's tiny relative to malbehavd-v1's ~1,285
     benign rows; this guarantees every split gets proportional
     representation from every source.
  2. Group by exact sequence, not just row. After id-based dedup (see
     data/download_behavioral.py), the dataset still has ~949 rows that are
     genuinely different real files (different sha256/id) whose captured
     API-call trace happens to be identical -- mostly malbehavd-v1's shorter
     sequences, where a generic ~10-call startup/cleanup pattern legitimately
     recurs across unrelated executables. A row-independent split let those
     coincidentally-identical sequences land in different splits (85 of them
     did, in the first attempt), which is real leakage: the model could see
     the exact same input in train and get evaluated on it in test. Forcing
     every row with an identical sequence into the same split closes this
     completely, at the cost of split proportions no longer being exactly
     80/10/10 (groups vary in size and can't be split).

Runs before any oversampling/class-weight computation, per the leakage
lesson from the research doc (Issue 5). This version has no synthetic
oversampling step, but the discipline still applies: nothing derived from
the full dataset (vocabulary, pos_weight) should be computed before this
split exists -- that happens in later stages, reading only the train split.

Usage:
    python -m scripts.split_behavioral \\
        --in data/processed/behavioral_dataset.parquet \\
        --train-out data/processed/behavioral_train.parquet \\
        --val-out   data/processed/behavioral_val.parquet \\
        --test-out  data/processed/behavioral_test.parquet
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger("cortex.scripts.split_behavioral")

DEFAULT_SEED = 42


def _build_groups(df: pd.DataFrame) -> dict[tuple, np.ndarray]:
    """Maps each distinct api_calls sequence to the array of row positions
    (df.index values) that share it."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for pos, seq in zip(df.index, df["api_calls"]):
        groups[tuple(seq)].append(pos)
    return {seq: np.array(positions) for seq, positions in groups.items()}


def stratified_group_split(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = _build_groups(df)

    # Bucket each group by (source, majority label) -- source is uniform
    # within a group (no cross-source duplicate sequences exist, verified
    # separately), but a handful of groups span both labels (genuinely
    # different real files, different ground truth, identical short trace);
    # those travel together under their majority label's bucket.
    buckets: dict[tuple, list[np.ndarray]] = defaultdict(list)
    mixed_label_groups = 0
    for seq, positions in groups.items():
        sources = df.loc[positions, "source"]
        labels = df.loc[positions, "label"]
        assert sources.nunique() == 1, f"group spans multiple sources: {sources.unique().tolist()}"
        source = sources.iloc[0]
        label_counts = Counter(labels)
        if len(label_counts) > 1:
            mixed_label_groups += 1
        label = label_counts.most_common(1)[0][0]
        buckets[(source, label)].append(positions)

    if mixed_label_groups:
        logger.warning(
            "%d group(s) contain rows with different ground-truth labels for an identical "
            "sequence (genuinely different real files); bucketed by majority label.",
            mixed_label_groups,
        )

    split_col = np.empty(len(df), dtype=object)
    for (source, label), group_list in sorted(buckets.items()):
        order = rng.permutation(len(group_list))
        shuffled = [group_list[i] for i in order]
        total_rows = sum(len(g) for g in shuffled)
        target_val = round(total_rows * val_frac)
        target_test = round(total_rows * test_frac)

        val_count = test_count = 0
        for positions in shuffled:
            if val_count < target_val:
                split_col[positions] = "val"
                val_count += len(positions)
            elif test_count < target_test:
                split_col[positions] = "test"
                test_count += len(positions)
            else:
                split_col[positions] = "train"

    df = df.copy()
    df["split"] = split_col
    return df


def verify_zero_leakage(df: pd.DataFrame) -> None:
    """Hard invariant check: every distinct api_calls sequence must map to
    exactly one split. Raises if violated -- this is not allowed to be a
    soft warning."""
    groups = _build_groups(df)
    violations = []
    for seq, positions in groups.items():
        splits = set(df.loc[positions, "split"])
        if len(splits) > 1:
            violations.append((seq, positions, splits))
    if violations:
        for seq, positions, splits in violations[:10]:
            logger.error("LEAKAGE: sequence (len=%d) spans splits %s", len(seq), sorted(splits))
        raise AssertionError(f"{len(violations)} sequence(s) still span more than one split -- grouping failed.")
    logger.info("Verified: 0 / %d distinct sequences span more than one split.", len(groups))


def report(df: pd.DataFrame, val_frac: float, test_frac: float) -> None:
    n = len(df)
    logger.info("Total rows: %d", n)
    for split, target_frac in (("train", 1 - val_frac - test_frac), ("val", val_frac), ("test", test_frac)):
        sub = df[df["split"] == split]
        actual_frac = len(sub) / n
        logger.info(
            "%s: n=%d (%.2f%% of total, target %.2f%%, delta %+.2f pp)",
            split, len(sub), 100 * actual_frac, 100 * target_frac, 100 * (actual_frac - target_frac),
        )
        for (source, label), group in sub.groupby(["source", "label"], sort=True):
            logger.info("  source=%-13s label=%d  n=%d", source, label, len(group))

    verify_zero_leakage(df)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--val-out", required=True)
    ap.add_argument("--test-out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    df = stratified_group_split(df, args.val_frac, args.test_frac, args.seed)
    report(df, args.val_frac, args.test_frac)

    for split, out_path in (("train", args.train_out), ("val", args.val_out), ("test", args.test_out)):
        sub = df[df["split"] == split].drop(columns=["split"]).reset_index(drop=True)
        sub.to_parquet(out_path, index=False)
        logger.info("Saved %s split (%d rows) to %s", split, len(sub), out_path)


if __name__ == "__main__":
    main()
