"""
Stratified train/val/test split for the CSE-CIC-IDS2018 network-flow
dataset, built from data.download_network's already-capped canonical
dataset (see that module's docstring for why the cap exists and how it's
chosen).

Stratification key: (capture_day, label_raw) -- every specific attack
scenario (not just the binary benign/malicious split) gets proportional
representation in every split, the same discipline used for
split_memory.py's (family, label) bucketing.

Unlike split_memory.py, this is a row-level stratified split, not a
group-protected one: CIC-MalMem-2022 had a real "same underlying malware
sample, ~10 memory dumps" structure that required keeping a sample's dumps
together to avoid leakage. CSE-CIC-IDS2018's flow records have no
analogous identity -- and deliberately so: data/download_network.py drops
Flow ID/Src IP/Src Port/Dst IP specifically so a model can't learn "which
fixed testbed machine" instead of a traffic pattern, which also means there
is no column left to define a "same underlying entity, multiple records"
group from even if we wanted to. The leakage safeguard here is instead the
same exact-duplicate-feature-vector hash check used for Cortex-Memory
(`verify_zero_duplicate_feature_hashes` below): any two rows whose full
78-feature vector is byte-identical are unioned into one group before
splitting, so duplicates (whatever their origin) can't span splits.

Before splitting, every row belonging to an exact-duplicate-feature-vector
group that contains BOTH benign and malicious labels is removed entirely
(train, val, AND test) via exclude_ambiguous_groups() -- genuine label
ambiguity in CICFlowMeter's 78-feature representation (identical feature
vector, contradictory ground truth), not leakage and not something any
classifier on these features could resolve. Excluded rows are saved
separately for inspection, not silently discarded.

Usage:
    python -m scripts.split_network \\
        --in data/processed/network_dataset.parquet \\
        --train-out data/processed/network_train.parquet \\
        --val-out   data/processed/network_val.parquet \\
        --test-out  data/processed/network_test.parquet \\
        --excluded-out data/processed/network_excluded_ambiguous.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from data.download_network import FEATURE_COLUMNS

logger = logging.getLogger("cortex.scripts.split_network")

DEFAULT_SEED = 42


def _feature_hashes(df: pd.DataFrame) -> pd.Series:
    values = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in values]
    return pd.Series(hashes, index=df.index)


def report_duplicate_hashes(df: pd.DataFrame) -> None:
    """Informational pass, run BEFORE splitting: how much exact-duplicate
    structure does this dataset actually have? Not assumed absent just
    because network flow data hasn't been checked before -- Cortex-Memory's
    own leakage-check work found real duplicates nobody expected going in."""
    feature_hash = _feature_hashes(df)
    check = df.assign(_h=feature_hash)
    sizes = check.groupby("_h").size()
    dup_groups = sizes[sizes > 1]
    logger.info(
        "=== Exact-duplicate-feature-vector check (pre-split) ===\n"
        "  %d / %d rows are exact duplicates of another row (%d duplicate groups, "
        "sizes ranging %d-%d)",
        int(dup_groups.sum()) if len(dup_groups) else 0, len(df), len(dup_groups),
        int(dup_groups.min()) if len(dup_groups) else 0, int(dup_groups.max()) if len(dup_groups) else 0,
    )
    if len(dup_groups):
        dup_rows = check[check["_h"].isin(dup_groups.index)]
        label_purity = dup_rows.groupby("_h")["label"].nunique()
        n_mixed_label = int((label_purity > 1).sum())
        logger.info(
            "  %d of %d duplicate group(s) span BOTH benign and malicious rows "
            "(same exact feature vector, different ground truth) -- %s",
            n_mixed_label, len(dup_groups),
            "removed entirely below (exclude_ambiguous_groups), not just from train" if n_mixed_label
            else "none -- every duplicate group is 100% benign or 100% malicious",
        )


def exclude_ambiguous_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Removes every row belonging to an exact-duplicate-feature-vector
    group that contains BOTH benign and malicious labels -- i.e. the
    identical 78-feature vector was observed with contradictory ground
    truth elsewhere in the data. This is NOT the cross-split leakage
    problem `verify_zero_duplicate_feature_hashes` guards against (nothing
    here spans splits); it's a property of CICFlowMeter's 78-feature
    representation itself -- these specific feature values genuinely do
    not determine the label. No classifier operating on these features
    could resolve them: keeping them in train teaches contradictory
    signal for identical inputs, and keeping them in val/test penalizes
    any model -- including a hypothetically perfect one -- for an
    unwinnable case. Excluded from train, val, AND test alike, not just
    train. Returns (clean_df, excluded_df) -- excluded rows are saved by
    main() to network_excluded_ambiguous.parquet for inspection, not
    silently discarded."""
    feature_hash = _feature_hashes(df)
    check = df.assign(_h=feature_hash)
    label_purity = check.groupby("_h")["label"].nunique()
    ambiguous_hashes = set(label_purity[label_purity > 1].index)
    ambiguous_mask = check["_h"].isin(ambiguous_hashes)

    excluded = df[ambiguous_mask].copy()
    # reset_index: stratified_group_split() below builds a plain
    # np.empty(len(df)) array and writes into it using df.index values
    # directly, which is only correct if the index is a gap-free 0..n-1
    # range. Boolean-mask filtering (df[~ambiguous_mask]) preserves the
    # ORIGINAL index values with gaps where excluded rows were removed --
    # a row that used to be at position 1,984,782 keeps that label even
    # though only ~1.95M rows remain, so writing into a ~1.95M-sized array
    # at that position raises IndexError. This was never exposed before
    # this function started pre-filtering the input to stratified_group_split()
    # -- every prior caller (including split_memory.py's identical pattern)
    # passed in a freshly-read, already gap-free DataFrame.
    clean = df[~ambiguous_mask].reset_index(drop=True)
    logger.warning(
        "Excluding %d row(s) across %d exact-duplicate-feature-vector group(s) that span "
        "both benign and malicious labels -- genuine label ambiguity in the CICFlowMeter "
        "feature representation, not leakage. Removed from train, val, AND test.",
        len(excluded), len(ambiguous_hashes),
    )
    return clean, excluded


def verify_zero_ambiguous_groups(df: pd.DataFrame) -> None:
    """Hard invariant, checked AFTER exclude_ambiguous_groups: no
    exact-duplicate-feature-vector group may span both benign and
    malicious labels. Proves the exclusion actually worked instead of
    trusting it did -- the same discipline as verify_zero_duplicate_feature_hashes
    below. Raises if violated, not a soft warning."""
    check = df.assign(_h=_feature_hashes(df))
    label_purity = check.groupby("_h")["label"].nunique()
    n_mixed = int((label_purity > 1).sum())
    if n_mixed:
        raise AssertionError(f"{n_mixed} exact-duplicate-feature-vector group(s) still span both labels after exclusion.")
    logger.info("Verified: 0 duplicate-feature-vector groups span both benign and malicious labels.")


def stratified_group_split(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int) -> pd.DataFrame:
    """Measured runtime note: on the real 2,000,000-row canonical dataset
    (~1.6M resulting groups after duplicate-hash merging), this function's
    per-group Python loop took ~25 minutes -- fine for a one-off split, but
    worth knowing before assuming a quick rerun if the sample cap or seed
    changes. The cost is the per-group majority-vote loop below, which
    split_memory.py's identical pattern never made visible at ~32K groups.
    Not optimized here since correctness on the one real run that produced
    the checked-in split mattered more than speculative speed for a script
    that doesn't need to run often; if repeated reruns become routine, the
    fix is to fast-path the (typically >90%) singleton-hash groups with a
    vectorized groupby instead of visiting every group individually."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["_group_id"] = _feature_hashes(df)  # rows sharing a hash are grouped; unique rows are singleton groups

    groups_raw: dict[str, list[int]] = defaultdict(list)
    for pos, gid in zip(df.index, df["_group_id"]):
        groups_raw[gid].append(pos)
    groups = {gid: np.array(positions) for gid, positions in groups_raw.items()}

    # Bucket each group by (capture_day, label_raw, label) -- same rationale
    # as split_memory.py's (family, label): every specific attack scenario
    # gets proportional representation in every split. label is included in
    # the bucket key (not just verified after the fact) because, unlike
    # memory's sample-id groups, a hash-duplicate group here could in
    # principle span two different label_raw values under the same binary
    # label (e.g. two different DDoS variants producing an identical
    # feature vector) -- majority vote decides the bucket the same way
    # split_memory.py handles a family-spanning merged group.
    buckets: dict[tuple, list[np.ndarray]] = defaultdict(list)
    mixed_stratum_groups = 0
    for gid, positions in groups.items():
        days = df.loc[positions, "capture_day"]
        label_raws = df.loc[positions, "label_raw"]
        labels = df.loc[positions, "label"]
        if days.nunique() > 1 or label_raws.nunique() > 1:
            mixed_stratum_groups += 1
        day = days.value_counts().idxmax()
        label_raw = label_raws.value_counts().idxmax()
        buckets[(day, label_raw, int(labels.value_counts().idxmax()))].append(positions)

    if mixed_stratum_groups:
        logger.warning(
            "%d exact-duplicate-feature-vector group(s) span more than one (capture_day, "
            "label_raw) -- bucketed by majority-row stratum for proportional-representation "
            "purposes; the hard split-leakage guarantee (below) is unaffected.",
            mixed_stratum_groups,
        )

    split_col = np.empty(len(df), dtype=object)
    for (day, label_raw, label), group_list in sorted(buckets.items(), key=lambda kv: (kv[0][0], kv[0][1])):
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

    df["split"] = split_col
    return df.drop(columns=["_group_id"])


def verify_zero_duplicate_feature_hashes(df: pd.DataFrame) -> None:
    """Hard invariant: no exact-duplicate feature vector may appear in more
    than one split. Should hold by construction (duplicates were grouped
    before allocation above) -- this proves it, rather than trusting it.
    Raises if violated, not a soft warning."""
    check = df.assign(_h=_feature_hashes(df))
    violations = []
    for h, sub in check.groupby("_h"):
        splits = set(sub["split"])
        if len(splits) > 1:
            violations.append((h, sub.index.tolist(), splits))
    if violations:
        for h, idxs, splits in violations[:10]:
            logger.error("DUPLICATE FEATURE VECTOR: hash=%s rows=%s spans splits %s", h[:12], idxs[:5], sorted(splits))
        raise AssertionError(f"{len(violations)} duplicate feature-vector hash(es) span more than one split.")
    logger.info(
        "Verified: 0 duplicate feature-vector hashes span more than one split (%d unique vectors / %d rows).",
        check["_h"].nunique(), len(check),
    )


def report_split(df: pd.DataFrame, val_frac: float, test_frac: float) -> None:
    n = len(df)
    logger.info("=== Split report ===")
    logger.info("Total rows: %d", n)
    for split, target_frac in (("train", 1 - val_frac - test_frac), ("val", val_frac), ("test", test_frac)):
        sub = df[df["split"] == split]
        actual_frac = len(sub) / n
        n_benign = int((sub["label"] == 0).sum())
        n_malicious = int((sub["label"] == 1).sum())
        logger.info(
            "%s: n=%d (%.2f%% of total, target %.2f%%, delta %+.2f pp) benign=%d malicious=%d",
            split, len(sub), 100 * actual_frac, 100 * target_frac, 100 * (actual_frac - target_frac),
            n_benign, n_malicious,
        )
        for (day, label_raw), group in sub.groupby(["capture_day", "label_raw"], sort=True):
            logger.info("    day=%-22s label_raw=%-25s n=%d", day, label_raw, len(group))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--val-out", required=True)
    ap.add_argument("--test-out", required=True)
    ap.add_argument("--excluded-out", required=True,
                     help="Where to save rows excluded for benign/malicious label ambiguity.")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    report_duplicate_hashes(df)

    df, excluded = exclude_ambiguous_groups(df)
    verify_zero_ambiguous_groups(df)
    excluded.to_parquet(args.excluded_out, index=False)
    logger.info("Saved %d excluded ambiguous row(s) to %s", len(excluded), args.excluded_out)

    df = stratified_group_split(df, args.val_frac, args.test_frac, args.seed)
    report_split(df, args.val_frac, args.test_frac)
    verify_zero_duplicate_feature_hashes(df)

    for split, out_path in (("train", args.train_out), ("val", args.val_out), ("test", args.test_out)):
        sub = df[df["split"] == split].drop(columns=["split"]).reset_index(drop=True)
        sub.to_parquet(out_path, index=False)
        logger.info("Saved %s split (%d rows) to %s", split, len(sub), out_path)


if __name__ == "__main__":
    main()
