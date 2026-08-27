"""
Val carve-out for the Quo Vadis Speakeasy-emulation dataset, built from
data.download_emulation's canonical dataset.

Unlike split_memory.py/split_network.py, this does NOT re-split train/test:
the dataset's authors already partitioned it by collection date (train =
Jan 2022, test = Apr 2022) specifically to capture concept drift -- a
deliberately more informative split than an arbitrary random one, and
preserved here rather than discarded. Test is never touched by this
script. Val is carved out of train only, the same way scripts/train_static.py
carves val out of its own train split (not a separate authored partition).

Duplicate detection: the dataset provides a per-entry `apihash` field.
Per instructions, this is validated empirically against a hand-computed
hash of the same entry's `api_names` sequence -- checked, not assumed --
before deciding whether to use it directly as the grouping key or fall
back to the hand-computed one. See validate_apihash()'s docstring for
what "validated" means here.

The full canonical dataset (all ep_types pooled) turned out to be 96.8%
exact-duplicate rows -- three orders of magnitude higher than
Cortex-Memory's 0.9% or Cortex-Network's 25%. Investigated, not accepted at
face value: two distinct causes. (1) 97.8% of non-module_entry rows
(thread/tls_callback_*/etc) have <=1 API call -- near-empty structural
noise, not behavioral signal. (2) Even module_entry alone (the real
signal) is 94.3% duplicated -- a genuine property of this dataset, likely
reflecting malware-builder-kit-generated variants that are byte-different
but API-name-identical. A naive group-and-protect approach (Memory/
Network's pattern) would let a single 4,312-row duplicate group dominate
whichever split it landed in -- confirmed as the actual cause of a first
attempt's val subset coming out 90% malicious. This script instead:

1. Restricts the modeling population to `ep_type == "module_entry"` rows
   only (see restrict_to_module_entry()) -- non-module_entry rows stay in
   the canonical parquet (data.download_emulation's output) but are
   excluded from train/val/test here, not silently discarded from the
   dataset entirely.
2. Collapses exact-duplicate api_names sequences to ONE representative row
   per unique sequence (see collapse_duplicates()) -- not grouped-and-kept-
   together like Memory/Network, since here a single dominant group is
   large enough to single-handedly skew a split's composition. A
   `duplicate_count` column records how many original rows shared each
   surviving sequence, so the collapsed volume isn't silently lost -- just
   deferred for possible frequency-weighted use later.
3. Collapses train and test SEPARATELY, never across the boundary: a
   sequence appearing in both partitions keeps one representative in each,
   preserving both "test is never touched" and the concept-drift comparison
   the authors' split exists for. A cross-partition duplicate is still
   reported (see report_duplicate_groups()), not hidden by the collapse.

Because collapsing already guarantees every row in a given partition has a
unique sequence, the val carve-out below is a plain stratified split by
(family, label) -- no hash-based group protection is needed (there are no
groups left to protect).

Usage:
    python -m scripts.split_emulation \\
        --in data/processed/emulation_dataset.parquet \\
        --train-out data/processed/emulation_train.parquet \\
        --val-out   data/processed/emulation_val.parquet \\
        --test-out  data/processed/emulation_test.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger("cortex.scripts.split_emulation")

DEFAULT_SEED = 42
_SEQ_SEP = "\x1f"  # unit separator -- won't appear inside an API name


def _sequence_hash(api_names) -> str:
    return hashlib.sha256(_SEQ_SEP.join(api_names).encode("utf-8")).hexdigest()


def validate_apihash(df: pd.DataFrame) -> str:
    """Checks, rather than assumes, whether the dataset-provided `apihash`
    field is a valid proxy for "identical api_names sequence" -- i.e.
    whether grouping by apihash gives the exact same groups as grouping by
    a hand-computed hash of the sequence itself. Two failure modes are
    checked independently:
      1. False merge: does one apihash value ever cover rows with
         genuinely DIFFERENT api_names sequences? (would make apihash
         unsafe to use alone -- it could hide a real difference)
      2. False split: does one true sequence ever get DIFFERENT apihash
         values? (would make apihash miss real duplicates)
    Returns "apihash" if both checks are clean (safe to use directly, and
    cheaper than rehashing every sequence downstream) or "sequence_hash"
    if not (falls back to the hand-computed column, logging exactly what
    was found). Real result on the full dataset: 686/6,699 apihash groups
    (10.2%) are false merges -- NOT safe to use alone."""
    my_hash = df["api_names"].map(_sequence_hash)
    check = df.assign(_my_hash=my_hash)

    apihash_to_myhash = check.groupby("apihash")["_my_hash"].nunique()
    false_merges = int((apihash_to_myhash > 1).sum())

    myhash_to_apihash = check.groupby("_my_hash")["apihash"].nunique()
    false_splits = int((myhash_to_apihash > 1).sum())

    logger.info(
        "=== apihash validation (against hand-computed sequence hash) ===\n"
        "  false merges (1 apihash -> >1 distinct sequences): %d / %d apihash groups\n"
        "  false splits (1 sequence -> >1 distinct apihash):  %d / %d sequence groups",
        false_merges, len(apihash_to_myhash), false_splits, len(myhash_to_apihash),
    )
    if false_merges == 0 and false_splits == 0:
        logger.info("apihash validated as an exact proxy for sequence identity -- using it directly.")
        return "apihash"
    logger.warning(
        "apihash did NOT validate as an exact proxy for sequence identity -- "
        "falling back to the hand-computed sequence hash for grouping."
    )
    return "sequence_hash"


def report_duplicate_groups(df: pd.DataFrame, hash_col: pd.Series) -> None:
    """Informational, run across the WHOLE incoming dataset (whatever rows
    are passed in -- all ep_types pooled, before module_entry restriction):
    how much exact-duplicate api_names structure exists, and does any of
    it span the dataset's own train/test partition? Reported, not fixed --
    see the module docstring for why a cross-split duplicate here isn't
    something this script alters."""
    check = df.assign(_h=hash_col)
    sizes = check.groupby("_h").size()
    dup_groups = sizes[sizes > 1]
    logger.info(
        "=== Exact-duplicate api_names-sequence check (whole dataset) ===\n"
        "  %d / %d rows are exact duplicates of another row (%d duplicate groups, sizes %d-%d)",
        int(dup_groups.sum()) if len(dup_groups) else 0, len(df), len(dup_groups),
        int(dup_groups.min()) if len(dup_groups) else 0, int(dup_groups.max()) if len(dup_groups) else 0,
    )
    if len(dup_groups) == 0:
        return
    dup_rows = check[check["_h"].isin(dup_groups.index)]
    cross_split = dup_rows.groupby("_h")["split"].nunique()
    n_cross = int((cross_split > 1).sum())
    logger.info(
        "  %d of %d duplicate group(s) span the dataset's OWN train/test partition "
        "(Jan 2022 vs Apr 2022 collection) -- reported as a property of the authors' "
        "split design, not altered here (test is never touched).",
        n_cross, len(dup_groups),
    )
    label_purity = dup_rows.groupby("_h")["label"].nunique()
    n_mixed = int((label_purity > 1).sum())
    logger.info(
        "  %d of %d duplicate group(s) span both benign and malicious labels "
        "(same exact api_names sequence, different ground truth)",
        n_mixed, len(dup_groups),
    )


def restrict_to_module_entry(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restricts the modeling population to ep_type == "module_entry" rows
    -- thread/tls_callback_*/etc are near-empty structural noise (97.8% of
    them have <=1 API call; the single largest duplicate group in the full
    dataset, 64,152 rows, was entirely 1-API-call thread/tls_callback
    entries). They are NOT dropped from the canonical parquet -- only
    excluded from train/val/test here.

    Returns (module_entry_df, dropped_files_df) -- dropped_files_df lists
    every sample_id that has ZERO module_entry rows at all (so restricting
    to module_entry would otherwise silently remove it from the modeling
    dataset entirely), one row per such file, for the caller to report."""
    is_me = df["ep_type"] == "module_entry"
    me_df = df[is_me].copy()

    ids_with_me = set(me_df["sample_id"])
    all_ids = set(df["sample_id"])
    dropped_ids = all_ids - ids_with_me
    dropped_df = df[df["sample_id"].isin(dropped_ids)].drop_duplicates("sample_id").copy()

    logger.info(
        "=== Restricting to module_entry rows ===\n"
        "  %d / %d rows kept (module_entry); %d / %d files have ZERO module_entry rows "
        "and are excluded from train/val/test entirely (still present in the canonical parquet)",
        len(me_df), len(df), len(dropped_df), df["sample_id"].nunique(),
    )
    if len(dropped_df):
        logger.info("  Label/family breakdown of the %d dropped file(s):", len(dropped_df))
        for (split, family, label), count in dropped_df.groupby(["split", "family", "label"]).size().items():
            logger.info("    split=%-6s family=%-16s label=%d n=%d", split, family, label, count)
    return me_df, dropped_df


def collapse_duplicates(df: pd.DataFrame, hash_col: pd.Series) -> pd.DataFrame:
    """Collapses exact-duplicate api_names sequences to ONE representative
    row per unique sequence, adding a `duplicate_count` column (how many
    original rows shared that sequence) so the collapsed volume is
    recorded, not silently lost. MUST be called separately per authored
    train/test partition (never on the two combined) -- collapsing across
    the boundary would let a sequence that happens to also appear in the
    other partition determine which partition "keeps" a representative,
    which would silently shrink test based on what's in train (or vice
    versa). The representative is chosen by lowest sample_id, purely for
    determinism -- which specific original row survives doesn't matter,
    only that exactly one does."""
    work = df.assign(_h=hash_col.loc[df.index])
    dup_counts = work.groupby("_h").size().rename("duplicate_count")
    representatives = work.sort_values("sample_id").groupby("_h", as_index=False).first()
    collapsed = representatives.merge(dup_counts, left_on="_h", right_index=True).drop(columns=["_h"])
    logger.info(
        "  collapsed %d rows -> %d unique sequences (dropped %d redundant copies, "
        "tracked in duplicate_count)",
        len(df), len(collapsed), len(df) - len(collapsed),
    )
    return collapsed


def separability_check(df: pd.DataFrame) -> None:
    """The same kind of check that caught Cortex-Memory's single-VM
    shortcut, adapted for sequence data: does a narrow set of benign
    sequences dominate the benign population the way Memory's benign
    samples traced back to one VM, and does a trivial property (raw
    sequence length) separate the classes on its own? Run on the
    module_entry population BEFORE collapsing (duplicate_count itself is
    the diversity signal -- collapsing first would hide exactly the
    concentration this check looks for)."""
    logger.info("=== Separability check (module_entry, pre-collapse) ===")
    n_unique = df["api_names"].map(_sequence_hash).nunique()

    benign = df[df["label"] == 0]
    malicious = df[df["label"] == 1]
    benign_hash = benign["api_names"].map(_sequence_hash)
    malicious_hash = malicious["api_names"].map(_sequence_hash)
    logger.info(
        "  overall: %d rows, %d unique sequences\n"
        "  benign: %d rows, %d unique sequences (%.1f%% unique)\n"
        "  malicious: %d rows, %d unique sequences (%.1f%% unique)",
        len(df), n_unique,
        len(benign), benign_hash.nunique(), 100 * benign_hash.nunique() / max(len(benign), 1),
        len(malicious), malicious_hash.nunique(), 100 * malicious_hash.nunique() / max(len(malicious), 1),
    )

    top5_frac = benign_hash.value_counts().head(5).sum() / max(len(benign), 1)
    top1_frac = benign_hash.value_counts().head(1).sum() / max(len(benign), 1)
    logger.info(
        "  top 5 benign unique-sequence groups cover %.1f%% of all benign rows "
        "(top 1 alone: %.1f%%) -- for comparison, Cortex-Memory's single-VM benign "
        "collection meant a handful of raw feature values covered nearly ALL benign "
        "rows; this is meaningfully more diverse, not the same failure mode",
        100 * top5_frac, 100 * top1_frac,
    )

    y = df["label"].to_numpy()
    x = df["n_apis"].to_numpy()
    auc = roc_auc_score(y, x)
    auc = max(auc, 1 - auc)
    logger.info(
        "  single-feature AUC of raw sequence length (n_apis) alone: %.4f -- %s",
        auc, "barely above chance, NOT a usable shortcut" if auc < 0.65 else "worth investigating further",
    )


def carve_val_from_train(train_df: pd.DataFrame, val_frac: float, seed: int) -> pd.DataFrame:
    """Plain stratified split by (family, label) -- no hash-based group
    protection needed, since collapse_duplicates() already guarantees
    every row in train_df has a unique api_names sequence within this
    partition."""
    rng = np.random.default_rng(seed)
    df = train_df.copy()

    buckets: dict[tuple, list[int]] = defaultdict(list)
    for idx, family, label in zip(df.index, df["family"], df["label"]):
        buckets[(family, int(label))].append(idx)

    subset_col = pd.Series(index=df.index, dtype=object)
    for (family, label), idxs in sorted(buckets.items()):
        idxs = np.array(idxs)
        order = rng.permutation(len(idxs))
        shuffled = idxs[order]
        target_val = round(len(shuffled) * val_frac)
        subset_col.loc[shuffled[:target_val]] = "val"
        subset_col.loc[shuffled[target_val:]] = "train"

    df["subset"] = subset_col
    return df


def report_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    for name, sub in (("train", train_df), ("val", val_df), ("test", test_df)):
        n = len(sub)
        n_benign = int((sub["label"] == 0).sum())
        n_malicious = int((sub["label"] == 1).sum())
        logger.info("%s: n=%d benign=%d malicious=%d", name, n, n_benign, n_malicious)
        for family, group in sub.groupby("family", sort=True):
            logger.info("    family=%-18s n=%d", family, len(group))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--val-out", required=True)
    ap.add_argument("--test-out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    if set(df["split"].unique()) != {"train", "test"}:
        raise ValueError(f"Expected split column values {{'train','test'}}, got {set(df['split'].unique())}")

    which_hash = validate_apihash(df)
    hash_col = df["apihash"] if which_hash == "apihash" else df["api_names"].map(_sequence_hash)
    report_duplicate_groups(df, hash_col)

    me_df, dropped_df = restrict_to_module_entry(df)
    separability_check(me_df)

    me_hash = me_df["apihash"] if which_hash == "apihash" else me_df["api_names"].map(_sequence_hash)
    train_me = me_df[me_df["split"] == "train"]
    test_me = me_df[me_df["split"] == "test"]

    logger.info("=== Collapsing exact duplicates within train (Jan 2022) ===")
    train_collapsed = collapse_duplicates(train_me, me_hash)
    logger.info("=== Collapsing exact duplicates within test (Apr 2022) ===")
    test_collapsed = collapse_duplicates(test_me, me_hash)

    carved = carve_val_from_train(train_collapsed, args.val_frac, args.seed)
    final_train = carved[carved["subset"] == "train"].drop(columns=["subset", "split"]).reset_index(drop=True)
    final_val = carved[carved["subset"] == "val"].drop(columns=["subset", "split"]).reset_index(drop=True)
    final_test = test_collapsed.drop(columns=["split"]).reset_index(drop=True)

    report_split(final_train, final_val, final_test)

    for name, sub, out_path in (
        ("train", final_train, args.train_out), ("val", final_val, args.val_out), ("test", final_test, args.test_out),
    ):
        sub.to_parquet(out_path, index=False)
        logger.info("Saved %s split (%d rows) to %s", name, len(sub), out_path)


if __name__ == "__main__":
    main()
