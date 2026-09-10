"""
Stratified train/val/test split for the CIC-MalMem-2022 memory-forensics
dataset, group-aware where the data supports it.

Real `Category` format (verified against the actual CSV, not assumed):
"<Type>-<Family>-<sha256 hex>-<dump#>.raw" for malicious rows (e.g.
"Ransomware-Ako-00a14062c268...-1.raw"), and the literal constant string
"Benign" for every benign row. `_sample_id()` strips the "-<dump#>.raw"
suffix to recover the sha256-identified malware sample a dump came from;
`_family()` further strips the hash to get the family/type name.

The dataset's paper (Carrier et al., ICISSP 2022) describes capturing
roughly 10 memory dumps per malware sample during dynamic analysis, which
this data confirms: malicious rows sharing an identical sample id (same
"<Type>-<Family>-<hash>" prefix) are different dumps of the same running
malware instance. A naive row-independent split would let dumps from the
same instance land in different splits -- the model could see one dump of a
sample in train and get evaluated on a near-identical dump of the *same*
sample in test, which is leakage, not generalization. `_assign_groups`
below checks the sample-id grouping empirically against the real per-label
size distribution (also surfaced directly in `report_group_sizes`), rather
than assuming it, and only applies group-based splitting where the
distribution actually supports it. (An earlier version of this grouping
logic grouped on the raw `Category` string directly, which happened to work
against a synthetic smoke-test dataset shaped with one Category string per
sample -- but the real data bakes the dump number into `Category` itself,
making every row's raw Category value unique. Grouping on raw Category
against real data would have reported a median group size of 1 for the
malicious class -- indistinguishable from "no real grouping" -- and
silently skipped the exact leakage protection this script exists to
provide. `_sample_id()` strips the dump-number suffix first so the grouping
key is the sample identity, not the per-dump string.)

Benign rows' constant `Category` value means Category carries zero
information for distinguishing one benign capture from another -- there is
no sample identifier for benign rows anywhere in the raw schema. No
group-based leakage protection is therefore possible for the benign class
from this data alone; each benign row ends up as its own singleton group
(equivalent to a plain row-independent split for that class specifically).
This is detected and logged explicitly by `_assign_groups`, not silently
assumed. It is also a real, documented limitation, not a fully solved
problem: near-duplicate benign memory snapshots (e.g. repeated captures of
the same idle baseline system state) could in principle still land across
multiple splits, since nothing in this dataset's schema lets us tell two
independently-genuine benign captures apart from two captures of the exact
same underlying system state. The exact-duplicate feature-vector hash check
below catches only *exact* duplicates, not near-duplicates -- see the
README's Known Limitations section.

After splitting, every row's feature vector is hashed and checked for exact
duplicates spanning more than one split (`verify_zero_duplicate_feature_hashes`),
independent of the group-based check above -- it catches accidental exact
duplicate rows that sample-id-based grouping wouldn't (e.g. a
data-collection artifact producing an identical feature vector under two
different Category labels).

Four splits, not three: `cal` sits between `test` and `train`. The greedy
per-bucket allocator fills val, then test, then cal, then train from one
seeded permutation of that bucket's groups. Inserting the `cal` phase
consumes no additional RNG draws, so val and test receive exactly the
groups they did when this script produced three splits -- byte-identical --
and `cal` is a deterministic slice carved from what would otherwise have
been train. `cal` is the split the Platt calibrator is fit on and every
`memory.*` value in config/thresholds.yaml is derived from; `val` is then
used for LightGBM early stopping only and `test` is read exactly once, for
reported numbers, after everything is frozen. See OPEN_ITEMS.md's "retrain
cluster" section for the full rationale. `--cal-frac` is deliberately
required with no default: memory uses 0.2 (an enlarged calibration set,
sized to match the benign count the pre-split-discipline val+test threshold
derivation relied on), unlike network's 0.1 -- the script refuses to guess.

Usage:
    python -m scripts.split_memory \\
        --in data/processed/memory_dataset.parquet \\
        --train-out data/processed/memory_train.parquet \\
        --val-out   data/processed/memory_val.parquet \\
        --test-out  data/processed/memory_test.parquet \\
        --cal-out   data/processed/memory_cal.parquet \\
        --cal-frac  0.2
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from data.download_memory import CATEGORY_COL, CLASS_COL, feature_columns

logger = logging.getLogger("cortex.scripts.split_memory")

DEFAULT_SEED = 42
_DUMP_SUFFIX_RE = re.compile(r"^(.+)-(\d+)\.raw$")
_SAMPLE_HASH_RE = re.compile(r"^(.+)-([0-9a-fA-F]{6,})$")


def _sample_id(category: str) -> str:
    """Strips the trailing "-<dump#>.raw" suffix to recover the underlying
    malware sample identity a dump was captured from (e.g.
    "Ransomware-Ako-00a14062...-1.raw" -> "Ransomware-Ako-00a14062...").
    Rows sharing an identical sample id are different memory dumps of the
    same running sample. "Benign" (no such suffix) is returned unchanged --
    there is no sample identity to recover from it."""
    m = _DUMP_SUFFIX_RE.match(category)
    return m.group(1) if m else category


def _family(category: str) -> str:
    """Strips the dump-number/".raw" suffix and the trailing sample hash to
    get the malware family/type name for stratification (e.g.
    "Ransomware-Ako-00a14062...-1.raw" -> "Ransomware-Ako"). Falls back to
    the sample id (or raw Category, e.g. "Benign") unchanged if no hash
    suffix is present."""
    sample = _sample_id(category)
    m = _SAMPLE_HASH_RE.match(sample)
    return m.group(1) if m else sample


def _assign_groups(df: pd.DataFrame) -> pd.Series:
    """Per label, decides whether sample-id-based grouping is real (median
    group size well above 1) or not, and returns a per-row group id
    accordingly -- the sample id where grouping holds, a synthetic per-row
    id (no grouping) where it doesn't. Computed empirically per label
    rather than assumed, since benign and malware are expected to differ
    here (see module docstring)."""
    sample_id = df[CATEGORY_COL].map(_sample_id)
    group_id = pd.Series(index=df.index, dtype=object)
    for label in sorted(df["label"].unique()):
        idx = df.index[df["label"] == label]
        sizes = sample_id.loc[idx].value_counts()
        median_size = float(sizes.median())
        # Require BOTH median size > 1 AND more than one distinct sample id.
        # Median alone is not enough: a single constant value across an
        # entire class (e.g. every benign row literally labeled "Benign",
        # which has no dump-number suffix for _sample_id to strip)
        # produces exactly one "group" whose size equals the whole class's
        # row count -- an enormous median that trivially passes a
        # size-only check, but is the degenerate case, not real per-sample
        # grouping. Caught by testing this against a synthetic dataset
        # shaped like the real one before running it on real data: the
        # first version of this check (median-only) silently put all 1800
        # synthetic benign rows into a single split.
        if median_size > 1.0 and len(sizes) > 1:
            logger.info(
                "label=%d: sample-id grouping looks real (median group size "
                "%.1f over %d distinct sample(s)) -- grouping by sample id.",
                label, median_size, len(sizes),
            )
            group_id.loc[idx] = sample_id.loc[idx]
        else:
            logger.info(
                "label=%d: sample-id grouping does NOT look real (median "
                "group size %.1f over %d distinct value(s), e.g. a constant "
                "Category with no dump-number suffix) -- falling back to one "
                "group per row (no grouping protection available for this "
                "class from this data).",
                label, median_size, len(sizes),
            )
            group_id.loc[idx] = [f"label{label}-row-{i}" for i in idx]
    return group_id


class _UnionFind:
    """Minimal disjoint-set structure, used only to merge sample-id groups
    that turn out to share an exact-duplicate feature vector (see
    `_assign_merged_groups`)."""

    def __init__(self) -> None:
        self._parent: dict = {}

    def find(self, x):
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def _feature_hashes(df: pd.DataFrame) -> pd.Series:
    feat_cols = [c for c in feature_columns(df) if c != "split"]
    values = df[feat_cols].to_numpy(dtype=np.float64)
    hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in values]
    return pd.Series(hashes, index=df.index)


def _assign_merged_groups(df: pd.DataFrame) -> pd.Series:
    """Combines two independent "must stay in the same split" constraints
    into one final group id per row:

    1. Sample-id groups from `_assign_groups()` -- dumps of the same
       malware sample (or, where that grouping isn't real, one group per
       row; see `_assign_groups`' docstring).
    2. Exact-duplicate feature-vector groups -- rows whose full 55-feature
       vector is byte-identical, regardless of Category/sample id. This
       catches two things `_assign_groups` structurally cannot: (a)
       near/exact-duplicate benign snapshots (benign has no per-sample
       identifier at all to group on), and (b) different malware samples
       whose coarse VolMemLyzer summary stats happen to collide exactly --
       confirmed present in the real data (11 groups / 25 rows on the full
       58,596-row dataset, none of them crossing benign/malware, 19 of the
       full 517 exact-duplicate groups crossing malware *families*).

    Any two rows tied by either constraint end up in the same merged group
    via union-find (e.g. if row A and row B share a sample id, and row B
    and row C share a duplicate feature hash, A/B/C all land in one group).
    This makes cross-split duplication structurally impossible rather than
    something `verify_zero_duplicate_feature_hashes` has to catch after the
    fact -- that check is still run as a hard invariant regardless, to
    prove the merge actually worked, not just trust that it did.

    This function exists because the first version of this script split on
    sample-id groups alone and `verify_zero_duplicate_feature_hashes` then
    failed for real on the full dataset: 20 hashes (80 rows) spanning
    splits, roughly evenly split between benign near-duplicates and
    cross-sample malicious collisions. Dropping those rows was considered
    and rejected -- they're genuine (if redundant) samples, not corrupt
    data -- in favor of keeping every row and just making sure duplicates
    of any origin stay together."""
    base_group_id = _assign_groups(df)
    feature_hash = _feature_hashes(df)

    uf = _UnionFind()
    hash_to_groups: dict[str, set] = defaultdict(set)
    for gid, h in zip(base_group_id, feature_hash):
        uf.find(gid)
        hash_to_groups[h].add(gid)
    for gids in hash_to_groups.values():
        gids = list(gids)
        for other in gids[1:]:
            uf.union(gids[0], other)

    return base_group_id.map(uf.find)


def report_group_sizes(df: pd.DataFrame) -> None:
    sample_id = df[CATEGORY_COL].map(_sample_id)
    families = df[CATEGORY_COL].map(_family)

    logger.info("=== Sample (dump-group) size statistics, by label ===")
    for label in sorted(df["label"].unique()):
        idx = df.index[df["label"] == label]
        sizes = sample_id.loc[idx].value_counts()
        logger.info(
            "label=%d: %d rows, %d distinct sample(s), "
            "dumps-per-sample min=%d median=%.1f mean=%.1f max=%d",
            label, len(idx), len(sizes),
            int(sizes.min()), float(sizes.median()), float(sizes.mean()), int(sizes.max()),
        )
        # Only print the full per-value breakdown when there are few
        # distinct values (e.g. benign's single "Benign" bucket) -- printing
        # ~2900 individual malicious sample hashes would be noise, not
        # signal; the family-level breakdown below is the useful view there.
        if len(sizes) <= 20:
            for val, count in sizes.sort_values(ascending=False).items():
                logger.info("    %-70s n=%d", val, count)

    logger.info("=== Family-level row counts, by label ===")
    fam_counts = df.assign(_family=families).groupby(["label", "_family"]).size()
    for (label, family), count in fam_counts.items():
        logger.info("  label=%d family=%-20s n=%d", label, family, count)


def stratified_group_split(df: pd.DataFrame, val_frac: float, test_frac: float,
                           cal_frac: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group_id = _assign_merged_groups(df)
    df = df.copy()
    df["_group_id"] = group_id
    df["_family"] = df[CATEGORY_COL].map(_family)

    groups_raw: dict[str, list[int]] = defaultdict(list)
    for pos, gid in zip(df.index, df["_group_id"]):
        groups_raw[gid].append(pos)
    groups = {gid: np.array(positions) for gid, positions in groups_raw.items()}

    # Bucket each group by (family, label) -- same rationale as
    # split_behavioral.py's (source, label) bucketing: guarantees every
    # malware family (and benign) gets proportional representation in every
    # split, not just overall label balance. label is a hard invariant here
    # (verified separately: no exact-duplicate-feature-vector group ever
    # crosses benign/malware) -- family is not: an exact-duplicate-feature
    # union can tie together two different malware families (confirmed: 19
    # of 517 such groups on the real data), so a merged group is bucketed
    # under whichever family has the most rows in it, same fallback
    # split_behavioral.py uses for a sequence group spanning multiple
    # labels. This only affects which bucket the group counts toward for
    # proportional-representation purposes, not correctness.
    buckets: dict[tuple, list[np.ndarray]] = defaultdict(list)
    mixed_family_groups = 0
    for gid, positions in groups.items():
        families = df.loc[positions, "_family"]
        labels = df.loc[positions, "label"]
        assert labels.nunique() == 1, f"group {gid!r} spans multiple labels: {labels.unique().tolist()}"
        if families.nunique() > 1:
            mixed_family_groups += 1
        family = families.value_counts().idxmax()
        buckets[(family, int(labels.iloc[0]))].append(positions)

    if mixed_family_groups:
        logger.warning(
            "%d merged group(s) span more than one malware family -- an "
            "exact-duplicate-feature-vector union tied together samples "
            "from different families. Bucketed by majority-row family; see "
            "_assign_merged_groups' docstring.",
            mixed_family_groups,
        )

    split_col = np.empty(len(df), dtype=object)
    for (family, label), group_list in sorted(buckets.items()):
        order = rng.permutation(len(group_list))
        shuffled = [group_list[i] for i in order]
        total_rows = sum(len(g) for g in shuffled)
        target_val = round(total_rows * val_frac)
        target_test = round(total_rows * test_frac)
        target_cal = round(total_rows * cal_frac)

        val_count = test_count = cal_count = 0
        for positions in shuffled:
            if val_count < target_val:
                split_col[positions] = "val"
                val_count += len(positions)
            elif test_count < target_test:
                split_col[positions] = "test"
                test_count += len(positions)
            elif cal_count < target_cal:
                # cal is carved AFTER val and test from the same seeded
                # per-bucket permutation, so val and test receive exactly
                # the groups they did before this phase existed
                # (byte-identical output); cal is a deterministic slice of
                # what would otherwise have been train.
                split_col[positions] = "cal"
                cal_count += len(positions)
            else:
                split_col[positions] = "train"

    df["split"] = split_col
    return df.drop(columns=["_group_id", "_family"])


def verify_zero_group_leakage(df: pd.DataFrame) -> None:
    """Hard invariant: every merged group (sample-id groups, further
    unioned with any exact-duplicate-feature-vector groups -- see
    `_assign_merged_groups`) must map to exactly one split. Raises if
    violated -- not a soft warning."""
    group_id = _assign_merged_groups(df)
    check = df.assign(_group_id=group_id)
    violations = []
    for gid, sub in check.groupby("_group_id"):
        splits = set(sub["split"])
        if len(splits) > 1:
            violations.append((gid, sub.index.tolist(), splits))
    if violations:
        for gid, idxs, splits in violations[:10]:
            logger.error("GROUP LEAKAGE: group=%r rows=%s spans splits %s", gid, idxs[:5], sorted(splits))
        raise AssertionError(f"{len(violations)} group(s) still span more than one split -- grouping failed.")
    logger.info("Verified: 0 / %d groups span more than one split.", check["_group_id"].nunique())


def verify_zero_duplicate_feature_hashes(df: pd.DataFrame) -> None:
    """Hard invariant: no exact-duplicate feature vector may appear in more
    than one split. Should always hold by construction now -- exact-duplicate
    rows are unioned into the same merged group by `_assign_merged_groups`
    before splitting, so this exists to prove that worked, not because it's
    expected to catch anything. Kept as a hard invariant regardless (raises,
    not a soft warning): trusting the merge without re-checking the actual
    output would be exactly the kind of unverified assumption this script's
    other invariants exist to avoid."""
    check = df.assign(_feature_hash=_feature_hashes(df))

    violations = []
    for h, sub in check.groupby("_feature_hash"):
        splits = set(sub["split"])
        if len(splits) > 1:
            violations.append((h, sub.index.tolist(), splits))
    if violations:
        for h, idxs, splits in violations[:10]:
            logger.error("DUPLICATE FEATURE VECTOR: hash=%s rows=%s spans splits %s", h[:12], idxs[:5], sorted(splits))
        raise AssertionError(f"{len(violations)} duplicate feature-vector hash(es) span more than one split.")
    logger.info(
        "Verified: 0 duplicate feature-vector hashes span more than one split (%d unique vectors / %d rows).",
        check["_feature_hash"].nunique(), len(check),
    )


def report_split(df: pd.DataFrame, val_frac: float, test_frac: float, cal_frac: float) -> None:
    n = len(df)
    logger.info("=== Split report ===")
    logger.info("Total rows: %d", n)
    for split, target_frac in (("train", 1 - val_frac - test_frac - cal_frac),
                               ("val", val_frac), ("test", test_frac), ("cal", cal_frac)):
        sub = df[df["split"] == split]
        actual_frac = len(sub) / n
        n_benign = int((sub["label"] == 0).sum())
        n_malware = int((sub["label"] == 1).sum())
        logger.info(
            "%s: n=%d (%.2f%% of total, target %.2f%%, delta %+.2f pp) benign=%d malware=%d",
            split, len(sub), 100 * actual_frac, 100 * target_frac, 100 * (actual_frac - target_frac),
            n_benign, n_malware,
        )
        families = sub[CATEGORY_COL].map(_family)
        for family, group in sub.groupby(families, sort=True):
            logger.info("    family=%-20s label=%s n=%d", family, sorted(group["label"].unique()), len(group))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--val-out", required=True)
    ap.add_argument("--test-out", required=True)
    ap.add_argument("--cal-out", required=True,
                    help="Calibration split -- Platt fit + config/thresholds.yaml derivation.")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--cal-frac", type=float, required=True,
                    help="Fraction of the dataset for the calibration split. Required, no "
                         "default: memory uses 0.2 (enlarged, matches the benign count the "
                         "pre-split-discipline val+test derivation relied on); network uses 0.1.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    if CLASS_COL in df.columns and "label" not in df.columns:
        raise ValueError(f"Expected a 'label' column (from data.download_memory.add_binary_label) in {args.inp}")

    report_group_sizes(df)
    df = stratified_group_split(df, args.val_frac, args.test_frac, args.cal_frac, args.seed)
    report_split(df, args.val_frac, args.test_frac, args.cal_frac)
    verify_zero_group_leakage(df)
    verify_zero_duplicate_feature_hashes(df)

    for split, out_path in (("train", args.train_out), ("val", args.val_out),
                            ("test", args.test_out), ("cal", args.cal_out)):
        sub = df[df["split"] == split].drop(columns=["split"]).reset_index(drop=True)
        sub.to_parquet(out_path, index=False)
        logger.info("Saved %s split (%d rows) to %s", split, len(sub), out_path)


if __name__ == "__main__":
    main()
