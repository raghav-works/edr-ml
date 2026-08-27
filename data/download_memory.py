"""
Cortex-Memory: load and validate the CIC-MalMem-2022 memory-forensics
dataset (Carrier et al., ICISSP 2022).

The dataset is gated behind a short access-request form on UNB's site
(https://www.unb.ca/cic/datasets/malmem-2022.html) and cannot be fetched
programmatically -- unlike EMBER2024/the behavioral sources, there is no
downloadable URL to wire in here. This module therefore does NOT download
anything; it loads and validates a CSV you've already obtained manually.
Kept the `download_memory.py` name for layout symmetry with
data/download_behavioral.py and data/download_ember2024.py, even though
"load" is the more literally accurate description of what it does.

Provenance: the working copy at data/raw/memory/Obfuscated-MalMem2022.csv
(gitignored, not committed) was pulled from the Kaggle mirror
"luccagodoy/obfuscated-malware-memory-2022-cic" on 2026-08-25, not the
official UNB/CIC portal linked above -- at the time of this pull, the
official portal's download was missing 3 of the dataset's 4 top-level
categories. The Kaggle mirror's row/column counts and value distributions
were checked against the dataset's published description before trusting it
(see the schema constants below); if you re-pull this, re-verify the same
way rather than assuming any mirror is complete.

Actual raw CSV shape (verified against the real file, not assumed):
58,596 rows, 57 columns in this exact order -- `Category` (first), 55
VolMemLyzer-derived numeric feature columns (pslist./handles./ldrmodules/
malfind/psxview/modules/svcscan/callbacks-prefixed), `Class` (last, values
exactly "Benign"/"Malware"). Column order is checked but not required for
correctness -- `feature_columns()` below detects feature columns by
exclusion, not position, so a reordered-but-otherwise-matching file would
still load correctly; the order check exists to flag an unexpected file
loudly rather than stay silent about a mismatch from what was verified here.
`Category` values: the literal constant string "Benign" for every benign
row (no per-row identifier -- confirmed against the real data, not a
parsing artifact), and "<Type>-<Family>-<sha256 hex>-<dump#>.raw" for
malicious rows (e.g. "Ransomware-Ako-00a14062c268...-1.raw") -- the hex
string identifies one specific malware sample, and multiple rows sharing an
identical "<Type>-<Family>-<hex>" prefix are different memory dumps
(dump# 1..~10) captured from that same running sample. See
scripts/split_memory.py for how this is used to prevent same-sample dumps
from spanning train/val/test.

Schema is validated structurally here -- column count, dtypes, exact
`Class` value set, first/last column identity -- not against a hardcoded
list of the 55 individual feature names. The precise name set was not
independently reproduced for this module, and a hardcoded name list risks
silently accepting a mismatched file (if names happen to collide by chance)
while missing a real mismatch elsewhere; the structural checks below catch
both failure modes without needing to trust a name list that might itself
be wrong.

Usage:
    python -m data.download_memory --in data/raw/memory/Obfuscated-MalMem2022.csv
    python -m data.download_memory --in data/raw/memory/Obfuscated-MalMem2022.csv --out data/processed/memory_dataset.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("cortex.data.memory")

EXPECTED_FEATURE_COUNT = 55
EXPECTED_ROW_COUNT = 58596  # logged as a soft check only, not enforced -- a
                              # different release or a legitimate subset of
                              # the dataset is still usable as long as the
                              # schema matches.
CATEGORY_COL = "Category"
CLASS_COL = "Class"
BENIGN_LABEL = "Benign"
MALWARE_LABEL = "Malware"


def load_and_validate(csv_path: str) -> pd.DataFrame:
    """Load the raw CIC-MalMem-2022 CSV and validate its schema.

    Raises ValueError with a specific message on any mismatch -- never
    silently proceeds on a file that doesn't match what this module was
    built against.
    """
    path = Path(csv_path)
    if not path.exists():
        raise ValueError(f"{csv_path} does not exist.")

    df = pd.read_csv(path)

    missing = [c for c in (CATEGORY_COL, CLASS_COL) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns {missing} not found in {csv_path}. "
            f"Columns present: {list(df.columns)}"
        )

    # Not required for correctness (feature_columns() below detects feature
    # columns by exclusion, not position), but this is the exact order
    # verified against the real file -- a mismatch here means the file
    # differs from what this module was checked against, worth flagging
    # loudly rather than silently proceeding.
    if df.columns[0] != CATEGORY_COL or df.columns[-1] != CLASS_COL:
        raise ValueError(
            f"Expected {CATEGORY_COL!r} as the first column and {CLASS_COL!r} as the "
            f"last column (the verified real layout); got first={df.columns[0]!r}, "
            f"last={df.columns[-1]!r}. Columns present: {list(df.columns)}"
        )

    feature_cols = [c for c in df.columns if c not in (CATEGORY_COL, CLASS_COL)]
    if len(feature_cols) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} feature columns, found "
            f"{len(feature_cols)}: {feature_cols}. This CSV's schema does not "
            f"match what data/download_memory.py was built against -- inspect "
            f"it manually (df.columns) before adjusting this check."
        )

    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(
            f"Expected all {EXPECTED_FEATURE_COUNT} feature columns to be numeric; "
            f"non-numeric columns found: {non_numeric}"
        )

    unexpected_classes = set(df[CLASS_COL].unique()) - {BENIGN_LABEL, MALWARE_LABEL}
    if unexpected_classes:
        raise ValueError(
            f"Expected {CLASS_COL!r} to only contain {{{BENIGN_LABEL!r}, {MALWARE_LABEL!r}}}, "
            f"found unexpected values: {unexpected_classes}"
        )

    if len(df) != EXPECTED_ROW_COUNT:
        logger.warning(
            "Row count %d does not match the dataset's documented %d -- "
            "proceeding anyway (could be a different release or a legitimate "
            "subset), but double-check this is the file you meant to load.",
            len(df), EXPECTED_ROW_COUNT,
        )

    logger.info("Loaded %d rows, %d feature columns from %s", len(df), len(feature_cols), csv_path)
    return df


def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Adds an integer `label` column: 0 = Benign, 1 = Malware."""
    df = df.copy()
    df["label"] = (df[CLASS_COL] == MALWARE_LABEL).astype(np.int32)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (CATEGORY_COL, CLASS_COL, "label")]


def report(df: pd.DataFrame) -> None:
    """Basic load-time report -- class balance and a Category-value
    overview. This is deliberately lightweight: real Category values are
    per-dump-unique for malicious rows (~28K distinct values across 58K
    rows), so dumping every value here would be noise, not signal. The
    real per-sample/per-family group-size investigation this dataset needs
    lives in scripts/split_memory.py (`report_group_sizes`), which groups
    on sample id, not raw Category, and is the right place for it."""
    n = len(df)
    n_benign = int((df["label"] == 0).sum())
    n_malware = int((df["label"] == 1).sum())
    logger.info(
        "Total rows: %d (benign=%d, malware=%d, %.1f%% malware)",
        n, n_benign, n_malware, 100 * n_malware / n,
    )

    cat_counts = df[CATEGORY_COL].value_counts()
    n_benign_cats = int((cat_counts.index == BENIGN_LABEL).sum())
    logger.info(
        "Distinct Category values: %d (%d benign constant + %d malicious, "
        "malicious values are per-dump so mostly unique -- see "
        "scripts/split_memory.py for the sample/family-level breakdown)",
        len(cat_counts), n_benign_cats, len(cat_counts) - n_benign_cats,
    )
    exact_dup_cats = cat_counts[cat_counts > 1]
    exact_dup_cats = exact_dup_cats[exact_dup_cats.index != BENIGN_LABEL]
    if len(exact_dup_cats):
        logger.warning(
            "%d malicious Category value(s) appear on more than one row "
            "(same '<Type>-<Family>-<hash>-<dump#>.raw' string repeated) -- "
            "%d rows total. Spot-checked: this is a mix of true exact-duplicate "
            "rows and same-label rows with different feature vectors (a label "
            "collision, not a data copy) -- both cases still keep their rows in "
            "the same sample-id group in scripts/split_memory.py, so this does "
            "not create a leakage risk, but it's a real quirk of the source "
            "data worth knowing about.",
            len(exact_dup_cats), int(exact_dup_cats.sum()),
        )


def save(df: pd.DataFrame, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to the manually-downloaded CIC-MalMem-2022 CSV.")
    ap.add_argument("--out", default=None, help="If omitted, only prints the report -- nothing is written.")
    args = ap.parse_args()

    df = load_and_validate(args.inp)
    df = add_binary_label(df)
    report(df)

    if args.out:
        save(df, args.out)
    else:
        logger.info("--out not given: report only, nothing written to disk.")


if __name__ == "__main__":
    main()
