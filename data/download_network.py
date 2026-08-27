"""
Cortex-Network: load and validate the CSE-CIC-IDS2018 network-flow dataset
(CICFlowMeter-V3-extracted flow records, Sharafaldin et al./CIC-ISCX,
mirrored on AWS Open Data -- s3://cse-cic-ids2018/, ca-central-1, no
account or access-request form required, unlike CIC-MalMem-2022).

The bucket has two top-level prefixes: "Original Network Traffic and Log
data/" (raw PCAPs + logs, per-day subfolders -- NOT used here) and
"Processed Traffic Data for ML Algorithms/" (10 CSVs, one per capture day,
already run through CICFlowMeter -- this is what this module loads). This
module does NOT sync from S3 itself; that's a one-time manual step:

    aws s3 sync --no-sign-request --region ca-central-1 \\
        "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" \\
        data/raw/network/

Real structure, verified against the actual downloaded files (not assumed
from the docs, which only claim "~80 features" and "multiple days"):

- 10 files, 6.4 GiB, spanning Feb 14 - Mar 2 2018. 16,233,002 rows total.
- Column count is NOT uniform: 9 of 10 files have 80 columns (79 features +
  Label). `Thuesday-20-02-2018` (also the dataset's size outlier: ~4 GiB
  vs ~330 MiB typical for the other 9) has 84 -- 4 extra leading identifier
  columns (Flow ID, Src IP, Src Port, Dst IP) present only in that file.
  See DROP_IDENTITY_COLUMNS below for why these are dropped, not aligned.
- 59 rows (out of 16.2M, across 3 files) are literal header-repeats: the
  header line reappears as a data row, so e.g. the Label column's value on
  that row is the literal string "Label", not a real label, and because a
  literal column-name string breaks numeric parsing, EVERY column in an
  affected file loads as `object` dtype until these rows are filtered out.
  See ALLOWED_LABELS / _filter_malformed_rows below.
- `Flow Byts/s` and `Flow Pkts/s` contain real NaN/Infinity values (division
  by zero when Flow Duration=0) -- confirmed present (thousands of rows) on
  a file with no other schema issues, so this is a genuine data property,
  not a symptom of the header-repeat rows above.
- Label taxonomy is inconsistent in casing/wording across files ("DDoS
  attacks-LOIC-HTTP" vs "DDOS attack-HOIC" / "DDOS attack-LOIC-UDP") --
  handled by scripts/split_network.py's binary-label derivation
  (`label.strip().lower() == "benign"`), not here; this module keeps the
  raw string as-is in `label_raw` rather than attempting to normalize
  spelling.

Usage:
    python -m data.download_network --in-dir data/raw/network \\
        --out data/processed/network_dataset.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("cortex.data.network")

EXPECTED_FILE_COUNT = 10
EXPECTED_TOTAL_ROWS = 16_233_002  # soft check only, like download_memory.py's EXPECTED_ROW_COUNT

LABEL_COL = "Label"
TIMESTAMP_COL = "Timestamp"
BENIGN_LABEL = "Benign"

# The 4 extra leading columns present ONLY in Thuesday-20-02-2018 (the raw
# per-flow identifiers CICFlowMeter can emit but which the other 9 files'
# processing pipeline stripped out). Dropped, not kept-and-aligned, for a
# specific reason beyond matching column counts: CSE-CIC-IDS2018 was
# captured on a small, FIXED testbed -- a handful of specific attacker and
# victim machines reused across the whole multi-day capture. A model with
# access to Src IP / Dst IP / Flow ID could trivially learn "traffic
# to/from this specific IP is malicious" (i.e. memorize which of the
# testbed's fixed machines played the attacker role) rather than learning
# any traffic-pattern signal that would transfer to a real network with
# different machines at different addresses. This is the same class of
# shortcut-learning risk already found and documented for Cortex-Memory
# (CIC-MalMem-2022's single-VM benign collection letting a model learn
# "which VM" instead of "is this malicious") -- here we can eliminate it
# entirely at the source instead of just documenting it, since the
# identity-carrying columns aren't the dataset's actual traffic-pattern
# signal to begin with.
DROP_IDENTITY_COLUMNS = ["Flow ID", "Src IP", "Src Port", "Dst IP"]

# Verified by reading data/raw/network/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv's
# actual header (one of the 9 files with the standard, non-Tuesday layout).
STANDARD_COLUMNS = [
    "Dst Port", "Protocol", "Timestamp", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean",
    "Fwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
    "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Tot",
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var", "FIN Flag Cnt",
    "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count",
    "ECE Flag Cnt", "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg",
    "Bwd Blk Rate Avg", "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts",
    "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts", "Fwd Seg Size Min", "Active Mean",
    "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label",
]
# 78 numeric flow-feature columns: STANDARD_COLUMNS minus Timestamp and
# Label. Timestamp is deliberately EXCLUDED from the feature set (kept only
# as metadata) for the same shortcut-learning reason as DROP_IDENTITY_COLUMNS
# above: each capture day is (almost) entirely one attack scenario (see the
# label taxonomy in the module docstring), so a raw wall-clock timestamp --
# or even the derived `capture_day` column below -- would let a model learn
# "which day/date this is" as a near-perfect proxy for the label, instead of
# learning anything about the actual traffic pattern. Both stay in the
# processed output for split-stratification and provenance, never as a
# training feature.
FEATURE_COLUMNS = [c for c in STANDARD_COLUMNS if c not in (TIMESTAMP_COL, LABEL_COL)]

# Verified by scanning the Label column of all 10 real files (16.2M values)
# and taking the distinct set -- not copied from the docs. "Label" itself
# (the literal header-repeat artifact) is deliberately excluded: any row
# whose Label doesn't match one of these 15 is dropped as malformed.
ALLOWED_LABELS = frozenset({
    "Benign", "Bot", "Brute Force -Web", "Brute Force -XSS", "DDOS attack-HOIC",
    "DDOS attack-LOIC-UDP", "DDoS attacks-LOIC-HTTP", "DoS attacks-GoldenEye", "DoS attacks-Hulk",
    "DoS attacks-SlowHTTPTest", "DoS attacks-Slowloris", "FTP-BruteForce", "Infilteration",
    "SQL Injection", "SSH-Bruteforce",
})


def _day_from_filename(path: Path) -> str:
    """"Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv" -> "Wednesday-14-02-2018".
    Keeps the dataset's own filename verbatim, including its "Thuesday" typo
    for the Feb 20 file -- not silently corrected, since that typo is part
    of how the source data identifies that specific (oversized, 84-column)
    file, and correcting it here would make it harder to trace a row back
    to its actual source filename."""
    return path.stem.split("_", 1)[0]


def load_and_validate_file(csv_path: str) -> pd.DataFrame:
    """Load one raw CSE-CIC-IDS2018 CSV, validate its schema, and clean it:
    align the Tuesday file's extra identity columns, drop malformed
    header-repeat rows, drop NaN/Infinity Flow Byts/s or Flow Pkts/s rows,
    and downcast numeric columns to float32. Returns a DataFrame with
    columns: `capture_day`, `Timestamp`, the 78 FEATURE_COLUMNS, `label_raw`,
    `label` (int32 binary target). Raises ValueError on any schema mismatch
    -- never silently proceeds on a file that doesn't match what this
    module was built against.
    """
    path = Path(csv_path)
    if not path.exists():
        raise ValueError(f"{csv_path} does not exist.")

    df = pd.read_csv(path, low_memory=False)

    if list(df.columns) == DROP_IDENTITY_COLUMNS + STANDARD_COLUMNS:
        df = df.drop(columns=DROP_IDENTITY_COLUMNS)
    elif list(df.columns) != STANDARD_COLUMNS:
        raise ValueError(
            f"{csv_path}: columns don't match the standard 80-column layout or the "
            f"84-column (Tuesday-file) layout with 4 extra leading identity columns. "
            f"Got {len(df.columns)} columns: {list(df.columns)}"
        )

    n_before = len(df)
    malformed_mask = ~df[LABEL_COL].isin(ALLOWED_LABELS)
    n_malformed = int(malformed_mask.sum())
    if n_malformed:
        unexpected = sorted(df.loc[malformed_mask, LABEL_COL].unique().tolist())
        logger.warning("%s: dropping %d malformed row(s), unexpected Label value(s): %s",
                        path.name, n_malformed, unexpected)
        df = df.loc[~malformed_mask].copy()

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    n_before_naninf = len(df)
    bad_rate_mask = (
        df["Flow Byts/s"].isna() | df["Flow Pkts/s"].isna()
        | np.isinf(df["Flow Byts/s"]) | np.isinf(df["Flow Pkts/s"])
    )
    n_bad_rate = int(bad_rate_mask.sum())
    if n_bad_rate:
        # Dropped rather than imputed -- these are division-by-zero
        # artifacts (Flow Duration=0), not missing measurements with a
        # sensible fill value; a small, known fraction of rows (~0.3-0.4%
        # on the files checked), so dropping costs negligible data and
        # avoids inventing a number for something that has none.
        df = df.loc[~bad_rate_mask].copy()

    for col in FEATURE_COLUMNS:
        df[col] = df[col].astype(np.float32)

    df["capture_day"] = _day_from_filename(path)
    df["label_raw"] = df[LABEL_COL]
    df["label"] = (df[LABEL_COL].str.strip().str.lower() != BENIGN_LABEL.lower()).astype(np.int32)
    df = df.drop(columns=[LABEL_COL])

    logger.info(
        "%s: %d rows loaded, %d malformed dropped, %d NaN/Inf-rate dropped, %d clean rows kept",
        path.name, n_before, n_malformed, n_bad_rate, len(df),
    )
    return df[["capture_day", TIMESTAMP_COL] + FEATURE_COLUMNS + ["label_raw", "label"]]


def scan_stratum_sizes(raw_dir: str) -> dict[tuple[str, str], int]:
    """Cheap first pass: read only the Label column of every file (not the
    full 78-feature row) to get exact (capture_day, label_raw) row counts,
    used by compute_sampling_plan() below. Malformed rows are excluded here
    too, so the plan isn't skewed by them."""
    sizes: dict[tuple[str, str], int] = {}
    for path in sorted(Path(raw_dir).glob("*.csv")):
        day = _day_from_filename(path)
        labels = pd.read_csv(path, usecols=[LABEL_COL])[LABEL_COL]
        labels = labels[labels.isin(ALLOWED_LABELS)]
        for label_raw, count in labels.value_counts().items():
            sizes[(day, label_raw)] = sizes.get((day, label_raw), 0) + int(count)
    return sizes


def compute_sampling_plan(
    stratum_sizes: dict[tuple[str, str], int], target_total_rows: int, min_stratum_rows: int,
) -> dict[tuple[str, str], int]:
    """Decides how many rows to keep from each (capture_day, label_raw)
    stratum: every stratum at or below `min_stratum_rows` is kept in FULL
    (protects the genuinely rare attack types -- e.g. SQL Injection is 87
    rows total across the whole dataset; proportional-only sampling at any
    real-world target size would round that to a handful of rows or zero).
    Every larger stratum is downsampled proportionally to its share of the
    remaining "downsamplable" pool, so the *relative* sizes of the large
    strata are preserved (a day/attack-type combination that's naturally
    3x bigger than another stays roughly 3x bigger in the sample) rather
    than being forced to equal size."""
    protected = {k: v for k, v in stratum_sizes.items() if v <= min_stratum_rows}
    downsamplable = {k: v for k, v in stratum_sizes.items() if v > min_stratum_rows}

    protected_total = sum(protected.values())
    downsamplable_total = sum(downsamplable.values())
    remaining_budget = max(target_total_rows - protected_total, 0)

    plan: dict[tuple[str, str], int] = dict(protected)
    if downsamplable_total > 0:
        rate = min(remaining_budget / downsamplable_total, 1.0)
        for k, v in downsamplable.items():
            plan[k] = round(v * rate)
    return plan


def build_canonical_dataset(
    raw_dir: str, target_total_rows: int, min_stratum_rows: int, seed: int,
) -> pd.DataFrame:
    """Loads, cleans, and stratified-samples all 10 files into one DataFrame
    -- one file fully in memory at a time, immediately reduced to its
    planned sample before the next file is loaded, so the full 16.2M-row
    uncapped dataset is never assembled in memory (its largest single file,
    Thuesday-20-02-2018, is 4 GiB of CSV text alone). See compute_sampling_plan()
    for how per-stratum sample sizes are chosen, and the module docstring
    for why a cap is needed at all (memory safety) given static_lgbm.py's
    own history of an OOM from a similar full-copy-everything pattern at
    a larger scale."""
    logger.info("Scanning stratum sizes (Label column only) across %s ...", raw_dir)
    stratum_sizes = scan_stratum_sizes(raw_dir)
    plan = compute_sampling_plan(stratum_sizes, target_total_rows, min_stratum_rows)
    logger.info(
        "Sampling plan: %d strata, target_total=%d, min_stratum_rows=%d, planned_total=%d",
        len(plan), target_total_rows, min_stratum_rows, sum(plan.values()),
    )

    rng = np.random.default_rng(seed)
    samples = []
    for path in sorted(Path(raw_dir).glob("*.csv")):
        day_df = load_and_validate_file(str(path))
        day = _day_from_filename(path)
        day_samples = []
        for label_raw, group in day_df.groupby("label_raw"):
            n_target = plan.get((day, label_raw), len(group))
            n_take = min(n_target, len(group))
            if n_take < len(group):
                idx = rng.choice(group.index.to_numpy(), size=n_take, replace=False)
                day_samples.append(group.loc[idx])
            else:
                day_samples.append(group)
        sampled = pd.concat(day_samples, ignore_index=True) if day_samples else day_df.iloc[0:0]
        logger.info("  %s: kept %d / %d rows after sampling", path.name, len(sampled), len(day_df))
        samples.append(sampled)
        del day_df, day_samples

    canonical = pd.concat(samples, ignore_index=True)
    logger.info("Canonical dataset: %d rows, %d columns", len(canonical), canonical.shape[1])
    return canonical


def report(df: pd.DataFrame) -> None:
    n = len(df)
    n_benign = int((df["label"] == 0).sum())
    n_malicious = int((df["label"] == 1).sum())
    logger.info("Total rows: %d (benign=%d, malicious=%d, %.1f%% malicious)",
                n, n_benign, n_malicious, 100 * n_malicious / n)
    logger.info("=== Rows per (capture_day, label_raw) ===")
    for (day, label_raw), count in df.groupby(["capture_day", "label_raw"]).size().items():
        logger.info("  day=%-22s label_raw=%-25s n=%d", day, label_raw, count)


def save(df: pd.DataFrame, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/raw/network", help="Directory containing the 10 synced CSVs.")
    ap.add_argument("--out", default=None, help="If omitted, only prints the report -- nothing is written.")
    ap.add_argument("--target-rows", type=int, default=2_000_000)
    ap.add_argument("--min-stratum-rows", type=int, default=15_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = build_canonical_dataset(args.in_dir, args.target_rows, args.min_stratum_rows, args.seed)
    report(df)

    if args.out:
        save(df, args.out)
    else:
        logger.info("--out not given: report only, nothing written to disk.")


if __name__ == "__main__":
    main()
