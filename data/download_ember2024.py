"""
Download EMBER2024 (joyce8/EMBER2024 on HuggingFace) and de-duplicate.

Known issue this script guards against: the public parquet release has been
observed to contain duplicated rows (roughly 2x the expected sample count for
a given split -- same sha256 appearing twice, most likely from an upstream
export step writing each shard twice or overlapping shard boundaries). We
de-duplicate on sha256 before anything else touches the data, and we log the
before/after counts so the discrepancy is always visible rather than silently
baked into training.

Deliberately does NOT use `datasets.load_dataset()`. That path was tried
first and hits a real bug in the `datasets` library's Arrow schema
inference: `load_dataset(..., split="test")` auto-concatenates every
file-format family's zip for that split and infers one merged schema across
them, and a field that's populated (string) in one format's shard but
entirely absent (typed as `null`) in another's makes the cast fail
(`TypeError: Couldn't cast array of type string to null`). It fails the same
way even restricted to a single format's shard, because the per-file JSON
schema inference chunk can miss a field that's rarely populated. Streaming
the raw JSONL out of each zip ourselves sidesteps Arrow schema inference
entirely.

Also deliberately restricted to PE-container formats. The HF repo actually
ships one zip per (file-format family, split): APK, Dot_Net, ELF, PDF,
Win32, Win64 (+ a `challenge.zip`, not handled here). This project's static
model (`features/pe_features.py`) is a PE-based (pefile) extractor, so only
the families that are genuinely valid PE containers are relevant: Win32 and
Win64 are native PE, and Dot_Net assemblies also carry a real PE/COFF header
(with a CLR runtime header layered on top), so pefile can parse them too.
APK (Android), ELF (Linux), and PDF are not PE format and are excluded.

Streams records instead of building one big in-memory DataFrame: measured
~65KB of Python-object overhead per raw EMBER2024 record, which would put a
full deduped test split (~540K rows) at ~35GB RSS and a full train split
(~2.3M rows after dedup) at over 100GB -- not safe to hold in memory at
once. Records are deduped on the fly (`keep first occurrence of a sha256`,
same semantics as the original `drop_duplicates(keep="first")`), vectorized
through features/ember2024_adapter.py (2568 float32 columns), and written to
parquet in bounded batches -- the bulky raw JSON is never held for more than
one batch at a time and never touches disk.

Usage:
    python -m data.download_ember2024 --split train --out data/processed/ember2024_train.parquet
    python -m data.download_ember2024 --split test  --out data/processed/ember2024_test.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from features.ember2024_adapter import record_to_vector
from features.pe_features import EMBER2024_FEATURE_COUNT

logger = logging.getLogger("cortex.data.ember2024")

HF_DATASET = "joyce8/EMBER2024"
PE_FORMATS = ["Win32", "Win64", "Dot_Net"]
BATCH_SIZE = 20_000


def _iter_raw_records(split: str) -> Iterator[dict]:
    """Stream every JSON record for one split across PE_FORMATS directly out
    of each format's zip, without going through `datasets.load_dataset()`."""
    for fmt in PE_FORMATS:
        filename = f"{fmt}_{split}.zip"
        logger.info("Fetching %s from %s", filename, HF_DATASET)
        zip_path = hf_hub_download(HF_DATASET, filename, repo_type="dataset")
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                with zf.open(name) as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)


def download_and_dedup(split: str, out_path: Path) -> None:
    """Stream all PE-format records for `split`, de-dup on sha256 (keep
    first occurrence), vectorize each deduped record through
    features/ember2024_adapter.py, and write to `out_path` as parquet in
    bounded-memory batches.

    Output schema: sha256, label, feature_0..feature_{EMBER2024_FEATURE_COUNT-1}
    (float32) -- exactly what scripts/train_static.py::_load_train_val /
    _load_cal expect as their parquet schema.
    """
    seen: set[str] = set()
    n_before = 0
    n_after = 0
    batch: list[dict] = []
    writer: Optional[pq.ParquetWriter] = None

    def flush() -> None:
        nonlocal writer, batch
        if not batch:
            return
        vectors = np.stack([record_to_vector(r) for r in batch]).T  # (2568, n), contiguous per feature
        columns = {f"feature_{i}": vectors[i] for i in range(EMBER2024_FEATURE_COUNT)}
        columns["sha256"] = [r["sha256"] for r in batch]
        columns["label"] = [r["label"] for r in batch]
        table = pa.table(columns)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), table.schema)
        writer.write_table(table)
        batch = []

    for rec in _iter_raw_records(split):
        n_before += 1
        sha = rec["sha256"]
        if sha in seen:
            continue
        seen.add(sha)
        n_after += 1
        batch.append(rec)
        if len(batch) >= BATCH_SIZE:
            flush()
        if n_before % 100_000 == 0:
            logger.info("...%d rows scanned, %d unique so far", n_before, n_after)

    flush()
    if writer is not None:
        writer.close()

    reduction = 100 * (1 - n_after / n_before) if n_before else 0.0
    logger.info("Deduplication: %d -> %d rows (%.1f%% removed)", n_before, n_after, reduction)
    if n_before > 0 and abs(n_after - n_before / 2) < 0.05 * (n_before / 2):
        logger.warning(
            "Row count dropped to ~50%% after dedup, consistent with the known "
            "'each sample present twice' issue in the EMBER2024 export."
        )
    logger.info("Saved deduplicated split to %s (%d rows)", out_path, n_after)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    download_and_dedup(args.split, out_path)


if __name__ == "__main__":
    main()
