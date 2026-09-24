"""
Build the canonical behavioral (API-call-sequence) dataset from Mal-API-2019,
MalbehavD-V1, and Carpenter (benign rows only). No synthetic backfill, no
Quo Vadis -- first version uses only these three public sources, per the
behavioral dataset plan.

Sources (all verified against the publishers' stated dataset stats before
being wired into this script -- see the Stage 1/2 inspections):
    Mal-API-2019:  https://github.com/ocatak/malware_api_class
                   mal-api-2019.zip (contains all_analysis_data.txt, one
                   space-separated lowercase API-call sequence per line) +
                   labels.csv (one family string per line, joined by line
                   position). 7,107 rows: Worms 1001, Virus 1001, Trojan
                   1001, Downloader 1001, Backdoor 1001, Dropper 891,
                   Spyware 832, Adware 379 -- all malicious, no benign
                   samples, so every row maps to label=1 regardless of
                   family (the family string is kept in `family_raw` for
                   provenance only, not used for training).
    MalbehavD-V1:  https://github.com/mpasco/MalbehavD-V1
                   MalBehavD-V1-dataset.csv, columns: sha256, labels,
                   0, 1, 2, ..., 152 (ragged -- the numbered columns are an
                   ordered API-call sequence, empty-padded to the row's
                   max width). 2,570 rows: 1,285 benign (labels=0), 1,285
                   malicious (labels=1).
    Carpenter:     https://www.kaggle.com/datasets/marcuscarpenter97/api-calls-generated-by-dynamic-malware-analysis
                   ("Behavioural Reports of Multi-Stage Malware" dataset).
                   The zip has three per-batch JSON files (each
                   {"name", "year", "apis": [[call, call, ...], ...],
                   "labels": [[one-hot over 15 categories], ...]}) --
                   "368.json" and "389.json" are malicious batches (their
                   one-hot labels never set the "benign" index), "benign.json"
                   is the only benign one, 101 sequences. Only benign.json is
                   used here -- the malicious batches are deliberately never
                   downloaded/extracted (they're ~2.5GB combined and
                   irrelevant, since Mal-API-2019/MalbehavD-V1 already cover
                   malicious volume).

Vocabulary checks (done manually before wiring each source into this script,
not repeated at runtime):
  - MalbehavD-V1's API names are the canonical Windows API casing (e.g.
    "CoCreateInstance"); Mal-API-2019's are already all-lowercase. Of
    Mal-API-2019's 278 unique tokens, 269 (96.8%) match a MalbehavD-V1 token
    exactly once both are lowercased -- no missing Nt/Zw prefixes, no
    truncation, a clean casing-only difference. The 9 that don't match
    include sentinel tokens "__anomaly__" and "__exception__" (Cuckoo
    Sandbox markers for anomalous/exception events, not real API calls).

    CORRECTED 2026-09-22 -- the original version of this docstring claimed
    these two sentinel tokens were "genuinely absent from MalbehavD-V1, not
    casing artifacts." That claim was factually wrong, not merely stale:
    checked directly against data/processed/behavioral_dataset.parquet,
    "__exception__" appears in 429 MalbehavD-V1 rows (78 benign-labeled,
    351 malicious-labeled), and "__anomaly__" appears in 7 Carpenter
    benign.json rows (all benign-labeled) -- confirmed by spot-checking a
    benign MalbehavD-V1 row with "__exception__" embedded among genuine
    Windows API call names, not a casing/matching artifact. Both sentinel
    tokens are present on both malicious- and benign-labeled rows, across
    more than one source. (The other 8 of the 9 non-matching tokens were
    not re-checked; this correction is scoped to the two sentinel tokens
    only.) See docs/TECHNICAL_NOTES.md's "Known limitation (behavioral sentinel tokens)"
    section for the resulting dataset-shortcut investigation and its
    outcome (no fix applied; aggregate dependence across the test split is
    low, but two individual rows show real per-row dependence on a single
    sentinel token).
  - Carpenter's benign.json uses the same canonical Windows API casing as
    MalbehavD-V1. Of its 212 unique tokens, 144 (67.9%) match the existing
    Mal-API-2019+MalbehavD-V1 vocabulary once lowercased (spot-checked
    LdrLoadDll, NtProtectVirtualMemory, NtClose, RegOpenKeyExW -- all match
    cleanly). The 68 that don't match are genuinely new real Windows API
    names (WinHttp* networking calls, additional Crypt*/Nt* functions), not
    casing artifacts.
Lowercasing all three sources is therefore sufficient to unify them into one
vocabulary space -- no further name remapping needed.

Usage:
    python -m data.download_behavioral
    python -m data.download_behavioral --out data/processed/behavioral_dataset.parquet
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("cortex.data.behavioral")

MAL_API_2019_ZIP_URL = "https://raw.githubusercontent.com/ocatak/malware_api_class/master/mal-api-2019.zip"
MAL_API_2019_LABELS_URL = "https://raw.githubusercontent.com/ocatak/malware_api_class/master/labels.csv"
MALBEHAVD_V1_CSV_URL = "https://raw.githubusercontent.com/mpasco/MalbehavD-V1/main/MalBehavD-V1-dataset.csv"
CARPENTER_ZIP_URL = "https://www.kaggle.com/api/v1/datasets/download/marcuscarpenter97/api-calls-generated-by-dynamic-malware-analysis"


def _download(url: str, cache_dir: Path, filename: Optional[str] = None) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / (filename or url.rsplit("/", 1)[-1])
    if not dest.exists():
        logger.info("Downloading %s -> %s", url, dest)
        urllib.request.urlretrieve(url, dest)
    else:
        logger.info("Using cached %s", dest)
    return dest


def _iter_mal_api_2019(cache_dir: Path) -> Iterator[dict]:
    zip_path = _download(MAL_API_2019_ZIP_URL, cache_dir)
    labels_path = _download(MAL_API_2019_LABELS_URL, cache_dir)

    with open(labels_path, newline="") as f:
        families = [line.strip() for line in f if line.strip()]

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("all_analysis_data.txt") as f:
            for i, line in enumerate(f):
                raw = line.rstrip(b"\n")
                api_calls = [tok.lower() for tok in raw.decode("ascii", "ignore").split()]
                yield {
                    "id": hashlib.sha256(raw).hexdigest(),
                    "source": "mal-api-2019",
                    "family_raw": families[i],
                    "label": 1,  # every Mal-API-2019 sample is malicious
                    "api_calls": api_calls,
                }


def _iter_malbehavd_v1(cache_dir: Path) -> Iterator[dict]:
    csv_path = _download(MALBEHAVD_V1_CSV_URL, cache_dir)

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Numbered columns ("0", "1", ...) in file order -- already ascending
        # in the source file, but sort explicitly by int value to be robust
        # to any future column reordering.
        numbered_idxs = sorted(
            (i for i, name in enumerate(header) if name.isdigit()),
            key=lambda i: int(header[i]),
        )
        for row in reader:
            api_calls = [row[i].lower() for i in numbered_idxs if row[i]]
            yield {
                "id": row[0],  # sha256
                "source": "malbehavd-v1",
                "family_raw": None,
                "label": int(row[1]),
                "api_calls": api_calls,
            }


def _iter_carpenter(cache_dir: Path) -> Iterator[dict]:
    zip_path = _download(CARPENTER_ZIP_URL, cache_dir, filename="carpenter.zip")

    with zipfile.ZipFile(zip_path) as zf:
        # Only benign.json is read -- 368.json/389.json (malicious batches,
        # ~2.5GB combined) are never extracted, since we only want Carpenter's
        # benign rows and already have plenty of malicious volume.
        with zf.open("data/Processed/benign.json") as f:
            data = json.load(f)
    if data.get("name") != "benign":
        raise ValueError(f"Expected Carpenter's benign.json to have name='benign', got {data.get('name')!r}")

    for i, seq in enumerate(data["apis"]):
        api_calls = [tok.lower() for tok in seq]
        yield {
            "id": "carpenter-benign-" + hashlib.sha256(" ".join(seq).encode()).hexdigest(),
            "source": "carpenter",
            "family_raw": None,
            "label": 0,  # benign.json only
            "api_calls": api_calls,
        }


def _dedup_by_id(records: list[dict]) -> list[dict]:
    """Drop rows with a duplicate `id` (keep first occurrence).

    Deliberately dedups on `id` (true record identity: a hash of the raw
    source line for mal-api-2019, the real file sha256 for malbehavd-v1/
    carpenter), NOT on the api_calls sequence itself. Those are not the
    same thing: many different real files -- especially malbehavd-v1's
    shorter sequences -- legitimately produce identical short API-call
    traces by coincidence (e.g. a generic ~10-call startup/cleanup
    sequence). Deduping on sequence would silently discard genuinely
    distinct real samples. Deduping on id only removes rows that are
    literal repeats of the same underlying record -- confirmed present in
    mal-api-2019 (879 rows across 290 duplicate ids, matching known source
    duplication comparable to EMBER2024's ~2x issue) and, to a much smaller
    degree, malbehavd-v1 (53 rows across 26 duplicate ids).
    """
    seen: set[str] = set()
    out = []
    n_before = len(records)
    for r in records:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        out.append(r)
    logger.info("Dedup by id: %d -> %d rows (%d removed)", n_before, len(out), n_before - len(out))
    return out


def build_canonical_records(cache_dir: Path) -> list[dict]:
    records = list(_iter_mal_api_2019(cache_dir))
    records.extend(_iter_malbehavd_v1(cache_dir))
    records.extend(_iter_carpenter(cache_dir))
    return _dedup_by_id(records)


def report(records: list[dict]) -> None:
    n = len(records)
    lengths = np.array([len(r["api_calls"]) for r in records])
    labels = np.array([r["label"] for r in records])
    sources = [r["source"] for r in records]

    logger.info("Total rows: %d", n)
    logger.info("Overall label balance: benign(0)=%d malicious(1)=%d", int((labels == 0).sum()), int((labels == 1).sum()))
    for src in sorted(set(sources)):
        mask = np.array([s == src for s in sources])
        logger.info(
            "  %s: n=%d benign=%d malicious=%d",
            src, int(mask.sum()), int(((labels == 0) & mask).sum()), int(((labels == 1) & mask).sum()),
        )

    logger.info(
        "api_calls length: min=%d median=%.1f p95=%.1f max=%d",
        int(lengths.min()), float(np.median(lengths)), float(np.percentile(lengths, 95)), int(lengths.max()),
    )
    for src in sorted(set(sources)):
        mask = np.array([s == src for s in sources])
        sl = lengths[mask]
        logger.info(
            "  %s length: min=%d median=%.1f p95=%.1f max=%d",
            src, int(sl.min()), float(np.median(sl)), float(np.percentile(sl, 95)), int(sl.max()),
        )

    n_empty = int((lengths == 0).sum())
    logger.info("Rows with empty api_calls: %d", n_empty)

    n_over_100 = int((lengths > 100).sum())
    logger.info(
        "Rows with api_calls longer than the tokenizer's 100-call cap: %d (%.1f%%)",
        n_over_100, 100 * n_over_100 / n,
    )


def save(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "id": [r["id"] for r in records],
        "source": [r["source"] for r in records],
        "family_raw": [r["family_raw"] for r in records],
        "label": [r["label"] for r in records],
        "api_calls": [r["api_calls"] for r in records],
    })
    pq.write_table(table, str(out_path))
    logger.info("Saved %d rows to %s", len(records), out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="If omitted, only prints the report -- nothing is written.")
    ap.add_argument("--cache-dir", default="data/raw_behavioral_cache")
    args = ap.parse_args()

    records = build_canonical_records(Path(args.cache_dir))
    report(records)

    if args.out:
        save(records, Path(args.out))
    else:
        logger.info("--out not given: report only, nothing written to disk.")


if __name__ == "__main__":
    main()
