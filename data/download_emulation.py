"""
Cortex-Emulation: load and validate the Quo Vadis Speakeasy-emulation
dataset (Trizna et al., 2022/2024; `dtrizna/quovadis-speakeasy` on
HuggingFace, Apache-2.0 -- confirmed via both the repo's `license` tag and
the actual card text, no additional restriction beyond the bare tag).

This module does NOT stream via the `datasets` library. The dataset card
itself warns that loading through `datasets` flattens each entry point
into its own row with an auto-inferred schema across ~93K structurally
inconsistent files (some carry `registry_access`/`file_access`/
`dropped_files`/`process_events`, most don't) -- risking silent schema
coercion we can't see. Instead: `huggingface_hub.snapshot_download()`
mirrors the raw JSON files locally (one-time manual step, same spirit as
data/download_memory.py's manual CSV download), and this module parses
them directly:

    python -c "from huggingface_hub import snapshot_download; \\
        snapshot_download(repo_id='dtrizna/quovadis-speakeasy', \\
        repo_type='dataset', local_dir='data/raw/emulation', max_workers=16)"

Real structure, verified against actual downloaded files (not the card,
which is wrong about this specific point): the card describes each
per-sample file as `{"sha256": ..., "entry_points": [...]}`. The real files
are a **bare JSON list** -- `[{...}, {...}]` -- with no wrapper object and
**no `sha256` field in the file contents at all**; the sample's identifier
is recoverable only from the filename (`<sha256>.json` or
`<sha256>.dat.json` for every category except `report_windows_syswow64`,
which uses the literal system-binary filename instead -- see
`_sample_id_from_filename`'s docstring). A loader trusting the card's
example structure would break immediately.

Each entry-point dict has more fields than the card's truncated example
shows. Always present: `ep_type` (`module_entry` or `thread`), `start_addr`,
`ep_args`, `apihash`, `apis` (list of `{pc, api_name, args, ret_val}`, not
just bare names), `ret_val`, `error`, `dynamic_code_segments`. Sparse/
optional (present only when the emulator observed that activity):
`handled_exceptions`, `registry_access`, `file_access`, `dropped_files`,
`network_events`, `process_events`.

v1 scope (see docs/TECHNICAL_NOTES.md's Cortex-Emulation section for the full reasoning):
only `api_name` tokens from `apis` are extracted for modeling, matching
Cortex-Behavioral's proven 1D-CNN + self-attention architecture. Every
other field -- the args/ret_val/pc detail inside `apis`, and the sparse
registry/file/dropped-file/process event categories -- is a real, known
signal that is NOT modeled in v1, but is not silently discarded either:
the complete original entry-point dict is preserved verbatim (as a JSON
string) in the `raw_entry_json` column of the processed output, so a
future v2 can encode it without re-parsing the raw files.

Labels: `report_clean` and `report_windows_syswow64` are benign; the other
7 folders (`report_backdoor`, `report_coinminer`, `report_dropper`,
`report_keylogger`, `report_ransomware`, `report_rat`, `report_trojan`)
are malware families -- a family taxonomy, not inherently binary. The raw
family name is kept alongside the derived binary target, same pattern as
Cortex-Memory's Category and Cortex-Network's label_raw.

Usage:
    python -m data.download_emulation --in-dir data/raw/emulation \\
        --out data/processed/emulation_dataset.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("cortex.data.emulation")

TRAIN_SUBDIR = "windows_emulation_trainset"
TEST_SUBDIR = "windows_emulation_testset"

# Verified against the real repo file listing via a full expected-vs-local
# diff (not just a count coincidence): 75,298 train + 17,402 test = 92,700
# .json files. Close to, but not exactly, the card's own numbers: its prose
# says 76,126 / 17,407 (clearly stale/rounded), and even its table --
# 75,301 / 17,402 -- is 3 off for train from what the repository actually
# contains. Use these verified numbers, not the card's, for anything that
# depends on an exact count.
EXPECTED_TRAIN_FILES = 75298
EXPECTED_TEST_FILES = 17402

BENIGN_FAMILIES = frozenset({"clean", "windows_syswow64"})

_SHA256_PREFIX_RE = re.compile(r"^([0-9a-f]{64})")

# Every entry-point dict must have at least these keys -- verified present
# in 100% of a 48-file structural sample across all 9 categories, train and
# test. The sparse event-category keys (registry_access, etc.) are NOT
# required -- they're genuinely optional per the real data, not missing due
# to a parsing bug, so requiring them would make load_and_validate_file
# reject the majority of real files.
REQUIRED_ENTRY_KEYS = frozenset({
    "ep_type", "start_addr", "ep_args", "apihash", "apis", "ret_val",
    "error", "dynamic_code_segments",
})


def _split_from_path(path: Path) -> str:
    parts = path.parts
    if TRAIN_SUBDIR in parts:
        return "train"
    if TEST_SUBDIR in parts:
        return "test"
    raise ValueError(f"{path}: not under {TRAIN_SUBDIR!r} or {TEST_SUBDIR!r}")


def _family_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("report_"):
            return part[len("report_"):]
    raise ValueError(f"{path}: no 'report_*' directory component found")


def _sample_id_from_filename(path: Path) -> str:
    """"<sha256>.json" or "<sha256>.dat.json" -> sha256, for every category
    except report_windows_syswow64. That one category -- confirmed across
    all 294 of its files (235 train + 59 test), and only that category --
    is named by the literal Windows system binary name instead
    (e.g. "AppVDllSurrogate.json"), not a sha256: these are known-legitimate
    system files, not anonymous malware samples needing hash-based
    identification, so the dataset's own creators evidently didn't bother
    hashing them. Falls back to the filename stem (suffixes stripped) in
    that case rather than raising -- a real, verified schema property, not
    a malformed name."""
    m = _SHA256_PREFIX_RE.match(path.name)
    if m:
        return m.group(1)
    stem = path.name
    for suffix in (".dat.json", ".json"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    raise ValueError(f"{path}: filename has neither a 64-hex-char sha256 prefix nor a recognized .json suffix")


def load_and_validate_file(json_path: Path) -> list[dict]:
    """Load one raw Quo Vadis emulation report and validate its structure:
    a bare JSON list of entry-point dicts, each with at least
    REQUIRED_ENTRY_KEYS. Raises ValueError on any mismatch -- never
    silently proceeds on a file that doesn't match what this module was
    built against."""
    with open(json_path) as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{json_path}: expected a top-level JSON list, got {type(data).__name__}")
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"{json_path}[{i}]: expected a dict entry point, got {type(entry).__name__}")
        missing = REQUIRED_ENTRY_KEYS - entry.keys()
        if missing:
            raise ValueError(f"{json_path}[{i}]: missing required keys {missing}")
    return data


def extract_records(json_path: Path) -> list[dict]:
    """One record per entry point in the file. `api_names` is the v1
    modeling signal (bare API-name tokens only); `raw_entry_json` preserves
    the complete original entry-point dict (apis with pc/args/ret_val, plus
    any sparse event categories present) for the deferred v2 -- see the
    module docstring."""
    split = _split_from_path(json_path)
    family = _family_from_path(json_path)
    label = 0 if family in BENIGN_FAMILIES else 1
    sample_id = _sample_id_from_filename(json_path)

    entries = load_and_validate_file(json_path)
    records = []
    for ep_index, entry in enumerate(entries):
        api_names = [a["api_name"] for a in entry["apis"]]
        records.append({
            "sample_id": sample_id,
            "ep_index": ep_index,
            "ep_type": entry["ep_type"],
            "family": family,
            "label": label,
            "split": split,
            "apihash": entry["apihash"],
            "api_names": api_names,
            "n_apis": len(api_names),
            # "error" is always present (in REQUIRED_ENTRY_KEYS) but is an
            # EMPTY dict {} when nothing went wrong, not null/None -- `is
            # not None` is wrong here, since {} is not None evaluates True,
            # which silently flagged every single entry as errored (100%
            # in an early run of this loader against the real 92,700-file
            # dataset, an obviously-wrong number that caught the bug).
            "has_error": bool(entry.get("error")),
            "raw_entry_json": json.dumps(entry),
        })
    return records


def _iter_json_files(raw_dir: Path) -> Iterator[Path]:
    for subdir in (TRAIN_SUBDIR, TEST_SUBDIR):
        base = raw_dir / subdir
        if not base.exists():
            raise ValueError(f"{base} does not exist -- did the snapshot_download complete?")
        yield from sorted(base.rglob("*.json"))


def build_canonical_dataset(raw_dir: str) -> pd.DataFrame:
    """Parses every file under windows_emulation_trainset/ and
    windows_emulation_testset/ (not the root-level example-ransomware.json
    or speakeasy-config.json, which aren't part of either split). One file
    at a time -- each is small (a few KB to ~500KB), so unlike
    data/download_network.py's 4 GiB single-file problem, no streaming/
    capping is needed here; the full ~93K-file, ~5.5 GiB dataset is safe to
    materialize as one DataFrame."""
    raw_path = Path(raw_dir)
    records: list[dict] = []
    n_files = 0
    for json_path in _iter_json_files(raw_path):
        records.extend(extract_records(json_path))
        n_files += 1
        if n_files % 10000 == 0:
            logger.info("Processed %d files, %d entry-point records so far...", n_files, len(records))
    logger.info("Processed %d files total, %d entry-point records.", n_files, len(records))
    return pd.DataFrame.from_records(records)


def report(df: pd.DataFrame) -> None:
    n = len(df)
    logger.info("Total entry-point records: %d (from %d distinct sample_id files)", n, df["sample_id"].nunique())

    logger.info("=== Rows per (split, family) ===")
    for (split, family), count in df.groupby(["split", "family"]).size().items():
        logger.info("  split=%-6s family=%-16s n=%d", split, family, count)

    logger.info("=== ep_type distribution ===")
    logger.info("  %s", df["ep_type"].value_counts().to_dict())

    n_apis = df["n_apis"].to_numpy()
    logger.info(
        "=== api_names length distribution (ALL entry points, every ep_type pooled) ===\n"
        "  min=%d max=%d mean=%.1f median=%.1f p90=%.1f p95=%.1f p99=%.1f",
        n_apis.min(), n_apis.max(), n_apis.mean(), np.median(n_apis),
        np.percentile(n_apis, 90), np.percentile(n_apis, 95), np.percentile(n_apis, 99),
    )
    # Broken out by ep_type: threads/TLS-callbacks are typically short
    # support code, not the sample's main behavior, and pooling them with
    # module_entry rows (the primary execution path) skews the "all
    # entry points" view toward near-zero -- this breakdown is what a
    # sequence-length decision should actually be grounded in.
    logger.info("=== api_names length distribution, by ep_type ===")
    for ep_type, sub in df.groupby("ep_type"):
        vals = sub["n_apis"].to_numpy()
        logger.info(
            "  ep_type=%-30s n=%6d min=%d max=%d mean=%.1f median=%.1f p90=%.1f p95=%.1f p99=%.1f",
            ep_type, len(vals), vals.min(), vals.max(), vals.mean(), np.median(vals),
            np.percentile(vals, 90), np.percentile(vals, 95), np.percentile(vals, 99),
        )

    n_error = int(df["has_error"].sum())
    logger.info("Entries with error set: %d / %d (%.1f%%)", n_error, n, 100 * n_error / n)
    with_err = df.loc[df["has_error"], "n_apis"]
    without_err = df.loc[~df["has_error"], "n_apis"]
    logger.info(
        "  n_apis WITH error:    mean=%.1f median=%.1f (n=%d)",
        with_err.mean(), with_err.median(), len(with_err),
    )
    logger.info(
        "  n_apis WITHOUT error: mean=%.1f median=%.1f (n=%d)",
        without_err.mean(), without_err.median(), len(without_err),
    )

    n_files_train = df.loc[df["split"] == "train", "sample_id"].nunique()
    n_files_test = df.loc[df["split"] == "test", "sample_id"].nunique()
    logger.info(
        "Distinct sample_id files: train=%d (expected %d) test=%d (expected %d)",
        n_files_train, EXPECTED_TRAIN_FILES, n_files_test, EXPECTED_TEST_FILES,
    )


def save(df: pd.DataFrame, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # api_names is a list column -- parquet handles list<string> natively,
    # no special serialization needed (unlike raw_entry_json, which is
    # already a JSON string precisely to avoid parquet's nested-struct
    # schema-inference headaches on genuinely heterogeneous dicts).
    df.to_parquet(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/raw/emulation")
    ap.add_argument("--out", default=None, help="If omitted, only prints the report -- nothing is written.")
    args = ap.parse_args()

    df = build_canonical_dataset(args.in_dir)
    report(df)

    if args.out:
        save(df, args.out)
    else:
        logger.info("--out not given: report only, nothing written to disk.")


if __name__ == "__main__":
    main()
