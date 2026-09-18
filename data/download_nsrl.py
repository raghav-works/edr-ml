"""
Download NIST's NSRL RDS "Modern" minimal hash-set database and build a
compact local SHA-256 allowlist artifact.

Backs one leg of the known-file allowlist (OPEN_ITEMS.md, "Static
false-positive severity cluster", addition A): inference/pipeline.py's
step 3b checks a scanned file's SHA-256 against the artifact this script
produces, and short-circuits static's ML judgment to ALLOW on a match --
see features/nsrl_allowlist.py for the loader/query side.

Why the "Modern" category, not "Legacy" or the full multi-category RDS:
NIST publishes RDS hash sets by category (Modern: software dated 2015 to
present; Legacy: 2014 and earlier; several others), and Modern is the only
category NIST currently publishes a trimmed "minimal" database for --
sized for exactly this kind of local-index use case, and the one whose
software population (current, actively-used software) matches this
project's stated concern: false positives landing on files the system
needs to keep running, e.g. core Windows system binaries.

Why RDSv3 / SQLite, not the legacy RDS 2.XX flat-text format: NIST has
completed the transition away from the RDS 2.XX text-file format and only
publishes RDSv3 (SQLite) going forward. RDSv3's minimal-database schema
exposes a FILE view (backed by an internal METADATA table) with a `sha256`
column already normalized to uppercase hex -- confirmed against NIST's own
RDSv3 schema documentation (RDSv3.pdf, distributed alongside the data on
NIST's S3 bucket). This script queries the FILE view, not METADATA
directly, so an internal schema change that keeps the public view stable
doesn't break this script.

Source URL: NIST maintains a version-independent alias,
rds.nsrl.nist.gov/RDS/current/rds_modernm.zip, that always points at the
current release's Modern-minimal RDSv3 database -- used here instead of a
version-numbered filename (e.g. RDS2026.09.1modernminimal.zip) specifically
so this script does not go stale at the next NSRL release.

This is a manual/offline step, run by a maintainer when the local artifact
needs building or refreshing -- not invoked at scan time or in CI. The
downloaded zip is large (NIST does not publish an exact minimal-set size
per release; treat it as a multi-hundred-MB to low-single-digit-GB
download) and the extracted SQLite database and output artifact together
need comparable free disk space; both are written under a TemporaryDirectory
and cleaned up automatically.

Output artifact format: N raw, sorted, concatenated 32-byte SHA-256
digests (N * 32 bytes total, no header) -- see build_allowlist_artifact()'s
docstring for why numpy's void ('V32') dtype is used for sorting rather
than a byte-string dtype.

Usage:
    python -m data.download_nsrl --out data/models/nsrl_sha256.bin

    # Re-build from an already-downloaded/extracted .sqlite file (skips the
    # network fetch -- also how tests exercise the extraction/build logic):
    python -m data.download_nsrl --out data/models/nsrl_sha256.bin \\
        --sqlite-path /path/to/already_extracted.sqlite
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np
import requests

logger = logging.getLogger("cortex.data.nsrl")

# Version-independent "current release" alias -- see module docstring.
NSRL_MODERN_MINIMAL_URL = "https://s3.amazonaws.com/rds.nsrl.nist.gov/RDS/current/rds_modernm.zip"
DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB
QUERY_BATCH_SIZE = 200_000
DIGEST_BYTES = 32  # SHA-256


def download_zip(url: str, dest: Path) -> None:
    logger.info("Downloading NSRL Modern-minimal hash set from %s", url)
    written = 0
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                fh.write(chunk)
                written += len(chunk)
        if total and written != total:
            raise RuntimeError(
                f"downloaded {written} bytes but the server advertised "
                f"Content-Length {total} -- treat this as a truncated/corrupt "
                "download, not a partial success"
            )
    logger.info("Downloaded %d bytes to %s", written, dest)


def find_sqlite_file(extract_dir: Path) -> Path:
    candidates = sorted(extract_dir.rglob("*.sqlite")) + sorted(extract_dir.rglob("*.db"))
    if not candidates:
        raise RuntimeError(f"no .sqlite/.db file found after extracting the NSRL zip into {extract_dir}")
    if len(candidates) > 1:
        raise RuntimeError(f"expected exactly one sqlite database, found {len(candidates)}: {candidates}")
    return candidates[0]


def iter_sha256_hex(sqlite_path: Path) -> Iterator[str]:
    """Yield every non-null sha256 hex string from the RDSv3 minimal
    database's FILE view, batched (QUERY_BATCH_SIZE) rather than loaded in
    one fetchall() -- the real Modern-minimal database has tens of millions
    of rows."""
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.cursor()
        try:
            cur.execute("SELECT sha256 FROM FILE")
        except sqlite3.OperationalError as exc:
            raise RuntimeError(
                f"NSRL database at {sqlite_path} does not expose the expected FILE "
                f"view with a sha256 column -- schema may have changed upstream: {exc}"
            ) from exc
        while True:
            rows = cur.fetchmany(QUERY_BATCH_SIZE)
            if not rows:
                return
            for (hex_str,) in rows:
                if hex_str:
                    yield hex_str
    finally:
        conn.close()


def build_allowlist_artifact(sqlite_path: Path, out_path: Path) -> int:
    """
    Convert every sha256 hex string in the RDSv3 minimal database into a
    raw 32-byte digest, sort them, and write the result as one flat binary
    file of concatenated digests: N * 32 raw bytes, sorted ascending, no
    header. features/nsrl_allowlist.py::NSRLAllowlist.load() memory-maps
    this file with numpy dtype='V32' and queries it with np.searchsorted.

    Uses numpy's void ('V32') dtype deliberately, NOT the more obvious
    fixed-width byte-string dtype ('S32'). Confirmed by direct testing
    during this feature's design that 'S32' silently strips trailing NUL
    (0x00) bytes on comparison -- a real risk here, not a hypothetical one:
    a SHA-256 digest ending in 0x00 is a ~1-in-256 event, so among tens of
    millions of hashes it is expected to happen many times, not a corner
    case to hand-wave past. Under 'S32' this produces both false negatives
    (a genuinely allowlisted hash never matching a lookup for itself) and
    false positives (two different digests that both happen to end in
    0x00, and share their other 31 bytes, comparing equal). 'V32' is opaque
    fixed-width binary data to numpy and was verified to preserve every
    byte exactly under sort / searchsorted / equality.

    Returns the number of digests written; raises RuntimeError if that
    number is zero (refuses to silently produce an empty allowlist that
    would just never match anything).
    """
    n = 0
    digests = bytearray()
    for hex_str in iter_sha256_hex(sqlite_path):
        try:
            d = bytes.fromhex(hex_str)
        except ValueError:
            logger.warning("skipping malformed sha256 value: %r", hex_str)
            continue
        if len(d) != DIGEST_BYTES:
            logger.warning("skipping sha256 value with unexpected length %d: %r", len(d), hex_str)
            continue
        digests.extend(d)
        n += 1
        if n % 1_000_000 == 0:
            logger.info("...%d digests read so far", n)

    if n == 0:
        raise RuntimeError(
            f"NSRL database at {sqlite_path} produced zero usable sha256 rows -- "
            "refusing to write an empty allowlist artifact"
        )

    arr = np.frombuffer(bytes(digests), dtype=f"V{DIGEST_BYTES}")
    arr = np.sort(arr)
    arr.tofile(out_path)
    logger.info("Wrote %d sorted SHA-256 digests (%d bytes) to %s", n, arr.nbytes, out_path)
    return n


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output path for the sorted digest artifact, e.g. data/models/nsrl_sha256.bin")
    ap.add_argument("--url", default=NSRL_MODERN_MINIMAL_URL, help="override the NSRL zip URL (default: NIST's version-independent 'current' Modern-minimal alias)")
    ap.add_argument("--sqlite-path", default=None, help="skip the download and build directly from an already-extracted .sqlite file")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.sqlite_path:
        build_allowlist_artifact(Path(args.sqlite_path), out_path)
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        zip_path = tmp_dir / "nsrl_modern_minimal.zip"
        download_zip(args.url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_dir)
        sqlite_path = find_sqlite_file(tmp_dir)
        build_allowlist_artifact(sqlite_path, out_path)


if __name__ == "__main__":
    main()
