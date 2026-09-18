"""
Local NSRL SHA-256 allowlist -- a memory-mapped, binary-searchable index
built by data/download_nsrl.py from NIST's NSRL RDS "Modern" minimal hash
set.

Backs one leg of the known-file allowlist (OPEN_ITEMS.md, "Static
false-positive severity cluster", addition A): a scanned file whose
SHA-256 is present in this list is treated as pre-vetted and short-circuits
static's ML judgment in inference/pipeline.py's step 3b. The other leg is
features/authenticode_trust.py's real Authenticode chain verification --
the two are independent, either is sufficient.

Deliberately NOT a live nsrlsvr-style lookup daemon queried per scan: this
repo has no live service dependencies anywhere (see ARCHITECTURE.md's
explicit "no live traffic capture, ... model-serving API, queue, database,
or web service" statement), and a per-scan network call to a lookup
service would both be a new failure mode and break that stated property
for no clear benefit over a local, memory-mapped index.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("cortex.features.nsrl_allowlist")

DIGEST_BYTES = 32  # SHA-256
_DTYPE = f"V{DIGEST_BYTES}"


class NSRLAllowlist:
    """Wraps a sorted array of raw 32-byte SHA-256 digests (the exact
    format data/download_nsrl.py::build_allowlist_artifact writes) and
    answers point-membership queries via binary search (np.searchsorted) --
    O(log N). Construct via load(), not directly."""

    def __init__(self, digests: np.ndarray):
        if digests.dtype != np.dtype(_DTYPE):
            raise ValueError(f"NSRLAllowlist expects dtype {_DTYPE}, got {digests.dtype}")
        # Trusts the artifact is sorted ascending, the same way models/*.py
        # trusts a .lgbm file's internal structure without re-validating it
        # byte-by-byte at load time: this class only ever loads artifacts
        # produced by build_allowlist_artifact() in this same repo, which
        # sorts unconditionally before writing.
        self._digests = digests

    @classmethod
    def load(cls, path: Path) -> Optional["NSRLAllowlist"]:
        """Returns None (does not raise) when `path` does not exist -- an
        unconfigured allowlist is a deployment choice, not an error.
        inference/pipeline.py treats a None NSRLAllowlist the same way it
        already treats an unconfigured memory/network model: that leg of
        the allowlist check is simply skipped, falling through to
        features/authenticode_trust.py's chain check and then, if that
        also misses, to static's normal ML judgment.

        Raises RuntimeError if `path` exists but is not a valid artifact
        (size not a multiple of DIGEST_BYTES) -- a corrupt/truncated
        artifact must fail loudly at startup, not silently match nothing or
        crash mid-scan.
        """
        path = Path(path)
        if not path.exists():
            logger.info("NSRL allowlist artifact not found at %s -- NSRL allowlist check disabled", path)
            return None
        size = path.stat().st_size
        if size == 0 or size % DIGEST_BYTES != 0:
            raise RuntimeError(
                f"NSRL allowlist artifact {path} has size {size} bytes, not a "
                f"positive multiple of {DIGEST_BYTES} -- refusing to load a "
                "corrupt or truncated artifact"
            )
        digests = np.memmap(path, dtype=_DTYPE, mode="r")
        logger.info("Loaded NSRL allowlist: %d digests from %s", len(digests), path)
        return cls(digests)

    def __len__(self) -> int:
        return len(self._digests)

    def contains(self, sha256_hex: str) -> bool:
        """False on any malformed input (wrong length, non-hex) rather than
        raising -- an allowlist lookup must never be the reason a scan
        fails; the caller's normal ML path is always the safe fallback."""
        try:
            target = bytes.fromhex(sha256_hex)
        except ValueError:
            return False
        if len(target) != DIGEST_BYTES:
            return False
        target_v = np.frombuffer(target, dtype=_DTYPE)[0]
        idx = np.searchsorted(self._digests, target_v)
        return idx < len(self._digests) and bytes(self._digests[idx]) == target
