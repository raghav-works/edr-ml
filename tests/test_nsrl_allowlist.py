"""
Tests for features/nsrl_allowlist.py -- the memory-mapped SHA-256 lookup
backing one leg of the known-file allowlist (addition A, OPEN_ITEMS.md
"Static false-positive severity cluster"). Builds small synthetic
artifacts directly (no real NSRL download, no network) using the same
'V32' format data/download_nsrl.py::build_allowlist_artifact writes.
"""
from __future__ import annotations

import numpy as np
import pytest

from features.nsrl_allowlist import DIGEST_BYTES, NSRLAllowlist


def _write_artifact(tmp_path, digests: list[bytes]):
    path = tmp_path / "nsrl_sha256.bin"
    path.write_bytes(b"".join(sorted(digests)))
    return path


def test_contains_present_and_absent_hashes(tmp_path):
    present = [bytes([i]) + b"\x11" * 31 for i in range(10)]
    path = _write_artifact(tmp_path, present)
    allow = NSRLAllowlist.load(path)
    assert len(allow) == 10
    for d in present:
        assert allow.contains(d.hex())
    assert not allow.contains(("\xff" * 32).encode("latin1").hex())


def test_digest_ending_in_null_byte_round_trips_correctly(tmp_path):
    # Direct regression test for the bug found while designing this format:
    # numpy's 'S32' byte-string dtype silently strips trailing NUL (0x00)
    # bytes on comparison, which would make a real SHA-256 digest ending in
    # 0x00 either never match itself (false negative) or collide with an
    # unrelated digest sharing the same leading 31 bytes (false positive).
    # 'V32' (used here) must not have this problem.
    ends_in_null = b"\xaa" * 31 + b"\x00"
    other = b"\xbb" * 32
    path = _write_artifact(tmp_path, [ends_in_null, other])
    allow = NSRLAllowlist.load(path)
    assert allow.contains(ends_in_null.hex())
    assert allow.contains(other.hex())
    # A different digest that shares the same leading 31 bytes but a
    # non-null last byte must NOT be treated as equal to ends_in_null.
    almost_same = b"\xaa" * 31 + b"\x01"
    assert not allow.contains(almost_same.hex())


def test_load_returns_none_for_missing_path(tmp_path):
    assert NSRLAllowlist.load(tmp_path / "does_not_exist.bin") is None


def test_load_raises_on_corrupt_size(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"\x00" * (DIGEST_BYTES - 1))  # not a multiple of 32
    with pytest.raises(RuntimeError):
        NSRLAllowlist.load(path)


def test_load_raises_on_empty_file(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    with pytest.raises(RuntimeError):
        NSRLAllowlist.load(path)


def test_contains_handles_malformed_input_without_raising(tmp_path):
    path = _write_artifact(tmp_path, [b"\x11" * 32])
    allow = NSRLAllowlist.load(path)
    assert allow.contains("not-hex-at-all") is False
    assert allow.contains("abcd") is False  # valid hex, wrong length
    assert allow.contains("") is False


def test_constructor_rejects_wrong_dtype():
    with pytest.raises(ValueError):
        NSRLAllowlist(np.zeros(4, dtype=np.uint8))
