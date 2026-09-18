"""
Tests for data/download_nsrl.py's extraction/build logic -- NOT the
network download itself (no test here touches NIST's servers). Builds a
tiny synthetic SQLite database mimicking the RDSv3 minimal database's FILE
view schema (a `sha256` text column) and exercises iter_sha256_hex() /
build_allowlist_artifact() / find_sqlite_file() against it.
"""
from __future__ import annotations

import hashlib
import sqlite3

import pytest

from data.download_nsrl import (
    build_allowlist_artifact,
    find_sqlite_file,
    iter_sha256_hex,
)
from features.nsrl_allowlist import NSRLAllowlist


def _make_sqlite(tmp_path, sha256_values):
    db_path = tmp_path / "nsrl_test.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE FILE (sha256 TEXT)")
    conn.executemany("INSERT INTO FILE (sha256) VALUES (?)", [(v,) for v in sha256_values])
    conn.commit()
    conn.close()
    return db_path


def _hexdigest(i: int) -> str:
    return hashlib.sha256(f"nsrl-fixture-{i}".encode()).hexdigest()


def test_iter_sha256_hex_yields_every_non_null_row(tmp_path):
    values = [_hexdigest(i) for i in range(20)] + [None]
    db_path = _make_sqlite(tmp_path, values)
    got = list(iter_sha256_hex(db_path))
    assert sorted(got) == sorted(values[:-1])  # the None row is skipped


def test_iter_sha256_hex_missing_file_view_raises(tmp_path):
    db_path = tmp_path / "no_file_view.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE something_else (x INTEGER)")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="does not expose the expected FILE view"):
        list(iter_sha256_hex(db_path))


def test_build_allowlist_artifact_round_trip(tmp_path):
    values = [_hexdigest(i) for i in range(500)]
    db_path = _make_sqlite(tmp_path, values)
    out_path = tmp_path / "allowlist.bin"

    n = build_allowlist_artifact(db_path, out_path)

    assert n == 500
    allow = NSRLAllowlist.load(out_path)
    assert len(allow) == 500
    for v in values:
        assert allow.contains(v)
    assert not allow.contains(_hexdigest(999999))


def test_build_allowlist_artifact_skips_malformed_hex(tmp_path, caplog):
    values = [_hexdigest(0), "not-valid-hex", "abcd"]  # last two: bad / wrong length
    db_path = _make_sqlite(tmp_path, values)
    out_path = tmp_path / "allowlist.bin"

    n = build_allowlist_artifact(db_path, out_path)

    assert n == 1
    allow = NSRLAllowlist.load(out_path)
    assert allow.contains(_hexdigest(0))


def test_build_allowlist_artifact_all_malformed_raises(tmp_path):
    db_path = _make_sqlite(tmp_path, ["not-valid-hex", "abcd"])
    out_path = tmp_path / "allowlist.bin"
    with pytest.raises(RuntimeError, match="zero usable sha256 rows"):
        build_allowlist_artifact(db_path, out_path)


def test_find_sqlite_file_requires_exactly_one(tmp_path):
    with pytest.raises(RuntimeError, match="no .sqlite/.db file found"):
        find_sqlite_file(tmp_path)

    (tmp_path / "a.sqlite").write_bytes(b"")
    assert find_sqlite_file(tmp_path) == tmp_path / "a.sqlite"

    (tmp_path / "b.sqlite").write_bytes(b"")
    with pytest.raises(RuntimeError, match="expected exactly one sqlite database"):
        find_sqlite_file(tmp_path)
