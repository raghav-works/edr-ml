"""
Tests for features/authenticode_trust.py -- the standalone Authenticode
chain-verification check backing the known-file allowlist (addition A,
OPEN_ITEMS.md "Static false-positive severity cluster"). Confirms the
three-way outcome taxonomy (verified / not_signed / parse_error /
chain_untrusted) against this repo's own committed PE fixtures -- no
network, no external NSRL data, no real trusted-signed sample (this repo
does not have one committed; that gap is why the self-signed-fixture case
below is the one genuinely load-bearing regression test here).
"""
from __future__ import annotations

import os

from features.authenticode_trust import verify_trusted_chain

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "pe_samples")


def _read(name: str) -> bytes:
    with open(os.path.join(_FIXTURE_DIR, name), "rb") as fh:
        return fh.read()


def test_self_signed_fixture_does_not_verify():
    # sample_signed64.exe is deliberately self-signed (O=Test Only, CN=Cortex
    # Endpoint Test Fixture). It must NEVER pass real chain verification --
    # if it did, the allowlist would trust every signed binary this repo's
    # own fixture generator produces. This is the single most important
    # case in this file: a false "verified" here is exactly the failure this
    # module exists to prevent.
    result = verify_trusted_chain(_read("sample_signed64.exe"))
    assert result.trusted is False
    assert result.reason == "chain_untrusted"


def test_unsigned_pe_is_not_signed():
    result = verify_trusted_chain(_read("sample_cli64.exe"))
    assert result.trusted is False
    assert result.reason == "not_signed"


def test_truncated_signed_pe_reads_as_not_signed():
    # Cutting off a signed binary before signify can locate the signature
    # is indistinguishable, to signify, from the file never having been
    # signed -- confirmed against this exact fixture. "not_signed" is the
    # honest description of what was found, not a hidden parse bug.
    result = verify_trusted_chain(_read("sample_signed64.exe")[:2000])
    assert result.trusted is False
    assert result.reason == "not_signed"


def test_garbage_bytes_is_parse_error():
    result = verify_trusted_chain(b"not a pe file at all" * 20)
    assert result.trusted is False
    assert result.reason == "parse_error"


def test_empty_bytes_is_parse_error():
    result = verify_trusted_chain(b"")
    assert result.trusted is False
    assert result.reason == "parse_error"


def test_never_raises_on_arbitrary_junk():
    # Defense-in-depth: an allowlist check must never crash a scan. Feed it
    # something that is neither a valid PE nor obviously empty/garbage.
    result = verify_trusted_chain(os.urandom(4096))
    assert result.trusted is False
    assert result.reason in ("parse_error", "not_signed", "chain_untrusted")
