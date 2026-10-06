"""
docs/CODE_REVIEW.md F17 -- truncated / cut-off PE files.

A PE whose section raw data, headers, or certificate table extend past the
end of the file must resolve to NEEDS_REVIEW (static ERROR, reason
static_pe_truncated + one detail code per rule) BEFORE the static model
scores it. Truncated copies are built from the committed fixtures, written
only to pytest's tmp_path (removed automatically) and never executed.
"""
from __future__ import annotations

import glob
import os
import struct

import numpy as np
import pefile
import pytest

from features.pe_features import PEFeatureExtractor, truncation_findings
from inference.pipeline import CortexPipeline
from inference.policy_engine import FinalDecision, StaticVerdict

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "pe_samples")
_FIXTURES = sorted(glob.glob(os.path.join(_FIXTURE_DIR, "*.exe")))
_SIGNED = os.path.join(_FIXTURE_DIR, "sample_signed64.exe")  # also the self-test reference PE
_UNSIGNED = os.path.join(_FIXTURE_DIR, "sample_cli64.exe")
_IDS = [os.path.basename(p) for p in _FIXTURES]


class _MustNotScoreModel:
    def predict_proba(self, *_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("static model scored a truncated PE")


class _BenignStaticModel:
    def predict_proba(self, X, *_args, **_kwargs):
        return np.full(len(X), 0.01, dtype=np.float64)


def _read(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _raw_ends(bytez: bytes) -> list[int]:
    pe = pefile.PE(data=bytez, fast_load=True)
    ends = [s.PointerToRawData + s.SizeOfRawData for s in pe.sections if s.SizeOfRawData > 0]
    pe.close()
    return ends


def _has_cert_table(bytez: bytes) -> bool:
    pe = pefile.PE(data=bytez, fast_load=True)
    has = pe.OPTIONAL_HEADER.DATA_DIRECTORY[4].Size > 0
    pe.close()
    return has


def _scan(tmp_path, bytez: bytes, model) -> "object":
    path = tmp_path / "probe.bin"
    path.write_bytes(bytez)
    return CortexPipeline(static_model=model, self_test=False).scan(str(path))


# ------------------------------------------------------------ full fixtures
@pytest.mark.parametrize("path", _FIXTURES, ids=_IDS)
def test_full_fixtures_have_no_truncation_findings(path):
    bytez = _read(path)
    pe = pefile.PE(data=bytez, fast_load=False)
    try:
        assert truncation_findings(pe, len(bytez)) == []
    finally:
        pe.close()


@pytest.mark.parametrize("path", _FIXTURES, ids=_IDS)
def test_full_fixtures_are_scored_normally(tmp_path, path):
    result = _scan(tmp_path, _read(path), _BenignStaticModel())
    assert result.static_verdict == StaticVerdict.ALLOW
    assert result.final_decision == FinalDecision.ALLOW
    assert not any(c.startswith(("static_pe_truncated", "pe_truncated:")) for c in result.reason_codes)


def test_self_test_reference_pe_is_unaffected():
    assert PEFeatureExtractor().self_test() == []
    CortexPipeline(static_model=_BenignStaticModel(), self_test=True)  # must not raise


# ------------------------------------------------------- truncated copies
@pytest.mark.parametrize("path", _FIXTURES, ids=_IDS)
@pytest.mark.parametrize("cut", ["1KB", "4KB", "half"])
def test_truncated_copy_is_needs_review_before_scoring(tmp_path, path, cut):
    full = _read(path)
    n = {"1KB": 1024, "4KB": 4096, "half": len(full) // 2}[cut]
    truncated = full[:n]

    result = _scan(tmp_path, truncated, _MustNotScoreModel())

    assert result.static_verdict == StaticVerdict.ERROR
    assert result.static_score is None
    assert result.final_decision == FinalDecision.NEEDS_REVIEW
    assert "static_pe_truncated" in result.reason_codes
    expected_overrun = max(_raw_ends(full)) - n
    assert f"pe_truncated:section_raw_beyond_eof:{expected_overrun}" in result.reason_codes
    assert ("pe_truncated:certificate_table_beyond_eof" in result.reason_codes) == _has_cert_table(full)
    assert "pe_truncated:headers_beyond_eof" not in result.reason_codes  # 0x400 headers survive a 1 KB cut


def test_last_section_overrun_by_one_byte_is_flagged(tmp_path):
    full = _read(_UNSIGNED)
    assert not _has_cert_table(full)
    cut = full[:max(_raw_ends(full)) - 1]

    result = _scan(tmp_path, cut, _MustNotScoreModel())

    assert result.final_decision == FinalDecision.NEEDS_REVIEW
    detail = [c for c in result.reason_codes if c.startswith("pe_truncated:")]
    assert detail == ["pe_truncated:section_raw_beyond_eof:1"]


def test_certificate_table_cut_alone_is_flagged(tmp_path):
    """Signed fixture cut 1 byte short: the certificate table (appended after
    the sections) is incomplete, section data is intact."""
    full = _read(_SIGNED)
    assert max(_raw_ends(full)) < len(full) - 1
    result = _scan(tmp_path, full[:-1], _MustNotScoreModel())
    detail = [c for c in result.reason_codes if c.startswith("pe_truncated:")]
    assert detail == ["pe_truncated:certificate_table_beyond_eof"]
    assert result.final_decision == FinalDecision.NEEDS_REVIEW


def test_size_of_headers_beyond_eof_is_flagged(tmp_path):
    full = bytearray(_read(_UNSIGNED))
    pe = pefile.PE(data=bytes(full), fast_load=True)
    off = pe.OPTIONAL_HEADER.get_field_absolute_offset("SizeOfHeaders")
    pe.close()
    struct.pack_into("<I", full, off, len(full) + 1)

    result = _scan(tmp_path, bytes(full), _MustNotScoreModel())

    assert "pe_truncated:headers_beyond_eof" in result.reason_codes
    assert result.final_decision == FinalDecision.NEEDS_REVIEW


def test_reason_codes_are_machine_parseable(tmp_path):
    result = _scan(tmp_path, _read(_SIGNED)[:4096], _MustNotScoreModel())
    for code in result.reason_codes:
        if code.startswith("pe_truncated:"):
            parts = code.split(":")
            assert parts[1] in {"section_raw_beyond_eof", "headers_beyond_eof",
                                "certificate_table_beyond_eof"}
            if parts[1] == "section_raw_beyond_eof":
                assert len(parts) == 3 and int(parts[2]) > 0
            else:
                assert len(parts) == 2


# ------------------------------------------------------------- parse once
@pytest.mark.parametrize("which", ["full", "truncated"])
def test_pipeline_parses_each_file_once(tmp_path, monkeypatch, which):
    calls = []

    # A subclass, not a plain function: pefile resolves class attributes
    # through the module-level name PE, so a function stand-in makes every
    # parse fail (and would make this count trivially 1).
    class CountingPE(pefile.PE):
        def __init__(self, *args, **kwargs):
            calls.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(pefile, "PE", CountingPE)
    full = _read(_SIGNED)
    bytez = full if which == "full" else full[:4096]
    model = _BenignStaticModel() if which == "full" else _MustNotScoreModel()

    result = _scan(tmp_path, bytez, model)

    assert len(calls) == 1
    # the single parse must have succeeded and driven the real outcome
    assert "invalid_or_non_pe_file" not in result.reason_codes
    if which == "full":
        assert result.static_verdict == StaticVerdict.ALLOW
        assert result.static_score is not None
    else:
        assert "static_pe_truncated" in result.reason_codes
