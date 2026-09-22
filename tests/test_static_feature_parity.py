"""
Feature-parity harness for features/pe_features.py -- checks 1-5 of the
item-7 scope (see OPEN_ITEMS.md). Runs entirely on committed synthetic PE
fixtures: no training data, no network, no model.

  1. vector contract     -- 2568-dim / float32 / finite; per-group slice
                            lengths == declared dims; histogram normalised;
                            general.size == real file size
  2. determinism         -- same bytes -> byte-identical vector, fresh
                            extractor instance each time
  3. signed-binary auth  -- the 8-dim authenticode group is populated on a
                            known-good signed PE (regression guard for the
                            silent _SIGNIFY_AVAILABLE=False all-zeros bug)
  4. ExportsInfo slot    -- slot 0 is the export COUNT, not the constant
                            hasher width (regression guard for the len(h) bug)
  5. adapter passthrough -- features/ember2024_adapter.record_to_vector on a
                            group-dict produces the exact same vector as the
                            live process_raw_features path

Real EMBER2024 record schema vs raw_features() output was checked, but not
here: a one-off investigation (2026-09-22, 140 real records pulled from
joyce8/EMBER2024, see OPEN_ITEMS.md's "Structural" section for the method
and result) confirmed the adapter's key/shape assumptions hold against real
HF records, with zero mismatches. It isn't an automated pytest in this file
because that would require a live HuggingFace pull inside the test suite
(network-dependent, slow, and a moving target if the upstream dataset
changes) for a check whose answer isn't expected to change on its own --
see OPEN_ITEMS.md for why a small hand-copied fixture wasn't added either.

NOT covered here (tracked in OPEN_ITEMS.md):
  - live-vs-thrember skew quantification (thrember needs pre-0.9 signify;
    this repo pins signify>=0.9 -- a separate venv is required)
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from features.ember2024_adapter import _GROUP_FIELDS, record_to_vector
from features.pe_features import (
    CRITICAL_FEATURE_GROUPS,
    EMBER2024_FEATURE_COUNT,
    AuthenticodeSignature,
    ExportsInfo,
    PEFeatureExtractor,
)

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "pe_samples")
_FIXTURES = sorted(glob.glob(os.path.join(_FIXTURE_DIR, "*.exe")))
_SIGNED = os.path.join(_FIXTURE_DIR, "sample_signed64.exe")
_IDS = [os.path.basename(p) for p in _FIXTURES]


@pytest.fixture(scope="module")
def extractor():
    return PEFeatureExtractor()


@pytest.fixture(scope="module")
def group_offsets(extractor):
    """name -> (start, dim) in the assembled 2568-vector, taken from the
    live group list so it can never drift from the real layout."""
    off, out = 0, {}
    for g in extractor._groups:
        out[g.name] = (off, g.dim)
        off += g.dim
    return out


def _read(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def test_fixtures_present():
    assert _FIXTURES, f"no PE fixtures in {_FIXTURE_DIR}"
    assert _SIGNED in _FIXTURES, "sample_signed64.exe fixture missing"


# --------------------------------------------------------------- check 1
def test_group_dims_sum_to_total(extractor):
    assert sum(g.dim for g in extractor._groups) == EMBER2024_FEATURE_COUNT


@pytest.mark.parametrize("path", _FIXTURES, ids=_IDS)
def test_vector_contract(extractor, group_offsets, path):
    bytez = _read(path)
    assert extractor.is_valid_pe(bytez), f"{path} did not parse as a PE"
    v = extractor.feature_vector(bytez)
    assert v.shape == (EMBER2024_FEATURE_COUNT,)
    assert v.dtype == np.float32
    assert np.isfinite(v).all(), "non-finite value in feature vector"

    hstart, hdim = group_offsets["histogram"]
    assert hdim == 256
    assert v[hstart:hstart + hdim].sum() == pytest.approx(1.0, abs=1e-4), \
        "byte histogram is not L1-normalised"

    # GeneralFileInfo: feature 0 is the raw file size.
    assert v[0] == pytest.approx(float(len(bytez)))


@pytest.mark.parametrize("path", _FIXTURES, ids=_IDS)
def test_per_group_slice_lengths(extractor, path):
    raw = extractor.raw_features(_read(path))
    for g in extractor._groups:
        part = g.process_raw_features(raw[g.name])
        assert part.shape == (g.dim,), f"group {g.name}: {part.shape} != ({g.dim},)"


# --------------------------------------------------------------- check 2
@pytest.mark.parametrize("path", _FIXTURES, ids=_IDS)
def test_determinism(path):
    bytez = _read(path)
    v1 = PEFeatureExtractor().feature_vector(bytez)
    v2 = PEFeatureExtractor().feature_vector(bytez)
    assert np.array_equal(v1, v2), "extraction is not deterministic"


# --------------------------------------------------------------- check 3
def test_signed_binary_authenticode_populated(extractor, group_offsets):
    bytez = _read(_SIGNED)
    raw = extractor.raw_features(bytez)
    grp = AuthenticodeSignature().process_raw_features(raw["authenticode"])
    # slot 0 = num_certs, slot 4 = parse_error (see AuthenticodeSignature)
    assert grp[0] >= 1, "signed fixture parsed 0 certs -- silent authenticode failure"
    assert grp[4] == 0, "parse_error set on a known-good signed fixture"

    astart, adim = group_offsets["authenticode"]
    assert adim == 8
    assert extractor.feature_vector(bytez)[astart:astart + adim].any(), \
        "authenticode slice is all-zero on a signed PE"


def test_unsigned_fixtures_have_zero_certs(extractor):
    for path in _FIXTURES:
        if path == _SIGNED:
            continue
        raw = extractor.raw_features(_read(path))
        grp = AuthenticodeSignature().process_raw_features(raw["authenticode"])
        assert grp[0] == 0, f"{os.path.basename(path)}: unexpected certs on an unsigned fixture"


# --------------------------------------------------------------- check 4
def test_exports_slot_is_count_not_constant():
    ei = ExportsInfo()
    assert ei.process_raw_features([])[0] == 0.0
    for n in (1, 7, 128, 413):
        names = [f"Export{i}" for i in range(n)]
        got = ei.process_raw_features(names)[0]
        assert got == float(n), (
            f"exports slot 0 == {got}, expected {n} "
            "(regression: the bug returned len(hash)==128 for any non-empty list)"
        )


@pytest.mark.parametrize("path", _FIXTURES, ids=_IDS)
def test_exports_slot_matches_raw_count(extractor, group_offsets, path):
    raw = extractor.raw_features(_read(path))
    start, _ = group_offsets["exports"]
    v = extractor.process_raw_features(raw)
    assert v[start] == float(len(raw["exports"]))


# --------------------------------------------------------------- check 5
def test_adapter_group_fields_match_extractor(extractor):
    assert list(_GROUP_FIELDS) == [g.name for g in extractor._groups], (
        "features/ember2024_adapter.py::_GROUP_FIELDS is out of sync with "
        "PEFeatureExtractor._groups -- a group was added/renamed on one side only"
    )


@pytest.mark.parametrize("path", _FIXTURES, ids=_IDS)
def test_adapter_passthrough_is_byte_identical(extractor, path):
    raw = extractor.raw_features(_read(path))
    v_live = extractor.process_raw_features(raw)
    v_adapter = record_to_vector(raw)
    assert np.array_equal(v_live, v_adapter), (
        "ember2024_adapter.record_to_vector diverged from the live "
        "process_raw_features path on identical group dicts"
    )


# ------------------------------------------------- check 6 (review item 6)
# Feature-group degradation is REPORTED, not silently zero-filled, and the
# startup self-test catches a broken extractor.

def test_clean_extraction_reports_no_degraded_groups(extractor):
    for path in _FIXTURES:
        _, degraded = extractor.feature_vector_with_report(_read(path))
        assert degraded == [], f"{os.path.basename(path)}: unexpected degraded groups {degraded}"


def test_raising_group_is_reported_and_zero_filled(monkeypatch):
    """A group that raises during raw extraction is recorded in the degraded
    list and its slice is zero-filled -- not left as a partial/NaN vector."""
    ex = PEFeatureExtractor()
    target = next(g for g in ex._groups if g.name == "imports")
    monkeypatch.setattr(target, "raw_features",
                        lambda bytez, pe: (_ for _ in ()).throw(RuntimeError("boom")))

    off = 0
    for g in ex._groups:
        if g.name == "imports":
            break
        off += g.dim

    vec, degraded = ex.feature_vector_with_report(_read(_SIGNED))
    assert vec.shape == (EMBER2024_FEATURE_COUNT,)
    assert np.isfinite(vec).all()
    assert "imports" in degraded
    assert not vec[off:off + target.dim].any(), "degraded 'imports' slice was not zero-filled"


def test_double_failure_still_yields_full_finite_vector(monkeypatch):
    """Even if the pe=None retry ALSO raises for every group, the vector is
    still (2568,), finite, all-zero -- and every group is reported degraded."""
    ex = PEFeatureExtractor()
    for g in ex._groups:
        monkeypatch.setattr(g, "raw_features",
                            lambda bytez, pe: (_ for _ in ()).throw(RuntimeError("boom")))
    vec, degraded = ex.feature_vector_with_report(_read(_SIGNED))
    assert vec.shape == (EMBER2024_FEATURE_COUNT,)
    assert np.isfinite(vec).all()
    assert (vec == 0).all()
    assert sorted(degraded) == sorted(g.name for g in ex._groups)


def test_authenticode_parse_error_is_reported_as_degraded(monkeypatch, extractor):
    """authenticode sets parse_error=1 instead of raising -- it must still
    surface in the degraded list."""
    target = next(g for g in extractor._groups if g.name == "authenticode")
    monkeypatch.setattr(
        target, "raw_features",
        lambda bytez, pe: {"num_certs": 0, "self_signed": 0, "empty_program_name": 0,
                           "no_countersigner": 0, "parse_error": 1, "chain_max_depth": 0,
                           "latest_signing_time": 0.0, "signing_time_diff": 0.0},
    )
    _, degraded = extractor.feature_vector_with_report(_read(_SIGNED))
    assert "authenticode" in degraded


def test_self_test_passes_on_bundled_reference(extractor):
    assert extractor.self_test() == []


def test_self_test_flags_a_broken_critical_group(monkeypatch, extractor):
    target = next(g for g in extractor._groups if g.name == "section")
    monkeypatch.setattr(target, "raw_features",
                        lambda bytez, pe: (_ for _ in ()).throw(RuntimeError("boom")))
    failures = extractor.self_test()
    assert "degraded_group:section" in failures
    assert "section" in CRITICAL_FEATURE_GROUPS


def test_self_test_reports_missing_reference_pe(extractor, tmp_path):
    out = extractor.self_test(pe_path=tmp_path / "nonexistent.exe")
    assert out and out[0].startswith("reference_pe_not_found:")


# --------------------------------------------------------------- misc
def test_non_pe_rejected(extractor):
    assert not extractor.is_valid_pe(b"This is definitely not a PE file.\n" * 40)
