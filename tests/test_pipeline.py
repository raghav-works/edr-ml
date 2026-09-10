"""
End-to-end pipeline coverage for:

  * review item 9 -- the "cannot analyze" cases (missing path / not-a-file /
    oversized / non-PE bytes) must resolve to NEEDS_REVIEW with a specific
    reason code, never a malware ALERT. These short-circuit in
    CortexPipeline.scan() before the static model is ever invoked.

  * review item 10 -- model health is recorded on ScanResult.signal_health,
    SEPARATE from the security verdict. A configured memory/network model
    that raises at runtime -> "model_error" (and NEEDS_REVIEW via its ERROR
    verdict); memory/network features supplied with no model wired ->
    "model_not_configured" AND the decision is unchanged (neutral).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from inference.pipeline import MAX_FILE_SIZE_BYTES, CortexPipeline
from inference.policy_engine import (
    FinalDecision, MemoryVerdict, NetworkVerdict, StaticVerdict,
)

_VALID_PE = os.path.join(
    os.path.dirname(__file__), "fixtures", "pe_samples", "sample_cli64.exe"
)
_MEMORY_FEATURES = np.zeros(62, dtype=np.float32)
_NETWORK_FEATURES = np.zeros(78, dtype=np.float32)


class _ExplodingStaticModel:
    """Stands in for models.static_lgbm.LGBMModel. scan() must never reach it
    on a cannot-analyze path; if it does, fail loudly rather than silently
    passing on some other behaviour."""

    def predict_proba(self, *_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("static model was invoked on a cannot-analyze path")


class _BenignStaticModel:
    """Returns a confidently-benign score so static -> ALLOW, giving the
    item-10 tests a clean ALLOW baseline to show a memory/network signal
    changing (or not changing) the outcome."""

    def predict_proba(self, X, *_args, **_kwargs):
        return np.full(len(X), 0.01, dtype=np.float64)


class _ExplodingModel:
    """A configured memory/network model whose predict_proba raises at
    runtime -- the item-10 "model_error" case."""

    def predict_proba(self, *_args, **_kwargs):
        raise RuntimeError("simulated model failure")


@pytest.fixture
def pipeline():
    # No behavioral / memory / network models: this fixture only exercises the
    # static-side "cannot analyze" short-circuits.
    return CortexPipeline(static_model=_ExplodingStaticModel())


def _assert_needs_review(result, expected_reason):
    assert result.final_decision == FinalDecision.NEEDS_REVIEW
    assert result.static_verdict == StaticVerdict.ERROR
    assert expected_reason in result.reason_codes
    # item 9: a file we could not analyze is NOT a malware finding and NOT a
    # clean pass.
    assert result.final_decision not in (
        FinalDecision.ALLOW, FinalDecision.ALERT,
        FinalDecision.BLOCK, FinalDecision.TERMINATE,
    )
    # item 10: a file-level failure with no memory/network signal supplied
    # leaves signal_health empty -- model health and "cannot analyze" are
    # tracked in different places.
    assert result.signal_health == {}


def test_missing_path_is_needs_review(pipeline, tmp_path):
    result = pipeline.scan(str(tmp_path / "does_not_exist.exe"))
    _assert_needs_review(result, "path_not_found")


def test_directory_is_needs_review(pipeline, tmp_path):
    d = tmp_path / "a_directory"
    d.mkdir()
    result = pipeline.scan(str(d))
    _assert_needs_review(result, "not_a_file")


def test_oversized_file_is_needs_review(pipeline, tmp_path):
    big = tmp_path / "huge.bin"
    # Sparse file: seek past the cap and write one byte. Logical size exceeds
    # MAX_FILE_SIZE_BYTES without actually writing 100+ MiB to disk.
    with open(big, "wb") as fh:
        fh.seek(MAX_FILE_SIZE_BYTES + 1)
        fh.write(b"\0")
    assert big.stat().st_size > MAX_FILE_SIZE_BYTES
    result = pipeline.scan(str(big))
    _assert_needs_review(result, "file_too_large")


def test_non_pe_bytes_is_needs_review(pipeline, tmp_path):
    junk = tmp_path / "notpe.txt"
    junk.write_bytes(b"this is plainly not a portable executable\n" * 50)
    result = pipeline.scan(str(junk))
    _assert_needs_review(result, "invalid_or_non_pe_file")


def test_security_event_carries_needs_review(pipeline, tmp_path):
    """The serialized event a caller ships downstream must also say
    NEEDS_REVIEW, not ALERT."""
    result = pipeline.scan(str(tmp_path / "nope.exe"))
    event = pipeline.to_security_event(result)
    assert event["final_decision"] == "NEEDS_REVIEW"
    assert "path_not_found" in event["reason_codes"]


# --------------------------------------------------------------------------
# review item 10 -- model health, separate from the security verdict
# --------------------------------------------------------------------------

def test_healthy_scan_has_empty_signal_health():
    """A valid PE scored by a working static model, no memory/network signal:
    signal_health stays empty."""
    pipe = CortexPipeline(static_model=_BenignStaticModel())
    result = pipe.scan(_VALID_PE)
    assert result.static_verdict == StaticVerdict.ALLOW
    assert result.final_decision == FinalDecision.ALLOW
    assert result.signal_health == {}


def test_memory_model_error_is_needs_review_with_health():
    """A configured memory model that raises: verdict ERROR -> NEEDS_REVIEW
    (item 9 routing), and signal_health['memory'] == 'model_error' so ops can
    tell it apart from a genuine file-analysis failure (item 10)."""
    pipe = CortexPipeline(static_model=_BenignStaticModel(), memory_model=_ExplodingModel())
    result = pipe.scan(_VALID_PE, memory_features=_MEMORY_FEATURES)

    assert result.memory_verdict == MemoryVerdict.ERROR
    assert result.final_decision == FinalDecision.NEEDS_REVIEW
    assert "memory_scan_error" in result.reason_codes
    assert result.signal_health == {"memory": "model_error"}

    event = pipe.to_security_event(result)
    assert event["signal_health"] == {"memory": "model_error"}


def test_network_model_error_is_needs_review_with_health():
    pipe = CortexPipeline(static_model=_BenignStaticModel(), network_model=_ExplodingModel())
    result = pipe.scan(_VALID_PE, network_features=_NETWORK_FEATURES)

    assert result.network_verdict == NetworkVerdict.ERROR
    assert result.final_decision == FinalDecision.NEEDS_REVIEW
    assert "network_scan_error" in result.reason_codes
    assert result.signal_health == {"network": "model_error"}


def test_memory_features_without_model_is_visible_but_neutral():
    """Features supplied, no memory model wired: signal_health flags it, but
    the decision is byte-for-byte what the same scan produces with no memory
    features at all -- a not-yet-deployed signal must not push scans to
    NEEDS_REVIEW (item 10, case A stays neutral)."""
    pipe = CortexPipeline(static_model=_BenignStaticModel())  # memory_model=None

    baseline = pipe.scan(_VALID_PE)
    with_feats = pipe.scan(_VALID_PE, memory_features=_MEMORY_FEATURES)

    assert with_feats.memory_verdict == MemoryVerdict.NOT_PROVIDED
    assert with_feats.final_decision == baseline.final_decision == FinalDecision.ALLOW
    assert with_feats.reason_codes == baseline.reason_codes
    assert with_feats.signal_health == {"memory": "model_not_configured"}
    assert baseline.signal_health == {}


def test_network_features_without_model_is_visible_but_neutral():
    pipe = CortexPipeline(static_model=_BenignStaticModel())  # network_model=None

    baseline = pipe.scan(_VALID_PE)
    with_feats = pipe.scan(_VALID_PE, network_features=_NETWORK_FEATURES)

    assert with_feats.network_verdict == NetworkVerdict.NOT_PROVIDED
    assert with_feats.final_decision == baseline.final_decision == FinalDecision.ALLOW
    assert with_feats.reason_codes == baseline.reason_codes
    assert with_feats.signal_health == {"network": "model_not_configured"}


def test_signal_health_does_not_reach_decide():
    """signal_health is diagnostic only: a memory model_error and a memory
    'model_not_configured' produce the same final_decision they would if
    decide() saw the raw verdicts (ERROR -> NEEDS_REVIEW, NOT_PROVIDED ->
    neutral). Guards that no one wired signal_health into the decision."""
    err = CortexPipeline(static_model=_BenignStaticModel(), memory_model=_ExplodingModel())
    unconfigured = CortexPipeline(static_model=_BenignStaticModel())

    r_err = err.scan(_VALID_PE, memory_features=_MEMORY_FEATURES)
    r_unconf = unconfigured.scan(_VALID_PE, memory_features=_MEMORY_FEATURES)

    assert r_err.final_decision == FinalDecision.NEEDS_REVIEW      # from ERROR verdict
    assert r_unconf.final_decision == FinalDecision.ALLOW          # NOT_PROVIDED stays neutral
