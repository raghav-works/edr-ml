"""
End-to-end pipeline coverage for the "cannot analyze" cases (review item 9):
a file the analyzer cannot even attempt to score must resolve to
NEEDS_REVIEW with a specific reason code -- never a malware ALERT.

These paths (missing path / not-a-file / oversized / non-PE bytes) all
short-circuit in CortexPipeline.scan() before the static model is ever
invoked, so a stub static model that raises if touched is enough -- and it
doubles as a guard that these paths really do not reach the model.
"""
from __future__ import annotations

import pytest

from inference.pipeline import MAX_FILE_SIZE_BYTES, CortexPipeline
from inference.policy_engine import FinalDecision, StaticVerdict


class _ExplodingStaticModel:
    """Stands in for models.static_lgbm.LGBMModel. scan() must never reach it
    on a cannot-analyze path; if it does, fail loudly rather than silently
    passing on some other behaviour."""

    def predict_proba(self, *_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("static model was invoked on a cannot-analyze path")


@pytest.fixture
def pipeline():
    # No behavioral / memory / network models: this suite only exercises the
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
