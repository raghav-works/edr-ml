"""
End-to-end pipeline coverage for:

  * review item 9 -- the "cannot analyze" cases (missing path / not-a-file /
    oversized / non-PE bytes) must resolve to NEEDS_REVIEW with a specific
    reason code, never a malware ALERT. These short-circuit in
    CortexPipeline.scan() before the static model is ever invoked.

  * review item 10 -- model health is recorded on ScanResult.signal_health,
    SEPARATE from the security verdict. A configured memory/network/
    behavioral model that raises at runtime -> "model_error" (and
    NEEDS_REVIEW via its ERROR verdict); evidence supplied with no model
    wired -> "model_not_configured" AND the decision is unchanged (neutral)
    -- consistent across all three signals (OPEN_ITEMS.md's tracked
    behavioral config-gap asymmetry, resolved).

  * review item 6 -- a degraded feature group is recorded on
    ScanResult.degraded_groups. A degraded CRITICAL group sets
    static_verdict = ERROR (-> NEEDS_REVIEW) with reason
    static_features_degraded; a non-critical one only sets
    signal_health["static"] = "degraded". The pipeline self-test runs at
    construction (self_test=False here to keep fixtures fast/isolated).
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from features.authenticode_trust import AuthenticodeTrustResult
from inference.pipeline import MAX_FILE_SIZE_BYTES, CortexPipeline
from inference.policy_engine import (
    BehavioralVerdict, FinalDecision, MemoryVerdict, NetworkVerdict, StaticVerdict,
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


class _MaliciousModel:
    """A configured memory/network model that confidently scores MALICIOUS
    -- 0.999999 clears both MEMORY_MALICIOUS_MIN and NETWORK_MALICIOUS_MIN
    (config/thresholds.yaml) with no need to import either threshold here."""

    def predict_proba(self, X, *_args, **_kwargs):
        return np.full(len(X), 0.999999, dtype=np.float64)


def _pipe(**kwargs) -> CortexPipeline:
    """CortexPipeline with the item-6 startup self-test disabled by default --
    it does real disk I/O + a full extraction of the bundled reference PE on
    every construction, which these tests neither need nor want. The self-test
    itself is covered by its own tests below and by test_static_feature_parity.
    """
    kwargs.setdefault("self_test", False)
    return CortexPipeline(**kwargs)


@pytest.fixture
def pipeline():
    # No behavioral / memory / network models: this fixture only exercises the
    # static-side "cannot analyze" short-circuits.
    return _pipe(static_model=_ExplodingStaticModel())


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
    pipe = _pipe(static_model=_BenignStaticModel())
    result = pipe.scan(_VALID_PE)
    assert result.static_verdict == StaticVerdict.ALLOW
    assert result.final_decision == FinalDecision.ALLOW
    assert result.signal_health == {}


def test_memory_model_error_is_needs_review_with_health():
    """A configured memory model that raises: verdict ERROR -> NEEDS_REVIEW
    (item 9 routing), and signal_health['memory'] == 'model_error' so ops can
    tell it apart from a genuine file-analysis failure (item 10)."""
    pipe = _pipe(static_model=_BenignStaticModel(), memory_model=_ExplodingModel())
    result = pipe.scan(_VALID_PE, memory_features=_MEMORY_FEATURES)

    assert result.memory_verdict == MemoryVerdict.ERROR
    assert result.final_decision == FinalDecision.NEEDS_REVIEW
    assert "memory_scan_error" in result.reason_codes
    assert result.signal_health == {"memory": "model_error"}

    event = pipe.to_security_event(result)
    assert event["signal_health"] == {"memory": "model_error"}


def test_network_model_error_is_needs_review_with_health():
    pipe = _pipe(static_model=_BenignStaticModel(), network_model=_ExplodingModel())
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
    pipe = _pipe(static_model=_BenignStaticModel())  # memory_model=None

    baseline = pipe.scan(_VALID_PE)
    with_feats = pipe.scan(_VALID_PE, memory_features=_MEMORY_FEATURES)

    assert with_feats.memory_verdict == MemoryVerdict.NOT_PROVIDED
    assert with_feats.final_decision == baseline.final_decision == FinalDecision.ALLOW
    assert with_feats.reason_codes == baseline.reason_codes
    assert with_feats.signal_health == {"memory": "model_not_configured"}
    assert baseline.signal_health == {}


def test_network_features_without_model_is_visible_but_neutral():
    pipe = _pipe(static_model=_BenignStaticModel())  # network_model=None

    baseline = pipe.scan(_VALID_PE)
    with_feats = pipe.scan(_VALID_PE, network_features=_NETWORK_FEATURES)

    assert with_feats.network_verdict == NetworkVerdict.NOT_PROVIDED
    assert with_feats.final_decision == baseline.final_decision == FinalDecision.ALLOW
    assert with_feats.reason_codes == baseline.reason_codes
    assert with_feats.signal_health == {"network": "model_not_configured"}


def test_behavioral_trace_without_model_is_visible_but_neutral(tmp_path):
    """An API-call trace supplied, no behavioral_model/tokenizer wired: this
    used to be BehavioralVerdict.ERROR (-> NEEDS_REVIEW), the one asymmetry
    against memory/network's NOT_PROVIDED-is-neutral design (item 10).
    Fixed to match: NOT_PROVIDED, decision unchanged, and now visible via
    signal_health the same way memory/network already are."""
    trace = tmp_path / "trace.json"
    trace.write_text('["NtCreateFile", "NtWriteFile"]')

    pipe = _pipe(static_model=_BenignStaticModel())  # behavioral_model=None, tokenizer=None

    baseline = pipe.scan(_VALID_PE)
    with_trace = pipe.scan(_VALID_PE, api_calls_json_path=str(trace))

    assert with_trace.behavioral_verdict == BehavioralVerdict.NOT_PROVIDED
    assert with_trace.final_decision == baseline.final_decision == FinalDecision.ALLOW
    assert with_trace.reason_codes == baseline.reason_codes
    assert with_trace.signal_health == {"behavioral": "model_not_configured"}
    assert baseline.signal_health == {}


def test_signal_health_does_not_reach_decide():
    """signal_health is diagnostic only: a memory model_error and a memory
    'model_not_configured' produce the same final_decision they would if
    decide() saw the raw verdicts (ERROR -> NEEDS_REVIEW, NOT_PROVIDED ->
    neutral). Guards that no one wired signal_health into the decision."""
    err = _pipe(static_model=_BenignStaticModel(), memory_model=_ExplodingModel())
    unconfigured = _pipe(static_model=_BenignStaticModel())

    r_err = err.scan(_VALID_PE, memory_features=_MEMORY_FEATURES)
    r_unconf = unconfigured.scan(_VALID_PE, memory_features=_MEMORY_FEATURES)

    assert r_err.final_decision == FinalDecision.NEEDS_REVIEW      # from ERROR verdict
    assert r_unconf.final_decision == FinalDecision.ALLOW          # NOT_PROVIDED stays neutral


# --------------------------------------------------------------------------
# review item 6 -- degraded feature groups, separate from the verdict
# --------------------------------------------------------------------------

def _force_report(monkeypatch, pipe, degraded):
    """Make the pipeline's extractor report `degraded` (and an all-zero
    vector) without needing a real extraction failure."""
    monkeypatch.setattr(
        pipe.feature_extractor, "feature_vector_with_report",
        lambda bytez, **_kw: (np.zeros(2568, dtype=np.float32), list(degraded)),
    )


def test_clean_scan_has_no_degraded_groups():
    pipe = _pipe(static_model=_BenignStaticModel())
    result = pipe.scan(_VALID_PE)
    assert result.degraded_groups == []
    assert "static" not in result.signal_health
    assert result.final_decision == FinalDecision.ALLOW


def test_critical_degraded_group_routes_to_needs_review(monkeypatch):
    pipe = _pipe(static_model=_BenignStaticModel())
    _force_report(monkeypatch, pipe, ["imports", "section"])
    result = pipe.scan(_VALID_PE)

    assert result.degraded_groups == ["imports", "section"]
    assert result.static_verdict == StaticVerdict.ERROR
    assert "static_features_degraded" in result.reason_codes
    assert result.signal_health["static"] == "degraded"
    assert result.final_decision == FinalDecision.NEEDS_REVIEW
    # score is still computed and kept for telemetry, even though the verdict
    # is set aside
    assert result.static_score is not None

    event = pipe.to_security_event(result)
    assert event["degraded_groups"] == ["imports", "section"]
    assert event["signal_health"] == {"static": "degraded"}


def test_noncritical_degraded_group_keeps_verdict(monkeypatch):
    pipe = _pipe(static_model=_BenignStaticModel())
    _force_report(monkeypatch, pipe, ["exports"])
    result = pipe.scan(_VALID_PE)

    assert result.degraded_groups == ["exports"]
    assert result.static_verdict == StaticVerdict.ALLOW          # score-derived, kept
    assert "static_features_degraded" not in result.reason_codes
    assert result.signal_health["static"] == "degraded"          # telemetry only
    assert result.final_decision == FinalDecision.ALLOW


def test_critical_degraded_still_yields_to_behavioral_malicious(monkeypatch, tmp_path):
    """A degraded critical static extraction must NOT suppress a caller-
    supplied API trace: behavioral still runs, and a completed MALICIOUS
    verdict is still surfaced (decide() rung 1 -> ALERT, since static ERROR
    does not corroborate it under F4)."""
    import torch

    class _MaliciousBehavioral:
        def __call__(self, x):
            return torch.tensor([6.0])          # sigmoid -> ~0.9975 -> MALICIOUS

    class _PassThroughTokenizer:
        def encode(self, calls):
            return np.zeros(100, dtype=np.int64), "ok"

    trace = tmp_path / "trace.json"
    trace.write_text('["NtCreateFile", "NtWriteFile"]')

    pipe = _pipe(static_model=_BenignStaticModel(),
                 behavioral_model=_MaliciousBehavioral(),
                 tokenizer=_PassThroughTokenizer())
    _force_report(monkeypatch, pipe, ["imports"])
    result = pipe.scan(_VALID_PE, api_calls_json_path=str(trace))

    assert result.static_verdict == StaticVerdict.ERROR          # critical degraded
    assert result.behavioral_verdict.value == "MALICIOUS"        # behavioral still ran
    # F4: static ERROR does not corroborate behavioral -> ALERT, not TERMINATE
    assert result.final_decision == FinalDecision.ALERT
    assert "behavioral_malicious_uncorroborated" in result.reason_codes
    assert "static_features_degraded" in result.reason_codes


# --------------------------------------------------------------------------
# review item 6 -- startup self-test
# --------------------------------------------------------------------------

def test_pipeline_self_test_passes_on_bundled_reference_pe():
    # Default self_test=True; the committed reference PE is clean, so
    # constructing the pipeline must not raise.
    CortexPipeline(static_model=_BenignStaticModel())


def test_pipeline_self_test_raises_on_broken_critical_group(monkeypatch):
    monkeypatch.setattr(
        "features.pe_features.PEFeatureExtractor.self_test",
        lambda self, pe_path=None: ["degraded_group:imports"],
    )
    with pytest.raises(RuntimeError, match="self_test"):
        CortexPipeline(static_model=_BenignStaticModel())


def test_pipeline_self_test_tolerates_noncritical_and_missing_reference(monkeypatch):
    for benign in (["degraded_group:pefilewarnings"], ["reference_pe_not_found:/nope"]):
        monkeypatch.setattr(
            "features.pe_features.PEFeatureExtractor.self_test",
            lambda self, pe_path=None, _b=benign: list(_b),
        )
        CortexPipeline(static_model=_BenignStaticModel())  # must not raise


# --------------------------------------------------------------------------
# known-file allowlist (OPEN_ITEMS.md "Static false-positive severity
# cluster", addition A) -- step 3b in scan(), before static's ML judgment.
# --------------------------------------------------------------------------

with open(_VALID_PE, "rb") as _fh:
    _VALID_PE_SHA256 = hashlib.sha256(_fh.read()).hexdigest()


class _StubAllowlist:
    """Stands in for features.nsrl_allowlist.NSRLAllowlist -- exercises
    _check_allowlist()'s wiring without needing a real built artifact
    (that class has its own dedicated test suite,
    tests/test_nsrl_allowlist.py)."""

    def __init__(self, matching_sha256: str):
        self._match = matching_sha256

    def contains(self, sha256_hex: str) -> bool:
        return sha256_hex == self._match


def test_allowlist_nsrl_hit_sets_allow_and_skips_ml_scoring():
    # static_model is an _ExplodingStaticModel: if step 4's feature
    # extraction + predict_proba() ran anyway, this test would fail loudly
    # via that model's own assertion, not silently pass.
    pipe = _pipe(static_model=_ExplodingStaticModel(),
                 nsrl_allowlist=_StubAllowlist(matching_sha256=_VALID_PE_SHA256))
    result = pipe.scan(_VALID_PE)

    assert result.static_verdict == StaticVerdict.ALLOW
    assert "static_allowlisted_nsrl" in result.reason_codes
    assert result.static_score is None            # nothing was scored
    assert result.degraded_groups == []            # nothing was extracted
    assert result.final_decision == FinalDecision.ALLOW


def test_allowlist_authenticode_chain_hit_sets_allow_and_skips_ml_scoring(monkeypatch):
    # No nsrl_allowlist configured; the chain-verification leg needs no
    # external artifact and is always attempted. This repo has no real
    # trusted-signed fixture (its one signed fixture is deliberately
    # self-signed -- see test_authenticode_trust.py), so the "verified"
    # outcome is stubbed at the pipeline's import site rather than produced
    # by a real chain -- features/authenticode_trust.py's own test suite
    # covers verify_trusted_chain()'s real behavior.
    monkeypatch.setattr(
        "inference.pipeline.verify_trusted_chain",
        lambda bytez: AuthenticodeTrustResult(True, "verified"),
    )
    pipe = _pipe(static_model=_ExplodingStaticModel())
    result = pipe.scan(_VALID_PE)

    assert result.static_verdict == StaticVerdict.ALLOW
    assert "static_allowlisted_authenticode_chain" in result.reason_codes
    assert result.static_score is None
    assert result.final_decision == FinalDecision.ALLOW


def test_allowlist_miss_falls_through_to_normal_ml_scoring():
    # Neither leg matches on the real, unsigned, non-allowlisted fixture:
    # static's normal ML path must run completely unchanged.
    pipe = _pipe(static_model=_BenignStaticModel())
    result = pipe.scan(_VALID_PE)

    assert result.static_verdict == StaticVerdict.ALLOW  # score-derived here, not allowlist-derived
    assert result.static_score is not None
    assert not any(code.startswith("static_allowlisted") for code in result.reason_codes)


def test_allowlist_hit_does_not_suppress_memory_malicious():
    """The single most important safety property of this feature: an
    allowlist match bypasses ONLY static's own judgment. Memory runs
    independently of static_verdict entirely (see scan()'s step 7) and
    must still reach its own MALICIOUS verdict -- and decide()'s outcome --
    on an allowlisted file."""
    pipe = _pipe(static_model=_ExplodingStaticModel(),
                 nsrl_allowlist=_StubAllowlist(matching_sha256=_VALID_PE_SHA256),
                 memory_model=_MaliciousModel())
    result = pipe.scan(_VALID_PE, memory_features=_MEMORY_FEATURES)

    assert result.static_verdict == StaticVerdict.ALLOW
    assert "static_allowlisted_nsrl" in result.reason_codes
    assert result.memory_verdict == MemoryVerdict.MALICIOUS
    assert result.final_decision == FinalDecision.ALERT  # memory's own capped ceiling, unaffected by the allowlist


def test_allowlist_hit_does_not_suppress_network_malicious():
    pipe = _pipe(static_model=_ExplodingStaticModel(),
                 nsrl_allowlist=_StubAllowlist(matching_sha256=_VALID_PE_SHA256),
                 network_model=_MaliciousModel())
    result = pipe.scan(_VALID_PE, network_features=_NETWORK_FEATURES)

    assert result.static_verdict == StaticVerdict.ALLOW
    assert "static_allowlisted_nsrl" in result.reason_codes
    assert result.network_verdict == NetworkVerdict.MALICIOUS
    assert result.final_decision == FinalDecision.ALERT  # network's own capped ceiling, unaffected by the allowlist


def test_allowlist_hit_does_not_suppress_behavioral_malicious(tmp_path):
    import torch

    class _MaliciousBehavioral:
        def __call__(self, x):
            return torch.tensor([6.0])  # sigmoid -> ~0.9975 -> MALICIOUS

    class _PassThroughTokenizer:
        def encode(self, calls):
            return np.zeros(100, dtype=np.int64), "ok"

    trace = tmp_path / "trace.json"
    trace.write_text('["NtCreateFile", "NtWriteFile"]')

    pipe = _pipe(static_model=_ExplodingStaticModel(),
                 nsrl_allowlist=_StubAllowlist(matching_sha256=_VALID_PE_SHA256),
                 behavioral_model=_MaliciousBehavioral(),
                 tokenizer=_PassThroughTokenizer())
    result = pipe.scan(_VALID_PE, api_calls_json_path=str(trace))

    assert result.static_verdict == StaticVerdict.ALLOW
    assert "static_allowlisted_nsrl" in result.reason_codes
    assert result.behavioral_verdict.value == "MALICIOUS"
    # F4: an allowlisted (signed/NSRL) static ALLOW does not corroborate, so
    # behavioral alone gives ALERT -- not suppressed, but not TERMINATE.
    assert result.final_decision == FinalDecision.ALERT
    assert "behavioral_malicious_uncorroborated" in result.reason_codes


# --------------------------------------------------------------------------
# docs/CODE_REVIEW.md F12 / F13 -- non-finite scores and PENDING, end to end

class _ConstStaticModel:
    def __init__(self, value):
        self.value = value

    def predict_proba(self, X, *_args, **_kwargs):
        return np.full(len(X), self.value, dtype=np.float64)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")],
                         ids=["nan", "+inf", "-inf"])
def test_non_finite_static_score_is_needs_review(value):
    result = _pipe(static_model=_ConstStaticModel(value)).scan(_VALID_PE)
    assert result.static_verdict == StaticVerdict.ERROR
    assert result.final_decision == FinalDecision.NEEDS_REVIEW
    assert "static_scan_error" in result.reason_codes


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")],
                         ids=["nan", "+inf", "-inf"])
def test_non_finite_memory_and_network_scores_are_error(value):
    pipe = _pipe(static_model=_BenignStaticModel(),
                 memory_model=_ConstStaticModel(value),
                 network_model=_ConstStaticModel(value))
    result = pipe.scan(_VALID_PE, memory_features=_MEMORY_FEATURES,
                       network_features=_NETWORK_FEATURES)
    assert result.memory_verdict == MemoryVerdict.ERROR
    assert result.network_verdict == NetworkVerdict.ERROR
    assert result.final_decision == FinalDecision.NEEDS_REVIEW


@pytest.mark.parametrize("logit", [float("nan"), float("inf"), float("-inf")],
                         ids=["nan", "+inf", "-inf"])
def test_non_finite_behavioral_score_is_error(tmp_path, logit):
    import torch

    class _ConstBehavioral:
        def __call__(self, x):
            return torch.tensor([logit])

    class _OkTokenizer:
        def encode(self, calls):
            return np.zeros(100, dtype=np.int64), "ok"

    trace = tmp_path / "trace.json"
    trace.write_text('["NtCreateFile", "NtWriteFile"]')
    pipe = _pipe(static_model=_BenignStaticModel(),
                 behavioral_model=_ConstBehavioral(), tokenizer=_OkTokenizer())
    result = pipe.scan(_VALID_PE, api_calls_json_path=str(trace))
    # Checked on the raw logit (Step 4): sigmoid(+-inf) would otherwise hide a
    # broken model behind a normal-looking 1.0 / 0.0.
    assert result.behavioral_verdict == BehavioralVerdict.ERROR
    assert result.behavioral_score is None
    assert "behavioral_logit_non_finite" in result.reason_codes
    assert result.final_decision == FinalDecision.NEEDS_REVIEW


def test_behavioral_pending_with_static_allow_is_allow_unverified(tmp_path):
    class _ShortTokenizer:
        def encode(self, calls):
            return np.zeros(100, dtype=np.int64), "too_short"

    class _UnusedBehavioral:
        def __call__(self, x):  # pragma: no cover - must not run on PENDING
            raise AssertionError("behavioral model scored a PENDING trace")

    trace = tmp_path / "trace.json"
    trace.write_text('["NtCreateFile", "NtWriteFile"]')
    pipe = _pipe(static_model=_BenignStaticModel(),
                 behavioral_model=_UnusedBehavioral(), tokenizer=_ShortTokenizer())
    result = pipe.scan(_VALID_PE, api_calls_json_path=str(trace))
    assert result.static_verdict == StaticVerdict.ALLOW
    assert result.behavioral_verdict == BehavioralVerdict.PENDING
    assert result.final_decision == FinalDecision.ALLOW_UNVERIFIED
    assert "behavioral_pending_unverified" in result.reason_codes
    assert result.to_dict()["final_decision"] == "ALLOW_UNVERIFIED"
