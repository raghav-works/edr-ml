"""
Truth table + invariants for inference/policy_engine.decide() and the
verdict_from_score helpers, plus the config/thresholds.yaml loader.

These are the cases verified by hand while building review item 2 (Cortex-
Static BLOCK interim-cap) and item 6 (Cortex-Network wiring / NOT_PROVIDED
vs ERROR); this makes them permanent.
"""
from __future__ import annotations

import inspect
import itertools

import pytest
import yaml

from inference import policy_engine as pe
from inference.policy_engine import (
    BEHAVIORAL_MALICIOUS_MIN,
    EMULATION_MALICIOUS_MIN,
    MEMORY_MALICIOUS_MIN,
    NETWORK_MALICIOUS_MIN,
    STATIC_ALLOW_MAX,
    STATIC_BLOCK_MIN,
    BehavioralVerdict as B,
    EmulationVerdict,
    FinalDecision as F,
    MemoryVerdict as M,
    NetworkVerdict as N,
    StaticVerdict as S,
    behavioral_verdict_from_score,
    decide,
    emulation_verdict_from_score,
    memory_verdict_from_score,
    network_verdict_from_score,
    static_verdict_from_score,
)

NM = M.NOT_PROVIDED
NN = N.NOT_PROVIDED


# ---------------------------------------------------------------- truth table
# (static, behavioral, memory, network) -> (final decision, reason that must appear)
CASES = [
    ((S.ALLOW, B.BENIGN, NM, NN), F.ALLOW, "no_malicious_evidence"),
    ((S.ALLOW, B.NOT_PROVIDED, NM, NN), F.ALLOW, "no_malicious_evidence"),
    ((S.ALLOW, B.PENDING, NM, NN), F.ALLOW, "no_malicious_evidence"),

    # item 2: static BLOCK is interim-capped to ALERT and must NOT pre-empt behavioral
    ((S.BLOCK, B.NOT_PROVIDED, NM, NN), F.ALERT, "static_block_capped_at_alert"),
    ((S.BLOCK, B.BENIGN, NM, NN), F.ALERT, "static_block_capped_at_alert"),
    ((S.BLOCK, B.MALICIOUS, NM, NN), F.TERMINATE, "behavioral_malicious"),
    ((S.BLOCK, B.NOT_PROVIDED, M.MALICIOUS, NN), F.ALERT, "memory_malicious"),
    ((S.BLOCK, B.NOT_PROVIDED, NM, N.MALICIOUS), F.ALERT, "network_malicious"),

    # static ALERT
    ((S.ALERT, B.BENIGN, NM, NN), F.ALERT, "static_alert"),
    ((S.ALERT, B.MALICIOUS, NM, NN), F.TERMINATE, "behavioral_malicious"),

    # static ALLOW + one downstream signal
    ((S.ALLOW, B.MALICIOUS, NM, NN), F.TERMINATE, "behavioral_malicious"),
    ((S.ALLOW, B.BENIGN, M.MALICIOUS, NN), F.ALERT, "memory_malicious"),
    ((S.ALLOW, B.BENIGN, NM, N.MALICIOUS), F.ALERT, "network_malicious"),
    ((S.ALLOW, B.BENIGN, M.MALICIOUS, N.MALICIOUS), F.ALERT, "memory_malicious"),

    # item 6: NOT_PROVIDED is neutral. item 9: a lone ERROR (nothing malicious
    # or suspicious from any channel) -> NEEDS_REVIEW, not ALERT.
    ((S.ALLOW, B.BENIGN, M.NOT_PROVIDED, N.NOT_PROVIDED), F.ALLOW, "no_malicious_evidence"),
    ((S.ALLOW, B.BENIGN, M.ERROR, NN), F.NEEDS_REVIEW, "memory_scan_error"),
    ((S.ALLOW, B.BENIGN, NM, N.ERROR), F.NEEDS_REVIEW, "network_scan_error"),
    ((S.ERROR, B.NOT_PROVIDED, NM, NN), F.NEEDS_REVIEW, "static_scan_error"),
    ((S.ALLOW, B.ERROR, NM, NN), F.NEEDS_REVIEW, "behavioral_scan_error"),

    # item 9: a completed malicious/suspicious signal still wins over another
    # signal's ERROR; the failed-signal code is retained in the reasons list.
    ((S.ERROR, B.MALICIOUS, NM, NN), F.TERMINATE, "static_scan_error"),
    ((S.ERROR, B.BENIGN, M.MALICIOUS, NN), F.ALERT, "static_scan_error"),
    ((S.ALERT, B.NOT_PROVIDED, NM, N.ERROR), F.ALERT, "network_scan_error"),

    # priority: behavioral MALICIOUS (TERMINATE) beats memory/network MALICIOUS (ALERT)
    ((S.ALLOW, B.MALICIOUS, M.MALICIOUS, N.MALICIOUS), F.TERMINATE, "behavioral_malicious"),
]


@pytest.mark.parametrize(
    "verdicts,expected,reason", CASES,
    ids=["-".join(v.value for v in vs) for vs, _, _ in CASES],
)
def test_decide_truth_table(verdicts, expected, reason):
    final, reasons = decide(*verdicts)
    assert final == expected
    assert reason in reasons


# ------------------------------------------------------ invariants over the product
_ALL = list(itertools.product(list(S), list(B), list(M), list(N)))
_B_NOT_MAL = (B.NOT_PROVIDED, B.PENDING, B.BENIGN, B.ERROR)


def test_decide_never_returns_block():
    """Item 2: static BLOCK is capped, so decide() must never return BLOCK."""
    assert all(decide(*v)[0] != F.BLOCK for v in _ALL)


def test_terminate_iff_behavioral_malicious():
    for s, b, m, n in _ALL:
        assert (decide(s, b, m, n)[0] == F.TERMINATE) == (b == B.MALICIOUS)


def test_network_alone_never_escalates_past_alert():
    """Item 6: network MALICIOUS with behavioral != MALICIOUS -> always ALERT
    (never ALLOW, never BLOCK/TERMINATE)."""
    for s, m in itertools.product(list(S), list(M)):
        for b in _B_NOT_MAL:
            assert decide(s, b, m, N.MALICIOUS)[0] == F.ALERT


def test_memory_alone_never_escalates_past_alert():
    for s, n in itertools.product(list(S), list(N)):
        for b in _B_NOT_MAL:
            assert decide(s, b, M.MALICIOUS, n)[0] == F.ALERT


def test_not_provided_is_neutral_for_memory_and_network():
    """Adding NOT_PROVIDED memory/network verdicts never changes the outcome
    static+behavioral alone produce."""
    for s, b in itertools.product(list(S), list(B)):
        assert decide(s, b)[0] == decide(s, b, M.NOT_PROVIDED, N.NOT_PROVIDED)[0]


def test_lone_error_returns_needs_review():
    """item 9: an ERROR from any single signal, with nothing malicious or
    suspicious from any channel, resolves to NEEDS_REVIEW -- was ALERT
    pre-item-9, and never ALLOW."""
    for bad in (
        (S.ALLOW, B.BENIGN, M.ERROR, N.NOT_PROVIDED),
        (S.ALLOW, B.BENIGN, M.NOT_PROVIDED, N.ERROR),
        (S.ALLOW, B.ERROR, M.NOT_PROVIDED, N.NOT_PROVIDED),
        (S.ERROR, B.NOT_PROVIDED, M.NOT_PROVIDED, N.NOT_PROVIDED),
    ):
        assert decide(*bad)[0] == F.NEEDS_REVIEW


def test_error_returns_needs_review():
    """item 9 invariant over the whole product: decide() returns NEEDS_REVIEW
    exactly when some signal is in ERROR AND no channel produced a completed
    finding (behavioral/memory/network MALICIOUS, or static ALERT/BLOCK).
    A completed finding always outranks another signal's failure."""
    for s, b, m, n in _ALL:
        any_error = (
            s == S.ERROR or b == B.ERROR or m == M.ERROR or n == N.ERROR
        )
        any_finding = (
            b == B.MALICIOUS or m == M.MALICIOUS or n == N.MALICIOUS
            or s in (S.ALERT, S.BLOCK)
        )
        expected = any_error and not any_finding
        assert (decide(s, b, m, n)[0] == F.NEEDS_REVIEW) == expected


def test_malicious_signal_wins_over_other_signal_error():
    """item 9: a signal that completed with a finding beats another signal's
    ERROR -- outcome is unchanged from pre-item-9. The failed-signal reason
    code is still recorded (audit-trail accumulation), and the code for the
    rung that drove the outcome is listed first."""
    final, reasons = decide(S.ERROR, B.MALICIOUS, NM, NN)
    assert final == F.TERMINATE
    assert reasons[0] == "behavioral_malicious"
    assert "static_scan_error" in reasons

    final, reasons = decide(S.ERROR, B.BENIGN, M.MALICIOUS, NN)
    assert final == F.ALERT
    assert reasons[0] == "memory_malicious"
    assert "static_scan_error" in reasons


def test_error_codes_accumulate():
    """Every signal in ERROR contributes its code, even when a higher rung
    decides the outcome. Regression: decide() used to return on the first
    ERROR rung and silently drop the rest."""
    final, reasons = decide(S.ERROR, B.ERROR, M.ERROR, N.ERROR)
    assert final == F.NEEDS_REVIEW
    assert set(reasons) == {
        "static_scan_error", "behavioral_scan_error",
        "memory_scan_error", "network_scan_error",
    }

    final, reasons = decide(S.ERROR, B.MALICIOUS, M.ERROR, N.ERROR)
    assert final == F.TERMINATE
    assert reasons[0] == "behavioral_malicious"
    assert set(reasons[1:]) == {
        "static_scan_error", "memory_scan_error", "network_scan_error",
    }


def test_capped_block_reason_code():
    """The pre-item-2 bare 'static_block' reason code is gone; a capped BLOCK
    reports 'static_block_capped_at_alert' whenever that branch is reached."""
    for s, b, m, n in _ALL:
        reasons = decide(s, b, m, n)[1]
        assert "static_block" not in reasons  # bare pre-cap code
        if s == S.BLOCK and b != B.MALICIOUS and m != M.MALICIOUS and n != N.MALICIOUS:
            assert "static_block_capped_at_alert" in reasons


# ------------------------------------------------------ verdict_from_score helpers
def test_static_verdict_from_score_bands():
    assert static_verdict_from_score(0.0) == S.ALLOW
    assert static_verdict_from_score(STATIC_ALLOW_MAX - 1e-9) == S.ALLOW
    assert static_verdict_from_score(STATIC_ALLOW_MAX) == S.ALERT
    assert static_verdict_from_score(STATIC_BLOCK_MIN - 1e-9) == S.ALERT
    assert static_verdict_from_score(STATIC_BLOCK_MIN) == S.BLOCK
    assert static_verdict_from_score(1.0) == S.BLOCK


def test_behavioral_verdict_from_score():
    assert behavioral_verdict_from_score(0.0) == B.BENIGN
    assert behavioral_verdict_from_score(BEHAVIORAL_MALICIOUS_MIN - 1e-9) == B.BENIGN
    assert behavioral_verdict_from_score(BEHAVIORAL_MALICIOUS_MIN) == B.MALICIOUS
    assert behavioral_verdict_from_score(1.0) == B.MALICIOUS


def test_memory_and_network_verdict_from_score():
    assert memory_verdict_from_score(MEMORY_MALICIOUS_MIN - 1e-9) == M.BENIGN
    assert memory_verdict_from_score(MEMORY_MALICIOUS_MIN) == M.MALICIOUS
    assert network_verdict_from_score(NETWORK_MALICIOUS_MIN - 1e-9) == N.BENIGN
    assert network_verdict_from_score(NETWORK_MALICIOUS_MIN) == N.MALICIOUS


def test_emulation_is_report_only_not_in_decide():
    assert emulation_verdict_from_score(EMULATION_MALICIOUS_MIN) == EmulationVerdict.MALICIOUS
    assert emulation_verdict_from_score(0.0) == EmulationVerdict.BENIGN
    assert "emulation" not in inspect.signature(decide).parameters
    assert "network_verdict" in inspect.signature(decide).parameters


# ------------------------------------------------------ threshold loading (SSOT)
def test_module_thresholds_match_yaml():
    y = yaml.safe_load(pe._THRESHOLDS_PATH.read_text(encoding="utf-8"))
    assert STATIC_ALLOW_MAX == y["static"]["allow_below"]
    assert STATIC_BLOCK_MIN == y["static"]["block_at_or_above"]
    assert BEHAVIORAL_MALICIOUS_MIN == y["behavioral"]["malicious_at_or_above"]
    assert MEMORY_MALICIOUS_MIN == y["memory"]["malicious_at_or_above"]
    assert NETWORK_MALICIOUS_MIN == y["network"]["malicious_at_or_above"]
    assert EMULATION_MALICIOUS_MIN == y["emulation"]["malicious_at_or_above"]


def test_static_thresholds_are_ordered():
    assert 0.0 < STATIC_ALLOW_MAX < STATIC_BLOCK_MIN < 1.0


def test_thr_helper_accepts_numbers_rejects_junk():
    assert pe._thr({"a": {"b": 0.5}}, "a", "b") == 0.5
    assert pe._thr({"a": 3}, "a") == 3.0
    with pytest.raises(RuntimeError, match="missing required key"):
        pe._thr({}, "a")
    with pytest.raises(RuntimeError, match="must be a number"):
        pe._thr({"a": "x"}, "a")
    with pytest.raises(RuntimeError, match="must be a number"):
        pe._thr({"a": True}, "a")


def test_load_thresholds_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(pe, "_THRESHOLDS_PATH", tmp_path / "nope.yaml")
    with pytest.raises(RuntimeError, match="not found"):
        pe._load_thresholds()

    bad = tmp_path / "bad.yaml"
    bad.write_text("this: [is, not: valid: yaml")
    monkeypatch.setattr(pe, "_THRESHOLDS_PATH", bad)
    with pytest.raises(RuntimeError, match="malformed"):
        pe._load_thresholds()

    lst = tmp_path / "list.yaml"
    lst.write_text("- 1\n- 2\n")
    monkeypatch.setattr(pe, "_THRESHOLDS_PATH", lst)
    with pytest.raises(RuntimeError, match="mapping"):
        pe._load_thresholds()
