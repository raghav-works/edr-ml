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

    # item 2: static BLOCK is interim-capped to ALERT (uncorroborated) and
    # must NOT pre-empt behavioral.
    ((S.BLOCK, B.NOT_PROVIDED, NM, NN), F.ALERT, "static_block_capped_at_alert"),
    ((S.BLOCK, B.BENIGN, NM, NN), F.ALERT, "static_block_capped_at_alert"),
    ((S.BLOCK, B.MALICIOUS, NM, NN), F.TERMINATE, "behavioral_malicious"),
    # addition B2: static BLOCK corroborated by memory/network MALICIOUS
    # escalates to BLOCK instead of the capped ALERT above -- behavioral
    # MALICIOUS still outranks it regardless (row above).
    ((S.BLOCK, B.NOT_PROVIDED, M.MALICIOUS, NN), F.BLOCK, "static_block_corroborated"),
    ((S.BLOCK, B.NOT_PROVIDED, NM, N.MALICIOUS), F.BLOCK, "static_block_corroborated"),

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


def test_decide_returns_block_iff_static_block_is_corroborated():
    """Superseded item-2 invariant: pre-B2, decide() never returned BLOCK at
    all (static's autonomous BLOCK was fully capped to ALERT). Addition B2
    (2026-09-18) narrows that: BLOCK is now reachable, but ONLY when static
    is independently BLOCK AND corroborated by memory or network MALICIOUS
    -- and even then, behavioral MALICIOUS's uncapped TERMINATE still wins.
    Every other combination in the full product must still never reach
    BLOCK, exactly as before."""
    for s, b, m, n in _ALL:
        final = decide(s, b, m, n)[0]
        expected_block = (
            b != B.MALICIOUS
            and s == S.BLOCK
            and (m == M.MALICIOUS or n == N.MALICIOUS)
        )
        assert (final == F.BLOCK) == expected_block


def test_terminate_iff_behavioral_malicious():
    for s, b, m, n in _ALL:
        assert (decide(s, b, m, n)[0] == F.TERMINATE) == (b == B.MALICIOUS)


def test_network_alone_never_escalates_past_alert():
    """Item 6: network MALICIOUS with behavioral != MALICIOUS and static NOT
    independently BLOCK -> always ALERT (never ALLOW, never TERMINATE).
    When static IS independently BLOCK, network's MALICIOUS verdict
    corroborates STATIC's verdict (addition B2) and the outcome becomes
    BLOCK -- that is static's escalation, not network's: network's OWN
    ceiling is still exactly ALERT in every case where static is not
    BLOCK, which is what "alone" means here."""
    for s, m in itertools.product(list(S), list(M)):
        for b in _B_NOT_MAL:
            final = decide(s, b, m, N.MALICIOUS)[0]
            if s == S.BLOCK:
                assert final == F.BLOCK  # corroborated static BLOCK (addition B2)
            else:
                assert final == F.ALERT


def test_memory_alone_never_escalates_past_alert():
    """Same reasoning as test_network_alone_never_escalates_past_alert:
    memory's own ceiling is ALERT except when it corroborates an
    independently-BLOCK static verdict (addition B2)."""
    for s, n in itertools.product(list(S), list(N)):
        for b in _B_NOT_MAL:
            final = decide(s, b, M.MALICIOUS, n)[0]
            if s == S.BLOCK:
                assert final == F.BLOCK  # corroborated static BLOCK (addition B2)
            else:
                assert final == F.ALERT


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


# --------------------------------------------------------- addition B1: corroboration
# "Static false-positive severity cluster" (OPEN_ITEMS.md). Two or more of
# {static ALERT/BLOCK, memory MALICIOUS, network MALICIOUS} agreeing adds
# "corroborated_multi_signal" to reasons WITHOUT changing FinalDecision.
# Behavioral is deliberately excluded as a corroboration input/target -- it
# already has uncapped TERMINATE authority on its own.

def test_single_signal_is_not_corroborated():
    for s, b, m, n in [
        (S.ALLOW, B.NOT_PROVIDED, M.MALICIOUS, NN),
        (S.ALLOW, B.NOT_PROVIDED, NM, N.MALICIOUS),
        (S.ALERT, B.NOT_PROVIDED, NM, NN),
        (S.BLOCK, B.NOT_PROVIDED, NM, NN),
    ]:
        final, reasons = decide(s, b, m, n)
        assert "corroborated_multi_signal" not in reasons


def test_behavioral_is_not_a_corroboration_input():
    """Behavioral MALICIOUS is uncapped (TERMINATE) on its own and must not
    count toward, or be counted as, corroboration -- only one of
    {static ALERT/BLOCK, memory, network} is present here, so no flag."""
    final, reasons = decide(S.ALERT, B.MALICIOUS, NM, NN)
    assert final == F.TERMINATE
    assert "corroborated_multi_signal" not in reasons


def test_two_signals_corroborate_without_changing_outcome():
    # NOTE: static BLOCK + memory/network MALICIOUS is deliberately NOT one
    # of these cases -- addition B2 (below) makes that combination change
    # the outcome to BLOCK on purpose. These three all involve static ALLOW
    # or ALERT, where corroboration is flag-only (B1) with no escalation
    # path (B2 only ever applies to static BLOCK).
    for s, b, m, n, expected_final in [
        (S.ALLOW, B.NOT_PROVIDED, M.MALICIOUS, N.MALICIOUS, F.ALERT),   # memory+network
        (S.ALERT, B.NOT_PROVIDED, M.MALICIOUS, NN, F.ALERT),            # static(ALERT)+memory
        (S.ALERT, B.NOT_PROVIDED, NM, N.MALICIOUS, F.ALERT),            # static(ALERT)+network
    ]:
        final, reasons = decide(s, b, m, n)
        assert final == expected_final
        assert "corroborated_multi_signal" in reasons


def test_three_signals_corroborate_once_not_thrice():
    final, reasons = decide(S.ALERT, B.NOT_PROVIDED, M.MALICIOUS, N.MALICIOUS)
    assert final == F.ALERT
    assert reasons.count("corroborated_multi_signal") == 1


def test_corroboration_flag_survives_under_terminate():
    """Corroboration is still recorded as audit information even when
    behavioral's own uncapped TERMINATE decides the outcome -- it cannot
    change a TERMINATE, but "was this also independently corroborated" is
    useful even then."""
    final, reasons = decide(S.ALERT, B.MALICIOUS, M.MALICIOUS, NN)
    assert final == F.TERMINATE
    assert reasons[0] == "behavioral_malicious"
    assert "corroborated_multi_signal" in reasons


def test_corroboration_does_not_leak_into_needs_review_or_allow():
    final, reasons = decide(S.ERROR, B.ERROR, M.ERROR, N.ERROR)
    assert final == F.NEEDS_REVIEW
    assert "corroborated_multi_signal" not in reasons

    final, reasons = decide(S.ALLOW, B.BENIGN, NM, NN)
    assert final == F.ALLOW
    assert "corroborated_multi_signal" not in reasons


# ------------------------------------------------- addition B2: corroborated BLOCK
# "Static false-positive severity cluster" (OPEN_ITEMS.md), mechanism (ii):
# static BLOCK demotes to ALERT by default; escalates to BLOCK only when
# corroborated by memory or network MALICIOUS. Additive to, not a
# replacement for, item 2's removal criteria for static acting alone.

def test_uncorroborated_block_still_demotes_to_alert():
    """The default is unchanged: static BLOCK alone, or BLOCK alongside a
    non-MALICIOUS memory/network verdict, still demotes to ALERT."""
    for m, n in [(NM, NN), (M.BENIGN, N.BENIGN), (M.ERROR, N.ERROR), (M.NOT_PROVIDED, N.ERROR)]:
        final, reasons = decide(S.BLOCK, B.NOT_PROVIDED, m, n)
        assert final == F.ALERT
        assert "static_block_capped_at_alert" in reasons
        assert "static_block_corroborated" not in reasons


def test_memory_corroborates_static_block():
    final, reasons = decide(S.BLOCK, B.NOT_PROVIDED, M.MALICIOUS, NN)
    assert final == F.BLOCK
    assert reasons[0] == "static_block_corroborated"
    assert "memory_malicious" in reasons
    assert "corroborated_multi_signal" in reasons
    assert "static_block_capped_at_alert" not in reasons  # not the demoted path


def test_network_corroborates_static_block():
    final, reasons = decide(S.BLOCK, B.NOT_PROVIDED, NM, N.MALICIOUS)
    assert final == F.BLOCK
    assert reasons[0] == "static_block_corroborated"
    assert "network_malicious" in reasons


def test_both_memory_and_network_corroborate_static_block_once():
    final, reasons = decide(S.BLOCK, B.NOT_PROVIDED, M.MALICIOUS, N.MALICIOUS)
    assert final == F.BLOCK
    assert reasons[0] == "static_block_corroborated"
    assert "memory_malicious" in reasons
    assert "network_malicious" in reasons
    assert reasons.count("static_block_corroborated") == 1


def test_behavioral_terminate_still_outranks_corroborated_block():
    """Behavioral's uncapped authority is untouched by B2: even a
    corroborated static BLOCK loses to a completed behavioral MALICIOUS."""
    final, reasons = decide(S.BLOCK, B.MALICIOUS, M.MALICIOUS, N.MALICIOUS)
    assert final == F.TERMINATE
    assert reasons[0] == "behavioral_malicious"


def test_static_alert_never_escalates_to_block_even_when_corroborated():
    """B2 only applies to static BLOCK, never to static ALERT -- a merely
    ALERT-scoring file cannot reach BLOCK no matter how much corroboration
    is available."""
    final, reasons = decide(S.ALERT, B.NOT_PROVIDED, M.MALICIOUS, N.MALICIOUS)
    assert final == F.ALERT
    assert "static_block_corroborated" not in reasons


def test_memory_or_network_alone_can_never_reach_block():
    """CRITICAL CONSTRAINT: corroboration unlocks ONLY static's own BLOCK
    verdict. Memory/network's own ceiling must stay ALERT in every case
    where static is not independently BLOCK, across the full product --
    not just the hand-picked cases above."""
    for s, b, m, n in _ALL:
        if s != S.BLOCK and b != B.MALICIOUS:
            assert decide(s, b, m, n)[0] != F.BLOCK


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
