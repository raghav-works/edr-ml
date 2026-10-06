"""
docs/CODE_REVIEW.md F3 / F21 / F12 follow-up -- behavioral input contract.

  * ApiTokenizer.encode canonicalises names exactly as training did
    (data/download_behavioral.py): lowercase, nothing else.
  * input-format diagnostics (empty / non-ASCII / whitespace names) are
    surfaced on ScanResult without changing scoring.
  * a scored window with too many <UNK> tokens is PENDING
    (behavioral_unk_rate_high), not scored.
  * the pipeline puts the behavioral model in eval mode and scores under
    torch.inference_mode(), so a model handed over in train mode is
    deterministic.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import pytest
import torch

from inference import pipeline as pipeline_mod
from inference.pipeline import CortexPipeline
from inference.policy_engine import (
    BEHAVIORAL_MAX_UNK_RATE, BehavioralVerdict, FinalDecision, StaticVerdict,
)
from models.behavioral_cnn import CortexBehavioralNet
from tokenizer.api_tokenizer import (
    MAX_SEQ_LEN, UNK_ID, ApiTokenizer, input_diagnostics, scored_unk_rate,
)

_VALID_PE = os.path.join(os.path.dirname(__file__), "fixtures", "pe_samples", "sample_cli64.exe")
_VOCAB = ["ntclose", "ntopenkey", "ldrloaddll", "ldrgetprocedureaddress", "regopenkeyexw",
          "ntcreatefile", "ntreadfile", "ntallocatevirtualmemory"]
_CAMEL = ["NtClose", "NtOpenKey", "LdrLoadDll", "LdrGetProcedureAddress", "RegOpenKeyExW",
          "NtCreateFile", "NtReadFile", "NtAllocateVirtualMemory"]


class _BenignStaticModel:
    def predict_proba(self, X, *_args, **_kwargs):
        return np.full(len(X), 0.01, dtype=np.float64)


class _RecordingBehavioral:
    """Constant logit; records whether it was called and in which grad mode."""

    def __init__(self, logit: float = -4.0):
        self.logit = logit
        self.calls = 0
        self.grad_enabled = []

    def __call__(self, x):
        self.calls += 1
        self.grad_enabled.append(torch.is_grad_enabled())
        return torch.tensor([self.logit])


def _trace(tmp_path, calls) -> str:
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(calls))
    return str(p)


def _pipe(model, tok=None):
    return CortexPipeline(static_model=_BenignStaticModel(), behavioral_model=model,
                          tokenizer=tok or ApiTokenizer(_VOCAB), self_test=False)


# ------------------------------------------------------------ F3: tokenizer
def test_camelcase_and_lowercase_encode_identically():
    tok = ApiTokenizer(_VOCAB)
    trace = (_CAMEL * 13)[:100]
    lower_ids, lower_status = tok.encode([n.lower() for n in trace])
    camel_ids, camel_status = tok.encode(trace)
    upper_ids, _ = tok.encode([n.upper() for n in trace])
    assert np.array_equal(camel_ids, lower_ids) and np.array_equal(upper_ids, lower_ids)
    assert camel_status == lower_status == "ok"
    assert UNK_ID not in camel_ids


def test_encode_is_lowercase_only_no_other_normalisation():
    tok = ApiTokenizer(["regopenkeyexw"])
    ids, _ = tok.encode(["RegOpenKeyExW", "RegOpenKeyEx", "RegOpenKeyExA", "ZwClose", " regopenkeyexw"])
    # only the exact case-insensitive match is known; no suffix stripping,
    # no Zw->Nt, no whitespace trimming
    assert ids[:5].tolist() == [tok.token_to_id["regopenkeyexw"], UNK_ID, UNK_ID, UNK_ID, UNK_ID]


def test_bad_names_stay_unk_and_are_not_dropped():
    tok = ApiTokenizer(_VOCAB)
    calls = ["ntclose", "", "ntclose", "Ñtclose", "nt close", "ntclose"]
    ids, _ = tok.encode(calls)
    assert ids[:6].tolist() == [2, UNK_ID, 2, UNK_ID, UNK_ID, 2]  # positions preserved


@pytest.mark.parametrize("calls,key,expected", [
    (["a", "", "b", ""], "empty_names", 2),
    (["a", "é", "ntclose", "日本"], "non_ascii_names", 2),
    (["a", "nt close", "x\ty", "ok"], "whitespace_names", 2),
])
def test_input_diagnostics_counts(calls, key, expected):
    diag = input_diagnostics(calls)
    assert diag[key] == expected
    assert all(v == 0 for k, v in diag.items() if k != key)


def test_scored_unk_rate_uses_only_the_scored_window():
    ids = np.full(MAX_SEQ_LEN, 0, dtype=np.int64)
    ids[:10] = [2, UNK_ID, 2, UNK_ID, 2, 2, 2, 2, 2, 2]
    assert scored_unk_rate(ids, 10) == pytest.approx(0.2)
    assert scored_unk_rate(ids, 0) == 0.0
    long_ids = np.full(MAX_SEQ_LEN, UNK_ID, dtype=np.int64)
    assert scored_unk_rate(long_ids, 5000) == 1.0  # only 100 slots are scored


# ----------------------------------------------------- diagnostics on result
@pytest.mark.parametrize("bad,key", [("", "empty_names"), ("ntclösé", "non_ascii_names"),
                                     ("nt close", "whitespace_names")])
def test_diagnostics_surface_on_result_and_log_warning(tmp_path, caplog, bad, key):
    calls = ["ntclose"] * 99 + [bad]  # 1% UNK -> still scored
    model = _RecordingBehavioral()
    with caplog.at_level(logging.WARNING, logger="cortex.pipeline"):
        result = _pipe(model).scan(_VALID_PE, api_calls_json_path=_trace(tmp_path, calls))
    diag = result.behavioral_input_diagnostics
    assert diag[key] == 1
    assert diag["unk_rate"] == pytest.approx(0.01)
    assert result.to_dict()["behavioral_input_diagnostics"] == diag
    assert any("input-format problems" in r.message for r in caplog.records)
    assert model.calls == 1  # scoring unchanged
    assert result.behavioral_verdict == BehavioralVerdict.BENIGN


def test_clean_trace_has_zero_diagnostics_and_no_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="cortex.pipeline"):
        result = _pipe(_RecordingBehavioral()).scan(
            _VALID_PE, api_calls_json_path=_trace(tmp_path, _CAMEL * 13))
    assert result.behavioral_input_diagnostics == {
        "empty_names": 0, "non_ascii_names": 0, "whitespace_names": 0, "unk_rate": 0.0}
    assert not any("input-format problems" in r.message for r in caplog.records)


def test_no_trace_means_no_diagnostics():
    result = _pipe(_RecordingBehavioral()).scan(_VALID_PE)
    assert result.behavioral_input_diagnostics is None


# --------------------------------------------------------------- UNK guard
def test_high_unk_trace_is_pending_not_scored(tmp_path):
    calls = ["ntclose"] * 50 + ["SomethingUnseen"] * 50  # 50% UNK
    model = _RecordingBehavioral(logit=10.0)  # would be MALICIOUS if scored
    result = _pipe(model).scan(_VALID_PE, api_calls_json_path=_trace(tmp_path, calls))

    assert model.calls == 0
    assert result.behavioral_verdict == BehavioralVerdict.PENDING
    assert result.behavioral_score is None
    assert "behavioral_unk_rate_high" in result.reason_codes
    # Step 2 rule: static ALLOW + behavioral PENDING -> ALLOW_UNVERIFIED
    assert result.static_verdict == StaticVerdict.ALLOW
    assert result.final_decision == FinalDecision.ALLOW_UNVERIFIED
    assert "behavioral_pending_unverified" in result.reason_codes


def test_camelcase_trace_is_not_mistaken_for_high_unk(tmp_path):
    model = _RecordingBehavioral()
    result = _pipe(model).scan(_VALID_PE, api_calls_json_path=_trace(tmp_path, _CAMEL * 13))
    assert model.calls == 1
    assert "behavioral_unk_rate_high" not in result.reason_codes


def test_unk_rate_exactly_at_threshold_is_still_scored(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "BEHAVIORAL_MAX_UNK_RATE", 0.10)
    calls = ["ntclose"] * 90 + ["Unseen"] * 10  # exactly 0.10
    model = _RecordingBehavioral()
    result = _pipe(model).scan(_VALID_PE, api_calls_json_path=_trace(tmp_path, calls))
    assert model.calls == 1
    assert result.behavioral_input_diagnostics["unk_rate"] == pytest.approx(0.10)


def test_configured_threshold_is_the_yaml_value():
    assert BEHAVIORAL_MAX_UNK_RATE == 0.10


# ---------------------------------------------------- F21: eval / no-grad
def _real_net(seed: int = 0) -> CortexBehavioralNet:
    torch.manual_seed(seed)
    return CortexBehavioralNet(vocab_size=len(_VOCAB) + 2, embed_dim=16)


def test_model_handed_over_in_train_mode_is_put_in_eval_and_deterministic(tmp_path):
    net = _real_net()
    net.train()
    pipe = _pipe(net)
    assert net.training is False  # set at construction

    path = _trace(tmp_path, _CAMEL * 13)
    scores = [pipe.scan(_VALID_PE, api_calls_json_path=path).behavioral_score for _ in range(5)]
    assert len(set(scores)) == 1 and scores[0] is not None


def test_model_switched_back_to_train_after_construction_still_scores_in_eval(tmp_path):
    net = _real_net()
    pipe = _pipe(net)
    path = _trace(tmp_path, _CAMEL * 13)
    first = pipe.scan(_VALID_PE, api_calls_json_path=path).behavioral_score
    net.train()
    again = pipe.scan(_VALID_PE, api_calls_json_path=path).behavioral_score
    assert again == first
    assert net.training is False


def test_scoring_runs_without_autograd(tmp_path):
    model = _RecordingBehavioral()
    _pipe(model).scan(_VALID_PE, api_calls_json_path=_trace(tmp_path, _CAMEL * 13))
    assert model.grad_enabled == [False]


# ------------------------------------- real artifacts (skipped if absent)
_REAL = ("data/models/api_vocab.json", "data/models/cortex_behavioral_best.pt",
         "data/processed/behavioral_test.parquet")


@pytest.mark.slow
@pytest.mark.skipif(not all(os.path.exists(p) for p in _REAL), reason="behavioral artifacts absent")
def test_real_vocab_camelcase_traces_score_like_lowercase():
    import pandas as pd
    tok = ApiTokenizer.load(_REAL[0])
    df = pd.read_parquet(_REAL[2]).head(200)
    for calls in df.api_calls:
        lower = list(calls)
        camel = [n[:1].upper() + n[1:] for n in lower]
        a, sa = tok.encode(lower)
        b, sb = tok.encode(camel)
        assert np.array_equal(a, b) and sa == sb
