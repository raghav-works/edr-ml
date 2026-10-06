"""
docs/CODE_REVIEW.md F11 -- memory/network vectors detached from file scans,
behind config/thresholds.yaml file_scan.attach_memory_network.

  * true (default): today's behaviour -- supplied vectors are scored and
    reach decide(). The rest of the suite runs with this value unchanged.
  * false: they are not scored for a file scan; decide() gets NOT_PROVIDED
    for both; a supplied vector is recorded as signal_health
    "detached_by_config" and logged, never dropped silently.
  * strict boolean parsing of the key.
"""
from __future__ import annotations

import itertools
import logging
import os

import numpy as np
import pytest

from inference import pipeline as pipeline_mod
from inference import policy_engine as pe
from inference.pipeline import CortexPipeline
from inference.policy_engine import (
    BehavioralVerdict, FinalDecision, MemoryVerdict, NetworkVerdict, StaticVerdict, decide,
)

_VALID_PE = os.path.join(os.path.dirname(__file__), "fixtures", "pe_samples", "sample_cli64.exe")
_MEMORY_FEATURES = np.zeros(62, dtype=np.float32)
_NETWORK_FEATURES = np.zeros(78, dtype=np.float32)


class _ConstModel:
    def __init__(self, value: float):
        self.value = value
        self.calls = 0

    def predict_proba(self, X, *_a, **_k):
        self.calls += 1
        return np.full(len(X), self.value, dtype=np.float64)


class _ExplodingModel:
    def __init__(self):
        self.calls = 0

    def predict_proba(self, *_a, **_k):
        self.calls += 1
        raise RuntimeError("simulated model failure")


_BENIGN, _BLOCK = 0.01, 0.999999  # static ALLOW / static BLOCK (>= block_at_or_above)


def _pipe(static=_BENIGN, memory=None, network=None):
    return CortexPipeline(static_model=_ConstModel(static), memory_model=memory,
                          network_model=network, self_test=False)


@pytest.fixture
def detached(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "FILE_SCAN_ATTACH_MEMORY_NETWORK", False)


# ----------------------------------------------------------- config
def test_default_is_attached():
    assert pe.FILE_SCAN_ATTACH_MEMORY_NETWORK is True
    assert pipeline_mod.FILE_SCAN_ATTACH_MEMORY_NETWORK is True


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, "no"])
def test_flag_parsing_is_strict(value):
    with pytest.raises(RuntimeError, match="must be true or false"):
        pe._flag({"file_scan": {"attach_memory_network": value}}, "file_scan", "attach_memory_network")


def test_missing_flag_is_an_error():
    with pytest.raises(RuntimeError, match="missing required key: file_scan.attach_memory_network"):
        pe._flag({"file_scan": {}}, "file_scan", "attach_memory_network")


@pytest.mark.parametrize("value", [True, False])
def test_flag_accepts_booleans(value):
    assert pe._flag({"file_scan": {"attach_memory_network": value}},
                    "file_scan", "attach_memory_network") is value


# ------------------------------------------------- true: unchanged
def test_attached_scores_memory_and_network():
    mem, net = _ConstModel(0.999999), _ConstModel(0.999999)
    result = _pipe(memory=mem, network=net).scan(
        _VALID_PE, memory_features=_MEMORY_FEATURES, network_features=_NETWORK_FEATURES)
    assert (mem.calls, net.calls) == (1, 1)
    assert result.memory_verdict == MemoryVerdict.MALICIOUS
    assert result.network_verdict == NetworkVerdict.MALICIOUS
    assert result.final_decision == FinalDecision.ALERT
    assert "memory_malicious" in result.reason_codes
    assert "detached_by_config" not in result.signal_health.values()


def test_attached_static_block_with_memory_corroboration_is_block():
    result = _pipe(static=_BLOCK, memory=_ConstModel(0.999999)).scan(
        _VALID_PE, memory_features=_MEMORY_FEATURES)
    assert result.static_verdict == StaticVerdict.BLOCK
    assert result.final_decision == FinalDecision.BLOCK


# ------------------------------------------------- false: detached
def test_detached_vectors_are_not_scored_and_not_provided(detached, caplog):
    mem, net = _ConstModel(0.999999), _ConstModel(0.999999)
    with caplog.at_level(logging.WARNING, logger="cortex.pipeline"):
        result = _pipe(memory=mem, network=net).scan(
            _VALID_PE, memory_features=_MEMORY_FEATURES, network_features=_NETWORK_FEATURES)
    assert (mem.calls, net.calls) == (0, 0)
    assert result.memory_verdict == MemoryVerdict.NOT_PROVIDED
    assert result.network_verdict == NetworkVerdict.NOT_PROVIDED
    assert result.memory_score is None and result.network_score is None
    assert result.signal_health == {"memory": "detached_by_config", "network": "detached_by_config"}
    assert result.to_dict()["signal_health"] == result.signal_health
    warnings = [r.message for r in caplog.records if "attach_memory_network" in r.message]
    assert len(warnings) == 2
    assert any(m.startswith("memory_features") for m in warnings)
    assert any(m.startswith("network_features") for m in warnings)
    # static ALLOW, nothing else -> the file's own decision
    assert result.final_decision == FinalDecision.ALLOW
    assert not {"memory_malicious", "network_malicious"} & set(result.reason_codes)


def test_detached_static_block_can_no_longer_become_block(detached):
    result = _pipe(static=_BLOCK, memory=_ConstModel(0.999999), network=_ConstModel(0.999999)).scan(
        _VALID_PE, memory_features=_MEMORY_FEATURES, network_features=_NETWORK_FEATURES)
    assert result.static_verdict == StaticVerdict.BLOCK
    assert result.final_decision == FinalDecision.ALERT
    assert "static_block_capped_at_alert" in result.reason_codes
    assert "corroborated_multi_signal" not in result.reason_codes


def test_detached_model_error_no_longer_forces_review(detached):
    mem = _ExplodingModel()
    result = _pipe(memory=mem).scan(_VALID_PE, memory_features=_MEMORY_FEATURES)
    assert mem.calls == 0
    assert result.signal_health == {"memory": "detached_by_config"}
    assert result.final_decision == FinalDecision.ALLOW


def test_detached_only_marks_signals_that_were_supplied(detached, caplog):
    with caplog.at_level(logging.WARNING, logger="cortex.pipeline"):
        only_net = _pipe(network=_ConstModel(0.5)).scan(_VALID_PE, network_features=_NETWORK_FEATURES)
        none = _pipe().scan(_VALID_PE)
    assert only_net.signal_health == {"network": "detached_by_config"}
    assert none.signal_health == {}
    assert len([r for r in caplog.records if "attach_memory_network" in r.message]) == 1


def test_detached_also_applies_when_the_file_cannot_be_analyzed(detached):
    result = _pipe(memory=_ConstModel(0.999999)).scan("/nonexistent/x.exe", memory_features=_MEMORY_FEATURES)
    assert result.signal_health == {"memory": "detached_by_config"}
    assert result.final_decision == FinalDecision.NEEDS_REVIEW  # not a memory ALERT


def test_detached_without_models_configured_is_still_detached(detached):
    result = _pipe().scan(_VALID_PE, memory_features=_MEMORY_FEATURES)
    assert result.signal_health == {"memory": "detached_by_config"}  # not model_not_configured


_MODELS = {"none": None, "benign": 0.0, "malicious": 0.999999, "error": "error"}


def _model(kind):
    v = _MODELS[kind]
    return None if v is None else _ExplodingModel() if v == "error" else _ConstModel(v)


@pytest.mark.parametrize("static", [_BENIGN, _BLOCK], ids=["static_allow", "static_block"])
@pytest.mark.parametrize("mem,net", list(itertools.product(_MODELS, _MODELS)))
@pytest.mark.parametrize("attach", [True, False])
def test_pipeline_passes_not_provided_when_detached(monkeypatch, static, mem, net, attach):
    monkeypatch.setattr(pipeline_mod, "FILE_SCAN_ATTACH_MEMORY_NETWORK", attach)
    result = _pipe(static=static, memory=_model(mem), network=_model(net)).scan(
        _VALID_PE, memory_features=_MEMORY_FEATURES, network_features=_NETWORK_FEATURES)
    if attach:
        expected = decide(result.static_verdict, BehavioralVerdict.NOT_PROVIDED,
                          result.memory_verdict, result.network_verdict)
    else:
        assert (result.memory_verdict, result.network_verdict) == (
            MemoryVerdict.NOT_PROVIDED, NetworkVerdict.NOT_PROVIDED)
        expected = decide(result.static_verdict, BehavioralVerdict.NOT_PROVIDED,
                          MemoryVerdict.NOT_PROVIDED, NetworkVerdict.NOT_PROVIDED)
    assert result.final_decision == expected[0]
    assert result.reason_codes[-len(expected[1]):] == expected[1]
