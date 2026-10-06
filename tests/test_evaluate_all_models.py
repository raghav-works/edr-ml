"""
Smoke test for scripts/evaluate_all_models.py -- the standalone
val+test evaluation harness re-run after every retrain / threshold change.

Marked `slow` (loads every trained model + real held-out parquet splits) and
skipped in full if any artifact is absent, so a fresh checkout without
data/models/ still gets a clean `pytest -q`.

This does NOT re-check the model numbers (that is the script's job, reviewed
by a human). It checks the harness stays wired up: every evaluator runs, and
every confusion matrix it returns is internally consistent (cells sum to n,
benign+malicious = n, rates in range). Each evaluator is called with a small
row cap so the whole test is a few seconds, not a few minutes.
"""
from __future__ import annotations

import os

import pytest

from scripts.evaluate_all_models import (
    BinaryEval,
    PrevalenceProjection,
    evaluate_behavioral,
    evaluate_emulation,
    evaluate_memory,
    evaluate_network,
    evaluate_static,
)

_REQUIRED = [
    "config/thresholds.yaml",
    "data/models/cortex_static.lgbm", "data/models/cortex_static.meta.json",
    "data/models/cortex_memory.lgbm", "data/models/cortex_memory.meta.json",
    "data/models/cortex_network.lgbm", "data/models/cortex_network.meta.json",
    "data/models/cortex_behavioral_best.pt", "data/models/api_vocab.json",
    "data/processed/ember2024_train.parquet", "data/processed/ember2024_test.parquet",
    "data/processed/memory_val.parquet", "data/processed/memory_test.parquet",
    "data/processed/network_val.parquet", "data/processed/network_test.parquet",
    "data/processed/network_train.parquet",  # per-attack-type breakdown reads its label_raw support counts
    "data/processed/behavioral_val.parquet", "data/processed/behavioral_test.parquet",
]
_MISSING = [p for p in _REQUIRED if not os.path.exists(p)]

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(bool(_MISSING), reason=f"evaluation artifacts absent (e.g. {_MISSING[:2]})"),
]

_LIMIT = 1500


def _check_binary_eval(ev: BinaryEval) -> None:
    assert isinstance(ev, BinaryEval)
    assert ev.tn + ev.fp + ev.fn + ev.tp == ev.n
    assert ev.n_benign + ev.n_malicious == ev.n
    assert ev.n_benign == ev.tn + ev.fp
    assert ev.n_malicious == ev.fn + ev.tp
    for rate in (ev.precision, ev.recall, ev.f1, ev.fpr):
        assert 0.0 <= rate <= 1.0
    assert ev.auc_roc != ev.auc_roc or 0.0 <= ev.auc_roc <= 1.0  # NaN allowed (one-class slice)


def _check_prevalence_projection(proj: PrevalenceProjection) -> None:
    assert isinstance(proj, PrevalenceProjection)
    assert 0.0 <= proj.fpr <= 1.0 and 0.0 <= proj.tpr <= 1.0
    assert proj.n_benign >= 0 and proj.n_malicious >= 0
    assert proj.fpr_ci95_upper is None or proj.fpr_ci95_upper > 0.0
    assert proj.rows, "projection has no prevalence rows"
    for r in proj.rows:
        assert 0.0 < r.prevalence < 1.0
        assert 0.0 <= r.ppv <= 1.0
        assert 0.0 <= r.alert_rate <= 1.0
        assert all(v >= 0.0 for v in r.fp_per.values())
        assert all(v >= 0.0 for v in r.tp_per.values())


def test_static_evaluator_runs_and_is_consistent():
    res = evaluate_static(limit=_LIMIT)
    assert set(res["splits"]) == {"val", "test"}
    assert set(res["prevalence"]) == {"val", "test"}
    for split in ("val", "test"):
        s = res["splits"][split]
        _check_binary_eval(s["allow_boundary"])
        _check_binary_eval(s["block_boundary"])
        p = res["prevalence"][split]
        _check_prevalence_projection(p["allow_boundary"])
        _check_prevalence_projection(p["block_boundary"])
        # 3-way rows each sum to that class's n
        for cls in ("benign", "malicious"):
            row = s["three_way"][cls]
            assert row["ALLOW"] + row["ALERT"] + row["BLOCK"] == row["n"]
        # ALLOW-boundary positives == ALERT + BLOCK cells
        tw = s["three_way"]
        ab = s["allow_boundary"]
        assert ab.fp == tw["benign"]["ALERT"] + tw["benign"]["BLOCK"]
        assert ab.tp == tw["malicious"]["ALERT"] + tw["malicious"]["BLOCK"]
        # BLOCK-boundary positives == BLOCK cells
        bb = s["block_boundary"]
        assert bb.fp == tw["benign"]["BLOCK"]
        assert bb.tp == tw["malicious"]["BLOCK"]


@pytest.mark.parametrize("evaluator", [evaluate_memory, evaluate_network])
def test_lgbm_signal_evaluators_run_and_are_consistent(evaluator):
    res = evaluator(limit=_LIMIT)
    assert set(res["splits"]) == {"val", "test"}
    assert set(res["prevalence"]) == {"val", "test"}
    for split in ("val", "test"):
        _check_binary_eval(res["splits"][split])
        _check_prevalence_projection(res["prevalence"][split])


def test_behavioral_evaluator_bands_and_totals():
    res = evaluate_behavioral(limit=_LIMIT)
    assert set(res["splits"]) == {"val", "test"}
    assert set(res["prevalence"]) == {"val", "test"}
    for split in ("val", "test"):
        s = res["splits"][split]
        dep, raw = s["deployment_total"], s["model_raw_total"]
        _check_binary_eval(dep)
        _check_binary_eval(raw)
        _check_prevalence_projection(res["prevalence"][split]["deployment_total"])
        band_n = 0
        for band, ev in s["bands"].items():
            if ev is not None:
                _check_binary_eval(ev)
                band_n += ev.n
        # every row lands in exactly one band
        assert band_n == raw.n
        # deployment total == short + ok+truncated bands
        scored = sum(ev.n for b, ev in s["bands"].items()
                     if b != "too_short" and ev is not None)
        assert dep.n == scored


def test_emulation_evaluator_is_report_only_but_runs():
    res = evaluate_emulation(limit=_LIMIT)
    if res is None:
        pytest.skip("emulation checkpoint / splits absent")
    assert set(res["splits"]) == {"val", "test"}
    for split in ("val", "test"):
        s = res["splits"][split]
        _check_binary_eval(s["deployment"])
        if s["model_raw"] is not None:  # legacy unmasked checkpoint
            _check_binary_eval(s["model_raw"])
            assert s["model_raw"].n == s["deployment"].n + s["pending"]
